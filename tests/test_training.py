import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.agents.ppo import networks as ppo_networks
from conftest import initial_params

from self_balancing_robot.actor import Actor, layers_from_brax
from self_balancing_robot.envs.balance_env import CRITIC_KEY
from self_balancing_robot.envs.wrappers import (
    TrainingWrapper,
    initial_curriculum,
    update_curriculum,
)
from self_balancing_robot.metrics import (
    METRIC_KEYS,
    plot_episode,
    plot_training_metrics,
    run_episode,
    summarize,
)


def _finish_episodes(
    env,
    curriculum,
    count,
    reached=1.0,
    commanded=1.0,
    still_speed=0.0,
    survive=True,
    reversal=False,
    level=None,
    batch=100,
    callback=None,
):
    """Feed ``count`` finished episodes to the curriculum, ``batch`` per step as in training.

    ``survive`` and ``reversal`` are a value for all episodes or one per episode.
    """
    survive = np.broadcast_to(survive, (count,))
    reversal = np.broadcast_to(reversal, (count,))
    for start in range(0, count, batch):
        n = min(batch, count - start)
        info = {
            "episode_stage": jnp.full(n, curriculum["stage"]),
            "episode_level": jnp.full(n, curriculum["level"] if level is None else level),
            "terminated": jnp.asarray(~survive[start : start + n]),
            "episode_reversal": jnp.asarray(reversal[start : start + n]),
            "episode_tracking": jnp.tile(jnp.array([reached, commanded, still_speed, 1.0]), (n, 1)),
        }
        curriculum = update_curriculum(env, curriculum, info, jnp.ones(n, bool), callback)
    return curriculum


def _position(curriculum):
    return int(curriculum["stage"]), int(curriculum["level"])


def _start(env, stage, level=1):
    return {**initial_curriculum(env), "stage": jnp.int32(stage), "level": jnp.int32(level)}


def test_curriculum_ramps_commands_then_reaches_domain_randomization(env):
    cfg = env.curriculum_config
    curriculum = _finish_episodes(env, initial_curriculum(env), cfg.evaluation_window)
    assert _position(curriculum) == (2, 1)
    for level in range(2, cfg.command_levels + 1):
        curriculum = _finish_episodes(env, curriculum, cfg.command_min_episodes_per_level)
        assert _position(curriculum) == (2, level)
    curriculum = _finish_episodes(env, curriculum, cfg.command_min_episodes_per_level)
    assert _position(curriculum) == (3, 1)


@pytest.mark.parametrize("stage", [2, 3])
def test_levels_require_speed_survival_and_standing_still(env, stage):
    cfg = env.curriculum_config
    minimum = (
        cfg.command_min_episodes_per_level if stage == 2 else cfg.randomization_min_episodes_per_level
    )
    ratio = cfg.command_speed_ratio if stage == 2 else cfg.randomization_speed_ratio
    still_limit = (
        cfg.command_still_speed_limit if stage == 2 else cfg.randomization_still_speed_limit
    )
    for failure in (
        {"reached": ratio - 0.05},  # too slow (v5 drove at ~60%)
        {"still_speed": still_limit + 0.005},  # rolls away when told to stop
        {"survive": np.arange(minimum) % 5 != 0},  # 80% survive: below both success rates
    ):
        curriculum = _finish_episodes(env, _start(env, stage), minimum, **failure)
        assert _position(curriculum) == (stage, 1), failure
    curriculum = _finish_episodes(env, _start(env, stage), minimum, reached=ratio + 0.01)
    assert _position(curriculum) == (stage, 2)


def test_a_full_window_is_not_enough_without_the_minimum_episodes(env):
    cfg = env.curriculum_config
    assert cfg.evaluation_window < cfg.randomization_min_episodes_per_level
    curriculum = _finish_episodes(env, _start(env, 3), cfg.evaluation_window)
    assert _position(curriculum) == (3, 1)
    assert int(curriculum["window_episodes"]) == 0  # the window was evaluated and restarted
    assert int(curriculum["level_episodes"]) == cfg.evaluation_window


def test_levels_require_surviving_reversals(env):
    cfg = env.curriculum_config
    minimum = cfg.command_min_episodes_per_level
    # 95% survive overall, but all falls are reversal episodes (a third of reversals fail).
    episode = np.arange(minimum)
    reversal = episode % 20 < 3
    survive = episode % 20 != 0
    curriculum = _finish_episodes(env, _start(env, 2), minimum, survive=survive, reversal=reversal)
    assert _position(curriculum) == (2, 1)
    curriculum = _finish_episodes(env, curriculum, cfg.evaluation_window, reversal=True)
    assert _position(curriculum) == (2, 2)


def test_episodes_from_an_earlier_level_are_not_counted(env):
    cfg = env.curriculum_config
    curriculum = _finish_episodes(env, _start(env, 2, 2), cfg.evaluation_window, level=1)
    assert int(curriculum["window_episodes"]) == 0 and int(curriculum["level_episodes"]) == 0


def test_last_level_stays(env):
    cfg = env.curriculum_config
    curriculum = _start(env, 3, cfg.randomization_levels)
    curriculum = _finish_episodes(env, curriculum, cfg.randomization_min_episodes_per_level)
    assert _position(curriculum) == (3, cfg.randomization_levels)


def test_curriculum_reports_every_window_to_the_host(env):
    cfg = env.curriculum_config
    reports = []
    _finish_episodes(env, initial_curriculum(env), cfg.evaluation_window, callback=reports.append)
    jax.effects_barrier()
    assert len(reports) == 1
    report = reports[0]
    assert (report["stage"], report["level"], report["new_stage"], report["new_level"]) == (1, 1, 2, 1)
    assert report["success_rate"] == 1.0


def test_training_wrapper_resets_finished_episodes_fresh_at_the_curriculum_level(env):
    wrapper = TrainingWrapper(env)
    batch = 8
    reset_fn = jax.jit(wrapper.reset)
    step_fn = jax.jit(wrapper.step)
    state = reset_fn(jax.random.split(jax.random.PRNGKey(0), batch))
    # Training has moved on to stage 3, level 2: new episodes start there.
    state = state.replace(info={**state.info, "curriculum": _start(env, 3, 2)})
    first_slopes = np.asarray(state.info["slope"])
    finished = np.zeros(batch, bool)
    for _ in range(150):  # unpowered, every robot falls within 1.5 s
        state = step_fn(state, jnp.zeros((batch, 2)))
        done = np.asarray(state.done) > 0
        if done.any():
            info = state.info
            # Bookkeeping describes the step just taken...
            assert np.all(np.asarray(info["episode_done"])[done] == 1.0)
            assert np.all(np.asarray(info["steps"])[done] > 0)
            # ... while the environment already runs a fresh episode.
            assert np.all(np.asarray(info["episode_step"])[done] == 0)
            assert np.all(np.asarray(info["stage"])[done] == 3)
            assert np.all(np.asarray(info["level"])[done] == 2)
            assert np.all(np.asarray(info["episode_level"])[done] == 2)
            finished |= done
        if finished.all():
            break
    assert finished.all()
    # Fresh resets draw new terrain at the new level (Brax's default auto-reset would repeat
    # the first, flat stage-1 episode); 10% of draws are flat on purpose.
    assert np.all(first_slopes == 0.0)
    slopes = np.asarray(state.info["slope"])
    assert np.count_nonzero(slopes) >= batch // 2
    assert len(np.unique(slopes[slopes != 0])) == np.count_nonzero(slopes)
    assert np.all(np.abs(slopes) <= env.slope_limit(3, 2) + 1e-6)


def test_actor_is_the_brax_policy_mean_and_blind_to_privileged_state(env):
    params, networks = initial_params(env)
    actor = Actor(layers_from_brax(params[1]))
    assert actor.sizes == [env.actor_observation_size, 128, 64, 32, 2]

    observation = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(9)).obs)
    observation = {key: value[None] * 50.0 for key, value in observation.items()}  # not tiny
    changed = {**observation, CRITIC_KEY: observation[CRITIC_KEY] + 10.0}
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    with jax.default_matmul_precision("highest"):
        actions = [np.asarray(policy(obs, jax.random.PRNGKey(0))[0]) for obs in (observation, changed)]
        values = [
            float(networks.value_network.apply(params[0], params[2], obs)[0])
            for obs in (observation, changed)
        ]
    np.testing.assert_allclose(actions[0], actions[1])
    assert values[0] != values[1]
    np.testing.assert_allclose(actor.predict(observation), np.clip(actions[0], -1, 1), atol=1e-5)
    # Training starts near zero action (small initial action layer, as in SB3).
    assert np.abs(actions[0]).max() < 0.1


def test_episode_trace_metrics_and_plots(env, tmp_path):
    zero = Actor([(np.zeros((2, env.actor_observation_size)), np.zeros(2))])
    trace = run_episode(env, zero, seed=11, max_steps=50, stage=2)
    assert len(trace["time"]) == len(trace["v_cmd"]) == trace["action"].shape[0]
    assert trace["end"] in {"fall", "max_steps"}
    metrics = summarize(trace)
    assert set(metrics) == set(METRIC_KEYS)
    assert all(np.isfinite(value) for value in metrics.values())

    plot_episode(trace, tmp_path / "episode.png")
    history = {"steps": [1, 2], "curriculum_progress": [1, 2]}
    for key in METRIC_KEYS:
        history["level_" + key] = [0.1, 0.2]
        history["final_" + key] = [0.1, 0.2]
    plot_training_metrics(history, tmp_path / "training.png")
    for name in ("episode.png", "training.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
