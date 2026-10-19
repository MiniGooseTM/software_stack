from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.agents.ppo import networks as ppo_networks
from conftest import initial_params, make_env

from self_balancing_robot.actor import Actor, layers_from_brax
from self_balancing_robot.curriculum import EVALUATION_WINDOW, Stage
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

# A small curriculum for the promotion mechanics, independent of the real list.
EASY = Stage(name="easy", success_rate=0.95)
TRACK = Stage(
    name="track",
    command_max=(0.3, 1.5),
    min_episodes=2_000,
    success_rate=0.95,
    speed_ratio=0.90,
    still_speed_limit=0.02,
    reversal_success_rate=0.85,
)
FINAL = replace(TRACK, name="final", min_episodes=3_000, success_rate=0.90)
SMALL = (EASY, TRACK, FINAL)


@pytest.fixture(scope="module")
def small():
    return make_env(curriculum=SMALL)


def _finish_episodes(
    env,
    curriculum,
    count,
    reached=1.0,
    commanded=1.0,
    still_speed=0.0,
    survive=True,
    reversal=False,
    stage=None,
    batch=100,
    callback=None,
):
    """Feed ``count`` finished episodes to the curriculum, ``batch`` per step as in training.

    ``survive`` and ``reversal`` are a value for all episodes or one per episode; ``stage``
    is the stage the episodes started at (default: the current one).
    """
    survive = np.broadcast_to(survive, (count,))
    reversal = np.broadcast_to(reversal, (count,))
    for start in range(0, count, batch):
        n = min(batch, count - start)
        info = {
            "episode_stage": jnp.full(n, curriculum["stage"] if stage is None else stage),
            "terminated": jnp.asarray(~survive[start : start + n]),
            "episode_reversal": jnp.asarray(reversal[start : start + n]),
            "episode_tracking": jnp.tile(jnp.array([reached, commanded, still_speed, 1.0]), (n, 1)),
        }
        curriculum = update_curriculum(env, curriculum, info, jnp.ones(n, bool), callback)
    return curriculum


def _at(env, stage):
    return {**initial_curriculum(env), "stage": jnp.int32(env.stage_index(stage))}


def _stage(curriculum):
    return SMALL[int(curriculum["stage"])].name


def test_training_moves_through_the_list_and_stays_at_the_last_stage(small):
    curriculum = _finish_episodes(small, initial_curriculum(small), EVALUATION_WINDOW)
    assert _stage(curriculum) == "track"
    curriculum = _finish_episodes(small, curriculum, TRACK.min_episodes)
    assert _stage(curriculum) == "final"
    curriculum = _finish_episodes(small, curriculum, 2 * FINAL.min_episodes)
    assert _stage(curriculum) == "final"


def test_a_stage_test_checks_speed_survival_standing_still_and_reversals(small):
    minimum = TRACK.min_episodes
    episode = np.arange(minimum)
    for failure in (
        {"reached": TRACK.speed_ratio - 0.05},  # too slow (v5 drove at ~60%)
        {"still_speed": TRACK.still_speed_limit + 0.005},  # rolls away when told to stop
        {"survive": episode % 5 != 0},  # 80% survive
        # 95% survive overall, but all falls are reversal episodes (a third of reversals fail).
        {"survive": episode % 20 != 0, "reversal": episode % 20 < 3},
    ):
        curriculum = _finish_episodes(small, _at(small, "track"), minimum, **failure)
        assert _stage(curriculum) == "track", failure
    passing = {"reached": TRACK.speed_ratio + 0.01, "reversal": True}
    curriculum = _finish_episodes(small, _at(small, "track"), minimum, **passing)
    assert _stage(curriculum) == "final"


def test_unchecked_criteria_are_not_checked(small):
    # "easy" only checks survival: slow, drifting and failed reversals do not matter.
    curriculum = _finish_episodes(
        small, _at(small, "easy"), EVALUATION_WINDOW, reached=0.0, still_speed=1.0, reversal=True
    )
    assert _stage(curriculum) == "track"


def test_a_full_window_is_not_enough_without_the_minimum_episodes(small):
    assert EVALUATION_WINDOW < TRACK.min_episodes
    curriculum = _finish_episodes(small, _at(small, "track"), EVALUATION_WINDOW)
    assert _stage(curriculum) == "track"
    assert int(curriculum["window_episodes"]) == 0  # the window was evaluated and restarted
    assert int(curriculum["stage_episodes"]) == EVALUATION_WINDOW


def test_episodes_from_an_earlier_stage_are_not_counted(small):
    curriculum = _finish_episodes(small, _at(small, "track"), EVALUATION_WINDOW, stage=0)
    assert int(curriculum["window_episodes"]) == 0 and int(curriculum["stage_episodes"]) == 0


def test_curriculum_reports_every_window_to_the_host(small):
    reports = []
    _finish_episodes(
        small, initial_curriculum(small), EVALUATION_WINDOW, callback=reports.append
    )
    jax.effects_barrier()
    assert len(reports) == 1
    assert (reports[0]["stage"], reports[0]["new_stage"]) == (0, 1)
    assert reports[0]["success_rate"] == 1.0


def test_training_wrapper_resets_finished_episodes_fresh_at_the_curriculum_stage(env):
    wrapper = TrainingWrapper(env)
    batch = 8
    reset_fn = jax.jit(wrapper.reset)
    step_fn = jax.jit(wrapper.step)
    state = reset_fn(jax.random.split(jax.random.PRNGKey(0), batch))
    # Training has moved on to "randomization 2/6": new episodes start there.
    stage = env.stage_index("randomization 2/6")
    state = state.replace(info={**state.info, "curriculum": _at(env, stage)})
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
            assert np.all(np.asarray(info["stage"])[done] == stage)
            assert np.all(np.asarray(info["episode_stage"])[done] == stage)
            finished |= done
        if finished.all():
            break
    assert finished.all()
    # Fresh resets draw new terrain at the new stage (Brax's default auto-reset would repeat
    # the first, flat episode); 10% of draws are flat on purpose.
    assert np.all(first_slopes == 0.0)
    slopes = np.asarray(state.info["slope"])
    assert np.count_nonzero(slopes) >= batch // 2
    assert len(np.unique(slopes[slopes != 0])) == np.count_nonzero(slopes)
    assert np.all(np.abs(slopes) <= env.curriculum[stage].slope_max + 1e-6)


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
    trace = run_episode(env, zero, seed=11, max_steps=50, stage="commands 0.1 m/s")
    assert len(trace["time"]) == len(trace["v_cmd"]) == trace["action"].shape[0]
    assert trace["end"] in {"fall", "max_steps"}
    metrics = summarize(trace)
    assert set(metrics) == set(METRIC_KEYS)
    assert all(np.isfinite(value) for value in metrics.values())

    plot_episode(trace, tmp_path / "episode.png")
    history = {"steps": [1, 2], "stage": [0, 1]}
    for key in METRIC_KEYS:
        history["stage_" + key] = [0.1, 0.2]
        history["final_" + key] = [0.1, 0.2]
    plot_training_metrics(history, tmp_path / "training.png", [s.name for s in env.curriculum])
    for name in ("episode.png", "training.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
