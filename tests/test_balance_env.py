import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import make_env, persistent, reset, set_info, step

from self_balancing_robot.config import RandomizationConfig, RewardConfig
from self_balancing_robot.envs.balance_env import (
    ACTOR_FRAME_SIZE,
    ACTOR_KEY,
    CRITIC_KEY,
    PRIVILEGED_SIZE,
    REWARD_TERMS,
    _tracking_kernel,
    axis_angle_quat,
    quat_to_mat,
    tilt,
)

PITCH = 3  # index of pitch in a frame (actor history and privileged state)


def _pitched(state, pitch, lift=0.0):
    """The state with the robot turned to `pitch` about the axle (and lifted by `lift`)."""
    quat = axis_angle_quat(jnp.array([1.0, 0.0, 0.0]), jnp.float32(pitch))
    qpos = state.data.qpos.at[3:7].set(quat).at[2].add(lift)
    return state.replace(data=state.data.replace(qpos=qpos))


def _mj_data(env, state) -> mujoco.MjData:
    """Full MuJoCo model at the state's pose (for geometry checks)."""
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = np.asarray(state.data.qpos)
    data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
    mujoco.mj_forward(env.mj_model, data)
    return data


def _run(env, state, action, steps):
    for _ in range(steps):
        state = step(env, state, action)
        if state.done:
            break
    return state


def test_environment_reset_and_step(env):
    state = reset(env, seed=123)
    assert state.obs[ACTOR_KEY].shape == (ACTOR_FRAME_SIZE * env.history_length,)
    assert state.obs[CRITIC_KEY].shape == (PRIVILEGED_SIZE,)
    assert env.actor_observation_size == ACTOR_FRAME_SIZE * env.history_length
    data = _mj_data(env, state)
    wheel_bottom = data.geom("left_wheel_collision").xpos[2] - data.model.geom(
        "left_wheel_collision"
    ).size[0]
    assert abs(wheel_bottom) < 1e-3

    state = step(env, state)
    assert np.isfinite(state.reward)
    assert state.done == 0 and state.info["truncation"] == 0
    assert state.info["stage"] == 1
    assert np.all(state.info["command"] == 0.0)
    assert set(state.info["reward_terms"]) == set(REWARD_TERMS)


def test_control_period_is_ten_milliseconds(env):
    # The ESP32 runs the policy every 10 ms; the physics runs whole substeps in between.
    assert np.isclose(env.dt, 0.01)
    assert np.isclose(env.n_substeps * env.sim_dt, env.dt)


def test_unpowered_robot_falls_without_wheel_torque(env):
    state = _run(env, reset(env, seed=0), (0.0, 0.0), int(1.5 / env.dt))
    assert state.done and state.info["terminated"]
    assert state.info["truncation"] == 0


def test_positive_action_drives_forward_and_differential_action_turns(env):
    state = _run(env, reset(env, seed=1), (0.3, 0.3), 5)
    assert state.info["forward_velocity"] > 0.0
    state = _run(env, reset(env, seed=1), (-0.3, 0.3), 5)
    assert state.info["yaw_rate"] > 0.0


def test_reward_components_follow_configured_weights(env):
    state = step(env, reset(env, seed=1))
    assert not state.info["terminated"]
    cfg, terms = env.reward_config, state.info["reward_terms"]
    assert terms["alive"] == 1.0
    expected = sum(getattr(cfg, f"{name}_weight") * terms[name] for name in REWARD_TERMS)
    assert np.isclose(state.reward, expected, rtol=1e-5)


def test_action_penalties_use_policy_action_not_delayed_action(env):
    state = set_info(reset(env, seed=5, stage=3), motor_latency=2)
    state = step(env, state, (0.5, -0.5))
    assert state.info["reward_terms"]["action"] == -0.5
    assert state.info["reward_terms"]["smoothness"] == -0.5


def test_timeout_is_reported_as_truncation(env):
    state = set_info(reset(env, seed=2), episode_step=env.max_episode_steps - 1)
    state = step(env, state)
    assert state.done and not state.info["terminated"]
    assert state.info["truncation"] == 1.0
    assert state.metrics["survived"] == 1.0


def test_pitch_failure_terminates_with_zero_reward(env):
    state = reset(env, seed=3)
    # Lift the robot so the base does not hit the ground first.
    state = _pitched(state, env.environment_config.upright_limit_radians + 0.1, lift=0.05)
    state = step(env, state)
    assert state.done and state.info["terminated"]
    assert state.info["truncation"] == 0
    assert state.reward == 0.0


def test_spinning_in_place_terminates_like_a_fall(env):
    limit = env.environment_config.yaw_rate_limit_rad_s
    assert limit >= 2 * env.curriculum_config.command_limits[1]  # never hit while tracking
    state = reset(env, seed=3)
    qvel = state.data.qvel.at[5].set(5 * limit)  # body z: yaw (wheel friction brakes it fast)
    state = step(env, state.replace(data=state.data.replace(qvel=qvel)))
    assert state.done and state.info["terminated"]
    assert state.reward == 0.0


def test_reset_with_same_seed_reproduces_initial_observation(env):
    first, second = reset(env, seed=4, stage=3), reset(env, seed=4, stage=3)
    np.testing.assert_allclose(first.obs[ACTOR_KEY], second.obs[ACTOR_KEY])
    np.testing.assert_allclose(first.obs[CRITIC_KEY], second.obs[CRITIC_KEY])


def test_actor_history_shifts_newest_frame_first(env):
    state = reset(env, seed=6)
    first = np.asarray(state.obs[ACTOR_KEY])
    # At reset the whole window is filled with the first frame.
    history = first.reshape(env.history_length, ACTOR_FRAME_SIZE)
    np.testing.assert_allclose(history, np.tile(history[0], (env.history_length, 1)))

    second = np.asarray(step(env, state, (0.2, 0.1)).obs[ACTOR_KEY])
    np.testing.assert_allclose(second[ACTOR_FRAME_SIZE : 2 * ACTOR_FRAME_SIZE], first[:ACTOR_FRAME_SIZE])
    np.testing.assert_allclose(second[ACTOR_FRAME_SIZE - 2 : ACTOR_FRAME_SIZE], [0.2, 0.1])


def test_sensor_latency_delays_the_actor_frame_but_not_the_privileged_state():
    env = make_env(
        randomization_config=RandomizationConfig(
            imu_angle_noise_rad=0.0, imu_rate_noise_rad_s=0.0, encoder_velocity_noise_rad_s=0.0
        )
    )
    state = set_info(reset(env, seed=7, stage=3), imu_latency=2)
    observations = []
    for _ in range(4):
        state = step(env, state, (0.3, 0.3))
        observations.append(state.obs)
    assert observations[3][ACTOR_KEY][PITCH] == observations[1][CRITIC_KEY][PITCH]
    assert observations[3][ACTOR_KEY][PITCH] != observations[3][CRITIC_KEY][PITCH]


def test_domain_randomization_changes_physics_only_in_stage_three(env):
    for stage in (1, 2):
        info = reset(env, seed=8, stage=stage, level=env.max_level(stage)).info
        assert info["mass_scale"] == 1.0 and info["friction_scale"] == 1.0
        assert np.all(info["com_offset"] == 0.0) and info["ground_normal"][2] == 1.0
        assert info["motor_latency"] == 0
    info = reset(env, seed=8, stage=3, level=env.max_level(3)).info
    assert info["mass_scale"] != 1.0 and info["friction_scale"] != 1.0
    assert info["ground_normal"][2] < 1.0
    model = env._episode_model(info)
    base = env._base_body_id
    assert np.isclose(model.body_mass[base], env.mj_model.body_mass[base] * info["mass_scale"])


def test_commands_stay_within_wheel_speed(env):
    full_speed = env.curriculum_config.command_limits[0]
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    commands = jax.vmap(lambda key: env.reset_episode(key, persistent(env, 2, 3)).info["command"])(
        keys
    )
    outer_wheel = np.abs(commands[:, 0]) + np.abs(commands[:, 1]) * env.half_track
    assert np.all(outer_wheel <= full_speed + 1e-6)
    linear, angular = env.wheel_feasible_command(full_speed, -1.5)  # joystick corner
    assert np.isclose(abs(linear) + abs(angular) * env.half_track, full_speed)
    assert np.isclose(angular / linear, -1.5 / full_speed)
    np.testing.assert_allclose(env.wheel_feasible_command(0.2, 0.5), [0.2, 0.5])  # feasible


def test_command_ramp_widens_commands_level_by_level(env):
    cfg = env.curriculum_config
    full = np.array(cfg.command_limits)
    np.testing.assert_allclose(env.command_limits(1, 1), 0.0)
    for level in (1, cfg.command_levels):
        limits = np.array(env.command_limits(2, level))
        np.testing.assert_allclose(limits, full * level / cfg.command_levels, rtol=1e-6)
        for seed in range(20):
            command = reset(env, seed=seed, stage=2, level=level).info["command"]
            assert np.all(np.abs(command) <= limits + 1e-6)
        assert env.slope_limit(2, level) == 0.0 and env.push_limit(2, level) == 0.0
    np.testing.assert_allclose(env.command_limits(3, 1), full)


def test_reversals_flip_direction_near_full_speed(env):
    cfg = env.curriculum_config
    info = reset(env, seed=21, stage=2, level=3).info
    info = {**info, "command": jnp.array([0.25, 0.5])}
    limit = float(env.command_limits(2, 3)[0])
    keys = jax.random.split(jax.random.PRNGKey(1), 4_000)

    def sample(allow_reversal, info=info):
        return jax.vmap(lambda key: env._sample_command(key, info, allow_reversal))(keys)

    commands, _, reversals = sample(True)
    commands, reversals = np.asarray(commands), np.asarray(reversals)
    assert np.isclose(reversals.mean(), cfg.reversal_probability, atol=0.02)
    linear = -commands[reversals, 0]
    assert np.all((linear >= cfg.reversal_min_fraction * limit - 1e-6) & (linear <= limit + 1e-6))
    assert np.all(commands[reversals, 1] == 0.0)
    assert not np.any(sample(False)[2])  # never at reset
    stage_one = {**info, "stage": jnp.int32(1), "level": jnp.int32(1)}
    commands, _, reversals = sample(True, stage_one)  # no commands in stage 1
    assert not np.any(reversals) and np.all(np.asarray(commands) == 0.0)


def test_speed_tracking_skips_acceleration_and_small_commands(env):
    info = reset(env, seed=0, stage=2, level=3).info
    settle_steps = int(np.ceil(env.curriculum_config.speed_ratio_settle_seconds / env.dt))

    def track(command, steps, forward_velocity=0.2):
        tracked = {
            **info,
            "command": jnp.asarray(command, jnp.float32),
            "command_age": jnp.float32(0.0),
            "episode_tracking": jnp.zeros(4),
        }
        for _ in range(steps):
            tracked = env._track_command(tracked, jnp.float32(forward_velocity))
        return np.asarray(tracked["episode_tracking"])

    assert np.all(track((0.3, 0.0), settle_steps - 1) == 0.0)  # still accelerating
    reached, commanded, _, _ = track((0.3, 0.0), settle_steps + 10)
    assert commanded > 0 and np.isclose(reached / commanded, 0.2 / 0.3)
    reached, commanded, _, _ = track((-0.3, 0.0), settle_steps + 10)
    assert np.isclose(reached / commanded, -0.2 / 0.3)  # wrong direction counts against
    assert np.all(track((0.08, 0.0), settle_steps + 10) == 0.0)  # under 1/3 of 0.3 m/s
    _, _, still_speed, still_steps = track((0.0, 1.0), settle_steps + 10)
    assert still_steps > 0 and np.isclose(still_speed / still_steps, 0.2)


def test_held_command_is_not_resampled(env):
    state = reset(env, seed=10, stage=2, level=3, command_held=True, command=[0.2, -0.5])

    def body(state, _):
        state = env.step(state, jnp.zeros(2))
        return state, state.info["command"]

    _, commands = jax.jit(lambda s: jax.lax.scan(body, s, None, int(6.0 / env.dt)))(state)
    np.testing.assert_allclose(commands, np.tile([0.2, -0.5], (len(commands), 1)), rtol=1e-6)


def test_physics_overrides_pin_friction_com_and_slope(env):
    pinned = {
        "override_slope": 0.3,
        "override_friction": 0.5,
        "override_com": [0.0, 0.004, -0.002],
    }
    for seed in range(3):  # overrides beat stage-3 randomization at every reset
        info = reset(env, seed=seed, stage=3, level=6, **pinned).info
        assert np.isclose(info["slope"], 0.3)
        assert np.isclose(info["friction_scale"] * env.nominal_ground_friction, 0.5)
        np.testing.assert_allclose(info["com_offset"], [0.0, 0.004, -0.002])
        model = env._episode_model(info)
        np.testing.assert_allclose(model.geom_friction[:, 0], 0.5, rtol=1e-6)
    info = reset(env, seed=13, stage=3, level=6).info
    assert not np.isclose(info["friction_scale"] * env.nominal_ground_friction, 0.5)


def test_robot_spawns_near_vertical_with_wheels_on_steep_slope(env):
    model = env.mj_model
    ground = model.geom("ground").id
    for seed in range(10):
        state = reset(env, seed=seed, override_slope=0.5)
        data = _mj_data(env, state)
        # The collision/render ground itself is tilted, not just the bookkeeping.
        ground_normal = data.geom_xmat[ground].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(ground_normal, state.info["ground_normal"], atol=1e-6)
        assert np.isclose(np.arccos(ground_normal[2]), 0.5, atol=1e-6)
        _, pitch = tilt(quat_to_mat(state.data.qpos[3:7]))
        assert abs(pitch) <= env.environment_config.initial_pitch_radians + 1e-6
        for name in ("left_wheel_collision", "right_wheel_collision"):
            distance = mujoco.mj_geomDistance(model, data, ground, model.geom(name).id, 1.0, None)
            assert abs(distance) < 1e-3


def test_pitch_rate_is_penalized_instead_of_pitch_angle(env):
    state = step(env, _pitched(reset(env, seed=14), 0.2))
    pitch_rate = state.data.qvel[3]  # about the axle (body x)
    assert np.isclose(state.info["reward_terms"]["pitch_rate"], -(pitch_rate**2))
    assert "pitch" not in state.info["reward_terms"]


def test_tracking_reward_still_rewards_progress_far_from_the_command():
    cfg = RewardConfig()
    reward = [
        float(
            _tracking_kernel(
                error,
                cfg.linear_velocity_variance,
                cfg.linear_velocity_wide_variance,
                cfg.linear_velocity_wide_fraction,
            )
        )
        for error in (0.35, 0.2, 0.1, 0.0)
    ]
    assert reward == sorted(reward)
    assert reward[1] - reward[0] > 0.05  # going from 0.05 to 0.2 m/s on a 0.4 command pays
    assert reward[-1] == 1.0


def test_stage_three_levels_ramp_slope_and_push_together(env):
    randomization = env.randomization_config
    for seed in range(5):
        info = reset(env, seed=seed, stage=3, level=1).info
        assert abs(info["slope"]) <= env.slope_limit(3, 1) + 1e-6
        assert np.all(np.abs(info["com_offset"]) <= env.com_offset_limit(3, 1) + 1e-6)
    last = env.max_level(3)
    limits = [float(env.slope_limit(3, level)) for level in range(1, last + 1)]
    assert limits == sorted(limits)
    assert np.isclose(env.slope_limit(3, last), randomization.slope_radians)
    assert np.isclose(env.push_limit(3, last), randomization.impulse_force_n)
    assert np.isclose(env.com_offset_limit(3, last), randomization.center_of_mass_offset_m)


def test_level_sampling_favours_the_newest_band_but_keeps_easier_values(env):
    cfg = env.curriculum_config
    levels = cfg.randomization_levels
    limit = env.slope_limit(3, levels)
    band_floor = limit - limit / levels

    def samples(level, count):
        keys = jax.random.split(jax.random.PRNGKey(level), count)
        return np.asarray(jax.vmap(lambda key: env.sample_level_magnitude(key, limit, level))(keys))

    values = samples(levels, 20_000)
    assert np.all((values >= 0) & (values <= limit + 1e-6))
    assert np.isclose(np.mean(values == 0), cfg.level_zero_probability, atol=0.01)
    assert np.isclose(np.mean(values >= band_floor), cfg.level_frontier_probability, atol=0.015)
    assert np.isclose(np.mean((values > 0) & (values < band_floor)), 0.5, atol=0.015)
    values = samples(1, 5_000)  # nothing below the first band: it gets that share
    assert np.isclose(np.mean(values == 0), cfg.level_zero_probability, atol=0.015)


def test_yaw_rate_reward_uses_the_narrow_kernel_only(env):
    state = step(env, reset(env, seed=17, stage=2, command_held=True, command=[0.0, 1.0]))
    error = state.info["yaw_rate"] - 1.0
    expected = np.exp(-(error**2) / env.reward_config.angular_velocity_variance)
    assert np.isclose(state.info["reward_terms"]["angular_velocity"], expected, rtol=1e-5)


@pytest.mark.parametrize("stage", [1, 3])
def test_jitted_step_keeps_the_state_structure(env, stage):
    # reset and step must return identical pytrees for lax.scan and the auto-reset select.
    state = reset(env, seed=0, stage=stage)
    stepped = step(env, state)
    assert jax.tree.structure(state) == jax.tree.structure(stepped)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
