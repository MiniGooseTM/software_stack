from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import make_env, persistent, reset, set_info, step

from self_balancing_robot.config import RewardConfig
from self_balancing_robot.curriculum import (
    BALANCE,
    CURRICULUM,
    FULL_COMMAND,
    NOMINAL_VOLTAGE,
    RANDOMIZATION_1,
    REVERSAL_MIN_FRACTION,
    SPEED_RATIO_SETTLE_S,
)
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
# Stages of the default curriculum used below.
COMMANDS_LOW = "commands 0.1 m/s"
COMMANDS_FULL = "commands 0.5 m/s"
RANDOMIZED = "randomization 1/6"
LAST = -1


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
    assert state.info["stage"] == 0
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


@pytest.mark.parametrize("stage", [0, LAST])
def test_reward_components_follow_the_stages_weights(env, stage):
    state = step(env, reset(env, seed=1, stage=stage), (0.2, -0.1))
    assert not state.info["terminated"]
    cfg, terms = CURRICULUM[stage].rewards, state.info["reward_terms"]
    assert terms["alive"] == 1.0
    expected = sum(getattr(cfg, f"{name}_weight") * terms[name] for name in REWARD_TERMS)
    assert np.isclose(state.reward, expected, rtol=1e-5)


def test_last_stages_weigh_smoothness_more():
    weights = [stage.rewards.smoothness_weight for stage in CURRICULUM]
    assert weights[-2:] == [5.0, 10.0]
    assert all(w == RewardConfig().smoothness_weight for w in weights[:-2])


def test_action_penalties_use_policy_action_not_delayed_action(env):
    state = set_info(reset(env, seed=5, stage=RANDOMIZED), motor_latency=2)
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
    assert limit >= 2 * FULL_COMMAND[1]  # never hit while tracking
    state = reset(env, seed=3)
    qvel = state.data.qvel.at[5].set(5 * limit)  # body z: yaw (wheel friction brakes it fast)
    state = step(env, state.replace(data=state.data.replace(qvel=qvel)))
    assert state.done and state.info["terminated"]
    assert state.reward == 0.0


def test_reset_with_same_seed_reproduces_initial_observation(env):
    first, second = reset(env, seed=4, stage=LAST), reset(env, seed=4, stage=LAST)
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
    quiet = replace(RANDOMIZATION_1, imu_angle_noise=0.0, imu_rate_noise=0.0, encoder_noise=0.0)
    env = make_env(curriculum=(BALANCE, quiet))
    state = set_info(reset(env, seed=7, stage=1), imu_latency=2)
    observations = []
    for _ in range(4):
        state = step(env, state, (0.3, 0.3))
        observations.append(state.obs)
    assert observations[3][ACTOR_KEY][PITCH] == observations[1][CRITIC_KEY][PITCH]
    assert observations[3][ACTOR_KEY][PITCH] != observations[3][CRITIC_KEY][PITCH]


def test_physics_is_randomized_only_where_the_stage_says_so(env):
    for stage in ("balance", COMMANDS_FULL):
        info = reset(env, seed=8, stage=stage).info
        assert info["mass_scale"] == 1.0 and info["friction_scale"] == 1.0
        assert np.all(info["com_offset"] == 0.0) and info["ground_normal"][2] == 1.0
        assert np.isclose(info["battery_voltage"], NOMINAL_VOLTAGE)
    assert reset(env, seed=8, stage="balance").info["motor_latency"] == 0
    info = reset(env, seed=8, stage=LAST).info
    assert info["mass_scale"] != 1.0 and info["friction_scale"] != 1.0
    assert info["ground_normal"][2] < 1.0
    model = env._episode_model(info)
    base = env._base_body_id
    assert np.isclose(model.body_mass[base], env.mj_model.body_mass[base] * info["mass_scale"])


def test_commands_stay_within_wheel_speed(env):
    full_speed = FULL_COMMAND[0]
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    start = persistent(env, COMMANDS_FULL)
    commands = jax.vmap(lambda key: env.reset_episode(key, start).info["command"])(keys)
    outer_wheel = np.abs(commands[:, 0]) + np.abs(commands[:, 1]) * env.half_track
    assert np.all(outer_wheel <= full_speed + 1e-6)
    linear, angular = env.wheel_feasible_command(full_speed, -1.5)  # joystick corner
    assert np.isclose(abs(linear) + abs(angular) * env.half_track, full_speed)
    assert np.isclose(angular / linear, -1.5 / full_speed)
    np.testing.assert_allclose(env.wheel_feasible_command(0.2, 0.5), [0.2, 0.5])  # feasible


def test_samples_stay_within_each_stages_maxima(env):
    keys = jax.random.split(jax.random.PRNGKey(2), 50)
    for index, stage in enumerate(CURRICULUM):
        start = persistent(env, index)
        info = jax.vmap(lambda key, start=start: env.reset_episode(key, start).info)(keys)
        assert np.all(np.abs(info["command"]) <= np.array(stage.command_max) + 1e-6), stage.name
        assert np.all(np.abs(info["slope"]) <= stage.slope_max + 1e-6), stage.name
        assert np.all(np.abs(info["com_offset"]) <= stage.com_offset_max + 1e-6), stage.name
        assert np.all(np.abs(info["mass_scale"] - 1) <= stage.mass_fraction + 1e-6), stage.name
        low, high = stage.battery_voltage
        voltage = info["battery_voltage"]
        assert np.all((voltage >= low - 1e-5) & (voltage <= high + 1e-5)), stage.name
        assert np.all(info["motor_latency"] <= stage.max_latency_steps), stage.name
        if stage.command_max[0] > 0:  # the new band is actually reached
            assert np.abs(info["command"][:, 0]).max() > 0.5 * stage.command_max[0], stage.name


def test_commands_are_full_stops_at_the_stop_probability(env):
    stage = CURRICULUM[env.stage_index(COMMANDS_FULL)]
    info = reset(env, seed=21, stage=COMMANDS_FULL).info
    keys = jax.random.split(jax.random.PRNGKey(3), 4_000)
    commands, _, reversals = jax.vmap(lambda key: env._sample_command(key, info, True))(keys)
    stops = np.all(np.asarray(commands) == 0.0, axis=1)
    # Plus ~1% from speed and yaw rate both drawing 0.
    expected = (1 - stage.reversal_probability) * (stage.stop_probability + 0.01)
    assert np.isclose(stops.mean(), expected, atol=0.02)
    assert not np.any(stops & np.asarray(reversals))


def test_reversals_flip_direction_near_full_speed(env):
    stage = CURRICULUM[env.stage_index(COMMANDS_FULL)]
    info = reset(env, seed=21, stage=COMMANDS_FULL).info
    info = {**info, "command": jnp.array([0.25, 0.5])}
    limit = stage.command_max[0]
    keys = jax.random.split(jax.random.PRNGKey(1), 4_000)

    def sample(allow_reversal, info=info):
        return jax.vmap(lambda key: env._sample_command(key, info, allow_reversal))(keys)

    commands, _, reversals = sample(True)
    commands, reversals = np.asarray(commands), np.asarray(reversals)
    assert np.isclose(reversals.mean(), stage.reversal_probability, atol=0.02)
    linear = -commands[reversals, 0]
    assert np.all((linear >= REVERSAL_MIN_FRACTION * limit - 1e-6) & (linear <= limit + 1e-6))
    assert np.all(commands[reversals, 1] == 0.0)
    assert not np.any(sample(False)[2])  # never at reset
    balance = {**info, "stage": jnp.int32(0)}
    commands, _, reversals = sample(True, balance)  # no commands while balancing
    assert not np.any(reversals) and np.all(np.asarray(commands) == 0.0)


def test_speed_tracking_skips_acceleration_and_small_commands(env):
    info = reset(env, seed=0, stage=COMMANDS_FULL).info
    settle_steps = int(np.ceil(SPEED_RATIO_SETTLE_S / env.dt))

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
    state = reset(env, seed=10, stage=COMMANDS_FULL, command_held=True, command=[0.2, -0.5])

    def body(state, _):
        state = env.step(state, jnp.zeros(2))
        return state, state.info["command"]

    _, commands = jax.jit(lambda s: jax.lax.scan(body, s, None, int(6.0 / env.dt)))(state)
    np.testing.assert_allclose(commands, np.tile([0.2, -0.5], (len(commands), 1)), rtol=1e-6)


def test_battery_voltage_scales_motor_speed_and_torque(env):
    nominal = env.mj_model
    wheel_speeds = []
    for voltage in (10.5, 12.5):
        state = set_info(reset(env, seed=6, stage=LAST), battery_voltage=voltage, motor_latency=0)
        model = env._episode_model(state.info)
        scale = voltage / NOMINAL_VOLTAGE
        np.testing.assert_allclose(
            model.actuator_ctrlrange, nominal.actuator_ctrlrange * scale, rtol=1e-6
        )
        np.testing.assert_allclose(
            model.actuator_forcerange, nominal.actuator_forcerange * scale, rtol=1e-6
        )
        state = _run(env, state, (1.0, 1.0), 5)
        wheel_speeds.append(float(state.data.qvel[env._wheel_qvel].mean()))
    # Full duty spins the wheels up faster on a charged battery.
    assert wheel_speeds[1] > wheel_speeds[0] > 0.0


def test_wheel_kicks_act_on_the_wheel_joints(env):
    """A kick is applied as a torque on the wheel joints while it lasts, nothing otherwise."""
    stage = CURRICULUM[LAST]
    state = set_info(
        reset(env, seed=4, stage=LAST),
        next_wheel_kick_time=0.05,
        next_impulse_time=1e3,
    )
    wheels = env._wheel_qvel
    kicks = []
    for _ in range(int(0.2 / env.dt)):
        state = step(env, state)
        assert not state.done
        applied = state.data.qfrc_applied[wheels]
        np.testing.assert_allclose(applied, state.info["wheel_kick"], atol=1e-7)
        kicks.append(np.any(state.info["wheel_kick"] != 0.0))
    # Steps 1-4 (t = 0..0.03 s) before the kick at 0.05 s, which lasts wheel_kick_duration_s.
    assert not any(kicks[:5])
    assert sum(kicks) == round(stage.wheel_kick_duration_s / env.dt)
    assert state.info["next_wheel_kick_time"] >= 0.05 + stage.wheel_kick_interval_s[0]


def test_wheel_kicks_only_where_the_stage_has_them(env):
    state = set_info(reset(env, seed=4, stage=COMMANDS_FULL), next_wheel_kick_time=0.0)
    state = step(env, state)
    assert np.all(state.info["wheel_kick"] == 0.0)
    assert np.all(state.data.qfrc_applied == 0.0)


def test_physics_overrides_pin_friction_com_and_slope(env):
    pinned = {
        "override_slope": 0.3,
        "override_friction": 0.5,
        "override_com": [0.0, 0.004, -0.002],
    }
    for seed in range(3):  # overrides beat stage-3 randomization at every reset
        info = reset(env, seed=seed, stage=LAST, **pinned).info
        assert np.isclose(info["slope"], 0.3)
        assert np.isclose(info["friction_scale"] * env.nominal_ground_friction, 0.5)
        np.testing.assert_allclose(info["com_offset"], [0.0, 0.004, -0.002])
        model = env._episode_model(info)
        np.testing.assert_allclose(model.geom_friction[:, 0], 0.5, rtol=1e-6)
    info = reset(env, seed=13, stage=LAST).info
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


@pytest.mark.parametrize("floor", [0.25, 0.0])
def test_magnitude_sampling_favours_the_new_band_but_keeps_easier_values(env, floor):
    maximum, zero, frontier = 0.3, 0.1, 0.4
    keys = jax.random.split(jax.random.PRNGKey(0), 20_000)
    values = np.asarray(
        jax.vmap(lambda key: env.sample_magnitude(key, floor, maximum, zero, frontier))(keys)
    )
    assert np.all((values >= 0) & (values <= maximum + 1e-6))
    assert np.isclose(np.mean(values == 0), zero, atol=0.01)
    if floor > 0:
        assert np.isclose(np.mean(values >= floor), frontier, atol=0.015)
        assert np.isclose(np.mean((values > 0) & (values < floor)), 1 - zero - frontier, atol=0.015)
    else:  # no band: every non-zero draw is uniform up to the maximum
        assert np.isclose(np.mean(values >= maximum / 2), (1 - zero) / 2, atol=0.015)


def test_yaw_rate_reward_uses_the_narrow_kernel_only(env):
    state = step(env, reset(env, seed=17, stage=COMMANDS_LOW, command_held=True, command=[0.0, 1.0]))
    error = state.info["yaw_rate"] - 1.0
    variance = CURRICULUM[env.stage_index(COMMANDS_LOW)].rewards.angular_velocity_variance
    expected = np.exp(-(error**2) / variance)
    assert np.isclose(state.info["reward_terms"]["angular_velocity"], expected, rtol=1e-5)


@pytest.mark.parametrize("stage", [0, LAST])
def test_jitted_step_keeps_the_state_structure(env, stage):
    # reset and step must return identical pytrees for lax.scan and the auto-reset select.
    state = reset(env, seed=0, stage=stage)
    stepped = step(env, state)
    assert jax.tree.structure(state) == jax.tree.structure(stepped)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
