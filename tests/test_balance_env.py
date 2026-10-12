import mujoco
import numpy as np
import pytest
import torch

from self_balancing_robot import SelfBalancingRobotEnv
from self_balancing_robot.config import EnvironmentConfig, RandomizationConfig
from self_balancing_robot.envs.balance_env import ACTOR_FRAME_SIZE, PRIVILEGED_SIZE, RewardConfig
from self_balancing_robot.metrics import (
    METRIC_KEYS,
    plot_episode,
    plot_training_metrics,
    run_episode,
    summarize,
)
from self_balancing_robot.policy import AsymmetricActorCriticPolicy
from self_balancing_robot.training_callback import (
    EvaluationVideoCallback,
    TrainingMetricsCallback,
)


def _set_pitch(env, pitch):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0.0, 0.0]), pitch)
    env.data.qpos[3:7] = quat
    mujoco.mj_forward(env.model, env.data)


def test_environment_reset_and_step():
    env = SelfBalancingRobotEnv()
    observation, info = env.reset(seed=123)
    expected_size = ACTOR_FRAME_SIZE * env.history_length + PRIVILEGED_SIZE
    assert observation.shape == (expected_size,)
    assert env.actor_observation_size == ACTOR_FRAME_SIZE * env.history_length
    assert env.observation_space.contains(observation)
    assert info == {}
    wheel_bottom = env.data.geom("left_wheel_collision").xpos[2] - env._wheel_radius
    assert abs(wheel_bottom) < 1e-3

    next_observation, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert next_observation.shape == (expected_size,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "pitch" in info
    assert info["curriculum_stage"] == 1
    assert info["v_cmd"] == 0.0
    assert "reward_linear_velocity" in info
    env.close()


def test_unpowered_robot_falls_without_wheel_torque():
    env = SelfBalancingRobotEnv()
    env.reset(seed=0)

    for _ in range(int(1.5 / env.dt)):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert info["terminated_reason"] in {"pitch_failure", "base_collision"}
    env.close()


def test_positive_action_drives_forward_and_differential_action_turns():
    env = SelfBalancingRobotEnv()
    env.reset(seed=1)
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.3, 0.3]))
    assert info["forward_velocity"] > 0.0

    env.reset(seed=1)
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([-0.3, 0.3]))
    assert info["yaw_rate"] > 0.0
    env.close()


def test_render_fps_uses_the_simulation_timestep_and_frame_skip():
    env = SelfBalancingRobotEnv(environment_config=EnvironmentConfig(frame_skip=3))
    assert env.frame_skip == 3
    assert env.metadata["render_fps"] == int(np.round(1.0 / env.dt))
    assert "render_fps" not in SelfBalancingRobotEnv.metadata
    env.close()


def test_training_start_replaces_old_plot_with_current_run(tmp_path):
    plot_path = tmp_path / "training_metrics.png"
    plot_path.write_bytes(b"stale plot")
    callback = TrainingMetricsCallback(plot_dir=tmp_path)
    callback.history["steps"].append(100)
    callback.history["test_reward"].append(0.4)

    callback._on_training_start()
    callback.eval_env.close()

    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not callback.history


def test_training_evaluation_callbacks_use_training_randomization_setting():
    metrics_callback = TrainingMetricsCallback(domain_randomization=False)
    video_callback = None
    try:
        video_callback = EvaluationVideoCallback(domain_randomization=False)
        assert metrics_callback.eval_env.domain_randomization is False
        assert video_callback.video_env.domain_randomization is False
    finally:
        metrics_callback.eval_env.close()
        if video_callback is not None:
            video_callback.video_env.close()


class _ZeroPolicy:
    def predict(self, observation, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_episode_trace_metrics_and_plots(tmp_path):
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    trace = run_episode(env, _ZeroPolicy(), seed=11, max_steps=50)
    env.close()
    assert len(trace["time"]) == len(trace["v_cmd"]) == trace["action"].shape[0]
    assert len(trace["position_x"]) == len(trace["position_y"]) == len(trace["time"])
    assert trace["initial_position"].shape == (2,)
    assert trace["end"] in {"pitch_failure", "base_collision", "max_steps"}
    metrics = summarize(trace)
    assert set(metrics) == set(METRIC_KEYS)
    assert all(np.isfinite(value) for value in metrics.values())

    plot_episode(trace, tmp_path / "episode.png")
    history = {"steps": [1, 2], "stage_min": [1, 2], "stage_mean": [1.5, 2.0]}
    for key in METRIC_KEYS:
        history["train_" + key] = [0.1, 0.2]
        history["test_" + key] = [0.1, 0.2]
    plot_training_metrics(history, tmp_path / "training.png")
    for name in ("episode.png", "training.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_sensor_update_intervals_hold_samples_between_sensor_reads():
    from self_balancing_robot.config import EnvironmentConfig

    env = SelfBalancingRobotEnv(
        environment_config=EnvironmentConfig(
            imu_update_interval_steps=2,
            encoder_update_interval_steps=3,
        )
    )
    env.reset(seed=17)
    imu_readings = iter((np.full(5, 1.0), np.full(5, 2.0)))
    encoder_readings = iter((np.full(2, 3.0), np.full(2, 4.0)))
    env._imu_measurement = lambda: next(imu_readings)
    env._encoder_measurement = lambda: next(encoder_readings)

    env._episode_step = 1
    held = env._actor_frame()
    np.testing.assert_allclose(held[2:7], env._held_imu_measurement)
    np.testing.assert_allclose(held[7:9], env._held_encoder_measurement)

    env._episode_step = 2
    new_imu = env._actor_frame()
    np.testing.assert_allclose(new_imu[2:7], np.full(5, 1.0))
    np.testing.assert_allclose(new_imu[7:9], env._held_encoder_measurement)

    env._episode_step = 3
    new_encoders = env._actor_frame()
    np.testing.assert_allclose(new_encoders[2:7], env._held_imu_measurement)
    np.testing.assert_allclose(new_encoders[7:9], np.full(2, 3.0))
    env.close()


def test_sensor_noise_does_not_change_domain_randomization_rng():
    from self_balancing_robot.config import EnvironmentConfig

    full_rate = SelfBalancingRobotEnv(curriculum_stage=3)
    reduced_rate = SelfBalancingRobotEnv(
        curriculum_stage=3,
        environment_config=EnvironmentConfig(
            imu_update_interval_steps=4,
            encoder_update_interval_steps=4,
        ),
    )
    full_rate.reset(seed=29)
    reduced_rate.reset(seed=29)
    np.testing.assert_allclose(full_rate.command, reduced_rate.command)
    for step in range(1, 12):
        full_rate._episode_step = reduced_rate._episode_step = step
        full_rate._actor_frame()
        reduced_rate._actor_frame()
    assert full_rate.np_random.random() == reduced_rate.np_random.random()
    full_rate.close()
    reduced_rate.close()


def test_sensor_update_intervals_must_be_positive():
    from self_balancing_robot.config import EnvironmentConfig

    with pytest.raises(ValueError, match="Sensor update intervals"):
        SelfBalancingRobotEnv(
            environment_config=EnvironmentConfig(imu_update_interval_steps=0)
        )


def test_reward_components_follow_configured_weights():
    env = SelfBalancingRobotEnv(
        reward_config=RewardConfig(
            alive_weight=2.0,
            pitch_rate_weight=0.0,
            linear_velocity_weight=0.0,
            angular_velocity_weight=0.0,
            action_weight=0.0,
            smoothness_weight=0.0,
        )
    )
    env.reset(seed=1)
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert reward == 2.0
    assert info["reward_alive"] == 1.0
    env.close()


def test_action_penalties_use_policy_action_not_delayed_action():
    env = SelfBalancingRobotEnv(curriculum_stage=3)
    env.reset(seed=5)
    env._motor_latency = 2
    action = np.array([0.5, -0.5])
    _, _, _, _, info = env.step(action)
    assert info["reward_action"] == -0.5
    assert info["reward_smoothness"] == -0.5
    env.close()


def test_timeout_is_reported_as_truncation():
    env = SelfBalancingRobotEnv(max_episode_steps=1)
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["terminated_reason"] == "timeout"
    env.close()


def test_pitch_failure_terminates_with_zero_reward():
    env = SelfBalancingRobotEnv()
    env.reset(seed=3)
    _set_pitch(env, env.environment_config.upright_limit_radians + 0.1)
    env.data.qpos[2] += 0.05  # lift the robot so the base does not hit the ground first
    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert not truncated
    assert reward == 0.0
    assert info["terminated_reason"] == "pitch_failure"
    env.close()


def test_reset_with_same_seed_reproduces_initial_observation():
    env = SelfBalancingRobotEnv(curriculum_stage=3)
    first, _ = env.reset(seed=4)
    second, _ = env.reset(seed=4)
    np.testing.assert_allclose(first, second)
    env.close()


def test_actor_history_shifts_newest_frame_first():
    env = SelfBalancingRobotEnv()
    first, _ = env.reset(seed=6)
    frame = slice(0, ACTOR_FRAME_SIZE)
    # At reset the whole window is filled with the first frame.
    history = first[: env.actor_observation_size].reshape(env.history_length, ACTOR_FRAME_SIZE)
    np.testing.assert_allclose(history, np.tile(history[0], (env.history_length, 1)))

    second, _, _, _, _ = env.step(np.array([0.2, 0.1]))
    np.testing.assert_allclose(second[ACTOR_FRAME_SIZE : 2 * ACTOR_FRAME_SIZE], first[frame])
    np.testing.assert_allclose(second[frame][-2:], [0.2, 0.1])
    env.close()


def test_sensor_latency_delays_the_actor_frame_but_not_the_privileged_state():
    env = SelfBalancingRobotEnv(
        curriculum_stage=3,
        randomization_config=RandomizationConfig(
            imu_angle_noise_rad=0.0, imu_rate_noise_rad_s=0.0, encoder_velocity_noise_rad_s=0.0
        ),
    )
    env.reset(seed=7)
    env._imu_latency = 2
    observations = [env.step(np.array([0.3, 0.3]))[0] for _ in range(4)]
    pitch_index = 3
    privileged_pitch = env.actor_observation_size + pitch_index
    assert observations[3][pitch_index] == observations[1][privileged_pitch]
    assert observations[3][pitch_index] != observations[3][privileged_pitch]
    env.close()


def test_domain_randomization_changes_physics_only_in_stage_three():
    env = SelfBalancingRobotEnv()
    nominal_mass = env.model.body_mass[env._base_body_id]
    env.reset(seed=8)
    assert env.model.body_mass[env._base_body_id] == nominal_mass
    env.set_curriculum_stage(3)
    env.reset(seed=8)
    assert env.model.body_mass[env._base_body_id] != nominal_mass
    assert env._ground_normal[2] < 1.0
    env.set_curriculum_stage(1)
    env.reset(seed=8)
    assert env.model.body_mass[env._base_body_id] == nominal_mass
    env.close()


def test_stage_three_keeps_commands_without_domain_randomization():
    env = SelfBalancingRobotEnv(curriculum_stage=3, domain_randomization=False)
    nominal_mass = env.model.body_mass[env._base_body_id]
    env.reset(seed=18)

    assert env.curriculum_stage == 3
    np.testing.assert_allclose(env.command_limits, env.curriculum_config.command_limits)
    assert env.model.body_mass[env._base_body_id] == nominal_mass
    np.testing.assert_allclose(env._com_offset, np.zeros(3))
    assert env._friction_scale == 1.0
    assert (env._motor_latency, env._imu_latency, env._encoder_latency) == (0, 0, 0)
    assert env.physics["slope"] == 0.0

    class UnexpectedSensorNoise:
        def normal(self, *args, **kwargs):
            raise AssertionError("domain-randomization sensor noise should be disabled")

    env._sensor_random = UnexpectedSensorNoise()
    env._episode_step = 1
    env._actor_frame()
    env._next_impulse_time = 0.0
    env._update_disturbance()
    np.testing.assert_array_equal(env._push_force, np.zeros(2))
    env.close()


def _finish_episodes(
    env, count, reached=1.0, commanded=1.0, still_speed=0.0, survive=True, reversal=False
):
    """Feed ``count`` finished episodes with the given tracking sums to the curriculum."""
    for _ in range(count):
        env._episode_tracking[:] = (reached, commanded, still_speed, 1.0)
        env._episode_reversal = reversal
        env._advance_curriculum(episode_success=survive)


def test_curriculum_ramps_commands_then_reaches_domain_randomization():
    env = SelfBalancingRobotEnv()
    cfg = env.curriculum_config
    _finish_episodes(env, cfg.evaluation_window)
    assert (env.curriculum_stage, env.difficulty_level) == (2, 1)
    for level in range(2, cfg.command_levels + 1):
        _finish_episodes(env, cfg.command_min_episodes_per_level)
        assert env.difficulty_level == level
    _finish_episodes(env, cfg.command_min_episodes_per_level)
    assert (env.curriculum_stage, env.difficulty_level) == (3, 1)
    env.close()


def test_commands_stay_within_wheel_speed():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    env.set_difficulty_level(env.max_level)
    assert env.curriculum_config.command_limits == (0.4, 2.5)
    full_speed = env.curriculum_config.command_limits[0]
    for seed in range(200):
        env.reset(seed=seed)
        linear, angular = env.command
        assert abs(linear) + abs(angular) * env._half_track <= full_speed + 1e-9
    angular_limit = env.curriculum_config.command_limits[1]
    env.hold_command(full_speed, -angular_limit)  # corner scales both, preserving direction
    linear, angular = env.command
    assert np.isclose(abs(linear) + abs(angular) * env._half_track, full_speed)
    assert np.isclose(angular / linear, -angular_limit / full_speed)
    env.hold_command(0.2, 0.5)  # already feasible: unchanged
    np.testing.assert_allclose(env.command, [0.2, 0.5])
    env.close()


def test_actor_command_scaling_uses_new_full_command_limits():
    env = SelfBalancingRobotEnv(curriculum_stage=3, domain_randomization=False)
    env.hold_command(0.4, 0.0)
    np.testing.assert_allclose(env._scaled_command(), [1.0, 0.0])
    env.hold_command(0.0, 2.5)
    np.testing.assert_allclose(env._scaled_command(), [0.0, 1.0])
    env.close()


def test_command_ramp_widens_commands_level_by_level():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    cfg = env.curriculum_config
    full = np.array(cfg.command_limits)
    for level in (1, cfg.command_levels):
        env.set_difficulty_level(level)
        limits = np.array(env.command_limits)
        np.testing.assert_allclose(limits, full * level / cfg.command_levels)
        for seed in range(20):
            env.reset(seed=seed)
            assert np.all(np.abs(env.command) <= limits + 1e-12)
    assert env.slope_limit == 0.0 and env.push_limit == 0.0
    env.set_curriculum_stage(3)
    np.testing.assert_allclose(env.command_limits, full)
    env.close()


@pytest.mark.parametrize("stage", [2, 3])
def test_levels_require_speed_survival_and_standing_still(stage):
    cfg = SelfBalancingRobotEnv().curriculum_config
    episodes = max(cfg.command_min_episodes_per_level, cfg.randomization_min_episodes_per_level)
    ratio = cfg.command_speed_ratio if stage == 2 else cfg.randomization_speed_ratio
    still_limit = (
        cfg.command_still_speed_limit if stage == 2 else cfg.randomization_still_speed_limit
    )
    for failure in (
        {"reached": ratio - 0.05},  # too slow (v5 drove at ~60%)
        {"still_speed": still_limit + 0.005},  # rolls away when told to stop
    ):
        env = SelfBalancingRobotEnv(curriculum_stage=stage)
        _finish_episodes(env, episodes, **failure)
        assert env.difficulty_level == 1, failure
        env.close()
    env = SelfBalancingRobotEnv(curriculum_stage=stage)
    for episode in range(episodes):  # 80% survive: below both success rates
        _finish_episodes(env, 1, survive=episode % 5 != 0)
    assert env.difficulty_level == 1
    env.close()
    env = SelfBalancingRobotEnv(curriculum_stage=stage)
    minimum = (
        cfg.command_min_episodes_per_level
        if stage == 2
        else cfg.randomization_min_episodes_per_level
    )
    _finish_episodes(env, minimum, reached=ratio + 0.01)
    assert env.difficulty_level == 2
    env.close()


def test_reversals_flip_direction_near_full_speed():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    env.set_difficulty_level(env.max_level)
    env.reset(seed=21)
    cfg = env.curriculum_config
    limit = env.command_limits[0]
    reversals = 0
    for _ in range(4_000):
        env._command[:] = (0.25, 0.5)
        env._episode_reversal = False
        env._sample_command(allow_reversal=True)
        if env._episode_reversal:
            reversals += 1
            linear, angular = env.command
            assert cfg.reversal_min_fraction * limit <= -linear <= limit and angular == 0.0
    assert np.isclose(reversals / 4_000, cfg.reversal_probability, atol=0.02)
    for _ in range(200):  # never at reset, and never in stage 1 (no commands)
        env._sample_command()
        assert not env._episode_reversal
    env.set_curriculum_stage(1)
    for _ in range(200):
        env._sample_command(allow_reversal=True)
        assert not env._episode_reversal and np.all(env.command == 0.0)
    env.close()


def test_levels_require_surviving_reversals():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    cfg = env.curriculum_config
    # 95% survive overall, but the only fall is one of three reversal episodes per window.
    for episode in range(cfg.command_min_episodes_per_level):
        phase = episode % cfg.evaluation_window
        _finish_episodes(env, 1, reversal=phase < 3, survive=phase != 0)
    assert env.difficulty_level == 1
    _finish_episodes(env, cfg.evaluation_window, reversal=True)  # all reversals survived
    assert env.difficulty_level == 2
    env.close()


def test_speed_tracking_skips_acceleration_and_small_commands():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    env.set_difficulty_level(env.max_level)
    settle_steps = int(np.ceil(env.curriculum_config.speed_ratio_settle_seconds / env.dt))

    def track(command, steps, forward_velocity=0.2):
        env._command[:] = command
        env._command_age = 0.0
        env._episode_tracking[:] = 0.0
        for _ in range(steps):
            env._track_command(forward_velocity)
        return env._episode_tracking.copy()

    assert np.all(track((0.3, 0.0), settle_steps) == 0.0)  # still accelerating
    reached, commanded, _, _ = track((0.3, 0.0), settle_steps + 10)
    assert commanded > 0 and np.isclose(reached / commanded, 0.2 / 0.3)
    reached, commanded, _, _ = track((-0.3, 0.0), settle_steps + 10)
    assert np.isclose(reached / commanded, -0.2 / 0.3)  # wrong direction counts against
    assert np.all(track((0.08, 0.0), settle_steps + 10) == 0.0)  # under 1/3 of 0.3 m/s
    _, _, still_speed, still_steps = track((0.0, 1.0), settle_steps + 10)
    assert still_steps > 0 and np.isclose(still_speed / still_steps, 0.2)
    env.close()


def test_actor_is_blind_to_privileged_state():
    env = SelfBalancingRobotEnv()
    policy = AsymmetricActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        actor_observation_size=env.actor_observation_size,
    )
    actor_layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    critic_layers = [m for m in policy.mlp_extractor.value_net if isinstance(m, torch.nn.Linear)]
    assert [layer.in_features for layer in actor_layers] == [env.actor_observation_size, 128, 64]
    assert [layer.out_features for layer in actor_layers] == [128, 64, 32]
    assert [layer.in_features for layer in critic_layers] == [PRIVILEGED_SIZE, 128]
    assert policy.action_net.in_features == 32
    assert policy.value_net.in_features == 64

    observation, _ = env.reset(seed=9)
    changed = observation.copy()
    changed[env.actor_observation_size :] += 10.0
    batch = torch.as_tensor(np.stack([observation, changed]), dtype=torch.float32)
    with torch.no_grad():
        actions = policy.get_distribution(batch).distribution.mean
        values = policy.predict_values(batch)
    torch.testing.assert_close(actions[0], actions[1])
    assert not torch.allclose(values[0], values[1])
    env.close()


def test_held_command_is_not_resampled():
    env = SelfBalancingRobotEnv(curriculum_stage=2)
    env.hold_command(0.2, -0.5)
    env.reset(seed=10)
    for _ in range(int(6.0 / env.dt)):
        _, _, terminated, _, info = env.step(np.zeros(2))
        if terminated:
            env.reset()
    np.testing.assert_allclose(env.command, [0.2, -0.5])
    assert info["v_cmd"] == 0.2
    env.close()


def test_physics_overrides_pin_friction_com_and_slope():
    env = SelfBalancingRobotEnv(curriculum_stage=3)
    env.reset(seed=12)
    env.set_physics(friction=0.5, com_offset=[0.0, 0.004, -0.002])
    # Friction and CoM apply immediately, without a reset.
    assert np.all(env.model.geom_friction[:, 0] == 0.5)
    np.testing.assert_allclose(
        env.model.body_ipos[env._base_body_id] - env._nominal_com, [0.0, 0.004, -0.002]
    )
    env.set_physics(slope=0.3)
    for seed in range(3):  # overrides survive resets and beat stage-3 randomization
        env.reset(seed=seed)
        assert env.physics["slope"] == 0.3
        assert env.physics["friction"] == 0.5
    env.clear_physics()
    env.reset(seed=13)
    assert env.physics["friction"] != 0.5
    env.close()


def test_robot_spawns_near_vertical_with_wheels_on_steep_slope():
    env = SelfBalancingRobotEnv()
    env.set_physics(slope=0.5)
    for seed in range(10):
        env.reset(seed=seed)
        # The collision/render ground itself is tilted, not just the bookkeeping.
        ground_normal = env.data.geom_xmat[env._ground_geom_id].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(ground_normal, env._ground_normal, atol=1e-9)
        assert np.isclose(np.arccos(ground_normal[2]), 0.5)
        _, pitch = env._tilt()
        assert abs(pitch) <= env.environment_config.initial_pitch_radians + 1e-6
        for name in ("left_wheel_collision", "right_wheel_collision"):
            distance = mujoco.mj_geomDistance(
                env.model, env.data, env._ground_geom_id, env.model.geom(name).id, 1.0, None
            )
            assert abs(distance) < 1e-3
    env.close()


def test_pitch_rate_is_penalized_instead_of_pitch_angle():
    env = SelfBalancingRobotEnv()
    env.reset(seed=14)
    _set_pitch(env, 0.2)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["reward_pitch_rate"] == -info["pitch_rate"] ** 2
    assert "reward_pitch" not in info
    env.close()


def test_tracking_reward_still_rewards_progress_far_from_the_command():
    from self_balancing_robot.envs.balance_env import _tracking_kernel

    cfg = RewardConfig()
    reward = [
        _tracking_kernel(
            error,
            cfg.linear_velocity_variance,
            cfg.linear_velocity_wide_variance,
            cfg.linear_velocity_wide_fraction,
        )
        for error in (0.35, 0.2, 0.1, 0.0)
    ]
    assert reward == sorted(reward)
    assert reward[1] - reward[0] > 0.05  # going from 0.05 to 0.2 m/s on a 0.4 command pays
    assert reward[-1] == 1.0


def test_stage_three_levels_ramp_slope_and_push_together():
    env = SelfBalancingRobotEnv(curriculum_stage=3)
    cfg, randomization = env.curriculum_config, env.randomization_config
    assert env.difficulty_level == 1
    for seed in range(5):
        env.reset(seed=seed)
        assert abs(env.physics["slope"]) <= env.slope_limit + 1e-12
        assert np.all(np.abs(env._com_offset) <= env.com_offset_limit + 1e-12)
    levels = [env.difficulty_level]
    # A full window of successes is not enough: each level needs a minimum of episodes.
    _finish_episodes(env, cfg.evaluation_window)
    assert env.difficulty_level == 1
    for _ in range(cfg.randomization_levels + 2):
        _finish_episodes(env, cfg.randomization_min_episodes_per_level)
        levels.append(env.difficulty_level)
    assert levels == sorted(levels) and levels[-1] == cfg.randomization_levels
    assert env.curriculum_stage == 3
    assert np.isclose(env.slope_limit, randomization.slope_radians)
    assert np.isclose(env.push_limit, randomization.impulse_force_n)
    assert np.isclose(env.com_offset_limit, randomization.center_of_mass_offset_m)
    env.set_curriculum_stage(3)  # e.g. evaluation: restarts at level 1
    assert env.difficulty_level == 1
    env.close()


def test_level_sampling_favours_the_newest_band_but_keeps_easier_values():
    env = SelfBalancingRobotEnv(curriculum_stage=3)
    env.reset(seed=16)
    cfg = env.curriculum_config
    env.set_difficulty_level(cfg.randomization_levels)
    limit = env.slope_limit
    band_floor = limit - limit / cfg.randomization_levels
    samples = np.array([env._sample_level_magnitude(limit) for _ in range(20_000)])
    assert np.all((samples >= 0) & (samples <= limit))
    assert np.isclose(np.mean(samples == 0), cfg.level_zero_probability, atol=0.01)
    assert np.isclose(np.mean(samples >= band_floor), cfg.level_frontier_probability, atol=0.015)
    assert np.isclose(np.mean((samples > 0) & (samples < band_floor)), 0.5, atol=0.015)

    env.set_difficulty_level(1)  # nothing below the first band: it gets that share
    samples = np.array([env._sample_level_magnitude(env.slope_limit) for _ in range(5_000)])
    assert np.isclose(np.mean(samples == 0), cfg.level_zero_probability, atol=0.015)
    env.close()


def test_yaw_rate_reward_uses_the_narrow_kernel_only():
    env = SelfBalancingRobotEnv()
    env.hold_command(0.0, 1.0)
    env.reset(seed=17)
    _, _, _, _, info = env.step(np.zeros(2))
    error = info["yaw_rate"] - info["omega_cmd"]
    expected = np.exp(-(error**2) / env.reward_config.angular_velocity_variance)
    assert np.isclose(info["reward_angular_velocity"], expected)
    env.close()
