import numpy as np

from self_balancing_robot import SelfBalancingRobotEnv
from self_balancing_robot.config import EnvironmentConfig
from self_balancing_robot.envs.balance_env import RewardConfig
from self_balancing_robot.training_callback import BalanceErrorCallback


def test_environment_reset_and_step():
    env = SelfBalancingRobotEnv()
    observation, info = env.reset(seed=123)
    assert observation.shape == (12,)
    assert env.observation_space.contains(observation)
    assert info == {}
    left_wheel_geom_id = env.model.geom("left_wheel_geom").id
    wheel_bottom = (
        env.data.geom_xpos[left_wheel_geom_id, 2] - env.model.geom_size[left_wheel_geom_id, 0]
    )
    assert abs(wheel_bottom) < 1e-3

    next_observation, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert next_observation.shape == (12,)
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
    assert info["terminated_reason"] == "pitch_failure"
    env.close()


def test_render_fps_uses_the_simulation_timestep_and_frame_skip():
    env = SelfBalancingRobotEnv(environment_config=EnvironmentConfig(frame_skip=2))
    assert env.frame_skip == 2
    assert env.metadata["render_fps"] == int(np.round(1.0 / env.dt))
    assert "render_fps" not in SelfBalancingRobotEnv.metadata
    env.close()


def test_training_start_replaces_old_plot_with_current_run(tmp_path):
    plot_path = tmp_path / "training_errors.png"
    plot_path.write_bytes(b"stale plot")
    callback = BalanceErrorCallback(plot_path=plot_path)
    callback.train_steps.append(100)
    callback.train_errors.append(0.5)
    callback.test_steps.append(100)
    callback.test_errors.append(0.4)

    callback._on_training_start()
    callback.eval_env.close()

    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not callback.train_steps
    assert not callback.test_steps


def test_reward_components_follow_configured_weights():
    env = SelfBalancingRobotEnv(
        reward_config=RewardConfig(
            alive_weight=2.0,
            pitch_weight=0.0,
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
    env.data.qpos[1] = env._upright_limit + 0.1
    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert not truncated
    assert reward == 0.0
    assert info["terminated_reason"] == "pitch_failure"
    env.close()


def test_reset_with_same_seed_reproduces_initial_observation():
    env = SelfBalancingRobotEnv()
    first, _ = env.reset(seed=4)
    second, _ = env.reset(seed=4)
    np.testing.assert_allclose(first, second)
    env.close()
