import numpy as np

from self_balancing_robot import SelfBalancingRobotEnv


def test_environment_reset_and_step():
    env = SelfBalancingRobotEnv()
    observation, info = env.reset(seed=123)
    assert observation.shape == (8,)
    assert env.observation_space.contains(observation)
    assert info == {}

    next_observation, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert next_observation.shape == (8,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "pitch" in info
    env.close()
