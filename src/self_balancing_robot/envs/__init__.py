"""MuJoCo environments for the self-balancing robot."""

import gymnasium as gym

from self_balancing_robot.envs.minigoose_env import MiniGooseEnv
from self_balancing_robot.config.config import MiniGooseEnvConfig

gym.register(
    id="MiniGoose-Balance-v0",
    entry_point="self_balancing_robot.envs.minigoose_env:MiniGooseEnv",
)

__all__ = ["MiniGooseEnvConfig", "MiniGooseEnv"]
