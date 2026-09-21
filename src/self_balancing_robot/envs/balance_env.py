from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


class SelfBalancingRobotEnv(MujocoEnv, gym.utils.EzPickle):
    """Planar two-wheel robot that learns to keep its chassis upright."""

    metadata: ClassVar = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):
        gym.utils.EzPickle.__init__(self, render_mode)
        xml_path = Path(__file__).parents[1] / "assets" / "self_balancing_robot.xml"
        MujocoEnv.__init__(
            self,
            model_path=str(xml_path),
            frame_skip=5,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            default_camera_config={"distance": 2.5, "elevation": -15, "azimuth": 90},
            render_mode=render_mode,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._upright_limit = 0.6
        self._max_position = 3.0

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos[:4], self.data.qvel[:4])).astype(np.float64)

    def reset_model(self) -> np.ndarray:
        self.set_state(
            self.init_qpos + self.np_random.uniform(-0.02, 0.02, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(-0.02, 0.02, size=self.model.nv),
        )
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        position, pitch = observation[:2]
        velocity, pitch_rate = observation[4:6]
        terminated = bool(abs(pitch) > self._upright_limit or abs(position) > self._max_position)
        reward = (
            1.0
            - 2.0 * pitch**2
            - 0.05 * pitch_rate**2
            - 0.02 * velocity**2
            - 0.01 * float(np.square(action).sum())
        )
        return observation, float(reward), terminated, False, {
            "x_position": float(position),
            "pitch": float(pitch),
        }

    def viewer_setup(self) -> None:
        self.viewer.cam.distance = 2.5
