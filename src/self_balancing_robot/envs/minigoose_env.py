from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from self_balancing_robot.config.config import MiniGooseEnvConfig

XML_PATH = Path(__file__).parents[1] / "assets" / "env_v0.xml"

# timestep 0.005 s * 1 = 200 Hz control
FRAME_SKIP = 1
TIMESTEP = 0.005  # must match <option timestep> in env_v0.xml

# Sensors the real robot has, in observation order (imu_orientation is sim-only)
SENSOR_ACCEL = "imu_accel"
SENSOR_GYRO = "imu_gyro"
SENSOR_LEFT_WHEEL_VEL = "left_wheel_vel"
SENSOR_RIGHT_WHEEL_VEL = "right_wheel_vel"
    
class MiniGooseEnv(MujocoEnv, gym.utils.EzPickle):
    """
    Environment for the MiniGoose self-balancing robot.

    Observation (4): pitch, pitch rate, left wheel velocity, right wheel velocity. 
    Action (2): left/right wheel velocity command, normalized to [-1, 1].

    Parameters come from a MiniGooseConfig; single fields can be overridden by keyword,
    e.g. MiniGooseEnv(max_tilt=0.3) or gym.make("MiniGoose-Balance-v0", max_tilt=0.3).
    """

    metadata: ClassVar = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        config: MiniGooseEnvConfig | None = None,
        render_mode: str | None = None,
        **overrides,
    ):
        # save config
        gym.utils.EzPickle.__init__(self, config, render_mode, **overrides)
        self.cfg = replace(config or MiniGooseEnvConfig(), **overrides)

        # initialize mujoco environment
        MujocoEnv.__init__(
            self,
            model_path=str(XML_PATH),
            frame_skip=FRAME_SKIP,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            # body 1 is base_link (body 0 is the world)
            default_camera_config={
                "trackbodyid": 1,
                "distance": 0.8,
                "elevation": -20.0,
                "azimuth": 45.0,
            },
            render_mode=render_mode,
        )
        # set action space to normalized [-1, 1] for each wheel
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._base_id = self.model.body("base_link").id

        # save sensors
        self.imu_accel = self.data.sensor(SENSOR_ACCEL) 
        self.imu_gyro = self.data.sensor(SENSOR_GYRO)
        self.left_wheel_vel = self.data.sensor(SENSOR_LEFT_WHEEL_VEL)
        self.right_wheel_vel = self.data.sensor(SENSOR_RIGHT_WHEEL_VEL)

        # initialize internal variables
        self._tilt = 0
        self._prev_action = np.zeros(2)
        self._init_xy = np.zeros(2)
        self._steps = 0

    def _get_obs(self) -> np.ndarray:
        # read data from sensors
        accel_x, accel_y, accel_z = self.imu_accel.data
        roll_rate, pitch_rate, yaw_rate = self.imu_gyro.data
        left_wheel_vel = self.left_wheel_vel.data[0]
        right_wheel_vel = self.right_wheel_vel.data[0]
        # TODO: add sensor noise
        
        # calculate tilt
        # TODO: check if the sign and axis is correct (rn we calculate roll as the tilt)
        tilt_from_accel = np.arctan2(accel_y, accel_z) # assumes that the accelerometer only measures gravity
        self._tilt = self.cfg.alpha * (self._tilt + roll_rate * self.model.opt.timestep) + (1 - self.cfg.alpha) * tilt_from_accel

        obs = np.array([self._tilt, roll_rate, left_wheel_vel, right_wheel_vel])
        return obs # tilt, tilt rate, left wheel velocity, right wheel velocity

    def _gt_tilt(self) -> float:
        """Ground-truth tilt about the body x axis (the wheel axle)."""
        R = self.data.xmat[self._base_id].reshape(3, 3)
        return float(np.arctan2(R[2, 1], R[2, 2]))

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel + self.np_random.uniform(-0.01, 0.01, size=self.model.nv) # start with small random velocity
        # The axle passes through the base origin, so tilting about it keeps the wheels on the ground
        tilt = self.np_random.uniform(-self.cfg.init_pitch_range, self.cfg.init_tilt_range)
        mujoco.mju_axisAngle2Quat(qpos[3:7], np.array([1.0, 0.0, 0.0]), tilt)
        self.set_state(qpos, qvel)

        self._tilt = tilt
        self._init_xy = qpos[:2].copy()
        self._prev_action = np.zeros(2)
        self._steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # add noise to actions
        noisy_action = action + self.np_random.normal(0.0, self.cfg.action_noise_std, size=2)
        self.do_simulation(noisy_action, self.frame_skip)
        self._steps += 1

        observation = self._get_obs()

        # calculate GT values for rewards
        R = self.data.xmat[self._base_id].reshape(3, 3)
        # Euler angles from rotation matrix (roll, pitch, yaw)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        # angle rates
        roll_rate = float(self.data.qvel[3]) 
        pitch_rate = float(self.data.qvel[4]) 
        yaw_rate = float(self.data.qvel[5]) 
        # speed and velocity
        velocity = float(np.dot(R, self.data.qvel[:3]))
        distance = float(np.linalg.norm(self.data.qpos[:2] - self._init_xy))

        tilt = roll
        tilt_rate = roll_rate
        
        reward_terms = {
            "alive": self.cfg.alive_bonus,
            "tilt": -self.cfg.w_tilt * tilt**2,
            "tilt_rate": -self.cfg.w_tilt_rate * tilt_rate**2,
            "velocity": -self.cfg.w_velocity * float(np.square(velocity).sum()),
            "distance": -self.cfg.w_distance * distance**2,
            "action": -self.cfg.w_action * float(np.square(action).sum()),
            "action_rate": -self.cfg.w_action_rate
            * float(np.square(action - self._prev_action).sum()),
        }
        reward = float(sum(reward_terms.values()))
        self._prev_action = noisy_action

        terminated = abs(tilt) > self.cfg.max_tilt
        truncated = self._steps >= self.cfg.max_episode_steps

        if self.render_mode == "human":
            self.render()
        info = {
            "tilt": tilt,
            "distance": distance,
            **{f"reward_{name}": value for name, value in reward_terms.items()},
        }
        return observation, reward, bool(terminated), bool(truncated), info
