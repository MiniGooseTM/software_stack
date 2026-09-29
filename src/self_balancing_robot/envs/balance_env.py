from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from self_balancing_robot.config import (
    DEFAULT_CURRICULUM_CONFIG,
    DEFAULT_ENVIRONMENT_CONFIG,
    DEFAULT_RANDOMIZATION_CONFIG,
    DEFAULT_REWARD_CONFIG,
    CurriculumConfig,
    CurriculumStage,
    EnvironmentConfig,
    RandomizationConfig,
    RewardConfig,
)


class SelfBalancingRobotEnv(MujocoEnv, gym.utils.EzPickle):
    """Planar two-wheel robot with command tracking and curriculum learning."""

    metadata: ClassVar = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
        randomization_config: RandomizationConfig | None = None,
        environment_config: EnvironmentConfig | None = None,
        max_episode_steps: int | None = None,
    ):
        gym.utils.EzPickle.__init__(
            self,
            render_mode,
            reward_config,
            curriculum_config,
            randomization_config,
            environment_config,
            max_episode_steps,
        )
        self.environment_config = environment_config or DEFAULT_ENVIRONMENT_CONFIG
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else self.environment_config.max_episode_steps
        )
        xml_path = Path(__file__).parents[1] / "assets" / "self_balancing_robot.xml"
        MujocoEnv.__init__(
            self,
            model_path=str(xml_path),
            frame_skip=self.environment_config.frame_skip,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            default_camera_config={
                "distance": self.environment_config.camera_distance_m,
                "elevation": self.environment_config.camera_elevation_degrees,
                "azimuth": self.environment_config.camera_azimuth_degrees,
            },
            render_mode=render_mode,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.curriculum_config = curriculum_config or DEFAULT_CURRICULUM_CONFIG
        self.randomization_config = randomization_config or DEFAULT_RANDOMIZATION_CONFIG
        self._upright_limit = self.environment_config.upright_limit_radians
        self._max_position = self.environment_config.max_position_m
        self._stage = CurriculumStage.STABILIZATION
        self._episode_step = 0
        self._command_time_remaining = 0.0
        self._next_impulse_time = np.inf
        self._episode_linear_error = 0.0
        self._episode_steps_for_metrics = 0
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._action_queue: deque[np.ndarray] = deque()
        self._latency_steps = 0
        self._stage_results: deque[bool] = deque(maxlen=self.curriculum_config.evaluation_window)
        self._stage2_errors: deque[float] = deque(maxlen=self.curriculum_config.evaluation_window)
        self._command = np.zeros(2, dtype=np.float64)

        self._chassis_body_id = self.model.body("chassis").id
        self._ground_geom_id = self.model.geom("ground").id
        self._chassis_geom_id = self.model.geom("chassis_geom").id
        self._nominal_body_pos = self.model.body_pos[self._chassis_body_id].copy()
        self._nominal_mass = self.model.body_mass[self._chassis_body_id]
        self._nominal_friction = self.model.geom_friction.copy()
        self._nominal_ground_quat = self.model.geom_quat[self._ground_geom_id].copy()
        self._wheel_radius = self.model.geom_size[self.model.geom("left_wheel_geom").id, 0]
        left_wheel_id = self.model.body("left_wheel").id
        right_wheel_id = self.model.body("right_wheel").id
        self._track_width = abs(
            self.model.body_pos[left_wheel_id, 1] - self.model.body_pos[right_wheel_id, 1]
        )

    def _initialize_simulation(self):
        model, data = super()._initialize_simulation()
        self.metadata = {
            **type(self).metadata,
            "render_fps": int(
                np.round(1.0 / (model.opt.timestep * self.environment_config.frame_skip))
            ),
        }
        return model, data

    @property
    def curriculum_stage(self) -> int:
        return self._stage

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def _get_true_kinematics(self) -> tuple[np.ndarray, float, float, float, float]:
        observation = np.concatenate((self.data.qpos[:4], self.data.qvel[:4]))
        position, pitch = observation[:2]
        linear_velocity, _pitch_rate = observation[4:6]
        wheel_left_velocity, wheel_right_velocity = observation[6:8]
        angular_velocity = (
            self._wheel_radius * (wheel_right_velocity - wheel_left_velocity) / self._track_width
        )
        return observation, float(position), float(pitch), float(linear_velocity), float(
            angular_velocity
        )

    def _get_obs(self, noisy: bool = True) -> np.ndarray:
        observation, _, _, _, _ = self._get_true_kinematics()
        result = np.concatenate((observation, self._command, self._previous_action)).astype(
            np.float64
        )
        if (
            noisy
            and self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION
            and self.randomization_config.observation_noise > 0
        ):
            result[:8] += self.np_random.normal(
                0.0, self.randomization_config.observation_noise, size=8
            )
        return result

    def _sample_command(self) -> None:
        if self._stage == CurriculumStage.STABILIZATION:
            self._command[:] = 0.0
        elif self._stage == CurriculumStage.MICRO_MOBILITY:
            linear_limit, angular_limit = self.curriculum_config.stage2_command_limits
            self._command[:] = self.np_random.uniform(
                [-linear_limit, -angular_limit], [linear_limit, angular_limit], size=2
            )
        else:
            linear_limit, angular_limit = self.curriculum_config.stage3_command_limits
            self._command[:] = self.np_random.uniform(
                [-linear_limit, -angular_limit], [linear_limit, angular_limit], size=2
            )
        self._command_time_remaining = self.np_random.uniform(
            *self.curriculum_config.command_resample_seconds
        )

    def _sample_randomization(self) -> None:
        self.model.body_mass[self._chassis_body_id] = self._nominal_mass
        self.model.body_pos[self._chassis_body_id] = self._nominal_body_pos
        self.model.geom_friction[:] = self._nominal_friction
        self.model.geom_quat[self._ground_geom_id] = self._nominal_ground_quat
        if self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            return
        cfg = self.randomization_config
        self.model.body_mass[self._chassis_body_id] *= self.np_random.uniform(
            1.0 - cfg.mass_fraction, 1.0 + cfg.mass_fraction
        )
        self.model.body_pos[self._chassis_body_id, :2] += self.np_random.uniform(
            -cfg.center_of_mass_offset_m, cfg.center_of_mass_offset_m, size=2
        )
        friction_scale = self.np_random.uniform(
            1.0 - cfg.friction_fraction, 1.0 + cfg.friction_fraction
        )
        self.model.geom_friction[self._ground_geom_id] = (
            self._nominal_friction[self._ground_geom_id] * friction_scale
        )
        slope = self.np_random.uniform(-cfg.slope_radians, cfg.slope_radians)
        self.model.geom_quat[self._ground_geom_id] = np.array(
            [np.cos(slope / 2), 0.0, np.sin(slope / 2), 0.0]
        )

    def reset_model(self) -> np.ndarray:
        self._sample_randomization()
        self._sample_command()
        self._episode_step = 0
        self._previous_action[:] = 0.0
        self._action_queue.clear()
        self._latency_steps = (
            int(
                self.np_random.integers(
                    0, self.randomization_config.max_latency_steps + 1
                )
            )
            if self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION
            else 0
        )
        self._episode_linear_error = 0.0
        self._episode_steps_for_metrics = 0
        self._next_impulse_time = self.np_random.uniform(
            *self.randomization_config.impulse_interval_seconds
        )
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                -self.environment_config.initial_state_noise,
                self.environment_config.initial_state_noise,
                size=self.model.nq,
            ),
            self.init_qvel
            + self.np_random.uniform(
                -self.environment_config.initial_state_noise,
                self.environment_config.initial_state_noise,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def _base_collision(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if self._ground_geom_id in geom_pair and self._chassis_geom_id in geom_pair:
                return True
        return False

    def _apply_disturbance_if_due(self) -> None:
        if (
            self._stage < CurriculumStage.DOMAIN_RANDOMIZATION
            or self._episode_step * self.dt < self._next_impulse_time
        ):
            return
        force = self.np_random.uniform(
            -self.randomization_config.impulse_force,
            self.randomization_config.impulse_force,
            size=2,
        )
        self.data.xfrc_applied[self._chassis_body_id, :2] = force
        self._next_impulse_time += self.np_random.uniform(
            *self.randomization_config.impulse_interval_seconds
        )

    def _advance_curriculum(self, episode_success: bool) -> None:
        if self._stage == CurriculumStage.STABILIZATION:
            self._stage_results.append(episode_success)
            if (
                len(self._stage_results) == self._stage_results.maxlen
                and np.mean(self._stage_results) >= self.curriculum_config.stage1_timeout_success_rate
            ):
                self._stage = CurriculumStage.MICRO_MOBILITY
        elif self._stage == CurriculumStage.MICRO_MOBILITY:
            self._stage2_errors.append(
                self._episode_linear_error / max(self._episode_steps_for_metrics, 1)
            )
            if (
                len(self._stage2_errors) == self._stage2_errors.maxlen
                and np.mean(self._stage2_errors)
                < self.curriculum_config.stage2_linear_error_threshold
            ):
                self._stage = CurriculumStage.COMMAND_TRACKING

    def step(self, action: np.ndarray):
        action = np.asarray(np.clip(action, self.action_space.low, self.action_space.high))
        self._action_queue.append(action.astype(np.float32))
        if not self._action_queue:
            applied_action = action
        elif self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION:
            while len(self._action_queue) <= self._latency_steps:
                self._action_queue.appendleft(np.zeros(2, dtype=np.float32))
            applied_action = self._action_queue.popleft()
        else:
            applied_action = self._action_queue.popleft()
        self._apply_disturbance_if_due()
        self.do_simulation(applied_action, self.frame_skip)
        self.data.xfrc_applied[self._chassis_body_id] = 0.0

        _, position, pitch, linear_velocity, angular_velocity = self._get_true_kinematics()
        self._episode_step += 1
        self._command_time_remaining -= self.dt
        if self._command_time_remaining <= 0:
            self._sample_command()
        linear_error = abs(linear_velocity - self._command[0])
        self._episode_linear_error += linear_error
        self._episode_steps_for_metrics += 1

        cfg = self.reward_config
        reward_terms = {
            "alive": 1.0,
            "pitch": np.exp(-(pitch**2) / cfg.pitch_variance),
            "linear_velocity": np.exp(
                -((linear_velocity - self._command[0]) ** 2) / cfg.linear_velocity_variance
            ),
            "angular_velocity": np.exp(
                -((angular_velocity - self._command[1]) ** 2) / cfg.angular_velocity_variance
            ),
            "action": -float(np.square(applied_action).sum()),
            "smoothness": -float(np.square(applied_action - self._previous_action).sum()),
        }
        self._previous_action = applied_action.copy()
        base_collision = self._base_collision()
        terminated = bool(
            abs(pitch) > self._upright_limit
            or abs(position) > self._max_position
            or base_collision
        )
        truncated = self._episode_step >= self.max_episode_steps
        if terminated or truncated:
            self._advance_curriculum(not terminated and truncated)
        reward = (
            cfg.alive_weight * reward_terms["alive"]
            + cfg.pitch_weight * reward_terms["pitch"]
            + cfg.linear_velocity_weight * reward_terms["linear_velocity"]
            + cfg.angular_velocity_weight * reward_terms["angular_velocity"]
            + cfg.action_weight * reward_terms["action"]
            + cfg.smoothness_weight * reward_terms["smoothness"]
        )
        if terminated:
            reward = 0.0
        info = {
            "x_position": position,
            "pitch": pitch,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "v_cmd": float(self._command[0]),
            "omega_cmd": float(self._command[1]),
            "curriculum_stage": self._stage,
            "episode_step": self._episode_step,
            "terminated_reason": (
                "pitch_failure"
                if abs(pitch) > self._upright_limit
                else "base_collision"
                if base_collision
                else "position_limit"
                if abs(position) > self._max_position
                else "timeout"
                if truncated
                else None
            ),
        }
        info.update({f"reward_{name}": float(value) for name, value in reward_terms.items()})
        return self._get_obs(), float(reward), terminated, truncated, info

    def viewer_setup(self) -> None:
        self.viewer.cam.distance = self.environment_config.camera_distance_m
