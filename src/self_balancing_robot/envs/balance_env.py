from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import mujoco
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

# One proprioceptive frame O_t, as the ESP32 builds it. The actor sees the last
# `history_length` frames, newest first. Commands are divided by the full command range
# (v / 0.3 m/s, omega / 1.5 rad/s) and wheel velocities by the motor no-load speed;
# everything else is in SI units (rad, rad/s). With raw commands (|v| <= 0.1 at the first
# ramp level) the policy ignored them and stood still; scaled, it learned to follow them.
ACTOR_FRAME_FIELDS = (
    "v_cmd",
    "omega_cmd",
    "roll",
    "pitch",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "left_wheel_velocity",
    "right_wheel_velocity",
    "previous_action_left",
    "previous_action_right",
)
ACTOR_FRAME_SIZE = len(ACTOR_FRAME_FIELDS)

# Privileged state S_t, seen only by the critic: the noise-free, latency-free frame
# plus physical truths that the actor has to infer from its history.
PRIVILEGED_FIELDS = (
    *(f"true_{name}" for name in ACTOR_FRAME_FIELDS),
    "forward_velocity",
    "lateral_velocity",
    "com_offset_x",
    "com_offset_y",
    "com_offset_z",
    "mass_deviation",
    "friction_deviation",
    "push_force_forward",
    "push_force_lateral",
    "motor_latency",
    "imu_latency",
    "encoder_latency",
    "slope_forward",
    "slope_lateral",
)
PRIVILEGED_SIZE = len(PRIVILEGED_FIELDS)


def _normalized(value: float | np.ndarray, limit: float) -> float | np.ndarray:
    return value / limit if limit > 0 else value * 0.0


def _tracking_kernel(
    error: float, narrow_variance: float, wide_variance: float, wide_fraction: float
) -> float:
    """Blend of a narrow and a wide Gaussian kernel, bounded in [0, 1]."""
    return float(
        (1.0 - wide_fraction) * np.exp(-(error**2) / narrow_variance)
        + wide_fraction * np.exp(-(error**2) / wide_variance)
    )


class SelfBalancingRobotEnv(MujocoEnv, gym.utils.EzPickle):
    """3D two-wheel robot for asymmetric actor-critic PPO with curriculum learning.

    The observation is ``[actor_history, privileged_state]``. The first
    ``actor_observation_size`` entries are what the deployed policy receives;
    the rest is only used by the critic during training.
    """

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
        curriculum_stage: int = CurriculumStage.STABILIZATION,
    ):
        gym.utils.EzPickle.__init__(
            self,
            render_mode,
            reward_config,
            curriculum_config,
            randomization_config,
            environment_config,
            max_episode_steps,
            curriculum_stage,
        )
        self.environment_config = environment_config or DEFAULT_ENVIRONMENT_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.curriculum_config = curriculum_config or DEFAULT_CURRICULUM_CONFIG
        self.randomization_config = randomization_config or DEFAULT_RANDOMIZATION_CONFIG
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else self.environment_config.max_episode_steps
        )
        self.history_length = self.environment_config.history_length
        self.actor_observation_size = ACTOR_FRAME_SIZE * self.history_length
        self.privileged_observation_size = PRIVILEGED_SIZE
        self._camera_config = {
            "type": mujoco.mjtCamera.mjCAMERA_TRACKING,
            "distance": self.environment_config.camera_distance_m,
            "elevation": self.environment_config.camera_elevation_degrees,
            "azimuth": self.environment_config.camera_azimuth_degrees,
        }
        xml_path = Path(__file__).parents[1] / "assets" / self.environment_config.model_file
        MujocoEnv.__init__(
            self,
            model_path=str(xml_path),
            frame_skip=self.environment_config.frame_skip,
            observation_space=spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.actor_observation_size + self.privileged_observation_size,),
                dtype=np.float64,
            ),
            default_camera_config=self._camera_config,
            render_mode=render_mode,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._base_body_id = self.model.body("base_link").id
        self._camera_config["trackbodyid"] = self._base_body_id
        self._ground_geom_id = self.model.geom("ground").id
        self._base_geom_id = self.model.geom("base_link_collision").id
        base_joint = self.model.joint("base_link_joint")
        self._base_qpos = int(base_joint.qposadr[0])
        self._base_qvel = int(base_joint.dofadr[0])
        wheel_joints = [self.model.joint(name) for name in ("left_wheel_joint", "right_wheel_joint")]
        self._wheel_qvel = np.array([int(joint.dofadr[0]) for joint in wheel_joints])
        # Normalized action a in [-1, 1] is the PWM duty; the velocity actuator turns
        # duty * no-load speed into DC-motor torque.
        self._max_wheel_speed = float(self.model.actuator_ctrlrange[:, 1].min())
        self._wheel_radius = float(self.model.geom("left_wheel_collision").size[0])
        mujoco.mj_forward(self.model, self.data)
        self._half_track = 0.5 * float(
            np.linalg.norm(
                self.data.geom("left_wheel_collision").xpos
                - self.data.geom("right_wheel_collision").xpos
            )
        )

        self._nominal_mass = float(self.model.body_mass[self._base_body_id])
        self._nominal_inertia = self.model.body_inertia[self._base_body_id].copy()
        self._nominal_com = self.model.body_ipos[self._base_body_id].copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self._terrain_mocap_id = int(self.model.body("terrain").mocapid[0])
        self._nominal_ground_friction = float(self._nominal_friction[self._ground_geom_id, 0])
        self._max_com_offset = self.randomization_config.center_of_mass_offset_m

        self._stage = CurriculumStage(curriculum_stage)
        self._difficulty_level = 1
        self._stage_results: deque[bool] = deque(maxlen=self.curriculum_config.evaluation_window)
        self._command = np.zeros(2, dtype=np.float64)
        self._command_time_remaining = 0.0
        self._command_held = False
        self._command_age = 0.0
        self._physics_overrides: dict[str, float | np.ndarray] = {}
        self._episode_step = 0
        # Per-episode sums for the level-up checks: (speed reached along the command,
        # commanded speed, |v| on zero speed commands, steps on zero speed commands).
        self._episode_tracking = np.zeros(4)
        self._stage_tracking: deque[np.ndarray] = deque(
            maxlen=self.curriculum_config.evaluation_window
        )
        # Whether this episode had a reversal command, and (had one, survived) per episode.
        self._episode_reversal = False
        self._stage_reversals: deque[tuple[bool, bool]] = deque(
            maxlen=self.curriculum_config.evaluation_window
        )
        self._level_episodes = 0
        self._previous_action = np.zeros(2, dtype=np.float64)
        self._action_queue: deque[np.ndarray] = deque()
        self._sensor_buffer: deque[np.ndarray] = deque()
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._motor_latency = 0
        self._imu_latency = 0
        self._encoder_latency = 0
        self._mass_scale = 1.0
        self._friction_scale = 1.0
        self._com_offset = np.zeros(3)
        self._ground_normal = np.array([0.0, 0.0, 1.0])
        self._slope = 0.0
        self._push_force = np.zeros(2)
        self._push_end_time = 0.0
        self._next_impulse_time = np.inf

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
        return int(self._stage)

    def set_curriculum_stage(self, stage: int) -> None:
        """Change the stage (starting at its level 1); applies at the next reset."""
        self._stage = CurriculumStage(stage)
        self.set_difficulty_level(1)

    @property
    def max_level(self) -> int:
        """Number of levels of the current stage (1 for stage 1)."""
        cfg = self.curriculum_config
        return {
            CurriculumStage.STABILIZATION: 1,
            CurriculumStage.COMMAND_RAMP: cfg.command_levels,
            CurriculumStage.DOMAIN_RANDOMIZATION: cfg.randomization_levels,
        }[self._stage]

    @property
    def difficulty_level(self) -> int:
        """Level within the current stage, 1 .. ``max_level``."""
        return self._difficulty_level

    @property
    def curriculum_progress(self) -> float:
        """Stage plus the fraction of its levels passed, e.g. 2.5 = halfway through stage 2."""
        return self._stage + (self._difficulty_level - 1) / self.max_level

    def set_difficulty_level(self, level: int) -> None:
        """Set the level of the current stage and restart the advancement statistics."""
        self._difficulty_level = int(np.clip(level, 1, self.max_level))
        self._level_episodes = 0
        self._stage_results.clear()
        self._stage_tracking.clear()
        self._stage_reversals.clear()

    @property
    def command_limits(self) -> tuple[float, float]:
        """Current maximum |v_cmd| (m/s) and |omega_cmd| (rad/s)."""
        linear, angular = self.curriculum_config.command_limits
        if self._stage == CurriculumStage.STABILIZATION:
            return 0.0, 0.0
        fraction = self._level_fraction() if self._stage == CurriculumStage.COMMAND_RAMP else 1.0
        return fraction * linear, fraction * angular

    @property
    def slope_limit(self) -> float:
        """Current maximum terrain slope (rad)."""
        return self._randomization_fraction() * self.randomization_config.slope_radians

    @property
    def push_limit(self) -> float:
        """Current maximum push force (N)."""
        return self._randomization_fraction() * self.randomization_config.impulse_force_n

    @property
    def com_offset_limit(self) -> float:
        """Current maximum CoM shift per axis (m)."""
        return self._randomization_fraction() * self.randomization_config.center_of_mass_offset_m

    def _level_fraction(self) -> float:
        return self._difficulty_level / self.max_level

    def _randomization_fraction(self) -> float:
        if self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            return 0.0
        return self._level_fraction()

    def _sample_level_magnitude(self, limit: float, level: int | None = None) -> float:
        """Magnitude in [0, limit] weighted towards the newest band of ``level`` (see config)."""
        cfg = self.curriculum_config
        level = self._difficulty_level if level is None else level
        band_floor = max(limit - limit / level, 0.0)
        draw = self.np_random.uniform()
        if draw < cfg.level_zero_probability:
            return 0.0
        # At level 1 nothing lies below the newest band, so that share goes to the band.
        if draw < cfg.level_zero_probability + cfg.level_frontier_probability or not band_floor:
            return float(self.np_random.uniform(band_floor, limit))
        return float(self.np_random.uniform(0.0, band_floor))

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def set_physics(
        self,
        slope: float | None = None,
        friction: float | None = None,
        com_offset: np.ndarray | None = None,
    ) -> None:
        """Pin physics values instead of randomizing them (e.g. from the teleop window).

        ``slope`` is the terrain incline in rad (rising along world +y), ``friction`` the
        floor friction coefficient and ``com_offset`` the CoM shift in m (body frame:
        x = axle, y = forward, z = up). Friction and CoM take effect immediately, the slope
        at the next reset, which places the robot on it. ``None`` keeps the current setting.
        """
        if slope is not None:
            self._physics_overrides["slope"] = float(slope)
        if friction is not None:
            self._physics_overrides["friction"] = float(friction)
            self._friction_scale = float(friction) / self._nominal_ground_friction
        if com_offset is not None:
            self._physics_overrides["com_offset"] = np.asarray(com_offset, dtype=np.float64)
            self._com_offset = self._physics_overrides["com_offset"].copy()
        self._apply_physics()

    def clear_physics(self) -> None:
        """Return slope, friction and CoM to the curriculum's randomization at the next reset."""
        self._physics_overrides.clear()

    @property
    def physics(self) -> dict[str, float | np.ndarray]:
        """Current slope (rad), floor friction coefficient and CoM offset (m)."""
        return {
            "slope": self._slope,
            "friction": self._friction_scale * self._nominal_ground_friction,
            "com_offset": self._com_offset.copy(),
        }

    def hold_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """Use a fixed command (e.g. from a joystick) instead of the curriculum's random ones."""
        self._command[:] = self.wheel_feasible_command(linear_velocity, angular_velocity)
        self._command_held = True

    def wheel_feasible_command(
        self, linear_velocity: float, angular_velocity: float
    ) -> tuple[float, float]:
        """Scale (v, omega) down together so the outer wheel stays within full speed.

        At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors
        give while balancing: every v4 fall on flat ground came from such commands. The
        limit is the full command speed (0.3 m/s). The ESP32 should limit joystick commands
        the same way.
        """
        full_speed = self.curriculum_config.command_limits[0]
        outer_wheel = abs(linear_velocity) + abs(angular_velocity) * self._half_track
        scale = min(1.0, full_speed / outer_wheel) if outer_wheel > 0 else 1.0
        return linear_velocity * scale, angular_velocity * scale

    # ------------------------------------------------------------------ state

    def _rotation(self) -> np.ndarray:
        return self.data.xmat[self._base_body_id].reshape(3, 3)

    def _heading_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit forward and lateral vectors of the robot projected on the world xy-plane."""
        forward = self._rotation()[:2, 1].copy()
        forward /= max(np.linalg.norm(forward), 1e-9)
        lateral = np.array([forward[1], -forward[0]])
        return forward, lateral

    def _tilt(self) -> tuple[float, float]:
        # Gravity direction in the body frame, as an accelerometer at rest would see it.
        # Body axes: x = wheel axle (pitch), y = forward (roll), z = up.
        up = self._rotation()[2]
        pitch = np.arctan2(up[1], up[2])
        roll = np.arctan2(-up[0], np.hypot(up[1], up[2]))
        return float(roll), float(pitch)

    def _imu_measurement(self) -> np.ndarray:
        roll, pitch = self._tilt()
        pitch_rate, roll_rate, yaw_rate = self.data.qvel[self._base_qvel + 3 : self._base_qvel + 6]
        return np.array([roll, pitch, roll_rate, pitch_rate, yaw_rate])

    def _encoder_measurement(self) -> np.ndarray:
        return self.data.qvel[self._wheel_qvel] / self._max_wheel_speed

    def _body_velocities(self) -> tuple[float, float, float]:
        """True forward speed, lateral speed and world yaw rate."""
        forward, lateral = self._heading_axes()
        linear = self.data.qvel[self._base_qvel : self._base_qvel + 2]
        angular_body = self.data.qvel[self._base_qvel + 3 : self._base_qvel + 6]
        yaw_rate = float((self._rotation() @ angular_body)[2])
        return float(linear @ forward), float(linear @ lateral), yaw_rate

    def _clean_frame(self) -> np.ndarray:
        return np.concatenate(
            (
                self._scaled_command(),
                self._imu_measurement(),
                self._encoder_measurement(),
                self._previous_action,
            )
        )

    def _actor_frame(self) -> np.ndarray:
        """Delayed, noisy measurements as the onboard sensors would report them."""
        self._sensor_buffer.appendleft(
            np.concatenate((self._imu_measurement(), self._encoder_measurement()))
        )
        while len(self._sensor_buffer) > self._max_sensor_latency() + 1:
            self._sensor_buffer.pop()
        oldest = len(self._sensor_buffer) - 1
        imu = self._sensor_buffer[min(self._imu_latency, oldest)][:5].copy()
        encoders = self._sensor_buffer[min(self._encoder_latency, oldest)][5:].copy()
        if self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION:
            cfg = self.randomization_config
            imu[:2] += self.np_random.normal(0.0, cfg.imu_angle_noise_rad, size=2)
            imu[2:] += self.np_random.normal(0.0, cfg.imu_rate_noise_rad_s, size=3)
            encoders += self.np_random.normal(
                0.0, cfg.encoder_velocity_noise_rad_s / self._max_wheel_speed, size=2
            )
        return np.concatenate((self._scaled_command(), imu, encoders, self._previous_action))

    def _scaled_command(self) -> np.ndarray:
        return self._command / np.asarray(self.curriculum_config.command_limits)

    def _max_sensor_latency(self) -> int:
        cfg = self.randomization_config
        return max(cfg.max_imu_latency_steps, cfg.max_encoder_latency_steps)

    def _privileged_state(self) -> np.ndarray:
        cfg = self.randomization_config
        forward, lateral = self._heading_axes()
        forward_velocity, lateral_velocity, _ = self._body_velocities()
        return np.concatenate(
            (
                self._clean_frame(),
                [forward_velocity, lateral_velocity],
                _normalized(self._com_offset, self._max_com_offset),
                [
                    _normalized(self._mass_scale - 1.0, cfg.mass_fraction),
                    _normalized(self._friction_scale - 1.0, cfg.friction_fraction),
                    _normalized(self._push_force @ forward, cfg.impulse_force_n),
                    _normalized(self._push_force @ lateral, cfg.impulse_force_n),
                    _normalized(self._motor_latency, cfg.max_motor_latency_steps),
                    _normalized(self._imu_latency, cfg.max_imu_latency_steps),
                    _normalized(self._encoder_latency, cfg.max_encoder_latency_steps),
                    _normalized(self._ground_normal[:2] @ forward, np.sin(cfg.slope_radians)),
                    _normalized(self._ground_normal[:2] @ lateral, np.sin(cfg.slope_radians)),
                ],
            )
        )

    def _get_obs(self) -> np.ndarray:
        self._history.appendleft(self._actor_frame())
        while len(self._history) < self.history_length:
            self._history.append(self._history[-1])
        return np.concatenate((*self._history, self._privileged_state()))

    # ------------------------------------------------------------- randomization

    def _sample_command(self, allow_reversal: bool = False) -> None:
        # Stage 2 favours the newest command band; stage 3 (full range, slopes and pushes)
        # draws 0 or uniform. Stage 1 limits are 0. Mid-episode, some changes are reversals.
        cfg = self.curriculum_config
        linear_limit = self.command_limits[0]
        if (
            allow_reversal
            and linear_limit > 0
            and self.np_random.uniform() < cfg.reversal_probability
        ):
            previous = self._command[0]
            direction = -np.sign(previous) if previous != 0 else self.np_random.choice((-1, 1))
            speed = self.np_random.uniform(cfg.reversal_min_fraction, 1.0) * linear_limit
            self._command[:] = (direction * speed, 0.0)
            self._episode_reversal = True
        else:
            level = 1 if self._stage == CurriculumStage.DOMAIN_RANDOMIZATION else None
            self._command[:] = self.wheel_feasible_command(
                *(
                    self._sample_level_magnitude(limit, level)
                    * self.np_random.choice((-1.0, 1.0))
                    for limit in self.command_limits
                )
            )
        self._command_time_remaining = self.np_random.uniform(
            *self.curriculum_config.command_resample_seconds
        )
        self._command_age = 0.0

    def _sample_randomization(self) -> np.ndarray:
        """Randomize the model for stage 3 and return the ground orientation quaternion."""
        cfg = self.randomization_config
        randomize = self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION
        self._mass_scale = (
            self.np_random.uniform(1.0 - cfg.mass_fraction, 1.0 + cfg.mass_fraction)
            if randomize
            else 1.0
        )
        self._com_offset = (
            np.array(
                [
                    self._sample_level_magnitude(self.com_offset_limit)
                    * self.np_random.choice((-1.0, 1.0))
                    for _ in range(3)
                ]
            )
            if randomize
            else np.zeros(3)
        )
        self._friction_scale = (
            self.np_random.uniform(1.0 - cfg.friction_fraction, 1.0 + cfg.friction_fraction)
            if randomize
            else 1.0
        )
        self._motor_latency, self._imu_latency, self._encoder_latency = (
            (
                int(self.np_random.integers(0, cfg.max_motor_latency_steps + 1)),
                int(self.np_random.integers(0, cfg.max_imu_latency_steps + 1)),
                int(self.np_random.integers(0, cfg.max_encoder_latency_steps + 1)),
            )
            if randomize
            else (0, 0, 0)
        )
        self._slope = (
            self._sample_level_magnitude(self.slope_limit) * self.np_random.choice((-1.0, 1.0))
            if randomize
            else 0.0
        )
        slope_axis_angle = self.np_random.uniform(-np.pi, np.pi)

        overrides = self._physics_overrides
        if "slope" in overrides:
            self._slope, slope_axis_angle = overrides["slope"], 0.0
        if "friction" in overrides:
            self._friction_scale = overrides["friction"] / self._nominal_ground_friction
        if "com_offset" in overrides:
            self._com_offset = overrides["com_offset"].copy()

        ground_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            ground_quat,
            np.array([np.cos(slope_axis_angle), np.sin(slope_axis_angle), 0.0]),
            self._slope,
        )
        self.data.mocap_quat[self._terrain_mocap_id] = ground_quat
        self._ground_normal = np.zeros(3)
        mujoco.mju_rotVecQuat(self._ground_normal, np.array([0.0, 0.0, 1.0]), ground_quat)
        self._apply_physics()
        return ground_quat

    def _apply_physics(self) -> None:
        self.model.body_mass[self._base_body_id] = self._nominal_mass * self._mass_scale
        self.model.body_inertia[self._base_body_id] = self._nominal_inertia * self._mass_scale
        self.model.body_ipos[self._base_body_id] = self._nominal_com + self._com_offset
        # MuJoCo uses the larger friction of the two contacting geoms, so scale all of them.
        self.model.geom_friction[:] = self._nominal_friction
        self.model.geom_friction[:, 0] *= self._friction_scale

    def reset_model(self) -> np.ndarray:
        ground_quat = self._sample_randomization()
        if not self._command_held:
            self._sample_command()
        self._episode_step = 0
        self._episode_tracking[:] = 0.0
        self._episode_reversal = False
        self._previous_action[:] = 0.0
        self._action_queue.clear()
        self._sensor_buffer.clear()
        self._history.clear()
        self._push_force[:] = 0.0
        self._push_end_time = 0.0
        self._next_impulse_time = self.np_random.uniform(
            *self.randomization_config.impulse_interval_seconds
        )

        # Random heading on the (possibly sloped) ground: the axle lies on the slope, the
        # body is turned about the axle so it starts near vertical (initial lean only).
        yaw_quat, on_slope_quat, pitch_quat, base_quat = (np.zeros(4) for _ in range(4))
        mujoco.mju_axisAngle2Quat(
            yaw_quat, np.array([0.0, 0.0, 1.0]), self.np_random.uniform(-np.pi, np.pi)
        )
        mujoco.mju_mulQuat(on_slope_quat, ground_quat, yaw_quat)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, on_slope_quat)
        up = rotation.reshape(3, 3)[2]
        pitch_on_slope = np.arctan2(up[1], up[2])
        initial_pitch = self.np_random.uniform(
            -self.environment_config.initial_pitch_radians,
            self.environment_config.initial_pitch_radians,
        )
        mujoco.mju_axisAngle2Quat(
            pitch_quat, np.array([1.0, 0.0, 0.0]), initial_pitch - pitch_on_slope
        )
        mujoco.mju_mulQuat(base_quat, on_slope_quat, pitch_quat)
        base_position = np.zeros(3)
        mujoco.mju_rotVecQuat(
            base_position, self.init_qpos[self._base_qpos : self._base_qpos + 3], ground_quat
        )
        qpos = self.init_qpos.copy()
        qpos[self._base_qpos : self._base_qpos + 3] = base_position
        qpos[self._base_qpos + 3 : self._base_qpos + 7] = base_quat
        noise = self.environment_config.initial_state_noise
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _update_disturbance(self) -> None:
        self.data.xfrc_applied[self._base_body_id] = 0.0
        if self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            return
        cfg = self.randomization_config
        time = self._episode_step * self.dt
        if time >= self._next_impulse_time:
            direction = self.np_random.uniform(-np.pi, np.pi)
            magnitude = self._sample_level_magnitude(self.push_limit)
            self._push_force = magnitude * np.array([np.cos(direction), np.sin(direction)])
            self._push_end_time = time + cfg.impulse_duration_seconds
            self._next_impulse_time += self.np_random.uniform(*cfg.impulse_interval_seconds)
        if time >= self._push_end_time:
            self._push_force[:] = 0.0
        self.data.xfrc_applied[self._base_body_id, :2] = self._push_force

    # --------------------------------------------------------------- episode

    def _base_collision(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if geom_pair == {self._ground_geom_id, self._base_geom_id}:
                return True
        return False

    def _track_command(self, forward_velocity: float) -> None:
        """Accumulate the speed-ratio and standing-still sums used to level up."""
        cfg = self.curriculum_config
        if self._command_age >= cfg.speed_ratio_settle_seconds:
            linear = self._command[0]
            if linear == 0.0:
                self._episode_tracking[2:] += (abs(forward_velocity), 1.0)
            elif abs(linear) >= max(
                cfg.speed_ratio_min_fraction * self.command_limits[0], cfg.speed_ratio_min_command
            ):
                # Speed along the commanded direction (driving backwards counts against).
                self._episode_tracking[:2] += (forward_velocity * np.sign(linear), abs(linear))
        self._command_age += self.dt

    def _advance_curriculum(self, episode_success: bool) -> None:
        cfg = self.curriculum_config
        self._stage_results.append(episode_success)
        self._stage_tracking.append(self._episode_tracking.copy())
        self._stage_reversals.append((self._episode_reversal, episode_success))
        self._level_episodes += 1
        if len(self._stage_results) < cfg.evaluation_window:
            return
        success_rate = float(np.mean(self._stage_results))
        reached, commanded, still_speed, still_steps = np.sum(self._stage_tracking, axis=0)
        speed_ratio = reached / commanded if commanded > 0 else 1.0
        mean_still_speed = still_speed / still_steps if still_steps > 0 else 0.0
        reversal_results = [success for had, success in self._stage_reversals if had]
        reversal_rate = float(np.mean(reversal_results)) if reversal_results else 1.0
        if self._stage == CurriculumStage.STABILIZATION:
            passed = success_rate >= cfg.stage1_timeout_success_rate
        else:
            ramp = self._stage == CurriculumStage.COMMAND_RAMP
            passed = (
                self._level_episodes
                >= (
                    cfg.command_min_episodes_per_level
                    if ramp
                    else cfg.randomization_min_episodes_per_level
                )
                and success_rate
                >= (cfg.command_level_success_rate if ramp else cfg.randomization_level_success_rate)
                and speed_ratio
                >= (cfg.command_speed_ratio if ramp else cfg.randomization_speed_ratio)
                and mean_still_speed
                <= (cfg.command_still_speed_limit if ramp else cfg.randomization_still_speed_limit)
                and reversal_rate >= cfg.reversal_success_rate
            )
        if not passed:
            return
        if self._difficulty_level < self.max_level:
            self.set_difficulty_level(self._difficulty_level + 1)
        elif self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            self.set_curriculum_stage(self._stage + 1)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_queue.append(action)
        while len(self._action_queue) <= self._motor_latency:
            self._action_queue.appendleft(np.zeros(2))
        applied_action = self._action_queue.popleft()
        self._update_disturbance()
        self.do_simulation(applied_action * self._max_wheel_speed, self.frame_skip)
        self._episode_step += 1

        roll, pitch = self._tilt()
        pitch_rate = float(self.data.qvel[self._base_qvel + 3])
        forward_velocity, _, yaw_rate = self._body_velocities()
        linear_error = forward_velocity - self._command[0]
        angular_error = yaw_rate - self._command[1]
        self._track_command(forward_velocity)

        # Rewards use the command the policy was conditioned on and its own action
        # (a_t, a_{t-1}), which is also what the ESP32 feeds back as the previous action.
        cfg = self.reward_config
        reward_terms = {
            "alive": 1.0,
            # Penalize pitch motion, not the lean itself: the balance angle depends on the
            # CoM position and slope, and leaning is needed to accelerate.
            "pitch_rate": -(pitch_rate**2),
            "linear_velocity": _tracking_kernel(
                linear_error,
                cfg.linear_velocity_variance,
                cfg.linear_velocity_wide_variance,
                cfg.linear_velocity_wide_fraction,
            ),
            "angular_velocity": np.exp(-(angular_error**2) / cfg.angular_velocity_variance),
            "action": -float(np.square(action).sum()),
            "smoothness": -float(np.square(action - self._previous_action).sum()),
        }
        self._previous_action = action.copy()

        pitch_failure = max(abs(pitch), abs(roll)) > self.environment_config.upright_limit_radians
        base_collision = self._base_collision()
        terminated = bool(pitch_failure or base_collision)
        truncated = self._episode_step >= self.max_episode_steps
        reward = (
            cfg.alive_weight * reward_terms["alive"]
            + cfg.pitch_rate_weight * reward_terms["pitch_rate"]
            + cfg.linear_velocity_weight * reward_terms["linear_velocity"]
            + cfg.angular_velocity_weight * reward_terms["angular_velocity"]
            + cfg.action_weight * reward_terms["action"]
            + cfg.smoothness_weight * reward_terms["smoothness"]
        )
        if terminated:
            reward = 0.0
        info = {
            "pitch": pitch,
            "pitch_rate": pitch_rate,
            "roll": roll,
            "forward_velocity": forward_velocity,
            "yaw_rate": yaw_rate,
            "v_cmd": float(self._command[0]),
            "omega_cmd": float(self._command[1]),
            "curriculum_stage": int(self._stage),
            "curriculum_level": self._difficulty_level,
            "episode_step": self._episode_step,
            "terminated_reason": (
                "pitch_failure"
                if pitch_failure
                else "base_collision"
                if base_collision
                else "timeout"
                if truncated
                else None
            ),
        }
        info.update({f"reward_{name}": float(value) for name, value in reward_terms.items()})
        if terminated or truncated:
            self._advance_curriculum(not terminated)

        self._command_time_remaining -= self.dt
        if self._command_time_remaining <= 0 and not self._command_held:
            self._sample_command(allow_reversal=True)
        return self._get_obs(), float(reward), terminated, truncated, info
