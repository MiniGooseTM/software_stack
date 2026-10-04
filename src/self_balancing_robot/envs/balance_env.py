"""MuJoCo MJX environment of the two-wheel robot, written for JAX (jit/vmap on GPU).

The environment is functional: `reset(rng)` and `step(state, action)` return a new
`State` and keep everything that changes over an episode (command, sensor and action
buffers, randomized physics, curriculum stage and level) in `state.info`. Training runs
thousands of copies with `jax.vmap` (see `envs/wrappers.py`); evaluation, teleop and
export run single copies through `jax.jit`, on GPU or CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

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

State = mjx_env.State

ASSETS = Path(__file__).parents[1] / "assets"

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

# Observation keys: the actor (ESP32 policy) reads "state", the critic "privileged_state".
ACTOR_KEY = "state"
CRITIC_KEY = "privileged_state"

# Info entries carried over from one episode to the next by `reset_episode`: the curriculum
# position and the teleop settings (held command, pinned physics; NaN = not pinned).
PERSISTENT_KEYS = (
    "stage",
    "level",
    "command_held",
    "command",
    "override_slope",
    "override_friction",
    "override_com",
)

# Per-step values summed over an episode by the training and evaluation wrappers.
METRIC_NAMES = ("linear_error", "angular_error", "abs_pitch_deg", "survived")
REWARD_TERMS = ("alive", "pitch_rate", "linear_velocity", "angular_velocity", "action", "smoothness")


def _tracking_kernel(
    error: jax.Array, narrow_variance: float, wide_variance: float, wide_fraction: float
) -> jax.Array:
    """Blend of a narrow and a wide Gaussian kernel, bounded in [0, 1]."""
    return (1.0 - wide_fraction) * jnp.exp(-(error**2) / narrow_variance) + wide_fraction * jnp.exp(
        -(error**2) / wide_variance
    )


def _normalized(value: jax.Array, limit: float) -> jax.Array:
    return value / limit if limit > 0 else value * 0.0


# ----------------------------------------------------------------- quaternions (w, x, y, z)


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def axis_angle_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
    axis = jnp.asarray(axis, dtype=jnp.float32)
    return jnp.concatenate([jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis])


def quat_to_mat(q: jax.Array) -> jax.Array:
    """Rotation matrix whose columns are the body axes in world coordinates."""
    w, x, y, z = q / jnp.linalg.norm(q)
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotate(vector: jax.Array, q: jax.Array) -> jax.Array:
    return quat_to_mat(q) @ vector


def tilt(rotation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(roll, pitch) from the gravity direction in the body frame, as an accelerometer at
    rest would see it. Body axes: x = wheel axle (pitch), y = forward (roll), z = up."""
    up = rotation[2]
    pitch = jnp.arctan2(up[1], up[2])
    roll = jnp.arctan2(-up[0], jnp.hypot(up[1], up[2]))
    return roll, pitch


def load_model(model_file: str, collision_only: bool = False) -> mujoco.MjModel:
    """Load the MuJoCo model; `collision_only` drops the visual meshes (MJX training model)."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / model_file))
    if collision_only:
        for geom in list(spec.geoms):
            if geom.contype == 0 and geom.conaffinity == 0:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    return spec.compile()


class SelfBalancingRobotEnv(mjx_env.MjxEnv):
    """3D two-wheel robot for asymmetric actor-critic PPO with curriculum learning.

    The observation is ``{"state": actor_history, "privileged_state": privileged_state}``.
    ``state`` (``actor_observation_size`` values) is what the deployed policy receives;
    ``privileged_state`` is only used by the critic during training.

    ``curriculum_stage`` and ``difficulty_level`` set where new episodes start; during
    training the curriculum wrapper moves them (see `envs/wrappers.py`) unless
    ``fixed_difficulty`` is set.
    """

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
        randomization_config: RandomizationConfig | None = None,
        environment_config: EnvironmentConfig | None = None,
        max_episode_steps: int | None = None,
        curriculum_stage: int = CurriculumStage.STABILIZATION,
        difficulty_level: int = 1,
        fixed_difficulty: bool = False,
        impl: str = "jax",
    ):
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

        # Full model (meshes) for rendering; the MJX model only has the collision geoms.
        # Both have the same bodies, joints and qpos layout.
        self._render_model = load_model(self.environment_config.model_file)
        self._mj_model = load_model(self.environment_config.model_file, collision_only=True)
        self._mj_model.opt.iterations = self.environment_config.solver_iterations
        self._mj_model.opt.ls_iterations = self.environment_config.solver_ls_iterations
        timestep = float(self._mj_model.opt.timestep)
        substeps = max(round(self.environment_config.control_period_s / timestep), 1)
        super().__init__(config_dict.create(ctrl_dt=substeps * timestep, sim_dt=timestep))
        self._impl = impl
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)

        model = self._mj_model
        self._base_body_id = model.body("base_link").id
        self._ground_geom_id = model.geom("ground").id
        self._base_geom_id = model.geom("base_link_collision").id
        base_joint = model.joint("base_link_joint")
        self._base_qpos = int(base_joint.qposadr[0])
        self._base_qvel = int(base_joint.dofadr[0])
        self._wheel_qvel = np.array(
            [int(model.joint(name).dofadr[0]) for name in ("left_wheel_joint", "right_wheel_joint")]
        )
        # Normalized action a in [-1, 1] is the PWM duty; the velocity actuator turns
        # duty * no-load speed into DC-motor torque.
        self._max_wheel_speed = float(model.actuator_ctrlrange[:, 1].min())
        self._wheel_radius = float(model.geom("left_wheel_collision").size[0])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        left = data.geom("left_wheel_collision").xpos
        right = data.geom("right_wheel_collision").xpos
        self._half_track = 0.5 * float(np.linalg.norm(left - right))
        # Speeds are measured at the middle of the axle, not at the body origin (which sits
        # off-centre, so turning would add a fake forward speed there).
        base_rotation = data.xmat[self._base_body_id].reshape(3, 3)
        self._axle_center = base_rotation.T @ (0.5 * (left + right) - data.xpos[self._base_body_id])

        self._nominal_mass = float(model.body_mass[self._base_body_id])
        self._nominal_inertia = model.body_inertia[self._base_body_id].copy()
        self._nominal_com = model.body_ipos[self._base_body_id].copy()
        self._nominal_ground_friction = float(model.geom_friction[self._ground_geom_id, 0])
        self._init_qpos = model.qpos0.copy()

        mjx_data = mjx.make_data(self._mjx_model)
        contact_geoms = np.asarray(mjx_data._impl.contact.geom)
        pair = {self._ground_geom_id, self._base_geom_id}
        self._base_contacts = np.array([set(map(int, g)) == pair for g in contact_geoms])

        # Evaluation environments stay at their stage and level (no curriculum).
        self.fixed_difficulty = fixed_difficulty
        self.start_stage = int(CurriculumStage(curriculum_stage))
        self.start_level = int(np.clip(difficulty_level, 1, self.max_level(self.start_stage)))

    # ------------------------------------------------------------------ MjxEnv interface

    @property
    def xml_path(self) -> str:
        return str(ASSETS / self.environment_config.model_file)

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Model with the visual meshes (rendering); same qpos layout as the MJX model."""
        return self._render_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def half_track(self) -> float:
        return self._half_track

    @property
    def max_wheel_speed(self) -> float:
        return self._max_wheel_speed

    @property
    def nominal_ground_friction(self) -> float:
        return self._nominal_ground_friction

    # ------------------------------------------------------------------ curriculum

    def max_level(self, stage) -> Any:
        """Number of levels of a stage (1 for stage 1); works on Python ints and arrays."""
        cfg = self.curriculum_config
        levels = (1, cfg.command_levels, cfg.randomization_levels)
        if isinstance(stage, (int, np.integer)):
            return levels[int(stage) - 1]
        return jnp.array(levels)[stage - 1]

    def _level_fraction(self, stage, level) -> jax.Array:
        return level / self.max_level(stage)

    def command_limits(self, stage, level) -> jax.Array:
        """Maximum |v_cmd| (m/s) and |omega_cmd| (rad/s) at a stage and level."""
        fraction = jnp.where(
            stage == CurriculumStage.COMMAND_RAMP, self._level_fraction(stage, level), 1.0
        )
        fraction = jnp.where(stage == CurriculumStage.STABILIZATION, 0.0, fraction)
        return fraction * jnp.asarray(self.curriculum_config.command_limits)

    def _randomization_fraction(self, stage, level) -> jax.Array:
        return jnp.where(
            stage >= CurriculumStage.DOMAIN_RANDOMIZATION, self._level_fraction(stage, level), 0.0
        )

    def slope_limit(self, stage, level) -> jax.Array:
        return self._randomization_fraction(stage, level) * self.randomization_config.slope_radians

    def push_limit(self, stage, level) -> jax.Array:
        return self._randomization_fraction(stage, level) * self.randomization_config.impulse_force_n

    def com_offset_limit(self, stage, level) -> jax.Array:
        return (
            self._randomization_fraction(stage, level)
            * self.randomization_config.center_of_mass_offset_m
        )

    def sample_level_magnitude(self, rng: jax.Array, limit: jax.Array, level) -> jax.Array:
        """Magnitude in [0, limit] weighted towards the newest band of ``level`` (see config)."""
        cfg = self.curriculum_config
        draw_key, frontier_key, below_key = jax.random.split(rng, 3)
        band_floor = jnp.maximum(limit - limit / level, 0.0)
        draw = jax.random.uniform(draw_key)
        frontier = jax.random.uniform(frontier_key, minval=band_floor, maxval=limit)
        below = jax.random.uniform(below_key, minval=0.0, maxval=band_floor)
        # At level 1 nothing lies below the newest band, so that share goes to the band.
        use_frontier = (draw < cfg.level_zero_probability + cfg.level_frontier_probability) | (
            band_floor <= 0.0
        )
        return jnp.where(
            draw < cfg.level_zero_probability, 0.0, jnp.where(use_frontier, frontier, below)
        )

    def wheel_feasible_command(self, linear_velocity, angular_velocity) -> jax.Array:
        """Scale (v, omega) down together so the outer wheel stays within full speed.

        At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors
        give while balancing: every v4 fall on flat ground came from such commands. The
        limit is the full command speed (0.3 m/s). The ESP32 should limit joystick commands
        the same way.
        """
        full_speed = self.curriculum_config.command_limits[0]
        outer_wheel = jnp.abs(linear_velocity) + jnp.abs(angular_velocity) * self._half_track
        scale = jnp.where(
            outer_wheel > 0, jnp.minimum(1.0, full_speed / jnp.maximum(outer_wheel, 1e-9)), 1.0
        )
        return jnp.stack([linear_velocity * scale, angular_velocity * scale])

    def _sample_command(
        self, rng: jax.Array, info: dict, allow_reversal: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """New (command, time until the next one, whether it is a reversal).

        Stage 2 favours the newest command band; stage 3 (full range, slopes and pushes)
        draws 0 or uniform. Stage 1 limits are 0. Mid-episode, some changes are reversals.
        """
        cfg = self.curriculum_config
        keys = jax.random.split(rng, 8)
        stage, level = info["stage"], info["level"]
        limits = self.command_limits(stage, level)
        previous = info["command"][0]
        direction = jnp.where(
            previous != 0,
            -jnp.sign(previous),
            jnp.where(jax.random.bernoulli(keys[0]), 1.0, -1.0),
        )
        speed = jax.random.uniform(keys[1], minval=cfg.reversal_min_fraction, maxval=1.0)
        reversal_command = jnp.stack([direction * speed * limits[0], 0.0])
        reversal = (
            allow_reversal & (limits[0] > 0) & (jax.random.uniform(keys[2]) < cfg.reversal_probability)
        )
        sample_level = jnp.where(stage == CurriculumStage.DOMAIN_RANDOMIZATION, 1, level)
        signs = jnp.where(jax.random.bernoulli(keys[3], shape=(2,)), 1.0, -1.0)
        random_command = self.wheel_feasible_command(
            self.sample_level_magnitude(keys[4], limits[0], sample_level) * signs[0],
            self.sample_level_magnitude(keys[5], limits[1], sample_level) * signs[1],
        )
        command = jnp.where(reversal, reversal_command, random_command)
        duration = jax.random.uniform(
            keys[6],
            minval=cfg.command_resample_seconds[0],
            maxval=cfg.command_resample_seconds[1],
        )
        return command, duration, reversal

    # ------------------------------------------------------------------ physics

    def _episode_model(self, info: dict) -> mjx.Model:
        """MJX model with this episode's mass, CoM and friction."""
        model = self._mjx_model
        base = self._base_body_id
        return model.replace(
            body_mass=model.body_mass.at[base].set(self._nominal_mass * info["mass_scale"]),
            body_inertia=model.body_inertia.at[base].set(
                self._nominal_inertia * info["mass_scale"]
            ),
            body_ipos=model.body_ipos.at[base].set(self._nominal_com + info["com_offset"]),
            # MuJoCo uses the larger friction of the two contacting geoms, so scale all.
            geom_friction=model.geom_friction.at[:, 0].multiply(info["friction_scale"]),
        )

    def _rotation(self, qpos: jax.Array) -> jax.Array:
        return quat_to_mat(qpos[self._base_qpos + 3 : self._base_qpos + 7])

    @staticmethod
    def heading_axes(rotation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Unit forward and lateral vectors of the robot projected on the world xy-plane."""
        forward = rotation[:2, 1]
        forward = forward / jnp.maximum(jnp.linalg.norm(forward), 1e-9)
        lateral = jnp.stack([forward[1], -forward[0]])
        return forward, lateral

    def _body_velocities(
        self, rotation: jax.Array, qvel: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """True forward speed and lateral speed of the axle centre, and world yaw rate."""
        forward, lateral = self.heading_axes(rotation)
        angular_world = rotation @ qvel[self._base_qvel + 3 : self._base_qvel + 6]
        # Free joint: linear velocity of the body origin in world frame.
        linear = qvel[self._base_qvel : self._base_qvel + 3] + jnp.cross(
            angular_world, rotation @ self._axle_center
        )
        return linear[:2] @ forward, linear[:2] @ lateral, angular_world[2]

    def _imu_measurement(self, rotation: jax.Array, qvel: jax.Array) -> jax.Array:
        roll, pitch = tilt(rotation)
        pitch_rate, roll_rate, yaw_rate = qvel[self._base_qvel + 3 : self._base_qvel + 6]
        return jnp.stack([roll, pitch, roll_rate, pitch_rate, yaw_rate])

    def _encoder_measurement(self, qvel: jax.Array) -> jax.Array:
        return qvel[self._wheel_qvel] / self._max_wheel_speed

    def _base_collision(self, data: mjx.Data) -> jax.Array:
        if not self._base_contacts.any():
            return jnp.array(False)
        dist = data._impl.contact.dist[self._base_contacts]
        return jnp.any(dist < 0)

    # ------------------------------------------------------------------ observation

    def _scaled_command(self, command: jax.Array) -> jax.Array:
        return command / jnp.asarray(self.curriculum_config.command_limits)

    def _observe(self, data: mjx.Data, info: dict, rng: jax.Array, first: bool) -> tuple[dict, dict]:
        """Push the current sensor readings into the buffers; return (obs, info)."""
        cfg = self.randomization_config
        rotation = self._rotation(data.qpos)
        measurement = jnp.concatenate(
            [self._imu_measurement(rotation, data.qvel), self._encoder_measurement(data.qvel)]
        )
        if first:
            sensors = jnp.tile(measurement, (self._sensor_buffer_length(), 1))
        else:
            sensors = jnp.roll(info["sensor_buffer"], 1, axis=0).at[0].set(measurement)
        imu = sensors[info["imu_latency"], :5]
        encoders = sensors[info["encoder_latency"], 5:]
        angle_key, rate_key, encoder_key = jax.random.split(rng, 3)
        noise = jnp.concatenate(
            [
                cfg.imu_angle_noise_rad * jax.random.normal(angle_key, (2,)),
                cfg.imu_rate_noise_rad_s * jax.random.normal(rate_key, (3,)),
                cfg.encoder_velocity_noise_rad_s
                / self._max_wheel_speed
                * jax.random.normal(encoder_key, (2,)),
            ]
        )
        noisy = info["stage"] >= CurriculumStage.DOMAIN_RANDOMIZATION
        measured = jnp.concatenate([imu, encoders]) + jnp.where(noisy, noise, 0.0)
        scaled_command = self._scaled_command(info["command"])
        frame = jnp.concatenate([scaled_command, measured, info["previous_action"]])
        if first:
            history = jnp.tile(frame, (self.history_length, 1))
        else:
            history = jnp.roll(info["history"], 1, axis=0).at[0].set(frame)

        forward, lateral = self.heading_axes(rotation)
        forward_velocity, lateral_velocity, _ = self._body_velocities(rotation, data.qvel)
        latencies = jnp.stack(
            [info["motor_latency"], info["imu_latency"], info["encoder_latency"]]
        ).astype(jnp.float32)
        max_latencies = np.array(
            [cfg.max_motor_latency_steps, cfg.max_imu_latency_steps, cfg.max_encoder_latency_steps],
            dtype=np.float32,
        )
        privileged = jnp.concatenate(
            [
                scaled_command,
                measurement,
                info["previous_action"],
                jnp.stack([forward_velocity, lateral_velocity]),
                _normalized(info["com_offset"], cfg.center_of_mass_offset_m),
                jnp.stack(
                    [
                        _normalized(info["mass_scale"] - 1.0, cfg.mass_fraction),
                        _normalized(info["friction_scale"] - 1.0, cfg.friction_fraction),
                        _normalized(info["push_force"] @ forward, cfg.impulse_force_n),
                        _normalized(info["push_force"] @ lateral, cfg.impulse_force_n),
                    ]
                ),
                latencies / np.maximum(max_latencies, 1.0),
                _normalized(
                    jnp.stack([info["ground_normal"][:2] @ forward, info["ground_normal"][:2] @ lateral]),
                    np.sin(cfg.slope_radians),
                ),
            ]
        )
        info = {**info, "sensor_buffer": sensors, "history": history}
        obs = {ACTOR_KEY: history.reshape(-1), CRITIC_KEY: privileged}
        return obs, info

    def _sensor_buffer_length(self) -> int:
        cfg = self.randomization_config
        return max(cfg.max_imu_latency_steps, cfg.max_encoder_latency_steps) + 1

    # ------------------------------------------------------------------ reset

    def initial_persistent(self) -> dict:
        """Persistent info for the first episode: start stage/level, nothing pinned."""
        return {
            "stage": jnp.int32(self.start_stage),
            "level": jnp.int32(self.start_level),
            "command_held": jnp.array(False),
            "command": jnp.zeros(2),
            "override_slope": jnp.float32(jnp.nan),
            "override_friction": jnp.float32(jnp.nan),
            "override_com": jnp.full(3, jnp.nan, dtype=jnp.float32),
        }

    def reset(self, rng: jax.Array) -> State:
        return self.reset_episode(rng, self.initial_persistent())

    def reset_episode(self, rng: jax.Array, persistent: dict) -> State:
        """Start an episode at the stage/level (and with the teleop settings) in `persistent`."""
        env_cfg = self.environment_config
        cfg = self.randomization_config
        keys = jax.random.split(rng, 20)
        stage, level = persistent["stage"], persistent["level"]
        randomize = stage >= CurriculumStage.DOMAIN_RANDOMIZATION

        def signed(key, limit):
            magnitude_key, sign_key = jax.random.split(key)
            sign = jnp.where(jax.random.bernoulli(sign_key), 1.0, -1.0)
            return self.sample_level_magnitude(magnitude_key, limit, level) * sign

        # Randomized physics (stage 3); teleop overrides take precedence.
        mass_scale = jnp.where(
            randomize,
            jax.random.uniform(keys[0], minval=1 - cfg.mass_fraction, maxval=1 + cfg.mass_fraction),
            1.0,
        )
        com_limit = self.com_offset_limit(stage, level)
        com_offset = jnp.where(
            randomize, jnp.stack([signed(k, com_limit) for k in jax.random.split(keys[1], 3)]), 0.0
        )
        friction_scale = jnp.where(
            randomize,
            jax.random.uniform(
                keys[2], minval=1 - cfg.friction_fraction, maxval=1 + cfg.friction_fraction
            ),
            1.0,
        )
        latency = [
            jnp.where(randomize, jax.random.randint(key, (), 0, maximum + 1), 0)
            for key, maximum in zip(
                keys[3:6],
                (
                    cfg.max_motor_latency_steps,
                    cfg.max_imu_latency_steps,
                    cfg.max_encoder_latency_steps,
                ),
            )
        ]
        slope = jnp.where(randomize, signed(keys[6], self.slope_limit(stage, level)), 0.0)
        slope_axis_angle = jax.random.uniform(keys[7], minval=-jnp.pi, maxval=jnp.pi)

        pinned_slope = ~jnp.isnan(persistent["override_slope"])
        slope = jnp.where(pinned_slope, persistent["override_slope"], slope)
        slope_axis_angle = jnp.where(pinned_slope, 0.0, slope_axis_angle)
        friction_scale = jnp.where(
            jnp.isnan(persistent["override_friction"]),
            friction_scale,
            persistent["override_friction"] / self._nominal_ground_friction,
        )
        com_offset = jnp.where(
            jnp.isnan(persistent["override_com"]), com_offset, persistent["override_com"]
        )

        ground_quat = axis_angle_quat(
            jnp.stack([jnp.cos(slope_axis_angle), jnp.sin(slope_axis_angle), 0.0]), slope
        )
        ground_normal = rotate(jnp.array([0.0, 0.0, 1.0]), ground_quat)

        # Random heading on the (possibly sloped) ground: the axle lies on the slope, the
        # body is turned about the axle so it starts near vertical (initial lean only).
        yaw_quat = axis_angle_quat(
            jnp.array([0.0, 0.0, 1.0]), jax.random.uniform(keys[8], minval=-jnp.pi, maxval=jnp.pi)
        )
        on_slope_quat = quat_mul(ground_quat, yaw_quat)
        up = quat_to_mat(on_slope_quat)[2]
        pitch_on_slope = jnp.arctan2(up[1], up[2])
        initial_pitch = jax.random.uniform(
            keys[9], minval=-env_cfg.initial_pitch_radians, maxval=env_cfg.initial_pitch_radians
        )
        pitch_quat = axis_angle_quat(jnp.array([1.0, 0.0, 0.0]), initial_pitch - pitch_on_slope)
        base_quat = quat_mul(on_slope_quat, pitch_quat)
        base_position = rotate(
            jnp.asarray(self._init_qpos[self._base_qpos : self._base_qpos + 3]), ground_quat
        )
        qpos = (
            jnp.asarray(self._init_qpos)
            .at[self._base_qpos : self._base_qpos + 3]
            .set(base_position)
            .at[self._base_qpos + 3 : self._base_qpos + 7]
            .set(base_quat)
        )
        noise = env_cfg.initial_state_noise
        qvel = jax.random.uniform(keys[10], (self._mj_model.nv,), minval=-noise, maxval=noise)
        data = mjx.make_data(self._mjx_model, impl=self._impl).replace(
            qpos=qpos, qvel=qvel, mocap_quat=ground_quat[None]
        )

        info = {
            "rng": keys[11],
            "stage": stage,
            "level": level,
            "episode_stage": stage,
            "episode_level": level,
            "command_held": persistent["command_held"],
            "override_slope": persistent["override_slope"],
            "override_friction": persistent["override_friction"],
            "override_com": persistent["override_com"],
            "command": persistent["command"],
            "episode_step": jnp.int32(0),
            "previous_action": jnp.zeros(2),
            "action_queue": jnp.zeros((cfg.max_motor_latency_steps + 1, 2)),
            "motor_latency": latency[0],
            "imu_latency": latency[1],
            "encoder_latency": latency[2],
            "mass_scale": mass_scale,
            "friction_scale": friction_scale,
            "com_offset": com_offset,
            "slope": slope,
            "ground_normal": ground_normal,
            "push_force": jnp.zeros(2),
            "push_end_time": jnp.float32(0.0),
            "next_impulse_time": jax.random.uniform(
                keys[12],
                minval=cfg.impulse_interval_seconds[0],
                maxval=cfg.impulse_interval_seconds[1],
            ),
            # Level-up sums: (speed reached along the command, commanded speed, |v| on zero
            # speed commands, steps on zero speed commands).
            "episode_tracking": jnp.zeros(4),
            "episode_reversal": jnp.array(False),
            "terminated": jnp.array(False),
            "truncation": jnp.float32(0.0),
            # Last step's readings (for plots and the teleop panel).
            "pitch": jnp.float32(0.0),
            "roll": jnp.float32(0.0),
            "forward_velocity": jnp.float32(0.0),
            "yaw_rate": jnp.float32(0.0),
            "reward_terms": {name: jnp.float32(0.0) for name in REWARD_TERMS},
        }
        command, duration, _ = self._sample_command(keys[13], info, allow_reversal=False)
        info["command"] = jnp.where(persistent["command_held"], persistent["command"], command)
        info["command_time_remaining"] = duration
        info["command_age"] = jnp.float32(0.0)
        info["sensor_buffer"] = jnp.zeros((self._sensor_buffer_length(), 7))
        info["history"] = jnp.zeros((self.history_length, ACTOR_FRAME_SIZE))
        obs, info = self._observe(data, info, keys[14], first=True)
        metrics = {name: jnp.float32(0.0) for name in METRIC_NAMES}
        return State(data, obs, jnp.float32(0.0), jnp.float32(0.0), metrics, info)

    # ------------------------------------------------------------------ step

    def _update_disturbance(self, info: dict, rng: jax.Array) -> dict:
        """Random pushes on the base (stage 3), as a force in the world xy-plane."""
        cfg = self.randomization_config
        direction_key, magnitude_key, interval_key = jax.random.split(rng, 3)
        time = info["episode_step"] * self.dt
        randomize = info["stage"] >= CurriculumStage.DOMAIN_RANDOMIZATION
        start = randomize & (time >= info["next_impulse_time"])
        direction = jax.random.uniform(direction_key, minval=-jnp.pi, maxval=jnp.pi)
        magnitude = self.sample_level_magnitude(
            magnitude_key, self.push_limit(info["stage"], info["level"]), info["level"]
        )
        push = jnp.where(
            start, magnitude * jnp.stack([jnp.cos(direction), jnp.sin(direction)]), info["push_force"]
        )
        push_end = jnp.where(start, time + cfg.impulse_duration_seconds, info["push_end_time"])
        next_impulse = info["next_impulse_time"] + jnp.where(
            start,
            jax.random.uniform(
                interval_key,
                minval=cfg.impulse_interval_seconds[0],
                maxval=cfg.impulse_interval_seconds[1],
            ),
            0.0,
        )
        push = jnp.where((time >= push_end) | ~randomize, 0.0, push)
        return {
            **info,
            "push_force": push,
            "push_end_time": push_end,
            "next_impulse_time": next_impulse,
        }

    def _track_command(self, info: dict, forward_velocity: jax.Array) -> dict:
        """Accumulate the speed-ratio and standing-still sums used to level up."""
        cfg = self.curriculum_config
        linear = info["command"][0]
        settled = info["command_age"] >= cfg.speed_ratio_settle_seconds
        still = settled & (linear == 0.0)
        limit = self.command_limits(info["stage"], info["level"])[0]
        moving = (
            settled
            & (linear != 0.0)
            & (
                jnp.abs(linear)
                >= jnp.maximum(cfg.speed_ratio_min_fraction * limit, cfg.speed_ratio_min_command)
            )
        )
        # Speed along the commanded direction (driving backwards counts against).
        update = jnp.where(
            moving,
            jnp.stack([forward_velocity * jnp.sign(linear), jnp.abs(linear), 0.0, 0.0]),
            jnp.where(still, jnp.stack([0.0, 0.0, jnp.abs(forward_velocity), 1.0]), 0.0),
        )
        return {
            **info,
            "episode_tracking": info["episode_tracking"] + update,
            "command_age": info["command_age"] + self.dt,
        }

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        rng, push_key, command_key, noise_key = jax.random.split(info["rng"], 4)
        info["rng"] = rng

        action = jnp.clip(action, -1.0, 1.0)
        queue = jnp.roll(info["action_queue"], 1, axis=0).at[0].set(action)
        applied_action = queue[info["motor_latency"]]
        info["action_queue"] = queue

        info = self._update_disturbance(info, push_key)
        xfrc = jnp.zeros((self._mj_model.nbody, 6)).at[self._base_body_id, :2].set(info["push_force"])
        data = state.data.replace(xfrc_applied=xfrc)
        data = mjx_env.step(
            self._episode_model(info), data, applied_action * self._max_wheel_speed, self.n_substeps
        )
        info["episode_step"] = info["episode_step"] + 1

        rotation = self._rotation(data.qpos)
        roll, pitch = tilt(rotation)
        pitch_rate = data.qvel[self._base_qvel + 3]
        forward_velocity, _, yaw_rate = self._body_velocities(rotation, data.qvel)
        linear_error = forward_velocity - info["command"][0]
        angular_error = yaw_rate - info["command"][1]
        info = self._track_command(info, forward_velocity)

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
            "angular_velocity": jnp.exp(-(angular_error**2) / cfg.angular_velocity_variance),
            "action": -jnp.sum(jnp.square(action)),
            "smoothness": -jnp.sum(jnp.square(action - info["previous_action"])),
        }
        info["previous_action"] = action

        env_cfg = self.environment_config
        pitch_failure = jnp.maximum(jnp.abs(pitch), jnp.abs(roll)) > env_cfg.upright_limit_radians
        spinning = jnp.abs(yaw_rate) > env_cfg.yaw_rate_limit_rad_s
        terminated = pitch_failure | spinning | self._base_collision(data)
        truncated = info["episode_step"] >= self.max_episode_steps
        reward = (
            cfg.alive_weight * reward_terms["alive"]
            + cfg.pitch_rate_weight * reward_terms["pitch_rate"]
            + cfg.linear_velocity_weight * reward_terms["linear_velocity"]
            + cfg.angular_velocity_weight * reward_terms["angular_velocity"]
            + cfg.action_weight * reward_terms["action"]
            + cfg.smoothness_weight * reward_terms["smoothness"]
        )
        reward = jnp.where(terminated, 0.0, reward)
        done = terminated | truncated
        info["terminated"] = terminated
        info["truncation"] = (truncated & ~terminated).astype(jnp.float32)
        info["pitch"] = pitch
        info["roll"] = roll
        info["forward_velocity"] = forward_velocity
        info["yaw_rate"] = yaw_rate
        info["reward_terms"] = {name: jnp.asarray(value, jnp.float32) for name, value in reward_terms.items()}

        # Mid-episode command changes (a reversal on the final step does not count for it).
        remaining = info["command_time_remaining"] - self.dt
        command, duration, reversal = self._sample_command(command_key, info, allow_reversal=True)
        resample = (remaining <= 0) & ~info["command_held"]
        info["command"] = jnp.where(resample, command, info["command"])
        info["command_time_remaining"] = jnp.where(resample, duration, remaining)
        info["command_age"] = jnp.where(resample, 0.0, info["command_age"])
        info["episode_reversal"] = info["episode_reversal"] | (resample & reversal & ~done)

        obs, info = self._observe(data, info, noise_key, first=False)
        metrics = {
            "linear_error": jnp.abs(linear_error),
            "angular_error": jnp.abs(angular_error),
            "abs_pitch_deg": jnp.degrees(jnp.abs(pitch)),
            "survived": info["truncation"],
        }
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jnp.float32),
            metrics=metrics,
            info=info,
        )
