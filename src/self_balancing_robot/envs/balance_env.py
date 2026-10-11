"""MuJoCo MJX environment of the two-wheel robot, written for JAX (jit/vmap on GPU).

The environment is functional: `reset(rng)` and `step(state, action)` return a new
`State` and keep everything that changes over an episode (command, sensor and action
buffers, randomized physics, curriculum stage) in `state.info`. Training runs
thousands of copies with `jax.vmap` (see `envs/wrappers.py`); evaluation, teleop and
export run single copies through `jax.jit`, on GPU or CPU.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from self_balancing_robot.config import (
    DEFAULT_ENVIRONMENT_CONFIG,
    DEFAULT_REWARD_CONFIG,
    EnvironmentConfig,
    RewardConfig,
)
from self_balancing_robot.curriculum import (
    BANDED,
    COMMAND_RESAMPLE_S,
    CURRICULUM,
    FULL_COMMAND,
    REVERSAL_MIN_FRACTION,
    SPEED_RATIO_MIN_COMMAND,
    SPEED_RATIO_MIN_FRACTION,
    SPEED_RATIO_SETTLE_S,
    Stage,
    stage_index,
    stage_table,
)

State = mjx_env.State

ASSETS = Path(__file__).parents[1] / "assets"

# One proprioceptive frame O_t, as the ESP32 builds it. The actor sees the last
# `history_length` frames, newest first. Commands are divided by the full command range
# (v / 0.4 m/s, omega / 2.5 rad/s) and wheel velocities by the motor no-load speed;
# everything else is in SI units (rad, rad/s). With raw commands (|v| <= 0.1 at the first
# command stage) the policy ignored them and stood still; scaled, it learned to follow them.
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
# stage and the teleop settings (held command, pinned physics; NaN = not pinned).
PERSISTENT_KEYS = (
    "stage",
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

    What each episode samples comes from a stage of ``curriculum`` (see `curriculum.py`).
    ``start_stage`` (index or name) is where episodes start; during training the curriculum
    wrapper moves on through the list (see `envs/wrappers.py`) unless ``fixed_difficulty``
    is set.
    """

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        environment_config: EnvironmentConfig | None = None,
        max_episode_steps: int | None = None,
        curriculum: tuple[Stage, ...] = CURRICULUM,
        start_stage: int | str = 0,
        fixed_difficulty: bool = False,
        impl: str = "jax",
    ):
        self.environment_config = environment_config or DEFAULT_ENVIRONMENT_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.curriculum = tuple(curriculum)
        # One row per stage; `stage_params` picks the current stage's row inside JAX code.
        self._table = stage_table(self.curriculum)
        # Shapes and the critic's input scale must not change between stages, so they come
        # from the largest values over the whole curriculum.
        table = self._table
        self._max_latency = int(table["max_latency_steps"].max())
        self._privileged_scale = {
            "com_offset": float(table["com_offset_max"].max()),
            "mass": float(table["mass_fraction"].max()),
            "friction": float(table["friction_fraction"].max()),
            "push": float(table["push_max"].max()),
            "slope": float(np.sin(table["slope_max"].max())),
            "latency": float(max(self._max_latency, 1)),
        }
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

        # Evaluation environments stay at their stage (no curriculum).
        self.fixed_difficulty = fixed_difficulty
        self.start_stage = self.stage_index(start_stage)

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

    @property
    def num_stages(self) -> int:
        return len(self.curriculum)

    def stage_index(self, stage: int | str) -> int:
        """Index of a curriculum stage given by name or index (-1 = the last one)."""
        return stage_index(self.curriculum, stage)

    def stage_params(self, stage) -> dict[str, jax.Array]:
        """The settings of a stage (`curriculum.stage_table` row); `stage` may be traced."""
        return jax.tree.map(lambda column: jnp.asarray(column)[stage], self._table)

    @staticmethod
    def sample_magnitude(
        rng: jax.Array, floor, maximum, zero_probability, frontier_probability
    ) -> jax.Array:
        """Magnitude in [0, maximum]: 0 with `zero_probability`, in the stage's new band
        [floor, maximum] with `frontier_probability`, otherwise below the band."""
        draw_key, frontier_key, below_key = jax.random.split(rng, 3)
        draw = jax.random.uniform(draw_key)
        frontier = jax.random.uniform(frontier_key, minval=floor, maxval=maximum)
        below = jax.random.uniform(below_key, minval=0.0, maxval=floor)
        # Without a band (floor 0) every non-zero draw is uniform up to the maximum.
        use_frontier = (draw < zero_probability + frontier_probability) | (floor <= 0.0)
        return jnp.where(draw < zero_probability, 0.0, jnp.where(use_frontier, frontier, below))

    def _sample_banded(self, rng: jax.Array, params: dict, quantity: str, signed: bool = True):
        """Draw `quantity` (command, slope, push, com_offset) of a stage, with a random sign."""
        maximum = params[BANDED[quantity]]
        floor = params[f"{quantity}_floor"]
        keys = jax.random.split(rng, maximum.size + 1)
        magnitude = jax.vmap(
            lambda key, lo, hi: self.sample_magnitude(
                key, lo, hi, params["zero_probability"], params["frontier_probability"]
            )
        )(keys[1:], jnp.ravel(floor), jnp.ravel(maximum)).reshape(maximum.shape)
        if not signed:
            return magnitude
        return magnitude * jnp.where(jax.random.bernoulli(keys[0], shape=maximum.shape), 1.0, -1.0)

    def wheel_feasible_command(self, linear_velocity, angular_velocity) -> jax.Array:
        """Scale (v, omega) down together so the outer wheel stays within full speed.

        At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors
        give while balancing: every v4 fall on flat ground came from such commands. The
        limit is the full command speed (0.4 m/s). The ESP32 should limit joystick commands
        the same way.
        """
        full_speed = FULL_COMMAND[0]
        outer_wheel = jnp.abs(linear_velocity) + jnp.abs(angular_velocity) * self._half_track
        scale = jnp.where(
            outer_wheel > 0, jnp.minimum(1.0, full_speed / jnp.maximum(outer_wheel, 1e-9)), 1.0
        )
        return jnp.stack([linear_velocity * scale, angular_velocity * scale])

    def _sample_command(
        self, rng: jax.Array, info: dict, allow_reversal: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """New (command, time until the next one, whether it is a reversal).

        Speed and yaw rate are drawn with the stage's band sampling; mid-episode, a change is
        a reversal with the stage's `reversal_probability`.
        """
        keys = jax.random.split(rng, 6)
        params = self.stage_params(info["stage"])
        limit = params["command_max"][0]
        previous = info["command"][0]
        direction = jnp.where(
            previous != 0,
            -jnp.sign(previous),
            jnp.where(jax.random.bernoulli(keys[0]), 1.0, -1.0),
        )
        speed = jax.random.uniform(keys[1], minval=REVERSAL_MIN_FRACTION, maxval=1.0)
        reversal_command = jnp.stack([direction * speed * limit, 0.0])
        reversal = (
            allow_reversal
            & (limit > 0)
            & (jax.random.uniform(keys[2]) < params["reversal_probability"])
        )
        random_command = self.wheel_feasible_command(
            *self._sample_banded(keys[3], params, "command")
        )
        command = jnp.where(reversal, reversal_command, random_command)
        duration = jax.random.uniform(
            keys[4], minval=COMMAND_RESAMPLE_S[0], maxval=COMMAND_RESAMPLE_S[1]
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
        return command / jnp.asarray(FULL_COMMAND)

    def _observe(self, data: mjx.Data, info: dict, rng: jax.Array, first: bool) -> tuple[dict, dict]:
        """Push the current sensor readings into the buffers; return (obs, info)."""
        params = self.stage_params(info["stage"])
        scale = self._privileged_scale
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
                params["imu_angle_noise"] * jax.random.normal(angle_key, (2,)),
                params["imu_rate_noise"] * jax.random.normal(rate_key, (3,)),
                params["encoder_noise"]
                / self._max_wheel_speed
                * jax.random.normal(encoder_key, (2,)),
            ]
        )
        measured = jnp.concatenate([imu, encoders]) + noise
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
        privileged = jnp.concatenate(
            [
                scaled_command,
                measurement,
                info["previous_action"],
                jnp.stack([forward_velocity, lateral_velocity]),
                _normalized(info["com_offset"], scale["com_offset"]),
                jnp.stack(
                    [
                        _normalized(info["mass_scale"] - 1.0, scale["mass"]),
                        _normalized(info["friction_scale"] - 1.0, scale["friction"]),
                        _normalized(info["push_force"] @ forward, scale["push"]),
                        _normalized(info["push_force"] @ lateral, scale["push"]),
                    ]
                ),
                latencies / scale["latency"],
                _normalized(
                    jnp.stack([info["ground_normal"][:2] @ forward, info["ground_normal"][:2] @ lateral]),
                    scale["slope"],
                ),
            ]
        )
        info = {**info, "sensor_buffer": sensors, "history": history}
        obs = {ACTOR_KEY: history.reshape(-1), CRITIC_KEY: privileged}
        return obs, info

    def _sensor_buffer_length(self) -> int:
        return self._max_latency + 1

    # ------------------------------------------------------------------ reset

    def initial_persistent(self) -> dict:
        """Persistent info for the first episode: start stage, nothing pinned."""
        return {
            "stage": jnp.int32(self.start_stage),
            "command_held": jnp.array(False),
            "command": jnp.zeros(2),
            "override_slope": jnp.float32(jnp.nan),
            "override_friction": jnp.float32(jnp.nan),
            "override_com": jnp.full(3, jnp.nan, dtype=jnp.float32),
        }

    def reset(self, rng: jax.Array) -> State:
        return self.reset_episode(rng, self.initial_persistent())

    def reset_episode(self, rng: jax.Array, persistent: dict) -> State:
        """Start an episode at the stage (and with the teleop settings) in `persistent`."""
        env_cfg = self.environment_config
        keys = jax.random.split(rng, 20)
        stage = persistent["stage"]
        params = self.stage_params(stage)

        # Randomized physics (zero ranges in early stages); teleop overrides take precedence.
        mass_scale = jax.random.uniform(
            keys[0], minval=1 - params["mass_fraction"], maxval=1 + params["mass_fraction"]
        )
        com_offset = self._sample_banded(keys[1], params, "com_offset")
        friction_scale = jax.random.uniform(
            keys[2], minval=1 - params["friction_fraction"], maxval=1 + params["friction_fraction"]
        )
        latency = [
            jax.random.randint(key, (), 0, params["max_latency_steps"] + 1) for key in keys[3:6]
        ]
        slope = self._sample_banded(keys[6], params, "slope")
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
            "episode_stage": stage,
            "command_held": persistent["command_held"],
            "override_slope": persistent["override_slope"],
            "override_friction": persistent["override_friction"],
            "override_com": persistent["override_com"],
            "command": persistent["command"],
            "episode_step": jnp.int32(0),
            "previous_action": jnp.zeros(2),
            "action_queue": jnp.zeros((self._max_latency + 1, 2)),
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
                keys[12], minval=params["push_interval_s"][0], maxval=params["push_interval_s"][1]
            ),
            # Sums for the stage's test: (speed reached along the command, commanded speed, |v| on zero
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
        """Random pushes on the base (if the stage has any), as a force in the world xy-plane."""
        params = self.stage_params(info["stage"])
        direction_key, magnitude_key, interval_key = jax.random.split(rng, 3)
        time = info["episode_step"] * self.dt
        pushes = params["push_max"] > 0
        start = pushes & (time >= info["next_impulse_time"])
        direction = jax.random.uniform(direction_key, minval=-jnp.pi, maxval=jnp.pi)
        magnitude = self._sample_banded(magnitude_key, params, "push", signed=False)
        push = jnp.where(
            start, magnitude * jnp.stack([jnp.cos(direction), jnp.sin(direction)]), info["push_force"]
        )
        push_end = jnp.where(start, time + params["push_duration_s"], info["push_end_time"])
        next_impulse = info["next_impulse_time"] + jnp.where(
            start,
            jax.random.uniform(
                interval_key,
                minval=params["push_interval_s"][0],
                maxval=params["push_interval_s"][1],
            ),
            0.0,
        )
        push = jnp.where((time >= push_end) | ~pushes, 0.0, push)
        return {
            **info,
            "push_force": push,
            "push_end_time": push_end,
            "next_impulse_time": next_impulse,
        }

    def _track_command(self, info: dict, forward_velocity: jax.Array) -> dict:
        """Accumulate the speed-ratio and standing-still sums of the stage's test."""
        linear = info["command"][0]
        settled = info["command_age"] >= SPEED_RATIO_SETTLE_S
        still = settled & (linear == 0.0)
        limit = self.stage_params(info["stage"])["command_max"][0]
        moving = (
            settled
            & (linear != 0.0)
            & (
                jnp.abs(linear)
                >= jnp.maximum(SPEED_RATIO_MIN_FRACTION * limit, SPEED_RATIO_MIN_COMMAND)
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
