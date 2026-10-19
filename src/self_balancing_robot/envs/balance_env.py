<<<<<<< HEAD
"""Gymnasium environment for the two-wheel self-balancing robot.

The observation is ``[actor_history, privileged_state]``. The first
``actor_observation_size`` values are what the deployed policy receives; the rest is
privileged simulation state used only by the training critic.
=======
"""MuJoCo MJX environment of the two-wheel robot, written for JAX (jit/vmap on GPU).

The environment is functional: `reset(rng)` and `step(state, action)` return a new
`State` and keep everything that changes over an episode (command, sensor and action
buffers, randomized physics, curriculum stage) in `state.info`. Training runs
thousands of copies with `jax.vmap` (see `envs/wrappers.py`); evaluation, teleop and
export run single copies through `jax.jit`, on GPU or CPU.
>>>>>>> gpu_training
"""

from __future__ import annotations

<<<<<<< HEAD
import math
from collections import deque
=======
>>>>>>> gpu_training
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
    EnvironmentConfig,
)
from self_balancing_robot.curriculum import (
    BANDED,
    COMMAND_RESAMPLE_S,
    CURRICULUM,
    FULL_COMMAND,
    NOMINAL_VOLTAGE,
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
<<<<<<< HEAD
# `history_length` frames, newest first. Commands and wheel velocities are normalized;
# everything else is in SI units (rad, rad/s).
=======
# `history_length` frames, newest first. Commands are divided by the full command range
# (v / 0.3 m/s, omega / 1.5 rad/s) and wheel velocities by the motor no-load speed;
# everything else is in SI units (rad, rad/s). With raw commands (|v| <= 0.1 at the first
# command stage) the policy ignored them and stood still; scaled, it learned to follow them.
>>>>>>> gpu_training
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
    "battery_voltage_deviation",
    "push_force_forward",
    "push_force_lateral",
    "wheel_kick_left",
    "wheel_kick_right",
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


<<<<<<< HEAD
def _weighted_reward(terms: dict[str, float], config: RewardConfig) -> float:
    """Sum each term scaled by its ``<name>_weight`` in the reward config."""
    return sum(getattr(config, f"{name}_weight") * value for name, value in terms.items())


def _end_reason(fell: bool, collided: bool, truncated: bool) -> str | None:
    if fell:
        return "pitch_failure"
    if collided:
        return "base_collision"
    if truncated:
        return "timeout"
    return None


class SelfBalancingRobotEnv(MujocoEnv, gym.utils.EzPickle):
    """3D two-wheel robot for asymmetric actor-critic PPO with curriculum learning."""
=======
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
>>>>>>> gpu_training

    What each episode samples comes from a stage of ``curriculum`` (see `curriculum.py`).
    ``start_stage`` (index or name) is where episodes start; during training the curriculum
    wrapper moves on through the list (see `envs/wrappers.py`) unless ``fixed_difficulty``
    is set.
    """

    def __init__(
        self,
        environment_config: EnvironmentConfig | None = None,
        max_episode_steps: int | None = None,
        curriculum: tuple[Stage, ...] = CURRICULUM,
        start_stage: int | str = 0,
        fixed_difficulty: bool = False,
        impl: str = "jax",
    ):
        self.environment_config = environment_config or DEFAULT_ENVIRONMENT_CONFIG
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
            "voltage": float(np.abs(table["battery_voltage"] - NOMINAL_VOLTAGE).max()),
            "push": float(table["push_max"].max()),
            "wheel_kick": float(table["wheel_kick_max"].max()),
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

<<<<<<< HEAD
        self._stage = CurriculumStage(curriculum_stage)
        self._bind_model()
        # `dt` and the buffer length are constant; the properties are called every step.
        self._dt = self.dt
        self._sensor_buffer_limit = 1 + max(
            self.randomization_config.max_imu_latency_steps,
            self.randomization_config.max_encoder_latency_steps,
        )
        self._init_episode_state()

    def _bind_model(self) -> None:
        """Look up model ids and the nominal physics values the curriculum scales."""
        self._base_body_id = self.model.body("base_link").id
        self._camera_config["trackbodyid"] = self._base_body_id
        self._ground_geom_id = self.model.geom("ground").id
        self._base_geom_id = self.model.geom("base_link_collision").id
        self._base_qpos = int(self.model.joint("base_link_joint").qposadr[0])
        self._base_qvel = int(self.model.joint("base_link_joint").dofadr[0])
        wheel_names = ("left_wheel_joint", "right_wheel_joint")
        self._wheel_qvel = np.array([int(self.model.joint(name).dofadr[0]) for name in wheel_names])
        # The normalized action is the PWM duty; the velocity actuator turns
=======
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
>>>>>>> gpu_training
        # duty * no-load speed into DC-motor torque.
        self._max_wheel_speed = float(model.actuator_ctrlrange[:, 1].min())
        self._nominal_ctrlrange = model.actuator_ctrlrange.copy()
        self._nominal_forcerange = model.actuator_forcerange.copy()
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

    def _sample_banded(
        self, rng: jax.Array, params: dict, quantity: str, signed: bool = True, shape=None
    ):
        """Draw `quantity` (a key of `BANDED`) of a stage, with a random sign; `shape` draws
        independent copies of a scalar quantity (e.g. one per wheel)."""
        maximum = params[BANDED[quantity]]
        floor = params[f"{quantity}_floor"]
        if shape is not None:
            maximum, floor = jnp.broadcast_to(maximum, shape), jnp.broadcast_to(floor, shape)
        keys = jax.random.split(rng, maximum.size + 1)
        magnitude = jax.vmap(
            lambda key, lo, hi: self.sample_magnitude(
                key, lo, hi, params["zero_probability"], params["frontier_probability"]
            )
        )(keys[1:], jnp.ravel(floor), jnp.ravel(maximum)).reshape(maximum.shape)
        if not signed:
            return magnitude
        return magnitude * jnp.where(jax.random.bernoulli(keys[0], shape=maximum.shape), 1.0, -1.0)

<<<<<<< HEAD
        self._nominal_mass = float(self.model.body_mass[self._base_body_id])
        self._nominal_inertia = self.model.body_inertia[self._base_body_id].copy()
        self._nominal_com = self.model.body_ipos[self._base_body_id].copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self._terrain_mocap_id = int(self.model.body("terrain").mocapid[0])
        self._nominal_ground_friction = float(self._nominal_friction[self._ground_geom_id, 0])
        self._max_com_offset = self.randomization_config.center_of_mass_offset_m

    def _init_episode_state(self) -> None:
        """Create the curriculum window, command and per-episode buffers."""
        window = self.curriculum_config.evaluation_window
        self._difficulty_level = 1
        self._stage_results: deque[bool] = deque(maxlen=window)
        self._command = np.zeros(2, dtype=np.float64)
        self._command_time_remaining = 0.0
        self._command_held = False
        self._command_age = 0.0
        self._physics_overrides: dict[str, float | np.ndarray] = {}
        self._episode_step = 0
        # Per-episode sums for the level-up checks: (speed reached along the command,
        # commanded speed, |v| on zero speed commands, steps on zero speed commands).
        self._episode_tracking = np.zeros(4)
        self._stage_tracking: deque[np.ndarray] = deque(maxlen=window)
        # Whether this episode had a reversal command, and (had one, survived) per episode.
        self._episode_reversal = False
        self._stage_reversals: deque[tuple[bool, bool]] = deque(maxlen=window)
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

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """Apply ctrl and advance n_frames, without the base class's per-call shape copy."""
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        # Force quantities (cacc, cfrc) are not computed unless the model has a force sensor.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    # -------------------------------------------------------------- curriculum

    @property
    def curriculum_stage(self) -> int:
        return int(self._stage)

    @property
    def difficulty_level(self) -> int:
        """Level within the current stage, 1 .. ``max_level``."""
        return self._difficulty_level

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
    def curriculum_progress(self) -> float:
        """Stage plus the fraction of its levels passed, e.g. 2.5 = halfway through stage 2."""
        return self._stage + (self._difficulty_level - 1) / self.max_level

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

    def set_curriculum_stage(self, stage: int) -> None:
        """Change the stage (starting at its level 1); applies at the next reset."""
        self._stage = CurriculumStage(stage)
        self.set_difficulty_level(1)

    def set_difficulty_level(self, level: int) -> None:
        """Set the level of the current stage and restart the advancement statistics."""
        self._difficulty_level = int(np.clip(level, 1, self.max_level))
        self._level_episodes = 0
        self._stage_results.clear()
        self._stage_tracking.clear()
        self._stage_reversals.clear()

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

    def _advance_curriculum(self, episode_success: bool) -> None:
        """Record an episode and advance a level (or stage) once its test passes."""
        self._stage_results.append(episode_success)
        self._stage_tracking.append(self._episode_tracking.copy())
        self._stage_reversals.append((self._episode_reversal, episode_success))
        self._level_episodes += 1
        if len(self._stage_results) < self.curriculum_config.evaluation_window:
            return
        success_rate, speed_ratio, still_speed, reversal_rate = self._window_statistics()
        if not self._passed_level(success_rate, speed_ratio, still_speed, reversal_rate):
            return
        if self._difficulty_level < self.max_level:
            self.set_difficulty_level(self._difficulty_level + 1)
        elif self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            self.set_curriculum_stage(self._stage + 1)

    def _window_statistics(self) -> tuple[float, float, float, float]:
        """Success rate, speed ratio, mean |v| on stops and reversal survival over the window."""
        reached, commanded, still_speed, still_steps = np.sum(self._stage_tracking, axis=0)
        reversal_results = [success for had, success in self._stage_reversals if had]
        return (
            float(np.mean(self._stage_results)),
            reached / commanded if commanded > 0 else 1.0,
            still_speed / still_steps if still_steps > 0 else 0.0,
            float(np.mean(reversal_results)) if reversal_results else 1.0,
        )

    def _passed_level(
        self, success_rate: float, speed_ratio: float, still_speed: float, reversal_rate: float
    ) -> bool:
        cfg = self.curriculum_config
        if self._stage == CurriculumStage.STABILIZATION:
            return success_rate >= cfg.stage1_timeout_success_rate
        if self._stage == CurriculumStage.COMMAND_RAMP:
            min_episodes = cfg.command_min_episodes_per_level
            required_success = cfg.command_level_success_rate
            required_speed = cfg.command_speed_ratio
            still_limit = cfg.command_still_speed_limit
        else:
            min_episodes = cfg.randomization_min_episodes_per_level
            required_success = cfg.randomization_level_success_rate
            required_speed = cfg.randomization_speed_ratio
            still_limit = cfg.randomization_still_speed_limit
        return (
            self._level_episodes >= min_episodes
            and success_rate >= required_success
            and speed_ratio >= required_speed
            and still_speed <= still_limit
            and reversal_rate >= cfg.reversal_success_rate
        )

    # --------------------------------------------------------------- commands

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def hold_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """Use a fixed command (e.g. from a joystick) instead of the curriculum's random ones."""
        self._command[:] = self.wheel_feasible_command(linear_velocity, angular_velocity)
        self._command_held = True

    def wheel_feasible_command(
        self, linear_velocity: float, angular_velocity: float
    ) -> tuple[float, float]:
        """Scale (v, omega) down together so the outer wheel stays within full speed.

        At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors
        give while balancing. The limit is the full command speed (0.3 m/s). The ESP32
        should limit joystick commands the same way.
        """
        full_speed = self.curriculum_config.command_limits[0]
        outer_wheel = abs(linear_velocity) + abs(angular_velocity) * self._half_track
        scale = min(1.0, full_speed / outer_wheel) if outer_wheel > 0 else 1.0
        return linear_velocity * scale, angular_velocity * scale

    def _sample_command(self, allow_reversal: bool = False) -> None:
        """Draw a new command; stage 2+ may instead reverse the current one."""
        cfg = self.curriculum_config
        linear_limit = self.command_limits[0]
        if (
            allow_reversal
            and linear_limit > 0
            and self.np_random.uniform() < cfg.reversal_probability
        ):
            self._command[:] = self._reversal_command(linear_limit)
            self._episode_reversal = True
        else:
            self._command[:] = self._random_command()
        self._command_time_remaining = self.np_random.uniform(*cfg.command_resample_seconds)
        self._command_age = 0.0

    def _reversal_command(self, linear_limit: float) -> tuple[float, float]:
        """The opposite direction at 75-100% of the limit, no turning."""
        previous = self._command[0]
        direction = -np.sign(previous) if previous != 0 else self.np_random.choice((-1, 1))
        speed = (
            self.np_random.uniform(self.curriculum_config.reversal_min_fraction, 1.0)
            * linear_limit
        )
        return float(direction * speed), 0.0

    def _random_command(self) -> tuple[float, float]:
        """A feasible random (v, omega); stage 3 draws the full range, stage 2 the newest band."""
        level = 1 if self._stage == CurriculumStage.DOMAIN_RANDOMIZATION else None
        linear, angular = (
            self._sample_level_magnitude(limit, level) * self.np_random.choice((-1.0, 1.0))
            for limit in self.command_limits
        )
        return self.wheel_feasible_command(linear, angular)

    # ----------------------------------------------------------- observations

    def _forward_direction(self) -> tuple[float, float]:
        """Unit forward vector (x, y) of the robot projected on the world plane."""
        xmat = self.data.xmat[self._base_body_id]
        fx, fy = float(xmat[1]), float(xmat[4])
        norm = max(math.hypot(fx, fy), 1e-9)
        return fx / norm, fy / norm

    def _heading_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit forward and lateral vectors of the robot projected on the world xy-plane."""
        fx, fy = self._forward_direction()
        return np.array([fx, fy]), np.array([fy, -fx])

    def _tilt(self) -> tuple[float, float]:
        # Gravity direction in the body frame, as an accelerometer at rest would see it.
        # Body axes: x = wheel axle (pitch), y = forward (roll), z = up.
        xmat = self.data.xmat[self._base_body_id]
        ux, uy, uz = float(xmat[6]), float(xmat[7]), float(xmat[8])
        return math.atan2(-ux, math.hypot(uy, uz)), math.atan2(uy, uz)

    def _imu_measurement(self) -> np.ndarray:
        """[roll, pitch, roll_rate, pitch_rate, yaw_rate] in the body frame."""
        roll, pitch = self._tilt()
        qvel, dof = self.data.qvel, self._base_qvel
        pitch_rate, roll_rate, yaw_rate = (
            float(qvel[dof + 3]),
            float(qvel[dof + 4]),
            float(qvel[dof + 5]),
        )
        return np.array([roll, pitch, roll_rate, pitch_rate, yaw_rate])

    def _encoder_measurement(self) -> np.ndarray:
        """Wheel velocities divided by the motor no-load speed."""
        return self.data.qvel[self._wheel_qvel] / self._max_wheel_speed

    def _body_velocities(
        self, forward: tuple[float, float] | None = None
    ) -> tuple[float, float, float]:
        """True forward speed, lateral speed and world yaw rate.

        ``forward`` is the unit forward vector from `_forward_direction`; pass it when the
        caller already has it.
        """
        fx, fy = forward if forward is not None else self._forward_direction()
        qvel, dof = self.data.qvel, self._base_qvel
        vx, vy = float(qvel[dof]), float(qvel[dof + 1])
        wx, wy, wz = float(qvel[dof + 3]), float(qvel[dof + 4]), float(qvel[dof + 5])
        xmat = self.data.xmat[self._base_body_id]
        yaw_rate = float(xmat[6]) * wx + float(xmat[7]) * wy + float(xmat[8]) * wz
        return vx * fx + vy * fy, vx * fy - vy * fx, yaw_rate

    def _clean_frame(self, fresh: np.ndarray | None = None) -> np.ndarray:
        """Frame with perfect, undelayed sensors (for the critic)."""
        if fresh is None:
            fresh = np.concatenate((self._imu_measurement(), self._encoder_measurement()))
        return np.concatenate(
            (self._scaled_command(), fresh[:5], fresh[5:], self._previous_action)
        )

    def _actor_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Delayed, noisy frame and the fresh sensor sample it was built from."""
        imu, encoders, fresh = self._delayed_sensors()
        if self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION:
            cfg = self.randomization_config
            imu[:2] += self.np_random.normal(0.0, cfg.imu_angle_noise_rad, size=2)
            imu[2:] += self.np_random.normal(0.0, cfg.imu_rate_noise_rad_s, size=3)
            encoders += self.np_random.normal(
                0.0, cfg.encoder_velocity_noise_rad_s / self._max_wheel_speed, size=2
            )
        frame = np.concatenate((self._scaled_command(), imu, encoders, self._previous_action))
        return frame, fresh

    def _delayed_sensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """IMU and encoder samples from their respective latencies, plus the fresh sample."""
        fresh = np.concatenate((self._imu_measurement(), self._encoder_measurement()))
        self._sensor_buffer.appendleft(fresh)
        while len(self._sensor_buffer) > self._sensor_buffer_limit:
            self._sensor_buffer.pop()
        oldest = len(self._sensor_buffer) - 1
        imu = self._sensor_buffer[min(self._imu_latency, oldest)][:5].copy()
        encoders = self._sensor_buffer[min(self._encoder_latency, oldest)][5:].copy()
        return imu, encoders, fresh

    def _scaled_command(self) -> np.ndarray:
        return self._command / np.asarray(self.curriculum_config.command_limits)

    def _privileged_state(
        self,
        fresh: np.ndarray | None = None,
        measurements: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        """Noise-free frame plus the physical truths the actor must infer.

        ``measurements`` is (forward x, forward y, forward speed, lateral speed) when the
        caller already computed them this step.
        """
        cfg = self.randomization_config
        if measurements is None:
            fx, fy = self._forward_direction()
            forward_velocity, lateral_velocity, _ = self._body_velocities((fx, fy))
        else:
            fx, fy, forward_velocity, lateral_velocity = measurements
        push_forward = self._push_force[0] * fx + self._push_force[1] * fy
        push_lateral = self._push_force[0] * fy - self._push_force[1] * fx
        normal_x, normal_y = float(self._ground_normal[0]), float(self._ground_normal[1])
        physics = (
            _normalized(self._mass_scale - 1.0, cfg.mass_fraction),
            _normalized(self._friction_scale - 1.0, cfg.friction_fraction),
            _normalized(push_forward, cfg.impulse_force_n),
            _normalized(push_lateral, cfg.impulse_force_n),
            _normalized(self._motor_latency, cfg.max_motor_latency_steps),
            _normalized(self._imu_latency, cfg.max_imu_latency_steps),
            _normalized(self._encoder_latency, cfg.max_encoder_latency_steps),
            _normalized(normal_x * fx + normal_y * fy, np.sin(cfg.slope_radians)),
            _normalized(normal_x * fy - normal_y * fx, np.sin(cfg.slope_radians)),
        )
        return np.concatenate(
            (
                self._clean_frame(fresh),
                [forward_velocity, lateral_velocity],
                _normalized(self._com_offset, self._max_com_offset),
                physics,
            )
        )

    def _get_obs(
        self, measurements: tuple[float, float, float, float] | None = None
    ) -> np.ndarray:
        frame, fresh = self._actor_frame()
        self._history.appendleft(frame)
        while len(self._history) < self.history_length:
            self._history.append(self._history[-1])
        return np.concatenate((*self._history, self._privileged_state(fresh, measurements)))

    # -------------------------------------------------------- randomization

    def _sample_randomization(self) -> np.ndarray:
        """Randomize the model for stage 3 and place the ground; returns its quaternion."""
        randomize = self._stage >= CurriculumStage.DOMAIN_RANDOMIZATION
        self._slope, axis_angle = self._sample_physics(randomize)
        ground_quat = self._place_ground(axis_angle)
        self._apply_physics()
        return ground_quat

    def _sample_physics(self, randomize: bool) -> tuple[float, float]:
        """Draw the physics of the episode; returns the slope (rad) and its axis angle (rad)."""
        cfg = self.randomization_config
        self._mass_scale = (
            self.np_random.uniform(1.0 - cfg.mass_fraction, 1.0 + cfg.mass_fraction)
            if randomize
            else 1.0
        )
        self._com_offset = self._random_com_offset() if randomize else np.zeros(3)
        self._friction_scale = (
            self.np_random.uniform(1.0 - cfg.friction_fraction, 1.0 + cfg.friction_fraction)
            if randomize
            else 1.0
        )
        latencies = [0, 0, 0]
        if randomize:
            latencies = [
                int(self.np_random.integers(0, maximum + 1))
                for maximum in (
                    cfg.max_motor_latency_steps,
                    cfg.max_imu_latency_steps,
                    cfg.max_encoder_latency_steps,
                )
            ]
        self._motor_latency, self._imu_latency, self._encoder_latency = latencies
        slope = (
            self._sample_level_magnitude(self.slope_limit) * self.np_random.choice((-1.0, 1.0))
            if randomize
            else 0.0
        )
        axis_angle = self.np_random.uniform(-np.pi, np.pi)

        overrides = self._physics_overrides
        if "slope" in overrides:
            slope, axis_angle = overrides["slope"], 0.0
        if "friction" in overrides:
            self._friction_scale = overrides["friction"] / self._nominal_ground_friction
        if "com_offset" in overrides:
            self._com_offset = overrides["com_offset"].copy()
        return float(slope), axis_angle

    def _random_com_offset(self) -> np.ndarray:
        return np.array(
            [
                self._sample_level_magnitude(self.com_offset_limit)
                * self.np_random.choice((-1.0, 1.0))
                for _ in range(3)
            ]
        )

    def _place_ground(self, axis_angle: float) -> np.ndarray:
        """Tilt the terrain body to ``self._slope`` about ``axis_angle``."""
        ground_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            ground_quat,
            np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0]),
            self._slope,
        )
        self.data.mocap_quat[self._terrain_mocap_id] = ground_quat
        self._ground_normal = np.zeros(3)
        mujoco.mju_rotVecQuat(self._ground_normal, np.array([0.0, 0.0, 1.0]), ground_quat)
        return ground_quat

    def _apply_physics(self) -> None:
        self.model.body_mass[self._base_body_id] = self._nominal_mass * self._mass_scale
        self.model.body_inertia[self._base_body_id] = self._nominal_inertia * self._mass_scale
        self.model.body_ipos[self._base_body_id] = self._nominal_com + self._com_offset
        # MuJoCo uses the larger friction of the two contacting geoms, so scale all of them.
        self.model.geom_friction[:] = self._nominal_friction
        self.model.geom_friction[:, 0] *= self._friction_scale

    # ---------------------------------------------------------------- episode

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

        qpos = self.init_qpos.copy()
        position, quat = self._spawn_pose(ground_quat)
        qpos[self._base_qpos : self._base_qpos + 3] = position
        qpos[self._base_qpos + 3 : self._base_qpos + 7] = quat
        noise = self.environment_config.initial_state_noise
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _spawn_pose(self, ground_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base position and orientation on the (possibly sloped) ground.

        The heading is random, the axle lies on the slope and the body is turned about
        the axle so it starts near vertical (only the initial lean is random).
        """
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
        position = np.zeros(3)
        mujoco.mju_rotVecQuat(
            position, self.init_qpos[self._base_qpos : self._base_qpos + 3], ground_quat
        )
        return position, base_quat

    def _update_disturbance(self) -> None:
        self.data.xfrc_applied[self._base_body_id] = 0.0
        if self._stage < CurriculumStage.DOMAIN_RANDOMIZATION:
            return
        cfg = self.randomization_config
        time = self._episode_step * self._dt
        if time >= self._next_impulse_time:
            direction = self.np_random.uniform(-np.pi, np.pi)
            magnitude = self._sample_level_magnitude(self.push_limit)
            self._push_force = magnitude * np.array([np.cos(direction), np.sin(direction)])
            self._push_end_time = time + cfg.impulse_duration_seconds
            self._next_impulse_time += self.np_random.uniform(*cfg.impulse_interval_seconds)
        if time >= self._push_end_time:
            self._push_force[:] = 0.0
        self.data.xfrc_applied[self._base_body_id, :2] = self._push_force

    def _base_collision(self) -> bool:
        ground, base = self._ground_geom_id, self._base_geom_id
        contacts = self.data.contact
        for i in range(self.data.ncon):
            contact = contacts[i]
            if (contact.geom1 == ground and contact.geom2 == base) or (
                contact.geom1 == base and contact.geom2 == ground
            ):
                return True
        return False

    def _track_command(self, forward_velocity: float) -> None:
        """Accumulate the speed-ratio and standing-still sums used to level up."""
        cfg = self.curriculum_config
        if self._command_age >= cfg.speed_ratio_settle_seconds:
            linear = self._command[0]
            if linear == 0.0:
                self._episode_tracking[2:] += (abs(forward_velocity), 1.0)
            elif self._counts_towards_speed_ratio(linear):
                # Speed along the commanded direction (driving backwards counts against).
                self._episode_tracking[:2] += (forward_velocity * np.sign(linear), abs(linear))
        self._command_age += self._dt

    def _counts_towards_speed_ratio(self, linear: float) -> bool:
        """Small commands are ignored: their speed signal is lost in balancing wobble."""
        cfg = self.curriculum_config
        return abs(linear) >= max(
            cfg.speed_ratio_min_fraction * self.command_limits[0], cfg.speed_ratio_min_command
        )

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied_action = self._delayed_action(action)
        self._update_disturbance()
        self.do_simulation(applied_action * self._max_wheel_speed, self.frame_skip)
        self._episode_step += 1

        roll, pitch = self._tilt()
        pitch_rate = float(self.data.qvel[self._base_qvel + 3])
        forward = self._forward_direction()
        forward_velocity, lateral_velocity, yaw_rate = self._body_velocities(forward)
        self._track_command(forward_velocity)

        # Rewards use the command the policy was conditioned on and its own action
        # (a_t, a_{t-1}), which is also what the ESP32 feeds back as the previous action.
        terms = self._reward_terms(action, forward_velocity, yaw_rate, pitch_rate)
        self._previous_action = action.copy()

        fell = max(abs(pitch), abs(roll)) > self.environment_config.upright_limit_radians
        collided = self._base_collision()
        terminated = bool(fell or collided)
        truncated = self._episode_step >= self.max_episode_steps
        reward = 0.0 if terminated else _weighted_reward(terms, self.reward_config)
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
            "terminated_reason": _end_reason(fell, collided, truncated),
        }
        info.update({f"reward_{name}": float(value) for name, value in terms.items()})
        if terminated or truncated:
            self._advance_curriculum(not terminated)

        self._command_time_remaining -= self._dt
        if self._command_time_remaining <= 0 and not self._command_held:
            self._sample_command(allow_reversal=True)
        measurements = (*forward, forward_velocity, lateral_velocity)
        return self._get_obs(measurements), float(reward), terminated, truncated, info

    def _delayed_action(self, action: np.ndarray) -> np.ndarray:
        """The action applied this step; motor latency delays it by a few steps."""
        self._action_queue.append(action)
        while len(self._action_queue) <= self._motor_latency:
            self._action_queue.appendleft(np.zeros(2))
        return self._action_queue.popleft()

    def _reward_terms(
        self, action: np.ndarray, forward_velocity: float, yaw_rate: float, pitch_rate: float
    ) -> dict[str, float]:
        cfg = self.reward_config
        linear_error = forward_velocity - self._command[0]
        angular_error = yaw_rate - self._command[1]
        return {
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

    # -------------------------------------------------------------- teleop API

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

    @property
    def nominal_friction(self) -> float:
        return self._nominal_ground_friction

    @property
    def half_track(self) -> float:
        return self._half_track

    @property
    def max_wheel_speed(self) -> float:
        """Motor no-load speed (rad/s) that a full duty command targets."""
        return self._max_wheel_speed

    @property
    def base_body_id(self) -> int:
        return self._base_body_id

    @property
    def heading_degrees(self) -> float:
        """Heading of the robot's forward axis, degrees from world +x."""
        forward, _ = self._heading_axes()
        return float(np.degrees(np.arctan2(forward[1], forward[0])))
=======
    def wheel_feasible_command(self, linear_velocity, angular_velocity) -> jax.Array:
        """Scale (v, omega) down together so the outer wheel stays within full speed.

        At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors
        give while balancing: every v4 fall on flat ground came from such commands. The
        limit is the full command speed (0.3 m/s). The ESP32 should limit joystick commands
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

        Speed and yaw rate are drawn with the stage's band sampling, or both 0 with the
        stage's `stop_probability`; mid-episode, a change is a reversal with the stage's
        `reversal_probability`.
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
        stop = jax.random.uniform(keys[5]) < params["stop_probability"]
        random_command = jnp.where(stop, 0.0, random_command)
        command = jnp.where(reversal, reversal_command, random_command)
        duration = jax.random.uniform(
            keys[4], minval=COMMAND_RESAMPLE_S[0], maxval=COMMAND_RESAMPLE_S[1]
        )
        return command, duration, reversal

    # ------------------------------------------------------------------ physics

    def _episode_model(self, info: dict) -> mjx.Model:
        """MJX model with this episode's mass, CoM, friction and motor voltage."""
        model = self._mjx_model
        base = self._base_body_id
        voltage_scale = info["battery_voltage"] / NOMINAL_VOLTAGE
        return model.replace(
            # The motors' no-load speed and stall torque are proportional to the voltage.
            actuator_ctrlrange=self._nominal_ctrlrange * voltage_scale,
            actuator_forcerange=self._nominal_forcerange * voltage_scale,
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
                        _normalized(info["battery_voltage"] - NOMINAL_VOLTAGE, scale["voltage"]),
                        _normalized(info["push_force"] @ forward, scale["push"]),
                        _normalized(info["push_force"] @ lateral, scale["push"]),
                    ]
                ),
                _normalized(info["wheel_kick"], scale["wheel_kick"]),
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
        battery_voltage = jax.random.uniform(
            keys[17], minval=params["battery_voltage"][0], maxval=params["battery_voltage"][1]
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
            "battery_voltage": battery_voltage,
            "com_offset": com_offset,
            "slope": slope,
            "ground_normal": ground_normal,
            "push_force": jnp.zeros(2),
            "push_end_time": jnp.float32(0.0),
            "next_impulse_time": jax.random.uniform(
                keys[12], minval=params["push_interval_s"][0], maxval=params["push_interval_s"][1]
            ),
            "wheel_kick": jnp.zeros(2),
            "wheel_kick_end_time": jnp.float32(0.0),
            "next_wheel_kick_time": jax.random.uniform(
                keys[16],
                minval=params["wheel_kick_interval_s"][0],
                maxval=params["wheel_kick_interval_s"][1],
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

    def _pulse(
        self, info: dict, rng: jax.Array, value, keys: tuple[str, str, str], timing: str
    ) -> dict:
        """Advance one kind of random pulse, stored in `info` under `keys` (value, end time,
        next start time): `value` starts at the next start time if the stage has any
        (`<timing>_max` > 0), lasts `<timing>_duration_s`, and the one after it follows
        `<timing>_interval_s` later."""
        params = self.stage_params(info["stage"])
        value_key, end_key, next_key = keys
        interval = params[f"{timing}_interval_s"]
        time = info["episode_step"] * self.dt
        enabled = params[f"{timing}_max"] > 0
        start = enabled & (time >= info[next_key])
        end_time = jnp.where(start, time + params[f"{timing}_duration_s"], info[end_key])
        next_time = info[next_key] + jnp.where(
            start, jax.random.uniform(rng, minval=interval[0], maxval=interval[1]), 0.0
        )
        current = jnp.where(start, value, info[value_key])
        current = jnp.where((time >= end_time) | ~enabled, 0.0, current)
        return {**info, value_key: current, end_key: end_time, next_key: next_time}

    def _update_disturbance(self, info: dict, rng: jax.Array) -> dict:
        """Random pushes on the base, as a force in the world xy-plane, and torque kicks on
        the wheels (if the stage has any)."""
        params = self.stage_params(info["stage"])
        keys = jax.random.split(rng, 5)
        direction_key, push_key, push_timing_key, kick_key, kick_timing_key = keys
        direction = jax.random.uniform(direction_key, minval=-jnp.pi, maxval=jnp.pi)
        magnitude = self._sample_banded(push_key, params, "push", signed=False)
        info = self._pulse(
            info,
            push_timing_key,
            magnitude * jnp.stack([jnp.cos(direction), jnp.sin(direction)]),
            ("push_force", "push_end_time", "next_impulse_time"),
            "push",
        )
        return self._pulse(
            info,
            kick_timing_key,
            self._sample_banded(kick_key, params, "wheel_kick", shape=(2,)),
            ("wheel_kick", "wheel_kick_end_time", "next_wheel_kick_time"),
            "wheel_kick",
        )

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
        # A torque on a wheel joint acts on the wheel and, reversed, on the base, like the
        # motor: a motor or gearbox torque error.
        qfrc = jnp.zeros(self._mj_model.nv).at[self._wheel_qvel].set(info["wheel_kick"])
        data = state.data.replace(xfrc_applied=xfrc, qfrc_applied=qfrc)
        # The PWM duty sets the motor voltage: duty * no-load speed at this episode's battery.
        speed = applied_action * self._max_wheel_speed * info["battery_voltage"] / NOMINAL_VOLTAGE
        data = mjx_env.step(self._episode_model(info), data, speed, self.n_substeps)
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
        # This stage's weights and kernel widths (`Stage.rewards`).
        cfg = self.stage_params(info["stage"])
        reward_terms = {
            "alive": 1.0,
            # Penalize pitch motion, not the lean itself: the balance angle depends on the
            # CoM position and slope, and leaning is needed to accelerate.
            "pitch_rate": -(pitch_rate**2),
            "linear_velocity": _tracking_kernel(
                linear_error,
                cfg["linear_velocity_variance"],
                cfg["linear_velocity_wide_variance"],
                cfg["linear_velocity_wide_fraction"],
            ),
            "angular_velocity": jnp.exp(-(angular_error**2) / cfg["angular_velocity_variance"]),
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
            cfg["alive_weight"] * reward_terms["alive"]
            + cfg["pitch_rate_weight"] * reward_terms["pitch_rate"]
            + cfg["linear_velocity_weight"] * reward_terms["linear_velocity"]
            + cfg["angular_velocity_weight"] * reward_terms["angular_velocity"]
            + cfg["action_weight"] * reward_terms["action"]
            + cfg["smoothness_weight"] * reward_terms["smoothness"]
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
>>>>>>> gpu_training
