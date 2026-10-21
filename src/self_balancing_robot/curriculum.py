"""
The training curriculum: an ordered list of stages, each with everything that defines it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np

from self_balancing_robot.config import DEFAULT_REWARD_CONFIG, RewardConfig

# (max linear velocity, max angular velocity)
FULL_COMMAND = (0.4, 2.5)
# Commands are redrawn every 3 to 5 s
COMMAND_RESAMPLE_S = (3.0, 5.0)
# probability of a command change being a full reversal
REVERSAL_MIN_FRACTION = 0.75

# number of finished episodes to check the success rate, speed ratio and still-speed limit
EVALUATION_WINDOW = 1_000

SPEED_RATIO_MIN_FRACTION = 1.0 / 3.0
SPEED_RATIO_MIN_COMMAND = 0.05
SPEED_RATIO_SETTLE_S = 1.0
# nominal battery voltage (V) for the ESP32 and the motors, used to scale the motor torque and speed
NOMINAL_VOLTAGE = 11.1


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: what the environment samples, and the test to leave it."""

    name: str

    # Largest |v| (m/s) and |omega| (rad/s) commanded.
    command_max: tuple[float, float] = (0.0, 0.0)
    # Chance that a command change mid-episode is a full reversal instead of a random command.
    reversal_probability: float = 0.0
    # Chance that a command that is not a reversal is a full stop (0, 0): the released joystick.
    stop_probability: float = 0.0
    # Terrain incline (rad), in a random direction.
    slope_max: float = 0.0
    # Pushes on the base (N, horizontal, random direction), lasting `push_duration_s`, one
    # every `push_interval_s`.
    push_max: float = 0.0
    push_interval_s: tuple[float, float] = (2.0, 4.0)
    push_duration_s: float = 0.1
    # Torque kicks on each wheel (N m), lasting `wheel_kick_duration_s`, one every
    # `wheel_kick_interval_s`: simulates bumps.
    wheel_kick_max: float = 0.0
    wheel_kick_interval_s: tuple[float, float] = (1.0, 3.0)
    wheel_kick_duration_s: float = 0.05
    # Centre-of-mass shift per axis (m).
    com_offset_max: float = 0.0
    # Base mass and floor friction: uniform within +- this fraction of nominal.
    mass_fraction: float = 0.0
    friction_fraction: float = 0.0
    # Battery voltage (V), uniform in (min, max)
    battery_voltage: tuple[float, float] = (NOMINAL_VOLTAGE, NOMINAL_VOLTAGE)
    # Motor, IMU and encoder delays: each 0 to this many control steps (10 ms).
    max_latency_steps: int = 0
    # Gaussian sensor noise (std) on IMU angles (rad), IMU rates (rad/s), wheel speeds (rad/s).
    imu_angle_noise: float = 0.0
    imu_rate_noise: float = 0.0
    encoder_noise: float = 0.0
    # How magnitudes are drawn (see the module docstring).
    zero_probability: float = 0.1
    frontier_probability: float = 0.4

    # --- The reward (weights and kernel widths) ---
    rewards: RewardConfig = DEFAULT_REWARD_CONFIG

    # --- The test to leave the stage, on a window of finished episodes (None = not checked) ---
    min_episodes: int = 0
    success_rate: float = 0.95
    # Speed along the command / commanded speed.
    speed_ratio: float | None = None
    # Mean |v| (m/s) on zero speed commands held at least SPEED_RATIO_SETTLE_S.
    still_speed_limit: float | None = None
    # Fraction of the episodes with a reversal that survive.
    reversal_success_rate: float | None = None


# --- The default curriculum ---

# Learn to balance
BALANCE = Stage(name="balance", success_rate=0.95)

# Introduce commands and sensor noise gradually.
IMU_ANGLE_NOISE = 0.01
IMU_RATE_NOISE = 0.05
ENCODER_NOISE = 0.3
COMMANDS_1 = replace(
    BALANCE,
    name="commands 0.1 m/s",
    command_max=(0.1, 0.625),
    reversal_probability=0.2,
    stop_probability=0.2,
    imu_angle_noise=IMU_ANGLE_NOISE / 4,
    imu_rate_noise=IMU_RATE_NOISE / 4,
    encoder_noise=ENCODER_NOISE / 4,
    min_episodes=2_000,
    success_rate=0.95,
    speed_ratio=0.90,
    still_speed_limit=0.02,
    reversal_success_rate=0.85,
)
COMMANDS_2 = replace(
    COMMANDS_1,
    name="commands 0.2 m/s",
    command_max=(0.2, 1.25),
    max_latency_steps=1,
    imu_angle_noise=IMU_ANGLE_NOISE * 2 / 4,
    imu_rate_noise=IMU_RATE_NOISE * 2 / 4,
    encoder_noise=ENCODER_NOISE * 2 / 4,
)
COMMANDS_3 = replace(
    COMMANDS_2,
    name="commands 0.3 m/s",
    command_max=(0.3, 1.875),
    imu_angle_noise=IMU_ANGLE_NOISE * 3 / 4,
    imu_rate_noise=IMU_RATE_NOISE * 3 / 4,
    encoder_noise=ENCODER_NOISE * 3 / 4,
)
COMMANDS_4 = replace(
    COMMANDS_3,
    name="commands 0.4 m/s",
    command_max=FULL_COMMAND,
    max_latency_steps=2,
    imu_angle_noise=IMU_ANGLE_NOISE,
    imu_rate_noise=IMU_RATE_NOISE,
    encoder_noise=ENCODER_NOISE,
)
# start with domain randomization
# --> pushes
# --> slopes
# --> wheel kicks
# --> COM offsets
# --> mass and friction randomization
PUSH_MAX = 0.5
COM_OFFSET_MAX = 0.008
WHEEL_KICK_MAX = 0.02
RANDOMIZATION_1 = replace(
    COMMANDS_4,
    name="randomization 1/6",
    slope_max=0.05,
    push_max=PUSH_MAX / 6,
    com_offset_max=COM_OFFSET_MAX / 6,
    wheel_kick_max=WHEEL_KICK_MAX / 6,
    mass_fraction=0.10,
    friction_fraction=0.40,
    battery_voltage=(10.5, 12.5),
    min_episodes=5_000,
    success_rate=0.90,
    speed_ratio=0.90,
    still_speed_limit=0.05,
)
RANDOMIZATION_2 = replace(
    RANDOMIZATION_1,
    name="randomization 2/6",
    slope_max=0.10,
    push_max=PUSH_MAX * 2 / 6,
    com_offset_max=COM_OFFSET_MAX * 2 / 6,
    wheel_kick_max=WHEEL_KICK_MAX * 2 / 6,
)
RANDOMIZATION_3 = replace(
    RANDOMIZATION_2,
    name="randomization 3/6",
    slope_max=0.15,
    push_max=PUSH_MAX * 3 / 6,
    com_offset_max=COM_OFFSET_MAX * 3 / 6,
    wheel_kick_max=WHEEL_KICK_MAX * 3 / 6,
)
RANDOMIZATION_4 = replace(
    RANDOMIZATION_3,
    name="randomization 4/6",
    slope_max=0.20,
    push_max=PUSH_MAX * 4 / 6,
    com_offset_max=COM_OFFSET_MAX * 4 / 6,
    wheel_kick_max=WHEEL_KICK_MAX * 4 / 6,
)
# increase the smoothness weight to get smoother policy without jittering
RANDOMIZATION_5 = replace(
    RANDOMIZATION_4,
    name="randomization 5/6",
    slope_max=0.25,
    push_max=PUSH_MAX * 5 / 6,
    com_offset_max=COM_OFFSET_MAX * 5 / 6,
    wheel_kick_max=WHEEL_KICK_MAX * 5 / 6,
    rewards=replace(DEFAULT_REWARD_CONFIG, smoothness_weight=5.0),
)
RANDOMIZATION_6 = replace(
    RANDOMIZATION_5,
    name="randomization 6/6",
    slope_max=0.30,
    push_max=PUSH_MAX,
    com_offset_max=COM_OFFSET_MAX,
    wheel_kick_max=WHEEL_KICK_MAX,
    rewards=replace(DEFAULT_REWARD_CONFIG, smoothness_weight=5.0),
)

CURRICULUM: tuple[Stage, ...] = (
    BALANCE,
    COMMANDS_1,
    COMMANDS_2,
    COMMANDS_3,
    COMMANDS_4,
    RANDOMIZATION_1,
    RANDOMIZATION_2,
    RANDOMIZATION_3,
    RANDOMIZATION_4,
    RANDOMIZATION_5,
    RANDOMIZATION_6,
)


# --- Helpers ---

# Quantities drawn with the band sampling, and the stage field holding their maximum.
BANDED = {
    "command": "command_max",
    "slope": "slope_max",
    "push": "push_max",
    "com_offset": "com_offset_max",
    "wheel_kick": "wheel_kick_max",
}


def validate(curriculum: tuple[Stage, ...]) -> None:
    """Raise ValueError if the curriculum cannot be run."""
    if not curriculum:
        raise ValueError("the curriculum has no stages")
    names = [stage.name for stage in curriculum]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names must be unique: {names}")
    for stage in curriculum:
        for name, value in _settings(stage):
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{stage.name}: {name} is negative")
            if isinstance(value, tuple) and min(value) < 0:
                raise ValueError(f"{stage.name}: {name} is negative")
        if stage.zero_probability + stage.frontier_probability > 1:
            raise ValueError(f"{stage.name}: zero + frontier probability is above 1")
        probabilities = (stage.reversal_probability, stage.stop_probability, stage.success_rate)
        if any(p > 1 for p in probabilities):
            raise ValueError(f"{stage.name}: a probability or rate is above 1")
        if any(c > full for c, full in zip(stage.command_max, FULL_COMMAND)):
            raise ValueError(f"{stage.name}: command_max is above FULL_COMMAND {FULL_COMMAND}")
        for field in ("push_interval_s", "wheel_kick_interval_s", "battery_voltage"):
            low, high = getattr(stage, field)
            if low > high:
                raise ValueError(f"{stage.name}: {field} must be (min, max)")


def stage_index(curriculum: tuple[Stage, ...], stage: str | int) -> int:
    """Index of a stage given by name or index (negative indices count from the end)."""
    names = [s.name for s in curriculum]
    if isinstance(stage, str) and stage not in names:
        try:
            stage = int(stage)
        except ValueError:
            raise ValueError(f"no stage {stage!r}; stages: {names}") from None
    if isinstance(stage, str):
        return names.index(stage)
    if not -len(curriculum) <= stage < len(curriculum):
        raise ValueError(f"stage {stage} out of range (0 to {len(curriculum) - 1})")
    return stage % len(curriculum)


def _settings(stage: Stage) -> list[tuple[str, object]]:
    """(name, value) of every setting of a stage, the reward's fields under their own names."""
    settings = []
    for field in fields(Stage):
        value = getattr(stage, field.name)
        if is_dataclass(value):
            settings += [(f.name, getattr(value, f.name)) for f in fields(value)]
        elif field.name != "name":
            settings.append((field.name, value))
    return settings


def stage_table(curriculum: tuple[Stage, ...]) -> dict[str, np.ndarray]:
    """The curriculum as arrays with one row per stage, for lookups inside JAX code.

    The reward's fields get a column each (`smoothness_weight`, ...). Adds the band floors
    of the banded quantities (`<quantity>_floor`: the previous stage's maximum if this stage
    raised it, else 0) and turns unchecked criteria (None) into values that always pass.
    """
    validate(curriculum)
    table: dict[str, list] = {}
    for stage in curriculum:
        for name, value in _settings(stage):
            if value is None:
                value = {"still_speed_limit": np.inf}.get(name, 0.0)
            table.setdefault(name, []).append(value)
    arrays = {
        name: np.asarray(
            values, dtype=np.int32 if name in ("max_latency_steps", "min_episodes") else np.float32
        )
        for name, values in table.items()
    }
    for quantity, field in BANDED.items():
        maxima = arrays[field]
        previous = np.concatenate([np.zeros_like(maxima[:1]), maxima[:-1]])
        arrays[f"{quantity}_floor"] = np.where(maxima > previous, previous, 0.0).astype(np.float32)
    return arrays
