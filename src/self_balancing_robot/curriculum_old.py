"""The training curriculum: an ordered list of stages, each with everything that defines it.

A `Stage` says what the environment samples (command range, slopes, pushes, randomized
physics, sensor latency and noise) and the test to leave it. Training starts at the first
stage of `CURRICULUM` and moves to the next one once a window of `EVALUATION_WINDOW`
episodes, pooled over all parallel environments, passes the stage's test (see
`envs/wrappers.py`). The last stage is never left.

To add or change a stage, edit `CURRICULUM` below; nothing else needs to change. Each stage is
written as the previous one plus what changes (`replace(previous, ...)`).

Magnitudes (command speed and yaw rate, slope, push, CoM shift) are drawn per stage as: 0 with
`zero_probability` (stand still, flat ground, no push), in the band the stage added (from the
previous stage's maximum up to this stage's) with `frontier_probability`, otherwise uniform
below that band, so easier values keep being practised. If a stage does not raise a
quantity, it is drawn as 0 or uniform up to its maximum. Signs and directions are random.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

# Full command range (v m/s, omega rad/s): observations are scaled by it, and every command
# is scaled down so the outer wheel never needs more than the full speed (see
# `SelfBalancingRobotEnv.wheel_feasible_command`). Up to v8 it was 0.4 m/s (~30% motor torque
# left for balancing; 0.5 m/s left ~14% and v4 fell more). 0.3 m/s leaves headroom for the
# speed bias a CoM shift causes: v8 with +1 cm drove ~0.15 m/s too fast, saturated the
# motors at 0.4 m/s and fell.
FULL_COMMAND = (0.3, 1.5)
# Commands are redrawn every 3 to 5 s. A reversal (see `Stage.reversal_probability`) goes to
# the opposite direction at this fraction to 100% of the stage's speed limit, no turning.
COMMAND_RESAMPLE_S = (3.0, 5.0)
REVERSAL_MIN_FRACTION = 0.75

# The curriculum is global: every environment trains at the same stage (with thousands of GPU
# environments each one only finishes a few dozen episodes per run, too few for a
# per-environment window). Episodes that finish at the current stage are pooled in windows of
# this many episodes, and each full window is checked against the stage's test.
EVALUATION_WINDOW = 1_000
# Speed ratio = speed along the command / commanded speed, counting commands of at least
# this fraction of the stage's speed limit (and at least SPEED_RATIO_MIN_COMMAND m/s) held
# at least SPEED_RATIO_SETTLE_S (the acceleration after each new command is ignored).
SPEED_RATIO_MIN_FRACTION = 1.0 / 3.0
SPEED_RATIO_MIN_COMMAND = 0.05
SPEED_RATIO_SETTLE_S = 1.0


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: what the environment samples, and the test to leave it."""

    name: str

    # --- What the environment samples (maxima; 0 = off) ---
    # Largest |v| (m/s) and |omega| (rad/s) commanded.
    command_max: tuple[float, float] = (0.0, 0.0)
    # Chance that a command change mid-episode is a full reversal instead of a random command.
    reversal_probability: float = 0.0
    # Terrain incline (rad), in a random direction.
    slope_max: float = 0.0
    # Pushes on the base (N, horizontal, random direction), lasting `push_duration_s`, one
    # every `push_interval_s`.
    push_max: float = 0.0
    push_interval_s: tuple[float, float] = (2.0, 4.0)
    push_duration_s: float = 0.1
    # Centre-of-mass shift per axis (m).
    com_offset_max: float = 0.0
    # Base mass and floor friction: uniform within +- this fraction of nominal.
    mass_fraction: float = 0.0
    friction_fraction: float = 0.0
    # Motor, IMU and encoder delays: each 0 to this many control steps (10 ms).
    max_latency_steps: int = 0
    # Gaussian sensor noise (std) on IMU angles (rad), IMU rates (rad/s), wheel speeds (rad/s).
    imu_angle_noise: float = 0.0
    imu_rate_noise: float = 0.0
    encoder_noise: float = 0.0
    # How magnitudes are drawn (see the module docstring).
    zero_probability: float = 0.1
    frontier_probability: float = 0.4

    # --- The test to leave the stage, on a window of finished episodes (None = not checked) ---
    # Episodes played at the stage before any window can pass.
    min_episodes: int = 0
    # Fraction of episodes reaching the timeout without falling.
    success_rate: float = 0.95
    # Speed along the command / commanded speed (see SPEED_RATIO_*).
    speed_ratio: float | None = None
    # Mean |v| (m/s) on zero speed commands held at least SPEED_RATIO_SETTLE_S.
    still_speed_limit: float | None = None
    # Fraction of the episodes with a reversal that survive.
    reversal_success_rate: float | None = None


# --- The default curriculum ---

# Learn to balance: zero command, flat ground, nominal physics, perfect sensors.
BALANCE = Stage(name="balance", success_rate=0.95)

# Learn to follow commands, the range growing in steps of 0.1 m/s (and 0.5 rad/s). Two fixed
# ranges (0.05, then 0.4 m/s) let v5/v6 pass the first barely moving (~20% of the command)
# and the second had to unlearn it. The first step stays at 0.1 m/s: starting from 0.05 m/s,
# v7 never passed it, as at 0.05 m/s the speed signal is lost in balancing wobble and gearbox
# friction (36% -> 4% of the command from 3M to 5M steps). Reversals: with random commands
# only 0.3-0.6% of changes were full reversals, and v9 fell on most joystick flips
# (+0.3 -> -0.3 m/s), braking so hard it pitched to ~30 deg. The speed ratio replaced an
# absolute error threshold that let v5 advance driving at ~60% of the command.
COMMANDS_1 = replace(
    BALANCE,
    name="commands 0.1 m/s",
    command_max=(0.1, 0.5),
    reversal_probability=0.2,
    min_episodes=2_000,
    success_rate=0.95,
    speed_ratio=0.90,
    still_speed_limit=0.02,
    reversal_success_rate=0.85,
)
COMMANDS_2 = replace(COMMANDS_1, name="commands 0.2 m/s", command_max=(0.2, 1.0))
COMMANDS_3 = replace(COMMANDS_2, name="commands 0.3 m/s", command_max=FULL_COMMAND)

# Full commands on an imperfect world. Mass, friction, latency and noise are on from the first
# step; slope, pushes and the CoM shift grow in six equal steps up to 0.3 rad, 1 N and 8 mm.
# - Slopes: the base hits the ground at 0.5 rad of tilt relative to the surface, so above
#   ~0.3 rad slopes cannot be driven straight up or down (v2: 0% survival above 0.3 rad).
# - CoM: the nominal CoM is 4.9 cm above the axle. Up to v7 the shift was 4.9 mm from the
#   start, and v7 still drifted at 0.1 m/s standing still with 5 mm forward and fell with
#   1 cm; v8 (1 cm) stalled with 5-7 mm. 8 mm needs ~9 deg of lean to balance.
# - Test: slopes cost speed uphill (at 0.3 rad, 0.4 m/s took almost full duty), so the speed
#   ratio is a bit lower; v6 (85% survival, no speed check) drifted back to ~70% of the
#   command. Standing still on slopes: v4 and v6 drifted at 0.07-0.10 m/s.
RANDOMIZATION_1 = replace(
    COMMANDS_3,
    name="randomization 1/6",
    slope_max=0.05,
    push_max=1.0 / 6,
    com_offset_max=0.008 / 6,
    mass_fraction=0.10,
    friction_fraction=0.40,
    max_latency_steps=2,
    imu_angle_noise=0.01,
    imu_rate_noise=0.05,
    encoder_noise=0.3,
    min_episodes=5_000,
    success_rate=0.90,
    speed_ratio=0.85,
    still_speed_limit=0.05,
)
RANDOMIZATION_2 = replace(
    RANDOMIZATION_1,
    name="randomization 2/6",
    slope_max=0.10,
    push_max=2.0 / 6,
    com_offset_max=0.008 * 2 / 6,
)
RANDOMIZATION_3 = replace(
    RANDOMIZATION_2,
    name="randomization 3/6",
    slope_max=0.15,
    push_max=3.0 / 6,
    com_offset_max=0.008 * 3 / 6,
)
RANDOMIZATION_4 = replace(
    RANDOMIZATION_3,
    name="randomization 4/6",
    slope_max=0.20,
    push_max=4.0 / 6,
    com_offset_max=0.008 * 4 / 6,
)
RANDOMIZATION_5 = replace(
    RANDOMIZATION_4,
    name="randomization 5/6",
    slope_max=0.25,
    push_max=5.0 / 6,
    com_offset_max=0.008 * 5 / 6,
)
RANDOMIZATION_6 = replace(
    RANDOMIZATION_5,
    name="randomization 6/6",
    slope_max=0.30,
    push_max=1.0,
    com_offset_max=0.008,
)

CURRICULUM: tuple[Stage, ...] = (
    BALANCE,
    COMMANDS_1,
    COMMANDS_2,
    COMMANDS_3,
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
}


def validate(curriculum: tuple[Stage, ...]) -> None:
    """Raise ValueError if the curriculum cannot be run."""
    if not curriculum:
        raise ValueError("the curriculum has no stages")
    names = [stage.name for stage in curriculum]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names must be unique: {names}")
    for stage in curriculum:
        for field in fields(stage):
            value = getattr(stage, field.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{stage.name}: {field.name} is negative")
            if isinstance(value, tuple) and min(value) < 0:
                raise ValueError(f"{stage.name}: {field.name} is negative")
        if stage.zero_probability + stage.frontier_probability > 1:
            raise ValueError(f"{stage.name}: zero + frontier probability is above 1")
        probabilities = (stage.reversal_probability, stage.success_rate)
        if any(p > 1 for p in probabilities):
            raise ValueError(f"{stage.name}: a probability or rate is above 1")
        if any(c > full for c, full in zip(stage.command_max, FULL_COMMAND)):
            raise ValueError(f"{stage.name}: command_max is above FULL_COMMAND {FULL_COMMAND}")
        if stage.push_interval_s[0] > stage.push_interval_s[1]:
            raise ValueError(f"{stage.name}: push_interval_s must be (min, max)")


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


def stage_table(curriculum: tuple[Stage, ...]) -> dict[str, np.ndarray]:
    """The curriculum as arrays with one row per stage, for lookups inside JAX code.

    Adds the band floors of the banded quantities (`<quantity>_floor`: the previous stage's
    maximum if this stage raised it, else 0) and turns unchecked criteria (None) into
    values that always pass.
    """
    validate(curriculum)
    table: dict[str, list] = {}
    for stage in curriculum:
        for field in fields(Stage):
            if field.name == "name":
                continue
            value = getattr(stage, field.name)
            if value is None:
                value = {"still_speed_limit": np.inf}.get(field.name, 0.0)
            table.setdefault(field.name, []).append(value)
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
