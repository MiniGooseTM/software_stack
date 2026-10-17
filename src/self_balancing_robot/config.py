"""Configuration for the environment, reward, curriculum, policy and training runs.

The defaults are the values the shipped policy was trained with; changing one usually
requires retraining. See ARCHITECTURE.md and REWARDS.md for the full rationale.
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class CurriculumStage(IntEnum):
    STABILIZATION = 1
    # The command range ramps in small levels, each checked for speed. Two fixed command
    # ranges instead let the policy pass the second stage barely moving.
    COMMAND_RAMP = 2
    DOMAIN_RANDOMIZATION = 3


@dataclass(frozen=True)
class RewardConfig:
    alive_weight: float = 2.0
    pitch_rate_weight: float = 0.1
    # Above the yaw weight so speed tracking still pays on slopes.
    linear_velocity_weight: float = 2.0
    angular_velocity_weight: float = 1.0
    action_weight: float = 0.01
    smoothness_weight: float = 0.05
    # Speed tracking blends a narrow kernel (precision near the command) with a wide one
    # (a gradient far from it). With only the narrow kernel the policy crawls towards the
    # command because speeding up earns no reward until it is close.
    linear_velocity_variance: float = 0.005
    linear_velocity_wide_variance: float = 0.25
    linear_velocity_wide_fraction: float = 0.5
    # Yaw rate uses the narrow kernel only: a wide one makes turning sloppy, and turning
    # never had the crawl problem.
    angular_velocity_variance: float = 0.3


@dataclass(frozen=True)
class CurriculumConfig:
    command_resample_seconds: tuple[float, float] = (3.0, 5.0)
    stage1_timeout_success_rate: float = 0.95
    # Full command range (v m/s, omega rad/s). 0.3 m/s leaves torque headroom for balancing
    # and for the speed bias a CoM shift causes. Commands are also scaled down together so
    # the outer wheel never needs more than this (see `wheel_feasible_command`).
    command_limits: tuple[float, float] = (0.3, 1.5)

    # Stages 2 and 3 have N levels; level k allows k/N of the stage's maxima (full commands
    # in stage 3). A magnitude is 0 with `level_zero_probability` (standing still, flat
    # ground, no push), in the newest band ((k-1)/N .. k/N) with `level_frontier_probability`,
    # otherwise uniform below that band so earlier levels are not forgotten.
    level_frontier_probability: float = 0.4
    level_zero_probability: float = 0.1
    # Chance that a mid-episode command change is a full reversal: the opposite direction at
    # 75-100% of the speed limit, no turning. Random commands almost never reverse, and the
    # policy fell on joystick flips until they were trained.
    reversal_probability: float = 0.2
    reversal_min_fraction: float = 0.75

    # A level is passed after its minimum number of episodes if, over the last
    # `evaluation_window` episodes, the policy survived, tracked the commanded speed (commands
    # held at least `speed_ratio_settle_seconds`, so acceleration is ignored), stood still on
    # zero commands and survived its reversals. Stage 3 allows a lower speed ratio because
    # slopes cost speed. The first command level stays at 0.1 m/s: below that the speed signal
    # is lost in balancing wobble.
    command_levels: int = 3
    command_level_success_rate: float = 0.95
    command_min_episodes_per_level: int = 40
    command_speed_ratio: float = 0.90
    command_still_speed_limit: float = 0.02
    randomization_levels: int = 6
    randomization_level_success_rate: float = 0.90
    randomization_min_episodes_per_level: int = 100
    randomization_speed_ratio: float = 0.85
    randomization_still_speed_limit: float = 0.05
    # Episodes with a reversal must also survive on their own (~7 of a 20-episode window
    # have one, so one fall among them is allowed).
    reversal_success_rate: float = 0.85
    speed_ratio_min_fraction: float = 1.0 / 3.0
    speed_ratio_min_command: float = 0.05
    speed_ratio_settle_seconds: float = 1.0
    evaluation_window: int = 20


@dataclass(frozen=True)
class RandomizationConfig:
    mass_fraction: float = 0.10
    # Maximum CoM shift per axis, reached at the last stage-3 level like slope and push.
    # 8 mm needs about 9 degrees of lean to balance, which the policy must learn.
    center_of_mass_offset_m: float = 0.008
    friction_fraction: float = 0.40
    # The base hits the ground at 0.5 rad of tilt relative to the surface, so steeper slopes
    # cannot be driven straight up or down.
    slope_radians: float = 0.3
    max_motor_latency_steps: int = 2
    max_imu_latency_steps: int = 2
    max_encoder_latency_steps: int = 2
    imu_angle_noise_rad: float = 0.01
    imu_rate_noise_rad_s: float = 0.05
    encoder_velocity_noise_rad_s: float = 0.3
    impulse_interval_seconds: tuple[float, float] = (2.0, 4.0)
    impulse_force_n: float = 1.0
    impulse_duration_seconds: float = 0.1


@dataclass(frozen=True)
class EnvironmentConfig:
    model_file: str = "XML-test.xml"
    frame_skip: int = 2
    max_episode_steps: int = 1_000
    history_length: int = 8
    initial_state_noise: float = 0.02
    initial_pitch_radians: float = 0.1
    upright_limit_radians: float = 0.6
    camera_distance_m: float = 0.8
    camera_elevation_degrees: float = -20.0
    camera_azimuth_degrees: float = 45.0


@dataclass(frozen=True)
class PolicyConfig:
    # The actor's 21.8k parameters (87 KB as float32, ~22k multiply-adds per step) fit the
    # ESP32's flash and the 10 ms control step.
    actor_layers: tuple[int, ...] = (128, 64, 32)
    critic_layers: tuple[int, ...] = (128, 64)
    log_std_init: float = -0.5


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 20_000_000
    output_path: Path = Path("models/ppo_balance")
    checkpoint_path: Path = Path("models/checkpoints")
    checkpoint_frequency: int = 1_000_000
    seed: int = 7
    evaluation_frequency: int = 200_000
    evaluation_episodes: int = 5
    evaluation_max_steps: int = 1_000
    evaluation_seed_offset: int = 1_000_000
    parallel_environments: int = 8
    n_steps: int = 1024
    batch_size: int = 2048
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    evaluation_model_path: Path = Path("models/ppo_balance.zip")
    plot_dir: Path = Path("plots")
    video_path: Path = Path("videos")
    video_frequency: int = 1_000_000


DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_CURRICULUM_CONFIG = CurriculumConfig()
DEFAULT_RANDOMIZATION_CONFIG = RandomizationConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_POLICY_CONFIG = PolicyConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
