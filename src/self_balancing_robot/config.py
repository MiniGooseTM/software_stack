from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class CurriculumStage(IntEnum):
    STABILIZATION = 1
    MICRO_MOBILITY = 2
    COMMAND_TRACKING = 3
    DOMAIN_RANDOMIZATION = 4


@dataclass(frozen=True)
class RewardConfig:
    alive_weight: float = 2.0
    pitch_weight: float = 0.5
    linear_velocity_weight: float = 1.5
    angular_velocity_weight: float = 1.0
    action_weight: float = 0.01
    smoothness_weight: float = 0.05
    pitch_variance: float = 0.05
    linear_velocity_variance: float = 0.25
    angular_velocity_variance: float = 0.5


@dataclass(frozen=True)
class CurriculumConfig:
    command_resample_seconds: tuple[float, float] = (3.0, 5.0)
    stage1_timeout_success_rate: float = 0.95
    stage2_linear_error_threshold: float = 0.01
    stage2_command_limits: tuple[float, float] = (0.05, 0.3)
    stage3_command_limits: tuple[float, float] = (0.3, 1.0)
    evaluation_window: int = 20


@dataclass(frozen=True)
class RandomizationConfig:
    mass_fraction: float = 0.10
    center_of_mass_offset_m: float = 0.01
    friction_fraction: float = 0.20
    slope_radians: float = 0.05
    max_latency_steps: int = 2
    observation_noise: float = 0.002
    impulse_interval_seconds: tuple[float, float] = (2.0, 4.0)
    impulse_force: float = 0.2


@dataclass(frozen=True)
class EnvironmentConfig:
    frame_skip: int = 3
    max_episode_steps: int = 1_000
    initial_state_noise: float = 0.02
    upright_limit_radians: float = 0.6
    max_position_m: float = 3.0
    camera_distance_m: float = 2.5
    camera_elevation_degrees: float = -15.0
    camera_azimuth_degrees: float = 90.0


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 100_000
    output_path: Path = Path("models/ppo_balance")
    seed: int = 7
    evaluation_frequency: int = 10_000
    evaluation_episodes: int = 5
    evaluation_max_steps: int = 1_000
    evaluation_seed_offset: int = 1_000_000
    parallel_environments: int = 8
    evaluation_model_path: Path = Path("models/ppo_balance.zip")
    training_plot_path: Path = Path("plots/training_errors.png")


DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_CURRICULUM_CONFIG = CurriculumConfig()
DEFAULT_RANDOMIZATION_CONFIG = RandomizationConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
