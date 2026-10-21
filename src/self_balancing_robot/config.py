"""Reward, environment, network and training settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    """The reward of every curriculum stage, unless a stage sets its own (`Stage.rewards`)."""

    alive_weight: float = 2.0
    pitch_rate_weight: float = 0.1

    linear_velocity_weight: float = 2.0
    angular_velocity_weight: float = 1.0
    action_weight: float = 0.05
    smoothness_weight: float = 1.0
    # use wide and narrow kernels for linear velocity, but only narrow for angular velocity
    linear_velocity_variance: float = 0.005
    linear_velocity_wide_variance: float = 0.25
    linear_velocity_wide_fraction: float = 0.5

    angular_velocity_variance: float = 0.3


@dataclass(frozen=True)
class EnvironmentConfig:
    model_file: str = "env_v0.xml"

    control_period_s: float = 0.01

    solver_iterations: int = 4
    solver_ls_iterations: int = 8
    max_episode_steps: int = 1_000
    history_length: int = 8
    initial_state_noise: float = 0.02
    initial_pitch_radians: float = 0.1
    upright_limit_radians: float = 0.6

    yaw_rate_limit_rad_s: float = 6.0
    camera_distance_m: float = 0.8
    camera_elevation_degrees: float = -20.0
    camera_azimuth_degrees: float = 45.0


@dataclass(frozen=True)
class PolicyConfig:
    actor_layers: tuple[int, ...] = (128, 64, 32)
    critic_layers: tuple[int, ...] = (128, 64, 32)
    init_noise_std: float = 0.6065


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 250_000_000
    num_envs: int = 4096
    unroll_length: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_cost: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 7
    run_dir: Path = Path("models/runs")
    evaluation_frequency: int = 5_000_000
    evaluation_episodes: int = 32
    evaluation_max_steps: int = 1_000
    evaluation_seed_offset: int = 1_000_000
    log_dir: Path = Path("logs")


DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_POLICY_CONFIG = PolicyConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
