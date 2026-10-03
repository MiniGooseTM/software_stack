from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class CurriculumStage(IntEnum):
    STABILIZATION = 1
    # Replaces the old micro-mobility (2) and full tracking (3) stages: two fixed command
    # ranges let v5/v6 pass stage 2 barely moving (~20% of the command) and stage 3 had to
    # unlearn it. The command range now grows in small levels, each checked for speed.
    COMMAND_RAMP = 2
    DOMAIN_RANDOMIZATION = 3


@dataclass(frozen=True)
class RewardConfig:
    alive_weight: float = 2.0
    pitch_rate_weight: float = 0.1
    # Above the yaw weight so speed tracking still pays on slopes (v5 at 1.5 drove at ~60%
    # of the commanded speed).
    linear_velocity_weight: float = 2.0
    angular_velocity_weight: float = 1.0
    action_weight: float = 0.01
    smoothness_weight: float = 0.05
    # Speed tracking blends a narrow kernel (precision near the command) with a wide one
    # (a gradient even far from it). With only the narrow kernel, v2 learned to crawl at
    # 0.05 m/s for 0.5 m/s commands because speeding up earned no reward until close.
    # At 0.02 the narrow kernel kept 96% of the reward at 0.04 m/s error, so small commands
    # were not worth following (v7 stalled at 0.05 m/s commands reaching ~20%) and the 90%
    # speed ratio was not worth reaching; at 0.005 it keeps 73%.
    linear_velocity_variance: float = 0.005
    linear_velocity_wide_variance: float = 0.25
    linear_velocity_wide_fraction: float = 0.5
    # Yaw rate uses the narrow kernel only: a wide one made v4 turn sloppily (0.24 rad/s
    # error vs 0.09) and turning never had the crawl problem.
    angular_velocity_variance: float = 0.3


@dataclass(frozen=True)
class CurriculumConfig:
    command_resample_seconds: tuple[float, float] = (3.0, 5.0)
    stage1_timeout_success_rate: float = 0.95
    # Full command range (v m/s, omega rad/s). 0.4 m/s leaves ~30% motor torque for
    # balancing (0.5 m/s left ~14% and v4 fell more). Commands are also scaled down so the
    # outer wheel never needs more than 0.4 m/s (see `wheel_feasible_command`).
    command_limits: tuple[float, float] = (0.4, 1.5)

    # Stages 2 and 3 have levels 1..N. Stage 2 level k allows commands up to k/N of
    # `command_limits`; stage 3 level k allows slopes up to k/N * slope_radians and pushes up
    # to k/N * impulse_force_n (commands always full range). A magnitude drawn at a level is
    # 0 with `level_zero_probability` (standing still, flat ground, no push), in the newest
    # band ((k-1)/N .. k/N of the maximum) with `level_frontier_probability`, otherwise
    # uniform below that band (no forgetting). Stage-3 commands: 0 or uniform over the range.
    level_frontier_probability: float = 0.4
    level_zero_probability: float = 0.1

    # An environment moves up a level (or, at the last level, to the next stage) once it has
    # played the minimum number of episodes at the level and, over its last
    # `evaluation_window` episodes:
    #  - survived at least the success rate;
    #  - reached the speed ratio: speed along the command / commanded speed, counting
    #    commands of at least `speed_ratio_min_fraction` of the level's maximum (and at least
    #    `speed_ratio_min_command`) held at least
    #    `speed_ratio_settle_seconds` (the acceleration after each new command is ignored);
    #  - stood still on zero speed commands: mean |v| at most the step's still-speed limit
    #    (stricter on flat ground; on slopes v4 and v6 drifted at 0.07-0.10 m/s).
    # 4 levels: 0.1, 0.2, 0.3, 0.4 m/s. With 8 (from 0.05 m/s) v7 never passed level 1: at
    # 0.05 m/s the speed signal is lost in balancing wobble and gearbox friction, and the
    # policy drifted back to standing still (36% -> 4% of the command from 3M to 5M steps).
    command_levels: int = 4
    command_level_success_rate: float = 0.95
    command_min_episodes_per_level: int = 40
    command_speed_ratio: float = 0.90
    command_still_speed_limit: float = 0.02
    randomization_levels: int = 6
    # Slopes cost speed uphill (at 0.3 rad, 0.4 m/s takes almost full duty), so the ratio is
    # a bit lower here; v6 (85% survival, no speed check) drifted back to ~70% of the command.
    randomization_level_success_rate: float = 0.90
    randomization_min_episodes_per_level: int = 100
    randomization_speed_ratio: float = 0.85
    randomization_still_speed_limit: float = 0.05
    speed_ratio_min_fraction: float = 1.0 / 3.0
    speed_ratio_min_command: float = 0.05
    speed_ratio_settle_seconds: float = 1.0
    evaluation_window: int = 20


@dataclass(frozen=True)
class RandomizationConfig:
    mass_fraction: float = 0.10
    # CoM shift per axis as a fraction of the nominal CoM height above the axle.
    center_of_mass_fraction: float = 0.10
    friction_fraction: float = 0.40
    # The base hits the ground at 0.5 rad of tilt relative to the surface, so slopes above
    # ~0.3 rad cannot be driven straight up or down (v2: 0% survival above 0.3 rad).
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
    actor_layers: tuple[int, ...] = (64, 32)
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
