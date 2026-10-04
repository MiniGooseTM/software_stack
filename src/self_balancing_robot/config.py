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
    # Full command range (v m/s, omega rad/s). Up to v8 it was 0.4 m/s (~30% motor torque
    # left for balancing; 0.5 m/s left ~14% and v4 fell more). 0.3 m/s leaves headroom for
    # the speed bias a CoM shift causes: v8 with +1 cm drove ~0.15 m/s too fast, saturated
    # the motors at 0.4 m/s and fell. Commands are also scaled down so the outer wheel never
    # needs more than the full speed (see `wheel_feasible_command`).
    command_limits: tuple[float, float] = (0.3, 1.5)

    # Stages 2 and 3 have levels 1..N. Stage 2 level k allows commands up to k/N of
    # `command_limits`; stage 3 level k allows slopes up to k/N * slope_radians and pushes up
    # to k/N * impulse_force_n (commands always full range). A magnitude drawn at a level is
    # 0 with `level_zero_probability` (standing still, flat ground, no push), in the newest
    # band ((k-1)/N .. k/N of the maximum) with `level_frontier_probability`, otherwise
    # uniform below that band (no forgetting). Stage-3 commands: 0 or uniform over the range.
    level_frontier_probability: float = 0.4
    level_zero_probability: float = 0.1
    # Chance that a command change mid-episode (stages 2 and 3) is a full reversal: the
    # opposite direction at 75-100% of the current speed limit, no turning. With random
    # commands only 0.3-0.6% of changes were full reversals, and v9 fell on most joystick
    # flips (+0.3 -> -0.3 m/s): it braked so hard it pitched to ~30 deg.
    reversal_probability: float = 0.2
    reversal_min_fraction: float = 0.75

    # The curriculum is global: every environment trains at the same stage and level (with
    # thousands of GPU environments each one only finishes a few dozen episodes per run, too few
    # for a per-environment window). Episodes that finish at the current level are pooled in
    # windows of `evaluation_window` episodes; training moves up a level (or, at the last
    # level, to the next stage) once the minimum number of episodes was played at the level
    # and a window:
    #  - survived at least the success rate;
    #  - reached the speed ratio: speed along the command / commanded speed, counting
    #    commands of at least `speed_ratio_min_fraction` of the level's maximum (and at least
    #    `speed_ratio_min_command`) held at least
    #    `speed_ratio_settle_seconds` (the acceleration after each new command is ignored);
    #  - stood still on zero speed commands: mean |v| at most the step's still-speed limit
    #    (stricter on flat ground; on slopes v4 and v6 drifted at 0.07-0.10 m/s).
    # 3 levels: 0.1, 0.2, 0.3 m/s (v7/v8: 4 levels to 0.4 m/s). The first level stays at
    # 0.1 m/s: with 8 levels (from 0.05 m/s) v7 never passed level 1, as at 0.05 m/s the
    # speed signal is lost in balancing wobble and gearbox friction and the policy drifted
    # back to standing still (36% -> 4% of the command from 3M to 5M steps).
    command_levels: int = 3
    command_level_success_rate: float = 0.95
    command_min_episodes_per_level: int = 2_000
    command_speed_ratio: float = 0.90
    command_still_speed_limit: float = 0.02
    randomization_levels: int = 6
    # Slopes cost speed uphill (at 0.3 rad, 0.4 m/s took almost full duty), so the ratio is
    # a bit lower here; v6 (85% survival, no speed check) drifted back to ~70% of the command.
    randomization_level_success_rate: float = 0.90
    randomization_min_episodes_per_level: int = 5_000
    randomization_speed_ratio: float = 0.85
    randomization_still_speed_limit: float = 0.05
    # Episodes with a reversal must also survive on their own (about a third of the episodes
    # have one).
    reversal_success_rate: float = 0.85
    speed_ratio_min_fraction: float = 1.0 / 3.0
    speed_ratio_min_command: float = 0.05
    speed_ratio_settle_seconds: float = 1.0
    evaluation_window: int = 1_000


@dataclass(frozen=True)
class RandomizationConfig:
    mass_fraction: float = 0.10
    # Maximum CoM shift per axis, reached at the last stage-3 level like slope and push. The
    # nominal CoM is 4.9 cm above the axle; up to v7 the shift was 10% of that (4.9 mm) from
    # level 1, and v7 still drifted at 0.1 m/s standing still with 5 mm forward and fell with
    # 1 cm; v8 (1 cm) stalled at the levels with 5-7 mm. 8 mm needs ~9 deg of lean to balance.
    center_of_mass_offset_m: float = 0.008
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
    model_file: str = "env_v0.xml"
    # Policy period, as on the ESP32; the physics runs `control_period_s / timestep` substeps
    # per action (timestep from the XML).
    control_period_s: float = 0.01
    # Constraint solver limits of the MJX model. Batched, every environment waits for the
    # slowest one, so MuJoCo's defaults (100 / 50) cost ~2.6x the time of 4 / 8 at 4096
    # environments; with at most 10 contacts the Newton solver converges in a few iterations.
    solver_iterations: int = 4
    solver_ls_iterations: int = 8
    max_episode_steps: int = 1_000
    history_length: int = 8
    initial_state_noise: float = 0.02
    initial_pitch_radians: float = 0.1
    upright_limit_radians: float = 0.6
    # Spinning faster than twice the largest yaw command ends the episode like a fall. Without
    # it, PPO on the GPU sometimes learned to balance while spinning in place at ~8 rad/s
    # (early on, spinning kept a barely trained robot up for longer), and the narrow yaw-rate
    # kernel gives no gradient back from that far away.
    yaw_rate_limit_rad_s: float = 3.0
    camera_distance_m: float = 0.8
    camera_elevation_degrees: float = -20.0
    camera_azimuth_degrees: float = 45.0


@dataclass(frozen=True)
class PolicyConfig:
    # v1-v7 used (64, 32): 7.8k weights. (128, 64, 32) has 21.8k (87 KB as float32, ~22k
    # multiply-adds per step), still far below the ESP32's flash and the 10 ms control step.
    actor_layers: tuple[int, ...] = (128, 64, 32)
    critic_layers: tuple[int, ...] = (128, 64)
    # Initial action noise std (exp(-0.5), SB3's log_std_init of the earlier runs).
    init_noise_std: float = 0.6065


@dataclass(frozen=True)
class TrainingConfig:
    # Brax PPO on GPU. One PPO iteration collects num_envs * unroll_length steps and splits
    # them into num_minibatches minibatches (num_envs must be a multiple of num_minibatches).
    total_timesteps: int = 200_000_000
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
    # Every `evaluation_frequency` steps training pauses to save a checkpoint and run
    # `evaluation_episodes` deterministic episodes at the current level and at the final
    # difficulty (in parallel on the GPU) for the plots and TensorBoard.
    evaluation_frequency: int = 5_000_000
    evaluation_episodes: int = 32
    evaluation_max_steps: int = 1_000
    evaluation_seed_offset: int = 1_000_000
    video_frequency: int = 25_000_000
    log_dir: Path = Path("logs")


DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_CURRICULUM_CONFIG = CurriculumConfig()
DEFAULT_RANDOMIZATION_CONFIG = RandomizationConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_POLICY_CONFIG = PolicyConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
