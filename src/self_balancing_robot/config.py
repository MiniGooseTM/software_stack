"""Reward, environment, network and training settings. The curriculum (what each training
stage samples and the test to leave it) is in `curriculum.py`."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    """The reward of every curriculum stage, unless a stage sets its own (`Stage.rewards`)."""

    alive_weight: float = 2.0
    pitch_rate_weight: float = 0.1
    # Above the yaw weight so speed tracking still pays on slopes (v5 at 1.5 drove at ~60%
    # of the commanded speed).
    linear_velocity_weight: float = 2.0
    angular_velocity_weight: float = 1.0
    action_weight: float = 0.05
    smoothness_weight: float = 1.0
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
    yaw_rate_limit_rad_s: float = 6.0
    camera_distance_m: float = 0.8
    camera_elevation_degrees: float = -20.0
    camera_azimuth_degrees: float = 45.0


@dataclass(frozen=True)
class PolicyConfig:
    # v1-v7 used (64, 32): 7.8k weights. (128, 64, 32) has 21.8k (87 KB as float32, ~22k
    # multiply-adds per step), still far below the ESP32's flash and the 10 ms control step.
    actor_layers: tuple[int, ...] = (128, 64, 32)
    critic_layers: tuple[int, ...] = (128, 64, 32)
    # Initial action noise std (exp(-0.5), SB3's log_std_init of the earlier runs).
    init_noise_std: float = 0.6065


@dataclass(frozen=True)
class TrainingConfig:
    # Brax PPO on GPU. One PPO iteration collects num_envs * unroll_length steps and splits
    # them into num_minibatches minibatches (num_envs must be a multiple of num_minibatches).
    total_timesteps: int = 500_000_000
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
    # `evaluation_episodes` deterministic episodes at the current curriculum stage and at the
    # last stage (in parallel on the GPU) for the plots and TensorBoard.
    evaluation_frequency: int = 5_000_000
    evaluation_episodes: int = 32
    evaluation_max_steps: int = 1_000
    evaluation_seed_offset: int = 1_000_000
    log_dir: Path = Path("logs")


DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_POLICY_CONFIG = PolicyConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
