from dataclasses import dataclass


@dataclass(frozen=True)
class MiniGooseEnvConfig:
    """
    Parameter config file for the MiniGoose self-balancing robot.

    """

    # Termination / episode
    max_tilt: float = 0.6  # rad from upright before the episode ends
    max_episode_steps: int = 2000  # truncation limit in control steps
    init_pitch_range: float = 0.1  # starting pitch is uniform in [-init_pitch_range, init_pitch_range] rad

    # Reward weights
    alive_bonus: float = 1.0
    w_tilt: float = 2.0
    w_tilt_rate: float = 0.05
    w_velocity: float = 0.1
    w_distance: float = 0.5
    w_action: float = 0.01
    w_action_rate: float = 0.01

    # complementary filter for pitch estimation
    alpha: float = 0.98  # trust in gyro integration vs accelerometer angle

    # Sensor noise (Gaussian std, in sensor units)
    accel_noise_std: float = 0.05  # m/s^2
    gyro_noise_std: float = 0.01  # rad/s
    wheel_pos_noise_std: float = 0.0  # rad
    wheel_vel_noise_std: float = 0.05  # rad/s

    # Actuator noise
    action_noise_std: float = 0.0  # in normalized action units [-1, 1]
    action_delay_steps: int = 0  # control steps between policy output and motor command
