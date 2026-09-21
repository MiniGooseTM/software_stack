# Self-balancing robot architecture

## Current vertical slice

`self_balancing_robot.xml` defines a planar chassis with a slide joint, pitch hinge, and two independently actuated wheels. `SelfBalancingRobotEnv` wraps the model as a Gymnasium MuJoCo environment. PPO uses eight parallel environments and saves policies under `models/`.

The observation is `[x, pitch, left_wheel_angle, right_wheel_angle, x_velocity, pitch_rate, left_wheel_rate, right_wheel_rate]`. The reward favors upright posture and penalizes pitch rate, drift, and control effort. An episode terminates when the robot falls beyond 0.6 radians or travels more than 3 m.

## Reproducible development paths

- Native development: `.venv` created by `make install`.
- Cross-platform CPU container: `Dockerfile`, suitable for macOS Docker Desktop and Ubuntu.
- Linux NVIDIA container: `Dockerfile.gpu`, for Ubuntu hosts configured with the NVIDIA Container Toolkit.

## Planned extensions

1. Replace the placeholder geometry and inertial parameters with the measured URDF robot.
2. Add IMU/encoder-style observations and actuator/sensor noise.
3. Add domain randomization, curriculum, and hardware-in-the-loop evaluation.