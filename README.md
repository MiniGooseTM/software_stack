# Self-balancing robot software stack

This repository contains the first simulation and reinforcement-learning setup for a homemade two-wheel self-balancing robot. The current model is planar: PPO controls two wheel motors and observes chassis pose/velocity, with the goal of keeping the chassis pitch near zero.

## Quick start with a Python virtual environment

Python 3.10–3.12 is supported. MuJoCo is installed from PyPI, so no separate MuJoCo download is required.

```bash
make install
make test
make train
```

The trained policy is saved to `models/ppo_balance.zip`; TensorBoard logs are written to `logs/`.

## Docker

The default image is CPU-only and works on macOS Docker Desktop and Ubuntu:

```bash
docker build -t self-balancing-robot .
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/logs:/app/logs" \
  self-balancing-robot --timesteps 100000
```

On an Ubuntu host with an NVIDIA runtime, use `Dockerfile.gpu` and add `--gpus all` to the run command. Training is headless; `MUJOCO_GL=egl` is set in both images.

## Project layout

- `src/self_balancing_robot/assets/self_balancing_robot.xml`: initial MuJoCo model.
- `src/self_balancing_robot/envs/balance_env.py`: Gymnasium environment and reward.
- `src/self_balancing_robot/train.py`: vectorized PPO training entry point.
- `src/self_balancing_robot/evaluate.py`: visual evaluation entry point.
- `tests/`: environment smoke tests.

The XML is intentionally a starting point. Before transferring a policy to hardware, replace dimensions, masses, friction, actuator limits, and sensor observations with measured robot values and add domain randomization.


## To access in University Cluster
```bash
ssh-keygen -t ed25519 -C "gpu-server"
cat ~/.ssh/id_ed25519.pub
```
add that as a PAT and then run
```bash
ssh -T git@github.com
```
you will get a message saying you've successfully authenticated, then feel free to clone
```bash
git clone git@github.com:MiniGooseTM/software_stack.git
```
