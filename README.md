# Self-balancing robot software stack

This repository contains the first simulation and reinforcement-learning setup for a homemade two-wheel self-balancing robot. The current model is planar: PPO controls two wheel motors and observes chassis pose/velocity, with the goal of keeping the chassis pitch near zero and following velocity commands.

## University cluster
Open the university cluster https://gpu1.eecs.kth.se/user/{USER} and in the terminal run
```bash
ssh-keygen -t ed25519 -C "gpu-server"
cat ~/.ssh/id_ed25519.pub
```
add that as a PAT in your Github settings SSH and then run in the cluster terminal
```bash
ssh -T git@github.com
```
you will get a message saying you've successfully authenticated, then feel free to clone
```bash
git clone git@github.com:MiniGooseTM/software_stack.git
```
after running the following you shouls be also able to commit and push
```bash
git config --global user.name "YOUR_GITHUB_USERNAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Python 3.10–3.12 is supported. MuJoCo is installed from PyPI, so no separate MuJoCo download is required.

```bash
make install
make test
make train
```

The trained policy is saved to `models/ppo_balance.zip`; TensorBoard logs are written to `logs/`.
Training also evaluates mean absolute pitch error on held-out episodes and saves the train/test
error graph to `src/plots/training_errors.png`. Evaluation frequency and episode count can be
changed with `--eval-freq` and `--eval-episodes`.

## Docker
There's no need for docker in the University Cluster. However it is available in case the cluster isn't.
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
