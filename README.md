# Self-balancing robot software stack

This repository contains the simulation and reinforcement-learning setup for a homemade two-wheel self-balancing robot. Training follows Option B in `ARCHITECTURE.md` (asymmetric actor-critic PPO) with the unified reward and curriculum in `REWARDS.md`. The robot (`assets/goose.xml`, on the ground of `assets/env_v0.xml`) is simulated in full 3D with MuJoCo MJX, thousands of copies in parallel on a GPU, and trained with Brax PPO (JAX). The actor is the small ReLU MLP meant for the ESP32: it outputs normalized PWM for the two wheel motors and sees only an 8-frame history of IMU, encoder, command, and previous-action readings. The critic additionally sees privileged simulation state (true velocities, CoM offset, mass, friction, pushes, latencies, slope) and is discarded after training.

## Setup

Python 3.11–3.13. MuJoCo, JAX and the CUDA libraries all come from PyPI; only an NVIDIA driver is needed for the GPU.

```bash
make install-cuda   # NVIDIA GPU (training); `make install` for CPU only
make test           # runs on the CPU
make benchmark      # environment steps per second on this machine
make train
```

On an RTX 4050 laptop GPU training runs ~165k environment steps per second (4,096 environments), so the default 200M steps take about 25 minutes. The first two evaluation intervals are slower: JAX compiles the training step (about a minute each).

## Training

Each run writes to its own directory, `models/runs/<date-time>/` (or `--run-dir`):

- `params.pkl`: the latest weights (actor, critic), saved every `--eval-freq` steps (5M);
- `curriculum.json`: curriculum stage (index and name) and the total number of steps;
- `config.json`: all settings of the run, including the curriculum;
- `plots/training_metrics.png`: reward, survival, velocity and yaw-rate tracking error and mean |pitch| of deterministic episodes at the current curriculum stage and at the last stage, and the curriculum stage over time; `plots/eval_episode_latest.png`: command vs response of one episode at the current stage, updated at every checkpoint;
- `plots/stages/` and `videos/`: for every stage passed, a plot (as above) and an MP4 of one episode at that stage, named after it (e.g. `stage_02_commands_0.2_m-s.png` / `.mp4`), plus one of the stage training ended at (`..._end`). The weights only reach the host at checkpoints, so these show the policy at the first checkpoint after the stage was passed (at most `--eval-freq` steps later). Videos are rendered in a separate process, trying the OpenGL backends GLFW (with a display), EGL and OSMesa in turn, so a machine without working OpenGL only loses the videos: the episode states stay in `videos/*.npz`; render them elsewhere with `python -m self_balancing_robot.render models/runs/<run>/videos/*.npz`.

The same numbers, the PPO losses and the curriculum go to TensorBoard: `.venv/bin/tensorboard --logdir logs`. Curriculum stage changes are also printed as they happen.

Continue a run with `make train TRAIN_ARGS="--resume models/runs/<run>"`; it picks up the weights and the curriculum stage (matched by name, so stages added earlier in the list do not shift it; `--start-stage <name or index>` overrides it), and `--timesteps` counts the extra steps. Other options: `--num-envs` (a multiple of 16), `--seed`, `--eval-freq`.

**The curriculum** is the list of stages in `src/self_balancing_robot/curriculum.py`: each stage sets what the environment samples (command range, slopes, pushes, randomized physics, latency, noise) and the test to leave it. Add or change stages there (REWARDS.md, section 3). Other defaults (reward, environment, network, PPO) are in `src/self_balancing_robot/config.py`; MuJoCo model and geometry values are defined in the XML.

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

Then start training in the background:

```bash
cd software_stack
scripts/train_gpu1.sh                      # or e.g. --timesteps 300000000, --resume models/runs/<run>
tail -f models/runs/<run>/train.log
```

The script installs the environment on first use (with `uv` if available, which also uses the exact versions of `uv.lock`; otherwise pip), checks that JAX sees the GPU, downloads OSMesa (software OpenGL for the videos, needed where EGL does not work, e.g. on the cluster) into `.mujoco-osmesa/` without sudo with `scripts/setup_osmesa.sh`, and runs training under `nohup`, so closing the browser tab does not stop it. JAX allocates GPU memory as needed (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) since the GPU may be shared. If the hub shuts the server down, training stops with it: continue from the last checkpoint with `--resume`. Copy a finished run (or just its `params.pkl` and `curriculum.json`) back to evaluate, drive or export it locally. Training uses one GPU.

## Evaluation, driving and export

`make evaluate` runs deterministic episodes of the newest run (or `--model <run dir or policy.bin>`) at the last curriculum stage (or `--stage <name or index>`, e.g. `EVALUATE_ARGS='--stage "commands 0.3 m/s"'`), in parallel, and prints per-episode metrics; `--plot-dir plots/eval` saves a plot of every episode and `--video eval.mp4` records the first one.

To drive a trained policy yourself, run `make teleop`. It simulates the last curriculum stage (noise, latency,
slopes and random pushes); `make teleop TELEOP_ARGS='--stage "commands 0.3 m/s"'` gives a clean simulation. A window opens with a camera behind and above the robot: hold the
arrow keys to drive and turn, `+`/`-` change the speed limit (starts at 100% = 0.3 m/s and 1.5 rad/s, the
training range; like in training, speed and turn rate are scaled down together when the outer wheel
would need more than 0.3 m/s, so the ESP32 should limit joystick commands the same way), `V` cycles the camera (chase, top-down, side, free mouse view), Backspace
resets, and Esc quits. To test robustness, press a key, type a number and Enter: `G` slope in rad
(the robot is placed on the new slope), `F` floor friction coefficient, `X` / `Y` / `Z` centre-of-mass
offset in mm (axle, forward, up). `N` restores nominal physics. The bottom-left panel shows the current
values. Teleop simulates on the CPU. `--model` also takes a policy file, e.g. the deployed `models/policy.bin`.

To deploy on the ESP32, `make export` writes the actor weights and the observation constants of the newest run (or `EXPORT_ARGS="--model models/runs/<run>"`) to `models/policy_weights.h` and to the policy file `models/policy.bin`, which the robot firmware loads (ARCHITECTURE.md, section 4).

## Project layout

- `src/self_balancing_robot/assets/goose.xml`: MuJoCo model of the robot (free-floating base, DC-motor wheel actuators, IMU site); `env_v0.xml` puts it on a ground plane that training tilts for slopes. `balance_pid.py` is a PID baseline on the same model.
- `src/self_balancing_robot/envs/balance_env.py`: MJX environment (JAX): actor/privileged observations, unified reward, command curriculum, domain randomization, sensor noise and latency.
- `src/self_balancing_robot/envs/wrappers.py`: batched training wrapper: fresh resets of finished episodes and the global curriculum.
- `src/self_balancing_robot/actor.py`: asymmetric actor-critic network for Brax, and the actor as a plain MLP (as on the ESP32) for evaluation, teleop and export.
- `src/self_balancing_robot/train.py`: Brax PPO training entry point; `scripts/train_gpu1.sh` runs it in the background on the cluster.
- `src/self_balancing_robot/evaluate.py`, `metrics.py`: evaluation episodes, plots and videos.
- `src/self_balancing_robot/teleop.py`: keyboard driving of a trained policy in a live window.
- `src/self_balancing_robot/export.py`: actor weights to a C header and a policy file for the ESP32.
- `src/self_balancing_robot/benchmark.py`: environment throughput.
- `tests/`: environment, curriculum, network and export tests.
- `models/archive/`: notes of earlier policies (v1–v10, trained with Stable-Baselines3 on 8 CPU environments and the earlier model `XML-test.xml`; their checkpoints are in the git history).

Before transferring a policy to hardware, replace dimensions, masses, friction, actuator limits, sensor observations, latency, and safety thresholds with measured robot values. Collect synchronized IMU, encoder, command, motor, battery, and fall-event logs to calibrate the model and define validation targets.
