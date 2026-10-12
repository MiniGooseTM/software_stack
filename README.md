# Self-balancing robot software stack

This repository contains the simulation and reinforcement-learning setup for a homemade two-wheel self-balancing robot. Training follows Option B in `ARCHITECTURE.md` (asymmetric actor-critic PPO) with the unified reward and curriculum in `REWARDS.md`. The robot is simulated in full 3D from `assets/XML-test.xml`. The actor is the small ReLU MLP meant for the ESP32: it outputs normalized PWM for the two wheel motors and sees only an 8-frame history of IMU, encoder, command, and previous-action readings. The critic additionally sees privileged simulation state (true velocities, CoM offset, mass, friction, pushes, latencies, slope) and is discarded after training.

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

The trained policy is saved to `models/ppo_balance.zip`, with checkpoints every 1M steps in `models/checkpoints/`; TensorBoard logs (including `curriculum/min_stage` and `curriculum/min_progress`, the stage plus the fraction of its levels passed) are written to `logs/`. The default run is 20M steps; use `--n-envs` to match the available CPU cores. To continue training a saved
model, pass `--resume models/ppo_balance.zip --start-stage 3 --start-level 4` (`--timesteps` then counts the extra
steps; the curriculum stage and level are not stored in the model, so pass the stage and level
that training printed at the end of the previous run). Stages: 1 balance, 2 command ramp, 3 slopes and pushes.
To train the same increasing stage-3 command difficulty without domain randomization, disable it
explicitly and save to a separate model path:
`make train TRAIN_ARGS="--domain-randomization none --output models/ppo_balance_no_dr.zip"`.
This still trains through stages 1 and 2 and reaches stage 3; at stage 3 it retains the full
command range while keeping mass, CoM, friction, slope, pushes, sensor noise and latency nominal/off.
The evaluation and video callbacks use the same setting. Use `--no-video` on headless servers to
skip video recording that requires an OpenGL context. Domain-randomization choice is not stored in
PPO checkpoints, so pass the same flag when resuming a run.
Training also evaluates held-out episodes at the current curriculum stage and writes to `src/plots/`:
`training_metrics.png` (train vs eval reward, survival, velocity and yaw-rate tracking error, mean
|pitch| and curriculum stage over timesteps) and `eval_episode_latest.png` / `eval_episode_final.png`
(commanded vs actual velocity and yaw rate, pitch and motor commands over one episode). The same
numbers are logged to TensorBoard under `eval/`. `make evaluate EVALUATE_ARGS="--stage 3 --plot-dir plots/eval"`
saves the per-episode plot for every evaluated episode. Evaluation frequency and episode count can be
changed with `--eval-freq` and `--eval-episodes`. `make evaluate EVALUATE_ARGS="--stage 3"` evaluates a
policy at the last level of a curriculum stage.

The same `make evaluate` command also runs quantitative suites:
`make evaluate EVALUATE_ARGS="--model models/ppo_balance.zip --suite all --episodes 5 --output results/evaluation.json"`.
The `balance` suite reports planar position drift and mean absolute pitch error at zero command;
`tracking` reports steady-state linear and angular velocity MAE over signed command setpoints,
excluding the first second by default. `robustness` compares the full-rate policy against IMU-only,
encoder-only, and combined sensor-rate reductions (50, 25 and 10 Hz by default), reporting survival,
survival time, pitch error and velocity MAE. Sensor rates must divide the simulation control rate;
use `--sensor-rates-hz` to choose rates or `--suite` to run one suite. The robustness suite always
uses stage 3 commands; `--domain-randomization none` disables randomized physics, sensor noise,
pushes, and latency while preserving stage 3, and `--domain-randomization full` enables them (the
quantitative default). Run either choice with `--suite all` to compare balance, tracking, and
robustness for that setting. The option also toggles randomization independently of `--stage` in the
standard evaluator. Results are printed as JSON and
optionally written to `--output`. The default `--suite standard` keeps the original per-episode
summary, plotting, and rendering behavior.

To drive a trained policy yourself, run `make teleop` (or `make teleop TELEOP_ARGS="--stage 3"` to add
noise, slopes and random pushes). A window opens with a camera behind and above the robot: hold the
arrow keys to drive and turn, `+`/`-` change the speed limit (starts at 100% = 0.3 m/s and 1.5 rad/s, the
training range; like in training, speed and turn rate are scaled down together when the outer wheel
would need more than 0.3 m/s, so the ESP32 should limit joystick commands the same way), `V` cycles the camera (chase, top-down, side, free mouse view), Backspace
resets, and Esc quits. To test robustness, press a key, type a number and Enter: `G` slope in rad
(the robot is placed on the new slope), `F` floor friction coefficient, `X` / `Y` / `Z` centre-of-mass
offset in mm (axle, forward, up). `N` restores nominal physics. The bottom-left panel shows the current
values. Use `--model` to drive an archived policy, e.g.
`models/archive/v1_pitch_angle_reward/ppo_balance.zip`. Training also records an MP4 of one evaluation episode every 1M steps
(`--video-freq`) and at the end of training in `videos/`. Shared defaults are in
`src/self_balancing_robot/config.py`; MuJoCo model and geometry values are defined in the XML.
Pass extra options to `make train` or `make evaluate` with `TRAIN_ARGS` or `EVALUATE_ARGS`.

To deploy on the ESP32, `make export` writes the actor weights and the observation constants of
`models/ppo_balance.zip` to `models/policy_weights.h` and to the policy file `models/policy.bin`,
which the robot firmware loads (ARCHITECTURE.md, section 4).

## Docker
There's no need for docker in the University Cluster. However it is available in case the cluster isn't.
The default image is CPU-only and works on macOS Docker Desktop and Ubuntu:

```bash
docker build -t self-balancing-robot .
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/logs:/app/logs" \
  self-balancing-robot
```

On an Ubuntu host with an NVIDIA runtime, use `Dockerfile.gpu` and add `--gpus all` to the run command. Training is headless; `MUJOCO_GL=egl` is set in both images.

## Project layout

- `src/self_balancing_robot/assets/XML-test.xml`: MuJoCo model of the robot used for training (free-floating base, DC-motor wheel actuators).
- `src/self_balancing_robot/assets/self_balancing_robot.xml`: earlier planar model, no longer used by the environment.
- `src/self_balancing_robot/envs/balance_env.py`: Gymnasium environment, actor/privileged observations, unified reward, command curriculum, and stage-four randomization.
- `src/self_balancing_robot/policy.py`: asymmetric actor-critic policy (ESP32 actor, privileged critic).
- `src/self_balancing_robot/train.py`: vectorized PPO training entry point.
- `src/self_balancing_robot/evaluate.py`: standard and quantitative evaluation entry point.
- `src/self_balancing_robot/evaluate_suite.py`: reusable quantitative balance, tracking, and sensor-rate metrics.
- `src/self_balancing_robot/teleop.py`: keyboard driving of a trained policy in a live window.
- `src/self_balancing_robot/export.py`: actor weights to a C header and a policy file for the ESP32.
- `tests/`: environment smoke tests.

The XML is intentionally a starting point. The environment implements four curriculum stages: upright stabilization, micro-mobility, full command tracking, and domain randomization/disturbance recovery. Before transferring a policy to hardware, replace dimensions, masses, friction, actuator limits, sensor observations, latency, and safety thresholds with measured robot values. Collect synchronized IMU, encoder, command, motor, battery, and fall-event logs to calibrate the model and define validation targets.
