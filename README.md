# MiniGoose software stack
This repository contains the training stack for the MiniGoose robot. 
It uses MuJoCo MJX and Brax PPO to train a policy in simulation, utilizing curriculum learning and domain randomization.
We can then export the policy into a binary file, which can be uploaded to the real robot.

## Setup

Python 3.11–3.13 needed. 
For GPU, you need a 

```bash
make install-cuda   # NVIDIA GPU (training); `make install` for CPU only
make test           # runs on the CPU
make train
```

On an RTX 4050 laptop GPU training runs ~450k environment steps per second (4,096 environments), so the default 200M steps take about 25 minutes.
## Repository structure
```
/software_stack
├── /src/self_balancing_robot
│   ├── /assets
│   │   ├── /meshes                 # STL meshes of the chassis and wheels
│   │   ├── goose.xml               # MuJoCo model of the robot 
│   │   ├── env_v0.xml              # The environment
│   │   └── model_test.py           # Checks that env_v0.xml loads
│   ├── /envs
│   │   ├── balance_env.py          # MJX environment
│   │   └── wrappers.py             # training wrapper
│   ├── actor.py                    # Actor-critic network PPO network
│   ├── config.py                   # Default reward, environment, network and PPO settings
│   ├── curriculum.py               # Curriculum stages
│   ├── train.py                   
│   ├── evaluate.py                 
│   ├── metrics.py                  # Episode metrics and plots
│   ├── teleop.py                   # Keyboard driving of a trained policy
│   ├── export.py                   # Policy export for the ESP32
│   └── render.py                   # Renders recorded episodes to MP4
│
├── /models
│   └── /runs                       # Training runs (not tracked)
│
├── /models_share                   # Policies used in the experiments
│   ├── /default
│   ├── /more_noise
│   └── /no_randomization
│
├── /scripts
│   ├── train_gpu1.sh               # Training on GPU cluster
│   └── setup_osmesa.sh             # Downloads OSMesa for headless video rendering
│
├── /tests                          # Environment, curriculum, training and export tests (make test)
├── /logs                           # TensorBoard logs (not tracked)
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Training

Each run writes to its own directory, `models/runs/<date-time>/`:

- `params.pkl`: the latest network weights saved every `--eval-freq` steps (5M)
- `curriculum.json`: curriculum stage (index and name) and the total number of steps
- `config.json`: all settings of the run, including the curriculum
- `plots/training_metrics.png`: metrics regarding the training
-  `plots/eval_episode_latest.png`: command vs response of one episode at the current stage
- `plots/stages/` and `videos/`: the same plots for each passed stage, plus a video of one episode from that stage.

The same numbers, the PPO losses and the curriculum go to TensorBoard: `.venv/bin/tensorboard --logdir logs`. Curriculum stage changes are also printed as they happen.

Continue a run with `make train TRAIN_ARGS="--resume models/runs/<run>"`; it picks up the weights and the curriculum stage (matched by name, so stages added earlier in the list do not shift it; `--start-stage <name or index>` overrides it), and `--timesteps` counts the extra steps. Other options: `--num-envs` (a multiple of 16), `--seed`, `--eval-freq`.

**The curriculum** is the list of stages in `src/self_balancing_robot/curriculum.py`: each stage sets what the environment samples (command range, slopes, pushes, randomized physics, latency, noise) and the test to leave it.
You can freely edit the curriculum to add/change the existing stages.

## Training on a cluster

Create a GitHub SSH token, and clone the repository.
```bash
git clone git@github.com:MiniGooseTM/software_stack.git
```
Then start training in the background:

```bash
cd software_stack
scripts/train_gpu1.sh                      # or e.g. --timesteps 300000000, --resume models/runs/<run>
tail -f models/runs/<run>/train.log        # tail to see the output
```
## Evaluation, driving and export

`make evaluate` runs deterministic episodes of the newest run (or `--model <run dir or policy.bin>`) at the last curriculum stage (or `--stage <name or index>`, e.g. `EVALUATE_ARGS='--stage "commands 0.4 m/s"'`), in parallel, and prints per-episode metrics; `--plot-dir plots/eval` saves a plot of every episode and `--video eval.mp4` records the first one.

To drive a trained policy yourself, run `make teleop`. 
Then use your arrow keys to send commands to the robot. For more information, check `src/self_balancing_robot/teleop.py`.

To deploy on the ESP32, `make export` writes the actor weights and the observation constants of the newest run (you can also specify the exact run). 
The export file creates a header file and a binary.
The binary file is uploaded onto the ESP32.