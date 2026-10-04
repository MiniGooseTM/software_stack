#!/usr/bin/env bash
# Train in the background on a single-GPU Linux machine, e.g. the KTH gpu1 JupyterHub.
#
#   scripts/train_gpu1.sh                                  # default run (config.py)
#   scripts/train_gpu1.sh --timesteps 300000000            # any train.py options
#   scripts/train_gpu1.sh --resume models/runs/<run>       # continue a run
#
# Sets up the environment on first use (uv if available, else pip), then starts training
# with nohup so it keeps running when the browser tab or terminal closes. Output goes to
# models/runs/<time>/train.log.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-.venv}
if command -v uv >/dev/null; then
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --extra cuda --quiet
else
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -e ".[cuda]"
fi
"$VENV/bin/python" -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print('GPU:', d.device_kind)"

# The GPU may be shared: allocate memory as needed instead of 75% of it up front.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

RUN_DIR=models/runs/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
nohup "$VENV/bin/python" -m self_balancing_robot.train --run-dir "$RUN_DIR" "$@" \
    > "$RUN_DIR/train.log" 2>&1 &
echo "$!" > "$RUN_DIR/train.pid"
echo "Training in the background (PID $!) in $RUN_DIR"
echo "  follow:      tail -f $RUN_DIR/train.log"
echo "  stop:        kill \$(cat $RUN_DIR/train.pid)"
echo "  TensorBoard: $VENV/bin/tensorboard --logdir logs"
