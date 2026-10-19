PYTHON ?= python3
VENV ?= .venv

.PHONY: install install-cuda train evaluate teleop export benchmark test lint clean

# CPU-only JAX: evaluation, teleop, export and tests (training works, but slowly).
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"

# NVIDIA GPU (CUDA 12 wheels, no system CUDA needed): everything, including training.
install-cuda:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[cuda,dev]"

train:
	$(VENV)/bin/python -m self_balancing_robot.train $(TRAIN_ARGS)

evaluate:
	$(VENV)/bin/python -m self_balancing_robot.evaluate $(EVALUATE_ARGS)

teleop:
	$(VENV)/bin/python -m self_balancing_robot.teleop $(TELEOP_ARGS)

export:
	$(VENV)/bin/python -m self_balancing_robot.export $(EXPORT_ARGS)

benchmark:
	$(VENV)/bin/python -m self_balancing_robot.benchmark $(BENCHMARK_ARGS)

# No third-party pytest plugins (e.g. ROS's, when its setup.bash is sourced).
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/python -m pytest

lint:
	$(VENV)/bin/python -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
