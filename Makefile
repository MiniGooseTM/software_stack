PYTHON ?= python3
VENV ?= .venv

.PHONY: install train evaluate test lint clean

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"

train:
	$(VENV)/bin/python -m self_balancing_robot.train $(TRAIN_ARGS)

evaluate:
	$(VENV)/bin/python -m self_balancing_robot.evaluate $(EVALUATE_ARGS)

test:
	$(VENV)/bin/python -m pytest

lint:
	$(VENV)/bin/python -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
