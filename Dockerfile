FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYTHONPATH=/app/src

WORKDIR /app
COPY pyproject.toml Makefile README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python", "-m", "self_balancing_robot.train"]
