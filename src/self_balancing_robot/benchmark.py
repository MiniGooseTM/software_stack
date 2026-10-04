"""Measure batched environment throughput (env steps per second) on the current device.

Runs the training wrapper (physics, rewards, fresh resets, curriculum) with random actions,
which is what PPO spends most of its time on. Example: ``python -m
self_balancing_robot.benchmark --num-envs 1024 4096 8192``.
"""

import argparse
import time

import jax

from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.envs.wrappers import TrainingWrapper


def benchmark(num_envs: int, stage: int, steps: int) -> tuple[float, float]:
    """(compile seconds, env steps per second)."""
    env = TrainingWrapper(
        SelfBalancingRobotEnv(curriculum_stage=stage, difficulty_level=99, fixed_difficulty=True),
        curriculum=False,
    )

    def rollout(state, key):
        def body(state, key):
            action = jax.random.uniform(key, (num_envs, 2), minval=-0.3, maxval=0.3)
            return env.step(state, action), None

        return jax.lax.scan(body, state, jax.random.split(key, steps))[0]

    rollout = jax.jit(rollout)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    start = time.perf_counter()
    state = jax.block_until_ready(rollout(state, jax.random.PRNGKey(1)))
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    jax.block_until_ready(rollout(state, jax.random.PRNGKey(2)))
    return compile_time, steps * num_envs / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--stage", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--steps", type=int, default=100, help="Control steps per measurement.")
    args = parser.parse_args()
    print(f"Device: {jax.devices()[0].device_kind}, stage {args.stage}")
    for num_envs in args.num_envs:
        compile_time, rate = benchmark(num_envs, args.stage, args.steps)
        print(f"{num_envs:>6} envs: {rate:>10,.0f} steps/s (compiled in {compile_time:.0f} s)")


if __name__ == "__main__":
    main()
