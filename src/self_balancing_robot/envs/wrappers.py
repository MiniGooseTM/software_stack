"""Batched training wrapper: vmap, fresh auto-reset and the global curriculum.

`wrap_for_training` is passed to `brax.training.agents.ppo.train` as ``wrap_env_fn``. It
replaces Brax's Vmap/Episode/AutoReset stack:

- finished environments start a *fresh* episode (`env.reset_episode`) instead of
  Brax's cached first state, so every episode draws its own slope, heading, physics and
  command at the current curriculum stage;
- the curriculum is global. Episodes that finish at the current stage are pooled across all
  environments in windows of `curriculum.EVALUATION_WINDOW` episodes, and each full window
  is checked against the stage's test (survival, speed ratio, standing still, reversals,
  minimum episodes; see `curriculum.Stage`). When it passes, new episodes start at the next
  stage of the list; running ones finish at the stage they started with (and are not
  counted).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import wrapper as playground_wrapper

from self_balancing_robot.curriculum import EVALUATION_WINDOW
from self_balancing_robot.envs.balance_env import (
    PERSISTENT_KEYS,
    SelfBalancingRobotEnv,
    State,
)

CurriculumCallback = Callable[[dict], None]


def initial_curriculum(env: SelfBalancingRobotEnv) -> dict:
    return {
        "stage": jnp.int32(env.start_stage),
        "stage_episodes": jnp.int32(0),
        "window_episodes": jnp.int32(0),
        "window_successes": jnp.int32(0),
        # Sums of `episode_tracking`: speed reached, speed commanded, |v| standing, steps standing.
        "window_tracking": jnp.zeros(4),
        "window_reversals": jnp.int32(0),
        "window_reversal_successes": jnp.int32(0),
    }


def window_statistics(curriculum: dict) -> dict:
    """Success rate, speed ratio, standing-still speed and reversal survival of a window."""
    episodes = jnp.maximum(curriculum["window_episodes"], 1)
    reached, commanded, still_speed, still_steps = curriculum["window_tracking"]
    reversals = curriculum["window_reversals"]
    return {
        "success_rate": curriculum["window_successes"] / episodes,
        "speed_ratio": jnp.where(commanded > 0, reached / jnp.maximum(commanded, 1e-9), 1.0),
        "still_speed": jnp.where(still_steps > 0, still_speed / jnp.maximum(still_steps, 1.0), 0.0),
        "reversal_rate": jnp.where(
            reversals > 0,
            curriculum["window_reversal_successes"] / jnp.maximum(reversals, 1),
            1.0,
        ),
    }


def update_curriculum(
    env: SelfBalancingRobotEnv,
    curriculum: dict,
    info: dict,
    done: jax.Array,
    callback: CurriculumCallback | None = None,
) -> dict:
    """Add the episodes that just finished (batched `info`, `done`); move to the next stage
    when a full window passes the current stage's test."""
    stage = curriculum["stage"]
    counted = done & (info["episode_stage"] == stage)
    success = counted & ~info["terminated"]
    reversal = counted & info["episode_reversal"]
    window = {
        "window_episodes": curriculum["window_episodes"] + counted.sum(dtype=jnp.int32),
        "window_successes": curriculum["window_successes"] + success.sum(dtype=jnp.int32),
        "window_tracking": curriculum["window_tracking"]
        + jnp.sum(jnp.where(counted[:, None], info["episode_tracking"], 0.0), axis=0),
        "window_reversals": curriculum["window_reversals"] + reversal.sum(dtype=jnp.int32),
        "window_reversal_successes": curriculum["window_reversal_successes"]
        + (reversal & success).sum(dtype=jnp.int32),
    }
    stage_episodes = curriculum["stage_episodes"] + counted.sum(dtype=jnp.int32)
    stats = window_statistics(window)
    window_full = window["window_episodes"] >= EVALUATION_WINDOW

    # The stage's test; unchecked criteria (None in the Stage) always pass.
    test = env.stage_params(stage)
    passed = (
        window_full
        & (stage_episodes >= test["min_episodes"])
        & (stats["success_rate"] >= test["success_rate"])
        & (stats["speed_ratio"] >= test["speed_ratio"])
        & (stats["still_speed"] <= test["still_speed_limit"])
        & (stats["reversal_rate"] >= test["reversal_success_rate"])
    )
    # The last stage is never left.
    new_stage = jnp.where(passed & (stage < env.num_stages - 1), stage + 1, stage)

    if callback is not None:

        def report(*values):
            names = ("stage", "new_stage", "stage_episodes", "window_episodes", *stats)
            callback({name: np.asarray(value).item() for name, value in zip(names, values)})

        jax.lax.cond(
            window_full,
            lambda: jax.debug.callback(
                report,
                stage,
                new_stage,
                stage_episodes,
                window["window_episodes"],
                *stats.values(),
            ),
            lambda: None,
        )

    # Windows tumble: a full window is evaluated once and then started again.
    empty = initial_curriculum(env)
    window = {key: jnp.where(window_full, empty[key], value) for key, value in window.items()}
    return {
        "stage": new_stage,
        "stage_episodes": jnp.where(new_stage != stage, 0, stage_episodes),
        **window,
    }


class TrainingWrapper(playground_wrapper.Wrapper):
    """Runs a batch of environments; resets finished ones fresh; advances the curriculum.

    Also keeps the bookkeeping Brax's PPO and evaluator expect in `state.info`:
    ``truncation``, ``steps``, ``episode_done`` and ``episode_metrics``.
    """

    def __init__(
        self,
        env: SelfBalancingRobotEnv,
        curriculum: bool = True,
        curriculum_callback: CurriculumCallback | None = None,
    ):
        super().__init__(env)
        self._curriculum = curriculum
        self._callback = curriculum_callback
        abstract = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
        self._env_keys = tuple(abstract.info)
        self._metric_keys = tuple(abstract.metrics)

    def reset(self, rng: jax.Array) -> State:
        state = jax.vmap(self.env.reset)(rng)
        batch = rng.shape[:-1]
        zeros = jnp.zeros(batch)
        info = {
            **state.info,
            "steps": zeros,
            "episode_done": zeros,
            "episode_metrics": {
                "sum_reward": zeros,
                "length": zeros,
                **{key: zeros for key in self._metric_keys},
            },
            "curriculum": initial_curriculum(self.env.unwrapped),
        }
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        info = state.info
        env = self.env.unwrapped
        inner = state.replace(info={key: info[key] for key in self._env_keys})
        stepped = jax.vmap(self.env.step)(inner, action)
        done = stepped.done > 0
        step_info = stepped.info

        curriculum = info["curriculum"]
        if self._curriculum:
            curriculum = update_curriculum(env, curriculum, step_info, done, self._callback)

        keys = jax.vmap(jax.random.split)(step_info["rng"])
        step_info = {**step_info, "rng": keys[:, 0]}
        persistent = {key: step_info[key] for key in PERSISTENT_KEYS}
        if self._curriculum:
            persistent["stage"] = jnp.broadcast_to(curriculum["stage"], done.shape)
        fresh = jax.vmap(env.reset_episode)(keys[:, 1], persistent)

        def where_done(reset_value, value):
            mask = done.reshape(done.shape + (1,) * (value.ndim - 1))
            return jnp.where(mask, reset_value, value)

        data = jax.tree.map(where_done, fresh.data, stepped.data)
        obs = jax.tree.map(where_done, fresh.obs, stepped.obs)
        env_info = jax.tree.map(where_done, fresh.info, step_info)

        previous_done = info["episode_done"]
        episode_metrics = {
            key: value * (1 - previous_done) for key, value in info["episode_metrics"].items()
        }
        episode_metrics["sum_reward"] += stepped.reward
        episode_metrics["length"] += 1
        for key in self._metric_keys:
            episode_metrics[key] += stepped.metrics[key]

        new_info = {
            **info,
            **env_info,
            # Values of the step just taken, not of the fresh episode.
            "truncation": step_info["truncation"],
            "steps": step_info["episode_step"].astype(jnp.float32),
            "episode_done": done.astype(jnp.float32),
            "episode_metrics": episode_metrics,
            "curriculum": curriculum,
        }
        return stepped.replace(data=data, obs=obs, info=new_info)


def wrap_for_training(
    env: SelfBalancingRobotEnv,
    episode_length: int | None = None,
    action_repeat: int = 1,
    randomization_fn=None,
    curriculum_callback: CurriculumCallback | None = None,
) -> TrainingWrapper:
    """``wrap_env_fn`` for Brax PPO (training and evaluation environments).

    The curriculum runs unless the env has ``fixed_difficulty``; the episode length comes
    from the env (truncation).
    """
    del episode_length, randomization_fn
    if action_repeat != 1:
        raise ValueError("action_repeat is not supported; the env sets its own control period")
    curriculum = not env.unwrapped.fixed_difficulty
    return TrainingWrapper(
        env,
        curriculum=curriculum,
        curriculum_callback=curriculum_callback if curriculum else None,
    )
