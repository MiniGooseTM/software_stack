import functools
import os

# Tests run on CPU (also where a GPU is present): small batches, no GPU memory needed.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest
from brax.training.acme import running_statistics, specs

from self_balancing_robot.actor import network_factory
from self_balancing_robot.envs.balance_env import (
    ACTOR_KEY,
    CRITIC_KEY,
    PRIVILEGED_SIZE,
    SelfBalancingRobotEnv,
)


@functools.cache
def make_env(**kwargs) -> SelfBalancingRobotEnv:
    """One environment (and one compilation) per configuration for the whole test session."""
    return SelfBalancingRobotEnv(**kwargs)


@functools.cache
def _jitted(env: SelfBalancingRobotEnv):
    return jax.jit(env.reset_episode), jax.jit(env.step)


def persistent(env: SelfBalancingRobotEnv, stage: int = 1, level: int = 1, **values) -> dict:
    result = {**env.initial_persistent(), "stage": jnp.int32(stage), "level": jnp.int32(level)}
    for name, value in values.items():
        result[name] = jnp.asarray(value, dtype=result[name].dtype)
    return result


def reset(env: SelfBalancingRobotEnv, seed: int = 0, stage: int = 1, level: int = 1, **values):
    return _jitted(env)[0](jax.random.PRNGKey(seed), persistent(env, stage, level, **values))


def step(env: SelfBalancingRobotEnv, state, action=(0.0, 0.0)):
    return _jitted(env)[1](state, jnp.asarray(action, dtype=jnp.float32))


def set_info(state, **values):
    info = dict(state.info)
    for name, value in values.items():
        info[name] = jnp.asarray(value, dtype=info[name].dtype)
    return state.replace(info=info)


def initial_params(env: SelfBalancingRobotEnv, seed: int = 0):
    """Freshly initialized Brax PPO parameters (normalizer, policy, value) and networks."""
    shapes = {ACTOR_KEY: (env.actor_observation_size,), CRITIC_KEY: (PRIVILEGED_SIZE,)}
    networks = network_factory()(shapes, env.action_size)
    key_policy, key_value = jax.random.split(jax.random.PRNGKey(seed))
    normalizer = running_statistics.init_state(
        {key: specs.Array(shape, jnp.dtype("float32")) for key, shape in shapes.items()}
    )
    params = (
        normalizer,
        networks.policy_network.init(key_policy),
        networks.value_network.init(key_value),
    )
    return params, networks


@pytest.fixture(scope="session")
def env() -> SelfBalancingRobotEnv:
    return make_env()
