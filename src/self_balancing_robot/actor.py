"""
The ESP32 policy: network definition for training and a plain MLP for inference.
"""
from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

from self_balancing_robot.config import DEFAULT_POLICY_CONFIG, DEFAULT_TRAINING_CONFIG, PolicyConfig
from self_balancing_robot.envs.balance_env import ACTOR_KEY, CRITIC_KEY

PARAMS_FILE = "params.pkl"
CURRICULUM_FILE = "curriculum.json"

Layers = list[tuple[np.ndarray, np.ndarray]]


def network_factory(config: PolicyConfig = DEFAULT_POLICY_CONFIG):
    """Brax PPO ``network_factory`` for the asymmetric actor-critic."""
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config.actor_layers),
        value_hidden_layer_sizes=tuple(config.critic_layers),
        activation=linen.relu,
        policy_obs_key=ACTOR_KEY,
        value_obs_key=CRITIC_KEY,
        distribution_type="normal",
        noise_std_type="log",
        init_noise_std=config.init_noise_std,
        policy_network_kernel_init_fn=jax.nn.initializers.orthogonal,
        policy_network_kernel_init_kwargs={"scale": float(np.sqrt(2))},
        value_network_kernel_init_fn=jax.nn.initializers.orthogonal,
        value_network_kernel_init_kwargs={"scale": float(np.sqrt(2))},
        mean_kernel_init_fn=jax.nn.initializers.orthogonal,
        mean_kernel_init_kwargs={"scale": 0.01},
    )


def forward(layers, x):
    """Unclipped actor output for one input or a batch (NumPy or JAX arrays)."""
    h = x
    for i, (weight, bias) in enumerate(layers):
        h = h @ weight.T + bias
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0) if isinstance(h, np.ndarray) else jnp.maximum(h, 0.0)
    return h


def layers_from_brax(policy_params) -> Layers:
    """Actor layers ``(W[out][in], b[out])`` from Brax policy parameters."""
    params = policy_params["params"]
    hidden = params["MLP_0"]
    dense = [hidden[f"hidden_{i}"] for i in range(len(hidden))] + [params["Dense_0"]]
    return [
        (np.asarray(layer["kernel"], dtype=np.float32).T, np.asarray(layer["bias"], dtype=np.float32))
        for layer in dense
    ]


class Actor:
    """The deployed policy: ReLU MLP on the actor observation, output clipped to [-1, 1]."""

    def __init__(self, layers: Layers, name: str = ""):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.name = name

    @property
    def input_size(self) -> int:
        return self.layers[0][0].shape[1]

    @property
    def sizes(self) -> list[int]:
        return [self.input_size, *(w.shape[0] for w, _ in self.layers)]

    @property
    def parameter_count(self) -> int:
        return sum(w.size + b.size for w, b in self.layers)

    def __call__(self, observation):
        """Clipped action for an observation dict or actor input (traceable by JAX)."""
        x = observation[ACTOR_KEY] if isinstance(observation, dict) else observation
        return jnp.clip(forward(self.layers, x), -1.0, 1.0)

    def predict(self, observation) -> np.ndarray:
        """Clipped action computed in NumPy float32, as on the ESP32."""
        x = observation[ACTOR_KEY] if isinstance(observation, dict) else observation
        return np.clip(forward(self.layers, np.asarray(x, np.float32)), -1.0, 1.0)

    @classmethod
    def load(cls, path: Path) -> Actor:
        """From a training run directory, its ``params.pkl``, or a ``policy.bin`` file."""
        path = Path(path)
        if path.suffix == ".bin":
            from self_balancing_robot.export import read_policy_file

            policy = read_policy_file(path.read_bytes())
            return cls([(w, b) for w, b, _ in policy["layers"]], name=policy["name"])
        params = load_params(path)
        return cls(layers_from_brax(params[1]), name=str(path))


def run_dir_of(path: Path) -> Path:
    path = Path(path)
    return path.parent if path.suffix == ".pkl" else path


def latest_run(runs: Path = DEFAULT_TRAINING_CONFIG.run_dir) -> Path:
    candidates = sorted(p for p in Path(runs).glob("*") if (p / PARAMS_FILE).exists())
    if not candidates:
        raise FileNotFoundError(f"no trained run with {PARAMS_FILE} in {runs}")
    return max(candidates, key=lambda p: (p / PARAMS_FILE).stat().st_mtime)


def resolve_model(path: Path | None) -> Path:
    """A model path from the command line; None means the newest training run."""
    return latest_run() if path is None else Path(path)


def save_params(run_dir: Path, params, curriculum: dict) -> None:
    """Brax parameters (normalizer, policy, value) and the curriculum position."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (PARAMS_FILE + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(jax_to_numpy(params), f)
    tmp.replace(run_dir / PARAMS_FILE)
    (run_dir / CURRICULUM_FILE).write_text(json.dumps(curriculum, indent=2) + "\n")


def load_params(path: Path):
    path = run_dir_of(path) / PARAMS_FILE
    with path.open("rb") as f:
        return pickle.load(f)


def load_curriculum(path: Path) -> dict:
    file = run_dir_of(path) / CURRICULUM_FILE
    return json.loads(file.read_text()) if file.exists() else {}


def jax_to_numpy(tree):
    return jax.tree.map(np.asarray, tree)
