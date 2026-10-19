"""Asymmetric actor-critic policy (ARCHITECTURE.md, Option B).

The flat observation is ``[actor_history, privileged_state]``. The actor only sees
the first ``actor_observation_size`` entries, so it can run on the ESP32; the
critic only sees the privileged state and is discarded after training.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file
from torch import nn

from self_balancing_robot.config import DEFAULT_POLICY_CONFIG


def _relu_mlp(input_size: int, layer_sizes: Sequence[int]) -> nn.Sequential:
    """Linear layers of the given widths with a ReLU after each one."""
    layers: list[nn.Module] = []
    for size in layer_sizes:
        layers += [nn.Linear(input_size, size), nn.ReLU()]
        input_size = size
    return nn.Sequential(*layers)


class AsymmetricMlpExtractor(nn.Module):
    """Actor and critic networks that read different halves of the observation."""

    def __init__(
        self,
        actor_observation_size: int,
        critic_observation_size: int,
        actor_layers: Sequence[int],
        critic_layers: Sequence[int],
    ):
        super().__init__()
        self.actor_observation_size = actor_observation_size
        self.policy_net = _relu_mlp(actor_observation_size, actor_layers)
        self.value_net = _relu_mlp(critic_observation_size, critic_layers)
        self.latent_dim_pi = actor_layers[-1]
        self.latent_dim_vf = critic_layers[-1]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[..., : self.actor_observation_size])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features[..., self.actor_observation_size :])


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """Actor: Linear(11*H) -> 128 -> 64 -> 32 -> 2, critic: Linear(S) -> 128 -> 64 -> 32 -> 1, ReLU only.

    The action mean is a plain linear output (no tanh), clipped to [-1, 1] by the
    environment and by the ESP32 firmware.
    """

    def __init__(
        self,
        *args,
        actor_observation_size: int,
        actor_layers: Sequence[int] = DEFAULT_POLICY_CONFIG.actor_layers,
        critic_layers: Sequence[int] = DEFAULT_POLICY_CONFIG.critic_layers,
        **kwargs,
    ):
        self.actor_observation_size = actor_observation_size
        self.actor_layers = tuple(actor_layers)
        self.critic_layers = tuple(critic_layers)
        kwargs.setdefault("log_std_init", DEFAULT_POLICY_CONFIG.log_std_init)
        kwargs["activation_fn"] = nn.ReLU
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.actor_observation_size,
            self.features_dim - self.actor_observation_size,
            self.actor_layers,
            self.critic_layers,
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(
            actor_observation_size=self.actor_observation_size,
            actor_layers=self.actor_layers,
            critic_layers=self.critic_layers,
        )
        return data


# Models up to v7 did not save their layer sizes and were trained with these.
LEGACY_LAYERS = {"actor_layers": (64, 32), "critic_layers": (128, 64)}


def load_model(path, **kwargs) -> PPO:
    """`PPO.load` that also opens models saved before the layer sizes were stored."""
    data, _, _ = load_from_zip_file(path, device="cpu")
    policy_kwargs = data["policy_kwargs"]
    if "actor_layers" not in policy_kwargs:
        custom_objects = kwargs.setdefault("custom_objects", {})
        custom_objects["policy_kwargs"] = {**LEGACY_LAYERS, **policy_kwargs}
    return PPO.load(path, **kwargs)
