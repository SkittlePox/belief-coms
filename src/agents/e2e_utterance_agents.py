"""The e2e utterance (speaker) agent.

One network maps the speaker's beliefs straight to its utterance. Run alone
(``--inference.mode e2e``) that is the whole agent. It is also the basis of inferential
communication: the ToM executors in ``agents.inference_agents`` use this network as the
speaker's own proposal policy and as the listener's model of its speaker. Names may change as
the inferential agents take shape.
"""

import dataclasses
from typing import Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp


def _dense_trunk(ego_belief, other_belief_estimate, second_order_ego_estimate, trunk_dims):
    """Concatenate the three belief levels, then a dense trunk."""
    x = jnp.concatenate([ego_belief, other_belief_estimate, second_order_ego_estimate], axis=-1)
    for dim in trunk_dims:
        x = nn.relu(nn.Dense(dim)(x))
    return x


class E2EUtteranceAgent(nn.Module):
    """
    Actor-critic agent for the utterance (speaker) role.

    Maps the speaker's three belief levels to a MultivariateNormalDiag over utterances and
    a scalar value estimate. Inputs may carry any leading batch axes, including none.

    Args:
        ego_belief: ``[..., S]`` b_e, the speaker's own belief.
        other_belief_estimate: ``[..., S]`` b̄_o, its estimate of its listener's belief.
        second_order_ego_estimate: ``[..., S]`` b̿_e, its estimate of what its listener
            believes about it.

    The actor is a dense trunk whose heads give the utterance mean, squashed to (0, 1) via
    sigmoid, and scale_diag, kept positive via softplus. The critic runs its own copy of
    the trunk (separate params) into a scalar.
    """

    utterance_action_dim: int
    trunk_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, ego_belief, other_belief_estimate, second_order_ego_estimate):
        trunk_inputs = (ego_belief, other_belief_estimate, second_order_ego_estimate)

        # Actor.
        x = _dense_trunk(*trunk_inputs, self.trunk_dims)
        mean = nn.sigmoid(nn.Dense(self.utterance_action_dim)(x))
        scale_diag = nn.softplus(nn.Dense(self.utterance_action_dim)(x)) + 1e-8
        pi = distrax.MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)

        # Critic: a second trunk call, so its own params.
        value = nn.Dense(1)(_dense_trunk(*trunk_inputs, self.trunk_dims))
        return pi, jnp.squeeze(value, axis=-1)


@dataclasses.dataclass(frozen=True)
class E2EUtteranceAgentConfig:
    """Architecture knobs for ``E2EUtteranceAgent``, the tyro-facing side of it.

    ``build()`` takes the env-derived ``utterance_action_dim``, which is not a user-facing
    choice. ``trunk_dims`` is a tuple (not a list) because it becomes an ``nn.Module``
    attribute, which must be hashable.
    """

    trunk_dims: Tuple[int, ...] = (128, 128, 128)

    def build(self, utterance_action_dim: int) -> nn.Module:
        return E2EUtteranceAgent(utterance_action_dim=utterance_action_dim, trunk_dims=self.trunk_dims)
