"""The e2e belief (listener) agent.

One network maps the listener's beliefs and the utterance it hears straight to its new
belief. Run alone (``--inference.mode e2e``) that is the whole agent. It is also the basis of
inferential communication: the ToM executors in ``agents.inference_agents`` use this network
as the listener's own policy and as the speaker's model of its listener. Names may change as
the inferential agents take shape.
"""

import dataclasses
from typing import Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp


def _conv_trunk(ego_belief, other_belief_estimate, second_order_ego_estimate, utterance_image, conv_features, conv_kernel, trunk_dims):
    """Conv stack over the utterance image, concatenate the three belief levels, then a dense trunk."""
    x = utterance_image[..., None]  # [..., D, D, 1]: one input channel
    for features in conv_features:
        x = nn.relu(nn.Conv(features=features, kernel_size=(conv_kernel, conv_kernel), padding="SAME")(x))
    x = x.reshape((*x.shape[:-3], -1))  # flatten the image dims, keeping any batch axes
    x = jnp.concatenate([x, ego_belief, other_belief_estimate, second_order_ego_estimate], axis=-1)
    for dim in trunk_dims:
        x = nn.relu(nn.Dense(dim)(x))
    return x


class E2EBeliefAgent(nn.Module):
    """
    Actor-critic agent for the belief (listener) role.

    Maps the listener's three belief levels and the rendered utterance it hears to a
    Dirichlet over its new belief b_e and a scalar value estimate. Inputs may carry any
    leading batch axes, including none.

    Args:
        ego_belief: ``[..., S]`` b_e, the listener's own belief.
        other_belief_estimate: ``[..., S]`` b̄_o, its estimate of its speaker's belief.
        second_order_ego_estimate: ``[..., S]`` b̿_e, its estimate of what its speaker
            believes about it.
        utterance_image: ``[..., D, D]`` the speaker's rendered utterance.

    The actor is a trunk (conv over the utterance image, then dense over it and the belief
    levels) whose head outputs log-likelihoods log L(utterance | s), essentially a
    denotational semantics. The Dirichlet's concentration is ``softplus(logits) + 1e-4``,
    where the logits are:

      - ``multiply_by_ego_belief=False``: log L alone.
      - ``multiply_by_ego_belief=True``: log b_e + log L, a log-space Bayesian update
        b'(s) ∝ b(s) * L(utterance | s). States where b_e(s) = 0 have log b_e(s) = -inf,
        so the new belief never gains support b_e did not have.
        NOTE: those states get concentration exactly 1e-4 regardless of params. A
        Dirichlet with alpha < 1 has unbounded density as x_s -> 0, and those coordinates
        sample as exactly 0.0 in float32, so log_prob of a sample is +inf (breaks the PPO
        ratio). Likely fix: a Dirichlet restricted to the support of b (zero-support
        coords fixed at 0, excluded from log_prob).

    The critic runs its own copy of the trunk (separate params) into a scalar.
    """

    conv_features: Tuple[int, ...]
    conv_kernel: int
    trunk_dims: Tuple[int, ...]
    multiply_by_ego_belief: bool

    @nn.compact
    def __call__(self, ego_belief, other_belief_estimate, second_order_ego_estimate, utterance_image):
        trunk_inputs = (ego_belief, other_belief_estimate, second_order_ego_estimate, utterance_image)
        arch = (self.conv_features, self.conv_kernel, self.trunk_dims)

        # Actor.
        x = _conv_trunk(*trunk_inputs, *arch)
        logits = nn.Dense(ego_belief.shape[-1])(x)  # log L(utterance | s)
        if self.multiply_by_ego_belief:
            logits = jnp.log(ego_belief) + logits
        pi = distrax.Dirichlet(concentration=nn.softplus(logits) + 1e-4)

        # Critic: a second trunk call, so its own params.
        value = nn.Dense(1)(_conv_trunk(*trunk_inputs, *arch))
        return pi, jnp.squeeze(value, axis=-1)


@dataclasses.dataclass(frozen=True)
class E2EBeliefAgentConfig:
    """Architecture knobs for ``E2EBeliefAgent``, the tyro-facing side of it.

    Fields are tuples (not lists) because they become ``nn.Module`` attributes, which must
    be hashable.
    """

    conv_features: Tuple[int, ...] = (32, 64)
    conv_kernel: int = 3
    trunk_dims: Tuple[int, ...] = (128, 128, 128)
    # Whether the new belief is a log-space update of the input ego_belief (see E2EBeliefAgent).
    multiply_by_ego_belief: bool = True

    def build(self) -> nn.Module:
        return E2EBeliefAgent(
            conv_features=self.conv_features,
            conv_kernel=self.conv_kernel,
            trunk_dims=self.trunk_dims,
            multiply_by_ego_belief=self.multiply_by_ego_belief,
        )
