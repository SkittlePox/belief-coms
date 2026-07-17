import dataclasses
import flax.linen as nn
from typing import Tuple
import distrax
import jax
import jax.numpy as jnp

# Architecture defaults, shared by the modules below and by ``UtteranceAgentConfig`` so
# that constructing a module directly (e.g. in tests) matches the tyro-config default.
DEFAULT_TRUNK_DIMS: Tuple[int, ...] = (128, 128, 128)


def _dense_trunk(own_belief, estimate_of_receiver_belief, trunk_dims):
    """Shared encoder: concatenate the two beliefs then a dense trunk. Returns the
    trunk activations; each head (actor/critic) owns its own copy of these params
    (they call this separately)."""
    x = jnp.concatenate([own_belief, estimate_of_receiver_belief], axis=-1)
    for dim in trunk_dims:
        x = nn.Dense(dim)(x)
        x = nn.relu(x)
    return x


class UtteranceActor(nn.Module):
    """
    Produces a MultivariateNormalDiag distribution over the utterance space.

    The sender concatenates its own belief with its estimate of the receiver's
    belief and maps through dense layers to a (mean, scale_diag) pair.  The
    mean is squashed to (0, 1) via sigmoid; scale_diag is kept positive via
    softplus.

    Args:
        utterance_action_dim: Dimensionality of the flat utterance vector.
        belief_dim:    Number of categories in each belief state.
    """

    utterance_action_dim: int
    belief_dim: int
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        x = _dense_trunk(own_belief, estimate_of_receiver_belief, self.trunk_dims)

        mean = nn.Dense(self.utterance_action_dim)(x)
        mean = nn.sigmoid(mean)

        scale_diag = nn.Dense(self.utterance_action_dim)(x)
        scale_diag = nn.softplus(scale_diag) + 1e-8

        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)


class UtteranceCritic(nn.Module):
    """
    Estimates the value of the sender's current state (own belief +
    estimate of receiver's belief), returning a scalar.

    Args:
        belief_dim: Number of categories in each belief state.
    """

    belief_dim: int
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        x = _dense_trunk(own_belief, estimate_of_receiver_belief, self.trunk_dims)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class ActorCriticUtteranceAgent(nn.Module):
    """
    Actor-critic agent for the utterance (sender) role.

    Given the sender's own belief and its estimate of the receiver's belief,
    produces a distribution over utterances (actor) and a scalar value estimate
    (critic).  The actor and critic have entirely separate parameters.

    Args:
        utterance_action_dim: Dimensionality of the flat utterance vector.
        belief_dim:    Number of categories in each belief state.
    """

    utterance_action_dim: int
    belief_dim: int
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        pi = UtteranceActor(utterance_action_dim=self.utterance_action_dim, belief_dim=self.belief_dim, trunk_dims=self.trunk_dims)(
            own_belief, estimate_of_receiver_belief
        )
        value = UtteranceCritic(belief_dim=self.belief_dim, trunk_dims=self.trunk_dims)(own_belief, estimate_of_receiver_belief)
        return pi, value


@dataclasses.dataclass(frozen=True)
class UtteranceAgentConfig:
    """Architecture knobs for ``ActorCriticUtteranceAgent``.

    Mirrors the ``BeliefAgentConfig``/``AssignmentConfig`` idiom: a tyro-facing dataclass
    of knobs whose ``build()`` returns the configured object, keeping the ``nn.Module``
    free of any config-parsing. ``build()`` takes the env-derived shapes
    (``utterance_action_dim``, ``belief_dim``) as arguments, since those are not
    user-facing architecture choices.

    This is an MLP-only agent, so the sole knob is the dense-trunk width/depth.
    ``trunk_dims`` is a tuple (not a list) because it becomes an ``nn.Module`` attribute,
    which must be hashable for jit's static treatment; tyro produces tuples from
    ``Tuple[int, ...]`` natively.
    """

    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    def build(self, utterance_action_dim: int, belief_dim: int) -> nn.Module:
        return ActorCriticUtteranceAgent(
            utterance_action_dim=utterance_action_dim,
            belief_dim=belief_dim,
            trunk_dims=self.trunk_dims,
        )


if __name__ == "__main__":
    utterance_action_dim = 16
    belief_dim = 5
    batch_size = 4

    # Build through the config to exercise the tyro-facing path (default arch).
    agent = UtteranceAgentConfig().build(utterance_action_dim=utterance_action_dim, belief_dim=belief_dim)
    key = jax.random.PRNGKey(0)

    own_belief = jax.random.dirichlet(key, alpha=jnp.ones(belief_dim), shape=(batch_size,))
    estimate_of_receiver_belief = jax.random.dirichlet(key, alpha=jnp.ones(belief_dim), shape=(batch_size,))

    params = agent.init(key, own_belief, estimate_of_receiver_belief)
    pi, value = agent.apply(params, own_belief, estimate_of_receiver_belief)

    samples = pi.sample(seed=key)

    print("Distribution mean shape:", pi.loc.shape)  # (batch_size, utterance_dim)
    print("Sample shape:", samples.shape)  # (batch_size, utterance_dim)
    print("Value shape:", value.shape)  # (batch_size,)
    print("Sample (first):", samples[0])
    print("Value (first):", value[0])
