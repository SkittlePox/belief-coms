import dataclasses
import flax.linen as nn
from typing import Sequence, Tuple
import distrax
import jax.numpy as jnp

# Architecture defaults, shared by the modules below and by ``BeliefAgentConfig`` so
# that constructing a module directly (e.g. in tests) matches the tyro-config default.
DEFAULT_CONV_FEATURES: Tuple[int, ...] = (32, 64)
DEFAULT_CONV_KERNEL: int = 3
DEFAULT_TRUNK_DIMS: Tuple[int, ...] = (128, 128, 128)


def _conv_trunk(x, previous_belief, conv_features, conv_kernel, trunk_dims, input_utterance_shape):
    """Shared encoder: conv stack over the utterance image, concatenate the previous
    belief, then a dense trunk. Returns the trunk activations; each head (actor/critic)
    owns its own copy of these params (they call this separately)."""
    x = x.reshape((-1, *input_utterance_shape, 1))
    for features in conv_features:
        x = nn.Conv(features=features, kernel_size=(conv_kernel, conv_kernel), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))
    x = jnp.concatenate([x, previous_belief], axis=-1)
    for dim in trunk_dims:
        x = nn.Dense(dim)(x)
        x = nn.relu(x)
    return x


class BeliefActor(nn.Module):
    """
    Produces a Dirichlet distribution over updated belief states, using a
    log-space Bayesian update to ensure belief support is never expanded.

    The network encodes the utterance via conv layers, concatenates with the
    previous belief, and outputs log-likelihoods log L(utterance | state) for
    each state. The updated belief concentrations are then computed as:

        Actual Bayesian update:
            b'(s) ∝ b(s) * L(utterance | s)

        As implemented (functionally the same) — Log-space Bayesian update:
            log b'(s) = log b(s) + log L(utterance | s)
            b' = softmax(log b + log L)

    In log space, states where b(s) = 0 have log b(s) = -inf, so they remain
    at zero after softmax regardless of the learned log-likelihood. This
    structurally prevents the agent from placing mass on states it was certain
    were impossible.

    L(utterance | s) is essentially a denotational semantics.

    The resulting log b' is used as the concentration parameter of a Dirichlet
    (after shifting to be positive), so samples are belief states on the simplex.

    Args:
        input_utterance_shape: Spatial dimensions of the utterance image, e.g. `(8, 8)`.
        belief_dim: Number of categories in the belief state simplex.
    """

    input_utterance_shape: Sequence[int]
    belief_dim: int
    conv_features: Tuple[int, ...] = DEFAULT_CONV_FEATURES
    conv_kernel: int = DEFAULT_CONV_KERNEL
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, previous_belief, utterance):
        x = _conv_trunk(utterance, previous_belief, self.conv_features, self.conv_kernel, self.trunk_dims, self.input_utterance_shape)
        log_likelihood = nn.Dense(self.belief_dim)(x)  # log L(utterance | state)

        # Log-space Bayesian update: log b'(s) = log b(s) + log L(utterance | s)
        log_prior = jnp.log(previous_belief)
        log_posterior = log_prior + log_likelihood

        # Use log_posterior as Dirichlet concentrations (shift to positive)
        concentration = nn.softplus(log_posterior) + 1e-4
        return distrax.Dirichlet(concentration=concentration)


class BeliefCritic(nn.Module):
    input_utterance_shape: Sequence[int]
    belief_dim: int
    conv_features: Tuple[int, ...] = DEFAULT_CONV_FEATURES
    conv_kernel: int = DEFAULT_CONV_KERNEL
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, previous_belief, utterance):
        x = _conv_trunk(utterance, previous_belief, self.conv_features, self.conv_kernel, self.trunk_dims, self.input_utterance_shape)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class ActorCriticBeliefAgent(nn.Module):
    """
    An actor-critic agent that operates over belief states rather than actions.

    Given an utterance (an image of shape `input_utterance_shape`) and a previous belief
    state (a distribution over `belief_dim` categories), the agent produces:
      - A Dirichlet distribution over the simplex, representing a distribution
        over possible next belief states. Sampling from this yields a categorical
        probability vector — a belief state.
      - A scalar value estimate for use as the critic in policy gradient training.

    The actor and critic have entirely separate parameters: each runs its own
    convolutional encoder over the utterance, concatenates with the previous
    belief, and passes through independent dense layers.

    Args:
        input_utterance_shape: Spatial dimensions of the utterance image, e.g. `(8, 8)`.
        belief_dim: Number of categories in the belief state simplex.
    """

    input_utterance_shape: Sequence[int]
    belief_dim: int
    conv_features: Tuple[int, ...] = DEFAULT_CONV_FEATURES
    conv_kernel: int = DEFAULT_CONV_KERNEL
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    @nn.compact
    def __call__(self, previous_belief, utterance):
        arch = dict(
            input_utterance_shape=self.input_utterance_shape,
            belief_dim=self.belief_dim,
            conv_features=self.conv_features,
            conv_kernel=self.conv_kernel,
            trunk_dims=self.trunk_dims,
        )
        pi = BeliefActor(**arch)(previous_belief, utterance)
        value = BeliefCritic(**arch)(previous_belief, utterance)
        return pi, value


@dataclasses.dataclass(frozen=True)
class BeliefAgentConfig:
    """Architecture knobs for ``ActorCriticBeliefAgent``.

    Mirrors the ``AssignmentConfig``/``CommunicationConfig`` idiom: a tyro-facing
    dataclass of knobs whose ``build()`` returns the configured object, keeping the
    ``nn.Module`` free of any config-parsing. ``build()`` takes the env-derived shapes
    (``input_utterance_shape``, ``belief_dim``) as arguments -- like assignment configs
    take runtime ``num_agents`` -- since those are not user-facing architecture choices.

    Fields are tuples (not lists) because they become ``nn.Module`` attributes, which
    must be hashable for jit's static treatment; tyro produces tuples from
    ``Tuple[int, ...]`` natively.
    """

    conv_features: Tuple[int, ...] = DEFAULT_CONV_FEATURES
    conv_kernel: int = DEFAULT_CONV_KERNEL
    trunk_dims: Tuple[int, ...] = DEFAULT_TRUNK_DIMS

    def build(self, input_utterance_shape: Sequence[int], belief_dim: int) -> nn.Module:
        return ActorCriticBeliefAgent(
            input_utterance_shape=input_utterance_shape,
            belief_dim=belief_dim,
            conv_features=self.conv_features,
            conv_kernel=self.conv_kernel,
            trunk_dims=self.trunk_dims,
        )


if __name__ == "__main__":
    import jax

    input_shape = (8, 8)
    belief_dim = 16
    batch_size = 4

    # Build through the config to exercise the tyro-facing path (default arch).
    agent = BeliefAgentConfig().build(input_utterance_shape=input_shape, belief_dim=belief_dim)
    key = jax.random.PRNGKey(0)

    utterance = jax.random.uniform(key, shape=(batch_size, *input_shape))
    previous_belief = jax.random.dirichlet(key, alpha=jnp.ones(belief_dim), shape=(batch_size,))

    params = agent.init(key, previous_belief, utterance)
    pi, value = agent.apply(params, previous_belief, utterance)

    samples = pi.sample(seed=key)
    sample_sums = samples.sum(axis=-1)

    print("Concentration shape:", pi.concentration.shape)  # (batch_size, belief_dim)
    print("Sample shape:", samples.shape)  # (batch_size, belief_dim)
    print("Value shape:", value.shape)  # (batch_size,)
    print("Sample sums (should all be ~1.0):", sample_sums)
    assert jnp.allclose(sample_sums, jnp.ones(batch_size), atol=1e-5), "Samples do not sum to 1!"
    print("All samples sum to 1. ✓")
    print("Sample belief state (first):", samples[0])
    print("Critic value (first):", value[0])
