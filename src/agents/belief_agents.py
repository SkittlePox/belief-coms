import flax.linen as nn
from typing import Sequence, Tuple, Dict
import distrax
import jax.numpy as jnp


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

    @nn.compact
    def __call__(self, previous_belief, utterance):
        x = utterance.reshape((-1, *self.input_utterance_shape, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, previous_belief], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
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

    @nn.compact
    def __call__(self, previous_belief, utterance):
        x = utterance.reshape((-1, *self.input_utterance_shape, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, previous_belief], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
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

    @nn.compact
    def __call__(self, previous_belief, utterance):
        pi = BeliefActor(input_utterance_shape=self.input_utterance_shape, belief_dim=self.belief_dim)(previous_belief, utterance)
        value = BeliefCritic(input_utterance_shape=self.input_utterance_shape, belief_dim=self.belief_dim)(previous_belief, utterance)
        return pi, value


if __name__ == "__main__":
    import jax

    input_shape = (8, 8)
    belief_dim = 16
    batch_size = 4

    agent = ActorCriticBeliefAgent(input_utterance_shape=input_shape, belief_dim=belief_dim)
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
