import flax.linen as nn
from typing import Sequence
import distrax
import jax
import jax.numpy as jnp


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

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        x = jnp.concatenate([own_belief, estimate_of_receiver_belief], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

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

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        x = jnp.concatenate([own_belief, estimate_of_receiver_belief], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
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

    @nn.compact
    def __call__(self, own_belief, estimate_of_receiver_belief):
        pi = UtteranceActor(utterance_action_dim=self.utterance_action_dim, belief_dim=self.belief_dim)(own_belief, estimate_of_receiver_belief)
        value = UtteranceCritic(belief_dim=self.belief_dim)(own_belief, estimate_of_receiver_belief)
        return pi, value


if __name__ == "__main__":
    utterance_action_dim = 16
    belief_dim = 5
    batch_size = 4

    agent = ActorCriticUtteranceAgent(utterance_action_dim=utterance_action_dim, belief_dim=belief_dim)
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
