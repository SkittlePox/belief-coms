import flax.linen as nn
import distrax
import jax
import jax.numpy as jnp


class UtteranceActor(nn.Module):
    """
    Produces a MultivariateNormalDiag distribution over the utterance space
    conditioned on a target belief vector.

    Args:
        utterance_dim: Dimensionality of the flat utterance vector.
        belief_dim:    Number of categories in the belief state.
    """

    utterance_dim: int
    belief_dim: int

    @nn.compact
    def __call__(self, target_belief):
        x = nn.Dense(128)(target_belief)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.utterance_dim)(x)
        mean = nn.sigmoid(mean)

        scale_diag = nn.Dense(self.utterance_dim)(x)
        scale_diag = nn.softplus(scale_diag) + 1e-8

        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)


class UtteranceCritic(nn.Module):
    """
    Estimates the value of a target belief state, returning a scalar.

    Args:
        belief_dim: Number of categories in the belief state.
    """

    belief_dim: int

    @nn.compact
    def __call__(self, target_belief):
        x = nn.Dense(128)(target_belief)
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

    Given a target belief, produces a distribution over utterances (actor)
    and a scalar value estimate (critic) with entirely separate parameters.

    Args:
        utterance_dim: Dimensionality of the flat utterance vector.
        belief_dim:    Number of categories in the belief state.
    """

    utterance_dim: int
    belief_dim: int

    @nn.compact
    def __call__(self, target_belief):
        pi = UtteranceActor(
            utterance_dim=self.utterance_dim, belief_dim=self.belief_dim
        )(target_belief)
        value = UtteranceCritic(belief_dim=self.belief_dim)(target_belief)
        return pi, value


if __name__ == "__main__":
    utterance_dim = 16
    belief_dim = 5
    batch_size = 4

    agent = ActorCriticUtteranceAgent(
        utterance_dim=utterance_dim, belief_dim=belief_dim
    )
    key = jax.random.PRNGKey(0)

    target_belief = jax.random.dirichlet(
        key, alpha=jnp.ones(belief_dim), shape=(batch_size,)
    )

    params = agent.init(key, target_belief)
    pi, value = agent.apply(params, target_belief)

    samples = pi.sample(seed=key)

    print("Distribution mean shape:", pi.loc.shape)  # (batch_size, utterance_dim)
    print("Sample shape:", samples.shape)  # (batch_size, utterance_dim)
    print("Value shape:", value.shape)  # (batch_size,)
    print("Sample (first):", samples[0])
    print("Value (first):", value[0])
