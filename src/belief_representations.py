from jax.random import categorical
import distrax, chex, jax
import jax.numpy as jnp
from functools import partial
from dynamics import *


class CategoricalBeliefState:
    """
        NOTE: This is not very fast
    """
    def __init__(self, number_of_unique_states, initial_belief_distribution=None):
        # We need to know the number of possible underlying states and the initial belief distribution over states
        self.belief_distribution = initial_belief_distribution if initial_belief_distribution != None else distrax.Categorical(probs=jnp.ones(number_of_unique_states)/number_of_unique_states)
        self.number_of_unique_states = number_of_unique_states
    
    def sample(self, key: chex.PRNGKey):
        return self.belief_distribution.sample(seed=key)
    
    def update(self, new_observation, previous_action, transition_model, observation_model):    # This assumes you have a transition model and an observation model!
        # The likelihood of a new state is proportional to the likelihood of the new observation given the action taken and the new state
        # multiplied by the sum of likelihoods of entering that new state.

        def calc_state_likelihood(state):
            # Observation model likelihood mult by transition likelihood
            
            def calc_transition_likelihood(previous_state, previous_action, previous_belief_distribution):
                return transition_model(state_num=previous_state, action_num=previous_action).prob(state) * previous_belief_distribution.prob(previous_state)

            transition_likelihood = jnp.sum(jax.vmap(calc_transition_likelihood, (0, None, None))(jnp.arange(self.number_of_unique_states), previous_action, self.belief_distribution))
            observation_likelihood = observation_model(state_num=state, action_num=previous_action).prob(new_observation)
            return observation_likelihood * transition_likelihood

        raw_belief_likelihoods = jax.vmap(calc_state_likelihood, (0,))(jnp.arange(self.number_of_unique_states)) # These are unnormalized at first

        belief_probs = raw_belief_likelihoods / jnp.sum(raw_belief_likelihoods)

        self.belief_distribution = distrax.Categorical(probs=belief_probs)

