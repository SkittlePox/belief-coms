from jax.random import categorical
import distrax, chex, jax
import jax.numpy as jnp
from functools import partial


class CategoricalBeliefState:
    def __init__(self, number_of_unique_states, number_of_unique_observations):
        # We need to know the number of possible underlying states and the initial belief distribution over states
        self.number_of_unique_states = number_of_unique_states
        self.number_of_unique_observations = number_of_unique_observations
    
    # def update(self, previous_belief_distribution, new_observation, previous_action, transition_model, observation_model):
    #     # The likelihood of a new state is proportional to the likelihood of the new observation given the action taken and the new state
    #     # multiplied by the sum of likelihoods of entering that new state.

    #     def calc_state_likelihood(state):
    #         def calc_transition_likelihood(previous_state):
    #             return transition_model(state_num=previous_state, action_num=previous_action).prob(state) * previous_belief_distribution.prob(previous_state)

    #         transition_likelihood = jnp.sum(jax.vmap(calc_transition_likelihood, (0,))(jnp.arange(self.number_of_unique_states)))
    #         observation_likelihood = observation_model(state_num=state, action_num=previous_action).prob(new_observation)
    #         return observation_likelihood * transition_likelihood

    #     raw_belief_likelihoods = jax.vmap(calc_state_likelihood, (0,))(jnp.arange(self.number_of_unique_states))
    #     belief_probs = raw_belief_likelihoods / jnp.sum(raw_belief_likelihoods)

    #     return distrax.Categorical(probs=belief_probs)
    
    def joint_update(self, previous_belief_distribution, new_observation, previous_action, joint_transition_model, joint_observation_model):
        # The likelihood of a new state is proportional to the likelihood of the new observation given the action taken and the new state
        # multiplied by the sum of likelihoods of entering that new state.

        def calc_state_likelihood(state):
            def calc_transition_likelihood(previous_state):
                return transition_model(state_num=previous_state, action_num=previous_action).prob(state) * previous_belief_distribution.prob(previous_state)

            transition_likelihood = jnp.sum(jax.vmap(calc_transition_likelihood, (0,))(jnp.arange(self.number_of_unique_states)))
            # Must marginalize over all possible actions the other agent may have taken!
            
            observation_likelihood = joint_observation_model(state_num=state, action_num=previous_action).prob(new_observation)
            return observation_likelihood * transition_likelihood

        raw_belief_likelihoods = jax.vmap(calc_state_likelihood, (0,))(jnp.arange(self.number_of_unique_states))
        belief_probs = raw_belief_likelihoods / jnp.sum(raw_belief_likelihoods)

        return distrax.Categorical(probs=belief_probs)


class DirichletBeliefOfBeliefState:
    def __init__(self, number_of_unique_states):
        self.number_of_unique_states = number_of_unique_states
