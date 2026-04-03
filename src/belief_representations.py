from stringprep import in_table_c11
from jax.random import categorical
import distrax, chex, jax
import jax.numpy as jnp
from functools import partial
from distributions import *


class CategoricalBeliefState:
    """Represents a belief over a set of possible underlying states. States are assumed to be categorical, so a belief can be represented by a single distrax categorical distribution.

    """
    def __init__(self, agent_id, num_unique_states, num_unique_observations, joint_transition_function, joint_observation_function, joint_action_constructor):
        self.agent_id = agent_id    # This is for indexing into the joint transition and observation functions
        self.num_unique_states = num_unique_states
        self.num_unique_observations = num_unique_observations
        self.joint_transition_function = joint_transition_function
        self.joint_observation_function = joint_observation_function
        self.joint_factory = JointCategoricalPair((num_unique_observations, num_unique_observations))
        self.joint_action_constructor = joint_action_constructor

    def update_with_observation_and_joint_action(self, belief_distribution: distrax.Categorical, observation, previous_joint_action):
        """Perform a Bayesian belief update given a new observation.

        Implements the standard POMDP belief update rule:

            b'(s') ∝ O(o | a, s') ∑_s T(s' | s, a) b(s)

        where:
          - b(s)  is the prior belief (probability of being in state s)
          - T(s' | s, a) is the transition model (probability of moving to s' from s under joint action a)
          - O(o | a, s') is the observation model (probability of observing o in state s' after action a)
          - b'(s') is the unnormalized posterior belief over next states s'

        The result is renormalized by distrax.Categorical to form a valid distribution.

        Because the observation model is joint over all agents — O(o1, o2 | s) — the agent's
        own marginal observation likelihood O(oi | s') is computed by marginalizing out the
        other agent's observation before evaluating the likelihood of the received observation.

        Args:
            belief_distribution: Current belief b(s) as a distrax.Categorical over states.
            observation: The observation received by this agent at the current timestep.
            previous_joint_action: The joint action taken by all agents at the previous timestep,
                used to condition the transition and observation models.

        Returns:
            A new distrax.Categorical representing the updated belief b'(s') (automatically
            normalized by distrax).
        """

        def state_likelihood(next_state):
            # ∑_s T(s' | s, a) b(s)
            def transition_contrib(state):
                return self.joint_transition_function(state, previous_joint_action).prob(next_state) * belief_distribution.prob(state)

            predicted_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))

            # O(oi | s') — marginalize the joint observation model down to this agent's view
            joint_obs = self.joint_observation_function(next_state)
            marginal_obs = jax.lax.cond(
                self.agent_id == 0,
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                lambda _: self.joint_factory.marginalize_var2(joint_obs),
                None
            )

            return marginal_obs.prob(observation) * predicted_prior

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def update_with_observation_only(self, ego_belief_distribution: distrax.Categorical, other_belief_distribution_estimate: distrax.Categorical, ego_observation, previous_ego_action, optimal_policy):
        """Perform a Bayesian belief update when the other agent's action is unobserved.

        Because the joint action is not directly observed, we marginalize over the other
        agent's possible actions weighted by their policy, giving:

            b'(s') ∝ ∑_a [ O(o | a, s') ∑_s T(s' | s, a) b(s) · π*(b̄_S)(a) ]

        where:
          - b(s)       is the ego agent's prior belief over states
          - T(s' | s, a) is the transition model conditioned on the joint action a
          - O(o | a, s') is the ego agent's marginal observation likelihood in state s'
          - π*(b̄_S)(a_other) is the probability the other agent takes action a_other
                              under their optimal policy given their belief b̄_S
          - The sum over a reduces to a sum over a_other since previous_ego_action is known;
            the joint action is reconstructed as a flat index from (ego_action, other_action)

        Args:
            ego_belief_distribution: The ego agent's current belief b(s) as a distrax.Categorical.
            other_belief_distribution_estimate: An estimate of other agent's current belief b̄_S, passed to
                optimal_policy to obtain a distribution over their actions.
            ego_observation: The observation received by the ego agent at this timestep.
            previous_ego_action: The ego agent's own action at the previous timestep (known).
            optimal_policy: A callable π* that takes a belief distribution and returns a
                distrax.Categorical over the other agent's actions.

        Returns:
            A new distrax.Categorical representing the updated belief b'(s').
        """
        # π*(b̄_S) — other agent's action distribution under their optimal policy
        other_action_dist = optimal_policy(other_belief_distribution_estimate)
        num_other_actions = other_action_dist.probs.shape[0]

        def state_likelihood(next_state):
            # O(oi | s') — marginalize joint obs model down to this agent's view
            joint_obs = self.joint_observation_function(next_state)
            marginal_obs = jax.lax.cond(
                self.agent_id == 0,
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                lambda _: self.joint_factory.marginalize_var2(joint_obs),
                None
            )
            obs_likelihood = marginal_obs.prob(ego_observation)  # O(oi | s')

            def action_contribution(other_action):
                # Reconstruct joint action from known ego action + candidate other action
                joint_action = self.joint_action_constructor(self.agent_id, previous_ego_action, other_action)

                # ∑_s T(s' | s, a) b(s)
                def transition_contrib(state):
                    return self.joint_transition_function(state, joint_action).prob(next_state) * ego_belief_distribution.prob(state)

                transition_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))
                return transition_prior * other_action_dist.prob(other_action)  # · π*(b̄_S)(a_other)

            # ∑_{a_other} [T-weighted prior · π*(b̄_S)(a_other)]
            marginalized_action_sum = jnp.sum(jax.vmap(action_contribution)(jnp.arange(num_other_actions)))
            return obs_likelihood * marginalized_action_sum  # O(oi | s') · ∑_a [...]

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def update_other_belief_estimate_with_observation_only(self, other_belief_distribution_estimate: distrax.Categorical, ego_observation, previous_ego_action, optimal_policy):
        """Update the ego agent's estimate of the other agent's belief, using only the ego agent's own observation.

        The ego agent cannot directly observe the other agent's observation, so it marginalizes
        over all possible other-agent observations, weighted by their likelihood conditioned on
        the ego agent's own observation:

            b̄'(s') ∝ ∑_{a_other} π*(b̄)(a_other) · ∑_{o_other} P(o_other | o_ego, s') · ∑_s T(s' | s, a) b̄(s)

        where:
          - b̄(s)          is the ego agent's current estimate of the other agent's belief
          - T(s' | s, a)  is the transition model under the reconstructed joint action a
          - P(o_other | o_ego, s') is the conditional observation likelihood for the other agent,
                                   derived from the joint observation model conditioned on the
                                   ego agent's known observation o_ego
          - π*(b̄)(a_other) is the probability the other agent takes action a_other under
                            their optimal policy given b̄
          - The joint action is reconstructed from previous_ego_action and each candidate a_other

        In effect this simulates how the other agent would update their belief, averaging over
        the observations they might have received given what the ego agent observed.

        Args:
            other_belief_distribution_estimate: The ego agent's current estimate of the other
                agent's belief b̄(s), as a distrax.Categorical over states.
            ego_observation: The observation received by the ego agent at this timestep.
            previous_ego_action: The ego agent's own action at the previous timestep (known).
            optimal_policy: A callable π* that takes a belief distribution and returns a
                distrax.Categorical over the other agent's actions.

        Returns:
            A new distrax.Categorical representing the updated estimate b̄'(s').
        """
        other_action_dist = optimal_policy(other_belief_distribution_estimate)
        num_other_actions = other_action_dist.probs.shape[0]

        def state_likelihood(next_state):
            joint_obs = self.joint_observation_function(next_state)

            their_marginal_obs = jax.lax.cond(
                self.agent_id == 0,
                lambda _: self.joint_factory.marginalize_var2(joint_obs),  # Order flipped: this is for the other agent
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                None
            )

            their_obs_likelihood = jax.lax.cond(
                self.agent_id == 0,
                lambda _: self.joint_factory.conditional_var2_given_var1(joint_obs, ego_observation),
                lambda _: self.joint_factory.conditional_var1_given_var2(joint_obs, ego_observation),
                None
            )

            def as_if_other_observed(other_obs):
                # likelihood that next_state yields other_obs * likelihood they observe other_obs given ego_obs
                # NOTE: THIS is not functioning as expected.
                return their_obs_likelihood.prob(other_obs)

            obs_component = jnp.sum(jax.vmap(as_if_other_observed)(jnp.arange(self.num_unique_observations))) 


            def as_if_other_took_action(other_action):
                joint_action = self.joint_action_constructor(self.agent_id, previous_ego_action, other_action)

                # ∑_s T(s' | s, a) b_other(s)
                def transition_contrib(state):
                    return self.joint_transition_function(state, joint_action).prob(next_state) * other_belief_distribution_estimate.prob(state)

                transition_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))
                return transition_prior * other_action_dist.prob(other_action)
            
            transition_component = jnp.sum(jax.vmap(as_if_other_took_action)(jnp.arange(num_other_actions))) 
            
            # Weight by optimal policy likelihood
            relative_state_likelihood = transition_component * obs_component
            return relative_state_likelihood

        props = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        probs = props / jnp.sum(props)
        return distrax.Categorical(probs=probs)
