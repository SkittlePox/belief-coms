from stringprep import in_table_c11
from jax.random import categorical
import distrax, chex, jax
import jax.numpy as jnp
from functools import partial
from .distributions import *


class CategoricalBeliefState:
    """Represents a belief over a set of possible underlying states. States are assumed to be categorical, so a belief can be represented by a single distrax categorical distribution.

    """
    def __init__(self, env_params):
        """Build a belief-update engine for a single DecPOMDP.

        Args:
            env_params: A FlexibleEnvParams (or any object exposing ``transition``
                [S, A, A, S], ``observation`` [S, A, A, O, O], ``num_states`` and
                ``num_actions``). The dense dynamics tensors are read directly via
                gathers, replacing the per-env callables this class used to take.
                Must be concrete (not a tracer) so the cardinalities are static.

        Note: pass a SINGLE game's params here, not the stacked (leading
        game-type axis) FlexibleEnvParams produced by ``assemble_environments``.
        """
        self.env_params = env_params
        self.num_unique_states = int(env_params.num_states)
        self.num_unique_actions = int(env_params.num_actions)
        # A single observation alphabet shared by all agents, read off the
        # observation tensor's trailing axis.
        self.num_unique_observations = env_params.observation.shape[-1]
        # The joint observation model O(o0, o1 | s', a) is still correlated, but
        # both agents use the same cardinality, so the two per-agent marginals
        # are the same size (this is what removes the earlier lax.cond mismatch).
        self.joint_factory = JointCategoricalPair((self.num_unique_observations, self.num_unique_observations))

    def joint_transition_function(self, state, joint_action) -> distrax.Categorical:
        """T(s' | s, a0, a1) as a gather into env_params.transition -> [S']."""
        agent_0_action, agent_1_action = joint_action
        return distrax.Categorical(probs=self.env_params.transition[state, agent_0_action, agent_1_action])

    def joint_observation_function(self, next_state, joint_action) -> distrax.Categorical:
        """O(o0, o1 | s', a0, a1) flattened to [O * O] (JointCategoricalPair order)."""
        agent_0_action, agent_1_action = joint_action
        probs = self.env_params.observation[next_state, agent_0_action, agent_1_action].reshape(-1)
        return distrax.Categorical(probs=probs)

    def joint_action_constructor(self, agent_id, ego_action, other_action):
        """Order (ego, other) actions into the (agent_0, agent_1) joint action."""
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None,
        )

    def update_with_observation_and_joint_action(
        self, 
        belief_distribution: distrax.Categorical, 
        observation, 
        previous_joint_action,
        agent_id = 0
    ):
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
            joint_obs = self.joint_observation_function(next_state, previous_joint_action)
            marginal_obs = jax.lax.cond(
                agent_id == 0,
                lambda _: self.joint_factory.marginalize_var2(joint_obs),
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                None
            )

            return marginal_obs.prob(observation) * predicted_prior

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def update_with_observation_only(
        self, 
        ego_belief_distribution: distrax.Categorical, 
        other_belief_distribution_estimate: distrax.Categorical, 
        ego_observation, 
        previous_ego_action, 
        other_optimal_policy,
        agent_id = 0
    ):
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
                other_optimal_policy to obtain a distribution over their actions....
            ego_observation: The observation received by the ego agent at this timestep.
            previous_ego_action: The ego agent's own action at the previous timestep (known).
            other_optimal_policy: A callable π* that takes a belief distribution and returns a
                distrax.Categorical over the other agent's actions.

        Returns:
            A new distrax.Categorical representing the updated belief b'(s').
        """
        # π*(b̄_S) — other agent's action distribution under their optimal policy
        other_action_dist = other_optimal_policy(other_belief_distribution_estimate)

        def state_likelihood(next_state):
            def contribution_for_other_action(other_action):
                joint_action = self.joint_action_constructor(agent_id, previous_ego_action, other_action)

                # O(oi | s') — marginalize joint obs model down to this agent's view
                joint_obs = self.joint_observation_function(next_state, joint_action)
                marginal_obs = jax.lax.cond(
                    agent_id == 0,
                    lambda _: self.joint_factory.marginalize_var2(joint_obs),
                    lambda _: self.joint_factory.marginalize_var1(joint_obs),
                    None
                )
                obs_likelihood = marginal_obs.prob(ego_observation)

                # ∑_s T(s' | s, a) b(s)
                def transition_contrib(state):
                    return self.joint_transition_function(state, joint_action).prob(next_state) * ego_belief_distribution.prob(state)

                transition_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))

                return obs_likelihood * transition_prior * other_action_dist.prob(other_action)

            return jnp.sum(jax.vmap(contribution_for_other_action)(jnp.arange(self.num_unique_actions)))

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def update_other_belief_estimate_with_observation_only(
        self,
        other_belief_distribution_estimate: distrax.Categorical,
        ego_observation,
        previous_ego_action,
        other_optimal_policy,
        agent_id = 0    # This is the ego agent's id!
    ):
        """Update the ego agent's estimate of the other agent's belief state.

        The ego agent cannot observe o_other directly, so it marginalizes over all
        possible other-agent observations, weighting each by its likelihood conditioned
        on the ego agent's own observation. For each hypothetical o_other, a full belief
        update is run from the other agent's perspective, and the results are averaged:

            b̄'(s') = ∑_{a_other} π*(b̄)(a_other) · ∑_{o_other} P(o_other | o_ego, a) · b̄_updated(s' | o_other, a)

        where:

            P(o_other | o_ego, a) = ∑_{s'} P(o_other | o_ego, a, s') · ∑_s T(s' | s, a) b̄(s)

            b̄_updated(s' | o_other, a) ∝ O_other(o_other | s', a) · ∑_s T(s' | s, a) b̄(s)

        and the joint action a is reconstructed from previous_ego_action and each
        candidate a_other drawn from π*(b̄).

        Args:
            other_belief_distribution_estimate: The ego agent's current estimate of the
                other agent's belief b̄(s), as a distrax.Categorical over states.
            ego_observation: The observation received by the ego agent at this timestep.
            previous_ego_action: The ego agent's action at the previous timestep.
            other_optimal_policy: A callable π* mapping a belief distribution to a
                distrax.Categorical over the other agent's actions.

        Returns:
            A new distrax.Categorical representing the updated estimate b̄'(s').
        """
        other_action_dist = other_optimal_policy(other_belief_distribution_estimate)

        def as_if_other_took_action(other_action):
            joint_action = self.joint_action_constructor(agent_id, previous_ego_action, other_action)

            def updated_bj_under_obs(other_obs):
                """Run agent j's full belief update under this hypothetical o_other."""
                # 1 - agent_id (not int(not ...)) so agent_id may be a traced array,
                # letting callers vmap these updates over per-agent roles.
                updated_bj_probs = self.update_with_observation_and_joint_action(
                    other_belief_distribution_estimate, other_obs, joint_action, agent_id=1 - agent_id
                ).probs
                return updated_bj_probs

            def weight_of_obs(other_obs):
                """P(o_other | o_ego, a) under the current estimated belief — marginalized over s'."""
                def per_state(next_state):
                    transition_prior = jnp.sum(
                        jax.vmap(lambda s: self.joint_transition_function(s, joint_action).prob(next_state)
                                * other_belief_distribution_estimate.prob(s))(jnp.arange(self.num_unique_states))
                    )
                    joint_obs = self.joint_observation_function(next_state, joint_action)
                    joint_obs_likelihood = jax.lax.cond(
                        agent_id == 0,
                        lambda x: self.joint_factory.prob(joint_obs, ego_observation, x),
                        lambda x: self.joint_factory.prob(joint_obs, x, ego_observation),
                        other_obs,
                    )
                    ego_marginal = jax.lax.cond(
                        agent_id == 0,
                        lambda _: self.joint_factory.marginalize_var1(joint_obs).prob(ego_observation),
                        lambda _: self.joint_factory.marginalize_var2(joint_obs).prob(ego_observation),
                        None,
                    )
                    return transition_prior * jnp.nan_to_num(joint_obs_likelihood / ego_marginal)

                return jnp.sum(jax.vmap(per_state)(jnp.arange(self.num_unique_states)))

            all_obs = jnp.arange(self.num_unique_observations)
            weights = jax.vmap(weight_of_obs)(all_obs)          # (O,)
            all_probs = jax.vmap(updated_bj_under_obs)(all_obs) # (O, S)

            # weighted average of updated beliefs over o_other
            return jnp.einsum('o,os->s', weights, all_probs) * other_action_dist.prob(other_action)

        # marginalize over other agent's action
        all_actions = jnp.arange(self.num_unique_actions)
        unnorm = jnp.sum(jax.vmap(as_if_other_took_action)(all_actions), axis=0)  # (S,)
        return distrax.Categorical(probs=unnorm / jnp.sum(unnorm))
