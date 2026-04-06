import jax
import jax.numpy as jnp


class DecPOMDPModel:
    def __init__(
        self,
        joint_transition_function,
        joint_reward_function,
        joint_observation_function,
        joint_action_constructor,
        num_unique_states,
        num_unique_observations,
        num_unique_actions,
    ):
        self.joint_transition_function = joint_transition_function
        self.joint_reward_function = joint_reward_function
        self.joint_observation_function = joint_observation_function
        self.joint_action_constructor = joint_action_constructor
        self.num_unique_states = num_unique_states
        self.num_unique_observations = num_unique_observations
        self.num_unique_actions = num_unique_actions

    def evaluate_expected_returns(
        self,
        state,
        ego_policy,
        other_policy,
        ego_belief,
        other_belief,
        belief_state_factory,
        ego_agent_id=0,
        evaluation_depth=4,
        discount_factor=0.9,
    ):
        """Evaluate the ego agent's expected discounted return from a given state.

        Computes a depth-limited lookahead by recursively expanding the Bellman
        equation over joint actions, next states, and observation pairs:

            V(s, b_ego, b_other, d) =
                ∑_{a_ego} π_ego(a_ego | b_ego)
                ∑_{a_other} π_other(a_other | b_other)
                [ ∑_{s'} T(s'|s,a) R(s,a,s')
                  + γ ∑_{s'} T(s'|s,a) ∑_{o_ego,o_other} O(o|s',a)
                    · V(s', b'_ego(o_ego,a), b'_other(o_other,a), d-1) ]

        where:
          - π_ego, π_other are the agents' policies, mapping a belief to a
            distribution over actions
          - T(s'|s,a) is the joint transition model
          - R(s,a,s') is the joint reward (ego agent's reward is extracted implicitly
            via joint_reward_function)
          - O(o_ego, o_other | s', a) is the joint observation model
          - b'_ego(o_ego, a) is the ego agent's updated belief after observing o_ego
            given joint action a, computed via belief_state_factory
          - b'_other(o_other, a) is the analogous update for the other agent's belief
          - d is the remaining evaluation depth; the recursion bottoms out at d=1
            (immediate reward only)

        The observation marginalization (inner ∑_o) is unrolled as a Python loop
        over all O² observation pairs so that belief updates — which produce new
        distrax.Categorical distributions and cannot themselves be vmapped — are
        computed once per pair outside the JAX vmap over next states.

        Args:
            state: The current world state (scalar int).
            ego_policy: Callable b_ego -> distrax.Categorical over ego actions.
            other_policy: Callable b_other -> distrax.Categorical over other-agent actions.
            ego_belief: Current ego belief b_ego as a distrax.Categorical over states.
            other_belief: Current other-agent belief estimate b_other as a distrax.Categorical.
            belief_state_factory: A CategoricalBeliefState instance used to perform
                belief updates via update_with_observation_and_joint_action.
            ego_agent_id: Integer index identifying the ego agent (0 or 1). Controls
                how joint observations are indexed. Default 0.
            evaluation_depth: Number of lookahead steps. At depth 1 only the immediate
                reward is returned; deeper values add discounted future terms. Default 2.
            discount_factor: γ ∈ [0, 1], the geometric discount applied at each step.
                Default 0.9.

        Returns:
            A scalar JAX array representing the ego agent's expected discounted return.
        """
        ego_action_dist = ego_policy(ego_belief)
        other_action_dist = other_policy(other_belief)

        def as_if_ego_acts(ego_action):
            def as_if_other_acts(other_action):
                joint_action = self.joint_action_constructor(
                    ego_agent_id, ego_action, other_action
                )
                next_state_dist = self.joint_transition_function(state, joint_action)
                rewards_for_each_state = jax.vmap(
                    self.joint_reward_function, in_axes=(None, None, 0)
                )(state, joint_action, jnp.arange(self.num_unique_states))
                immediate = jnp.sum(rewards_for_each_state * next_state_dist.probs)

                if evaluation_depth <= 1:
                    return immediate * other_action_dist.prob(other_action)

                # γ ∑_{s'} T(s'|s,a) ∑_{o_ego,o_other} O(o|s',a) · V(s', b'_ego, b'_other, depth-1)
                # Beliefs are per-obs-pair (not per s'), so we loop over obs in Python and vmap over s'.
                future = jnp.zeros(())
                for ego_obs in range(self.num_unique_observations):
                    for other_obs in range(self.num_unique_observations):
                        new_ego_belief = belief_state_factory.update_with_observation_and_joint_action(
                            ego_belief, ego_obs, joint_action, agent_id=ego_agent_id
                        )
                        new_other_belief = belief_state_factory.update_with_observation_and_joint_action(
                            other_belief,
                            other_obs,
                            joint_action,
                            agent_id=int(not ego_agent_id),
                        )

                        def per_next_state(
                            next_state,
                            _ego_obs=ego_obs,
                            _other_obs=other_obs,
                            _new_ego_belief=new_ego_belief,
                            _new_other_belief=new_other_belief,
                        ):
                            joint_obs_dist = self.joint_observation_function(
                                next_state, joint_action
                            )
                            obs_prob = jax.lax.cond(
                                ego_agent_id == 0,
                                lambda _: belief_state_factory.joint_factory.prob(
                                    joint_obs_dist, _ego_obs, _other_obs
                                ),
                                lambda _: belief_state_factory.joint_factory.prob(
                                    joint_obs_dist, _other_obs, _ego_obs
                                ),
                                None,
                            )
                            future_v = self.evaluate_expected_returns(
                                next_state,
                                ego_policy,
                                other_policy,
                                _new_ego_belief,
                                _new_other_belief,
                                belief_state_factory,
                                ego_agent_id,
                                evaluation_depth - 1,
                                discount_factor,
                            )
                            return (
                                next_state_dist.prob(next_state) * obs_prob * future_v
                            )

                        future = future + jnp.sum(
                            jnp.nan_to_num(jax.vmap(per_next_state)(jnp.arange(self.num_unique_states)))
                        )

                return (immediate + discount_factor * future) * other_action_dist.prob(
                    other_action
                )

            return jnp.sum(
                jax.vmap(as_if_other_acts)(jnp.arange(self.num_unique_actions))
            ) * ego_action_dist.prob(ego_action)

        return jnp.sum(jax.vmap(as_if_ego_acts)(jnp.arange(self.num_unique_actions)))

