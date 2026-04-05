import jax
import jax.numpy as jnp

def evaluate_expected_returns(
    state,
    ego_policy,
    other_policy,
    ego_belief,
    other_belief,
    joint_transition_function,
    joint_reward_function,
    joint_observation_function,
    joint_action_constructor,
    belief_state_factory,
    num_unique_states,
    num_unique_observations,
    num_unique_actions,
    ego_agent_id=0,
    evaluation_depth=2,
    discount_factor=0.9
):
    # we can call this function recursively I think. Nvm it's not jax compatible.
    # Start with evaluating the current rewards
    cumulative_disc_reward = 0.0
    ego_action_dist = ego_policy(ego_belief)
    other_action_dist = other_policy(other_belief)
    
    # Over all possible actions the ego agent could take
    def as_if_ego_acts(ego_action):
        def as_if_other_acts(other_action):
            joint_action = joint_action_constructor(ego_agent_id, ego_action, other_action)
            reward = jnp.sum(joint_reward_function(state, joint_action))
            
            
            pass # Something * likelihood of other action
        return jnp.sum(jax.vmap(as_if_other_acts)(jnp.arange(num_unique_actions))) *  ego_action_dist.prob(ego_action)# likelihood of ego action
    return jnp.sum(jax.vmap(as_if_ego_acts)(jnp.arange(num_unique_actions)))
