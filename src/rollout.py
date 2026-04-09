import distrax
import jax.numpy as jnp
import jax
from guessing_game import GuessingGame
from belief_representations import CategoricalBeliefState

def basic_rollout():
    env = GuessingGame()

    # Make basic agents
    agent_0_policy = env._optimal_policy
    agent_1_policy = env._optimal_policy

    # Make belief state stuff
    initial_belief_state = distrax.Categorical(probs=jnp.ones(3))
    belief_factory = CategoricalBeliefState(num_unique_states=3, num_unique_observations=3, num_unique_actions=3, joint_transition_function=env._joint_transition_function, joint_observation_function=env._joint_observation_function, joint_action_constructor=env._joint_action_constructor)

    def execute_single_rollout(rng):
        env_rng, agent_0_rng, agent_1_rng, env_rng_2, next_rng = jax.random.split(rng, 5)
        env_state, agent_observations = env.reset(env_rng)
        agent_0_observation, agent_1_observation = agent_observations

        agent_0_belief = belief_factory.update_with_observation_and_joint_action(initial_belief_state, agent_0_observation, (-1, -1), agent_id=0)   # Just for initial belief update. Joint action is irrelevant here
        agent_1_belief = belief_factory.update_with_observation_and_joint_action(initial_belief_state, agent_1_observation, (-1, -1), agent_id=1)

        agent_0_action = agent_0_policy(agent_0_belief).sample(seed=agent_0_rng)
        agent_1_action = agent_1_policy(agent_1_belief).sample(seed=agent_1_rng)

        next_state, next_agent_observations, agent_rewards, is_final_state = env.step_env(env_rng_2, env_state, (agent_0_action, agent_1_action))
        return agent_rewards, next_rng

    def scan_body(rng, _):
        rewards, next_rng = execute_single_rollout(rng)
        return next_rng, rewards

    init_rng = jax.random.key(0)
    final_rng, all_rewards = jax.lax.scan(scan_body, init_rng, None, length=1000)
    return all_rewards

if __name__ == "__main__":
    all_rewards = basic_rollout()
    print(all_rewards)
    print(jnp.mean(all_rewards[0]))

