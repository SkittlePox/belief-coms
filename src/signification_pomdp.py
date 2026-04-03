from mimetypes import init
import jax, chex
import copy
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from belief_representations import *
from distributions import *

@struct.dataclass
class State:
    sender_agent: chex.Array            # Either a 0 or 1 corresponding to agent 0 or agent 1

    sender_signal_image: chex.Array     # The rendered image shown to the receiver
    sender_signal_action: chex.Array    # The action the sender took (which will be rendered and shown to receiver)
    
    agent_0_world_observation: chex.Array   # This is an integer corresponding to an observation
    agent_1_world_observation: chex.Array

    optimal_receiver_action: chex.Array     # This is an integer corresponding to the optimal receiver action

    final_state: chex.Array


class ImageSigPOMDP(MultiAgentEnv):
    def __init__(self) -> None:
        super().__init__(num_agents=2)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: State):
        # There are only three possible discrete observations, plus the sender_signal_image is sent to the receiver
        return state.agent_0_world_observation, state.agent_1_world_observation, state.sender_signal_image
    
    def step_env(self, key: chex.PRNGKey, state: State, actions: chex.Array):
        # Actions are sent in the order (receiver action, sender action) and are shape ((1), (28))
        receiver_action, sender_action = actions
        
        # 3 is the wait action!
        agent_reward = jax.lax.cond(receiver_action == 3, lambda _: -0.1, jax.lax.cond(receiver_action == state.optimal_receiver_action, lambda _: 1.0, lambda _: -0.1, None), None)

        next_environment_state = State(
            sender_agent=state.sender_agent,

            sender_signal_image=jnp.zeros(32, 32),  # We need to render the speaker action into an image here.
            sender_signal_action=sender_action,

            agent_0_world_observation=state.agent_0_world_observation,
            agent_1_world_observation=state.agent_1_world_observation,

            optimal_receiver_action=state.optimal_receiver_action,

            final_state=agent_reward == 1.0,
        )
        
        return next_environment_state, self.get_obs(key, next_environment_state, (agent_reward, agent_reward), next_environment_state.final_state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, sender_agent=jnp.array(-1), agent_world_observations_and_optimal_action=jnp.array([-1, -1, -1])):
        # Agent role selection (sender or receiver)
        sender_agent_key, key = jax.random.split(key)
        sender_agent = jax.lax.cond(sender_agent == jnp.array(-1), lambda _: jax.random.randint(sender_agent_key, (1), 0, 2)[0], lambda _: sender_agent, None) # Choose a value of shape (1) between 0 and 1 inclusive, corresponding to agent 0 or agent 1

        # Agent world observation selection (A, B, or C, corresponding to 0, 1, or 2)
        observation_key, key = jax.random.split(key)
        agent_world_observations_and_optimal_action = jax.lax.cond(jnp.array_equal(agent_world_observations_and_optimal_action, jnp.array([-1, -1, -1])), lambda _: jax.random.permutation(observation_key, jnp.arange(3)), lambda _: agent_world_observations_and_optimal_action, None)

        initial_environment_state = State(
            sender_agent=sender_agent,

            sender_signal_image=jnp.zeros((32, 32)),
            sender_signal_action=jnp.zeros(28),

            agent_0_world_observation=agent_world_observations_and_optimal_action[0],
            agent_1_world_observation=agent_world_observations_and_optimal_action[1],

            optimal_receiver_action=agent_world_observations_and_optimal_action[2],
            
            final_state=jnp.array(0)
        )

        return (initial_environment_state, self.get_obs(key, initial_environment_state))

    @partial(jax.jit, static_argnums=(0,))
    def _joint_transition_function(self, state_num, joint_action):
        """ There are only 5 possible states. The initial state of 3 where nothing can happen, the final state 4 where you get reward, and then the states 0-2 corresponding to optimal actions 0-2
        """
        # If you are in state 3, the initial state, there is a 1/3 chance of transition of states 0-2
        # If you are in states 0-2, depending on the receiver action you will either stay in the same state or transition to the final state (4)

        # This function should be represented as T(s'|s, a)

        (sender_action, receiver_action) = joint_action

        def state_0_1_2():
            idx = jax.lax.cond(state_num == receiver_action, lambda _: 4, lambda _: state_num, None)
            probs = jnp.zeros(5).at[idx].set(1.0)
            return probs

        def state_3():
            probs = jnp.zeros(5).at[0:3].set([1.0, 1.0, 1.0])
            return probs/3

        def state_4():
            return jnp.zeros(5)

        probs = jax.lax.switch(state_num, [state_0_1_2, state_0_1_2, state_0_1_2, state_3, state_4])

        return distrax.Categorical(probs=probs)

    @partial(jax.jit, static_argnums=(0,))
    def _joint_observation_function(self, state_num, joint_action):
        """ There are only 3 environment observations: A, B, C corresponding to 0, 1, 2
        """
        # This function should be represented as O(o1, o2|s), a 2D categorical distribution, dim 3
        #                       Agent 0
        #                  A       B       C
        #              +-------+-------+-------+
        #            A | (0,0) | (0,1) | (0,2) |
        #              +-------+-------+-------+
        #  Agent 1   B | (1,0) | (1,1) | (1,2) |
        #              +-------+-------+-------+
        #            C | (2,0) | (2,1) | (2,2) |
        #              +-------+-------+-------+
        #                 (agent 0, agent 1)
        
        # 3 by 3 grid = 9 probabilities
        

        def state_0():
            return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0])
        
        def state_1():
            return jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        
        def state_2():
            return jnp.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        probs = jax.lax.switch(state_num, [state_0, state_1, state_2])  # These are the only states that matter I think...
        return distrax.Categorical(probs=probs)

    def _joint_action_constructor(self, agent_id, ego_action, other_action):
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None
        )


def optimal_policy(belief_distribution: distrax.Categorical):
    """ Returns a probability distribution over possible actions in the DecPOMDP.
    belief_distribution is a categorical distrax distribution over all possible states. (Assuming 3 for simplicity)
    """
    # I'm not sure how to write this for arbitrary distributions...
    # The funny thing is I think in this case it's actually literally just the belief distribution! The optimal action distribution is the same as the belief distribution here...
    return belief_distribution
        

if __name__ == '__main__':
    env = ImageSigPOMDP()
    key = jax.random.key(10)
    env_state, observations = env.reset(key)
    print(env_state)


    ### Transition function tests
    # Actions do nothing in initial state. 1/3 chance of ending up in 0, 1, or 2
    print(env._joint_transition_function(3, (-1, 0)).probs)
    print(env._joint_transition_function(3, (-1, 1)).probs)

    # In states 0, 1, 2 you either stay in your current state or get to the end state
    print(env._joint_transition_function(0, (-1, 0)).probs)
    print(env._joint_transition_function(0, (-1, 1)).probs)

    # Nothing happens in the end state
    print(env._joint_transition_function(4, (-1, 0)).probs)

    ### Observation function tests
    factory = JointCategoricalPair(vars_num_categories=(3, 3))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(0, 0)))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(1, 0)))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(2, 0)))


    ### Belief update test
    initial_belief = distrax.Categorical(probs=jnp.zeros(5).at[0:3].set([1.0, 1.0, 1.0])/3)
    print(initial_belief.probs)

    belief_factory = CategoricalBeliefState(num_unique_states=3, num_unique_observations=3, num_unique_actions=4, joint_transition_function=env._joint_transition_function, joint_observation_function=env._joint_observation_function, joint_action_constructor=env._joint_action_constructor)

    new_belief = belief_factory.update_with_observation_and_joint_action(initial_belief, 1, (-1, 3))
    print(new_belief.probs)

    uniform_belief = distrax.Categorical(probs=jnp.ones(3))
    
    # their_beliefs = belief_factory.update_with_observation(uniform_belief, uniform_belief, 1, 4, optimal_policy)

    print(initial_belief.probs)
    their_beliefs = belief_factory.update_other_belief_estimate_with_observation_only(initial_belief, 0, 3, optimal_policy) # Prob for index 1 should be non-zero!

    print(their_beliefs.probs)
