import jax, chex
import copy
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from belief_representations import *


@struct.dataclass
class State:    # Underlying state of the decision-process
    agent_0_is_sending: chex.Array
    agent_1_is_sending: chex.Array

    sender_curtain_up: chex.Array
    receiver_curtain_up: chex.Array

    observation_behind_sender_curtain: chex.Array
    observation_behind_receiver_curtain: chex.Array

    optimal_receiver_action: chex.Array

    # +----------------------------+
    # |              ░             |
    # |   Agent 0    ░    +-----+  |
    # |   [Sender]   ░    |  C  |  |
    # |              ░    +-----+  |
    # |              ░             |
    # |               ---- Wall ---|
    # |              ░             |
    # |   Agent 1    ░    +-----+  |
    # |  [Receiver]  ░    |  A  |  |
    # |              ░    +-----+  |
    # |              ░             |
    # |       *                    |
    # |  [A] [B] [C]               |
    # +----------------------------+
    #   ░ = curtain  * = optimal receiver action

    # Each agent can see the other agent, and when the
    # curtain is up they can see their corresponding observation
    # (e.g. Agent 1 can see 'A' and Agent 0 and see 'C')

    # At t = 0, agents can only see each other and whether they are a sender or receiver
    #               actions do nothing on this step
    # At t = 1, the curtains lift and the agents can see their corresponding observations
    #               the sender can now take an action (a drawing)
    # At t = 2, the receiver can see the sender's action (a drawing)
    #               the receiver can now take a discrete action in (A, B, C)
    # At t = 3, the episode ends, agents are rewarded based on the action that the receiver took.
    #               1 if correct action, -0.1 otherwise


class SimplePOMDP(MultiAgentEnv):
    def __init__(self):
        super().__init__(num_agents=2)
        with open("ascii_simple_pomdp.txt") as f:
            self.template = f.read()

    @partial(jax.jit, static_argnums=(0,))  # This is safe because 'self' is never used in this function
    def get_obs(self, key: chex.PRNGKey, state: State):
        # The agent must observe whether it is a sender or receiver, and receiver the actual observations from the other agent and from behind the curtain.
        sender_obs = jax.lax.cond(state.sender_curtain_up, lambda _: state.observation_behind_sender_curtain, lambda _: jnp.array(-1), None)        # An observation of -1 means that you are the sender
        receiver_obs = jax.lax.cond(state.receiver_curtain_up, lambda _: state.observation_behind_receiver_curtain, lambda _: jnp.array(-2), None)  # An observation of -2 means that you are the receiver
        agent_obs = jax.lax.cond(state.agent_0_is_sending, lambda _: (sender_obs, receiver_obs), lambda _: (receiver_obs, sender_obs), None)
        return agent_obs    # This returns a tuple of observations corresponding to (agent_0, agent_1)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: State, actions: chex.Array):
        receiver_action = jax.lax.cond(jnp.logical_not(state.agent_0_is_sending), lambda _: actions[0], lambda _: actions[1], None)    # I could probably also do this using actions.at[state.agent_1_is_sending]
        # If the receiver agent takes the optimal action then there's a reward, 
        agent_rewards = jax.lax.cond(state.receiver_curtain_up, lambda _: jnp.ones(2) * (receiver_action == state.optimal_receiver_action), lambda _: jnp.array([0, 0], dtype=jnp.float32), None)
        agent_dones = agent_rewards

        next_environment_state = State(
            agent_0_is_sending=state.agent_0_is_sending,
            agent_1_is_sending=state.agent_1_is_sending,
            sender_curtain_up=jnp.array(1),
            receiver_curtain_up=jnp.array(1),
            observation_behind_sender_curtain=state.observation_behind_sender_curtain,
            observation_behind_receiver_curtain=state.observation_behind_receiver_curtain,
            optimal_receiver_action=state.optimal_receiver_action
        )
        
        return next_environment_state, self.get_obs(key, next_environment_state), agent_rewards, agent_dones
    
    @partial(jax.jit, static_argnums=(0,))  # We don't need to cache the SimplePOMDP object so static_argnums=(0,)
    def reset(self, key: chex.PRNGKey, optimal_receiver_action_prescription=jnp.array(-1), curtain_observation_prescription=jnp.array([-1, -1]), agent_role_prescription=jnp.array([-1, -1])):
        # The optimal action is selected at random if not prescribed. One of three possible discrete actions.
        optimal_receiver_action_key, key = jax.random.split(key)
        def pick_random_action(_):
            return jax.random.randint(optimal_receiver_action_key, (1), 0, 3)[0]  # Choose a value between 0 and 2 inclusive, corresponding to A, B, C
        optimal_receiver_action = jax.lax.cond(optimal_receiver_action_prescription == jnp.array(-1), pick_random_action, lambda _: optimal_receiver_action_prescription, None)
        
        # The remaining observations are randomly assigned to the sender and receiver roles if they aren't prescribed
        remaining_curtain_observation_key, key = jax.random.split(key)
        def pick_curtain_observations(_):
            return jax.random.permutation(remaining_curtain_observation_key, jnp.delete(jnp.arange(3), optimal_receiver_action, assume_unique_indices=True)) # Shuffle the observation array after removing the optimal action
        agent_observations = jax.lax.cond(jnp.array_equal(curtain_observation_prescription, jnp.array([-1, -1])), pick_curtain_observations, lambda _: curtain_observation_prescription, None)
        sender_curtain_observation, receiver_curtain_observation = agent_observations[0], agent_observations[1]

        # Finally, assign the agents a sender/receiver role if they aren't prescribed
        agent_role_assignment_key, key = jax.random.split(key)
        def pick_agent_roles(_):
            return jax.random.permutation(agent_role_assignment_key, jnp.arange(2))
        agent_roles = jax.lax.cond(jnp.array_equal(agent_role_prescription, jnp.array([-1, -1])), pick_agent_roles, lambda _: curtain_observation_prescription, None)
        agent_0_is_sending, agent_1_is_sending = agent_roles[0], agent_roles[1]

        initial_environment_state = State(
            agent_0_is_sending=agent_0_is_sending,
            agent_1_is_sending=agent_1_is_sending,
            sender_curtain_up=jnp.array(0),
            receiver_curtain_up=jnp.array(0),
            observation_behind_sender_curtain=sender_curtain_observation,
            observation_behind_receiver_curtain=receiver_curtain_observation,
            optimal_receiver_action=optimal_receiver_action
        )
        
        return (initial_environment_state, self.get_obs(key, initial_environment_state)) # This returns a tuple of the observations and the environment state
    
    # State, Action -> Next State
    def abstract_transition_function(self, state_num, action_num) -> distrax.Categorical:\
        # The underlying state of the world actually doesn't change I think...
        return distrax.Categorical(probs=jnp.zeros(3).at[state_num].set(1))
    
    def abstract_observation_function(self, state_num, action_num) -> distrax.Categorical:
        obbs = jnp.ones(3) * 0.5
        obbs = obbs.at[state_num].set(0)
        return distrax.Categorical(probs=obbs)

    def ascii_state(self, state: State):
        state_visual = copy.deepcopy(self.template)
        state_visual = state_visual.replace("!", "0" if state.agent_0_is_sending else "1")
        state_visual = state_visual.replace("#", "1" if state.agent_0_is_sending else "0")

        observation_list = ["A", "B", "C"]
        state_visual = state_visual.replace("@", observation_list[state.observation_behind_sender_curtain])
        state_visual = state_visual.replace("$", observation_list[state.observation_behind_receiver_curtain])

        if state.sender_curtain_up:
            state_visual = state_visual.replace("░", " ")
        
        its_A = " *         "
        its_B = "     *     "
        its_C = "         * "

        state_visual = state_visual.replace("%", [its_A, its_B, its_C][state.optimal_receiver_action])
        
        print(state_visual)


if __name__ == '__main__':
    env = SimplePOMDP()
    env_state, observations = env.reset(jax.random.key(8))
    env.ascii_state(env_state)
    print(observations)
    # print(env.reset(jax.random.key(5)))
    # print(env.reset(jax.random.key(6)))
    # print(env.reset(jax.random.key(7)))
    # print(env.reset(jax.random.key(8)))
    # print(env.reset(jax.random.key(9)))
    # print(env.reset(jax.random.key(10)))

    agent_0_belief = CategoricalBeliefState(3)
    agent_1_belief = CategoricalBeliefState(3)
    print(agent_0_belief.belief_distribution.probs)

    env_state, observations, rewards, dones = env.step_env(jax.random.key(10), env_state, jnp.array([0, 0]))
    env.ascii_state(env_state)

    print(observations, rewards, dones)

    agent_0_belief.update(observations[0], 0, env.abstract_transition_function, env.abstract_observation_function)
    print(agent_0_belief.belief_distribution.probs)


