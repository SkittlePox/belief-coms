import jax, chex
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:    # Underlying state of the decision-process
    agent_0_is_sending: chex.Array
    agent_1_is_sending: chex.Array

    sender_curtain_up: chex.Array
    receiver_curtain_up: chex.Array

    observation_behind_sender_curtain: chex.Array
    observation_behind_receiver_curtain: chex.Array

    optimal_action: chex.Array

    # +----------------------------+
    # |              ░             |
    # |   Agent 0    ░    +-----+  |
    # |  [Speaker]   ░    |  C  |  |
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
    #   ░ = curtain  * = optimal action

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

    def get_obs(self, key: chex.PRNGKey, state: State):
        # The agent must observe whether it is a sender or receiver
        pass

    def step_env(self, key: chex.PRNGKey, state: State, actions):
        pass

    def reset(self, key: chex.PRNGKey, optimal_action_prescription=None, curtain_observation_prescription=None, agent_role_prescription=None):
        # The optimal action is selected at random if not prescribed. One of three possible discrete actions.
        if optimal_action_prescription:
            optimal_action = optimal_action_prescription
        else:
            optimal_action_key, key = jax.random.split(key)
            optimal_action = jax.random.randint(optimal_action_key, (1), 0, 3)
        
        # The remaining observations are randomly assigned to the sender and receiver roles if they aren't prescribed
        if curtain_observation_prescription:
            sender_curtain_observation, receiver_curtain_observation = curtain_observation_prescription
        else:
            remaining_curtain_observations = [0, 1, 2]
            remaining_curtain_observations.remove(optimal_action)
            remaining_curtain_observation_key, key = jax.random.split(key)
            sender_curtain_observation_idx = jax.random.randint(remaining_curtain_observation_key, (1), 0, 2)
            sender_curtain_observation = remaining_curtain_observations[sender_curtain_observation_idx.item()]
            receiver_curtain_observation = remaining_curtain_observations[1-sender_curtain_observation_idx.item()]

        # Finally, assign the agents a sender/receiver role if they aren't prescribed
        if agent_role_prescription:
            agent_0_is_sending, agent_1_is_sending = agent_role_prescription
        else:
            agent_role_assignment_key, key = jax.random.split(key)
            agent_0_is_sending = jax.random.randint(agent_role_assignment_key, (1), 0, 2)
            agent_1_is_sending = 1 - agent_0_is_sending

        initial_environment_state = State(
            agent_0_is_sending=agent_0_is_sending,
            agent_1_is_sending=agent_1_is_sending,
            sender_curtain_up=jnp.array([0]),
            receiver_curtain_up=jnp.array([0]),
            observation_behind_sender_curtain=jnp.array([sender_curtain_observation]),
            observation_behind_receiver_curtain=jnp.array([receiver_curtain_observation]),
            optimal_action=optimal_action
        )
        
        return initial_environment_state # This returns a tuple of the observations and the environment state

if __name__ == '__main__':
    env = SimplePOMDP()
    env_state = env.reset(jax.random.key(5))
    print(env_state)
