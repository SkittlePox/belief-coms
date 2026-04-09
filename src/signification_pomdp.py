import jax, chex
import jax.numpy as jnp
import distrax
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from flax import struct
from guessing_game import State

# This is not as efficient as signification game because each state needs to be iterated on twice before actions in the underlying env are taken
@struct.dataclass
class AugmentedState:
    underlying_state: State     # State from GuessingGame
    sender_agent: chex.Array

    agent_0_utterance_action: chex.Array        # May eventually include an RNG key to render the utterance action
    agent_0_belief_action: chex.Array

    agent_1_utterance_action: chex.Array
    agent_1_belief_action: chex.Array

    agent_0_belief_state: distrax.Categorical
    agent_1_belief_state: distrax.Categorical

    agent_0s_estimate_of_agent_1s_belief_state: distrax.Categorical
    agent_1s_estimate_of_agent_0s_belief_state: distrax.Categorical

    message_status: chex.Array  # 0: unsent, 1: sent, 2: read



class SignificationPOMDPGuessingGame(MultiAgentEnv):
    """
    This is a POMDP centered around communicative actions.
    It accepts an underlying decision process and interleaves communication between timesteps.
    Handles belief updates automatically.
    Assumes communication only happens in one direction from sender to receiver.
    """

    def __init__(self, downstairs_env, initial_belief_distribution, belief_factory) -> None:
        super().__init__(num_agents=2)
        self.underlying_env = downstairs_env
        self.initial_belief_distribution = initial_belief_distribution
        self.belief_factory = belief_factory

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: AugmentedState):
        """
        Each agent gets their belief state I think... Should they also have the belief estimate of the other agent??
        """
        return (state.agent_0_belief_state, state.agent_0s_estimate_of_agent_1s_belief_state), (state.agent_1_belief_state, state.agent_1s_estimate_of_agent_0s_belief_state)

    def step_env(self, key: chex.PRNGKey, state: AugmentedState, actions: chex.Array):
        agent_0_actions, agent_1_actions = actions
        agent_0_utterance_action, agent_0_belief_action = agent_0_actions
        agent_1_utterance_action, agent_1_belief_action = agent_1_actions



        # If the message has been read we need to execute the action... if not we need to pass it back to the agents...

        def message_unsent(_):
            # The sender has generated an utterance to send based on its belief state and its estimate of the receiver's belief state
            # We need to add that into the state, flip the bit, and move on
            next_environment_state = AugmentedState(
                underlying_state=state.underlying_state,
                sender_agent=state.sender_agent,

                agent_0_utterance_action=agent_0_utterance_action,
                agent_0_belief_action=agent_0_belief_action,

                agent_1_utterance_action=agent_1_utterance_action,
                agent_1_belief_action=agent_1_belief_action,

                agent_0_belief_state=agent_0_belief_action,
                agent_1_belief_state=agent_1_belief_action,

                agent_0s_estimate_of_agent_1s_belief_state=state.agent_0s_estimate_of_agent_1s_belief_state,  # This should probably update but it's kind of irrelevant for this specific 1-step environment
                agent_1s_estimate_of_agent_0s_belief_state=state.agent_1s_estimate_of_agent_0s_belief_state,

                message_status=jnp.array(1)
            )
 
            return next_environment_state, self.get_obs(key, next_environment_state), (0.0, 0.0)

        def message_sent(_):
            # The utterance must be sent to the receiver agent now to be updated...
            next_environment_state = AugmentedState(
                underlying_state=state.underlying_state,
                sender_agent=state.sender_agent,

                agent_0_utterance_action=agent_0_utterance_action,
                agent_0_belief_action=agent_0_belief_action,

                agent_1_utterance_action=agent_1_utterance_action,
                agent_1_belief_action=agent_1_belief_action,

                agent_0_belief_state=agent_0_belief_action,
                agent_1_belief_state=agent_1_belief_action,

                agent_0s_estimate_of_agent_1s_belief_state=state.agent_0s_estimate_of_agent_1s_belief_state,  # This should probably update but it's kind of irrelevant for this specific 1-step environment
                agent_1s_estimate_of_agent_0s_belief_state=state.agent_1s_estimate_of_agent_0s_belief_state,

                message_status=jnp.array(1)
            )
            
        def message_read(_):
            pass

        return jax.lax.switch(state.message_status, [message_unsent, message_sent, message_read])


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        initial_underlying_state, agent_0_world_observation, agent_1_world_observation = self.underlying_env.reset(key)

        agent_0_belief_state = self.belief_factory.update_with_observation_and_joint_action(self.initial_belief_distribution, agent_0_world_observation, previous_joint_action=(-1, -1), agent_id=0)
        agent_1_belief_state = self.belief_factory.update_with_observation_and_joint_action(self.initial_belief_distribution, agent_1_world_observation, previous_joint_action=(-1, -1), agent_id=1)

        agent_0s_estimate_of_agent_1s_belief_state = self.belief_factory.update_other_belief_estimate_with_observation_only(self.initial_belief_distribution, agent_0_world_observation, 0, self.underlying_env._optimal_policy, agent_id=0)
        agent_1s_estimate_of_agent_0s_belief_state = self.belief_factory.update_other_belief_estimate_with_observation_only(self.initial_belief_distribution, agent_1_world_observation, 0, self.underlying_env._optimal_policy, agent_id=1)

        initial_environment_state = AugmentedState(
            underlying_state=initial_underlying_state,
            sender_agent=initial_underlying_state.sender_agent,

            agent_0_utterance_action=jnp.zeros(5),
            agent_0_belief_action=jnp.ones_like(self.initial_belief_distribution),

            agent_1_utterance_action=jnp.zeros(5),
            agent_1_belief_action=jnp.ones_like(self.initial_belief_distribution),

            agent_0_belief_state=agent_0_belief_state,
            agent_1_belief_state=agent_1_belief_state,

            agent_0s_estimate_of_agent_1s_belief_state=agent_0s_estimate_of_agent_1s_belief_state,
            agent_1s_estimate_of_agent_0s_belief_state=agent_1s_estimate_of_agent_0s_belief_state,

            message_status=jnp.array(0)
        )

        return (initial_environment_state, self.get_obs(key, initial_environment_state))
