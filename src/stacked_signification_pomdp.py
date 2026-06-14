import jax, chex
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from flax import struct
from typing import Any, Callable
from functools import partial

@struct.dataclass
class StackedState:
    """
    Full state for stacked SignificationPOMDP
    """

    augmented_pomdp_states: Any  # An array of AugmentedStates, each one corresponding to a significationPOMDP

    sender_agent_channel_assignment: chex.Array
    receiver_agent_channel_assignment: chex.Array

    sender_network_ego_belief_observations: chex.Array
    sender_network_other_belief_estimate_observations: chex.Array

    receiver_network_utterance_observations: chex.Array   # These should be utterances
    receiver_network_belief_observations: chex.Array

    sender_network_rewards: chex.Array
    receiver_network_rewards: chex.Array

    sender_agent_alive: int
    receiver_agent_alive: int

    iteration: int

class StackedSignificationPOMDP(MultiAgentEnv):
    """
    This class has multiple agents that are passed to a series of SignificationPOMDPs

    This class assigns agents to games according to a scheduler
    """

    def __init__(self, num_agents: int, env_maplist: Callable, env_schedule_function: Callable, ) -> None:
        """
        Args:
            num_agents
            env_schedule_function
        """
        super().__init__(num_agents)

        # self.underlying_sigpomdp = underlying_sigpomdp


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):

        # Assign senders and receivers to channels
        k1, key = jax.random.split(key)
        shuffled_agents = jax.random.permutation(k1, self.num_agents)
        channel_map = shuffled_agents.reshape((self.num_agents/2, 2))


        initial_environment_state = StackedState(
            augmented_pomdp_states=
        )
        return initial_environment_state, self.get_obs(key, initial_environment_state)

