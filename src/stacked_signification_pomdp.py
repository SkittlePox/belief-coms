import jax, chex
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from flax import struct
from typing import Any
from functools import partial
from routing import RouteFn
from envs.guessing_game import *


@struct.dataclass
class FlexibleEnvParams:
    """Dense dynamics tensors for a (variant of a) DecPOMDP.

    Everything that distinguishes one environment variant from another lives
    here as plain arrays, so different variants are just different EnvParams
    instances *with identical shapes* — which is what keeps a future
    vmap-over-environments clean (no per-env Python branching / lax.switch).

    The conditioning below is the most general the current consumers
    (tools/model.py, tools/belief_representations.py) can actually feed, given
    what each call site has in scope:

        transition   T(s' | s, a0, a1)         shape [S, A, A, S]
        observation  O(o0, o1 | s', a0, a1)    shape [S, A, A, O0, O1]
        reward       R_i(s, a0, a1, s')        shape [N, S, A, A, S]
        num_actions
        num_states
        initial_belief_states

    Notes on the deliberate choices:
      * Observation conditions on the *next* state s' (its first axis) and on
        the joint action — never on the previous state.
      * Both agents get their own action axis (a0, a1) even when one agent's
        action is currently inert, so the shapes stay uniform across variants.
      * Reward carries a leading agent axis (N) so per-agent (non-shared)
        reward structures are expressible later.
    """

    transition: chex.Array
    observation: chex.Array
    reward: chex.Array
    num_actions: chex.Array
    num_states: chex.Array
    initial_belief_states: chex.Array   # Indexed by agent role

    
@struct.dataclass
class StackedState:
    agent_utterance_actions_unrendered = chex.Array
    agent_utterance_actions_rendered = chex.Array
    # agent_belief_action = chex.Array    # This is the new belief an agent holds 

    true_agent_belief_states = chex.Array         # These are the agents' true belief states
    other_estimated_agent_belief_states = chex.Array   # These are the estimated belief states of agents according to the agents they are engaged in games with

    global_rng_key = chex.Array

class StackedSignificationPOMDP():
    """
    """

    def __init__(self, num_agents: int, all_env_parameters: FlexibleEnvParams, routing_fn: RouteFn, communication_pattern) -> None:
        """
        Args:
            num_agents
            env_schedule_function
        """
        self.num_agents = num_agents
        self.all_env_parameters = all_env_parameters
        self.routing_fn = routing_fn
        pass

    def step_env(self, key: chex.PRNGKey, state, actions: chex.Array):
        """
        This function's job is basically just to listen to the routing function and handle all the communication processes and belief updates, etc. It actually does a lot.
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """
        NOTE: Resetting only happens once I think. The env basically continues on forever according to the routing function.
        """

        # So agents are either seeing beliefs and utterances and returning beliefs or they are seeing beliefs and something else and returning utterances.

        routing_key, key = jax.random.split(key)

        initial_route = self.routing_fn(key=routing_key, iteration=0)

        # Each agent's initial belief comes from the env parameters of the game
        agent_game_types = initial_route.game_set[initial_route.agent_game_assignment]  # [num_agents]
        agent_roles = initial_route.agent_role_assignment                              # [num_agents]
        agent_initial_belief_states = self.all_env_parameters.initial_belief_states[
            agent_game_types, agent_roles
        ]  # [num_agents, *belief_shape]

        # Each agent's initial estimate of the OTHER role's belief in the same game.
        # (Assumes 2 roles per game, so "the other role" is 1 - role.)
        other_roles = 1 - agent_roles
        est_other_initial_belief_states = self.all_env_parameters.initial_belief_states[
            agent_game_types, other_roles
        ]  # [num_agents, *belief_shape]

        

        # initial_environment_state = FlexibleGuessingGameState(
        #     ego_belief_states=agent_initial_belief_states,
        #     est_other_belief_states=est_other_initial_belief_states,
        #     agent_game_id=initial_route.agent_game_assignment,
        #     agent_role_id=agent_roles,
        #     game_states=jnp.ones(1)
        # )

        # return (initial_environment_state, self.get_obs(key, initial_environment_state))

        return None


if __name__ == "__main__":
    # env = StackedSignificationPOMDP(num_agents=10, all_env_parameters=)
    pass


