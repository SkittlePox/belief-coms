import jax, chex
import jax.numpy as jnp
import distrax
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from flax import struct
from typing import Any, Callable, Sequence
from functools import partial
from routing import RouteFn
# FlexibleEnvParams / OptimalPolicy are defined in envs.flexible_env (the leaf of
# the env-definition dependency graph) and consumed here.
from envs.flexible_env import FlexibleEnvParams, OptimalPolicy


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

    def __init__(self, num_agents: int, all_env_parameters: FlexibleEnvParams, optimal_policies: Sequence[Sequence[OptimalPolicy]], routing_fn: RouteFn, communication_pattern, skip_first_communication_step: bool) -> None:
        """
        Args:
            num_agents
            all_env_parameters: Stacked FlexibleEnvParams, indexed by game type along the leading axis.
            optimal_policies: Nested table of optimal policies indexed [game_type][role].
                Each entry is an arbitrary callable Categorical(belief) -> Categorical(action).
                Stored separately from all_env_parameters because callables are not
                traceable pytree data and cannot be gathered by a traced index.
            routing_fn
        """
        self.num_agents = num_agents
        self.all_env_parameters = all_env_parameters
        self.routing_fn = routing_fn
        self.skip_first_communication_step = skip_first_communication_step

        # Policy table indexed [game_type][role]. Flatten once for lax.switch
        # dispatch: the table is static (Python-level), so the flat list of
        # callables is available at trace time even though the (game_type, role)
        # selection index is traced.
        self.optimal_policies = optimal_policies
        self.num_game_types = len(optimal_policies)
        self.num_roles = len(optimal_policies[0])
        self._flat_policies = tuple(
            optimal_policies[game_type][role]
            for game_type in range(self.num_game_types)
            for role in range(self.num_roles)
        )

    def _agent_policy(self, game_type, role, belief_distribution: distrax.Categorical) -> distrax.Categorical:
        """Select and apply the optimal policy for a single (game_type, role).

        `game_type` and `role` may be traced; dispatch goes through lax.switch over
        the flattened [game_type * num_roles + role] index. All policies must accept
        the same belief shape and return the same action-distribution shape (pad /
        mask actions across game types if they differ).
        """
        flat_index = game_type * self.num_roles + role
        return jax.lax.switch(flat_index, self._flat_policies, belief_distribution)

    def agent_action_distributions(self, agent_game_types, agent_roles, agent_belief_probs):
        """Vectorized optimal action distribution for every agent.

        Args:
            agent_game_types: [num_agents] game-type index for each agent.
            agent_roles:      [num_agents] role index for each agent.
            agent_belief_probs: [num_agents, num_states] each agent's belief as raw probs.

        Returns:
            [num_agents, num_actions] each agent's action-distribution probs.
        """
        def one_agent(game_type, role, belief_probs):
            belief = distrax.Categorical(probs=belief_probs)
            return self._agent_policy(game_type, role, belief).probs

        return jax.vmap(one_agent)(agent_game_types, agent_roles, agent_belief_probs)

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

        # Each agent's optimal action distribution under its game-type/role policy,
        # given its initial belief. (The other-role policy is available the same way
        # via agent_action_distributions(agent_game_types, other_roles, ...), e.g. for
        # modelling the other agent in belief updates.)
        agent_initial_action_distributions = self.agent_action_distributions(
            agent_game_types, agent_roles, agent_initial_belief_states
        )  # [num_agents, num_actions]
        # TODO: thread beliefs / action distributions into the stacked state.

        if self.skip_first_communication_step:
            # We want the agents to move forward with a single joint action before invoking communication
            pass
        else:
            # We want the agents to communicate before stepping ahead in the environment
            pass



        return None


if __name__ == "__main__":
    from routing import simple_routing_fn
    from envs.factory import assemble_environments, guessing_game_spec

    # Build the stacked params + policy table from one game type (the guessing game).
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])

    env = StackedSignificationPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
        communication_pattern=None,
        skip_first_communication_step=False,
    )

    # reset compiles & runs the belief gather + policy vmap (returns None for now).
    env.reset(jax.random.key(0))

    # Direct dispatch check: each agent's belief is routed through its
    # (game_type, role) policy. Both guessing-game roles are identity here, so we
    # confirm each agent gets its OWN belief back through the correct branch.
    belief_role_0 = jnp.array([0.6, 0.3, 0.1, 0.0])
    belief_role_1 = jnp.array([0.2, 0.2, 0.6, 0.0])
    dists = env.agent_action_distributions(
        agent_game_types=jnp.array([0, 0]),
        agent_roles=jnp.array([0, 1]),
        agent_belief_probs=jnp.stack([belief_role_0, belief_role_1]),
    )
    print("role 0 action probs:", dists[0])
    print("role 1 action probs:", dists[1])
    assert jnp.allclose(dists[0], belief_role_0)
    assert jnp.allclose(dists[1], belief_role_1)
    print("ok: env built via factory; policies dispatch by (game_type, role)")


