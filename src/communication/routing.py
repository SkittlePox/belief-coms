import dataclasses
import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Callable, Union

@struct.dataclass
class AgentGameRoleRoute:
    """
    This routing class contains all the logic for what games agents are subjected to.
    Each AgentGameRoleRoute represents an assignment of agents to a full episode in a game (with underlying_env_steps_per_episode timesteps).
    """
    game_set: chex.Array                   # [num agents / 2], index i represents game i's game type index
    agent_game_assignment: chex.Array      # [num agents], index i represents agent i's assigned game (between 0 and num_agents/2)
    agent_role_assignment: chex.Array      # [num agents], index i represents agent i's assigned role in a game (either 0 or 1)
    underlying_env_steps_per_episode: chex.Array  # scalar int: how many underlying-DecPOMDP steps this episode runs (lockstep across all games; early-terminating games are masked, not re-routed)

# RouteFn is sampled once per episode; `iteration` is the episode index.
RouteFn = Callable[[chex.PRNGKey, int], AgentGameRoleRoute]  # (key, iteration) -> route

# Dyadic games: exactly two agents (two roles) per game.
AGENTS_PER_GAME = 2


def simple_routing_fn(
    num_agents: int = 10,
    game_type_id: int = 0,
    underlying_env_steps_per_episode: int = 10,
) -> RouteFn:
    """Build a RouteFn that randomly assigns agents to two-agent games of one type.

    Every game is of type ``game_type_id`` and holds two agents. The agents are
    randomly partitioned across ``num_agents // 2`` games, and within each game they
    are randomly given distinct roles 0 and 1.

    With ``num_agents=10`` and ``game_type_id=0`` this routes 10 agents into 5 games
    of id 0, each with one role-0 and one role-1 agent.

    The returned RouteFn takes ``(key, iteration)``, where ``iteration`` is the
    episode index. This simple router does not let the assignment depend on the
    iteration; it re-randomizes purely from ``key``.

    Args:
        num_agents: Total number of agents to route.
        game_type_id: The game-type index assigned to every game.
        underlying_env_steps_per_episode: Constant number of underlying-DecPOMDP steps
            each episode runs.

    Returns:
        A ``RouteFn`` mapping (key, iteration) -> AgentGameRoleRoute.
    """

    def route(key: chex.PRNGKey, iteration: int) -> AgentGameRoleRoute:
        num_games = num_agents // AGENTS_PER_GAME

        # Every game shares the same game type.
        game_set = jnp.full((num_games,), game_type_id, dtype=jnp.int32)

        # Lay out the agents into ordered (game, role) slots:
        #   slot s -> game (s // AGENTS_PER_GAME), role (s % AGENTS_PER_GAME)
        # so the slots already contain each game exactly once per role. We then
        # shuffle *which agent occupies which slot* to randomize assignments,
        # which keeps every game balanced (one agent of each role) by construction.
        slots = jnp.arange(num_agents)
        game_per_slot = slots // AGENTS_PER_GAME
        role_per_slot = slots % AGENTS_PER_GAME

        agent_for_slot = jax.random.permutation(key, num_agents)  # slot -> agent id

        agent_game_assignment = (
            jnp.zeros((num_agents,), dtype=jnp.int32).at[agent_for_slot].set(game_per_slot)
        )
        agent_role_assignment = (
            jnp.zeros((num_agents,), dtype=jnp.int32).at[agent_for_slot].set(role_per_slot)
        )

        return AgentGameRoleRoute(
            game_set=game_set,
            agent_game_assignment=agent_game_assignment,
            agent_role_assignment=agent_role_assignment,
            underlying_env_steps_per_episode=jnp.asarray(
                underlying_env_steps_per_episode, dtype=jnp.int32
            ),
        )

    return route


def fixed_pairs_routing_fn(
    num_agents: int = 10,
    underlying_env_steps_per_episode: int = 10,
) -> RouteFn:
    """Build a RouteFn with a fixed, iteration-independent assignment.

    Agents are partitioned into consecutive pairs: agent i is placed in game
    ``i // 2`` with role ``i % 2`` -- games (0,1), (2,3), ... Unlike
    ``simple_routing_fn``, the assignment never re-randomizes -- the same agents play
    the same roles in the same games every episode, regardless of ``key`` or
    ``iteration``. Every game is game type 0, so unlike simple_routing_fn there is no
    ``game_type_id`` knob.

    Args:
        num_agents: Total number of agents to route.
        underlying_env_steps_per_episode: Constant number of underlying-DecPOMDP steps
            per episode (no schedule -- the whole route is fixed across iterations).

    Returns:
        A ``RouteFn`` that returns the same AgentGameRoleRoute for every (key, iteration).
    """
    num_games = num_agents // AGENTS_PER_GAME

    # The assignment is fully determined by the args, so build it once and close over
    # it; `route` ignores key and iteration entirely.
    slots = jnp.arange(num_agents)
    fixed_route = AgentGameRoleRoute(
        game_set=jnp.zeros((num_games,), dtype=jnp.int32),               # all game type 0
        agent_game_assignment=(slots // AGENTS_PER_GAME).astype(jnp.int32),
        agent_role_assignment=(slots % AGENTS_PER_GAME).astype(jnp.int32),
        underlying_env_steps_per_episode=jnp.asarray(
            underlying_env_steps_per_episode, dtype=jnp.int32
        ),
    )

    def route(key: chex.PRNGKey, iteration: int) -> AgentGameRoleRoute:
        return fixed_route

    return route


# --- Routing configs ---------------------------------------------------------
# One config dataclass per routing family, each colocated with the builder it wraps
# and exposing ``build() -> RouteFn``. ``RoutingConfig`` is the union of all families;
# training.config nests it in ExperimentConfig and tyro renders a 2+-member union as
# subcommands (pick one family at the CLI). To add a family: write its builder above,
# add a frozen *RoutingConfig dataclass with build() here, and extend the union.


@dataclasses.dataclass(frozen=True)
class SimpleRoutingConfig:
    """Randomly assign agents to fixed-size games of a single type (simple_routing_fn)."""

    num_agents: int = 10
    game_type_id: int = 0
    underlying_env_steps_per_episode: int = 10

    def build(self) -> RouteFn:
        return simple_routing_fn(
            num_agents=self.num_agents,
            game_type_id=self.game_type_id,
            underlying_env_steps_per_episode=self.underlying_env_steps_per_episode,
        )


@dataclasses.dataclass(frozen=True)
class FixedPairsRoutingConfig:
    """Fixed consecutive agent pairings, identical every episode (fixed_pairs_routing_fn).

    Carries no ``game_type_id`` (every game is type 0) -- an illustration that routing
    families need not share a parameter set.
    """

    num_agents: int = 10
    underlying_env_steps_per_episode: int = 10

    def build(self) -> RouteFn:
        return fixed_pairs_routing_fn(
            num_agents=self.num_agents,
            underlying_env_steps_per_episode=self.underlying_env_steps_per_episode,
        )


# Union over routing families. Every member must expose ``build() -> RouteFn``; tyro
# renders a 2+-member union as CLI subcommands (pick one family).
RoutingConfig = Union[SimpleRoutingConfig, FixedPairsRoutingConfig]


if __name__ == "__main__":
    key = jax.random.key(0)

    route_fn = simple_routing_fn(
        num_agents=10, game_type_id=0, underlying_env_steps_per_episode=7
    )
    route = route_fn(key, iteration=0)

    print("game_set:              ", route.game_set)
    print("agent_game_assignment: ", route.agent_game_assignment)
    print("agent_role_assignment: ", route.agent_role_assignment)
    print("env steps per episode: ", route.underlying_env_steps_per_episode)
    assert route.underlying_env_steps_per_episode == 7, "constant length returned verbatim"

    # Sanity checks: every game holds exactly one agent of each role.
    for game in range(route.game_set.shape[0]):
        in_game = route.agent_game_assignment == game
        agents = jnp.where(in_game)[0]
        roles = route.agent_role_assignment[in_game]
        print(f"  game {game}: agents={agents.tolist()} roles={roles.tolist()}")
        assert agents.shape[0] == 2, "each game should have 2 agents"
        assert set(roles.tolist()) == {0, 1}, "each game should have one of each role"
    print("ok: 10 agents routed to 5 games of id 0, balanced roles, episode length prescribed over time")
