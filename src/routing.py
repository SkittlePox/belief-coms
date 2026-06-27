import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Callable, Union

@struct.dataclass
class AgentGameRoleRoute:
    """
    This routing class contains all the logic for what games agents are subjected to.
    """
    game_set: chex.Array                   # [num agents / 2], index i represents game i's game type index
    agent_game_assignment: chex.Array      # [num agents], index i represents agent i's assigned game (between 0 and num_agents/2)
    agent_role_assignment: chex.Array      # [num agents], index i represents agent i's assigned role in a game (either 0 or 1)
    horizon: chex.Array                    # scalar int: number of underlying-DecPOMDP steps this episode runs (lockstep across all games; early-terminating games are masked, not re-routed)

# RouteFn is sampled once per episode; `iteration` is the episode index, which is
# what lets the route prescribe a different horizon (and assignment) over time.
RouteFn = Callable[[chex.PRNGKey, int], AgentGameRoleRoute]  # (key, iteration) -> route


def simple_routing_fn(
    num_agents: int = 10,
    game_type_id: int = 0,
    agents_per_game: int = 2,
    horizon: Union[int, Callable[[chex.Numeric], chex.Numeric]] = 10,
) -> RouteFn:
    """Build a RouteFn that randomly assigns agents to fixed-size games of one type.

    Every game is of type ``game_type_id`` and holds exactly ``agents_per_game``
    agents. The agents are randomly partitioned across ``num_agents //
    agents_per_game`` games, and within each game they are randomly given distinct
    roles ``0 .. agents_per_game - 1``.

    With ``num_agents=10``, ``agents_per_game=2`` and ``game_type_id=0`` this
    routes 10 agents into 5 games of id 0, each with one role-0 and one role-1
    agent.

    The returned RouteFn takes ``(key, iteration)``, where ``iteration`` is the
    episode index. This simple router does not let the *assignment* depend on the
    iteration (it re-randomizes purely from ``key``), but the iteration drives the
    ``horizon`` schedule, so H can be prescribed over time.

    Args:
        num_agents: Total number of agents to route.
        game_type_id: The game-type index assigned to every game.
        agents_per_game: Number of (distinct) roles / agents per game.
        horizon: Episode length H, in underlying-DecPOMDP steps. Either a constant
            int (same H every episode) or a callable ``iteration -> H`` (a schedule,
            e.g. a curriculum that lengthens episodes over training). May return a
            traced value; it is cast to an int32 scalar in the route.

    Returns:
        A ``RouteFn`` mapping (key, iteration) -> AgentGameRoleRoute.
    """

    # Normalize a constant int into a (trivial) schedule so `route` has one path.
    horizon_fn = horizon if callable(horizon) else (lambda iteration: horizon)

    def route(key: chex.PRNGKey, iteration: int) -> AgentGameRoleRoute:
        num_games = num_agents // agents_per_game

        # Every game shares the same game type.
        game_set = jnp.full((num_games,), game_type_id, dtype=jnp.int32)

        # Lay out the agents into ordered (game, role) slots:
        #   slot s -> game (s // agents_per_game), role (s % agents_per_game)
        # so the slots already contain each game exactly once per role. We then
        # shuffle *which agent occupies which slot* to randomize assignments,
        # which keeps every game balanced (one agent of each role) by construction.
        slots = jnp.arange(num_agents)
        game_per_slot = slots // agents_per_game
        role_per_slot = slots % agents_per_game

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
            horizon=jnp.asarray(horizon_fn(iteration), dtype=jnp.int32),
        )

    return route


if __name__ == "__main__":
    key = jax.random.key(0)

    route_fn = simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2, horizon=7)
    route = route_fn(key, iteration=0)

    print("game_set:              ", route.game_set)
    print("agent_game_assignment: ", route.agent_game_assignment)
    print("agent_role_assignment: ", route.agent_role_assignment)
    print("horizon:               ", route.horizon)
    assert route.horizon == 7, "constant horizon should be returned verbatim"

    # Scheduled horizon: H grows with the episode index (a simple curriculum).
    sched_fn = simple_routing_fn(
        num_agents=10, agents_per_game=2, horizon=lambda iteration: 5 + 2 * iteration
    )
    horizons = [int(sched_fn(key, iteration=it).horizon) for it in range(4)]
    print("scheduled horizons:    ", horizons)
    assert horizons == [5, 7, 9, 11], "horizon schedule should track the episode index"

    # Sanity checks: every game holds exactly one agent of each role.
    for game in range(route.game_set.shape[0]):
        in_game = route.agent_game_assignment == game
        agents = jnp.where(in_game)[0]
        roles = route.agent_role_assignment[in_game]
        print(f"  game {game}: agents={agents.tolist()} roles={roles.tolist()}")
        assert agents.shape[0] == 2, "each game should have 2 agents"
        assert set(roles.tolist()) == {0, 1}, "each game should have one of each role"
    print("ok: 10 agents routed to 5 games of id 0, balanced roles, horizon prescribed over time")
