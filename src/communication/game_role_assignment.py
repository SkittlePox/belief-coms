import dataclasses
import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Callable, Union


@struct.dataclass
class AgentGameRoleAssignment:
    """
    An assignment of agents to games and roles for a full episode.
    Each AgentGameRoleAssignment represents an assignment of agents to a full episode in a game (with underlying_env_steps_per_episode timesteps).
    """

    game_set: chex.Array  # [num agents / 2], index i represents game i's game type index
    agent_game_assignment: chex.Array  # [num agents], index i represents agent i's assigned game (between 0 and num_agents/2)
    agent_role_assignment: chex.Array  # [num agents], index i represents agent i's assigned role in a game (either 0 or 1)
    underlying_env_steps_per_episode: (
        chex.Array
    )  # scalar int: how many underlying-DecPOMDP steps this episode runs (lockstep across all games; early-terminating games are masked, not re-routed)


# AssignmentFn is sampled once per episode; `iteration` is the episode index.
AssignmentFn = Callable[[chex.PRNGKey, int], AgentGameRoleAssignment]  # (key, iteration) -> assignment

# Dyadic games: exactly two agents (two roles) per game.
AGENTS_PER_GAME = 2


def simple_assignment_fn(
    num_agents: int = 10,
    game_type_id: int = 0,
    underlying_env_steps_per_episode: int = 10,
) -> AssignmentFn:
    """Build an AssignmentFn that randomly assigns agents to two-agent games of one type.

    Every game is of type ``game_type_id`` and holds two agents. The agents are
    randomly partitioned across ``num_agents // 2`` games, and within each game they
    are randomly given distinct roles 0 and 1.

    With ``num_agents=10`` and ``game_type_id=0`` this assigns 10 agents into 5 games
    of id 0, each with one role-0 and one role-1 agent.

    The returned AssignmentFn takes ``(key, iteration)``, where ``iteration`` is the
    episode index. This simple assigner does not let the assignment depend on the
    iteration; it re-randomizes purely from ``key``.

    Args:
        num_agents: Total number of agents to assign.
        game_type_id: The game-type index assigned to every game.
        underlying_env_steps_per_episode: Constant number of underlying-DecPOMDP steps
            each episode runs.

    Returns:
        An ``AssignmentFn`` mapping (key, iteration) -> AgentGameRoleAssignment.
    """

    def assign(key: chex.PRNGKey, iteration: int) -> AgentGameRoleAssignment:
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

        agent_game_assignment = jnp.zeros((num_agents,), dtype=jnp.int32).at[agent_for_slot].set(game_per_slot)
        agent_role_assignment = jnp.zeros((num_agents,), dtype=jnp.int32).at[agent_for_slot].set(role_per_slot)

        return AgentGameRoleAssignment(
            game_set=game_set,
            agent_game_assignment=agent_game_assignment,
            agent_role_assignment=agent_role_assignment,
            underlying_env_steps_per_episode=jnp.asarray(underlying_env_steps_per_episode, dtype=jnp.int32),
        )

    return assign


def fixed_pairs_assignment_fn(
    num_agents: int = 10,
    underlying_env_steps_per_episode: int = 10,
) -> AssignmentFn:
    """Build an AssignmentFn with a fixed, iteration-independent assignment.

    Agents are partitioned into consecutive pairs: agent i is placed in game
    ``i // 2`` with role ``i % 2`` -- games (0,1), (2,3), ... Unlike
    ``simple_assignment_fn``, the assignment never re-randomizes -- the same agents play
    the same roles in the same games every episode, regardless of ``key`` or
    ``iteration``. Every game is game type 0, so unlike simple_assignment_fn there is no
    ``game_type_id`` knob.

    Args:
        num_agents: Total number of agents to assign.
        underlying_env_steps_per_episode: Constant number of underlying-DecPOMDP steps
            per episode.

    Returns:
        An ``AssignmentFn`` that returns the same AgentGameRoleAssignment for every (key, iteration).
    """
    num_games = num_agents // AGENTS_PER_GAME

    # The assignment is fully determined by the args, so build it once and close over
    # it; `assign` ignores key and iteration entirely.
    slots = jnp.arange(num_agents)
    fixed_assignment = AgentGameRoleAssignment(
        game_set=jnp.zeros((num_games,), dtype=jnp.int32),  # all game type 0
        agent_game_assignment=(slots // AGENTS_PER_GAME).astype(jnp.int32),
        agent_role_assignment=(slots % AGENTS_PER_GAME).astype(jnp.int32),
        underlying_env_steps_per_episode=jnp.asarray(underlying_env_steps_per_episode, dtype=jnp.int32),
    )

    def assign(key: chex.PRNGKey, iteration: int) -> AgentGameRoleAssignment:
        return fixed_assignment

    return assign


# --- Assignment configs -------------------------------------------------------
# One config dataclass per assignment family, each colocated with the builder it wraps
# and exposing ``build() -> AssignmentFn``. ``AssignmentConfig`` is the union of all
# families; training.config nests it in ExperimentConfig and tyro renders a 2+-member
# union as subcommands (pick one family at the CLI). To add a family: write its builder
# above, add a frozen *AssignmentConfig dataclass with build() here, and extend the union.


@dataclasses.dataclass(frozen=True)
class SimpleAssignmentConfig:
    """Randomly assign agents to fixed-size games of a single type (simple_assignment_fn)."""

    num_agents: int = 10
    game_type_id: int = 0
    underlying_env_steps_per_episode: int = 10

    def build(self) -> AssignmentFn:
        return simple_assignment_fn(
            num_agents=self.num_agents,
            game_type_id=self.game_type_id,
            underlying_env_steps_per_episode=self.underlying_env_steps_per_episode,
        )


@dataclasses.dataclass(frozen=True)
class FixedPairsAssignmentConfig:
    """Fixed consecutive agent pairings, identical every episode (fixed_pairs_assignment_fn).

    Carries no ``game_type_id`` (every game is type 0) -- an illustration that assignment
    families need not share a parameter set.
    """

    num_agents: int = 10
    underlying_env_steps_per_episode: int = 10

    def build(self) -> AssignmentFn:
        return fixed_pairs_assignment_fn(
            num_agents=self.num_agents,
            underlying_env_steps_per_episode=self.underlying_env_steps_per_episode,
        )


# Union over assignment families. Every member must expose ``build() -> AssignmentFn``; tyro
# renders a 2+-member union as CLI subcommands (pick one family).
AssignmentConfig = Union[SimpleAssignmentConfig, FixedPairsAssignmentConfig]


if __name__ == "__main__":
    key = jax.random.key(0)

    assign_fn = simple_assignment_fn(num_agents=10, game_type_id=0, underlying_env_steps_per_episode=7)
    assignment = assign_fn(key, iteration=0)

    print("game_set:              ", assignment.game_set)
    print("agent_game_assignment: ", assignment.agent_game_assignment)
    print("agent_role_assignment: ", assignment.agent_role_assignment)
    print("env steps per episode: ", assignment.underlying_env_steps_per_episode)
    assert assignment.underlying_env_steps_per_episode == 7, "constant length returned verbatim"

    # Sanity checks: every game holds exactly one agent of each role.
    for game in range(assignment.game_set.shape[0]):
        in_game = assignment.agent_game_assignment == game
        agents = jnp.where(in_game)[0]
        roles = assignment.agent_role_assignment[in_game]
        print(f"  game {game}: agents={agents.tolist()} roles={roles.tolist()}")
        assert agents.shape[0] == 2, "each game should have 2 agents"
        assert set(roles.tolist()) == {0, 1}, "each game should have one of each role"
    print("ok: 10 agents assigned to 5 games of id 0, balanced roles, episode length prescribed over time")
