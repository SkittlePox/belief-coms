import chex
from flax import struct
from typing import Callable

@struct.dataclass
class AgentGameRoleRoute:
    """
    This is an extremely flexible representation for capturing the game role assignment for a populations among a set of games.
    The fundamental unit is the global role. Each agent has a unique global role at each time step, and each global role has a local game and game role.
    """
    agents_global_role: chex.Array      # [num agents], index i represents agent i's global role
    global_to_local_role: chex.Array    # [num agents or num global roles], index i represents global role i's role in a local game, almost always a 0 or 1
    global_to_local_game: chex.Array    # [num agents or num global roles], index i represent global role i's participating in a specific game x[i] = j

RouteFn = Callable[[chex.PRNGKey, int], AgentGameRoleRoute]
