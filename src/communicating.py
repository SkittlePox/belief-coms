import chex
from flax import struct
from typing import Callable

@struct.dataclass
class AgentGameCommunicationScheme:
    """
    This routing class contains all the logic for what games agents are subjected to.
    """
    game_set: chex.Array                   # [num agents / 2], index i represents game i's game type index
    agent_game_assignment: chex.Array      # [num agents], index i represents agent i's assigned game (between 0 and num_agents/2)
    agent_role_assignment: chex.Array      # [num agents], index i represents agent i's assigned role in a game (either 0 or 1)

RouteFn = Callable[[chex.PRNGKey, int], AgentGameCommunicationScheme]
