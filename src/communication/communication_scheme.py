"""Communication schemes for the StackedSignificationDecPOMDP.

A CommunicationScheme is a fixed sequence of communication rounds had between
agents in between environment timesteps. The environment is paused until all
the rounds of communication have occured.

CommunicationSchemeFn maps an in-game iteration to the scheme in force at that
iteration; the env (see stacked_signification_decpomdp.py) calls it each
underlying-env step, which is what lets a scheme change over training.

Dyadic games have two speakers: A is speaker 0, B is speaker 1.
``who_speaks[r, i] == 1`` means speaker i utters in round r (the listener is the
other agent, so it is not stored).

Design Q&A
----------
Q: Can different games run different communication schemes at the same timestep? A 
   StackedSignificationDecPOMDP can run multiple games in parallel.
A: No. ``communication_scheme_fn(iteration)`` returns a *single* CommunicationScheme
   that governs every game at that iteration. The scheme may vary *across* iterations
   (write a CommunicationSchemeFn whose output depends on ``iteration``), but within
   an iteration all games communicate identically -- there is no per-game scheme
   axis. Supporting per-game schemes would mean indexing the scheme by game (the way
   ``game_set`` indexes game type in game_role_assignment.py); we intentionally do not.
Q: Can we have different communication rounds depending on the underlying timestep of 
   the environment? E.g., so that Agent A and Agent B can take turns speaking with
   the environment changing state in between?
A: No. Not as implemented below. Each set of rounds occurs in between each env timestep.
"""

import dataclasses
import chex
import jax.numpy as jnp
from flax import struct
from typing import Callable, Literal

# Dyadic games: two speakers. A is speaker 0, B is speaker 1.
NUM_SPEAKERS = 2
SPEAKER_A, SPEAKER_B = 0, 1


@struct.dataclass
class CommunicationScheme:
    """The pattern of communication between agents that occurs in between environment timesteps.

    Agents get to communicate in between timesteps in the underlying environment.
    The objects in this class represent exactly what that pattern of communication 
    looks like. All rounds of communication happen before interaction with the environment
    is resumed.


    ``who_speaks[r, i] == 1`` means speaker i utters in round r. ``total_num_rounds``
    is who_speaks's real length (its leading dimension); it is stored explicitly so a
    scheme's length survives any downstream padding/stacking.
    """

    who_speaks: chex.Array  # [num_rounds, num_speakers]
    total_num_rounds: chex.Array  # scalar int

    @classmethod
    def from_rows(cls, rows) -> "CommunicationScheme":
        """Build a scheme from a [num_rounds, num_speakers] list of 0/1 rows."""
        who_speaks = jnp.array(rows, dtype=jnp.int32)
        return cls(who_speaks, jnp.array(who_speaks.shape[0], dtype=jnp.int32))


# The selectable schemes: name -> who_speaks rows.
_SCHEME_ROWS: dict[str, list[list[int]]] = {
    "a_to_b": [[1, 0]],  # A->B, one round
    "b_to_a": [[0, 1]],  # B->A, one round
    "a_to_b_thrice": [[1, 0], [1, 0], [1, 0]],  # A->B, three rounds
    "both_speak": [[1, 1]],  # A->B & B->A, one round
}

CommunicationSchemeName = Literal["a_to_b", "b_to_a", "a_to_b_thrice", "both_speak"]


# A CommunicationSchemeFn maps an in-game iteration to the scheme in force then,
# analogous to AssignmentFn in game_role_assignment.py. ``iteration`` is the cumulative in-game
# (signification-game) iteration count -- not the per-episode env step count. Every
# scheme below is constant (iteration-independent); for an iteration-varying scheme,
# write a CommunicationSchemeFn that branches on ``iteration``, e.g. a curriculum:
#   def curriculum_scheme_fn(iteration):
#       return jax.lax.cond(iteration < 100, a_to_b_scheme_fn, both_speak_scheme_fn, iteration)
# (the branches must return equal-shaped schemes; a_to_b and both_speak are both 1 round.)
CommunicationSchemeFn = Callable[[chex.Numeric], CommunicationScheme]


def get_scheme_fn(name: CommunicationSchemeName) -> CommunicationSchemeFn:
    """Resolve a scheme name into its (constant) CommunicationSchemeFn."""
    scheme = CommunicationScheme.from_rows(_SCHEME_ROWS[name])
    return lambda iteration: scheme


# Convenience module-level SchemeFns (imported by the env and visualizations).
a_to_b_scheme_fn = get_scheme_fn("a_to_b")
b_to_a_scheme_fn = get_scheme_fn("b_to_a")
a_to_b_thrice_scheme_fn = get_scheme_fn("a_to_b_thrice")
both_speak_scheme_fn = get_scheme_fn("both_speak")


@dataclasses.dataclass
class CommunicationConfig:
    """Which communication scheme governs the games.

    A flat name selector, not a union of families like AssignmentConfig -- communication
    is one-of-N, so there are no per-family params. Exposes the same ``build()``
    interface as the assignment configs so callers resolve both uniformly.
    """

    scheme: CommunicationSchemeName = "a_to_b"

    def build(self) -> CommunicationSchemeFn:
        return get_scheme_fn(self.scheme)


if __name__ == "__main__":
    for name in _SCHEME_ROWS:
        scheme = get_scheme_fn(name)(0)
        print(f"{name}: who_speaks={scheme.who_speaks.tolist()} " f"total_num_rounds={int(scheme.total_num_rounds)}")
