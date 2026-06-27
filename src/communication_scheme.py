import jax
import chex
import jax.numpy as jnp
from flax import struct
from typing import Callable, Sequence


@struct.dataclass
class CommunicationScheme:
    """A fixed sequence of communication rounds for one in-game iteration.

    ``who_speaks[r, i] == 1`` means speaker i utters in round r (the listener is the
    other agent, so it is not stored). ``total_num_rounds`` is the real number of
    rounds, i.e. who_speaks's first dimension before any padding -- so when schemes
    of different lengths are stacked (see ``stack_schemes``) the padding rounds can
    be ignored on read.
    """

    who_speaks: chex.Array              # [num_rounds, num_speakers]
    total_num_rounds: chex.Array        # [int]


# Dyadic games: two speakers. A is speaker 0, B is speaker 1. A column of
# who_speaks is a speaker; who_speaks[r, i] == 1 means speaker i utters in round r
# (the listener is the other agent, so it need not be stored).
NUM_SPEAKERS = 2
SPEAKER_A, SPEAKER_B = 0, 1


def _scheme(who_speaks_rows) -> CommunicationScheme:
    """Build a CommunicationScheme from a [num_rounds, num_speakers] list of rows."""
    who_speaks = jnp.array(who_speaks_rows, dtype=jnp.int32)
    return CommunicationScheme(
        who_speaks=who_speaks,
        total_num_rounds=jnp.array(who_speaks.shape[0], dtype=jnp.int32),
    )


def a_to_b() -> CommunicationScheme:
    """A->B. One round: A speaks."""
    return _scheme([[1, 0]])


def b_to_a() -> CommunicationScheme:
    """B->A. One round: B speaks."""
    return _scheme([[0, 1]])


def a_to_b_thrice() -> CommunicationScheme:
    """A->B, A->B, A->B. Three rounds: A speaks each round."""
    return _scheme([[1, 0], [1, 0], [1, 0]])


def both_speak() -> CommunicationScheme:
    """A->B & B->A. One round: both A and B speak."""
    return _scheme([[1, 1]])


def stack_schemes(schemes: Sequence[CommunicationScheme]) -> CommunicationScheme:
    """Stack multiple schemes into one, padding who_speaks to a common round count.

    Each scheme's who_speaks [num_rounds, num_speakers] is right-padded with silent
    rounds (all-zero) up to the largest num_rounds across schemes, then stacked
    along a leading scheme axis. total_num_rounds keeps each scheme's real
    (pre-padding) length, so the padding can be ignored when a scheme is read.

    Returns a CommunicationScheme with:
        who_speaks:       [num_schemes, max_rounds, num_speakers]
        total_num_rounds: [num_schemes]
    """
    max_rounds = max(s.who_speaks.shape[0] for s in schemes)

    def pad(who_speaks):
        num_rounds, num_speakers = who_speaks.shape
        silent = jnp.zeros((max_rounds - num_rounds, num_speakers), dtype=who_speaks.dtype)
        return jnp.concatenate([who_speaks, silent], axis=0)

    return CommunicationScheme(
        who_speaks=jnp.stack([pad(s.who_speaks) for s in schemes]),
        total_num_rounds=jnp.stack([s.total_num_rounds for s in schemes]),
    )


# A CommunicationSchemeFn maps an iteration to the scheme in force at that iteration,
# analogous to RouteFn in routing.py. The `iteration` is the cumulative number of
# in-game (overhead / signification-game) iterations -- NOT the per-episode
# underlying-env step count -- so the scheme can change over the course of training.
CommunicationSchemeFn = Callable[[chex.Numeric], CommunicationScheme]


def simple_communication_scheme_fn(
    schemes_by_phase: Sequence[CommunicationScheme],
) -> CommunicationSchemeFn:
    """Build a CommunicationSchemeFn that cycles schemes by iteration.

    ``schemes_by_phase[p]`` is used whenever ``iteration % num_phases == p``. Pass
    two schemes for an odd/even split. The schemes are padded to a common round
    count (via ``stack_schemes``) so the selection works under a traced iteration;
    the returned scheme's ``total_num_rounds`` still reports its real length.

    Args:
        schemes_by_phase: One scheme per phase, selected cyclically by iteration.

    Returns:
        A ``CommunicationSchemeFn`` mapping iteration -> CommunicationScheme.
    """
    stacked = stack_schemes(schemes_by_phase)  # leading axis is the phase
    num_phases = len(schemes_by_phase)

    def scheme_fn(iteration: chex.Numeric) -> CommunicationScheme:
        phase = iteration % num_phases
        return jax.tree.map(lambda leaf: leaf[phase], stacked)

    return scheme_fn


# Convenience CommunicationSchemeFns: each accepts an iteration (like RouteFn) and
# returns a scheme, so they can be passed wherever a CommunicationSchemeFn is
# expected. The constant ones ignore the iteration.
def a_to_b_scheme_fn(iteration: chex.Numeric) -> CommunicationScheme:
    """A->B at every iteration (the iteration is ignored)."""
    return a_to_b()


def a_to_b_thrice_scheme_fn(iteration: chex.Numeric) -> CommunicationScheme:
    """A->B three times at every iteration (the iteration is ignored)."""
    return a_to_b_thrice()


def both_speak_scheme_fn(iteration: chex.Numeric) -> CommunicationScheme:
    """A->B & B->A at every iteration (the iteration is ignored)."""
    return both_speak()


def alternating_a_to_b_and_b_to_a_scheme_fn(iteration: chex.Numeric) -> CommunicationScheme:
    """Alternate A->B (even iterations) and B->A (odd iterations).

    Just the phase cycler over ``[a_to_b, b_to_a]``."""
    return simple_communication_scheme_fn([a_to_b(), b_to_a()])(iteration)


if __name__ == "__main__":
    for name, scheme in [
        ("a_to_b", a_to_b()),
        ("a_to_b_thrice", a_to_b_thrice()),
        ("both_speak", both_speak()),
    ]:
        print(f"{name}:")
        print("  who_speaks:\n", scheme.who_speaks)
        print("  total_num_rounds:", scheme.total_num_rounds)

    # Iteration-driven selection: even iterations use a_to_b, odd use a_to_b_thrice.
    # The two schemes differ in length, so stack_schemes pads the shorter one; the
    # returned scheme's total_num_rounds still reports the real (pre-padding) length.
    scheme_fn = simple_communication_scheme_fn([a_to_b(), a_to_b_thrice()])
    for iteration in range(4):
        scheme = scheme_fn(iteration)
        print(f"iteration {iteration} (phase {iteration % 2}):")
        print("  who_speaks:\n", scheme.who_speaks)
        print("  total_num_rounds:", scheme.total_num_rounds)

    # The convenience scheme fns: constant ones plus alternating A<->B.
    print("constant scheme fns at iteration 0:")
    for name, fn in [
        ("a_to_b_scheme_fn", a_to_b_scheme_fn),
        ("a_to_b_thrice_scheme_fn", a_to_b_thrice_scheme_fn),
        ("both_speak_scheme_fn", both_speak_scheme_fn),
    ]:
        print(f"  {name}: who_speaks={fn(0).who_speaks.tolist()}")

    print("alternating_a_to_b_and_b_to_a_scheme_fn:")
    for iteration in range(4):
        scheme = alternating_a_to_b_and_b_to_a_scheme_fn(iteration)
        print(f"  iteration {iteration}: who_speaks={scheme.who_speaks.tolist()}")
