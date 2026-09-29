"""Sanity checks for tests/stacked_harness.py itself (not the env's behavior).

Run with::

    uv run pytest tests/test_stacked_harness.py
"""

import numpy as np
import pytest

from stacked_harness import Harness, padded_scheme_fn, partners, schedule, sentinel_utterances


def test_reset_shapes_and_fixed_pairs():
    h = Harness.build(num_agents=4)
    state, (beliefs, estimates, utterances) = h.reset(seed=0)
    S = h.env.belief_dim
    assert beliefs.shape == estimates.shape == (4, S)
    assert utterances.shape == (4, h.env.utterance_action_dim)
    # fixed_pairs: agent i -> game i // 2, role i % 2, so partners are (0,1), (2,3).
    np.testing.assert_array_equal(partners(h.env, state), [1, 0, 3, 2])
    assert schedule(state) == (0, 0, 0, 0, 0, 0)


def test_sentinel_rows_are_distinct_and_nonzero():
    u = np.asarray(sentinel_utterances(4, 3))
    assert u.shape == (4, 3)
    np.testing.assert_array_equal(u[:, 0], [1, 2, 3, 4])
    assert (u == u[:, :1]).all()


def test_padded_scheme_keeps_real_round_count():
    scheme = padded_scheme_fn([[1, 0], [0, 1]], pad_to=4)(0)
    assert scheme.who_speaks.shape == (4, 2)
    assert int(scheme.total_num_rounds) == 2


@pytest.mark.parametrize("estimate_mode", ["oracle", "keep"])
@pytest.mark.parametrize("act_first", [False, True])
def test_rollout_runs_and_keeps_beliefs_valid(estimate_mode, act_first):
    """A few episodes of passthrough driving: every stored belief stays a distribution."""
    h = Harness.build(scheme="a_to_b_thrice", num_agents=4, horizon=2, act_first=act_first, estimate_mode=estimate_mode)
    roll = h.rollout(seed=0, num_steps=6 * 5)  # 6 calls per a_to_b_thrice block, 5 blocks
    assert len(roll.states) == len(roll.obs) == 31
    assert len(roll.pre_act_states) == len(roll.rewards) == len(roll.actions) == 30
    for state in roll.states:
        for b in (state.true_agent_belief_states, state.estimated_agent_belief_states):
            b = np.asarray(b)
            assert np.isfinite(b).all() and (b >= 0).all()
            np.testing.assert_allclose(b.sum(-1), 1.0, atol=1e-5)
    # 5 blocks with horizon 2 must have crossed at least one episode boundary.
    assert schedule(roll.states[-1])[5] >= 1
