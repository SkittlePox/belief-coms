"""Sanity checks for tests/stacked_harness.py itself, plus step_env wiring checks run through it.

Run with::

    uv run pytest tests/test_stacked_harness.py
"""

import jax
import numpy as np
import pytest

from stacked_harness import Harness, padded_scheme_fn, partners, schedule, sentinel_utterances


def test_reset_shapes_and_fixed_pairs():
    h = Harness.build(num_agents=4)
    state, (beliefs, estimates, second_order_estimates, utterances) = h.reset(seed=0)
    S = h.env.belief_dim
    assert beliefs.shape == estimates.shape == second_order_estimates.shape == (4, S)
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
def test_rollout_runs_and_keeps_beliefs_valid(estimate_mode):
    """A few episodes of passthrough driving: every stored belief stays a distribution."""
    h = Harness.build(scheme="a_to_b_thrice", num_agents=4, horizon=2, estimate_mode=estimate_mode)
    roll = h.rollout(seed=0, num_steps=6 * 5)  # 6 calls per a_to_b_thrice block, 5 blocks
    assert len(roll.states) == len(roll.obs) == 31
    assert len(roll.post_comm_true_beliefs) == len(roll.post_comm_estimated_beliefs) == len(roll.rewards) == len(roll.actions) == 30
    for state in roll.states:
        for b in (state.true_agent_belief_states, state.estimated_agent_belief_states, state.second_order_agent_belief_estimates):
            b = np.asarray(b)
            assert np.isfinite(b).all() and (b >= 0).all()
            np.testing.assert_allclose(b.sum(-1), 1.0, atol=1e-5)
    # 5 blocks with horizon 2 must have crossed at least one episode boundary.
    assert schedule(roll.states[-1])[5] >= 1


def test_belief_stage_records_listeners_estimates_of_speakers():
    """The utterance stage writes the speakers' estimates of the listeners (the listeners'
    subject rows); the belief stage writes the listeners' estimates of the speakers (the
    speakers' subject rows) and the listeners' own b̿_e rows. Under a_to_b (one round per
    block) that belief stage is the act, which runs on -- and reports -- the estimates as
    just written."""
    h = Harness.build(scheme="a_to_b")
    state, _ = h.reset(seed=0)
    speakers = np.asarray(state.game_roles.agent_role_assignment) == 0  # a_to_b: role 0 speaks
    listeners = ~speakers

    def skewed(at):
        est = state.true_agent_belief_states.at[:, at].add(0.5)
        return est / est.sum(axis=-1, keepdims=True)

    about_listeners, about_speakers, listeners_second_order = skewed(0), skewed(1), skewed(2)
    base = h.actions(state)
    actions = base._replace(
        speaker_other_belief_estimate_actions=about_listeners,
        listener_other_belief_estimate_actions=about_speakers,
        listener_second_order_ego_estimate_actions=listeners_second_order,
    )

    after_utt, *_ = h.step(jax.random.key(1), state, actions)
    np.testing.assert_array_equal(after_utt.estimated_agent_belief_states[listeners], about_listeners[listeners])
    np.testing.assert_array_equal(after_utt.estimated_agent_belief_states[speakers], state.estimated_agent_belief_states[speakers])

    after_belief, _, _, _, post_comm_estimated = h.step(jax.random.key(2), after_utt, actions)
    expected = np.where(speakers[:, None], about_speakers, about_listeners)
    np.testing.assert_array_equal(post_comm_estimated, expected)
    assert int(after_belief.game_counters.underlying_env_timestep) == 1, "the belief stage acted"


def test_stages_record_second_order_estimates_by_holder():
    """b̿_e is stored by holder: the utterance stage writes the SPEAKERS' rows (and leaves
    every true belief alone), the belief stage the LISTENERS' rows. Checked under
    a_to_b_thrice, whose first belief stage does not act (an act would move every b̿_e row
    through the observation update)."""
    h = Harness.build(scheme="a_to_b_thrice")
    state, _ = h.reset(seed=0)
    listeners = np.asarray(state.game_roles.agent_role_assignment) == 1  # a_to_b: role 0 speaks
    speakers = ~listeners

    def skewed(at):
        est = state.true_agent_belief_states.at[:, at].add(0.5)
        return est / est.sum(axis=-1, keepdims=True)

    speakers_second_order, listeners_second_order = skewed(1), skewed(2)

    actions = h.actions(state)._replace(speaker_second_order_ego_estimate_actions=speakers_second_order)
    after_utt, *_ = h.step(jax.random.key(1), state, actions)
    np.testing.assert_array_equal(after_utt.second_order_agent_belief_estimates[speakers], speakers_second_order[speakers])
    np.testing.assert_array_equal(
        after_utt.second_order_agent_belief_estimates[listeners], state.second_order_agent_belief_estimates[listeners]
    )
    np.testing.assert_array_equal(after_utt.true_agent_belief_states, state.true_agent_belief_states)

    actions = h.actions(after_utt)._replace(listener_second_order_ego_estimate_actions=listeners_second_order)
    after_belief, *_ = h.step(jax.random.key(2), after_utt, actions)
    assert int(after_belief.game_counters.underlying_env_timestep) == 0, "no act on the first round"
    np.testing.assert_array_equal(after_belief.second_order_agent_belief_estimates[listeners], listeners_second_order[listeners])
    np.testing.assert_array_equal(after_belief.second_order_agent_belief_estimates[speakers], speakers_second_order[speakers])
