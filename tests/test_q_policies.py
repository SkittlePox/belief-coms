"""Tests for the Q-function policies: the guessing game's Q functions, the Boltzmann
policy, the policy bundle, and the stacked env's priors for untracked belief levels.

Run with::

    uv run pytest tests/test_q_policies.py
"""

import functools

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.env_assembly import assemble_environments
from envs.flexible_env import boltzmann_policy
from envs.guessing_game import guessing_game_spec
from stacked_harness import Harness

GAMMA = 0.9
WAIT = 3


def _categorical(probs):
    return distrax.Categorical(probs=jnp.array(probs, dtype=jnp.float32))


def _brute_force_presser_q(params, belief, depth=120, gamma=GAMMA):
    """Q(b, a) by a depth-limited Bellman lookahead read straight off the env tensors.

    The observer's action is inert, so it is fixed to wait. Reachable beliefs are few
    (supports shrink, weights keep their ratios), so memoizing on the rounded belief keeps
    this cheap even at a depth where γ^depth is negligible.
    """
    T = np.asarray(params.transition)[:, :, WAIT]  # [S, A, S']
    O = np.asarray(params.observation)[:, :, WAIT].sum(-1)  # presser's marginal [S', A, O]
    R = np.asarray(params.reward)[0][:, :, WAIT]  # [S, A, S']
    num_states, num_actions, num_observations = T.shape[0], T.shape[1], O.shape[-1]

    @functools.lru_cache(maxsize=None)
    def value(key, d):
        return max(q(key, a, d) for a in range(num_actions)) if d > 0 else 0.0

    def q(key, a, d):
        b = np.array(key)
        reward = sum(b[s] * T[s, a] @ R[s, a] for s in range(num_states))
        if d <= 1:
            return reward
        predicted = b @ T[:, a]
        future = 0.0
        for o in range(num_observations):
            unnormalized = predicted * O[:, a, o]
            mass = unnormalized.sum()
            if mass > 1e-12:
                future += mass * value(tuple(np.round(unnormalized / mass, 12)), d - 1)
        return reward + gamma * future

    key = tuple(np.round(np.asarray(belief, dtype=np.float64), 12))
    return np.array([q(key, a, depth) for a in range(num_actions)])


BELIEFS = [
    [1 / 3, 1 / 3, 1 / 3, 0.0],  # the prior
    [1.0, 0.0, 0.0, 0.0],  # certain
    [0.5, 0.5, 0.0, 0.0],  # two referents, even
    [0.8, 0.2, 0.0, 0.0],  # two referents, past the press threshold
    [0.6, 0.3, 0.1, 0.0],
    [0.2, 0.3, 0.1, 0.4],  # some mass on done
    [0.0, 0.0, 0.0, 1.0],  # certainly done: every action is worth 0
] + [list(np.random.default_rng(seed).dirichlet(np.full(4, 0.5))) for seed in range(8)]


@pytest.mark.parametrize("belief", BELIEFS, ids=[f"belief{i}" for i in range(len(BELIEFS))])
def test_presser_q_matches_brute_force_lookahead(belief):
    """The closed-form presser Q equals the optimal Q found by searching the real tensors."""
    spec = guessing_game_spec(discount_factor=GAMMA)
    b = _categorical(belief)
    got = np.asarray(spec.q_functions[0](b, b))
    want = _brute_force_presser_q(spec.params, belief)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_presser_q_ignores_the_estimate_of_the_observer():
    """The observer is inert in T, O and R, so b̄_o cannot move the presser's values."""
    q = guessing_game_spec().q_functions[0]
    b = _categorical([0.6, 0.3, 0.1, 0.0])
    np.testing.assert_allclose(q(b, _categorical([1, 0, 0, 0])), q(b, _categorical([0, 0, 1, 0])))


def test_presser_prefers_waiting_on_the_prior_and_pressing_when_certain():
    q = guessing_game_spec().q_functions[0]
    assert int(jnp.argmax(q(_categorical([1 / 3, 1 / 3, 1 / 3, 0]), _categorical([1 / 3, 1 / 3, 1 / 3, 0])))) == WAIT
    assert int(jnp.argmax(q(_categorical([0, 0, 1, 0]), _categorical([1 / 3, 1 / 3, 1 / 3, 0])))) == 2


def test_observer_policy_is_uniform():
    spec = guessing_game_spec()
    q_values = spec.q_functions[1](_categorical([0.6, 0.3, 0.1, 0.0]), _categorical([0, 1, 0, 0]))
    np.testing.assert_allclose(boltzmann_policy(q_values, spec.inverse_temperature).probs, 0.25, atol=1e-6)


@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize("which", ["ego", "estimate"])
def test_nan_beliefs_give_nan_q_values(role, which):
    """Tripwire: a NaN belief must not come out as a healthy-looking policy, even in a Q
    function that does not read that belief."""
    q = guessing_game_spec().q_functions[role]
    healthy, broken = _categorical([0.5, 0.5, 0, 0]), _categorical([jnp.nan] * 4)
    q_values = q(broken, healthy) if which == "ego" else q(healthy, broken)
    assert np.all(np.isnan(np.asarray(q_values)))


@pytest.mark.parametrize("inverse_temperature", [0.0, 1.0, 10.0])
def test_boltzmann_gives_unavailable_actions_zero_probability(inverse_temperature):
    """-inf Q (padding) means probability 0 at any β -- including β = 0, where β · -inf is NaN."""
    probs = np.asarray(boltzmann_policy(jnp.array([1.0, 0.0, -jnp.inf]), inverse_temperature).probs)
    assert probs[2] == 0.0
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    expected = np.exp(inverse_temperature * np.array([1.0, 0.0]))
    np.testing.assert_allclose(probs[:2], expected / expected.sum(), atol=1e-6)


def test_policy_bundle_dispatches_traced_game_type_and_role():
    """The bundle reproduces boltzmann(Q_role(b_e, b̄_o), β) with traced indices."""
    spec = guessing_game_spec()
    _, bundle = assemble_environments([guessing_game_spec])
    ego_belief, estimate = _categorical([0.6, 0.3, 0.1, 0.0]), _categorical([0.2, 0.2, 0.6, 0.0])
    for role in (0, 1):
        got = jax.jit(lambda r: bundle(jnp.array(0), r, ego_belief, estimate).probs)(jnp.array(role))
        want = boltzmann_policy(spec.q_functions[role](ego_belief, estimate), spec.inverse_temperature).probs
        np.testing.assert_allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("is_first_act", [True, False])
def test_untracked_level_predictions(is_first_act):
    """The predictions that reach levels above b̿_e never use a lower level in place of an
    untracked one. Both are Boltzmann policies with a stand-in belief U in every untracked
    slot: ego_action_as_they_predict = π_e(b̿_e, U), other_action_as_they_think_we_predict =
    π_o(U, U). U is the initial state distribution on the first act, uniform over all real
    states (terminal included) afterwards."""
    harness = Harness.build()
    env = harness.env
    params = jax.tree.map(lambda leaf: leaf[0], env.all_env_parameters)
    second_order = _categorical([0.7, 0.2, 0.1, 0.0])
    stand_in = _categorical([1 / 3, 1 / 3, 1 / 3, 0.0] if is_first_act else [0.25, 0.25, 0.25, 0.25])
    for role in (0, 1):
        ego_action_as_they_predict, other_action_as_they_think_we_predict = env._untracked_level_predictions(
            0, role, params, second_order, jnp.array(is_first_act)
        )
        np.testing.assert_allclose(ego_action_as_they_predict.probs, env.policy_bundle(0, role, second_order, stand_in).probs, atol=1e-6)
        np.testing.assert_allclose(
            other_action_as_they_think_we_predict.probs, env.policy_bundle(0, 1 - role, stand_in, stand_in).probs, atol=1e-6
        )
    # The presser's prediction really depends on b̿_e: a sharp b̿_e makes it press that button.
    presser_prediction, _ = env._untracked_level_predictions(0, 0, params, _categorical([0, 0, 1, 0]), jnp.array(is_first_act))
    assert int(jnp.argmax(presser_prediction.probs)) == 2


def test_untracked_presser_prediction_is_not_uniform_actions():
    """From the observer's seat, the presser's action as the presser thinks the observer
    predicts it is π_presser(U, U): mostly waiting, not uniform over actions."""
    env = Harness.build().env
    params = jax.tree.map(lambda leaf: leaf[0], env.all_env_parameters)
    for is_first_act in (True, False):
        _, presser_prediction = env._untracked_level_predictions(0, 1, params, _categorical([0.25] * 4), jnp.array(is_first_act))
        assert int(jnp.argmax(presser_prediction.probs)) == WAIT
        assert float(presser_prediction.probs[WAIT]) > 0.5


def test_first_act_flag_reaches_the_untracked_predictions(monkeypatch):
    """_step_underlying_env passes "is this the episode's first act" down to
    _untracked_level_predictions.

    The guessing game cannot show this on its own: U is symmetric over the referents, so
    every π(U, ·) presses each button equally, and a symmetric P(press) cancels out of
    the observer's b̿_e. Wrap the real predictions so that ONLY on the first act the
    presser's predicted action skews toward button 0: the observer's b̿_e must then
    differ between a first act and a later act, while b_e and b̄_o must not."""
    harness = Harness.build()
    env = harness.env
    real = env._untracked_level_predictions

    def flagged(game_type, role, game_params, second_order_ego_estimate, is_first_act):
        ego, other = real(game_type, role, game_params, second_order_ego_estimate, is_first_act)
        skewed = jnp.array([0.9, 0.0, 0.0, 0.1])
        return ego, distrax.Categorical(probs=jnp.where(is_first_act, skewed, other.probs))

    monkeypatch.setattr(env, "_untracked_level_predictions", flagged)
    state, _ = harness.reset(0)
    args = (
        jax.random.key(1),
        state.game_states,
        state.game_roles,
        state.true_agent_belief_states,
        state.estimated_agent_belief_states,
        state.second_order_agent_belief_estimates,
    )
    first = env._step_underlying_env(*args, jnp.asarray(0, dtype=jnp.int32))
    later = env._step_underlying_env(*args, jnp.asarray(5, dtype=jnp.int32))
    np.testing.assert_array_equal(first[5], later[5])  # same sampled actions
    np.testing.assert_allclose(first[1], later[1])  # b_e
    np.testing.assert_allclose(first[2], later[2])  # b̄_o
    observers = np.asarray(state.game_roles.agent_role_assignment) == 1
    presser_ids = ~observers
    np.testing.assert_allclose(np.asarray(first[3])[presser_ids], np.asarray(later[3])[presser_ids])  # observer's action is inert
    assert not np.allclose(np.asarray(first[3])[observers], np.asarray(later[3])[observers])  # observer's b̿_e moved
