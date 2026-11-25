"""Tests for the Q-function policies: the guessing game's Q functions, the Boltzmann
policy, the policy bundle, and the stacked env's belief-closure switch.

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


def test_unknown_belief_closure_is_rejected():
    with pytest.raises(ValueError, match="belief_closure"):
        Harness.build(belief_closure="third_order")


def test_belief_closure_changes_the_pressers_second_order_estimate():
    """The closure sets how the observer is assumed to predict the presser's action. The
    presser's actions move the state, so what the presser thinks the observer now thinks
    of it (b̿_e) depends on that assumption after the first act."""
    second_order = {}
    for closure in ("uniform", "stationary"):
        harness = Harness.build(scheme="a_to_b", num_agents=4, horizon=5, belief_closure=closure, estimate_mode="keep")
        roll = harness.rollout(seed=0, num_steps=2)  # a_to_b: one round (2 calls), then the act
        assert int(roll.states[2].game_counters.underlying_env_timestep) == 1
        second_order[closure] = np.asarray(roll.states[2].second_order_agent_belief_estimates)
    # With fixed_pairs_assignment_fn the pressers are the even agents, and row i is held by agent i.
    assert not np.allclose(second_order["uniform"][0::2], second_order["stationary"][0::2], atol=1e-3)
