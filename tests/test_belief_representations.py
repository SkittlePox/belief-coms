"""Tests for tools/belief_representations.py.

Run with::

    uv run pytest tests/test_belief_representations.py
"""

import itertools

import distrax
import jax
import pytest
import jax.numpy as jnp
import numpy as np

from envs.flexible_env import FlexibleEnvParams
from envs.guessing_game import (
    guessing_game_spec,
    role_0_optimal_policy,
    role_1_optimal_policy,
)
from tools.belief_representations import (
    initial_belief,
    initial_other_belief_estimate,
    update_other_belief_estimate_with_observation_only,
    update_with_observation_and_joint_action,
    update_with_observation_only,
)


def test_other_belief_estimate_keeps_states_the_partner_cannot_rule_out():
    """update_other_belief_estimate_with_observation_only must not rule out a state
    that the partner (whose action the ego did NOT reveal to it) cannot rule out.

    Scenario (guessing game). Agent 0 is the presser (role 0), agent 1 the observer
    (role 1). On this step agent 0 takes action 0 (presses button 0) and observes the
    referent symbol 0; agent 1 takes action 3 (wait) and observes symbol 1. We check
    agent 0's ESTIMATE of agent 1's belief.

    Key asymmetry: agent 1 never observes agent 0's action. So from agent 1's point of
    view, button 0 was not necessarily pressed, and state 0 is NOT ruled out. Indeed,
    agent 1's actual belief after seeing symbol 1 (computed the way agent 1 truly
    updates -- update_with_observation_only, marginalizing over agent 0's unseen
    action) puts mass 0.5 on state 0:

        agent 1's true belief after o1=1  ==  [0.5, 0.0, 0.5, 0.0]

    Therefore agent 0's estimate of agent 1's belief -- a mixture over the observations
    agent 1 might have received -- MUST place nonzero probability on state 0 (the
    o1=1 outcome alone contributes state-0 mass and has nonzero likelihood given o0=0).

    The bug: update_other_belief_estimate_with_observation_only models agent 1's update
    with update_with_observation_and_joint_action, i.e. it conditions on agent 0's
    ACTUAL action (button 0). Pressing button 0 sends state 0 -> the done state, so the
    modeled agent-1 posterior wrongly assigns state 0 probability 0. The estimate then
    excludes a state the real agent 1 believes with probability 0.5.
    """
    params, _ = guessing_game_spec()

    # Uniform prior over the three referent states {0, 1, 2} (state 3 is terminal).
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))

    estimate = update_other_belief_estimate_with_observation_only(
        params,
        ego_belief=initial_belief(params, 0, agent_id=0),  # the ego's OWN belief
        other_belief_estimate=uniform_referent,  # ego's prior estimate of agent 1
        ego_observation=0,  # agent 0 saw referent symbol 0
        ego_action=0,  # agent 0 pressed button 0
        other_policy=role_1_optimal_policy,  # agent 1's policy
        agent_id=0,
    )

    # Reference: agent 1's ACTUAL belief had it seen symbol 1 (the observation it in fact
    # received). Agent 1 marginalizes over agent 0's unseen action, so it cannot rule out
    # state 0. This is the belief agent 0's estimate is supposed to be consistent with.
    agent1_true_belief = update_with_observation_only(
        params,
        ego_belief=uniform_referent,  # agent 1's own prior
        other_belief_estimate=uniform_referent,  # agent 1's estimate of agent 0
        ego_observation=1,  # agent 1 saw symbol 1
        ego_action=3,  # agent 1's own action (wait)
        other_policy=role_0_optimal_policy,  # agent 0's policy, marginalized over by agent 1
        agent_id=1,
    )

    # Agent 1 genuinely believes state 0 is possible (sanity-check the reference).
    assert agent1_true_belief.probs[0] > 0.0, (
        "reference: agent 1 (not seeing agent 0's action) should keep state-0 mass; " f"got {agent1_true_belief.probs}"
    )

    # The estimate must not rule out state 0 that agent 1 itself cannot rule out.
    assert estimate.probs[0] > 1e-6, (
        "agent 0's estimate of agent 1's belief wrongly assigns zero probability to "
        f"state 0 (estimate={estimate.probs}), even though agent 1 -- which never "
        f"observed agent 0's button press -- believes state 0 with probability "
        f"{float(agent1_true_belief.probs[0]):.3f}. The estimate conditions on agent "
        "0's own action, which agent 1 did not observe."
    )


def test_ego_belief_is_mandatory():
    """Omitting the ego's own belief must fail loudly, not silently degrade.

    It used to default to b̄, which quietly roughly doubled the error (worst case 0.17 ->
    0.37): b̄ is the OTHER agent's view of the world, so standing it in here asks them to
    guess their own observation from their own belief, which is nearly uninformative. The
    ego's belief is the only channel by which the ego's private knowledge reaches the
    estimate, so it is now required.
    """
    params, policies = guessing_game_spec()
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))
    rest = dict(ego_observation=0, ego_action=0, other_policy=policies[1], agent_id=0)

    with pytest.raises(TypeError):  # missing keyword-only argument
        update_other_belief_estimate_with_observation_only(params, other_belief_estimate=uniform_referent, **rest)

    with pytest.raises(ValueError, match="ego_belief is required"):
        update_other_belief_estimate_with_observation_only(params, ego_belief=None, other_belief_estimate=uniform_referent, **rest)


def test_the_reset_observation_beats_the_prior():
    """Do not seed the estimate with the prior once you have actually observed something.

    Before anyone looks, the estimate of a partner's belief IS the prior -- there is nothing
    to compute. But the two agents' reset observations are generated by the same state, so
    the ego's own observation is evidence about theirs, and hence about what they now
    believe. Using it moves the estimate off the prior, and that movement is information,
    not noise. This is the trap in "just start everyone at the prior".
    """
    params, _ = guessing_game_spec()

    blind = np.asarray(params.initial_state_distribution)  # the best you can do UNOBSERVED
    looking = np.asarray(initial_other_belief_estimate(params, jnp.array(1), agent_id=0).probs)

    np.testing.assert_allclose(blind, [1 / 3, 1 / 3, 1 / 3, 0.0], atol=1e-5)
    # The ego saw symbol 1 and ruled out state 1 for ITSELF. The two agents always see
    # DIFFERENT symbols, so the other agent certainly did not see 1: it saw 2 (state 0) or
    # 0 (state 2), 50/50, and so believes [.5, .5, 0] or [0, .5, .5]. It can never have
    # ruled state 1 out -- its estimated belief in state 1 RISES to 0.5.
    np.testing.assert_allclose(looking, [0.25, 0.5, 0.25, 0.0], atol=1e-4)
    assert not np.allclose(blind, looking, atol=1e-2)


def test_impossible_observations_produce_nan():
    """An observation with zero likelihood under the belief must yield NaN, loudly.

    In the guessing game state s never emits symbol s, so a belief certain of state 0
    that then sees symbol 0 has seen something impossible. The environment should never
    produce that, so the NaN is a deliberate tripwire for bugs elsewhere -- it must not
    be silently replaced by some finite fallback. (The exact-returns evaluator enumerates
    impossible observation pairs by design and handles the NaN itself.)
    """
    params, _ = guessing_game_spec()
    certain_of_0 = distrax.Categorical(probs=jnp.array([1.0, 0.0, 0.0, 0.0]))
    wait = int(params.num_actions) - 1

    updated = update_with_observation_and_joint_action(params, certain_of_0, 0, (wait, wait), agent_id=0)
    assert np.all(np.isnan(updated.probs))

    updated = update_with_observation_only(
        params,
        ego_belief=certain_of_0,
        other_belief_estimate=certain_of_0,
        ego_observation=0,
        ego_action=wait,
        other_policy=role_1_optimal_policy,
        agent_id=0,
    )
    assert np.all(np.isnan(updated.probs))

    # Symbol 3 is only emitted from the terminal state, which the prior excludes.
    assert np.all(np.isnan(initial_belief(params, 3, agent_id=0).probs))


# --- Agreement with a brute-force reference ----------------------------------------
#
# The implementation is vectorized in the ego frame; the reference below is plain
# loops written straight from the docstring formulas, in agent order. Random dynamics
# with a correlated, ASYMMETRIC joint observation model, so swapping agents (or their
# observation axes) cannot cancel out.

S, A, O = 4, 3, 3


def _random_simplex(rng, *shape):
    x = rng.random(shape) ** 3  # skewed, so some entries are near zero
    return x / x.sum(-1, keepdims=True)


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    T = _random_simplex(rng, S, A, A, S)
    Oj = _random_simplex(rng, S, A, A, O * O).reshape(S, A, A, O, O)
    P0 = _random_simplex(rng, S)
    params = FlexibleEnvParams(
        transition=jnp.array(T, dtype=jnp.float32),
        observation=jnp.array(Oj, dtype=jnp.float32),
        reward=jnp.zeros((2, S, A, A, S)),
        num_actions=jnp.array(A),
        num_states=jnp.array(S),
        initial_state_distribution=jnp.array(P0, dtype=jnp.float32),
        terminal_mask=jnp.zeros(S),
    )
    W = rng.normal(size=(S, A))  # a fixed nonlinear belief -> action-probs map

    def policy(belief):
        return distrax.Categorical(logits=jnp.log(belief.probs + 1e-9) @ W)

    def policy_np(b):
        z = np.log(b + 1e-9) @ W
        z = np.exp(z - z.max())
        return z / z.sum()

    return params, T, Oj, P0, policy, policy_np, rng


def _joint(agent_id, ego, other):
    return (ego, other) if agent_id == 0 else (other, ego)


def _obs_own(Oj, sp, a, agent_id, o):
    """The marginal likelihood of agent `agent_id`'s own observation."""
    grid = Oj[sp, a[0], a[1]]
    return grid[o, :].sum() if agent_id == 0 else grid[:, o].sum()


def _obs_pair(Oj, sp, a, agent_id, o_ego, o_other):
    grid = Oj[sp, a[0], a[1]]
    return grid[o_ego, o_other] if agent_id == 0 else grid[o_other, o_ego]


def _predict(T, b, a, sp):
    return sum(T[s, a[0], a[1], sp] * b[s] for s in range(S))


def _norm(v):
    return v / v.sum()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("agent_id", [0, 1])
@pytest.mark.parametrize("traced", [False, True], ids=["static", "traced"])
def test_matches_brute_force_reference(seed, agent_id, traced):
    params, T, Oj, P0, policy, policy_np, rng = _random_problem(seed)
    b, b_bar = _random_simplex(rng, S), _random_simplex(rng, S)
    obs, a_ego, a_other = (int(x) for x in rng.integers([O, A, A]))
    prior_h = _random_simplex(rng, A)
    other_id = 1 - agent_id
    aid = jnp.array(agent_id) if traced else agent_id
    belief, belief_bar = distrax.Categorical(probs=jnp.array(b)), distrax.Categorical(probs=jnp.array(b_bar))

    def close(got, want):
        np.testing.assert_allclose(np.asarray(got.probs), want, atol=1e-5)

    # update_with_observation_and_joint_action
    a = _joint(agent_id, a_ego, a_other)
    want = _norm(np.array([_obs_own(Oj, sp, a, agent_id, obs) * _predict(T, b, a, sp) for sp in range(S)]))
    close(jax.jit(lambda aid_: update_with_observation_and_joint_action(params, belief, obs, a, agent_id=aid_))(aid), want)

    # update_with_observation_only
    pi = policy_np(b_bar)
    want = np.zeros(S)
    for x, sp in itertools.product(range(A), range(S)):
        a = _joint(agent_id, a_ego, x)
        want[sp] += pi[x] * _obs_own(Oj, sp, a, agent_id, obs) * _predict(T, b, a, sp)
    got = update_with_observation_only(
        params, ego_belief=belief, other_belief_estimate=belief_bar, ego_observation=obs, ego_action=a_ego, other_policy=policy, agent_id=aid
    )
    close(got, _norm(want))

    # initial_belief
    close(initial_belief(params, obs, agent_id=aid), _norm(np.array([P0[s] * _obs_own(Oj, s, (0, 0), agent_id, obs) for s in range(S)])))

    # initial_other_belief_estimate
    want = np.zeros(S)
    for j in range(O):
        w = sum(P0[s] * _obs_pair(Oj, s, (0, 0), agent_id, obs, j) for s in range(S))
        want += w * _norm(np.array([P0[s] * _obs_own(Oj, s, (0, 0), other_id, j) for s in range(S)]))
    close(initial_other_belief_estimate(params, obs, agent_id=aid), _norm(want))

    # update_other_belief_estimate_with_observation_only, both modes, both priors
    for mode, h_prior in itertools.product(["mixture", "soft_evidence"], [None, prior_h]):
        p_h = np.ones(A) / A if h_prior is None else h_prior
        want = np.zeros(S)
        for x, j in itertools.product(range(A), range(O)):
            a = _joint(agent_id, a_ego, x)
            w = pi[x] * sum(_predict(T, b, a, sp) * _obs_pair(Oj, sp, a, agent_id, obs, j) for sp in range(S))
            post = np.zeros(S)
            for h, sp in itertools.product(range(A), range(S)):
                a_h = _joint(agent_id, h, x)
                post[sp] += p_h[h] * _predict(T, b_bar, a_h, sp) * _obs_own(Oj, sp, a_h, other_id, j)
            want += w * (_norm(post) if mode == "mixture" else post)
        got = update_other_belief_estimate_with_observation_only(
            params,
            ego_belief=belief,
            other_belief_estimate=belief_bar,
            ego_observation=obs,
            ego_action=a_ego,
            other_policy=policy,
            agent_id=aid,
            ego_action_prior=None if h_prior is None else jnp.array(h_prior),
            mode=mode,
        )
        close(got, _norm(want))
