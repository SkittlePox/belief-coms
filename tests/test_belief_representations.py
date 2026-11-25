"""Tests for tools/belief_representations.py.

Run with::

    uv run pytest tests/test_belief_representations.py
"""

import dataclasses
import itertools

import distrax
import jax
import pytest
import jax.numpy as jnp
import numpy as np

from envs.flexible_env import FlexibleEnvParams
from envs.guessing_game import guessing_game_spec
from tools.belief_representations import (
    initial_belief,
    initial_other_belief_estimate,
    initial_second_order_ego_estimate,
    update_other_belief_estimate_with_observation_only,
    update_second_order_ego_estimate_with_observation_only,
    update_with_observation_and_joint_action,
    update_with_observation_only,
)


# The guessing game has 4 actions (3 buttons and wait).
UNIFORM_ACTIONS = distrax.Categorical(probs=jnp.ones(4) / 4)
# A presser pressing each button 0-2 with probability 1/3 and never waiting. The belief
# updates take action distributions as given, so the worked examples fix this one by hand
# rather than deriving it from a policy. (The observer's action is inert in the guessing
# game, so any distribution over it gives the same updates; UNIFORM_ACTIONS is used.)
PRESS_EACH_BUTTON = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))


def test_update_with_observation_only_worked_example():
    """A hand-checkable update: the observer waits, sees symbol 1, and rules out state 1.

    Guessing game. States 0-2 are referents, state 3 is "done". Agent 1 (the observer)
    starts uniform over the referents, waits (its action is inert), and sees symbol 1.
    It never sees agent 0's action, so it marginalizes over it: agent 0's policy on a
    uniform belief is taken to press each button 0-2 with probability 1/3.

    1. Predict -- sum over agent 0's possible presses k of (1/3) · T(s' | s, k):
       from referent s, pressing k == s ends the game (s -> done) and k != s leaves s
       put. Each referent therefore keeps 1/3 · 2/3 = 2/9 and sends 1/3 · 1/3 = 1/9 to
       done, so

           predicted = [2/9, 2/9, 2/9, 1/3]

    2. Likelihood of agent 1 seeing symbol 1 in each next state s':
       in referent s the agent sees one of the two symbols != s, 50/50; in done it
       sees only the done symbol. So

           likelihood = [1/2, 0, 1/2, 0]

    3. Multiply and normalize:

           [1/9, 0, 1/9, 0]  ->  [0.5, 0, 0.5, 0]

    State 1 is ruled out by the observation (state s never shows symbol s). "Done" is
    ruled out too: agent 0 might well have ended the game, but then agent 1 would have
    seen the done symbol instead.
    """
    env_params = guessing_game_spec().params
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))
    wait = 3

    updated = update_with_observation_only(
        env_params,
        ego_belief=uniform_referent,  # agent 1's own belief
        ego_observation=1,  # agent 1 saw symbol 1
        ego_action_tminus1=wait,  # agent 1 waited
        other_action_distribution=PRESS_EACH_BUTTON,  # how agent 1 thinks agent 0 acted
        ego_role=1,
    )

    np.testing.assert_allclose(updated.probs, [0.5, 0.0, 0.5, 0.0], atol=1e-6)


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
    params = guessing_game_spec().params

    # Uniform prior over the three referent states {0, 1, 2} (state 3 is terminal).
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))

    estimate = update_other_belief_estimate_with_observation_only(
        params,
        ego_belief=initial_belief(params, 0, ego_role=0),  # the ego's OWN belief
        other_belief_estimate=uniform_referent,  # ego's prior estimate of agent 1
        ego_observation=0,  # agent 0 saw referent symbol 0
        ego_action=0,  # agent 0 pressed button 0
        other_action_distribution=UNIFORM_ACTIONS,  # agent 1's actions (inert)
        ego_role=0,
        ego_action_prior=UNIFORM_ACTIONS,  # what agent 1 assumes agent 0 did
    )

    # Reference: agent 1's ACTUAL belief had it seen symbol 1 (the observation it in fact
    # received). Agent 1 marginalizes over agent 0's unseen action, so it cannot rule out
    # state 0. This is the belief agent 0's estimate is supposed to be consistent with.
    agent1_true_belief = update_with_observation_only(
        params,
        ego_belief=uniform_referent,  # agent 1's own prior
        ego_observation=1,  # agent 1 saw symbol 1
        ego_action_tminus1=3,  # agent 1's own action (wait)
        other_action_distribution=PRESS_EACH_BUTTON,  # agent 0's action, marginalized over by agent 1
        ego_role=1,
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
    params = guessing_game_spec().params
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))
    rest = dict(
        ego_observation=0,
        ego_action=0,
        other_action_distribution=UNIFORM_ACTIONS,
        ego_role=0,
        ego_action_prior=UNIFORM_ACTIONS,
    )

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
    params = guessing_game_spec().params

    blind = np.asarray(params.initial_state_distribution)  # the best you can do UNOBSERVED
    looking = np.asarray(initial_other_belief_estimate(params, jnp.array(1), ego_role=0).probs)

    np.testing.assert_allclose(blind, [1 / 3, 1 / 3, 1 / 3, 0.0], atol=1e-5)
    # The ego saw symbol 1 and ruled out state 1 for ITSELF. The two agents always see
    # DIFFERENT symbols, so the other agent certainly did not see 1: it saw 2 (state 0) or
    # 0 (state 2), 50/50, and so believes [.5, .5, 0] or [0, .5, .5]. It can never have
    # ruled state 1 out -- its estimated belief in state 1 RISES to 0.5.
    np.testing.assert_allclose(looking, [0.25, 0.5, 0.25, 0.0], atol=1e-4)
    assert not np.allclose(blind, looking, atol=1e-2)


def test_initial_second_order_ego_estimate_worked_example():
    """Our estimate of THEIR estimate of our belief, right after reset (guessing game).

    Agent 0 (the ego) sees symbol 1. The two agents always see different symbols, both
    different from the state, so:

      - state 0: they saw 2.  state 2: they saw 0.  50/50.

    They don't know what we saw, so each of their observations leaves them with two
    possibilities for us:

      - They saw 2 -> state 0 (we saw 1, believe [1/2, 0, 1/2, 0])
                   or state 1 (we saw 0, believe [0, 1/2, 1/2, 0]).
        Their estimate of us: [1/4, 1/4, 1/2, 0].
      - They saw 0 -> state 1 (we saw 2, believe [1/2, 1/2, 0, 0])
                   or state 2 (we saw 1, believe [1/2, 0, 1/2, 0]).
        Their estimate of us: [1/2, 1/4, 1/4, 0].

    Averaging the two: [3/8, 1/4, 3/8, 0]. Our own belief is [1/2, 0, 1/2, 0], but we
    expect them to think we still hold state 1 at 1/4 -- they never saw our symbol.
    """
    params = guessing_game_spec().params

    # Premise: their estimate of us, for each of the two observations they could have had.
    np.testing.assert_allclose(initial_other_belief_estimate(params, 2, ego_role=1).probs, [1 / 4, 1 / 4, 1 / 2, 0], atol=1e-6)
    np.testing.assert_allclose(initial_other_belief_estimate(params, 0, ego_role=1).probs, [1 / 2, 1 / 4, 1 / 4, 0], atol=1e-6)

    second_order = initial_second_order_ego_estimate(params, 1, ego_role=0)
    np.testing.assert_allclose(second_order.probs, [3 / 8, 1 / 4, 3 / 8, 0], atol=1e-6)


def test_impossible_observations_produce_nan():
    """An observation with zero likelihood under the belief must yield NaN, loudly.

    In the guessing game state s never emits symbol s, so a belief certain of state 0
    that then sees symbol 0 has seen something impossible. The environment should never
    produce that, so the NaN is a deliberate tripwire for bugs elsewhere -- it must not
    be silently replaced by some finite fallback. (The exact-returns evaluator enumerates
    impossible observation pairs by design and handles the NaN itself.)
    """
    params = guessing_game_spec().params
    certain_of_0 = distrax.Categorical(probs=jnp.array([1.0, 0.0, 0.0, 0.0]))
    wait = int(params.num_actions) - 1

    updated = update_with_observation_and_joint_action(params, certain_of_0, 0, (wait, wait), ego_role=0)
    assert np.all(np.isnan(updated.probs))

    updated = update_with_observation_only(
        params,
        ego_belief=certain_of_0,
        ego_observation=0,
        ego_action_tminus1=wait,
        other_action_distribution=UNIFORM_ACTIONS,
        ego_role=0,
    )
    assert np.all(np.isnan(updated.probs))

    estimate = update_other_belief_estimate_with_observation_only(
        params,
        ego_belief=certain_of_0,
        other_belief_estimate=certain_of_0,
        ego_observation=0,
        ego_action=wait,
        other_action_distribution=UNIFORM_ACTIONS,
        ego_role=0,
        ego_action_prior=UNIFORM_ACTIONS,
    )
    assert np.all(np.isnan(estimate.probs))

    second_order = update_second_order_ego_estimate_with_observation_only(
        params,
        ego_belief=certain_of_0,
        other_belief_estimate=certain_of_0,
        second_order_ego_estimate=certain_of_0,
        ego_observation=0,
        ego_action=wait,
        other_action_distribution=UNIFORM_ACTIONS,
        ego_role=0,
        ego_action_prior=UNIFORM_ACTIONS,
        other_action_prior=UNIFORM_ACTIONS,
    )
    assert np.all(np.isnan(second_order.probs))

    # Symbol 3 is only emitted from the terminal state, which the prior excludes.
    assert np.all(np.isnan(initial_belief(params, 3, ego_role=0).probs))
    assert np.all(np.isnan(initial_other_belief_estimate(params, 3, ego_role=0).probs))
    assert np.all(np.isnan(initial_second_order_ego_estimate(params, 3, ego_role=0).probs))


def test_contradictory_estimates_produce_nan():
    """A branch WE find possible but THEY find impossible must yield NaN, not be dropped.

    Guessing game, ego = agent 0. We waited, saw symbol 1, and our belief is uniform over
    the referents, so as far as we know the state is 0 (they saw 2) or 2 (they saw 0).
    But our estimate of THEIR belief says they are certain the state is 2.

    That is a contradiction: if the state is 0 -- which we find possible -- they saw 2,
    which is impossible under a belief certain of state 2. Their posterior for that branch
    is 0/0. Silently dropping it would leave a finite, healthy-looking estimate built from
    the other branch alone. The estimates are inconsistent with each other, so the update
    must return NaN instead.
    """
    params = guessing_game_spec().params
    uniform_referent = distrax.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))
    certain_of_2 = distrax.Categorical(probs=jnp.array([0.0, 0.0, 1.0, 0.0]))
    wait = int(params.num_actions) - 1

    estimate = update_other_belief_estimate_with_observation_only(
        params,
        ego_belief=uniform_referent,  # we find state 0 possible
        other_belief_estimate=certain_of_2,  # ...but think they are certain of state 2
        ego_observation=1,
        ego_action=wait,
        other_action_distribution=UNIFORM_ACTIONS,
        ego_role=0,
        ego_action_prior=UNIFORM_ACTIONS,
    )
    assert np.all(np.isnan(estimate.probs))


# --- Agreement with a brute-force reference ----------------------------------------
#
# The belief functions are vectorized (einsum, in the ego frame). Each test below
# recomputes the same quantity with plain Python loops written straight from that
# function's docstring formula, directly on the agent-ordered tensors, and checks the
# two agree. The dynamics are random, with a correlated and ASYMMETRIC joint
# observation model, so mixing up the two agents (or their observation axes) cannot
# cancel out by symmetry.
#
# Every test runs on the same 16 random cases: 4 seeds x ego agent 0 or 1 x ego_role
# passed as a plain int ("static") or as a traced value under jax.jit ("traced").

S, A, O = 4, 3, 3  # states, actions, observations

RANDOM_CASES = list(itertools.product(range(4), [0, 1], [False, True]))
RANDOM_CASE_IDS = [f"seed{seed}-agent{ego_role}-{'traced' if traced else 'static'}" for seed, ego_role, traced in RANDOM_CASES]


@dataclasses.dataclass
class RandomCase:
    env_params: FlexibleEnvParams
    T: np.ndarray  # T[s, a0, a1, s']            (agent order)
    O_joint: np.ndarray  # O[s', a0, a1, o0, o1]  (agent order)
    prior: np.ndarray  # P0[s]
    policy_probs: object  # a fixed nonlinear belief -> action-probs map on numpy vectors
    ego_belief: np.ndarray
    other_belief_estimate: np.ndarray
    ego_observation: int
    ego_action: int
    other_action: int
    ego_action_prior: np.ndarray
    second_order_ego_estimate: np.ndarray
    other_action_prior: np.ndarray
    ego_role: int
    traced: bool

    @property
    def other_id(self):
        return 1 - self.ego_role

    def run(self, belief_function):
        """Call belief_function(ego_role) -- under jax.jit with a traced ego_role if traced."""
        if self.traced:
            return jax.jit(belief_function)(jnp.array(self.ego_role))
        return belief_function(self.ego_role)


def _random_simplex(rng, *shape):
    x = rng.random(shape) ** 3  # skewed, so some entries are near zero
    return x / x.sum(-1, keepdims=True)


def _random_case(seed, ego_role, traced):
    rng = np.random.default_rng(seed)
    T = _random_simplex(rng, S, A, A, S)
    O_joint = _random_simplex(rng, S, A, A, O * O).reshape(S, A, A, O, O)
    prior = _random_simplex(rng, S)
    env_params = FlexibleEnvParams(
        transition=jnp.array(T, dtype=jnp.float32),
        observation=jnp.array(O_joint, dtype=jnp.float32),
        reward=jnp.zeros((2, S, A, A, S)),
        num_actions=jnp.array(A),
        num_states=jnp.array(S),
        initial_state_distribution=jnp.array(prior, dtype=jnp.float32),
        terminal_mask=jnp.zeros(S),
    )
    W = rng.normal(size=(S, A))  # a fixed nonlinear belief -> action-probs map

    def policy_probs(belief_probs):
        z = np.exp(np.log(belief_probs + 1e-9) @ W)
        return z / z.sum()

    ego_observation, ego_action, other_action = (int(x) for x in rng.integers([O, A, A]))
    return RandomCase(
        env_params=env_params,
        T=T,
        O_joint=O_joint,
        prior=prior,
        policy_probs=policy_probs,
        ego_belief=_random_simplex(rng, S),
        other_belief_estimate=_random_simplex(rng, S),
        ego_observation=ego_observation,
        ego_action=ego_action,
        other_action=other_action,
        ego_action_prior=_random_simplex(rng, A),
        # Drawn last, so the earlier fields are unchanged by their addition.
        second_order_ego_estimate=_random_simplex(rng, S),
        other_action_prior=_random_simplex(rng, A),
        ego_role=ego_role,
        traced=traced,
    )


# Reference helpers. All take actions as an (agent 0, agent 1) tuple, like the env.


def _agent_order(ego_role, ego_action, other_action):
    """The (agent 0, agent 1) joint action, given the ego's and the other's actions."""
    return (ego_action, other_action) if ego_role == 0 else (other_action, ego_action)


def _predicted_prob(T, belief, joint_action, next_state):
    """∑_s T(s' | s, a) b(s)"""
    a0, a1 = joint_action
    return sum(T[s, a0, a1, next_state] * belief[s] for s in range(S))


def _own_observation_prob(O_joint, next_state, joint_action, ego_role, observation):
    """O_i(o_i | s', a): agent `ego_role`'s observation, the other agent's summed out."""
    a0, a1 = joint_action
    grid = O_joint[next_state, a0, a1]  # [o0, o1]
    return grid[observation, :].sum() if ego_role == 0 else grid[:, observation].sum()


def _joint_observation_prob(O_joint, next_state, joint_action, ego_role, ego_observation, other_observation):
    """O(o_ego, o_other | s', a), with the observations given in ego/other order."""
    a0, a1 = joint_action
    grid = O_joint[next_state, a0, a1]  # [o0, o1]
    return grid[ego_observation, other_observation] if ego_role == 0 else grid[other_observation, ego_observation]


def _normalized(v):
    return v / v.sum()


def _assert_close(got, want):
    np.testing.assert_allclose(np.asarray(got.probs), want, atol=1e-5)


# The estimate updates, written as loops from the seat of agent `updater_role` (the
# updater) about its partner. The second-order references call them from the OTHER
# agent's seat, exactly as the second-order functions do.


def _reference_branch_weights(c, updater_role, updater_belief, updater_observation, updater_action, partner_action_probs):
    """w(a_partner, o_partner) ∝ π(a_partner) · ∑_{s'} pred(s') · O(o_updater, o_partner | s', a),
    pred(s') = ∑_s T(s' | s, a) b_updater(s), with a = (a_updater, a_partner). Shape [A, O]."""
    weights = np.zeros((A, O))
    for partner_action in range(A):
        joint_action = _agent_order(updater_role, updater_action, partner_action)
        for partner_observation in range(O):
            weights[partner_action, partner_observation] = partner_action_probs[partner_action] * sum(
                _predicted_prob(c.T, updater_belief, joint_action, next_state)
                * _joint_observation_prob(c.O_joint, next_state, joint_action, updater_role, updater_observation, partner_observation)
                for next_state in range(S)
            )
    return weights


def _reference_estimate_update(
    c, updater_role, updater_belief, partner_belief_estimate, updater_observation, updater_action, partner_action_probs, updater_action_prior
):
    """Steps 1-5 of update_other_belief_estimate_with_observation_only, from `updater_role`'s seat:

        b_{a,o}(s') ∝ ∑_h P(h) · [∑_s T(s' | s, h, a_partner) b̄(s)] · O_partner(o_partner | s', h, a_partner)
        b̄'(s')      ∝ ∑_{a_partner, o_partner} w(a_partner, o_partner) · b_{a,o}(s')

    where h ranges over the updater actions the partner thinks it might have taken.
    """
    partner_role = 1 - updater_role
    weights = _reference_branch_weights(c, updater_role, updater_belief, updater_observation, updater_action, partner_action_probs)
    mixture = np.zeros(S)
    for partner_action in range(A):
        for partner_observation in range(O):
            # The partner's posterior, marginalizing the updater action h it never saw.
            partner_posterior = np.zeros(S)
            for h in range(A):
                hypothetical_joint_action = _agent_order(updater_role, h, partner_action)
                for next_state in range(S):
                    partner_posterior[next_state] += (
                        updater_action_prior[h]
                        * _predicted_prob(c.T, partner_belief_estimate, hypothetical_joint_action, next_state)
                        * _own_observation_prob(c.O_joint, next_state, hypothetical_joint_action, partner_role, partner_observation)
                    )
            mixture += weights[partner_action, partner_observation] * _normalized(partner_posterior)
    return _normalized(mixture)


def _reference_initial_estimate(c, updater_role, updater_observation):
    """b̄_0(s) = ∑_{o_partner} w(o_partner) · b_{o_partner}(s), from `updater_role`'s seat, where
    w(o_partner) ∝ ∑_s P0(s) · O(o_updater, o_partner | s, reset_action) and
    b_{o_partner}(s) ∝ P0(s) · O_partner(o_partner | s, reset_action)."""
    reset_action = (0, 0)
    mixture = np.zeros(S)
    for partner_observation in range(O):
        weight = sum(
            c.prior[state] * _joint_observation_prob(c.O_joint, state, reset_action, updater_role, updater_observation, partner_observation)
            for state in range(S)
        )
        partner_posterior = np.array(
            [c.prior[state] * _own_observation_prob(c.O_joint, state, reset_action, 1 - updater_role, partner_observation) for state in range(S)]
        )
        if weight > 0:
            mixture += weight * _normalized(partner_posterior)
    return _normalized(mixture)


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_update_with_observation_and_joint_action_matches_reference(seed, ego_role, traced):
    """b'(s') ∝ O_ego(o | s', a) · ∑_s T(s' | s, a) b(s)"""
    c = _random_case(seed, ego_role, traced)
    joint_action = _agent_order(c.ego_role, c.ego_action, c.other_action)

    want = np.zeros(S)
    for next_state in range(S):
        want[next_state] = _own_observation_prob(c.O_joint, next_state, joint_action, c.ego_role, c.ego_observation) * _predicted_prob(
            c.T, c.ego_belief, joint_action, next_state
        )

    got = c.run(
        lambda ego_role: update_with_observation_and_joint_action(
            c.env_params, distrax.Categorical(probs=jnp.array(c.ego_belief)), c.ego_observation, joint_action, ego_role=ego_role
        )
    )
    _assert_close(got, _normalized(want))


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_update_with_observation_only_matches_reference(seed, ego_role, traced):
    """b'(s') ∝ ∑_{a_other} π(a_other) · O_ego(o | s', a) · ∑_s T(s' | s, a) b(s),
    with a = (ego_action, a_other)."""
    c = _random_case(seed, ego_role, traced)
    other_action_probs = c.policy_probs(c.other_belief_estimate)  # π, passed in as given

    want = np.zeros(S)
    for other_action in range(A):
        joint_action = _agent_order(c.ego_role, c.ego_action, other_action)
        for next_state in range(S):
            want[next_state] += (
                other_action_probs[other_action]
                * _own_observation_prob(c.O_joint, next_state, joint_action, c.ego_role, c.ego_observation)
                * _predicted_prob(c.T, c.ego_belief, joint_action, next_state)
            )

    got = c.run(
        lambda ego_role: update_with_observation_only(
            c.env_params,
            ego_belief=distrax.Categorical(probs=jnp.array(c.ego_belief)),
            ego_observation=c.ego_observation,
            ego_action_tminus1=c.ego_action,
            other_action_distribution=distrax.Categorical(probs=jnp.array(other_action_probs)),
            ego_role=ego_role,
        )
    )
    _assert_close(got, _normalized(want))


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_initial_belief_matches_reference(seed, ego_role, traced):
    """b_0(s) ∝ P0(s) · O_ego(o_0 | s, reset_action), with reset_action = (0, 0)."""
    c = _random_case(seed, ego_role, traced)
    reset_action = (0, 0)

    want = np.zeros(S)
    for state in range(S):
        want[state] = c.prior[state] * _own_observation_prob(c.O_joint, state, reset_action, c.ego_role, c.ego_observation)

    got = c.run(lambda ego_role: initial_belief(c.env_params, c.ego_observation, ego_role=ego_role))
    _assert_close(got, _normalized(want))


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_initial_other_belief_estimate_matches_reference(seed, ego_role, traced):
    """b̄_0(s) = ∑_{o_other} w(o_other) · b_{o_other}(s), where
    w(o_other) ∝ ∑_s P0(s) · O(o_ego, o_other | s, reset_action) and
    b_{o_other}(s) ∝ P0(s) · O_other(o_other | s, reset_action)."""
    c = _random_case(seed, ego_role, traced)
    want = _reference_initial_estimate(c, c.ego_role, c.ego_observation)
    got = c.run(lambda ego_role: initial_other_belief_estimate(c.env_params, c.ego_observation, ego_role=ego_role))
    _assert_close(got, want)


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_update_other_belief_estimate_matches_reference(seed, ego_role, traced):
    """Steps 1-5 of the docstring:

        w(a_other, o_other) ∝ π(a_other) · ∑_{s'} pred_ego(s') · O(o_ego, o_other | s', a)
        b_{a,o}(s')         ∝ ∑_h P(h) · [∑_s T(s' | s, h, a_other) b̄(s)] · O_other(o_other | s', h, a_other)
        b̄'(s')              ∝ ∑_{a_other, o_other} w(a_other, o_other) · b_{a,o}(s')

    where pred_ego(s') = ∑_s T(s' | s, a_ego, a_other) b_ego(s), and h ranges over the
    ego actions the other agent thinks we might have taken. Each b_{a,o} is normalized
    before mixing.
    """
    c = _random_case(seed, ego_role, traced)
    ego_action_prior = c.ego_action_prior  # P(h), a random distribution
    other_action_probs = c.policy_probs(c.other_belief_estimate)  # π(a_other), passed in as given
    want = _reference_estimate_update(
        c, c.ego_role, c.ego_belief, c.other_belief_estimate, c.ego_observation, c.ego_action, other_action_probs, ego_action_prior
    )

    got = c.run(
        lambda ego_role: update_other_belief_estimate_with_observation_only(
            c.env_params,
            ego_belief=distrax.Categorical(probs=jnp.array(c.ego_belief)),
            other_belief_estimate=distrax.Categorical(probs=jnp.array(c.other_belief_estimate)),
            ego_observation=c.ego_observation,
            ego_action=c.ego_action,
            other_action_distribution=distrax.Categorical(probs=jnp.array(other_action_probs)),
            ego_role=ego_role,
            ego_action_prior=distrax.Categorical(probs=jnp.array(ego_action_prior)),
        )
    )
    _assert_close(got, want)


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_initial_second_order_ego_estimate_matches_reference(seed, ego_role, traced):
    """b̿_0(s) = ∑_{o_other} w(o_other) · [their initial estimate of us, given o_other](s),
    w(o_other) ∝ ∑_s P0(s) · O(o_ego, o_other | s, reset_action)."""
    c = _random_case(seed, ego_role, traced)
    reset_action = (0, 0)

    want = np.zeros(S)
    for other_observation in range(O):
        weight = sum(
            c.prior[state] * _joint_observation_prob(c.O_joint, state, reset_action, c.ego_role, c.ego_observation, other_observation)
            for state in range(S)
        )
        want += weight * _reference_initial_estimate(c, c.other_id, other_observation)

    got = c.run(lambda ego_role: initial_second_order_ego_estimate(c.env_params, c.ego_observation, ego_role=ego_role))
    _assert_close(got, _normalized(want))


@pytest.mark.parametrize("seed, ego_role, traced", RANDOM_CASES, ids=RANDOM_CASE_IDS)
def test_update_second_order_ego_estimate_matches_reference(seed, ego_role, traced):
    """b̿_e'(s) ∝ ∑_{a_other, o_other} w(a_other, o_other) · U(a_other, o_other)(s), where w is
    the first-order branch weight (OUR side) and U is the first-order estimate update run
    from THEIR seat: their belief b̄_o, their estimate of us b̿_e, their observation and
    action (a_other, o_other), our action as they predict it, and their action as they
    think we predict it."""
    c = _random_case(seed, ego_role, traced)
    other_action_probs = c.policy_probs(c.other_belief_estimate)  # their action, as we predict it
    ego_action_prior = c.ego_action_prior  # our action, as they predict it
    other_action_prior = c.other_action_prior  # their action, as they think we predict it

    weights = _reference_branch_weights(c, c.ego_role, c.ego_belief, c.ego_observation, c.ego_action, other_action_probs)
    want = np.zeros(S)
    for other_action in range(A):
        for other_observation in range(O):
            their_estimate_of_us = _reference_estimate_update(
                c,
                c.other_id,
                c.other_belief_estimate,
                c.second_order_ego_estimate,
                other_observation,
                other_action,
                ego_action_prior,
                other_action_prior,
            )
            want += weights[other_action, other_observation] * their_estimate_of_us

    got = c.run(
        lambda ego_role: update_second_order_ego_estimate_with_observation_only(
            c.env_params,
            ego_belief=distrax.Categorical(probs=jnp.array(c.ego_belief)),
            other_belief_estimate=distrax.Categorical(probs=jnp.array(c.other_belief_estimate)),
            second_order_ego_estimate=distrax.Categorical(probs=jnp.array(c.second_order_ego_estimate)),
            ego_observation=c.ego_observation,
            ego_action=c.ego_action,
            other_action_distribution=distrax.Categorical(probs=jnp.array(other_action_probs)),
            ego_role=ego_role,
            ego_action_prior=distrax.Categorical(probs=jnp.array(ego_action_prior)),
            other_action_prior=distrax.Categorical(probs=jnp.array(other_action_prior)),
        )
    )
    _assert_close(got, _normalized(want))
