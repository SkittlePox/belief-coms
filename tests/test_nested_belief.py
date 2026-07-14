"""Tests for tools/nested_belief.py.

Run with::

    uv run pytest tests/test_nested_belief.py

These are self-contained. The accuracy of the tower against EXACT recursive inference is
checked separately, in the memo-decpomdp repo, which has the history-enumerating reference
model (itself validated against a brute-force enumeration).
"""

import numpy as np
import pytest

from envs.guessing_game import guessing_game_spec
from tools.nested_belief import build_nested_belief_step

PARAMS, _ = guessing_game_spec()
WAIT = 3

# role_0_optimal_policy is probability matching, so it puts zero mass on the wait action
# (the belief never has mass on the terminal state). The optimal presser NEVER waits, so
# any history in which it does is OFF-POLICY -- see the module docstring, and
# test_off_policy_histories_drift_silently below.
ON_POLICY = [
    (0, [1, 0], [0]),
    (0, [1, 2], [2]),
    (0, [2, 1], [1]),
    (0, [0, 1], [1]),
]


def run_tower(model, observations, actions):
    tower = model.initial_tower(observations[0])
    for action, observation in zip(actions, observations[1:]):
        tower = model(tower, action, observation)
    return tower


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("ego_role, observations, actions", ON_POLICY)
def test_every_level_stays_a_distribution(depth, ego_role, observations, actions):
    model = build_nested_belief_step(PARAMS, ego_role=ego_role, depth=depth)
    tower = run_tower(model, observations, actions)

    assert len(tower) == depth + 1
    for level, belief in enumerate(tower):
        assert np.isfinite(belief).all(), level
        assert belief.sum() == pytest.approx(1.0, abs=1e-4), level
        assert (belief >= -1e-6).all(), level


def test_the_tower_alternates_perspective():
    """bel[0] is mine and bel[1] is theirs, and they must not collapse into each other.

    bel[0] is conditioned on the ego's own observation. bel[1] is not -- they did not see
    it -- and neither is bel[2], their estimate of my belief.
    """
    model = build_nested_belief_step(PARAMS, ego_role=0, depth=2)
    tower = model.initial_tower(1)   # the ego saw symbol 1, so it rules out state 1

    np.testing.assert_allclose(tower[0], [0.5, 0.0, 0.5, 0.0], atol=1e-4)
    # They did not see symbol 1, so they cannot have ruled out state 1...
    assert tower[1][1] > 0.15, tower[1]
    # ...and their estimate of MY belief cannot have ruled it out either.
    assert tower[2][1] > 0.15, tower[2]


def test_depth_2_is_where_their_model_of_my_action_stops_being_uniform():
    """The point of the tower: depth 2 changes bel[1] and depth 1 cannot.

    At depth 1 the other agent's model of MY action bottoms out at uniform. At depth 2 it
    is pi_me(bel[2]) -- driven by their estimate of my belief -- so bel[1] must differ.
    """
    obs, acts = [1, 2], [2]
    shallow = run_tower(build_nested_belief_step(PARAMS, ego_role=0, depth=1), obs, acts)
    deeper = run_tower(build_nested_belief_step(PARAMS, ego_role=0, depth=2), obs, acts)

    assert not np.allclose(shallow[1], deeper[1], atol=1e-3), (shallow[1], deeper[1])


def test_extra_depth_is_inert_when_the_modeled_action_is_inert():
    """depth=3 must equal depth=2 in THIS game, and that is a feature.

    Level 3 exists to predict role 1's action. In the guessing game role 1's action enters
    neither the transition nor the observation -- agent 1 is inert -- so refining the model
    of it cannot change anything, and the tower correctly reports that it does not. A depth
    knob that kept "improving" here would be computing noise. Expect depth to matter in a
    game where both agents' actions do.
    """
    obs, acts = [1, 2], [2]
    d2 = run_tower(build_nested_belief_step(PARAMS, ego_role=0, depth=2), obs, acts)
    d3 = run_tower(build_nested_belief_step(PARAMS, ego_role=0, depth=3), obs, acts)

    for level in range(3):
        np.testing.assert_allclose(d2[level], d3[level], atol=1e-4)


def test_off_policy_histories_drift_silently():
    """The failure mode to know about: off-policy histories do not raise, they drift.

    The nested levels never condition on the ego's ACTUAL action -- they only marginalize
    it under the policy they attribute to the ego -- so an off-policy action can never
    contradict them. Here the presser waits twice, which its own optimal policy says it
    never does. The tower keeps returning clean, normalized, confidently wrong beliefs.

    This test does not assert the estimate is good. It asserts the silence, so that the
    behaviour is on the record.
    """
    model = build_nested_belief_step(PARAMS, ego_role=0, depth=2)
    tower = run_tower(model, [1, 2, 2], [WAIT, WAIT])   # off-policy: the presser never waits

    for belief in tower:
        assert np.isclose(belief.sum(), 1.0)   # no complaint, no NaN, no zero mass


def test_initial_tower_uses_the_reset_observation():
    """Seeding must not start every level at the prior.

    At t=0 the other agent has already seen ITS reset observation, correlated with the
    ego's through the state. So bel[1] has already moved off the uniform prior. Starting
    the tower at the environment prior throws a whole step of evidence away.
    """
    model = build_nested_belief_step(PARAMS, ego_role=0, depth=2)
    prior = np.asarray(PARAMS.initial_state_distribution)
    tower = model.initial_tower(1)

    assert not np.allclose(tower[1], prior, atol=1e-3), tower[1]


def _asymmetric_env():
    """A deliberately hostile FlexibleEnvParams: nothing like the guessing game.

    |S| != |A| != |O|; BOTH agents' actions drive the transition (the state only advances
    when they agree); observations are correlated across agents (not an outer product) and
    action-dependent; and the state prior is skewed rather than uniform.
    """
    import jax.numpy as jnp
    from envs.flexible_env import FlexibleEnvParams

    S, A, O, N = 3, 2, 2, 2
    rng = np.random.default_rng(0)

    T = np.zeros((S, A, A, S))
    for s in range(S):
        for a0 in range(A):
            for a1 in range(A):
                T[s, a0, a1, (s + 1) % S if a0 == a1 else s] = 1.0

    obs = rng.random((S, A, A, O, O)) + 0.05
    obs /= obs.sum(axis=(-1, -2), keepdims=True)

    return FlexibleEnvParams(
        transition=jnp.asarray(T),
        observation=jnp.asarray(obs),
        reward=jnp.asarray(rng.random((N, S, A, A, S))),
        num_actions=jnp.array(A),
        num_states=jnp.array(S),
        initial_state_distribution=jnp.array([0.6, 0.3, 0.1]),
        terminal_mask=jnp.zeros(S),
    )


def test_runs_on_an_arbitrary_flexible_env():
    """Nothing about the guessing game may be baked in."""
    import jax
    import jax.numpy as jnp

    params = _asymmetric_env()

    @jax.jit
    def policy_weight(role, s, a):
        # jnp, not np: role/s/a arrive as traced indices.
        w = jnp.array([[[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]],
                       [[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]])
        return w[role, s, a]

    model = build_nested_belief_step(params, ego_role=0, depth=2,
                                     policy_weight=policy_weight)
    tower = run_tower(model, [0, 1, 0], [0, 1])

    assert len(tower) == 3
    for belief in tower:
        assert belief.shape == (3,)
        assert np.isfinite(belief).all()
        assert belief.sum() == pytest.approx(1.0, abs=1e-4)


def test_default_policy_weight_is_refused_when_it_is_meaningless():
    """1[s == a] only makes sense when action indices ARE state indices.

    On any other environment it would leave some states with no action, and every level of
    the tower would rest on a policy that means nothing. Fail loudly rather than return a
    confident wrong answer -- which is exactly what it used to do.
    """
    with pytest.raises(ValueError, match="action indices are state indices"):
        build_nested_belief_step(_asymmetric_env(), ego_role=0, depth=2)
