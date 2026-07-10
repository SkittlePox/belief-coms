"""Tests for tools/belief_representations.py.

Run either directly (matches this repo's ``python -m`` convention)::

    PYTHONPATH=. uv run python -m tools.test_belief_representations

or under pytest::

    PYTHONPATH=. uv run pytest src/tools/test_belief_representations.py
"""

import distrax
import jax.numpy as jnp

from envs.guessing_game import (
    guessing_game_spec,
    role_0_optimal_policy,
    role_1_optimal_policy,
)
from tools.belief_representations import CategoricalBeliefState


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
    belief_factory = CategoricalBeliefState(params)

    # Uniform prior over the three referent states {0, 1, 2} (state 3 is terminal).
    uniform_referent = jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0])

    ego_action = 0  # agent 0 pressed button 0
    ego_observation = 0  # agent 0 saw referent symbol 0
    partner_policy = role_1_optimal_policy  # agent 1's policy

    estimate = belief_factory.update_other_belief_estimate_with_observation_only(
        distrax.Categorical(probs=uniform_referent),  # ego's prior estimate of agent 1
        ego_observation,
        ego_action,
        partner_policy,
        agent_id=0,
    )

    # Reference: agent 1's ACTUAL belief had it seen symbol 1 (the observation it in fact
    # received). Agent 1 marginalizes over agent 0's unseen action, so it cannot rule out
    # state 0. This is the belief agent 0's estimate is supposed to be consistent with.
    agent1_true_belief = belief_factory.update_with_observation_only(
        distrax.Categorical(probs=uniform_referent),  # agent 1's own prior
        distrax.Categorical(probs=uniform_referent),  # agent 1's estimate of agent 0
        1,  # agent 1 saw symbol 1
        3,  # agent 1's own action (wait)
        role_0_optimal_policy,  # agent 0's policy, marginalized over by agent 1
        agent_id=1,
    )

    # Agent 1 genuinely believes state 0 is possible (sanity-check the reference).
    assert agent1_true_belief.probs[0] > 0.0, (
        "reference: agent 1 (not seeing agent 0's action) should keep state-0 mass; "
        f"got {agent1_true_belief.probs}"
    )

    # The estimate must not rule out state 0 that agent 1 itself cannot rule out.
    assert estimate.probs[0] > 1e-6, (
        "agent 0's estimate of agent 1's belief wrongly assigns zero probability to "
        f"state 0 (estimate={estimate.probs}), even though agent 1 -- which never "
        f"observed agent 0's button press -- believes state 0 with probability "
        f"{float(agent1_true_belief.probs[0]):.3f}. The estimate conditions on agent "
        "0's own action, which agent 1 did not observe."
    )


if __name__ == "__main__":
    test_other_belief_estimate_keeps_states_the_partner_cannot_rule_out()
    print("ok: other-belief-estimate keeps partner-plausible states")
