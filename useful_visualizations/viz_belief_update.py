"""Interactively inspect the observation-only belief update of a single DecPOMDP agent.

This is a debugger for the exact update the stacked env runs on an act (see
``StackedSignificationDecPOMDP._agent_belief_updates``): given an initial belief and an
initial estimate of the partner's belief, you feed a hand-picked sequence of observations
(and, optionally, the ego agent's own actions) and watch how

  * ``true belief``      -- the ego agent's own belief b(s), and
  * ``estimated belief`` -- the ego agent's estimate of its partner's belief b_bar(s)

evolve. Both are driven by the real belief engine
(``CategoricalBeliefState.update_with_observation_only`` and
``update_other_belief_estimate_with_observation_only``) marginalizing the partner's
unobserved action through the partner's optimal policy -- no reimplementation.

HOW TO USE: edit the manual-specification block below (``OBSERVATIONS`` is the main knob;
``EGO_ACTIONS`` defaults to "wait" every step so the update is pure observation filtering),
then re-run. Each observation you list becomes one step in the trajectory.

The dashboard shows:
  * two [step x state] heatmaps -- the true-belief and estimated-belief trajectories, so the
    whole evolution (and any divergence between them) reads at a glance; each row is labeled
    with the observation/action that produced it.
  * an animated grouped-bar panel (true vs estimate over states) for the scrubbed step.
"""

from __future__ import annotations

import distrax
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.guessing_game import guessing_game_spec
from tools.belief_representations import CategoricalBeliefState

from . import _figures as F

# ======================= manual specification (edit me) ======================
# The observation sequence to feed the update, one int per step in [0, num_observations).
OBSERVATIONS = [1, 2, 1, 2]
# The ego agent's OWN action at each step (the update conditions on it). None -> "wait"
# (the last action) every step, i.e. a passive observer whose action never moves the world,
# so belief change is pure observation filtering. Otherwise a list aligned with OBSERVATIONS.
EGO_ACTIONS = None
# Which role the ego agent plays: 0 = presser (its belief drives its button), 1 = observer.
EGO_ROLE = 0
# Initial belief / initial partner-belief estimate. None -> the game's uniform prior over
# non-terminal states. Otherwise a length-S probability vector (need not be normalized).
INITIAL_BELIEF = None
INITIAL_ESTIMATE = None
# =============================================================================


def _action_labels(num_actions):
    """Guessing-game action names: press s0..s{A-2}, then wait."""
    return [f"press s{a}" for a in range(num_actions - 1)] + ["wait"]


def _normalize(vec, size):
    """Coerce a user-supplied vector (or None -> uniform prior handled by caller) to probs."""
    v = np.asarray(vec, dtype=float)
    assert v.shape == (size,), f"expected length-{size} vector, got shape {v.shape}"
    assert v.sum() > 0, "belief vector must have positive mass"
    return v / v.sum()


def _rollout():
    """Run the observation-only updates; return (steps_meta, true_traj, est_traj).

    true_traj / est_traj are [num_frames, S] arrays (row 0 is the initial belief/estimate,
    then one row per specified observation). steps_meta[i] is a label for frame i.
    """
    params, policies = guessing_game_spec()
    factory = CategoricalBeliefState(params)
    S = int(params.num_states)
    A = int(params.num_actions)
    partner_policy = policies[1 - EGO_ROLE]  # the OTHER role's optimal policy pi*

    prior = np.asarray(params.initial_belief_states[EGO_ROLE])
    init_belief = prior if INITIAL_BELIEF is None else _normalize(INITIAL_BELIEF, S)
    init_estimate = prior if INITIAL_ESTIMATE is None else _normalize(INITIAL_ESTIMATE, S)

    actions = EGO_ACTIONS if EGO_ACTIONS is not None else [A - 1] * len(OBSERVATIONS)
    assert len(actions) == len(OBSERVATIONS), "EGO_ACTIONS must align with OBSERVATIONS"

    action_labels = _action_labels(A)
    belief = distrax.Categorical(probs=init_belief)
    estimate = distrax.Categorical(probs=init_estimate)

    true_traj = [np.asarray(belief.probs)]
    est_traj = [np.asarray(estimate.probs)]
    meta = ["start (prior)"]

    for obs, act in zip(OBSERVATIONS, actions):
        # Both updates read the SAME pre-update (belief, estimate) -- mirror the env: the
        # true update marginalizes the partner action via the estimate, and the estimate
        # update refreshes the partner-belief estimate. Compute both, then advance.
        new_true = factory.update_with_observation_only(
            belief, estimate, obs, act, partner_policy, agent_id=EGO_ROLE
        )
        new_estimate = factory.update_other_belief_estimate_with_observation_only(
            estimate, obs, act, partner_policy, agent_id=EGO_ROLE
        )
        belief, estimate = new_true, new_estimate
        true_traj.append(np.asarray(belief.probs))
        est_traj.append(np.asarray(estimate.probs))
        meta.append(f"o={obs}, {action_labels[act]}")

    return meta, np.asarray(true_traj), np.asarray(est_traj), S


def render() -> str:
    meta, true_traj, est_traj, S = _rollout()
    state_labels = [f"s={s}" for s in range(S)]
    # y labels for the trajectory heatmaps: the step index + the observation that made it.
    row_labels = [f"t{i - 1}: {m}" if i > 0 else m for i, m in enumerate(meta)]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"colspan": 2, "type": "xy"}, None]],
        row_heights=[0.56, 0.44], vertical_spacing=0.14, horizontal_spacing=0.10,
        subplot_titles=(
            "true belief  b(s)  trajectory   [step × state]",
            "estimated belief  b̄(s)  (partner)  trajectory   [step × state]",
            "belief at scrubbed step:  true  vs  estimate  ▶",
        ),
    )

    # Row 1: whole-trajectory heatmaps (rows = steps top-down, cols = states).
    fig.add_trace(
        F.heatmap_trace(
            true_traj, x=state_labels, y=row_labels, colorscale=F.PROB_SCALE,
            zmin=0, zmax=1, hover="P", showscale=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        F.heatmap_trace(
            est_traj, x=state_labels, y=row_labels, colorscale=F.PROB_SCALE,
            zmin=0, zmax=1, hover="P", showscale=False,
        ),
        row=1, col=2,
    )
    for c in (1, 2):
        fig.update_yaxes(autorange="reversed", row=1, col=c)

    # Row 2: animated grouped bars (true vs estimate) for the scrubbed step.
    true_bar_idx = len(fig.data)
    fig.add_trace(
        go.Bar(x=state_labels, y=true_traj[0], name="true b(s)", marker_color=F.ACCENT),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=state_labels, y=est_traj[0], name="estimate b̄(s)", marker_color=F.ACCENT_2),
        row=2, col=1,
    )
    est_bar_idx = true_bar_idx + 1
    fig.update_yaxes(range=[0, 1.08], title_text="probability", row=2, col=1)

    frames, labels = [], []
    for i in range(len(meta)):
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Bar(x=state_labels, y=true_traj[i], name="true b(s)", marker_color=F.ACCENT),
                    go.Bar(x=state_labels, y=est_traj[i], name="estimate b̄(s)", marker_color=F.ACCENT_2),
                ],
                traces=[true_bar_idx, est_bar_idx],
            )
        )
        labels.append(row_labels[i])

    fig.update_layout(
        height=980,
        barmode="group",
        margin=dict(l=60, r=30, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0.5, xanchor="center"),
        title=dict(text=(
            f"Belief update inspector — ego role {EGO_ROLE}, "
            f"observations {OBSERVATIONS}<br>"
            f"<span style='font-size:13px;color:{F.MUTED}'>"
            f"true belief vs estimate of partner's belief, driven by the real "
            f"observation-only update</span>"
        )),
    )
    F.attach_slider(fig, frames, labels, slider_prefix="step ")
    return F.write(fig, "viz_belief_update")


if __name__ == "__main__":
    print(render())
