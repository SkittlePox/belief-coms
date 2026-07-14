"""Live, engine-backed belief-update inspector (Dash).

Every interaction runs the REAL belief engine -- no precompute, no JS reimplementation.
You supply a JOINT step -- an observation AND an action for each of the two agents -- press
"Apply step", and the app advances BOTH agents' belief updates in lockstep, exactly as
``StackedSignificationDecPOMDP._agent_belief_updates`` does for each agent. Each agent's
update consumes only its OWN (observation, action); the partner's action is unobserved and
is marginalized through the partner's optimal policy.

Timing within one step: action -> state transitions -> observation. The action you supply
for an agent is the action IT takes; the observation you supply is what it then receives as
a consequence (the belief update transitions under the action, then conditions on the obs).

All FOUR beliefs are drawn as [step x state] trajectory heatmaps, paired as the two natural
sanity checks -- does each agent's estimate of its partner track the partner's real belief?

  * agent 0 true b0(s)      vs   agent 1's estimate of 0   b̄_{1->0}(s)
  * agent 1 true b1(s)      vs   agent 0's estimate of 1   b̄_{0->1}(s)

plus grouped bars for the latest step. Undo pops the last step; Reset re-initializes both
agents to the game's prior.

Run:  uv run python -m useful_visualizations.viz_belief_update_dash
then open http://127.0.0.1:8050  (set PORT / HOST env vars to override).

The heavy lifting lives in pure functions (``initial_store``, ``apply_step``, ``undo``,
``figures_from_store``) so the belief logic is testable without a browser; the Dash layout
and callbacks are thin wrappers.
"""

from __future__ import annotations

import os

import distrax
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from envs.guessing_game import guessing_game_spec
from tools.belief_representations import CategoricalBeliefState

from . import _figures as F

# --- The single-game engine, built once (params + per-role optimal policies). -----------
_PARAMS, _POLICIES = guessing_game_spec()
_FACTORY = CategoricalBeliefState(_PARAMS)
S = int(_PARAMS.num_states)
A = int(_PARAMS.num_actions)
O = _PARAMS.observation.shape[-1]
_PRIOR = np.asarray(_PARAMS.initial_belief_states[0])  # uniform over non-terminal states

STATE_LABELS = [f"s={s}" for s in range(S)]
# The last observation symbol is the dedicated "done" signal, emitted only in the
# terminal state (see guessing_game.build_observation_tensor).
DONE_OBS = O - 1
OBS_LABELS = [f"o={o}" for o in range(O - 1)] + [f"o={DONE_OBS} (done)"]
OBS_OPTIONS = [{"label": OBS_LABELS[o], "value": o} for o in range(O)]
ACTION_LABELS = [f"press s{a}" for a in range(A - 1)] + ["wait"]
ACTION_OPTIONS = [{"label": lab, "value": a} for a, lab in enumerate(ACTION_LABELS)]


# ============================ pure engine + state helpers ============================


def step_belief(role, belief, estimate, obs, action):
    """One real observation-only update for one agent. Returns (new_belief, new_estimate).

    Mirrors StackedSignificationDecPOMDP._agent_belief_updates: the partner's unobserved
    action is marginalized through the OTHER role's optimal policy, so only this agent's
    own ``obs`` and ``action`` enter the update.
    """
    partner_policy = _POLICIES[1 - role]
    b = distrax.Categorical(probs=np.asarray(belief, dtype=float))
    e = distrax.Categorical(probs=np.asarray(estimate, dtype=float))
    new_true = _FACTORY.update_with_observation_only(b, e, int(obs), int(action), partner_policy, agent_id=int(role))
    new_est = _FACTORY.update_other_belief_estimate_with_observation_only(e, int(obs), int(action), partner_policy, agent_id=int(role))
    return np.asarray(new_true.probs), np.asarray(new_est.probs)


def initial_store():
    """A fresh trajectory: row 0 is the prior for every belief and estimate.

    Keys are subject-indexed. ``true{i}`` is agent i's own belief; ``est{i}`` is agent i's
    estimate of its partner (so ``est0`` = b̄_{0->1}, ``est1`` = b̄_{1->0}).
    """
    return {
        "true0": [_PRIOR.tolist()],
        "est0": [_PRIOR.tolist()],
        "true1": [_PRIOR.tolist()],
        "est1": [_PRIOR.tolist()],
        "steps": [],  # list of {"o0","o1","a0","a1"}
        "note": "",  # last-action message (e.g. a rejected impossible observation)
    }


def _degenerate(*prob_arrays):
    """True if any posterior failed to normalize -- an impossible (zero-probability) obs.

    The engine returns a normalized categorical when the evidence has support; when the
    observation is impossible given the belief (0/0) the probs come back NaN / all-zero.
    """
    for p in prob_arrays:
        p = np.asarray(p, dtype=float)
        if not np.all(np.isfinite(p)) or abs(p.sum() - 1.0) > 1e-6:
            return True
    return False


def apply_step(store, o0, a0, o1, a1):
    """Append one engine-computed JOINT step: advance both agents from their own (o, a).

    If either agent's update is degenerate (the observation is impossible given that
    agent's current belief), the whole step is REJECTED -- nothing is appended (so the two
    agents' trajectories stay row-aligned) and ``note`` explains which agent was at fault.
    """
    nt0, ne0 = step_belief(0, store["true0"][-1], store["est0"][-1], o0, a0)
    nt1, ne1 = step_belief(1, store["true1"][-1], store["est1"][-1], o1, a1)

    bad = [i for i, ps in ((0, (nt0, ne0)), (1, (nt1, ne1))) if _degenerate(*ps)]
    if bad:
        who = " and ".join(f"agent {i}" for i in bad)
        return {
            **store,
            "note": (
                f"⚠ step rejected: {who} received an impossible observation given its current "
                f"belief (o0={_obs_str(o0)} after a0={ACTION_LABELS[a0]}, "
                f"o1={_obs_str(o1)} after a1={ACTION_LABELS[a1]}). "
                f"The 'done' symbol is only reachable via the button that transitions into the "
                f"done state."
            ),
        }

    return {
        "true0": store["true0"] + [nt0.tolist()],
        "est0": store["est0"] + [ne0.tolist()],
        "true1": store["true1"] + [nt1.tolist()],
        "est1": store["est1"] + [ne1.tolist()],
        "steps": store["steps"] + [{"o0": int(o0), "a0": int(a0), "o1": int(o1), "a1": int(a1)}],
        "note": "",
    }


def undo(store):
    """Pop the last step (no-op at the initial prior)."""
    if not store["steps"]:
        return {**store, "note": ""}
    return {
        "true0": store["true0"][:-1],
        "est0": store["est0"][:-1],
        "true1": store["true1"][:-1],
        "est1": store["est1"][:-1],
        "steps": store["steps"][:-1],
        "note": "",
    }


def _obs_str(o):
    return "done" if o == DONE_OBS else str(o)


def _row_labels(store):
    labels = ["start (prior)"]
    for i, s in enumerate(store["steps"]):
        labels.append(f"t{i}: a0={ACTION_LABELS[s['a0']]}, o0={_obs_str(s['o0'])} | " f"a1={ACTION_LABELS[s['a1']]}, o1={_obs_str(s['o1'])}")
    return labels


def _trajectory_heatmap(matrix, row_labels, title, color):
    fig = go.Figure(
        F.heatmap_trace(
            np.asarray(matrix),
            x=STATE_LABELS,
            y=row_labels,
            colorscale=F.PROB_SCALE,
            zmin=0,
            zmax=1,
            hover="P",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=title, font=dict(color=color)),
        height=380,
        margin=dict(l=90, r=20, t=50, b=40),
        template=None,
    )
    return fig


def _latest_bar(store, own_key, est_key, title):
    bar = go.Figure()
    bar.add_bar(x=STATE_LABELS, y=store[own_key][-1], name="true b(s)", marker_color=F.ACCENT)
    bar.add_bar(x=STATE_LABELS, y=store[est_key][-1], name="partner's estimate b̄(s)", marker_color=F.ACCENT_2)
    bar.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=50, r=20, t=50, b=40),
        yaxis=dict(range=[0, 1.08], title="probability"),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return bar


def figures_from_store(store):
    """Four trajectory heatmaps (paired true vs partner-estimate) + two latest-step bars."""
    labels = _row_labels(store)
    # Pair 1: agent 0's own belief vs agent 1's estimate OF agent 0 (est1).
    true0 = _trajectory_heatmap(store["true0"], labels, "agent 0 true  b₀(s)", F.ACCENT)
    est_of_0 = _trajectory_heatmap(store["est1"], labels, "agent 1's estimate of 0  b̄₁→₀(s)", F.ACCENT_2)
    # Pair 2: agent 1's own belief vs agent 0's estimate OF agent 1 (est0).
    true1 = _trajectory_heatmap(store["true1"], labels, "agent 1 true  b₁(s)", F.ACCENT)
    est_of_1 = _trajectory_heatmap(store["est0"], labels, "agent 0's estimate of 1  b̄₀→₁(s)", F.ACCENT_2)

    bar0 = _latest_bar(store, "true0", "est1", "latest: agent 0 — true vs 1's estimate of 0")
    bar1 = _latest_bar(store, "true1", "est0", "latest: agent 1 — true vs 0's estimate of 1")
    return true0, est_of_0, true1, est_of_1, bar0, bar1


# ================================== Dash app ==================================

app = Dash(__name__)
app.title = "Belief update inspector"


def _agent_inputs(idx):
    return html.Div(
        [
            html.Label(f"agent {idx}", style={"fontWeight": "600"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("action"),
                            dcc.Dropdown(id=f"action{idx}", options=ACTION_OPTIONS, value=A - 1, clearable=False, style={"width": "150px"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("observation"),
                            dcc.Dropdown(id=f"obs{idx}", options=OBS_OPTIONS, value=0, clearable=False, style={"width": "110px"}),
                        ]
                    ),
                ],
                style={"display": "flex", "gap": "10px"},
            ),
        ]
    )


_controls = html.Div(
    [
        _agent_inputs(0),
        _agent_inputs(1),
        html.Div(
            [
                html.Button("Apply step", id="apply", n_clicks=0, style={"height": "38px"}),
                html.Button("Undo", id="undo", n_clicks=0, style={"height": "38px"}),
                html.Button("Reset", id="reset", n_clicks=0, style={"height": "38px"}),
            ],
            style={"display": "flex", "gap": "8px", "alignItems": "flex-end"},
        ),
    ],
    style={"display": "flex", "gap": "28px", "alignItems": "flex-end", "flexWrap": "wrap"},
)


def _heat_row(a_id, b_id):
    return html.Div(
        [dcc.Graph(id=a_id, style={"flex": "1 1 420px"}), dcc.Graph(id=b_id, style={"flex": "1 1 420px"})],
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
    )


app.layout = html.Div(
    [
        html.H2("Belief update inspector — live, engine-backed (joint step)"),
        html.P(
            "Supply an action + resulting observation for each agent, then Apply. "
            "Both agents' beliefs advance together; each partner's action is "
            "marginalized through their optimal policy. Timing within a step: "
            "action → state transitions → observation."
        ),
        _controls,
        dcc.Store(id="store", data=initial_store()),
        html.Div(id="banner", style={"minHeight": "0", "margin": "8px 0"}),
        html.H4("agent 0's belief vs how agent 1 models it"),
        _heat_row("true0_heat", "est_of_0_heat"),
        html.H4("agent 1's belief vs how agent 0 models it"),
        _heat_row("true1_heat", "est_of_1_heat"),
        html.Div(
            [dcc.Graph(id="bar0", style={"flex": "1 1 420px"}), dcc.Graph(id="bar1", style={"flex": "1 1 420px"})],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
)


@app.callback(
    Output("store", "data"),
    Input("apply", "n_clicks"),
    Input("undo", "n_clicks"),
    Input("reset", "n_clicks"),
    State("obs0", "value"),
    State("action0", "value"),
    State("obs1", "value"),
    State("action1", "value"),
    State("store", "data"),
    prevent_initial_call=True,
)
def _update_store(_apply, _undo, _reset, o0, a0, o1, a1, store):
    trigger = ctx.triggered_id
    if trigger == "reset":
        return initial_store()
    if trigger == "undo":
        return undo(store)
    if trigger == "apply":
        return apply_step(store, o0, a0, o1, a1)
    return no_update


@app.callback(Output("banner", "children"), Input("store", "data"))
def _banner(store):
    note = store.get("note", "")
    if not note:
        return None
    return html.Div(
        note,
        style={
            "padding": "10px 14px",
            "borderRadius": "6px",
            "background": "#fff4e5",
            "border": "1px solid #f0b775",
            "color": "#8a4b00",
        },
    )


@app.callback(
    Output("true0_heat", "figure"),
    Output("est_of_0_heat", "figure"),
    Output("true1_heat", "figure"),
    Output("est_of_1_heat", "figure"),
    Output("bar0", "figure"),
    Output("bar1", "figure"),
    Input("store", "data"),
)
def _redraw(store):
    return figures_from_store(store)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050")), debug=False)
