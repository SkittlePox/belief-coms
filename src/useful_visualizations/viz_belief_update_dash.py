"""Live, engine-backed belief-update inspector (Dash).

Every interaction runs the REAL belief engine -- no precompute, no JS reimplementation.
Click an observation (and choose the ego agent's own action), press "Apply step", and the
app calls ``CategoricalBeliefState.update_with_observation_only`` and
``update_other_belief_estimate_with_observation_only`` (partner action marginalized through
the partner's optimal policy) exactly as ``StackedSignificationDecPOMDP._agent_belief_updates``
does, then redraws:

  * true belief  b(s)      -- the ego agent's own belief, and
  * estimated belief b̄(s)  -- its estimate of the partner's belief,

as [step x state] trajectory heatmaps plus a grouped bar for the latest step. Undo pops the
last step; Reset (or changing role) re-initializes to the game's prior.

Run:  PYTHONPATH=. uv run python -m useful_visualizations.viz_belief_update_dash
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
OBS_OPTIONS = [{"label": f"observe o={o}", "value": o} for o in range(O)]
ACTION_LABELS = [f"press s{a}" for a in range(A - 1)] + ["wait"]
ACTION_OPTIONS = [{"label": lab, "value": a} for a, lab in enumerate(ACTION_LABELS)]


# ============================ pure engine + state helpers ============================

def step_belief(role, belief, estimate, obs, action):
    """One real observation-only update. Returns (new_belief, new_estimate) as np arrays.

    Mirrors StackedSignificationDecPOMDP._agent_belief_updates: the partner's unobserved
    action is marginalized through the OTHER role's optimal policy.
    """
    partner_policy = _POLICIES[1 - role]
    b = distrax.Categorical(probs=np.asarray(belief, dtype=float))
    e = distrax.Categorical(probs=np.asarray(estimate, dtype=float))
    new_true = _FACTORY.update_with_observation_only(
        b, e, int(obs), int(action), partner_policy, agent_id=int(role)
    )
    new_est = _FACTORY.update_other_belief_estimate_with_observation_only(
        e, int(obs), int(action), partner_policy, agent_id=int(role)
    )
    return np.asarray(new_true.probs), np.asarray(new_est.probs)


def initial_store(role=0):
    """A fresh trajectory: row 0 is the prior for both belief and estimate."""
    return {
        "role": int(role),
        "true": [_PRIOR.tolist()],
        "est": [_PRIOR.tolist()],
        "steps": [],  # list of {"obs": int, "action": int}
    }


def apply_step(store, obs, action):
    """Append one engine-computed step to the trajectory."""
    role = store["role"]
    new_true, new_est = step_belief(role, store["true"][-1], store["est"][-1], obs, action)
    out = {
        "role": role,
        "true": store["true"] + [new_true.tolist()],
        "est": store["est"] + [new_est.tolist()],
        "steps": store["steps"] + [{"obs": int(obs), "action": int(action)}],
    }
    return out


def undo(store):
    """Pop the last step (no-op at the initial prior)."""
    if not store["steps"]:
        return store
    return {
        "role": store["role"],
        "true": store["true"][:-1],
        "est": store["est"][:-1],
        "steps": store["steps"][:-1],
    }


def _row_labels(store):
    labels = ["start (prior)"]
    for i, s in enumerate(store["steps"]):
        labels.append(f"t{i}: o={s['obs']}, {ACTION_LABELS[s['action']]}")
    return labels


def _trajectory_heatmap(matrix, row_labels, title):
    fig = go.Figure(
        F.heatmap_trace(
            np.asarray(matrix), x=STATE_LABELS, y=row_labels, colorscale=F.PROB_SCALE,
            zmin=0, zmax=1, hover="P", showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title, height=420, margin=dict(l=90, r=20, t=50, b=40),
        template=None,
    )
    return fig


def figures_from_store(store):
    """(true-trajectory heatmap, estimate-trajectory heatmap, latest-step grouped bar)."""
    labels = _row_labels(store)
    true_fig = _trajectory_heatmap(store["true"], labels, "true belief  b(s)   [step × state]")
    est_fig = _trajectory_heatmap(store["est"], labels, "estimated belief  b̄(s)   [step × state]")

    bar = go.Figure()
    bar.add_bar(x=STATE_LABELS, y=store["true"][-1], name="true b(s)", marker_color=F.ACCENT)
    bar.add_bar(x=STATE_LABELS, y=store["est"][-1], name="estimate b̄(s)", marker_color=F.ACCENT_2)
    bar.update_layout(
        barmode="group", height=340, margin=dict(l=50, r=20, t=50, b=40),
        yaxis=dict(range=[0, 1.08], title="probability"),
        title=f"latest step: {labels[-1]}   (true vs estimate)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return true_fig, est_fig, bar


# ================================== Dash app ==================================

app = Dash(__name__)
app.title = "Belief update inspector"

_controls = html.Div(
    [
        html.Div([
            html.Label("ego role"),
            dcc.Dropdown(id="role", options=[{"label": "0 (presser)", "value": 0},
                                             {"label": "1 (observer)", "value": 1}],
                         value=0, clearable=False, style={"width": "150px"}),
        ]),
        html.Div([
            html.Label("observation"),
            dcc.Dropdown(id="obs", options=OBS_OPTIONS, value=0, clearable=False,
                         style={"width": "160px"}),
        ]),
        html.Div([
            html.Label("ego action"),
            dcc.Dropdown(id="action", options=ACTION_OPTIONS, value=A - 1, clearable=False,
                         style={"width": "160px"}),
        ]),
        html.Button("Apply step", id="apply", n_clicks=0, style={"height": "38px"}),
        html.Button("Undo", id="undo", n_clicks=0, style={"height": "38px"}),
        html.Button("Reset", id="reset", n_clicks=0, style={"height": "38px"}),
    ],
    style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "flexWrap": "wrap"},
)

app.layout = html.Div(
    [
        html.H2("Belief update inspector — live, engine-backed"),
        html.P("Every step runs the real observation-only update (partner action "
               "marginalized through the partner's optimal policy)."),
        _controls,
        dcc.Store(id="store", data=initial_store(0)),
        html.Div(
            [dcc.Graph(id="true_heat", style={"flex": "1 1 420px"}),
             dcc.Graph(id="est_heat", style={"flex": "1 1 420px"})],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        dcc.Graph(id="bar"),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
)


@app.callback(
    Output("store", "data"),
    Input("apply", "n_clicks"),
    Input("undo", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("role", "value"),
    State("obs", "value"),
    State("action", "value"),
    State("store", "data"),
    prevent_initial_call=True,
)
def _update_store(_apply, _undo, _reset, role, obs, action, store):
    trigger = ctx.triggered_id
    if trigger in ("reset", "role"):
        return initial_store(role)
    if trigger == "undo":
        return undo(store)
    if trigger == "apply":
        return apply_step(store, obs, action)
    return no_update


@app.callback(
    Output("true_heat", "figure"),
    Output("est_heat", "figure"),
    Output("bar", "figure"),
    Input("store", "data"),
)
def _redraw(store):
    return figures_from_store(store)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8050")),
            debug=False)
