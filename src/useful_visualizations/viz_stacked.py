"""Visualize the StackedSignificationDecPOMDP: a full state decode plus a scrubbable
temporal trace of ``step_env``.

We assemble the env exactly like ``stacked_signification_decpomdp.py``'s ``__main__``
(guessing game, ``a_to_b_thrice`` so the round cursor visibly walks 0,0,1,1,2,2, and a
2-step episode horizon so a boundary appears), then drive ``step_env`` with the same
scripted stand-in inputs that harness uses -- agents are not learned here; this is
inspection. Each returned ``StackedSignificationState`` becomes one slider frame.

The dashboard decodes, per step:

  * Routing -- for each game (column) the two agents (rows) and their roles, so the
    dyadic pairing is visible.
  * World -- each game's true DecPOMDP state.
  * Scheduler -- the four counters as bars; the stage / round / episode / ACT /
    BOUNDARY read out in the figure title, which updates as you scrub.
  * Beliefs -- ``true_agent_belief_states`` and ``estimated_agent_belief_states`` as
    two [agent, state] probability heatmaps (subject-indexed: "what i believes" vs
    "what i's partner thinks i believes"), so belief adoption on belief-stages and
    Bayesian updates on acts are both visible.
  * Reward -- ``last_agent_rewards`` (zero except on acts).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stacked_signification_decpomdp import (
    StackedSignificationDecPOMDP,
    UTTERANCE_STAGE,
)
from envs.env_assembly import assemble_environments
from envs.guessing_game import guessing_game_spec
from routing import simple_routing_fn
from communication_scheme import a_to_b_thrice_scheme_fn

from . import _figures as F

NUM_AGENTS = 6
NUM_STEPS = 12  # two acts (steps 6 & 12); the 2nd act hits the horizon -> a boundary

# Two-tone discrete scale for roles in the routing panel.
ROLE_SCALE = [[0.0, F.ACCENT], [0.5, F.ACCENT], [0.5, F.ACCENT_2], [1.0, F.ACCENT_2]]
# Warm light->orange scale for the "actions taken" panel (distinct from the teal beliefs).
ACTION_SCALE = [[0.0, "#fff5eb"], [1.0, F.ACCENT_2]]


def _action_labels(num_actions):
    """Guessing-game action names: press s0..s{A-2}, then wait."""
    return [f"press s{a}" for a in range(num_actions - 1)] + ["wait"]


def _build_env():
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])
    env = StackedSignificationDecPOMDP(
        num_agents=NUM_AGENTS,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(
            num_agents=NUM_AGENTS, agents_per_game=2, underlying_env_steps_per_episode=2
        ),
        communication_scheme_fn=a_to_b_thrice_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )
    return env


def _rollout(env):
    """Drive NUM_STEPS step_env calls with scripted inputs; return per-step snapshots."""
    # reset / step_env now return (state, observations); we read the state.
    state, _obs = env.reset(jax.random.key(0))
    num_states = int(state.true_agent_belief_states.shape[-1])

    # Scripted stand-ins (mirror the module's __main__): every agent utters all-ones,
    # proposes a valid one-hot belief, and carries a valid full-support post-utterance
    # estimate (a padding-state one-hot would break the act's marginalization).
    utt = jnp.ones((NUM_AGENTS, env.utterance_action_dim))
    valid_belief = jnp.zeros((NUM_AGENTS, num_states)).at[:, 0].set(1.0)
    other_est = state.true_agent_belief_states.at[:, 0].add(0.5)
    other_est = other_est / other_est.sum(axis=-1, keepdims=True)

    snaps = [_snapshot(state)]
    for t in range(NUM_STEPS):
        state, _obs = env.step_env(jax.random.key(t), state, utt, other_est, valid_belief)
        snaps.append(_snapshot(state))
    return snaps


def _snapshot(state):
    """Pull the plottable fields out of a StackedSignificationState as numpy."""
    return dict(
        agent_game=np.asarray(state.agent_game_assignment),
        agent_role=np.asarray(state.agent_role_assignment),
        game_states=np.asarray(state.game_states),
        stage=int(state.communicative_round_stage),
        round_cursor=int(state.underlying_communication_round_iterator),
        total_rounds=int(state.active_total_num_rounds),
        env_iter=int(state.underlying_env_iteration),
        cum_env=int(state.cumulative_env_iteration),
        cum_round=int(state.cumulative_communication_round_iterator),
        episode=int(state.episode_index),
        true_beliefs=np.asarray(state.true_agent_belief_states),
        est_beliefs=np.asarray(state.estimated_agent_belief_states),
        rewards=np.asarray(state.last_agent_rewards),
        actions=np.asarray(state.last_agent_actions),  # -1 on non-act steps
    )


def _routing_matrix(snap, num_games):
    """[agent, game] = role if the agent plays that game, else NaN (a gap)."""
    m = np.full((NUM_AGENTS, num_games), np.nan)
    for a in range(NUM_AGENTS):
        m[a, snap["agent_game"][a]] = snap["agent_role"][a]
    return m


def _actions_matrix(snap, num_actions):
    """[agent, action] one-hot at the action each agent took; all-zero where -1 (no act)."""
    m = np.zeros((NUM_AGENTS, num_actions))
    for a in range(NUM_AGENTS):
        act = int(snap["actions"][a])
        if act >= 0:
            m[a, act] = 1.0
    return m


def _traces(snap, num_games, num_states, num_actions):
    """The seven animatable traces, in a fixed order (matches the subplot cells)."""
    agent_labels = [f"agent {i}" for i in range(NUM_AGENTS)]
    game_labels = [f"game {g}" for g in range(num_games)]
    state_labels = [f"s={s}" for s in range(num_states)]
    action_labels = _action_labels(num_actions)

    routing = go.Heatmap(
        z=_routing_matrix(snap, num_games), x=game_labels, y=agent_labels,
        colorscale=ROLE_SCALE, zmin=0, zmax=1, showscale=False,
        text=[["" if np.isnan(v) else f"role {int(v)}" for v in row]
              for row in _routing_matrix(snap, num_games)],
        texttemplate="%{text}", textfont=dict(size=10), hoverongaps=False,
        hovertemplate="%{y} in %{x}: %{text}<extra></extra>",
    )
    world = F.heatmap_trace(
        snap["game_states"][None, :].astype(float), x=game_labels, y=["state"],
        colorscale=F.PROB_SCALE, zmin=0, zmax=num_states - 1,
        text=True, text_fmt="s={:.0f}", showscale=False, hover="state",
    )
    counters = F.bar_trace(
        [snap["env_iter"], snap["cum_env"], snap["round_cursor"], snap["cum_round"]],
        labels=["env_iter", "cum_env", "round", "cum_round"], color=F.ACCENT,
    )
    true_b = F.heatmap_trace(
        snap["true_beliefs"], x=state_labels, y=agent_labels,
        colorscale=F.PROB_SCALE, zmin=0, zmax=1, showscale=False, hover="P",
    )
    est_b = F.heatmap_trace(
        snap["est_beliefs"], x=state_labels, y=agent_labels,
        colorscale=F.PROB_SCALE, zmin=0, zmax=1, showscale=False, hover="P",
    )
    actions_m = _actions_matrix(snap, num_actions)
    actions = go.Heatmap(
        z=actions_m, x=action_labels, y=agent_labels,
        colorscale=ACTION_SCALE, zmin=0, zmax=1, showscale=False,
        text=[[action_labels[j] if actions_m[i, j] > 0 else "" for j in range(num_actions)]
              for i in range(NUM_AGENTS)],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} took %{x}<extra></extra>",
    )
    rewards = go.Bar(
        x=agent_labels, y=snap["rewards"], showlegend=False,
        marker_color=[F.ACCENT_2 if r >= 0 else "#c0392b" for r in snap["rewards"]],
        text=[f"{r:+.2f}" for r in snap["rewards"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:+.3f}<extra></extra>",
    )
    return [routing, world, counters, true_b, est_b, actions, rewards]


def _title(step, snap, is_act, is_boundary):
    stage = "UTTERANCE" if snap["stage"] == UTTERANCE_STAGE else "BELIEF"
    tags = ""
    if is_act:
        tags += "   · ACT (world steps)"
    if is_boundary:
        tags += "   · EPISODE BOUNDARY (re-route)"
    head = "reset" if step < 0 else f"step {step}"
    return (
        f"StackedSignificationDecPOMDP — {head}<br>"
        f"<span style='font-size:13px;color:{F.MUTED}'>"
        f"stage={stage}   round {snap['round_cursor'] + 1}/{snap['total_rounds']}   "
        f"env_iter={snap['env_iter']}   episode={snap['episode']}{tags}</span>"
    )


def render() -> str:
    env = _build_env()
    snaps = _rollout(env)
    num_games = NUM_AGENTS // 2
    num_states = snaps[0]["true_beliefs"].shape[-1]
    num_actions = int(env.all_env_parameters.transition.shape[2])  # padded action count

    # Per-step flags inferred from consecutive snapshots.
    is_act = [False] + [snaps[i]["cum_env"] > snaps[i - 1]["cum_env"] for i in range(1, len(snaps))]
    is_boundary = [False] + [snaps[i]["episode"] > snaps[i - 1]["episode"] for i in range(1, len(snaps))]

    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"colspan": 3, "type": "xy"}, None, None]],
        row_heights=[0.30, 0.44, 0.26], vertical_spacing=0.11, horizontal_spacing=0.08,
        subplot_titles=(
            "routing — agents ▸ (game, role)",
            "world — each game's true state",
            "scheduler counters",
            "true beliefs  b_i(s)   [agent × state]",
            "estimated beliefs  (estimate ABOUT agent i)",
            "actions taken  [agent × action]  (lit only on ACT steps)",
            "last agent rewards  (nonzero only on acts)",
        ),
    )

    base = _traces(snaps[0], num_games, num_states, num_actions)
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1)]
    for trace, (r, c) in zip(base, positions):
        fig.add_trace(trace, row=r, col=c)

    # Grids read top-down (agent 0 on top); bar axes get sensible ranges.
    for (r, c) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]:
        fig.update_yaxes(autorange="reversed", row=r, col=c)
    # Headroom so the outside value labels on the bars don't clip.
    counter_max = max(
        max(s["env_iter"], s["cum_env"], s["round_cursor"], s["cum_round"]) for s in snaps
    )
    fig.update_yaxes(title_text="count", range=[0, counter_max * 1.25 + 1], row=1, col=3)
    fig.update_yaxes(title_text="reward", range=[-1.4, 1.4], row=3, col=1)

    frames, labels = [], []
    for i, snap in enumerate(snaps):
        step = i - 1  # snap 0 is the reset
        frames.append(
            go.Frame(
                name=str(i),
                data=_traces(snap, num_games, num_states, num_actions),
                traces=[0, 1, 2, 3, 4, 5, 6],
                layout=go.Layout(title=dict(text=_title(step, snap, is_act[i], is_boundary[i]))),
            )
        )
        tag = " ★act" if is_act[i] else ""
        tag += " ⟲bdry" if is_boundary[i] else ""
        labels.append(("reset" if step < 0 else f"step {step}") + tag)

    fig.update_layout(
        height=1080,
        margin=dict(l=60, r=30, t=90, b=60),
        title=dict(text=_title(-1, snaps[0], False, False)),
    )
    F.attach_slider(fig, frames, labels, slider_prefix="")
    return F.write(fig, "viz_stacked")


if __name__ == "__main__":
    print(render())
