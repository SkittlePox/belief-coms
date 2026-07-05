"""Visualize the StackedSignificationDecPOMDP: a full state decode plus a scrubbable
temporal trace of ``step_env``.

We assemble the env exactly like ``stacked_signification_decpomdp.py``'s ``__main__``
(guessing game, ``b_to_a`` -- a single round per block, so B speaks then an act lands
every two steps -- and a 2-step episode horizon so boundaries appear), then drive
``step_env`` with the same
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
  * Act I/O -- ``last_agent_actions`` and ``last_agent_observations`` as [agent x k]
    one-hot heatmaps (lit only on acts), alongside ``last_agent_rewards`` (zero except
    on acts).
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
from communication_scheme import b_to_a_scheme_fn

from . import _figures as F

NUM_AGENTS = 2  # one dyadic game
NUM_STEPS = 12  # b_to_a: 1 round/block -> an act every 2 steps (6 acts); horizon 4 -> boundary every ~3 acts

# Two-tone discrete scale for roles in the routing panel.
ROLE_SCALE = [[0.0, F.ACCENT], [0.5, F.ACCENT], [0.5, F.ACCENT_2], [1.0, F.ACCENT_2]]
# Warm light->orange scale for the "actions taken" panel (distinct from the teal beliefs).
ACTION_SCALE = [[0.0, "#fff5eb"], [1.0, F.ACCENT_2]]
# Cool violet scale for the "observations received" panel (distinct from beliefs & actions).
OBS_SCALE = [[0.0, "#f2eef7"], [1.0, "#8856a7"]]


def _action_labels(num_actions):
    """Guessing-game action names: press s0..s{A-2}, then wait."""
    return [f"press s{a}" for a in range(num_actions - 1)] + ["wait"]


def _obs_labels(num_obs):
    """Observation names o=0..o={O-1}."""
    return [f"o={o}" for o in range(num_obs)]


def _build_env():
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])
    env = StackedSignificationDecPOMDP(
        num_agents=NUM_AGENTS,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(
            num_agents=NUM_AGENTS, agents_per_game=2, underlying_env_steps_per_episode=4
        ),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )
    return env


def _rollout(env):
    """Drive NUM_STEPS steps with scripted inputs; return the per-frame snapshots.

    Act steps are split into two frames: a pre-act substate (communication resolved, world
    about to step) and the post-act state (world stepped). Communication-only steps stay a
    single frame. Each snapshot is tagged with its step index and phase for labeling.
    """
    # reset returns (state, observations); step_env_with_substate returns
    # (pre_act_state, state, observations, rewards). We read everything we plot off the
    # states, so we drop obs and rewards.
    state, _obs = env.reset(jax.random.key(0))
    num_states = int(state.true_agent_belief_states.shape[-1])

    # Scripted stand-ins (mirror the module's __main__): every agent utters all-ones,
    # proposes a valid one-hot belief, and carries a valid full-support post-utterance
    # estimate (a padding-state one-hot would break the act's marginalization).
    utt = jnp.ones((NUM_AGENTS, env.utterance_action_dim))
    valid_belief = jnp.zeros((NUM_AGENTS, num_states)).at[:, 0].set(1.0)
    other_est = state.true_agent_belief_states.at[:, 0].add(0.5)
    other_est = other_est / other_est.sum(axis=-1, keepdims=True)

    def tagged(state_, step, phase):
        snap = _snapshot(state_)
        snap["step"] = step
        snap["phase"] = phase
        return snap

    # The reset frame is special: with skip_first_communication_step=True it has already
    # acted once inside reset() (an act that is not surfaced as a separate substate).
    snaps = [tagged(state, -1, "reset")]
    for t in range(NUM_STEPS):
        pre_act, state, _obs, _rewards = env.step_env_with_substate(
            jax.random.key(t), state, utt, other_est, valid_belief
        )
        # An act step advances cumulative_env_iteration; only then is the pre-act substate
        # (communication resolved, world not yet stepped) a distinct, informative frame, so
        # we splice it in as sub-frame "a" ahead of the post-act world-step frame "b".
        is_act = int(state.cumulative_env_iteration) > int(pre_act.cumulative_env_iteration)
        if is_act:
            snaps.append(tagged(pre_act, t, "pre_act"))
        snaps.append(tagged(state, t, "act" if is_act else "comm"))
    return snaps


def _snapshot(state):
    """Pull the plottable fields out of a StackedSignificationState as numpy."""
    return dict(
        agent_game=np.asarray(state.agent_game_assignment),
        agent_role=np.asarray(state.agent_role_assignment),
        game_states=np.asarray(state.game_states),
        stage=int(state.communicative_round_stage),
        round_cursor=int(state.communication_round_iterator),
        total_rounds=int(state.active_total_num_rounds),
        env_iter=int(state.underlying_env_iteration),
        cum_env=int(state.cumulative_env_iteration),
        cum_round=int(state.cumulative_communication_round_iterator),
        episode=int(state.episode_index),
        true_beliefs=np.asarray(state.true_agent_belief_states),
        est_beliefs=np.asarray(state.estimated_agent_belief_states),
        rewards=np.asarray(state.last_agent_rewards),
        actions=np.asarray(state.last_agent_actions),  # -1 on non-act steps
        observations=np.asarray(state.last_agent_observations),  # -1 on non-act steps
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


def _obs_matrix(snap, num_obs):
    """[agent, obs] one-hot at the observation each agent drew; all-zero where -1 (no act)."""
    m = np.zeros((NUM_AGENTS, num_obs))
    for a in range(NUM_AGENTS):
        o = int(snap["observations"][a])
        if o >= 0:
            m[a, o] = 1.0
    return m


def _traces(snap, num_games, num_states, num_actions, num_obs):
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
    obs_m = _obs_matrix(snap, num_obs)
    obs_labels = _obs_labels(num_obs)
    observations = go.Heatmap(
        z=obs_m, x=obs_labels, y=agent_labels,
        colorscale=OBS_SCALE, zmin=0, zmax=1, showscale=False,
        text=[[obs_labels[j] if obs_m[i, j] > 0 else "" for j in range(num_obs)]
              for i in range(NUM_AGENTS)],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} saw %{x}<extra></extra>",
    )
    return [routing, world, counters, true_b, est_b, actions, rewards, observations]


def _step_label(snap):
    """Frame label from the snapshot's phase: reset, step N, or the a/b sub-frames of an act."""
    if snap["phase"] == "reset":
        return "reset"
    label = f"step {snap['step']}"
    if snap["phase"] == "pre_act":
        return label + "a"   # communication resolved, world about to step
    if snap["phase"] == "act":
        return label + "b"   # world stepped
    return label             # communication-only step


def _title(snap, is_act, is_boundary):
    stage = "UTTERANCE" if snap["stage"] == UTTERANCE_STAGE else "BELIEF"
    tags = ""
    if snap["phase"] == "pre_act":
        tags += "   · communication resolved (world about to step)"
    if is_act:
        tags += "   · ACT (world steps)"
    if is_boundary:
        tags += "   · EPISODE BOUNDARY (re-route)"
    head = _step_label(snap)
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
    num_obs = int(env.all_env_parameters.observation.shape[-1])  # per-agent observation count

    # Per-frame flags. The act frame ("b") is the world-step; a boundary shows up as the
    # episode index incrementing on that same post-act frame.
    is_act = [snap["phase"] == "act" for snap in snaps]
    is_boundary = [False] + [snaps[i]["episode"] > snaps[i - 1]["episode"] for i in range(1, len(snaps))]

    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"colspan": 2, "type": "xy"}, None, {"type": "xy"}]],
        row_heights=[0.30, 0.44, 0.26], vertical_spacing=0.11, horizontal_spacing=0.08,
        subplot_titles=(
            "routing — agents ▸ (game, role)",
            "world — each game's true state",
            "scheduler counters",
            "true beliefs  b_i(s)   [agent × state]",
            "estimated beliefs  (estimate ABOUT agent i)",
            "actions taken  [agent × action]  (lit only on ACT steps)",
            "last agent rewards  (nonzero only on acts)",
            "observations received  [agent × obs]  (lit only on ACT steps)",
        ),
    )

    base = _traces(snaps[0], num_games, num_states, num_actions, num_obs)
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 3)]
    for trace, (r, c) in zip(base, positions):
        fig.add_trace(trace, row=r, col=c)

    # Grids read top-down (agent 0 on top); bar axes get sensible ranges.
    for (r, c) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 3)]:
        fig.update_yaxes(autorange="reversed", row=r, col=c)
    # Headroom so the outside value labels on the bars don't clip.
    counter_max = max(
        max(s["env_iter"], s["cum_env"], s["round_cursor"], s["cum_round"]) for s in snaps
    )
    fig.update_yaxes(title_text="count", range=[0, counter_max * 1.25 + 1], row=1, col=3)
    fig.update_yaxes(title_text="reward", range=[-1.4, 1.4], row=3, col=1)

    frames, labels = [], []
    for i, snap in enumerate(snaps):
        frames.append(
            go.Frame(
                name=str(i),
                data=_traces(snap, num_games, num_states, num_actions, num_obs),
                traces=[0, 1, 2, 3, 4, 5, 6, 7],
                layout=go.Layout(title=dict(text=_title(snap, is_act[i], is_boundary[i]))),
            )
        )
        tag = " ✎comm-done" if snap["phase"] == "pre_act" else ""
        tag += " ★act" if is_act[i] else ""
        tag += " ⟲bdry" if is_boundary[i] else ""
        labels.append(_step_label(snap) + tag)

    fig.update_layout(
        height=1080,
        margin=dict(l=60, r=30, t=90, b=60),
        title=dict(text=_title(snaps[0], False, False)),
    )
    F.attach_slider(fig, frames, labels, slider_prefix="")
    return F.write(fig, "viz_stacked")


if __name__ == "__main__":
    print(render())
