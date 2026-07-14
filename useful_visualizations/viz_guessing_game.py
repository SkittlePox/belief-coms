"""Visualize the guessing game's dynamics tensors and how ``FlexibleEnv`` consumes them.

Static panels (top) read straight off the ``FlexibleEnvParams`` built by
``guessing_game_spec``:

  * transition ``T[S, A, A, S]`` -- the next state for each (state, button); agent 1's
    action is inert, so we fix a1=0. Pressing a0==s sends you to the absorbing done
    state (3); every other button keeps you put.
  * observation ``O[S, A, A, O, O]`` -- action-independent, so we fix a=(0,0). Shown as
    each agent's per-symbol marginal ``O(o | s')`` (state s never emits symbol s) and,
    for one state, the joint ``O(o0, o1 | s')`` = outer product of identical marginals.
  * reward ``R[N, S, A, A, S]`` -- shared across agents, independent of a1 and s':
    +1 correct button, -1 wrong, -0.1 wait, 0 in the done state.

Animated panel (bottom): the "learning by waiting" rollout from ``guessing_game.py``.
The presser repeatedly waits and updates its belief via the same belief engine the env
uses; scrub the slider to watch each observed symbol rule out a state until the belief
collapses to the truth -- how the observation tensor is actually *used*.
"""

from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.flexible_env import FlexibleEnv
from envs.guessing_game import guessing_game_spec
from tools.belief_representations import CategoricalBeliefState

from . import _figures as F


def _belief_by_waiting(params, true_state=0, n_waits=12, seed=0):
    """Presser waits, observes, updates belief; return per-step (probs, symbol)."""
    env = FlexibleEnv(params)
    belief_factory = CategoricalBeliefState(params)
    wait = env.num_actions - 1
    wait_joint_action = (wait, wait)
    presser_obs_dist = params.observation[true_state, 0, 0].sum(axis=1)  # O(o | true_state)

    belief = distrax.Categorical(probs=params.initial_belief_states[0])
    frames = [(np.asarray(belief.probs), None)]  # start: the uniform prior
    key = jax.random.key(seed)
    for _ in range(n_waits):
        key, obs_key = jax.random.split(key)
        obs = distrax.Categorical(probs=presser_obs_dist).sample(seed=obs_key)
        belief = belief_factory.update_with_observation_and_joint_action(
            belief, obs, wait_joint_action, agent_id=0
        )
        frames.append((np.asarray(belief.probs), int(obs)))
        if belief.probs[true_state] > 0.999:
            break
    return frames


def render() -> str:
    params, _ = guessing_game_spec()
    S = int(params.num_states)
    A = int(params.num_actions)
    O = params.observation.shape[-1]
    done = S - 1

    T = np.asarray(params.transition)          # [S, A, A, S]
    R = np.asarray(params.reward)              # [N, S, A, A, S]
    Obs = np.asarray(params.observation)       # [S, A, A, O, O]

    # Transition: next state s' for each (s, a0), with a1 inert (fixed at 0).
    next_state = T[:, :, 0, :].argmax(axis=-1).astype(float)  # [S, A]
    # Reward: R independent of agent, a1 and s' here -> take agent 0, a1=0, s'=0.
    reward_grid = R[0, :, :, 0, 0]  # [S, A]
    # Observation marginal per agent: O(o | s') = sum out the other agent.
    obs_marg = Obs[:, 0, 0, :, :].sum(axis=2)  # [S, O]
    # Joint observation for one referent state (state 0): outer product structure.
    joint_state = 0
    obs_joint = Obs[joint_state, 0, 0]  # [O, O]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"colspan": 2, "type": "xy"}, None],
        ],
        row_heights=[0.5, 0.5], vertical_spacing=0.17, horizontal_spacing=0.09,
        subplot_titles=(
            "transition  T:  next state s'  (a1 inert;  a0==s → done)",
            "reward  R(s, a0):  +1 hit / −1 miss / −0.1 wait / 0 done",
            "observation marginal  O(o | s')   (state s never emits symbol s)",
            f"joint obs  O(o0, o1 | s'={joint_state})  = outer(marginal, marginal)",
            "FlexibleEnv rollout — presser's belief while waiting  ▶",
        ),
    )

    state_labels = [f"s={i}" + (" (done)" if i == done else "") for i in range(S)]
    action_labels = [f"a0={a}" + (" wait" if a == A - 1 else "") for a in range(A)]
    symbol_labels = [f"o={o}" for o in range(O)]

    # Transition next-state grid (color by next-state index; text is the index).
    fig.add_trace(
        F.heatmap_trace(
            next_state, x=action_labels, y=state_labels, colorscale=F.PROB_SCALE,
            zmin=0, zmax=S - 1, text=True, text_fmt="s'={:.0f}", showscale=False, hover="next s'",
        ),
        row=1, col=1,
    )
    # Reward grid (diverging).
    fig.add_trace(
        F.heatmap_trace(
            reward_grid, x=action_labels, y=state_labels, colorscale=F.REWARD_SCALE,
            zmin=-1, zmax=1, text=True, text_fmt="{:+.1f}", showscale=False, hover="reward",
        ),
        row=1, col=2,
    )
    # Observation marginal.
    fig.add_trace(
        F.heatmap_trace(
            obs_marg, x=symbol_labels, y=state_labels, colorscale=F.PROB_SCALE,
            zmin=0, zmax=1, showscale=False, hover="P(o|s')",
        ),
        row=1, col=3,
    )
    # Joint observation for state 0.
    fig.add_trace(
        F.heatmap_trace(
            obs_joint, x=[f"o1={o}" for o in range(O)], y=[f"o0={o}" for o in range(O)],
            colorscale=F.PROB_SCALE, zmin=0, zmax=obs_joint.max(), showscale=False, hover="P(o0,o1|s')",
        ),
        row=2, col=1,
    )

    # Belief-by-waiting rollout: one animated bar trace (added last).
    rollout = _belief_by_waiting(params, true_state=0)
    belief_trace_idx = len(fig.data)
    fig.add_trace(
        F.bar_trace(rollout[0][0], state_labels, color=F.ACCENT_2),
        row=2, col=2,
    )

    # Row-0-on-top for the grids; belief bar on a probability axis.
    for r, c in [(1, 1), (1, 2), (1, 3), (2, 1)]:
        fig.update_yaxes(autorange="reversed", row=r, col=c)
    fig.update_yaxes(range=[0, 1.18], title_text="belief", row=2, col=2)

    frames, labels = [], []
    for i, (probs, sym) in enumerate(rollout):
        frames.append(
            go.Frame(
                name=str(i),
                data=[F.bar_trace(probs, state_labels, color=F.ACCENT_2)],
                traces=[belief_trace_idx],
            )
        )
        labels.append("start (prior)" if sym is None else f"t={i - 1}: saw o={sym}")

    fig.update_layout(
        height=1000,
        margin=dict(l=60, r=30, t=80, b=60),
        title="Guessing game — dynamics tensors and how FlexibleEnv uses them",
    )
    F.attach_slider(fig, frames, labels, slider_prefix="wait ")
    return F.write(fig, "viz_guessing_game")


if __name__ == "__main__":
    print(render())
