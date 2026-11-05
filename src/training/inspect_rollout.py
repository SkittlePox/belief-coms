"""Rollout inspector: scan ``rollout.env_step`` on the stacked env and print every stage.

Builds the env and freshly initialized (untrained) agent populations exactly as
``ippo.make_train`` does, then scans ``env_step`` for ``num_steps`` stages -- the same
scan ``collect_rollout`` runs -- additionally keeping the env state after each step so
the world/belief bookkeeping can be read alongside each ``Transition``.

Accepts the same experiment flags as ``training.launch``. Run from ``src/``:
    uv run python -m training.inspect_rollout --num-steps 16 \\
        --communication.scheme a_to_b \\
        role-assignment:fixed-pairs-assignment-config \\
        --role-assignment.num-agents 4 --role-assignment.underlying-env-steps-per-episode 3
"""

from __future__ import annotations
from functools import partial
from typing import Annotated

import jax
import numpy as np
import tyro

from communication.stacked_signification_decpomdp import UTTERANCE_STAGE
from tools.utterance_rendering import paint_multiple_splines
from training.config import ExperimentConfig
from training.ippo import initialize_belief_agents, initialize_utterance_agents
from training.rollout import env_step


def run_rollout(config: ExperimentConfig, num_steps: int):
    """Build env + agents from ``config`` and scan ``env_step`` ``num_steps`` times.

    Returns ``(initial_env_state, transitions, env_states)``: ``transitions`` and
    ``env_states`` carry a leading ``[num_steps]`` axis, and ``env_states[t]`` is the env
    state AFTER step t (so the state step t acted from is ``env_states[t - 1]``, or
    ``initial_env_state`` for t = 0).
    """
    num_agents = config.role_assignment.num_agents
    env = config.environment.build(
        num_agents=num_agents,
        assignment_fn=config.role_assignment.build(),
        communication_scheme_fn=config.communication.build(),
    )
    image_dim = config.environment.utterance_image_dim
    render_utterance_fn = lambda rng, utterances: paint_multiple_splines(utterances, image_dim)
    inference_strategy = config.inference.build()

    rng = jax.random.key(config.jax_seed)
    belief_rng, utterance_rng, env_rng, rollout_rng = jax.random.split(rng, 4)
    belief_train_states = initialize_belief_agents(
        config.belief_agents,
        config.belief_optimizer,
        belief_rng,
        num_agents,
        input_utterance_shape=(image_dim, image_dim),
        belief_dim=env.belief_dim,
    )
    utterance_train_states = initialize_utterance_agents(
        config.utterance_agents,
        config.utterance_optimizer,
        utterance_rng,
        num_agents,
        utterance_action_dim=config.environment.utterance_action_dim,
        belief_dim=env.belief_dim,
    )
    env_state, obs = env.reset(env_rng)

    step_fn = partial(
        env_step,
        env=env,
        render_utterance_fn=render_utterance_fn,
        utterance_image_dim=image_dim,
        inference_strategy=inference_strategy,
    )

    def traced_step(carry, _):
        carry, transition = step_fn(carry, None)
        return carry, (transition, carry[2])  # carry[2] is the post-step env state

    @jax.jit
    def scan(carry):
        return jax.lax.scan(traced_step, carry, xs=None, length=num_steps)[1]

    init_carry = (belief_train_states, utterance_train_states, env_state, obs, rollout_rng)
    transitions, env_states = jax.device_get(scan(init_carry))
    return jax.device_get(env_state), transitions, env_states


# --- Printing ----------------------------------------------------------------------------


def _vec(v, max_dims=None) -> str:
    v = np.asarray(v)
    suffix = ""
    if max_dims is not None and v.shape[-1] > max_dims:
        v, suffix = v[..., :max_dims], " ..."
    return np.array2string(v, precision=3, suppress_small=True, floatmode="fixed", max_line_width=10_000) + suffix


def _at(tree, t):
    return jax.tree.map(lambda leaf: leaf[t], tree)


def print_assignment(state, header: str):
    roles = state.game_roles
    print(f"\n##### {header} #####")
    print(f"  agent_game_assignment : {_vec(roles.agent_game_assignment)}")
    print(f"  agent_role_assignment : {_vec(roles.agent_role_assignment)}")
    print(f"  partner_agent         : {_vec(roles.partner_agent)}")
    print(f"  game_types            : {_vec(roles.game_types)}")
    print(f"  episode_horizon       : {int(state.game_counters.episode_horizon)}")
    print(f"  active_who_speaks     : {np.asarray(state.dialog.active_who_speaks).tolist()}  (rows = rounds, cols = roles)")


def print_initial_state(state):
    print_assignment(state, "reset")
    print(f"  game_states           : {_vec(state.game_states)}")
    print(f"  counters              : episode {int(state.game_counters.episode_index)}, "
          f"env t {int(state.game_counters.underlying_env_timestep)}, cum t {int(state.game_counters.cumulative_env_timestep)}")
    for agent in range(state.true_agent_belief_states.shape[0]):
        print(f"  agent {agent}: true belief {_vec(state.true_agent_belief_states[agent])}   "
              f"estimate about agent {_vec(state.estimated_agent_belief_states[agent])}")


def print_step(t, prev, tr, post, max_utterance_dims):
    """Print step t: ``prev`` is the state it acted from, ``post`` the state after it."""
    on_utterance_stage = int(tr.stage) == UTTERANCE_STAGE
    stage_name = "UTTERANCE" if on_utterance_stage else "BELIEF"
    round_idx = int(prev.dialog.communication_round_iterator)
    num_rounds = int(prev.dialog.active_total_num_rounds)
    pc, nc = prev.game_counters, post.game_counters
    acted = bool((post.last_agent_actions >= 0).any())

    print(f"\n=== step {t} | {stage_name} stage | round {round_idx + 1}/{num_rounds} | "
          f"episode {int(pc.episode_index)}->{int(nc.episode_index)} | "
          f"env t {int(pc.underlying_env_timestep)}->{int(nc.underlying_env_timestep)} | "
          f"cum t {int(pc.cumulative_env_timestep)}->{int(nc.cumulative_env_timestep)} | "
          f"acted={acted} | done={bool(tr.done)}")
    print(f"  game_states: {_vec(prev.game_states)} -> {_vec(post.game_states)}")

    who_speaks = np.asarray(prev.dialog.active_who_speaks[round_idx])
    roles = prev.game_roles
    for agent in range(tr.reward.shape[0]):
        role = int(roles.agent_role_assignment[agent])
        speaks = bool(who_speaks[role])
        listens = bool(who_speaks[1 - role])
        tag = "+".join(x for x, on in (("speaks", speaks), ("listens", listens)) if on) or "idle"
        print(f"  agent {agent} (game {int(roles.agent_game_assignment[agent])}, role {role}, "
              f"partner {int(roles.partner_agent[agent])}, {tag})")
        if on_utterance_stage:
            print(f"      in : own belief {_vec(tr.utterance_own_belief[agent])}   "
                  f"partner estimate {_vec(tr.utterance_partner_estimate[agent])}")
            print(f"      out: utterance {_vec(tr.utterance_action[agent], max_utterance_dims)}   "
                  f"log_prob {tr.utterance_log_prob[agent]:.3f}   value {tr.utterance_value[agent]:.3f}")
            print(f"      stored utterance (delivered) {_vec(post.agent_utterance_actions_unrendered[agent], max_utterance_dims)}")
        else:
            print(f"      in : previous belief {_vec(tr.belief_previous_belief[agent])}   "
                  f"utterance image sum {tr.belief_utterance_image[agent].sum():.1f}")
            print(f"      out: belief action {_vec(tr.belief_action[agent])}   "
                  f"log_prob {tr.belief_log_prob[agent]:.3f}   value {tr.belief_value[agent]:.3f}")
        print(f"      env: action {int(post.last_agent_actions[agent])}   obs {int(post.last_agent_observations[agent])}   "
              f"reward {tr.reward[agent]:.3f}")
        print(f"      post: true belief {_vec(post.true_agent_belief_states[agent])}   "
              f"estimate about agent {_vec(post.estimated_agent_belief_states[agent])}")

    if bool(tr.done):
        print_assignment(post, f"episode {int(nc.episode_index)} begins (post-step {t})")


def main(
    experiment_config: Annotated[ExperimentConfig, tyro.conf.arg(name="")],
    num_steps: int = 16,
    max_utterance_dims: int = 6,
):
    """Print a stage-by-stage trace of one rollout.

    Args:
        num_steps: Number of env stages (step_env calls) to roll out.
        max_utterance_dims: Utterance vectors are truncated to this many dims when printed.
    """
    initial_state, transitions, env_states = run_rollout(experiment_config, num_steps)
    print_initial_state(initial_state)
    prev = initial_state
    for t in range(num_steps):
        post = _at(env_states, t)
        print_step(t, prev, _at(transitions, t), post, max_utterance_dims)
        prev = post


if __name__ == "__main__":
    tyro.cli(
        main,
        config=(
            tyro.conf.UsePythonSyntaxForLiteralCollections,
            tyro.conf.FlagConversionOff,
        ),
    )
