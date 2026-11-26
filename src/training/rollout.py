"""Trajectory collection: one ``env_step`` (a ``jax.lax.scan`` body) and the
``collect_rollout`` wrapper that scans it.

A single ``env_step`` advances the stacked env by exactly ONE ``step_env`` call --
i.e. one communicative *stage* (utterance or belief), not a whole round. Both agent
populations are run every step regardless of stage; the env internally consumes only
the stage-relevant action (see ``StackedSignificationDecPOMDP.step_env``) and
zeros the rest. We therefore record BOTH populations' actions/values/log-probs plus the
``stage`` at every step, and rely on a later masking pass (returns/PPO) to keep only the
entries that actually drove the env on each stage.

Each ``Transition`` also stores the exact inputs each agent consumed (its belief levels
and, for the listener, the rendered utterance) so the PPO update can re-run the policies
under fresh params without re-deriving them from the state.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from communication.stacked_signification_decpomdp import BELIEF_STAGE


class Transition(NamedTuple):
    """One step's worth of rollout data, everything the update pass will need.

    Leading axis of every array is ``[num_agents]`` (the scan adds a further leading
    ``[num_steps]`` axis when it stacks these). ``stage`` is the scalar
    ``UTTERANCE_STAGE``/``BELIEF_STAGE`` this step ran in; ``reward`` is the per-agent
    env reward, real only on act steps and NaN otherwise. The NaN is a placeholder for a
    value to be filled in retroactively once communication finishes; until that
    happens, consumers such as ``compute_advantages`` must not use it as a number.
    """

    stage: jnp.ndarray  # scalar int: which communicative stage this step was
    reward: jnp.ndarray  # [num_agents] per-agent env reward from this step
    done: jnp.ndarray  # scalar bool: did this step cross an episode boundary (re-route)?

    # --- Utterance (speaker) population --------------------------------------
    utterance_inputs: Tuple[jnp.ndarray, ...]  # (b_e, b̄_o, b̿_e), each [num_agents, belief_dim]  (inputs consumed)
    utterance_action: jnp.ndarray  # [num_agents, utterance_action_dim]
    utterance_log_prob: jnp.ndarray  # [num_agents]
    utterance_value: jnp.ndarray  # [num_agents]

    # --- Belief (listener) population ----------------------------------------
    belief_inputs: Tuple[jnp.ndarray, ...]  # (b_e, b̄_o, b̿_e, utterance image), each [num_agents, ...]  (inputs consumed)
    belief_action: jnp.ndarray  # [num_agents, belief_dim]  (a next-belief on the simplex)
    belief_log_prob: jnp.ndarray  # [num_agents]
    belief_value: jnp.ndarray  # [num_agents]


def env_step(carry, _, env, render_utterance_fn, utterance_image_dim, inference):
    """One ``jax.lax.scan`` step: act with both populations, step the env, emit a Transition.

    No Python control flow (this is a scan body). ``env``, ``render_utterance_fn``,
    ``utterance_image_dim`` and ``inference`` are bound via ``partial`` in
    ``collect_rollout``; the scanned inputs are unused (``xs=None``).

    ``inference`` (an ``agent_execution.AgentInferenceConfig``) decides whether each
    population runs e2e or with ToM; this step hands it both populations and stays agnostic
    to which.

    Carry: ``(belief_train_states, utterance_train_states, env_state, last_obs, rng)``.
    The two train states are carried read-only (params are fixed during a rollout) so this
    step is self-contained -- it pulls each population's ``apply_fn`` and ``params`` straight
    off its train state.
    """
    belief_train_states, utterance_train_states, env_state, last_obs, rng = carry
    # last_obs is get_obs(state) from the PREVIOUS step, each [num_agents, ...] and indexed
    # so row i is what agent i consumes: its own belief (b_e), its estimate of its partner
    # (b̄_o), its second-order estimate (b̿_e), and the utterance it hears (NaN on the
    # utterance stage, when there is none yet) -- see get_obs.
    ego_beliefs, other_belief_estimates, second_order_ego_estimates, obs_partner_utterances = last_obs

    rng, utt_rng, bel_rng, render_rng, step_rng = jax.random.split(rng, 5)

    # The stage THIS action is taken in (before the env advances it), and the episode we
    # are in (to detect a boundary crossing after the step, for the GAE ``done`` flag).
    stage = env_state.dialog.communicative_round_stage
    episode_index_before = env_state.game_counters.episode_index

    # --- Utterance (speaker) population --------------------------------------
    utterance_inputs = (ego_beliefs, other_belief_estimates, second_order_ego_estimates)
    utterance_actions, utterance_log_probs, utterance_values = inference.run_utterance_agents(
        utt_rng, utterance_train_states, belief_train_states, *utterance_inputs, render_utterance_fn
    )

    # --- Belief (listener) population ----------------------------------------
    # The belief agent consumes a RENDERED utterance image, so paint the raw utterance
    # vectors the obs handed us (each agent's partner-speaker's utterance) onto canvases.
    # get_obs NaNs the utterances on the UTTERANCE stage (the belief output there is masked
    # out of GAE/PPO anyway), and painting NaN splines would bake NaNs into the stored image
    # -- which can later poison masked reductions/gradients. So render only on the belief
    # stage and hand a zero canvas back otherwise. lax.cond (not where) genuinely skips the
    # paint on utterance steps, since env_step is scanned, not vmapped.
    num_agents = obs_partner_utterances.shape[0]
    on_belief_stage = stage == BELIEF_STAGE
    utterance_images = jax.lax.cond(
        on_belief_stage,
        lambda: render_utterance_fn(render_rng, obs_partner_utterances),
        lambda: jnp.zeros((num_agents, utterance_image_dim, utterance_image_dim), dtype=jnp.float32),
    )  # [num_agents, D, D]
    belief_inputs = (ego_beliefs, other_belief_estimates, second_order_ego_estimates, utterance_images)
    listener_ego_belief_actions, belief_log_probs, belief_values = inference.run_belief_agents(
        bel_rng, belief_train_states, utterance_train_states, *belief_inputs, render_utterance_fn
    )

    # --- Belief estimates (step_env's b̄_o and b̿_e inputs) ---------------------
    # SKELETON: the agents don't produce these yet, so pass the env's current estimates
    # through unchanged. Shapes and plumbing are correct, but the estimates never move
    # from communication; replace once we decide what produces them.
    speaker_other_belief_estimate_actions = env_state.estimated_agent_belief_states
    speaker_second_order_ego_estimate_actions = env_state.second_order_agent_belief_estimates
    listener_other_belief_estimate_actions = env_state.estimated_agent_belief_states
    listener_second_order_ego_estimate_actions = env_state.second_order_agent_belief_estimates

    # --- Step the environment one stage --------------------------------------
    env_state, next_obs, rewards, _, _ = env.step_env(
        step_rng,
        env_state,
        utterance_actions,
        speaker_other_belief_estimate_actions,
        speaker_second_order_ego_estimate_actions,
        listener_ego_belief_actions,
        listener_other_belief_estimate_actions,
        listener_second_order_ego_estimate_actions,
    )

    # An episode boundary re-routes assignments and resamples world states/beliefs, so the
    # value after it belongs to a fresh episode -- GAE must not bootstrap across it. The
    # boundary is a scalar event (all games share the horizon), detected by the episode
    # index advancing during this step.
    done = env_state.game_counters.episode_index != episode_index_before

    transition = Transition(
        stage=stage,
        reward=rewards,
        done=done,
        utterance_inputs=utterance_inputs,
        utterance_action=utterance_actions,
        utterance_log_prob=utterance_log_probs,
        utterance_value=utterance_values,
        belief_inputs=belief_inputs,
        belief_action=listener_ego_belief_actions,
        belief_log_prob=belief_log_probs,
        belief_value=belief_values,
    )

    carry = (belief_train_states, utterance_train_states, env_state, next_obs, rng)
    return carry, transition


def collect_rollout(
    belief_train_states,
    utterance_train_states,
    env,
    env_state,
    last_obs,
    rng,
    num_steps,
    render_utterance_fn,
    utterance_image_dim,
    inference,
):
    """Scan ``env_step`` ``num_steps`` times, threading the env state forward.

    ``inference`` (e2e or ToM) is bound into every step. Returns
    ``(env_state, last_obs, rng, transitions)`` where ``transitions`` is a ``Transition``
    whose leaves carry a leading ``[num_steps]`` axis, and the first three are the
    post-rollout carry to fold back into the training loop.
    """
    step_fn = partial(
        env_step,
        env=env,
        render_utterance_fn=render_utterance_fn,
        utterance_image_dim=utterance_image_dim,
        inference=inference,
    )
    init_carry = (belief_train_states, utterance_train_states, env_state, last_obs, rng)
    (_belief_train_states, _utterance_train_states, env_state, last_obs, rng), transitions = jax.lax.scan(
        step_fn, init_carry, xs=None, length=num_steps
    )
    return env_state, last_obs, rng, transitions
