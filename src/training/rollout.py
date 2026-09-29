"""Trajectory collection: one ``env_step`` (a ``jax.lax.scan`` body) and the
``collect_rollout`` wrapper that scans it.

A single ``env_step`` advances the stacked env by exactly ONE ``step_env`` call --
i.e. one communicative *stage* (utterance or belief), not a whole round. Both agent
populations are run every step regardless of stage; the env internally consumes only
the stage-relevant action (see ``StackedSignificationDecPOMDP._step_env_impl``) and
zeros the rest. We therefore record BOTH populations' actions/values/log-probs plus the
``stage`` at every step, and rely on a later masking pass (returns/PPO) to keep only the
entries that actually drove the env on each stage.

Each ``Transition`` also stores the exact inputs each agent consumed (own belief, partner
estimate, rendered utterance) so the PPO update can re-run the policies under fresh params
without re-deriving them from the state.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

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
    utterance_action: jnp.ndarray  # [num_agents, utterance_action_dim]
    utterance_log_prob: jnp.ndarray  # [num_agents]
    utterance_value: jnp.ndarray  # [num_agents]
    utterance_own_belief: jnp.ndarray  # [num_agents, belief_dim]  (input consumed)
    utterance_partner_estimate: jnp.ndarray  # [num_agents, belief_dim]  (input consumed)

    # --- Belief (listener) population ----------------------------------------
    belief_action: jnp.ndarray  # [num_agents, belief_dim]  (a next-belief on the simplex)
    belief_log_prob: jnp.ndarray  # [num_agents]
    belief_value: jnp.ndarray  # [num_agents]
    belief_previous_belief: jnp.ndarray  # [num_agents, belief_dim]  (input consumed)
    belief_utterance_image: jnp.ndarray  # [num_agents, image_dim, image_dim]  (input consumed)


def env_step(carry, _, env, render_utterance_fn, utterance_image_dim, inference_strategy):
    """One ``jax.lax.scan`` step: act with both populations, step the env, emit a Transition.

    No Python control flow (this is a scan body). ``env``, ``render_utterance_fn``,
    ``utterance_image_dim`` and ``inference_strategy`` are bound via ``partial`` in
    ``collect_rollout``; the scanned inputs are unused (``xs=None``).

    ``inference_strategy`` (an ``agent_execution.InferenceStrategy``) decides whether each
    population thinks first-order or with ToM; this step just hands it each population as an
    ``(apply_fn, params)`` pair plus the partner population, and stays agnostic to which.

    Carry: ``(belief_train_states, utterance_train_states, env_state, last_obs, rng)``.
    The two train states are carried read-only (params are fixed during a rollout) so this
    step is self-contained -- it pulls each population's ``apply_fn`` and ``params`` straight
    off its train state.
    """
    belief_train_states, utterance_train_states, env_state, last_obs, rng = carry
    # last_obs is get_obs(state) from the PREVIOUS step: (beliefs, estimated_beliefs,
    # utterances), each [num_agents, ...]. On the belief stage the belief group is NaN'd
    # and on the utterance stage the utterances are NaN'd -- see get_obs. We only feed the
    # gathered partner quantities from here; the own belief we read from the state directly.
    _obs_beliefs, obs_partner_estimates, obs_partner_utterances = last_obs

    rng, utt_rng, bel_rng, render_rng, step_rng = jax.random.split(rng, 5)

    # The stage THIS action is taken in (before the env advances it), and the episode we
    # are in (to detect a boundary crossing after the step, for the GAE ``done`` flag).
    stage = env_state.dialog.communicative_round_stage
    episode_index_before = env_state.game_counters.episode_index

    # Each agent's own current belief, straight from the world state (always valid, unlike
    # the obs copy which the belief stage NaNs). This is the "existing belief" both
    # populations condition on.
    own_beliefs = env_state.true_agent_belief_states  # [num_agents, belief_dim]

    # Each population as an (apply_fn, params) handle, so the strategy can be handed both
    # its own and its partner's model uniformly (the partner is unused in first-order mode).
    utterance_pop = (utterance_train_states.apply_fn, utterance_train_states.params)
    belief_pop = (belief_train_states.apply_fn, belief_train_states.params)

    # --- Utterance (speaker) population --------------------------------------
    # Conditions on own belief + the estimate of the receiver-partner it is engaged with.
    utterance_actions, utterance_log_probs, utterance_values = inference_strategy.run_utterance_agents(
        utt_rng,
        utterance_pop,
        belief_pop,
        own_beliefs,
        obs_partner_estimates,
        render_utterance_fn,
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
    belief_actions, belief_log_probs, belief_values = inference_strategy.run_belief_agents(
        bel_rng,
        belief_pop,
        utterance_pop,
        own_beliefs,
        utterance_images,
        render_utterance_fn,
    )

    # --- Post-utterance belief estimate (step_env's third input) -------------
    # SKELETON: step_env also wants each speaker's refreshed estimate of its
    # listener-partner's belief (SUBJECT-indexed: row i = estimate ABOUT agent i). For now
    # pass the existing estimate through unchanged so shapes/plumbing are correct; replace
    # with the speaker's ToM re-estimate (belief model run on its own utterance) later.
    belief_estimate_post_utterance = env_state.estimated_agent_belief_states

    # --- Step the environment one stage --------------------------------------
    env_state, next_obs, rewards = env.step_env(
        step_rng,
        env_state,
        utterance_actions,
        belief_estimate_post_utterance,
        belief_actions,
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
        utterance_action=utterance_actions,
        utterance_log_prob=utterance_log_probs,
        utterance_value=utterance_values,
        utterance_own_belief=own_beliefs,
        utterance_partner_estimate=obs_partner_estimates,
        belief_action=belief_actions,
        belief_log_prob=belief_log_probs,
        belief_value=belief_values,
        belief_previous_belief=own_beliefs,
        belief_utterance_image=utterance_images,
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
    inference_strategy,
):
    """Scan ``env_step`` ``num_steps`` times, threading the env state forward.

    ``inference_strategy`` (first-order or ToM) is bound into every step. Returns
    ``(env_state, last_obs, rng, transitions)`` where ``transitions`` is a ``Transition``
    whose leaves carry a leading ``[num_steps]`` axis, and the first three are the
    post-rollout carry to fold back into the training loop.
    """
    step_fn = partial(
        env_step,
        env=env,
        render_utterance_fn=render_utterance_fn,
        utterance_image_dim=utterance_image_dim,
        inference_strategy=inference_strategy,
    )
    init_carry = (belief_train_states, utterance_train_states, env_state, last_obs, rng)
    (_belief_train_states, _utterance_train_states, env_state, last_obs, rng), transitions = jax.lax.scan(
        step_fn, init_carry, xs=None, length=num_steps
    )
    return env_state, last_obs, rng, transitions
