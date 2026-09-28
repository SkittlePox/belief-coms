"""Generalized Advantage Estimation (GAE) for the two agent populations.

Turns a rollout's ``Transition`` batch (leading ``[num_steps]`` axis; see
``training.rollout``) into per-population advantages and value targets, ready for the PPO
update.

Why this is not a plain single-stream GAE
------------------------------------------
The stacked env is *staged*: ``step_env`` strictly alternates UTTERANCE -> BELIEF ->
UTTERANCE -> ... and env reward is produced only on *act* steps (which are always belief
stages). So the two populations decide on interleaved steps and their reward timing
differs:

  * Belief (listener) agents decide on BELIEF stages, and the act reward lands on those
    same steps -- textbook GAE, just skipping the intervening utterance steps.
  * Utterance (speaker) agents decide on UTTERANCE stages, but the reward their utterance
    earns only materializes on the *immediately following* belief stage. Strict U->B
    alternation guarantees that follower is exactly one step later, so we shift the reward
    (and the ``done`` flag) one step earlier onto the speaker's decision step and then run
    the same GAE.

``_masked_gae`` runs one reverse scan over the full step axis but only *emits* an advantage
(and advances its bootstrap value) on the steps where the population is ``active``; on the
other population's steps it passes the accumulator through untouched. That skipping is what
makes "decide every other step" behave like a contiguous trajectory for each population.

Advantages/targets are only meaningful at each population's own stages; entries on the
other population's stages are zero (advantage) / garbage (target) and MUST be masked out in
the loss via ``transitions.stage`` -- the same masks (`UTTERANCE_STAGE`/`BELIEF_STAGE`) used
here.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from communication.stacked_signification_decpomdp import BELIEF_STAGE, UTTERANCE_STAGE


class AdvantageOutputs(NamedTuple):
    """Per-population GAE results; every array is ``[num_steps, num_agents]``.

    ``*_targets`` are the value-function regression targets (advantage + value). Valid only
    at that population's own stages (see module docstring).
    """

    utterance_advantages: jnp.ndarray
    utterance_targets: jnp.ndarray
    belief_advantages: jnp.ndarray
    belief_targets: jnp.ndarray


def _masked_gae(values, rewards, dones, active, last_value, gamma, gae_lambda):
    """GAE over the full step axis, accruing only on ``active`` steps.

    Args (all leading axis ``[num_steps]``):
        values:  ``[num_steps, num_agents]`` this population's value estimate per step.
        rewards: ``[num_steps, num_agents]`` reward credited to this population per step.
        dones:   ``[num_steps]`` episode-boundary flag (float 0/1) aligned to ``rewards``.
        active:  ``[num_steps]`` bool -- steps this population actually decided on.
        last_value: ``[num_agents]`` bootstrap value for the step after the rollout.

    On an inactive step the carry (running gae + next bootstrap value) is passed through
    unchanged and a zero advantage is emitted, so consecutive active steps compose as one
    trajectory. Returns ``(advantages, targets)``, each ``[num_steps, num_agents]``.
    """

    def _step(carry, transition):
        gae, next_value = carry
        value, reward, done, act = transition
        cont = 1.0 - done  # stop bootstrapping across an episode boundary
        delta = reward + gamma * next_value * cont - value
        new_gae = delta + gamma * gae_lambda * cont * gae
        # Only advance the accumulator / bootstrap value on this population's own steps.
        gae = jnp.where(act, new_gae, gae)
        next_value = jnp.where(act, value, next_value)
        advantage = jnp.where(act, new_gae, 0.0)
        return (gae, next_value), advantage

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones, active),
        reverse=True,
    )
    return advantages, advantages + values


def compute_advantages(
    transitions,
    gamma: float,
    gae_lambda: float,
    last_utterance_value: Optional[jnp.ndarray] = None,
    last_belief_value: Optional[jnp.ndarray] = None,
) -> AdvantageOutputs:
    """Per-population advantages + value targets from a rollout ``Transition`` batch.

    ``transitions`` leaves carry a leading ``[num_steps]`` axis. ``gamma``/``gae_lambda`` are
    the usual discount and GAE trace-decay. The two bootstrap values are the value estimates
    of each population's next decision *after* the rollout window; they default to zeros
    (i.e. treat the window as ending an episode). Wire in a real bootstrap -- by running each
    population's critic on the final carried observation -- for a continuing rollout.
    """
    stage = transitions.stage  # [num_steps]
    reward = transitions.reward  # [num_steps, num_agents]
    done = transitions.done.astype(reward.dtype)  # [num_steps]

    is_utterance_step = stage == UTTERANCE_STAGE
    is_belief_step = stage == BELIEF_STAGE

    num_agents = reward.shape[-1]
    if last_belief_value is None:
        last_belief_value = jnp.zeros((num_agents,), dtype=reward.dtype)
    if last_utterance_value is None:
        last_utterance_value = jnp.zeros((num_agents,), dtype=reward.dtype)

    # Belief (listener): decides on belief stages; reward/done sit on those same steps.
    belief_advantages, belief_targets = _masked_gae(
        transitions.belief_value,
        reward,
        done,
        is_belief_step,
        last_belief_value,
        gamma,
        gae_lambda,
    )

    # Utterance (speaker): decides on utterance stages, but earns its reward one step later
    # (the following belief/act stage). Shift reward + done one step earlier onto the
    # speaker's decision step; the last step has no follower so it pads with zero and relies
    # on the bootstrap value.
    shifted_reward = jnp.concatenate([reward[1:], jnp.zeros_like(reward[:1])], axis=0)
    shifted_done = jnp.concatenate([done[1:], jnp.zeros_like(done[:1])], axis=0)
    utterance_advantages, utterance_targets = _masked_gae(
        transitions.utterance_value,
        shifted_reward,
        shifted_done,
        is_utterance_step,
        last_utterance_value,
        gamma,
        gae_lambda,
    )

    return AdvantageOutputs(
        utterance_advantages=utterance_advantages,
        utterance_targets=utterance_targets,
        belief_advantages=belief_advantages,
        belief_targets=belief_targets,
    )
