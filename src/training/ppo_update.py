"""PPO loss + gradient step for the two agent populations.

Consumes a rollout's ``Transition`` batch (``training.rollout``) and its per-population
advantages/targets (``training.advantages``), and returns updated train states plus
metrics. The two populations (belief/listener, utterance/speaker) are trained
independently -- each with its own params, optimizer, and critic -- so the machinery here
is written once and applied to each.

Independence & vmap
-------------------
Every agent has its own parameter set (leading ``[num_agents]`` axis on the batched
``TrainState``), and IPPO trains each on its own trajectory column. So the per-agent loss +
gradient step (``_update_one_agent``) is written for a single agent and ``vmap``-ed over the
``num_agents`` axis; agent quantities sit on axis 1 of the ``[num_steps, num_agents, ...]``
rollout arrays (mapped with ``in_axes=1``), while the per-step ``stage`` mask has no agent
axis (``in_axes=None``).

Staged masking
--------------
The env alternates utterance/belief stages and each population only decides on its own
stage. We recompute the policy on EVERY stored step (cheap, and keeps shapes uniform) but
the loss is a **masked mean** over just this population's stages -- the same
``UTTERANCE_STAGE``/``BELIEF_STAGE`` split used to build the advantages. Off-stage entries
of the stored transitions can be non-finite (e.g. the speaker ran on the belief stage's
NaN partner-estimate), so the loss ``nan_to_num``s the recompute inputs first: a NaN in a
masked-out term still poisons gradients (``0 * NaN = NaN``), so it must never enter the
graph, not merely be zeroed after the fact.

Recompute uses the first-order network forward pass (not any ToM search) regardless of how
the rollout was collected: the rollout stores each action's log-prob under that same network
policy, so the PPO ratio is consistent. That is also why this module never needs the
inference strategy.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from communication.stacked_signification_decpomdp import BELIEF_STAGE, UTTERANCE_STAGE


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters, shared by both populations.

    Grouped as a sub-config (like the agent/optimizer/inference configs) rather than loose
    top-level fields. No ``build()``: it is pure data read straight by ``ppo_update``.
    """

    clip_eps: float = 0.2  # PPO surrogate + value clip range
    vf_coef: float = 0.5  # value-loss weight
    ent_coef: float = 0.01  # entropy-bonus weight
    update_epochs: int = 4  # passes over the rollout, per population, per iteration
    num_minibatches: int = 4  # minibatches the step axis is split into each epoch
    # (num_steps_per_epoch must be divisible by num_minibatches)


def _ppo_loss(params, apply_fn, net_input_0, net_input_1, actions, old_log_probs, old_values, advantages, targets, stage_mask, clip_eps, vf_coef, ent_coef):
    """Masked clipped-PPO loss for ONE agent over its rollout column.

    All per-step arrays have leading ``[num_steps]``. ``net_input_0`` / ``net_input_1`` are
    the two positional inputs this population's network consumes (belief: previous_belief,
    utterance_image; utterance: own_belief, partner_estimate). ``stage_mask`` is the boolean
    per-step mask of this population's own stages. Returns ``(loss, aux_metrics)``.
    """
    # Sanitize everything derived from off-stage steps (all masked out below): the env NaNs
    # the irrelevant observations, so the stored recompute inputs / old stats can be NaN
    # there, and so can `targets` (== advantage + value, and the value is NaN off-stage). A
    # NaN anywhere poisons the gradient even when its step is masked (0 * NaN = NaN), so it
    # must never enter the graph. On-stage steps are finite, so this is a no-op where it
    # matters.
    net_input_0 = jnp.nan_to_num(net_input_0)
    net_input_1 = jnp.nan_to_num(net_input_1)
    actions = jnp.nan_to_num(actions)
    old_log_probs = jnp.nan_to_num(old_log_probs)
    old_values = jnp.nan_to_num(old_values)
    advantages = jnp.nan_to_num(advantages)
    targets = jnp.nan_to_num(targets)

    pi, values = apply_fn(params, net_input_0, net_input_1)
    new_log_probs = pi.log_prob(actions)  # [num_steps]
    entropy = pi.entropy()  # [num_steps]

    mask = stage_mask.astype(jnp.float32)  # [num_steps]
    normalizer = jnp.maximum(mask.sum(), 1.0)  # this population always has >=1 own stage

    def masked_mean(x):
        return (x * mask).sum() / normalizer

    # Normalize advantages over this population's own steps only (off-stage advantages are
    # zero and must not skew the statistics).
    adv_mean = masked_mean(advantages)
    adv_var = masked_mean((advantages - adv_mean) ** 2)
    advantages = (advantages - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)

    # Clipped surrogate (actor) loss.
    ratio = jnp.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -jnp.minimum(unclipped, clipped)

    # Clipped value loss (PureJaxRL style): penalize the larger of the raw and clipped errors.
    values_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    value_loss = 0.5 * jnp.maximum((values - targets) ** 2, (values_clipped - targets) ** 2)

    actor_loss = masked_mean(actor_loss)
    value_loss = masked_mean(value_loss)
    entropy = masked_mean(entropy)
    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    # Diagnostics (masked): approximate KL and the fraction of steps hitting the clip.
    approx_kl = masked_mean(old_log_probs - new_log_probs)
    clip_frac = masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    aux = {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return total_loss, aux


def _update_one_agent(train_state, net_input_0, net_input_1, actions, old_log_probs, old_values, advantages, targets, stage_mask, clip_eps, vf_coef, ent_coef):
    """One agent's value-and-grad + optimizer step; ``vmap``-ed over agents by the caller."""

    def loss_fn(params):
        return _ppo_loss(
            params, train_state.apply_fn, net_input_0, net_input_1, actions, old_log_probs,
            old_values, advantages, targets, stage_mask, clip_eps, vf_coef, ent_coef,
        )

    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=grads), aux


# Map the single-agent update over the num_agents axis. Agent-indexed arrays are on axis 1
# of the [num_steps, num_agents, ...] rollout; the train state batches on axis 0; the
# per-step stage mask and the scalar coefficients are shared (None).
_update_all_agents = jax.vmap(
    _update_one_agent,
    in_axes=(0, 1, 1, 1, 1, 1, 1, 1, None, None, None, None),
)


def _update_population(rng, train_states, net_input_0, net_input_1, actions, old_log_probs, old_values, advantages, targets, stage_mask, config: PPOConfig):
    """Run ``config.update_epochs`` minibatched gradient steps for one population.

    Each epoch reshuffles the step axis and splits it into ``config.num_minibatches``
    minibatches, taking one gradient step per minibatch (the standard PPO inner loop). The
    permutation is shared across agents -- they see the same reordering of timesteps, but
    each still trains on its own column. Returns ``(train_states, metrics)`` with metrics
    averaged over agents, minibatches, and epochs.
    """
    num_steps = stage_mask.shape[0]
    assert num_steps % config.num_minibatches == 0, (
        f"num_steps_per_epoch ({num_steps}) must be divisible by num_minibatches "
        f"({config.num_minibatches})"
    )
    minibatch_size = num_steps // config.num_minibatches

    # Everything with a step axis, bundled so one permutation/reshape hits all of it in
    # lockstep (keeping each step's mask aligned with its data). Step axis is axis 0 for
    # every leaf: agent-indexed arrays are [num_steps, num_agents, ...]; stage_mask is
    # [num_steps] (no agent axis).
    data = (net_input_0, net_input_1, actions, old_log_probs, old_values, advantages, targets, stage_mask)

    def _epoch(train_states, epoch_rng):
        perm = jax.random.permutation(epoch_rng, num_steps)
        shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), data)
        # Split the (shuffled) step axis into [num_minibatches, minibatch_size, ...].
        minibatches = jax.tree.map(
            lambda x: x.reshape(config.num_minibatches, minibatch_size, *x.shape[1:]),
            shuffled,
        )

        def _minibatch(train_states, mb):
            net0, net1, acts, old_lp, old_v, adv, tgt, mask = mb
            train_states, aux = _update_all_agents(
                train_states, net0, net1, acts, old_lp, old_v, adv, tgt, mask,
                config.clip_eps, config.vf_coef, config.ent_coef,
            )
            # aux leaves carry a leading num_agents axis; average over agents.
            return train_states, jax.tree.map(jnp.mean, aux)

        train_states, mb_metrics = jax.lax.scan(_minibatch, train_states, minibatches)
        # mb_metrics leaves carry a leading num_minibatches axis; average over minibatches.
        return train_states, jax.tree.map(jnp.mean, mb_metrics)

    train_states, metrics = jax.lax.scan(_epoch, train_states, jax.random.split(rng, config.update_epochs))
    # metrics leaves now carry a leading update_epochs axis; average over epochs too.
    return train_states, jax.tree.map(jnp.mean, metrics)


def ppo_update(rng, belief_train_states, utterance_train_states, transitions, advantages, config: PPOConfig):
    """Update both populations from one rollout; returns the new train states + metrics.

    ``transitions`` is the rollout ``Transition`` batch (leading ``[num_steps]`` axis) and
    ``advantages`` is the ``AdvantageOutputs`` from ``compute_advantages``. ``rng`` seeds the
    per-epoch minibatch shuffles (split once per population). Metrics are returned under
    ``belief_*`` / ``utterance_*`` prefixes.
    """
    utterance_rng, belief_rng = jax.random.split(rng)
    is_utterance_step = transitions.stage == UTTERANCE_STAGE
    is_belief_step = transitions.stage == BELIEF_STAGE

    # Utterance (speaker): network consumes (own_belief, partner_estimate).
    utterance_train_states, utterance_metrics = _update_population(
        utterance_rng,
        utterance_train_states,
        transitions.utterance_own_belief,
        transitions.utterance_partner_estimate,
        transitions.utterance_action,
        transitions.utterance_log_prob,
        transitions.utterance_value,
        advantages.utterance_advantages,
        advantages.utterance_targets,
        is_utterance_step,
        config,
    )

    # Belief (listener): network consumes (previous_belief, utterance_image).
    belief_train_states, belief_metrics = _update_population(
        belief_rng,
        belief_train_states,
        transitions.belief_previous_belief,
        transitions.belief_utterance_image,
        transitions.belief_action,
        transitions.belief_log_prob,
        transitions.belief_value,
        advantages.belief_advantages,
        advantages.belief_targets,
        is_belief_step,
        config,
    )

    metrics = {
        **{f"utterance_{k}": v for k, v in utterance_metrics.items()},
        **{f"belief_{k}": v for k, v in belief_metrics.items()},
    }
    return belief_train_states, utterance_train_states, metrics
