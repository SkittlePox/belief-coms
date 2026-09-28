"""Per-population policy execution (SKELETON).

These functions run a whole agent *population* for a single environment step: given
the shared ``apply_fn`` and a parameter pytree whose leaves carry a leading
``[num_agents]`` axis (the batched TrainState layout from ``ippo._batched_train_states``),
they sample an action for every agent and return the per-agent action, log-prob and
value.

They are deliberately thin: the rollout (``training.rollout``) owns the plumbing --
which inputs each population consumes, how utterances get rendered, how the env is
stepped -- while these own only "run this population on these inputs". That separation
is what lets the real execution (ToM variants, action transforms/clipping, deterministic
vs. sampled heads, etc.) be filled in here later without touching the rollout.

Ben: the bodies below are placeholder wiring -- correct shapes, real sampling -- but no
domain-specific logic yet. Replace as needed.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp

# A population handle the executors need: the shared ``apply_fn`` (static) plus the
# batched ``params`` (leading ``[num_agents]`` axis). Bundled so an executor can be
# handed BOTH its own population and its partner's with one argument each -- which is
# exactly what a ToM (second-order) executor requires and a first-order one ignores.
Population = Tuple[object, object]  # (apply_fn, params)


def execute_utterance_agents(rng, apply_fn, params, own_beliefs, partner_estimates):
    """SKELETON -- run the utterance (speaker) population for one step.

    Args:
        rng: PRNGKey; split per agent internally.
        apply_fn: shared ``utterance_train_states.apply_fn`` (ActorCriticUtteranceAgent).
        params: utterance params with a leading ``[num_agents]`` axis.
        own_beliefs: ``[num_agents, belief_dim]`` -- each agent's own belief.
        partner_estimates: ``[num_agents, belief_dim]`` -- each agent's estimate of the
            belief of the receiver it is engaged with.

    Returns ``(actions, log_probs, values)``:
        actions:   ``[num_agents, utterance_action_dim]``
        log_probs: ``[num_agents]``
        values:    ``[num_agents]``

    TODO(Ben): real execution -- any action post-processing (clip/transform), and
    whether utterance agents get a ToM search variant like the reference speaker.
    """
    num_agents = own_beliefs.shape[0]
    agent_rngs = jax.random.split(rng, num_agents)

    def run_one(agent_rng, agent_params, own_belief, partner_estimate):
        # Add a batch dim of 1 (the modules assume a leading batch axis), run, unbatch.
        pi, value = apply_fn(agent_params, own_belief[None], partner_estimate[None])
        action, log_prob = pi.sample_and_log_prob(seed=agent_rng)
        return action[0], log_prob[0], value[0]

    return jax.vmap(run_one, in_axes=(0, 0, 0, 0))(agent_rngs, params, own_beliefs, partner_estimates)


def execute_belief_agents(rng, apply_fn, params, previous_beliefs, utterance_images):
    """SKELETON -- run the belief (listener) population for one step.

    Args:
        rng: PRNGKey; split per agent internally.
        apply_fn: shared ``belief_train_states.apply_fn`` (ActorCriticBeliefAgent).
        params: belief params with a leading ``[num_agents]`` axis.
        previous_beliefs: ``[num_agents, belief_dim]`` -- each agent's existing belief
            (its ``previous_belief`` input; the log-space Bayesian update never expands
            support, so this must be a valid full/partial-support distribution).
        utterance_images: ``[num_agents, image_dim, image_dim]`` -- the rendered utterance
            each agent hears.

    Returns ``(actions, log_probs, values)``:
        actions:   ``[num_agents, belief_dim]`` -- each a next-belief on the simplex.
        log_probs: ``[num_agents]``
        values:    ``[num_agents]``

    TODO(Ben): real execution -- e.g. deterministic (mode) vs. sampled belief, and any
    ToM variant.
    """
    num_agents = previous_beliefs.shape[0]
    agent_rngs = jax.random.split(rng, num_agents)

    def run_one(agent_rng, agent_params, previous_belief, utterance_image):
        pi, value = apply_fn(agent_params, previous_belief[None], utterance_image[None])
        action, log_prob = pi.sample_and_log_prob(seed=agent_rng)
        return action[0], log_prob[0], value[0]

    return jax.vmap(run_one, in_axes=(0, 0, 0, 0))(agent_rngs, params, previous_beliefs, utterance_images)


# =============================================================================
# Second-order (Theory-of-Mind) execution -- SKELETON
# =============================================================================
# The first-order executors above run an agent's OWN network and stop. A ToM executor
# additionally reasons THROUGH a model of its partner: the speaker scores candidate
# utterances by how they would move the *listener's* belief, and the listener inverts the
# *speaker's* model to refine its own belief. That is why every ToM executor takes BOTH
# populations. See ``execute_tom_speaker`` / ``execute_tom_listener`` in
# ``siggame_reference/ippo_ff.py`` for the reference recursion (adapted here: the belief
# agent emits a Dirichlet over the belief simplex rather than a categorical over referents,
# so the scoring/inversion math is intentionally left for you to define).
#
# Partner alignment: these skeletons vmap the partner population with ``in_axes=0``, i.e.
# they treat ``partner_params[i]`` as the model of the agent that agent ``i`` is engaged
# with. The rollout currently passes each population index-aligned; wiring in the true
# partner gather (via ``env_state.game_roles``) is part of making ToM correct.


def execute_tom_utterance_agents(
    rng,
    utterance_apply_fn,
    utterance_params,
    belief_apply_fn,
    belief_params,
    own_beliefs,
    partner_estimates,
    render_utterance_fn,
    n_search,
):
    """SKELETON -- second-order (ToM) utterance (speaker) population.

    Same return contract as :func:`execute_utterance_agents`
    ``(actions, log_probs, values)``, but each agent additionally has access to the belief
    (listener) population so it can search over utterances that best steer the listener.

    Args (beyond the first-order version):
        belief_apply_fn, belief_params: the belief (listener) population, so the speaker can
            simulate the listener's update.
        render_utterance_fn: raw-utterance -> image, needed to feed candidate utterances
            through the (image-consuming) belief model.
        n_search: number of candidate utterances proposed per agent.
    """
    num_agents = own_beliefs.shape[0]
    agent_rngs = jax.random.split(rng, num_agents)

    def run_one(agent_rng, utt_params_i, bel_params_i, own_belief, partner_estimate):
        # 1) Propose n_search candidate utterances from the agent's own utterance policy.
        pi, value = utterance_apply_fn(utt_params_i, own_belief[None], partner_estimate[None])
        candidates = pi.sample(seed=agent_rng, sample_shape=(n_search,))[:, 0]  # [n_search, utt_dim]

        # 2) TODO(Ben): score each candidate by how well it moves the LISTENER's belief
        #    toward the speaker's own belief. Sketch:
        #        images = render_utterance_fn(agent_rng, candidates)            # [n_search, D, D]
        #        listener_posteriors = belief_apply_fn(bel_params_i,
        #                                              partner_estimate broadcast, images)
        #        score = -KL(listener_posterior || own_belief)  (or cross-entropy, etc.)
        #    then select argmax, or sample softmax(beta * score). See execute_tom_speaker.
        #
        # PLACEHOLDER until the scorer exists: take the first candidate (a plain sample),
        # so this runs and returns correct shapes while behaving first-order.
        chosen = candidates[0]
        log_prob = pi.log_prob(chosen[None])[0]
        return chosen, log_prob, value[0]

    return jax.vmap(run_one, in_axes=(0, 0, 0, 0, 0))(agent_rngs, utterance_params, belief_params, own_beliefs, partner_estimates)


def execute_tom_belief_agents(
    rng,
    belief_apply_fn,
    belief_params,
    utterance_apply_fn,
    utterance_params,
    previous_beliefs,
    utterance_images,
    render_utterance_fn,
    n_samples,
):
    """SKELETON -- second-order (ToM) belief (listener) population.

    Same return contract as :func:`execute_belief_agents` ``(actions, log_probs, values)``,
    but each agent additionally has access to the utterance (speaker) population so it can
    invert the speaker's model when forming its new belief.

    Args (beyond the first-order version):
        utterance_apply_fn, utterance_params: the utterance (speaker) population, so the
            listener can ask "which world states would have produced the utterance I heard?".
        render_utterance_fn: raw-utterance -> image, for comparing hypothesized speaker
            utterances against the heard one in image space.
        n_samples: number of speaker-utterance samples drawn per hypothesis.
    """
    num_agents = previous_beliefs.shape[0]
    agent_rngs = jax.random.split(rng, num_agents)

    def run_one(agent_rng, bel_params_i, utt_params_i, previous_belief, utterance_image):
        pi, value = belief_apply_fn(bel_params_i, previous_belief[None], utterance_image[None])

        # TODO(Ben): refine `pi` by inverting the SPEAKER model. Sketch:
        #    for each hypothesized world state, sample the utterances the speaker would emit
        #    (utterance_apply_fn + utt_params_i), render them, and weight states by how
        #    consistent the heard `utterance_image` is with each hypothesis (times a P_R
        #    prior). Combine with the first-order posterior. See execute_tom_listener.
        #
        # PLACEHOLDER until the inversion exists: sample from the first-order belief policy,
        # so this runs and returns correct shapes while behaving first-order.
        action, log_prob = pi.sample_and_log_prob(seed=agent_rng)
        return action[0], log_prob[0], value[0]

    return jax.vmap(run_one, in_axes=(0, 0, 0, 0, 0))(agent_rngs, belief_params, utterance_params, previous_beliefs, utterance_images)


# =============================================================================
# Inference strategy: a uniform interface the rollout calls, so env_step is agnostic
# to whether the agents think first-order or with ToM.
# =============================================================================


class InferenceStrategy:
    """Interface the rollout uses to run each population.

    Both methods share a uniform signature -- own population, partner population, the
    per-agent inputs, and the renderer -- so ``env_step`` calls them identically no matter
    the strategy. A first-order strategy ignores the partner population; a ToM strategy uses
    it.
    """

    def run_utterance_agents(self, rng, utterance_pop: Population, belief_pop: Population, own_beliefs, partner_estimates, render_utterance_fn):
        raise NotImplementedError

    def run_belief_agents(self, rng, belief_pop: Population, utterance_pop: Population, previous_beliefs, utterance_images, render_utterance_fn):
        raise NotImplementedError


class FirstOrderInference(InferenceStrategy):
    """Each agent runs only its own network (the naive path); partner models unused."""

    def run_utterance_agents(self, rng, utterance_pop, belief_pop, own_beliefs, partner_estimates, render_utterance_fn):
        apply_fn, params = utterance_pop
        return execute_utterance_agents(rng, apply_fn, params, own_beliefs, partner_estimates)

    def run_belief_agents(self, rng, belief_pop, utterance_pop, previous_beliefs, utterance_images, render_utterance_fn):
        apply_fn, params = belief_pop
        return execute_belief_agents(rng, apply_fn, params, previous_beliefs, utterance_images)


@dataclasses.dataclass(frozen=True)
class ToMInference(InferenceStrategy):
    """Second-order path: each agent reasons through its partner's model.

    Holds the ToM search hyperparameters (static Python scalars, so they stay out of the
    traced computation) and dispatches to the ToM executor skeletons above.
    """

    speaker_n_search: int
    listener_n_samples: int

    def run_utterance_agents(self, rng, utterance_pop, belief_pop, own_beliefs, partner_estimates, render_utterance_fn):
        utt_apply_fn, utt_params = utterance_pop
        bel_apply_fn, bel_params = belief_pop
        return execute_tom_utterance_agents(
            rng,
            utt_apply_fn,
            utt_params,
            bel_apply_fn,
            bel_params,
            own_beliefs,
            partner_estimates,
            render_utterance_fn,
            self.speaker_n_search,
        )

    def run_belief_agents(self, rng, belief_pop, utterance_pop, previous_beliefs, utterance_images, render_utterance_fn):
        bel_apply_fn, bel_params = belief_pop
        utt_apply_fn, utt_params = utterance_pop
        return execute_tom_belief_agents(
            rng,
            bel_apply_fn,
            bel_params,
            utt_apply_fn,
            utt_params,
            previous_beliefs,
            utterance_images,
            render_utterance_fn,
            self.listener_n_samples,
        )


@dataclasses.dataclass(frozen=True)
class AgentInferenceConfig:
    """Tyro-facing selector for how agents reason during rollouts.

    Mirrors the ``*Config.build()`` idiom used across the codebase: a dataclass of knobs
    whose ``build()`` returns the configured runtime object (here an ``InferenceStrategy``).
    ``mode`` picks first-order vs. ToM; the remaining fields are ToM search knobs, ignored
    in first-order mode.
    """

    mode: str = "first_order"  # "first_order" | "tom"
    speaker_n_search: int = 5
    listener_n_samples: int = 50

    def build(self) -> InferenceStrategy:
        if self.mode == "first_order":
            return FirstOrderInference()
        if self.mode == "tom":
            return ToMInference(speaker_n_search=self.speaker_n_search, listener_n_samples=self.listener_n_samples)
        raise ValueError(f"unknown inference mode {self.mode!r} (expected 'first_order' or 'tom')")
