"""Inferential (Theory-of-Mind) communication -- SKELETON.

Population executors in which each agent reasons through a model of its partner. They are
built on the e2e networks (``e2e_belief_agents``, ``e2e_utterance_agents``): an agent's own
e2e network proposes or forms its belief, and its partner population's e2e network is the
model it reasons through. ``AgentInferenceConfig`` runs these in "tom" mode. Both currently
behave like the e2e agents: they run the agent's own network and take a plain sample.
"""

from __future__ import annotations

import jax

# The e2e executor (``training.agent_execution.execute_agents``) runs an agent's
# OWN network and stops. A ToM executor additionally reasons THROUGH a model of its
# partner: the speaker scores candidate utterances by how they would move the *listener's*
# belief, and the listener inverts the *speaker's* model to refine its own belief. That is
# why every ToM executor takes BOTH populations. See ``execute_tom_speaker`` /
# ``execute_tom_listener`` in ``siggame_reference/ippo_ff.py`` for the reference recursion (adapted here: the belief
# agent emits a Dirichlet over the belief simplex rather than a categorical over referents,
# so the scoring/inversion math is intentionally left for you to define).
#
# Partner alignment: these skeletons vmap the partner population with ``in_axes=0``, i.e.
# they treat ``partner_params[i]`` as the model of the agent that agent ``i`` is engaged
# with. The rollout currently passes each population index-aligned; wiring in the true
# partner gather (via ``env_state.game_roles``) is part of making ToM correct.


def execute_tom_utterance_agents(
    rng,
    utterance_train_states,
    belief_train_states,
    ego_beliefs,
    other_belief_estimates,
    second_order_ego_estimates,
    render_utterance_fn,
    n_search,
):
    """SKELETON -- second-order (ToM) utterance (speaker) population.

    Same return contract as ``training.agent_execution.execute_agents``
    ``(actions, log_probs, values)``, but each agent additionally has access to the belief
    (listener) population so it can search over utterances that best steer the listener.

    Args (beyond the e2e version):
        belief_train_states: the belief (listener) population, so the speaker can simulate
            the listener's update.
        render_utterance_fn: raw-utterance -> image, needed to feed candidate utterances
            through the (image-consuming) belief model.
        n_search: number of candidate utterances proposed per agent.
    """
    agent_rngs = jax.random.split(rng, ego_beliefs.shape[0])

    def run_one(agent_rng, utt_params_i, bel_params_i, ego_belief, other_belief_estimate, second_order_ego_estimate):
        # 1) Propose n_search candidate utterances from the agent's own utterance policy.
        pi, value = utterance_train_states.apply_fn(utt_params_i, ego_belief, other_belief_estimate, second_order_ego_estimate)
        candidates = pi.sample(seed=agent_rng, sample_shape=(n_search,))  # [n_search, utt_dim]

        # 2) TODO(Ben): score each candidate by how well it moves the LISTENER's belief
        #    toward the speaker's own belief. Sketch:
        #        images = render_utterance_fn(agent_rng, candidates)            # [n_search, D, D]
        #        listener_pi, _ = belief_train_states.apply_fn(bel_params_i,
        #            other_belief_estimate, ego_belief, second_order_ego_estimate (each broadcast),
        #            images)
        #        score = -KL(listener posterior || own_belief)  (or cross-entropy, etc.)
        #    then select argmax, or sample softmax(beta * score). See execute_tom_speaker.
        #
        # PLACEHOLDER until the scorer exists: take the first candidate (a plain sample),
        # so this runs and returns correct shapes while behaving like the e2e agent.
        chosen = candidates[0]
        return chosen, pi.log_prob(chosen), value

    return jax.vmap(run_one)(
        agent_rngs, utterance_train_states.params, belief_train_states.params, ego_beliefs, other_belief_estimates, second_order_ego_estimates
    )


def execute_tom_belief_agents(
    rng,
    belief_train_states,
    utterance_train_states,
    ego_beliefs,
    other_belief_estimates,
    second_order_ego_estimates,
    utterance_images,
    render_utterance_fn,
    n_samples,
):
    """SKELETON -- second-order (ToM) belief (listener) population.

    Same return contract as ``training.agent_execution.execute_agents``
    ``(actions, log_probs, values)``, but each agent additionally has access to the
    utterance (speaker) population so it can invert the speaker's model when forming its
    new belief.

    Args (beyond the e2e version):
        utterance_train_states: the utterance (speaker) population, so the listener can ask
            "which world states would have produced the utterance I heard?".
        render_utterance_fn: raw-utterance -> image, for comparing hypothesized speaker
            utterances against the heard one in image space.
        n_samples: number of speaker-utterance samples drawn per hypothesis.
    """
    agent_rngs = jax.random.split(rng, ego_beliefs.shape[0])

    def run_one(agent_rng, bel_params_i, utt_params_i, ego_belief, other_belief_estimate, second_order_ego_estimate, utterance_image):
        pi, value = belief_train_states.apply_fn(bel_params_i, ego_belief, other_belief_estimate, second_order_ego_estimate, utterance_image)

        # TODO(Ben): refine `pi` by inverting the SPEAKER model. Sketch:
        #    for each hypothesized world state, sample the utterances the speaker would emit
        #    (utterance_train_states.apply_fn + utt_params_i), render them, and weight states by how
        #    consistent the heard `utterance_image` is with each hypothesis (times a P_R
        #    prior). Combine with the e2e posterior. See execute_tom_listener.
        #
        # PLACEHOLDER until the inversion exists: sample from the e2e belief policy,
        # so this runs and returns correct shapes while behaving like the e2e agent.
        action, log_prob = pi.sample_and_log_prob(seed=agent_rng)
        return action, log_prob, value

    return jax.vmap(run_one)(
        agent_rngs,
        belief_train_states.params,
        utterance_train_states.params,
        ego_beliefs,
        other_belief_estimates,
        second_order_ego_estimates,
        utterance_images,
    )
