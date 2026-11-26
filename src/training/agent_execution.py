"""Running an agent population for one environment step.

A population is a batched ``TrainState``: one shared ``apply_fn`` and params whose leaves
carry a leading ``[num_agents]`` axis (see ``ippo._batched_train_states``). Running it
means each agent's network on its own row of the inputs, sampling an action, and returning
the per-agent ``(actions, log_probs, values)``.

``AgentInferenceConfig`` picks how each population runs: e2e (each agent's own network,
``execute_agents`` below) or ToM (reasoning through the partner population's model; the
skeletons in ``agents.inference_agents``).
"""

from __future__ import annotations

import dataclasses

import jax

from agents.inference_agents import execute_tom_belief_agents, execute_tom_utterance_agents


def execute_agents(rng, train_states, *inputs):
    """Run an e2e population for one step: each agent's own network on its row of ``inputs``.

    Args:
        rng: PRNGKey; split per agent internally.
        train_states: the population's batched TrainState (E2EUtteranceAgent / E2EBeliefAgent).
        *inputs: the network's positional inputs, each ``[num_agents, ...]``; row i is what
            agent i consumes.

    Returns ``(actions, log_probs, values)``:
        actions:   ``[num_agents, utterance_action_dim]`` utterances or
                   ``[num_agents, belief_dim]`` new beliefs, row i held by agent i
        log_probs: ``[num_agents]``
        values:    ``[num_agents]``
    """

    def run_one(agent_rng, agent_params, *agent_inputs):
        pi, value = train_states.apply_fn(agent_params, *agent_inputs)
        action, log_prob = pi.sample_and_log_prob(seed=agent_rng)
        return action, log_prob, value

    num_agents = inputs[0].shape[0]
    return jax.vmap(run_one)(jax.random.split(rng, num_agents), train_states.params, *inputs)


@dataclasses.dataclass(frozen=True)
class AgentInferenceConfig:
    """How agents run during rollouts: ``mode`` is "e2e" or "tom".

    The rollout calls ``run_utterance_agents`` / ``run_belief_agents``, which take both
    populations so a ToM agent can reason through its partner's model; e2e ignores the
    partner. The remaining fields are ToM search knobs, ignored in e2e mode. ``mode`` is a
    static Python value, so the dispatch happens once at trace time.
    """

    mode: str = "e2e"  # "e2e" | "tom"
    speaker_n_search: int = 5
    listener_n_samples: int = 50

    def __post_init__(self):
        if self.mode not in ("e2e", "tom"):
            raise ValueError(f"unknown inference mode {self.mode!r} (expected 'e2e' or 'tom')")

    def run_utterance_agents(
        self, rng, utterance_train_states, belief_train_states, ego_beliefs, other_belief_estimates, second_order_ego_estimates, render_utterance_fn
    ):
        if self.mode == "e2e":
            return execute_agents(rng, utterance_train_states, ego_beliefs, other_belief_estimates, second_order_ego_estimates)
        return execute_tom_utterance_agents(
            rng,
            utterance_train_states,
            belief_train_states,
            ego_beliefs,
            other_belief_estimates,
            second_order_ego_estimates,
            render_utterance_fn,
            self.speaker_n_search,
        )

    def run_belief_agents(
        self,
        rng,
        belief_train_states,
        utterance_train_states,
        ego_beliefs,
        other_belief_estimates,
        second_order_ego_estimates,
        utterance_images,
        render_utterance_fn,
    ):
        if self.mode == "e2e":
            return execute_agents(rng, belief_train_states, ego_beliefs, other_belief_estimates, second_order_ego_estimates, utterance_images)
        return execute_tom_belief_agents(
            rng,
            belief_train_states,
            utterance_train_states,
            ego_beliefs,
            other_belief_estimates,
            second_order_ego_estimates,
            utterance_images,
            render_utterance_fn,
            self.listener_n_samples,
        )
