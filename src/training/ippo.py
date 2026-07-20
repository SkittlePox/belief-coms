"""Independent PPO (IPPO)."""

from __future__ import annotations
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState

from training.config import ExperimentConfig
from training.optimizer import OptimizerConfig
from agents.belief_agents import BeliefAgentConfig
from agents.utterance_agents import UtteranceAgentConfig


def _batched_train_states(network: nn.Module, tx, rng, num_agents, *init_inputs) -> TrainState:
    """Init ``num_agents`` independent parameter sets and wrap each in a TrainState.

    A single network *definition* is shared by every agent, but each agent gets its own
    randomly initialized params (one per split of ``rng``) and its own optimizer state.
    Rather than a Python list of TrainStates, we ``vmap`` the per-agent init so the result
    is one TrainState pytree with a leading ``num_agents`` axis on the array leaves
    (params, opt_state, step); the static fields (``apply_fn``, ``tx``) are shared. That
    batched layout is what lets the downstream update ``vmap``/scan over agents.

    ``init_inputs`` are the dummy forward-pass inputs (batch dimension of 1) whose shapes
    seed the parameter shapes; they are identical across agents, so they stay outside the
    vmap.
    """

    def init_one(key):
        params = network.init(key, *init_inputs)
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    return jax.vmap(init_one)(jax.random.split(rng, num_agents))


def initialize_belief_agents(
    agent_config: BeliefAgentConfig,
    optimizer_config: OptimizerConfig,
    rng,
    num_agents: int,
    input_utterance_shape: Sequence[int],
    belief_dim: int,
) -> Tuple[nn.Module, TrainState]:
    """Build one belief-agent network and ``num_agents`` initialized train states.

    Returns ``(network, train_states)`` where ``network`` is the shared module definition
    (its ``apply`` is the batched TrainState's ``apply_fn``) and ``train_states`` is the
    vmapped TrainState (leading ``num_agents`` axis on params/opt_state).
    """
    network = agent_config.build(input_utterance_shape=input_utterance_shape, belief_dim=belief_dim)
    tx = optimizer_config.build()

    # Dummy inputs to shape-init the params: a valid previous belief (uniform over the
    # simplex -- zeros would send BeliefActor's log-prior to -inf) and a rendered
    # utterance image. Batch dim of 1; shared across agents.
    dummy_belief = jnp.full((1, belief_dim), 1.0 / belief_dim)
    dummy_utterance = jnp.zeros((1, *input_utterance_shape))

    train_states = _batched_train_states(network, tx, rng, num_agents, dummy_belief, dummy_utterance)
    return network, train_states


def initialize_utterance_agents(
    agent_config: UtteranceAgentConfig,
    optimizer_config: OptimizerConfig,
    rng,
    num_agents: int,
    utterance_action_dim: int,
    belief_dim: int,
) -> Tuple[nn.Module, TrainState]:
    """Build one utterance-agent network and ``num_agents`` initialized train states.

    Same contract as ``initialize_belief_agents``; the utterance agent reads the sender's
    own belief and its estimate of the receiver's belief (both length ``belief_dim``).
    """
    network = agent_config.build(utterance_action_dim=utterance_action_dim, belief_dim=belief_dim)
    tx = optimizer_config.build()

    # Dummy inputs: the sender's own belief and its estimate of the receiver's belief.
    dummy_own_belief = jnp.full((1, belief_dim), 1.0 / belief_dim)
    dummy_estimate = jnp.full((1, belief_dim), 1.0 / belief_dim)

    train_states = _batched_train_states(network, tx, rng, num_agents, dummy_own_belief, dummy_estimate)
    return network, train_states


def make_train(config: ExperimentConfig):

    # Each sub-config knows how to build its runtime function, so resolution is
    # uniform: assignment -> AssignmentFn, communication -> CommunicationSchemeFn.
    assignment_fn = config.role_assignment.build()
    scheme_fn = config.communication.build()
    num_agents = config.role_assignment.num_agents

    # The env config assembles the stacked DecPOMDP params + optimal-policy table from
    # the selected games and wires in the two runtime functions above.
    env = config.environment.build(
        num_agents=num_agents,
        assignment_fn=assignment_fn,
        communication_scheme_fn=scheme_fn,
    )

    # Env-derived shapes the agents need. belief_dim is the (padded) world-state
    # cardinality (length of every belief vector). A belief agent consumes a rendered
    # utterance image of side utterance_image_dim; the utterance agent emits a flat
    # utterance of length utterance_action_dim (which must match the env's).
    belief_dim = env.belief_dim
    utterance_action_dim = config.environment.utterance_action_dim
    image_dim = config.environment.utterance_image_dim
    input_utterance_shape = (image_dim, image_dim)

    def train(rng):
        # One independent population per role: num_agents belief agents and num_agents
        # utterance agents, each a distinct parameter set + optimizer state batched under
        # a leading num_agents axis.
        belief_rng, utterance_rng = jax.random.split(rng)
        belief_network, belief_train_states = initialize_belief_agents(
            config.belief_agents,
            config.belief_optimizer,
            belief_rng,
            num_agents,
            input_utterance_shape=input_utterance_shape,
            belief_dim=belief_dim,
        )
        utterance_network, utterance_train_states = initialize_utterance_agents(
            config.utterance_agents,
            config.utterance_optimizer,
            utterance_rng,
            num_agents,
            utterance_action_dim=utterance_action_dim,
            belief_dim=belief_dim,
        )

        # TODO: reset the env, run the rollout collecting per-agent trajectories, and run
        # the PPO update over both populations.

    return train
