"""Tests for the e2e agents (agents/) and the population executor (training/agent_execution.py).

Run with::

    uv run pytest tests/test_agents.py
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.e2e_belief_agents import E2EBeliefAgentConfig
from agents.e2e_utterance_agents import E2EUtteranceAgentConfig
from training.agent_execution import execute_agents

BELIEF_DIM = 5
UTTERANCE_ACTION_DIM = 12
IMAGE_DIM = 8
NUM_AGENTS = 3


def _beliefs(key, batch_shape):
    """Three belief levels (b_e, b̄_o, b̿_e) with the given leading batch axes."""
    return tuple(jax.random.dirichlet(k, jnp.ones(BELIEF_DIM), shape=batch_shape) for k in jax.random.split(key, 3))


def _utterance_agent(key, batch_shape):
    agent = E2EUtteranceAgentConfig(trunk_dims=(16,)).build(utterance_action_dim=UTTERANCE_ACTION_DIM)
    return agent, _beliefs(key, batch_shape), UTTERANCE_ACTION_DIM


def _belief_agent(key, batch_shape, multiply_by_ego_belief=False):
    belief_key, image_key = jax.random.split(key)
    image = jax.random.uniform(image_key, (*batch_shape, IMAGE_DIM, IMAGE_DIM))
    agent = E2EBeliefAgentConfig(conv_features=(4,), trunk_dims=(16,), multiply_by_ego_belief=multiply_by_ego_belief).build()
    return agent, (*_beliefs(belief_key, batch_shape), image), BELIEF_DIM


AGENTS = pytest.mark.parametrize("make", [_utterance_agent, _belief_agent], ids=["utterance", "belief"])


@AGENTS
@pytest.mark.parametrize("batch_shape", [(), (4,)], ids=["unbatched", "batched"])
def test_output_shapes(make, batch_shape):
    """The modules accept any leading batch axes, including none (what the executor passes)."""
    init_key, input_key, sample_key = jax.random.split(jax.random.key(0), 3)
    agent, inputs, action_dim = make(input_key, batch_shape)
    params = agent.init(init_key, *inputs)
    pi, value = agent.apply(params, *inputs)
    action, log_prob = pi.sample_and_log_prob(seed=sample_key)

    assert action.shape == (*batch_shape, action_dim)
    assert log_prob.shape == batch_shape
    assert value.shape == batch_shape


@pytest.mark.parametrize("multiply_by_ego_belief", [False, True])
def test_belief_samples_are_on_the_simplex(multiply_by_ego_belief):
    init_key, input_key, sample_key = jax.random.split(jax.random.key(1), 3)
    agent, inputs, _ = _belief_agent(input_key, (4,), multiply_by_ego_belief)
    pi, _ = agent.apply(agent.init(init_key, *inputs), *inputs)
    np.testing.assert_allclose(pi.sample(seed=sample_key).sum(-1), 1.0, atol=1e-5)


def test_multiply_by_ego_belief_preserves_support():
    """With the switch on, states the input ego_belief rules out get the 1e-4 floor
    concentration regardless of params; with it off they don't."""
    init_key, input_key = jax.random.split(jax.random.key(2))
    for multiply, expect_floor in ((True, True), (False, False)):
        agent, (ego, other, second, image), _ = _belief_agent(input_key, (4,), multiply)
        ego = ego.at[:, 0].set(0.0)
        ego = ego / ego.sum(-1, keepdims=True)
        pi, _ = agent.apply(agent.init(init_key, ego, other, second, image), ego, other, second, image)
        assert np.allclose(pi.concentration[:, 0], 1e-4) == expect_floor


@AGENTS
def test_execute_agents_runs_each_agent_on_its_own_row(make):
    """Each agent's executor output matches running that agent's params on its row alone."""
    init_key, input_key, rng = jax.random.split(jax.random.key(3), 3)
    agent, inputs, action_dim = make(input_key, (NUM_AGENTS,))
    row0 = [x[0] for x in inputs]
    params = jax.vmap(lambda k: agent.init(k, *row0))(jax.random.split(init_key, NUM_AGENTS))

    train_states = types.SimpleNamespace(apply_fn=agent.apply, params=params)  # the TrainState fields execute_agents reads
    actions, log_probs, values = execute_agents(rng, train_states, *inputs)

    assert actions.shape == (NUM_AGENTS, action_dim)
    assert log_probs.shape == values.shape == (NUM_AGENTS,)
    for i in range(NUM_AGENTS):
        pi, value = agent.apply(jax.tree.map(lambda x: x[i], params), *(x[i] for x in inputs))
        np.testing.assert_allclose(values[i], value, rtol=1e-5)
        np.testing.assert_allclose(log_probs[i], pi.log_prob(actions[i]), rtol=1e-4)
