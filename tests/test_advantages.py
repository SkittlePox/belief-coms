"""Tests for training/advantages.py (staged GAE).

Run with::

    uv run pytest tests/test_advantages.py

The oracle here is a plain contiguous GAE applied to each population's *own* decision
substream (belief agents on belief stages; speakers on utterance stages, with the reward
they earn taken from the following belief stage). ``compute_advantages`` must match it on
the decision steps and emit zeros on the other population's steps.
"""

import types

import jax.numpy as jnp
import numpy as np

from communication.stacked_signification_decpomdp import BELIEF_STAGE, UTTERANCE_STAGE
from training.advantages import compute_advantages

GAMMA = 0.9
LAM = 0.8


def _vanilla_gae(values, rewards, dones, last_value, gamma, lam):
    """Textbook GAE over a contiguous trajectory. Arrays are ``[T, num_agents]``."""
    T = values.shape[0]
    advantages = np.zeros_like(values)
    gae = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(T)):
        cont = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * cont - values[t]
        gae = delta + gamma * lam * cont * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


def _make_transitions(stage, reward, done, belief_value, utterance_value):
    return types.SimpleNamespace(
        stage=jnp.asarray(stage),
        reward=jnp.asarray(reward),
        done=jnp.asarray(done),
        belief_value=jnp.asarray(belief_value),
        utterance_value=jnp.asarray(utterance_value),
    )


def _oracle(stage, reward, done, belief_value, utterance_value):
    """Independent per-population GAE by extracting each substream, scattering back."""
    num_steps, num_agents = reward.shape
    belief_adv = np.zeros((num_steps, num_agents))
    utt_adv = np.zeros((num_steps, num_agents))

    # Belief (listener): decisions on belief stages; reward/done on that same step.
    bel_idx = [t for t in range(num_steps) if stage[t] == BELIEF_STAGE]
    if bel_idx:
        sub = _vanilla_gae(
            belief_value[bel_idx], reward[bel_idx], done[bel_idx],
            np.zeros(num_agents), GAMMA, LAM,
        )
        for k, t in enumerate(bel_idx):
            belief_adv[t] = sub[k]

    # Utterance (speaker): decisions on utterance stages; reward/done taken from the
    # following step (guaranteed a belief stage by strict alternation), zero-padded at the
    # end.
    utt_idx = [t for t in range(num_steps) if stage[t] == UTTERANCE_STAGE]
    if utt_idx:
        r = np.array([reward[t + 1] if t + 1 < num_steps else np.zeros(num_agents) for t in utt_idx])
        d = np.array([done[t + 1] if t + 1 < num_steps else np.zeros(num_agents) for t in utt_idx])
        sub = _vanilla_gae(utterance_value[utt_idx], r, d, np.zeros(num_agents), GAMMA, LAM)
        for k, t in enumerate(utt_idx):
            utt_adv[t] = sub[k]

    return utt_adv, belief_adv


def test_staged_gae_matches_oracle_no_boundary():
    # Stages strictly alternate U,B,U,B,U,B; reward only on belief (act) steps.
    stage = np.array([UTTERANCE_STAGE, BELIEF_STAGE, UTTERANCE_STAGE, BELIEF_STAGE, UTTERANCE_STAGE, BELIEF_STAGE])
    num_agents = 2
    reward = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [0.5, 2.0], [0.0, 0.0], [3.0, 1.0]])
    done = np.zeros(6)
    rng = np.random.default_rng(0)
    belief_value = rng.normal(size=(6, num_agents))
    utterance_value = rng.normal(size=(6, num_agents))

    out = compute_advantages(
        _make_transitions(stage, reward, done, belief_value, utterance_value),
        GAMMA, LAM,
    )
    # done broadcast to [T, num_agents] for the oracle.
    done2 = np.repeat(done[:, None], num_agents, axis=1)
    exp_utt, exp_bel = _oracle(stage, reward, done2, belief_value, utterance_value)

    np.testing.assert_allclose(np.asarray(out.belief_advantages), exp_bel, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out.utterance_advantages), exp_utt, rtol=1e-5, atol=1e-6)


def test_staged_gae_respects_episode_boundary():
    # A boundary on the middle belief step (t=3) must stop bootstrapping across it.
    stage = np.array([UTTERANCE_STAGE, BELIEF_STAGE, UTTERANCE_STAGE, BELIEF_STAGE, UTTERANCE_STAGE, BELIEF_STAGE])
    num_agents = 1
    reward = np.array([[0.0], [1.0], [0.0], [2.0], [0.0], [3.0]])
    done = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    rng = np.random.default_rng(1)
    belief_value = rng.normal(size=(6, num_agents))
    utterance_value = rng.normal(size=(6, num_agents))

    out = compute_advantages(
        _make_transitions(stage, reward, done, belief_value, utterance_value),
        GAMMA, LAM,
    )
    done2 = np.repeat(done[:, None], num_agents, axis=1)
    exp_utt, exp_bel = _oracle(stage, reward, done2, belief_value, utterance_value)

    np.testing.assert_allclose(np.asarray(out.belief_advantages), exp_bel, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out.utterance_advantages), exp_utt, rtol=1e-5, atol=1e-6)


def test_targets_are_advantage_plus_value():
    stage = np.array([UTTERANCE_STAGE, BELIEF_STAGE, UTTERANCE_STAGE, BELIEF_STAGE])
    reward = np.array([[0.0], [1.0], [0.0], [2.0]])
    done = np.zeros(4)
    belief_value = np.array([[0.1], [0.2], [0.3], [0.4]])
    utterance_value = np.array([[0.5], [0.6], [0.7], [0.8]])

    out = compute_advantages(
        _make_transitions(stage, reward, done, belief_value, utterance_value),
        GAMMA, LAM,
    )
    np.testing.assert_allclose(
        np.asarray(out.belief_targets), np.asarray(out.belief_advantages) + belief_value, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(out.utterance_targets), np.asarray(out.utterance_advantages) + utterance_value, rtol=1e-6
    )
