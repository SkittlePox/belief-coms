"""Shared test harness for communication/stacked_signification_decpomdp.py.

The stacked env's step_env takes three LEARNED inputs every call (utterances, the
speaker's post-utterance estimate of its listener, the listener's new belief). To test
the env's bookkeeping exactly we replace those networks with fixed, known drivers:

  * utterances: a SENTINEL per agent -- row i is filled with ``i + 1`` -- so a value
    that ends up in the wrong row (wrong partner, wrong role) is a visible wrong number
    rather than a silently plausible one. Zero is avoided because the env writes zeros
    for non-speakers.
  * beliefs: PASSTHROUGH -- each listener "adopts" its current true belief, so
    communication is a no-op and the true beliefs stay the exact Bayesian posteriors
    produced by the act steps.
  * estimates: ORACLE (default) -- the estimate ABOUT agent i is set to agent i's true
    belief -- or KEEP, which leaves the stored estimates untouched.

Usage::

    h = Harness.build(scheme="a_to_b_thrice", num_agents=4, horizon=2)
    state, obs = h.reset(seed=0)
    roll = h.rollout(seed=0, num_steps=12)
    [schedule(s) for s in roll.states]

With the default ``fixed_pairs_assignment_fn``, agent i is in game ``i // 2`` with role
``i % 2``, so partners are (0, 1), (2, 3), ... and role 0 (the presser) is the even agent.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal, NamedTuple, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

from communication.communication_scheme import CommunicationScheme, get_scheme_fn
from communication.game_role_assignment import fixed_pairs_assignment_fn
from communication.stacked_signification_decpomdp import StackedSignificationDecPOMDP, StackedSignificationState
from envs.env_assembly import assemble_environments, guessing_game_spec

EstimateMode = Literal["oracle", "keep"]


class Actions(NamedTuple):
    """The three per-call step_env inputs, with step_env's own index conventions."""

    utterances: chex.Array  # [num_agents, utterance_action_dim], AGENT-indexed
    estimates: chex.Array  # [num_agents, S], SUBJECT-indexed (row i = estimate ABOUT agent i)
    beliefs: chex.Array  # [num_agents, S], AGENT-indexed (row i = agent i's new belief)


class Rollout(NamedTuple):
    """A driven trajectory. ``states[0]`` / ``obs[0]`` are from reset; entry t+1 is after
    step t. ``pre_act_states``, ``rewards`` and ``actions`` have one entry per step."""

    states: list
    obs: list
    pre_act_states: list
    rewards: list
    actions: list


def sentinel_utterances(num_agents: int, utterance_action_dim: int) -> chex.Array:
    """Row i filled with ``i + 1``: identifies which agent's utterance landed where."""
    return jnp.broadcast_to(jnp.arange(1, num_agents + 1, dtype=jnp.float32)[:, None], (num_agents, utterance_action_dim))


def padded_scheme_fn(rows: Sequence[Sequence[int]], pad_to: int) -> Callable:
    """A constant scheme whose who_speaks is zero-padded to ``pad_to`` rows while
    total_num_rounds stays the REAL length -- for testing that padding is never walked."""
    real = CommunicationScheme.from_rows(rows)
    assert pad_to >= real.who_speaks.shape[0]
    padding = jnp.zeros((pad_to - real.who_speaks.shape[0], real.who_speaks.shape[1]), dtype=jnp.int32)
    scheme = CommunicationScheme(jnp.concatenate([real.who_speaks, padding]), real.total_num_rounds)
    return lambda iteration: scheme


def schedule(state: StackedSignificationState) -> tuple:
    """The scheduling counters as plain ints, for comparing against hand-written traces:
    (stage, round, underlying_env_timestep, cumulative_env_timestep, comm_steps, episode_index)."""
    return (
        int(state.dialog.communicative_round_stage),
        int(state.dialog.communication_round_iterator),
        int(state.game_counters.underlying_env_timestep),
        int(state.game_counters.cumulative_env_timestep),
        int(state.dialog.cumulative_communication_round_iterator),
        int(state.game_counters.episode_index),
    )


def partners(env: StackedSignificationDecPOMDP, state: StackedSignificationState) -> np.ndarray:
    """[num_agents] partner index per agent, via the env's own _partner_agent."""
    roles = state.game_roles
    return np.asarray(env._partner_agent(roles.agent_game_assignment, roles.agent_role_assignment, state.game_states.shape[0]))


@dataclasses.dataclass
class Harness:
    env: StackedSignificationDecPOMDP
    estimate_mode: EstimateMode = "oracle"

    def __post_init__(self):
        # Jit once per harness; the env is closed over (not an argument), so the trace is reused.
        self._step = jax.jit(self.env.step_env_with_substate)

    @classmethod
    def build(
        cls,
        scheme: str | Callable = "a_to_b",
        num_agents: int = 4,
        horizon: int = 3,
        act_first: bool = False,
        utterance_action_dim: int = 3,
        assignment_fn: Callable | None = None,
        estimate_mode: EstimateMode = "oracle",
    ) -> "Harness":
        """A small guessing-game env. ``scheme`` is a scheme name or a CommunicationSchemeFn."""
        stacked_params, optimal_policies = assemble_environments([guessing_game_spec])
        env = StackedSignificationDecPOMDP(
            num_agents=num_agents,
            all_env_parameters=stacked_params,
            optimal_policies=optimal_policies,
            assignment_fn=assignment_fn or fixed_pairs_assignment_fn(num_agents=num_agents, underlying_env_steps_per_episode=horizon),
            communication_scheme_fn=get_scheme_fn(scheme) if isinstance(scheme, str) else scheme,
            utterance_action_dim=utterance_action_dim,
            skip_first_communication_step=act_first,
        )
        return cls(env, estimate_mode)

    def reset(self, seed: int = 0):
        return self.env.reset(jax.random.key(seed))

    def actions(self, state: StackedSignificationState) -> Actions:
        """Sentinel utterances, passthrough beliefs, oracle/keep estimates for ``state``."""
        estimates = state.true_agent_belief_states if self.estimate_mode == "oracle" else state.estimated_agent_belief_states
        return Actions(
            utterances=sentinel_utterances(self.env.num_agents, self.env.utterance_action_dim),
            estimates=estimates,
            beliefs=state.true_agent_belief_states,
        )

    def step(self, key: chex.PRNGKey, state: StackedSignificationState, actions: Actions | None = None):
        """One step_env call; returns (pre_act_state, state, obs, rewards)."""
        actions = self.actions(state) if actions is None else actions
        return self._step(key, state, actions.utterances, actions.estimates, actions.beliefs)

    def rollout(self, seed: int = 0, num_steps: int = 10, actions_fn: Callable | None = None) -> Rollout:
        """Reset then drive ``num_steps`` step_env calls. ``actions_fn(state) -> Actions``
        overrides the default drivers."""
        actions_fn = actions_fn or self.actions
        reset_key, walk_key = jax.random.split(jax.random.key(seed))
        state, obs = self.env.reset(reset_key)
        roll = Rollout([state], [obs], [], [], [])
        for step_key in jax.random.split(walk_key, num_steps):
            actions = actions_fn(state)
            pre_act, state, obs, rewards = self.step(step_key, state, actions)
            roll.states.append(state)
            roll.obs.append(obs)
            roll.pre_act_states.append(pre_act)
            roll.rewards.append(rewards)
            roll.actions.append(actions)
        return roll
