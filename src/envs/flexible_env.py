"""Generic dense-DecPOMDP environment.

This is the *foundation* of the env-definition dependency graph: it defines
``FlexibleEnvParams`` (the dense tensors describing one DecPOMDP), the
``OptimalPolicy`` type, and ``FlexibleEnv`` — a generic runtime that drives any
``FlexibleEnvParams`` (reset / step_env / get_obs plus the ``_joint_*`` gather
API). Per-game-type definitions (e.g. envs/guessing_game.py) build a
``FlexibleEnvParams``; envs/env_assembly.py stacks them. Neither this module nor the
env-definition modules import the stacked env, keeping the import graph acyclic.
"""

import jax, chex
import distrax
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Callable


# A policy maps a belief (Categorical over states) to an action distribution
# (Categorical over actions). Arbitrary Python callables — kept OUT of
# FlexibleEnvParams (which holds only traceable array data).
OptimalPolicy = Callable[[distrax.Categorical], distrax.Categorical]


@struct.dataclass
class FlexibleEnvParams:
    """Dense dynamics tensors for a (variant of a) DecPOMDP.

    Everything that distinguishes one environment variant from another lives
    here as plain arrays, so different variants are just different instances
    *with identical shapes* — which is what keeps a vmap-over-environments clean
    (no per-env Python branching / lax.switch).

    Conditioning (the most general the consumers can feed):

        transition   T(s' | s, a0, a1)         shape [S, A, A, S]
        observation  O(o0, o1 | s', a0, a1)    shape [S, A, A, O, O]
        reward       R_i(s, a0, a1, s')        shape [N, S, A, A, S]

    ``num_states`` / ``num_actions`` record the *real* (pre-padding) cardinality
    so consumers can mask the zero-padded tail. ``initial_state_distribution``
    [S] is the prior over the true world state at reset, and it is also every
    agent's prior belief. ``terminal_mask`` [S] is 1 on terminal states.

    There is deliberately no per-role initial belief. Every agent starts from
    ``initial_state_distribution``, and so does every level of a nested belief
    hierarchy: my prior estimate of your belief, and of your estimate of mine, are
    all the same distribution. (That is the law of total expectation -- averaging a
    posterior over its own prior predictive returns the prior -- so a per-role field
    could only ever hold a copy of this one.) Agents diverge the moment reset emits
    observations, not before; see tools.belief_representations.initial_belief.

    Information asymmetry belongs in the OBSERVATION function, not the prior: to
    give one agent privileged knowledge of the state, let it observe the state. An
    agent whose prior differs from ``initial_state_distribution`` is not better
    informed, it is simply wrong about the world.

    When stacked across game types (see factory.assemble_environments) every
    field gains a leading game-type axis.
    """

    transition: chex.Array
    observation: chex.Array
    reward: chex.Array
    num_actions: chex.Array
    num_states: chex.Array
    initial_state_distribution: chex.Array  # [S], the world prior AND every agent's prior
    terminal_mask: chex.Array  # [S], 1 on terminal states


@struct.dataclass
class FlexibleEnvState:
    state_index: chex.Array  # The DecPOMDP world state
    done: chex.Array


class FlexibleEnv:
    """Generic runtime over a single FlexibleEnvParams.

    reset/step_env/get_obs use positional (agent_0, agent_1) tuples; dynamics are
    read from ``self.params`` via the ``_joint_*`` gather API (the single reader),
    so rollouts and the DecPOMDP/belief consumers never diverge.
    """

    def __init__(self, params: FlexibleEnvParams) -> None:
        self.params = params
        self.num_states = int(params.num_states)
        self.num_actions = int(params.num_actions)
        self.num_observations = params.observation.shape[-1]
        # num agents = leading axis of the reward tensor
        self.num_agents = params.reward.shape[0]

    @partial(jax.jit, static_argnums=(0,))
    def _joint_transition_function(self, state, joint_action) -> distrax.Categorical:
        """T(s' | s, a0, a1) as a gather into params.transition -> [S']."""
        agent_0_action, agent_1_action = joint_action
        return distrax.Categorical(probs=self.params.transition[state, agent_0_action, agent_1_action])

    @partial(jax.jit, static_argnums=(0,))
    def _joint_observation_function(self, next_state, joint_action) -> distrax.Categorical:
        """O(o0, o1 | s', a0, a1) flattened to [O * O] (JointCategoricalPair order)."""
        agent_0_action, agent_1_action = joint_action
        probs = self.params.observation[next_state, agent_0_action, agent_1_action].reshape(-1)
        return distrax.Categorical(probs=probs)

    def _joint_reward_function(self, state, joint_action, next_state):
        """R_i(s, a0, a1, s') as a gather into params.reward -> per-agent tuple."""
        agent_0_action, agent_1_action = joint_action
        rewards = self.params.reward[:, state, agent_0_action, agent_1_action, next_state]
        return tuple(rewards[i] for i in range(self.num_agents))

    def _joint_action_constructor(self, agent_id, ego_action, other_action):
        """Order (ego, other) actions into the (agent_0, agent_1) joint action."""
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: FlexibleEnvState):
        """Sample both agents' observations from the joint O(o0, o1 | s', a).

        The joint is flattened row-major (index = o0 * O + o1), so we decode
        o0 = flat // O, o1 = flat % O. Observation is queried with a no-op action;
        envs whose observations depend on the action should override this.
        """
        joint_obs_dist = self._joint_observation_function(state.state_index, (0, 0))
        flat_obs = joint_obs_dist.sample(seed=key)
        return flat_obs // self.num_observations, flat_obs % self.num_observations

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """Sample the initial world state from params.initial_state_distribution."""
        state_key, obs_key = jax.random.split(key)
        state_index = distrax.Categorical(probs=self.params.initial_state_distribution).sample(seed=state_key)
        state = FlexibleEnvState(state_index=state_index, done=jnp.array(0))
        return state, self.get_obs(obs_key, state)

    def step_env(self, key: chex.PRNGKey, state: FlexibleEnvState, joint_action):
        """Sample a transition + reward from the dynamics; done from terminal_mask."""
        transition_key, obs_key = jax.random.split(key)
        s = state.state_index

        next_index = self._joint_transition_function(s, joint_action).sample(seed=transition_key)
        agent_rewards = self._joint_reward_function(s, joint_action, next_index)
        done = self.params.terminal_mask[next_index]

        next_state = FlexibleEnvState(state_index=next_index, done=done)
        return next_state, self.get_obs(obs_key, next_state), agent_rewards, done
