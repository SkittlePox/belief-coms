"""Guessing game: a stripped-down DecPOMDP defined as a FlexibleEnvParams spec.

This module holds the guessing game's *definition* — the eager tensor builders,
the per-role optimal policies, and ``guessing_game_spec`` (an EnvSpec). The
generic runtime lives in envs/flexible_env.py (FlexibleEnv); assembly across
game types lives in envs/env_assembly.py.

Q&A
---
Q: Is state 3 (the 4th state) a termination state?
A: Yes. It's absorbing, gives 0 reward, emits a dedicated "done" observation (the
   last observation symbol), and is never an initial state. States 0-2 are the
   referents; you reach state 3 only by pressing the matching button (a0 == s).

Q: Is the observation function deterministic?
A: In the referent states, no: each agent sees one of the two referent symbols != s,
   50/50, drawn independently per agent and per step (action-independent). In the
   done state it IS deterministic: both agents see the dedicated done symbol, so a
   belief update on that observation collapses onto the terminal state.

Q: Can the button-presser learn the true state just by waiting?
A: Yes. Each observed symbol k rules out state k, so after seeing both non-true
   symbols the belief collapses to the true state. P(identified after n waits) =
   1 - 0.5^(n-1) (~3 waits on average), at a cost of -0.1 per wait. So the presser
   can solo-solve the game; communication only speeds it up, it isn't necessary.
   Note: this is not a k-POMDP and there is no static randomness in this environment.
"""

import jax
import distrax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from envs.flexible_env import FlexibleEnvParams, OptimalPolicy


def build_transition_tensor(num_states, num_actions, done_state):
    """T(s' | s, a0, a1), shape [S, A, A, S].

    Pressing the button matching the state (a0 == s) sends you to the absorbing
    done state; any other action keeps you put. Agent 1's action is inert.
    """
    T = np.zeros((num_states, num_actions, num_actions, num_states))
    for s in range(num_states):
        for a0 in range(num_actions):
            for a1 in range(num_actions):
                if s == done_state:
                    next_state = done_state          # absorbing
                elif a0 == s:
                    next_state = done_state          # correct button -> done
                else:
                    next_state = s                   # stay
                T[s, a0, a1, next_state] = 1.0
    return jnp.asarray(T)


def build_observation_tensor(num_states, num_actions, num_observations, done_state):
    """O(o0, o1 | s', a0, a1), shape [S, A, A, O, O].

    Single shared observation alphabet: both agents draw from the same set of
    `num_observations` symbols. The LAST symbol (index num_observations - 1) is a
    dedicated "done" symbol emitted only in the terminal state; the remaining
    `num_observations - 1` symbols are the referent symbols. In a non-terminal
    (referent) state s each agent independently sees, uniformly, one of the referent
    symbols that are NOT s (so symbol k rules out state k, and the done symbol is
    never emitted). In the done state both agents deterministically see the done
    symbol, which signals to belief updating that a terminal state was entered. The
    joint is the outer product of the two identical per-agent marginals;
    action-independent here, so we broadcast over the action axes.
    """
    O = np.zeros((num_states, num_actions, num_actions, num_observations, num_observations))
    done_symbol = num_observations - 1
    num_referent_symbols = num_observations - 1   # every symbol except the done symbol
    for s in range(num_states):
        row = np.zeros(num_observations)
        if s == done_state:
            row[done_symbol] = 1.0                # deterministic "done" signal
        else:
            others = [k for k in range(num_referent_symbols) if k != s]
            row[others] = 1.0 / len(others)       # uniform over referent symbols != s
        O[s, :, :, :, :] = np.outer(row, row)     # independent identical marginals
    return jnp.asarray(O)


def build_reward_tensor(num_states, num_actions, num_agents, done_state):
    """R_i(s, a0, a1, s'), shape [N, S, A, A, S].

    Shared reward across agents. +1 for the button matching the state, -1 for a
    wrong button, -0.1 for the wait action; 0 in the done state. Independent of
    agent 1's action and of s'.
    """
    R = np.zeros((num_agents, num_states, num_actions, num_actions, num_states))
    for s in range(num_states):
        reward_array = np.array([-1.0, -1.0, -1.0, -0.1])
        if s < num_actions:
            reward_array[s] = 1.0
        for a0 in range(num_actions):
            value = 0.0 if s == done_state else reward_array[a0]
            R[:, s, a0, :, :] = value        # all agents, all a1, all s'
    return jnp.asarray(R)


# --------------------------------------------------------------------------- #
# Optimal policies (Categorical(belief over states) -> Categorical(over actions))
# --------------------------------------------------------------------------- #

def role_0_optimal_policy(belief: distrax.Categorical) -> distrax.Categorical:
    """Button presser: press the state you believe is the truth.

    State index == button index, so the optimal action distribution is just the
    belief over states.
    """
    return belief


def role_1_optimal_policy(belief: distrax.Categorical) -> distrax.Categorical:
    """Observer: action is inert, so any action distribution is optimal."""
    return belief


def guessing_game_spec(
    num_states=4, num_actions=4, num_observations=4, num_agents=2, done_state=3
) -> Tuple[FlexibleEnvParams, Tuple[OptimalPolicy, OptimalPolicy]]:
    """EnvSpec for the guessing game: (FlexibleEnvParams, per-role policies)."""
    # Uniform prior over the non-terminal states; same for the world state and
    # for both roles' initial beliefs.
    nonterminal = jnp.ones(num_states).at[done_state].set(0.0)
    nonterminal = nonterminal / nonterminal.sum()

    params = FlexibleEnvParams(
        transition=build_transition_tensor(num_states, num_actions, done_state),
        observation=build_observation_tensor(num_states, num_actions, num_observations, done_state),
        reward=build_reward_tensor(num_states, num_actions, num_agents, done_state),
        num_actions=jnp.array(num_actions),
        num_states=jnp.array(num_states),
        initial_belief_states=jnp.broadcast_to(nonterminal, (num_agents, num_states)),
        initial_state_distribution=nonterminal,
        terminal_mask=jnp.zeros(num_states).at[done_state].set(1.0),
    )
    policies = (role_0_optimal_policy, role_1_optimal_policy)
    return params, policies


if __name__ == "__main__":
    from envs.flexible_env import FlexibleEnv
    from tools.belief_representations import CategoricalBeliefState

    params, _ = guessing_game_spec()
    env = FlexibleEnv(params)
    key = jax.random.key(10)

    ### Basic environment loop with a random policy ###
    for episode in range(3):
        key, reset_key = jax.random.split(key)
        env_state, observations = env.reset(reset_key)
        print(f"\n=== Episode {episode} ===")
        print(f"reset state: state_index={env_state.state_index} (optimal button action)")
        print(f"initial obs (agent_0, agent_1): {observations}")

        episode_return = 0.0
        for t in range(5):
            key, action_key, step_key = jax.random.split(key, 3)
            agent_0_action = jax.random.randint(action_key, (), 0, env.num_actions)
            joint_action = (agent_0_action, jnp.array(0))

            env_state, next_obs, (r0, r1), done = env.step_env(step_key, env_state, joint_action)
            episode_return += float(r0)
            print(f"  t={t}: action={int(agent_0_action)}, reward=({float(r0):.2f}, {float(r1):.2f}), "
                  f"obs={next_obs}, done={bool(done)}")
            if bool(done):
                break
        print(f"episode return: {episode_return:.2f}")

    ### Demonstration: the presser learns the true state just by waiting ###
    # Observations are stochastic but state-distinctive (state s never emits symbol
    # s), so each observed symbol rules out one state. Once the presser has seen
    # both non-true symbols, its belief collapses to certainty -- no communication
    # needed. Here the presser repeatedly waits, observes, and updates its belief.
    belief_factory = CategoricalBeliefState(params)
    true_state = 0
    wait = env.num_actions - 1                       # the "wait" action
    wait_joint_action = (wait, wait)                 # presser waits; other agent inert
    presser_obs_dist = params.observation[true_state, 0, 0].sum(axis=1)  # O(o | true_state)

    print(f"\n=== Learning by waiting (presser, true state = {true_state}) ===")
    belief = distrax.Categorical(probs=params.initial_belief_states[0])
    print(f"  start : belief = {belief.probs}")
    key = jax.random.key(0)
    for t in range(12):
        key, obs_key = jax.random.split(key)
        observation = distrax.Categorical(probs=presser_obs_dist).sample(seed=obs_key)
        belief = belief_factory.update_with_observation_and_joint_action(
            belief, observation, wait_joint_action, agent_id=0
        )
        identified = bool(belief.probs[true_state] > 0.999)
        note = "  <- identified the true state!" if identified else ""
        print(f"  t={t:<2}: saw symbol {int(observation)} -> belief = {belief.probs}{note}")
        if identified:
            break
