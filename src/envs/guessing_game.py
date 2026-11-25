"""Guessing game: a stripped-down DecPOMDP defined as a FlexibleEnvParams spec.

This module holds the guessing game's *definition* — the eager tensor builders,
the per-role Q functions, and ``guessing_game_spec`` (an EnvSpec). The
generic runtime lives in envs/flexible_env.py (FlexibleEnv); assembly across
game types lives in envs/env_assembly.py.

Q&A
---
Q: Is state 3 (the 4th state) a termination state?
A: Yes. It's absorbing, gives 0 reward, emits a dedicated "done" observation (the
   last observation symbol), and is never an initial state. States 0-2 are the
   referents; you reach state 3 only by pressing the matching button (a0 == s).

Q: Is the observation function deterministic?
A: In the referent states, no: the two agents see the two referent symbols != s,
   one each, with the assignment 50/50 (so their symbols always differ, and together
   they pin down s). Draws are independent across steps and action-independent. In the
   done state it IS deterministic: both agents see the dedicated done symbol, so a
   belief update on that observation collapses onto the terminal state.

Q: Can the button-presser learn the true state just by waiting?
A: Yes. Each observed symbol k rules out state k, so after seeing both non-true
   symbols the belief collapses to the true state. P(identified after n waits) =
   1 - 0.5^(n-1) (~3 waits on average), at a cost of -0.1 per wait. So the presser
   can solo-solve the game; communication only speeds it up, it isn't necessary.
   (With communication it's solvable in one step: the observer's symbol is always the
   one the presser didn't see.)
   Note: this is not a k-POMDP and there is no static randomness in this environment.
"""

import jax
import distrax
import jax.numpy as jnp
import numpy as np

from envs.flexible_env import FlexibleEnvParams, GameSpec

# Rewards (shared by both agents). build_reward_tensor and presser_q_function both read
# these, so the hand-derived Q stays in step with the tensor.
CORRECT_PRESS_REWARD = 1.0  # pressing the button matching the state (then the game ends)
WRONG_PRESS_REWARD = -1.0  # pressing any other referent button (the state stays put)
WAIT_REWARD = -0.1  # the wait action (the last action)

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
                    next_state = done_state  # absorbing
                elif a0 == s:
                    next_state = done_state  # correct button -> done
                else:
                    next_state = s  # stay
                T[s, a0, a1, next_state] = 1.0
    return jnp.asarray(T)


def build_observation_tensor(num_states, num_actions, num_observations, done_state):
    """O(o0, o1 | s', a0, a1), shape [S, A, A, O, O].

    Single shared observation alphabet: both agents draw from the same set of
    `num_observations` symbols. The LAST symbol (index num_observations - 1) is a
    dedicated "done" symbol emitted only in the terminal state; the remaining
    `num_observations - 1` symbols are the referent symbols. In a non-terminal
    (referent) state s the two agents see two DIFFERENT referent symbols, both != s,
    with the ordered pair (o0, o1) drawn uniformly (so symbol k rules out state k, the
    done symbol is never emitted, and the two agents' symbols together identify s when
    there are 3 referents). The observations are therefore correlated: each agent's
    marginal is uniform over the referent symbols != s, but the joint is not their
    outer product. In the done state both agents deterministically see the done
    symbol, which signals to belief updating that a terminal state was entered.
    Action-independent here, so we broadcast over the action axes.
    """
    O = np.zeros((num_states, num_actions, num_actions, num_observations, num_observations))
    done_symbol = num_observations - 1
    num_referent_symbols = num_observations - 1  # every symbol except the done symbol
    for s in range(num_states):
        joint = np.zeros((num_observations, num_observations))
        if s == done_state:
            joint[done_symbol, done_symbol] = 1.0  # deterministic "done" signal
        else:
            others = [k for k in range(num_referent_symbols) if k != s]
            pairs = [(i, j) for i in others for j in others if i != j]
            for i, j in pairs:
                joint[i, j] = 1.0 / len(pairs)  # uniform over distinct ordered pairs != s
        O[s, :, :, :, :] = joint
    return jnp.asarray(O)


def build_reward_tensor(num_states, num_actions, num_agents, done_state):
    """R_i(s, a0, a1, s'), shape [N, S, A, A, S].

    Shared reward across agents. +1 for the button matching the state, -1 for a
    wrong button, -0.1 for the wait action; 0 in the done state. Independent of
    agent 1's action and of s'.
    """
    R = np.zeros((num_agents, num_states, num_actions, num_actions, num_states))
    for s in range(num_states):
        reward_array = np.array([WRONG_PRESS_REWARD, WRONG_PRESS_REWARD, WRONG_PRESS_REWARD, WAIT_REWARD])
        if s < num_actions:
            reward_array[s] = CORRECT_PRESS_REWARD
        for a0 in range(num_actions):
            value = 0.0 if s == done_state else reward_array[a0]
            R[:, s, a0, :, :] = value  # all agents, all a1, all s'
    return jnp.asarray(R)


# --------------------------------------------------------------------------- #
# Q functions: (b_e, b̄_o) -> [A] action values, turned into Boltzmann policies
# --------------------------------------------------------------------------- #

# The Q functions below are derived for the default layout: referents 0, 1, 2; done
# state 3; buttons 0, 1, 2; wait = action 3. guessing_game_spec refuses other sizes.
NUM_REFERENTS = 3
# For each button k, the two referents that remain possible if k turns out wrong.
_OTHER_REFERENTS = jnp.array([[1, 2], [0, 2], [0, 1]])


def _nan_tripwire(ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical):
    """0, or NaN if either belief is NaN.

    Added to Q values that do not read every belief, so a NaN belief still yields a NaN
    policy instead of a healthy-looking one.
    """
    return 0.0 * (ego_belief.probs.sum() + other_belief_estimate.probs.sum())


def presser_q_function(discount_factor: float):
    """The presser's (role 0) exact Q function, for the infinite-horizon discounted game.

    Only b_e matters. The observer's action is inert in T, O and R, so nothing the
    observer believes or does can change the presser's return inside the DecPOMDP; b̄_o
    would matter only through communication, which is outside it. b̄_o is accepted for
    the shared Q signature and otherwise ignored.

    Why a closed form exists
    ------------------------
    Observation likelihoods are flat: in referent s the presser sees one of the two
    referent symbols != s, each with probability 1/2. So an observation only RULES OUT
    a referent and never reweights the survivors. The same holds after a wrong press of
    k: the game would have ended had k been right, so k is ruled out too. The belief
    therefore only ever moves between the supports {3 referents} -> {2} -> {1}, keeping
    the original relative weights, and the value is solved one support size at a time.

    Values are written for UNNORMALIZED referent weights (each is degree-1 homogeneous),
    which keeps every formula division-free. Mass on the done state contributes 0 to every
    action and drops out. With C, W, w the correct/wrong/wait rewards and γ the discount:

      Certain of referent i:  press i, worth C (waiting or a wrong press only delays it).

      Two referents left, weights (a, b):  each wait resolves the state with probability
      1/2 (the presser sees the other survivor's symbol) and otherwise changes nothing,
      so the best plan is "press the likelier now" or "wait until resolved, then press":

          Ṽ₂(a, b)    = max( max(a,b)·C + min(a,b)·(W + γC),   (a + b)·V_wait )
          V_wait      = (w + γC/2) / (1 − γ/2)        (fixed point of waiting)

      Three-referent belief p (the general case; it reduces to the two cases above when
      weights are zero):

          Q(p, press k) = p_k·C + (p_i + p_j)·(W + γC/2) + (γ/2)·Ṽ₂(p_i, p_j)
          Q(p, wait)    = (p_0 + p_1 + p_2)·w + (γ/2)·∑_o Ṽ₂(p restricted to referents ≠ o)

      where {i, j} are the referents other than k. After a wrong press of k, the next
      symbol is k (no news, 1/2) or the other wrong referent (resolved, 1/2). A wait
      rules out exactly one referent, symbol o arriving with weight (p_i + p_j)/2.

    This is the optimal Q, i.e. it assumes greedy play afterwards. The Boltzmann policy
    built on it is softer than that, and the episode horizon is ignored (infinite
    horizon); both are deliberate approximations.

    Returns:
        q_function(ego_belief, other_belief_estimate) -> [4] values, for buttons 0-2
        then wait.
    """
    gamma = discount_factor
    wait_until_resolved = (WAIT_REWARD + gamma / 2 * CORRECT_PRESS_REWARD) / (1 - gamma / 2)

    def two_referent_value(a, b):
        press_likelier = jnp.maximum(a, b) * CORRECT_PRESS_REWARD + jnp.minimum(a, b) * (WRONG_PRESS_REWARD + gamma * CORRECT_PRESS_REWARD)
        return jnp.maximum(press_likelier, (a + b) * wait_until_resolved)

    def q_function(ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical) -> jnp.ndarray:
        p = ego_belief.probs[:NUM_REFERENTS]  # referent weights; done mass drops out
        p_i, p_j = p[_OTHER_REFERENTS[:, 0]], p[_OTHER_REFERENTS[:, 1]]  # [3]: survivors if button k is wrong
        # Ṽ₂ of the two referents other than k. Indexed by k it serves the presses; indexed
        # by the ruled-out symbol o it serves the wait.
        survivors_value = two_referent_value(p_i, p_j)  # [3]

        press = p * CORRECT_PRESS_REWARD + (p_i + p_j) * (WRONG_PRESS_REWARD + gamma / 2 * CORRECT_PRESS_REWARD) + gamma / 2 * survivors_value
        wait = p.sum() * WAIT_REWARD + gamma / 2 * survivors_value.sum()
        return jnp.append(press, wait) + _nan_tripwire(ego_belief, other_belief_estimate)

    return q_function


def observer_q_function(ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical) -> jnp.ndarray:
    """The observer's (role 1) Q function: the same value for every action.

    The observer's action is inert in T, O and R, so Q_1(b_e, b̄_o, a) does not depend on
    a. Its level -- the team's value as seen from the observer's seat, which does depend
    on b_e and b̄_o -- cancels out of a Boltzmann policy, so zeros are returned instead:
    the policy is uniform at any β.
    """
    return jnp.zeros(4) + _nan_tripwire(ego_belief, other_belief_estimate)


def guessing_game_spec(
    num_states=4,
    num_actions=4,
    num_observations=4,
    num_agents=2,
    done_state=3,
    discount_factor=0.9,
    inverse_temperature=10.0,
) -> GameSpec:
    """EnvSpec for the guessing game: (FlexibleEnvParams, per-role Q functions, β).

    ``discount_factor`` is the γ the presser's Q is solved under; ``inverse_temperature``
    is the β of both roles' Boltzmann policies.
    """
    if (num_states, num_actions, num_observations, done_state) != (4, 4, 4, 3):
        raise NotImplementedError("the guessing game's Q functions are derived for 3 referents, a done state 3 and wait = action 3")

    # Uniform prior over the non-terminal states. This is both the world's initial
    # state distribution and every agent's initial belief -- there is no separate
    # per-role prior; see FlexibleEnvParams.
    nonterminal = jnp.ones(num_states).at[done_state].set(0.0)
    nonterminal = nonterminal / nonterminal.sum()

    params = FlexibleEnvParams(
        transition=build_transition_tensor(num_states, num_actions, done_state),
        observation=build_observation_tensor(num_states, num_actions, num_observations, done_state),
        reward=build_reward_tensor(num_states, num_actions, num_agents, done_state),
        num_actions=jnp.array(num_actions),
        num_states=jnp.array(num_states),
        initial_state_distribution=nonterminal,
        terminal_mask=jnp.zeros(num_states).at[done_state].set(1.0),
    )
    q_functions = (presser_q_function(discount_factor), observer_q_function)
    return GameSpec(params=params, q_functions=q_functions, inverse_temperature=inverse_temperature)

if __name__ == "__main__":
    from envs.flexible_env import FlexibleEnv
    from tools.belief_representations import update_with_observation_and_joint_action

    params = guessing_game_spec().params
    env = FlexibleEnv(params)
    key = jax.random.key(10)

    ### Joint observation likelihoods O(o0, o1 | s') per state ###
    O = np.asarray(params.observation)
    num_obs = O.shape[-1]
    print(f"observation tensor shape [s', a0, a1, o0, o1]: {O.shape}")
    print(f"action-independent: {bool(np.allclose(O, O[:, :1, :1]))}")
    for s in range(O.shape[0]):
        joint = O[s, 0, 0]  # action-independent, so any (a0, a1) slice is representative
        print(f"\ns'={s}{' (done)' if params.terminal_mask[s] else ''}   rows = o0 (agent 0), cols = o1 (agent 1)")
        print("         " + "".join(f"o1={j:<5}" for j in range(num_obs)))
        for i in range(num_obs):
            print(f"  o0={i}   " + "".join(f"{joint[i, j]:<8.3g}" for j in range(num_obs)))

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
            print(f"  t={t}: action={int(agent_0_action)}, reward=({float(r0):.2f}, {float(r1):.2f}), " f"obs={next_obs}, done={bool(done)}")
            if bool(done):
                break
        print(f"episode return: {episode_return:.2f}")

    ### Demonstration: the presser learns the true state just by waiting ###
    # Observations are stochastic but state-distinctive (state s never emits symbol
    # s), so each observed symbol rules out one state. Once the presser has seen
    # both non-true symbols, its belief collapses to certainty -- no communication
    # needed. Here the presser repeatedly waits, observes, and updates its belief.
    true_state = 0
    wait = env.num_actions - 1  # the "wait" action
    wait_joint_action = (wait, wait)  # presser waits; other agent inert
    presser_obs_dist = params.observation[true_state, 0, 0].sum(axis=1)  # O(o | true_state)

    print(f"\n=== Learning by waiting (presser, true state = {true_state}) ===")
    belief = distrax.Categorical(probs=params.initial_state_distribution)
    print(f"  start : belief = {belief.probs}")
    key = jax.random.key(0)
    for t in range(12):
        key, obs_key = jax.random.split(key)
        observation = distrax.Categorical(probs=presser_obs_dist).sample(seed=obs_key)
        belief = update_with_observation_and_joint_action(params, belief, observation, wait_joint_action, ego_role=0)
        identified = bool(belief.probs[true_state] > 0.999)
        note = "  <- identified the true state!" if identified else ""
        print(f"  t={t:<2}: saw symbol {int(observation)} -> belief = {belief.probs}{note}")
        if identified:
            break
