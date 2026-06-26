from mimetypes import init
import jax, chex
import copy
import distrax
import jax.numpy as jnp
import numpy as np
from flax import struct
from functools import partial
from tools.belief_representations import CategoricalBeliefState
from tools.distributions import JointCategoricalPair
from tools.model import DecPOMDPModel


@struct.dataclass
class GuessingGameState:
    state_index: chex.Array   # The DecPOMDP state: 0..2 = optimal button action, done_state = terminal
    done: chex.Array


@struct.dataclass
class EnvParams:
    """Dense dynamics tensors for a (variant of a) DecPOMDP.

    Everything that distinguishes one environment variant from another lives
    here as plain arrays, so different variants are just different EnvParams
    instances *with identical shapes* — which is what keeps a future
    vmap-over-environments clean (no per-env Python branching / lax.switch).

    The conditioning below is the most general the current consumers
    (tools/model.py, tools/belief_representations.py) can actually feed, given
    what each call site has in scope:

        transition   T(s' | s, a0, a1)         shape [S, A, A, S]
        observation  O(o0, o1 | s', a0, a1)    shape [S, A, A, O0, O1]
        reward       R_i(s, a0, a1, s')        shape [N, S, A, A, S]

    Notes on the deliberate choices:
      * Observation conditions on the *next* state s' (its first axis) and on
        the joint action — never on the previous state.
      * Both agents get their own action axis (a0, a1) even when one agent's
        action is currently inert, so the shapes stay uniform across variants.
      * Reward carries a leading agent axis (N) so per-agent (non-shared)
        reward structures are expressible later.
    """

    transition: chex.Array
    observation: chex.Array
    reward: chex.Array


def build_transition_tensor(num_states, num_actions, done_state):
    """T(s' | s, a0, a1), shape [S, A, A, S].

    Guessing game: pressing the button matching the state (a0 == s) sends you to
    the absorbing done state; any other action keeps you put. Agent 1's action
    is inert. The done state is absorbing.
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
    `num_observations` symbols. Each agent independently sees, uniformly, one of
    the two symbols that are NOT the true state, so the joint is the outer product
    of the two identical per-agent marginals. Action-independent here, so we
    broadcast over the action axes. In the done state both marginals are uniform.
    """
    O = np.zeros((num_states, num_actions, num_actions, num_observations, num_observations))
    base = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    for s in range(num_states):
        row = np.ones(num_observations) / num_observations if s == done_state else base[s]
        O[s, :, :, :, :] = np.outer(row, row)   # independent identical marginals
    return jnp.asarray(O)


def build_reward_tensor(num_states, num_actions, num_agents, done_state):
    """R_i(s, a0, a1, s'), shape [N, S, A, A, S].

    Guessing game: shared reward across agents. +1 for the button matching the
    state, -1 for a wrong button, -0.1 for the wait action; 0 in the done state.
    Independent of agent 1's action and of s' here.
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


class GuessingGame:
    """
    This is a stripped-down environment for testing purposes.

    Deliberately a plain class — it does not implement the JaxMARL
    MultiAgentEnv interface (dict-keyed, agent-named observations/actions and
    the auto-resetting `step`). reset/step_env/get_obs use simple positional
    (agent_0, agent_1) tuples, and dynamics are read from `self.params`.
    """

    def __init__(
        self,
        num_states=4,
        num_actions=4,
        num_observations=3,
        num_agents=2,
        done_state=3,
    ) -> None:
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        # A single observation alphabet shared by all agents.
        self.num_observations = num_observations
        self.done_state = done_state

        # Dense dynamics tensors. Built eagerly (plain numpy loops) so the
        # high-rank tensors stay readable; consumed at runtime as simple gathers.
        self.params = EnvParams(
            transition=build_transition_tensor(num_states, num_actions, done_state),
            observation=build_observation_tensor(
                num_states,
                num_actions,
                num_observations,
                done_state,
            ),
            reward=build_reward_tensor(num_states, num_actions, num_agents, done_state),
        )

        self.initial_belief_agent_0 = distrax.Categorical(
            probs=jnp.ones(4).at[3].set(0.0)
        )
        self.initial_belief_agent_1 = distrax.Categorical(
            probs=jnp.ones(4).at[3].set(0.0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: GuessingGameState):
        """Both agents' observations are sampled jointly from O(o0, o1 | s', a).

        The joint is flattened row-major (index = o0 * O + o1), so we decode
        o0 = flat // O and o1 = flat % O. Observation is action-independent in
        this env, so we query O with a no-op action.
        """
        joint_obs_dist = self._joint_observation_function(state.state_index, (0, 0))
        flat_obs = joint_obs_dist.sample(seed=key)
        agent_0_obs = flat_obs // self.num_observations
        agent_1_obs = flat_obs % self.num_observations
        return agent_0_obs, agent_1_obs

    def step_env(
        self, key: chex.PRNGKey, state: GuessingGameState, joint_action: chex.Array
    ):
        """
        Actions are sent in the order (agent_0, agent_1). Agent 0 is the button
        presser; agent 1's action is currently inert. The transition and reward
        are sampled/read from the _joint_* dynamics API (the single reader of
        self.params), so rollouts and the DecPOMDP model never diverge.
        """
        transition_key, obs_key = jax.random.split(key)
        s = state.state_index

        next_index = self._joint_transition_function(s, joint_action).sample(
            seed=transition_key
        )
        agent_rewards = self._joint_reward_function(s, joint_action, next_index)

        next_environment_state = GuessingGameState(
            state_index=next_index,
            done=jnp.array(next_index == self.done_state),
        )

        return (
            next_environment_state,
            self.get_obs(obs_key, next_environment_state),
            agent_rewards,
            next_environment_state.done,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """
        Agent 0 is always the button presser. The state index (= optimal button
        action) is drawn uniformly from the non-terminal states.
        """
        state_key, obs_key = jax.random.split(key)
        state_index = jax.random.randint(state_key, (), 0, self.done_state)

        initial_environment_state = GuessingGameState(
            state_index=state_index, done=jnp.array(0)
        )

        return (
            initial_environment_state,
            self.get_obs(obs_key, initial_environment_state),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _joint_transition_function(self, state_num, joint_action):
        """T(s' | s, a0, a1) as a gather into params.transition -> [S']."""
        agent_0_action, agent_1_action = joint_action
        probs = self.params.transition[state_num, agent_0_action, agent_1_action]
        return distrax.Categorical(probs=probs)

    @partial(jax.jit, static_argnums=(0,))
    def _joint_observation_function(self, next_state, joint_action):
        """O(o0, o1 | s', a0, a1) as a gather into params.observation.

        Returns the joint distribution flattened to [O * O], matching what
        JointCategoricalPair(vars_num_categories=(O, O)) expects.
        """
        agent_0_action, agent_1_action = joint_action
        probs = self.params.observation[
            next_state, agent_0_action, agent_1_action
        ].reshape(-1)
        return distrax.Categorical(probs=probs)

    def _joint_action_constructor(self, agent_id, ego_action, other_action):
        """
        This will likely be the default for all environments
        """
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None,
        )

    def _joint_reward_function(self, state, joint_action, next_state):
        """R_i(s, a0, a1, s') as a gather into params.reward -> per-agent tuple."""
        agent_0_action, agent_1_action = joint_action
        rewards = self.params.reward[
            :, state, agent_0_action, agent_1_action, next_state
        ]
        return (rewards[0], rewards[1])

    def _agent_0_optimal_policy(self, belief_distribution: distrax.Categorical):
        """
        Returns a probability distribution over possible actions in the DecPOMDP.
        belief_distribution is a categorical distrax distribution over all possible states.
        """
        return belief_distribution

    def _agent_1_optimal_policy(self, belief_distribution: distrax.Categorical):
        """
        Returns a probability distribution over possible actions in the DecPOMDP.
        belief_distribution is a categorical distrax distribution over all possible states.
        """
        return distrax.Categorical(probs=[1.0])

    def _is_terminal_state(self, state):
        return state == 4


if __name__ == "__main__":
    env = GuessingGame()
    key = jax.random.key(10)
    env_state, observations = env.reset(key)
    print(env_state)

    ### Basic environment loop with a random policy ###
    # Agent 0 is the button presser (4 actions: 0-2 buttons, 3 wait). Agent 1's action is unused.
    num_episodes = 3
    max_steps = 5
    num_agent_0_actions = 4

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        env_state, observations = env.reset(reset_key)
        print(f"\n=== Episode {episode} ===")
        print(
            f"reset state: state_index={env_state.state_index} "
            f"(optimal button action)"
        )
        print(f"initial obs (agent_0, agent_1): {observations}")

        episode_return = 0.0
        for t in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)

            # Random policy: agent 0 picks a random button/wait action, agent 1's action is unused.
            agent_0_action = jax.random.randint(action_key, (), 0, num_agent_0_actions)
            agent_1_action = jnp.array(-1)
            joint_action = (agent_0_action, agent_1_action)

            next_state, next_obs, (r0, r1), done = env.step_env(
                step_key, env_state, joint_action
            )
            episode_return += float(r0)

            print(
                f"  t={t}: action={int(agent_0_action)}, "
                f"reward=({float(r0):.2f}, {float(r1):.2f}), "
                f"obs={next_obs}, done={bool(done)}"
            )

            env_state = next_state
            if bool(done):
                break

        print(f"episode return: {episode_return:.2f}")

    ### Transition function tests

    # In states 0, 1, 2 you either stay in your current state or get to the end state
    print(env._joint_transition_function(0, (3, -1)).probs)
    print(env._joint_transition_function(0, (0, -1)).probs)

    ### Observation function tests
    factory = JointCategoricalPair(vars_num_categories=(3, 3))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(0, (0, 0))))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(1, (0, 0))))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(2, (0, 0))))

    ### Belief update test
    initial_belief = env.initial_belief_agent_1
    print(initial_belief.probs)

    belief_factory = CategoricalBeliefState(
        num_unique_states=4,
        num_unique_observations=3,
        num_unique_actions=1,
        joint_transition_function=env._joint_transition_function,
        joint_observation_function=env._joint_observation_function,
        joint_action_constructor=env._joint_action_constructor,
    )

    new_belief = belief_factory.update_with_observation_and_joint_action(
        belief_distribution=initial_belief,
        observation=0,
        previous_joint_action=(2, 0),
        agent_id=1,
    )
    print(new_belief.probs)

    uniform_belief = distrax.Categorical(probs=jnp.ones(4))

    their_beliefs = env.initial_belief_agent_0

    print(new_belief.probs)
    their_beliefs = belief_factory.update_other_belief_estimate_with_observation_only(
        other_belief_distribution_estimate=their_beliefs,
        ego_observation=0,
        previous_ego_action=3,
        other_optimal_policy=env._agent_1_optimal_policy,
        agent_id=0,
    )  # Prob for index 0 should be highest actually, because they cannot see 0
    print(their_beliefs.probs)

    model = DecPOMDPModel(
        joint_transition_function=env._joint_transition_function,
        joint_reward_function=env._joint_reward_function,
        joint_observation_function=env._joint_observation_function,
        joint_action_constructor=env._joint_action_constructor,
        num_unique_states=3,
        num_unique_observations=3,
        num_unique_actions=4,
    )

    cum_ret = model.evaluate_expected_returns(
        0,
        env._agent_0_optimal_policy,
        env._agent_1_optimal_policy,
        new_belief,
        initial_belief,
        belief_factory,
    )
    print(cum_ret)
