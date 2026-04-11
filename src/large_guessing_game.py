import jax, chex
import jax.numpy as jnp
import distrax
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from belief_representations import *
from distributions import *
from model import *


@struct.dataclass
class State:
    sender_agent: chex.Array

    agent_0_world_observation: chex.Array
    agent_1_world_observation: chex.Array

    optimal_receiver_action: chex.Array

    done: chex.Array


class LargeGuessingGame(MultiAgentEnv):
    """
    A generalized guessing game with a configurable number of referents N (default 5).

    In each episode, one referent is secretly designated as the target (optimal action).
    Each agent observes a single distinct non-target referent; their two observations
    are guaranteed to differ from the target and from each other.  When N > 3 there are
    additional unseen distractor referents that neither agent observed.

    The sender communicates to help the receiver guess the target.  With N > 3, the
    target cannot be determined from either agent's observation alone -- both agents'
    combined knowledge still leaves (N-3) unseen distractors as candidates alongside
    the true target, so effective communication is necessary.

    This environment is a drop-in replacement for GuessingGame and is compatible with
    SignificationPOMDP.  The State struct is identical; only N changes.

    Parameters
    ----------
    num_referents:
        Number of distinct referents.  Must be >= 3.  Defaults to 5.
    """

    def __init__(self, num_referents: int = 5) -> None:
        assert num_referents >= 3, "num_referents must be >= 3"
        super().__init__(num_agents=2)
        self.num_referents = num_referents

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: State):
        return state.agent_0_world_observation, state.agent_1_world_observation

    def step_env(self, key: chex.PRNGKey, state: State, joint_action: chex.Array):
        agent_0_action, agent_1_action = joint_action

        receiver_action = jax.lax.cond(
            state.sender_agent == 0,
            lambda _: agent_1_action,
            lambda _: agent_0_action,
            None,
        )
        agent_reward = jax.lax.cond(
            receiver_action == state.optimal_receiver_action,
            lambda _: 1.0,
            lambda _: -1.0,
            None,
        )

        next_environment_state = State(
            sender_agent=-1,
            agent_0_world_observation=-1,
            agent_1_world_observation=-1,
            optimal_receiver_action=state.optimal_receiver_action,
            done=jnp.array(1),
        )

        return (
            next_environment_state,
            self.get_obs(key, next_environment_state),
            (agent_reward, agent_reward),
            next_environment_state.done,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        sender_agent=jnp.array(-1),
        agent_world_observations_and_optimal_action=jnp.array([-1, -1, -1]),
    ):
        N = self.num_referents

        # Agent role selection
        sender_agent_key, key = jax.random.split(key)
        sender_agent = jax.lax.cond(
            sender_agent == jnp.array(-1),
            lambda _: jax.random.randint(sender_agent_key, (1), 0, 2)[0],
            lambda _: sender_agent,
            None,
        )

        # Permute all N referents.
        #   perm[0]   -> agent 0's observation
        #   perm[1]   -> agent 1's observation
        #   perm[N-1] -> target / optimal action
        #   perm[2:N-1] -> unseen distractors (for N > 3)
        # All three selected values are distinct by construction.
        observation_key, key = jax.random.split(key)

        def sample_observations(_):
            perm = jax.random.permutation(observation_key, jnp.arange(N))
            return jnp.array([perm[0], perm[1], perm[N - 1]])

        agent_world_observations_and_optimal_action = jax.lax.cond(
            jnp.array_equal(
                agent_world_observations_and_optimal_action, jnp.array([-1, -1, -1])
            ),
            sample_observations,
            lambda _: agent_world_observations_and_optimal_action,
            None,
        )

        initial_environment_state = State(
            sender_agent=sender_agent,
            agent_0_world_observation=agent_world_observations_and_optimal_action[0],
            agent_1_world_observation=agent_world_observations_and_optimal_action[1],
            optimal_receiver_action=agent_world_observations_and_optimal_action[2],
            done=jnp.array(0),
        )

        return (initial_environment_state, self.get_obs(key, initial_environment_state))

    @partial(jax.jit, static_argnums=(0,))
    def _joint_transition_function(self, state_num, joint_action):
        """N possible states, one per referent.  The state is absorbing."""
        N = self.num_referents
        probs = jnp.zeros(N).at[state_num].set(1.0)
        return distrax.Categorical(probs=probs)

    @partial(jax.jit, static_argnums=(0,))
    def _joint_observation_function(self, state_num, joint_action):
        """
        Joint observation function over N x N pairs (agent_0_obs, agent_1_obs).

        In state s (target = s), the valid joint observations (o0, o1) satisfy:
            o0 != s,  o1 != s,  o0 != o1
        There are (N-1)*(N-2) such pairs, each equally likely.

        The probability array is flattened in row-major order:
            flat_index = o0 * N + o1
        """
        N = self.num_referents
        a = jnp.arange(N)
        # Build N x N boolean validity mask
        valid = (
            (a[:, None] != state_num)
            & (a[None, :] != state_num)
            & (a[:, None] != a[None, :])
        )
        num_valid = (N - 1) * (N - 2)
        probs = (valid.astype(jnp.float32) / num_valid).flatten()
        return distrax.Categorical(probs=probs)

    def _joint_action_constructor(self, agent_id, ego_action, other_action):
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None,
        )

    def _joint_reward_function(self, state, joint_action, next_state):
        return joint_action[0] == state

    def _optimal_policy(self, belief_distribution: distrax.Categorical):
        """The optimal action distribution equals the belief distribution."""
        return belief_distribution

    def _agent_0_optimal_policy(self, belief_distribution: distrax.Categorical):
        return belief_distribution

    def _agent_1_optimal_policy(self, belief_distribution: distrax.Categorical):
        return belief_distribution


if __name__ == "__main__":
    N = 5
    env = LargeGuessingGame(num_referents=N)
    key = jax.random.key(10)
    env_state, observations = env.reset(key)
    print("=== Initial state ===")
    print(env_state)

    ### Transition function tests
    print("\n--- Transition function ---")
    print("State 0:", env._joint_transition_function(0, (-1, 0)).probs)
    print("State 3:", env._joint_transition_function(3, (-1, 1)).probs)

    ### Observation function tests
    print("\n--- Observation function (joint sample per state) ---")
    factory = JointCategoricalPair(vars_num_categories=(N, N))
    for s in range(N):
        sample = factory.sample_joint_distribution(
            key, env._joint_observation_function(s, 0)
        )
        print(f"  State {s}: {sample}")

    ### Belief update test
    print("\n--- Belief update ---")
    initial_belief = distrax.Categorical(probs=jnp.ones(N) / N)

    # num_unique_actions = N (valid guesses 0..N-1) + 1 (null action)
    belief_factory = CategoricalBeliefState(
        num_unique_states=N,
        num_unique_observations=N,
        num_unique_actions=N + 1,
        joint_transition_function=env._joint_transition_function,
        joint_observation_function=env._joint_observation_function,
        joint_action_constructor=env._joint_action_constructor,
    )

    new_belief = belief_factory.update_with_observation_and_joint_action(
        initial_belief, 1, (-1, N), 0
    )
    print("Updated belief after obs=1:", new_belief.probs)

    their_belief = belief_factory.update_other_belief_estimate_with_observation_only(
        initial_belief, 1, N, env._optimal_policy
    )
    print("Agent 0's estimate of agent 1's belief:", their_belief.probs)

    ### Expected returns
    model = DecPOMDPModel(
        env._joint_transition_function,
        env._joint_reward_function,
        env._joint_observation_function,
        env._joint_action_constructor,
        N,
        N,
        N + 1,
    )

    cum_ret = model.evaluate_expected_returns(
        0,
        env._optimal_policy,
        env._optimal_policy,
        new_belief,
        initial_belief,
        belief_factory,
    )
    print("Expected returns:", cum_ret)
