from mimetypes import init
import jax, chex
import copy
import distrax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from tools.belief_representations import CategoricalBeliefState
from tools.distributions import JointCategoricalPair
from tools.model import DecPOMDPModel


@struct.dataclass
class GuessingGameState:
    hidden_variable: (
        chex.Array
    )  # This is an integer corresponding to the variable the 'sender' sees
    optimal_button_action: (
        chex.Array
    )  # This is an integer corresponding to the optimal button action
    done: chex.Array


class GuessingGame(MultiAgentEnv):
    """
    This is a stripped-down environment for testing purposes
    """

    def __init__(self) -> None:
        super().__init__(num_agents=2)
        self.initial_belief_agent_0 = distrax.Categorical(probs=jnp.ones(4).at[3].set(0.0))
        self.initial_belief_agent_1 = distrax.Categorical(probs=jnp.ones(4).at[3].set(0.0))

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: GuessingGameState):
        # There are only three possible discrete observations
        return jnp.array(-1), state.hidden_variable

    def step_env(
        self, key: chex.PRNGKey, state: GuessingGameState, joint_action: chex.Array
    ):
        """
        Actions are sent in the order (agent_0, agent_1) and are shape (1) and (1)
        Agent 0 is always the button presser. Agent 1 does not have any actions, or rather its actions are unused.
        Assuming 3 buttons, and a wait action, so 4 actions
        """
        agent_0_action, _agent_1_action = joint_action

        reward_array = (
            jnp.array([-1.0, -1.0, -1.0, -0.1]).at[state.optimal_button_action].set(1.0)
        )
        agent_reward = reward_array[agent_0_action]

        is_done = agent_0_action == state.optimal_button_action

        next_environment_state = GuessingGameState(
            hidden_variable=state.hidden_variable,
            optimal_button_action=state.optimal_button_action,
            done=jnp.array(is_done),
        )

        return (
            next_environment_state,
            self.get_obs(key, next_environment_state),
            (agent_reward, agent_reward),
            next_environment_state.done,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """
        Agent 0 is always the button presser. Agent 1 always sees the hidden variable.
        """

        optimal_button_key, key = jax.random.split(key)
        perm = jax.random.permutation(optimal_button_key, jnp.arange(3))

        initial_environment_state = GuessingGameState(
            hidden_variable=perm[0], optimal_button_action=perm[1], done=jnp.array(0)
        )

        return (initial_environment_state, self.get_obs(key, initial_environment_state))

    @partial(jax.jit, static_argnums=(0,))
    def _joint_transition_function(self, state_num, joint_action):
        """There are 4 possible states. States 0-2 corresponding to optimal actions 0-2 and State 3 is the done state"""
        # If you are in states 0-2, depending on the receiver action you will either stay in the same state or transition to the final state (4)
        # This function should be represented as T(s'|s, a)

        probs = jax.lax.cond(
            joint_action[0] == state_num,
            lambda _: jnp.array([0.0, 0.0, 0.0, 1.0]),
            lambda _: jnp.zeros(4).at[state_num].set(1.0),
            None,
        )

        # probs = jnp.zeros(3).at[state_num].set(1.0)

        return distrax.Categorical(probs=probs)

    @partial(jax.jit, static_argnums=(0,))
    def _joint_observation_function(self, state_num, joint_action):
        """
        There are only 3 environment observations: A, B, C corresponding to 0, 1, 2
        Only agent 1 sees them.
        """
        # This function should be represented as O(o1, o2|s), a 2D categorical distribution, dim 3
        #                       Agent 0
        #                  A       B       C
        #              +-------+-------+-------+
        #            A | (0,0) | (0,1) | (0,2) |
        #              +-------+-------+-------+
        #  Agent 1   B | (1,0) | (1,1) | (1,2) |
        #              +-------+-------+-------+
        #            C | (2,0) | (2,1) | (2,2) |
        #              +-------+-------+-------+
        #                 (agent 0, agent 1)

        # This should be a 1 by 3 grid actually
        # Agent 0 sees nothing but Agent 1 sees the hidden variable

        probs = jnp.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])[
            state_num
        ]

        # def state_0():
        #     return jnp.array([0.0, 0.5, 0.5])

        # def state_0():
        #     return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0])

        # def state_1():
        #     return jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])

        # def state_2():
        #     return jnp.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        # probs = jax.lax.switch(state_num, [state_0, state_1, state_2])  # These are the only states that matter I think...
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
        """
        The reward for the done state (state idx 3) is 0
        """
        reward_array = jnp.array([-1.0, -1.0, -1.0, -0.1]).at[state].set(1.0)
        reward = jax.lax.cond(
            state == 3,
            lambda _: (0.0, 0.0),
            lambda _: (reward_array[joint_action[0]], reward_array[joint_action[0]]),
            None,
        )

        return reward
        # return (joint_action[0] == state, joint_action[0] == state)

    def _agent_0_optimal_policy(self, belief_distribution: distrax.Categorical):
        """Returns a probability distribution over possible actions in the DecPOMDP.
        belief_distribution is a categorical distrax distribution over all possible states. (Assuming 3 for simplicity)
        """
        return belief_distribution

    def _agent_1_optimal_policy(self, belief_distribution: distrax.Categorical):
        """Returns a probability distribution over possible actions in the DecPOMDP.
        belief_distribution is a categorical distrax distribution over all possible states. (Assuming 3 for simplicity)
        """
        return distrax.Categorical(probs=[1.0])

    def _is_terminal_state(self, state):
        return state == 4

    # def _optimal_policy(self, belief_distribution: distrax.Categorical):
    #     """ Returns a probability distribution over possible actions in the DecPOMDP.
    #     belief_distribution is a categorical distrax distribution over all possible states. (Assuming 3 for simplicity)
    #     """
    #     # The optimal action distribution is the same as the belief distribution...
    #     return belief_distribution

    # def _agent_1_optimal_policy(self, belief_distribution: distrax.Categorical):
    #     """
    #         Returns a probability distribution over possible actions in the DecPOMDP.
    #     """
    #     # The optimal action distribution is the same as the belief distribution...
    #     return distrax.Categorical(probs=jnp.array([1]))


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
            f"reset state: hidden_variable={env_state.hidden_variable}, "
            f"optimal_button_action={env_state.optimal_button_action}"
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
    factory = JointCategoricalPair(vars_num_categories=(1, 3))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(0, 0)))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(1, 0)))
    print(factory.sample_joint_distribution(key, env._joint_observation_function(2, 0)))

    ### Belief update test
    initial_belief = env.initial_belief_agent_1
    print(initial_belief.probs)

    belief_factory = CategoricalBeliefState(
        num_unique_states=4,
        num_unique_observations_per_agent=(1, 3),
        num_unique_actions=1,
        joint_transition_function=env._joint_transition_function,
        joint_observation_function=env._joint_observation_function,
        joint_action_constructor=env._joint_action_constructor,
    )

    new_belief = belief_factory.update_with_observation_and_joint_action(
        belief_distribution=initial_belief,
        observation=0,
        previous_joint_action=(2, 0),
        agent_id=1
    )
    print(new_belief.probs)

    uniform_belief = distrax.Categorical(probs=jnp.ones(4))

    their_beliefs = env.initial_belief_agent_0

    print(new_belief.probs)
    their_beliefs = belief_factory.update_other_belief_estimate_with_observation_only(
        other_belief_distribution_estimate=their_beliefs,
        ego_observation=0,
        previous_ego_action=3,
        other_optimal_policy=env._optimal_policy,
        agent_id=0
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
        env._optimal_policy,
        env._optimal_policy,
        new_belief,
        initial_belief,
        belief_factory,
    )
    print(cum_ret)
