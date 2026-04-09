import jax, chex
import jax.numpy as jnp
import distrax
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from flax import struct
from guessing_game import State


# This is not as efficient as signification game because each state needs to be iterated on twice before actions in the underlying env are taken
# I will eventually double-up on this I think...
@struct.dataclass
class AugmentedState:
    """
    Full state of the SignificationPOMDPGuessingGame, augmenting the underlying GuessingGame
    state with communication and belief-tracking fields.

    Each environment step is split into two micro-steps gated by ``message_status``:

    * **Phase 0 (unsent):** The sender produces an utterance; the state records it and
      flips ``message_status`` to 1.  No underlying-env action occurs yet.
    * **Phase 1 (sent):** The receiver has seen the utterance and returned its updated
      belief.  Both agents act in the underlying environment, beliefs are updated, and
      ``message_status`` resets to 0.

    Fields
    ------
    underlying_state:
        The current state of the wrapped GuessingGame environment.
    sender_agent:
        Integer (0 or 1) indicating which agent is the sender this round.
    agent_0_utterance_action:
        The utterance produced by agent 0.  Non-null only when agent 0 is the sender
        and the message has just been generated (phase 0 → 1 transition).
    agent_1_belief_action_post_utterance_from_previous_state:
        Agent 1's updated belief after receiving agent 0's utterance, carried over from
        the previous full step.  Non-null only when agent 0 was the sender last round;
        kept for logging / observation of agent 0.
    agent_1_utterance_action:
        The utterance produced by agent 1.  Non-null only when agent 1 is the sender.
    agent_0_belief_action_post_utterance_from_previous_state:
        Agent 0's updated belief after receiving agent 1's utterance, carried over from
        the previous full step.  Non-null only when agent 1 was the sender last round.
    agent_0_belief_state:
        Agent 0's current Bayesian belief over the hidden world state.
    agent_1_belief_state:
        Agent 1's current Bayesian belief over the hidden world state.
    agent_0s_estimate_of_agent_1s_belief_state:
        Agent 0's model of what agent 1 believes about the world state.
    agent_1s_estimate_of_agent_0s_belief_state:
        Agent 1's model of what agent 0 believes about the world state.
    message_status:
        0 if no message has been sent this round (phase 0), 1 if a message has been
        sent and the receiver is about to act (phase 1).
    done:
        1 when the episode has terminated, 0 otherwise.
    """

    underlying_state: State  # State from GuessingGame
    sender_agent: chex.Array

    # May eventually include an RNG key to render the utterance action
    agent_0_utterance_action: chex.Array  ## Used only when sender_agent == agent_0
    agent_1_belief_action_post_utterance_from_previous_state: (
        chex.Array
    )  ## Used only when sender_agent == agent_0, and kept only for logging
    #
    agent_1_utterance_action: chex.Array  ## Used only when sender_agent == agent_1
    agent_0_belief_action_post_utterance_from_previous_state: (
        chex.Array
    )  ## Used only when sender_agent == agent_1, and kept only for logging
    #######

    agent_0_belief_state: distrax.Categorical
    agent_1_belief_state: distrax.Categorical

    agent_0s_estimate_of_agent_1s_belief_state: distrax.Categorical
    agent_1s_estimate_of_agent_0s_belief_state: distrax.Categorical

    message_status: chex.Array  # 0: unsent, 1: sent
    done: chex.Array


class SignificationPOMDPGuessingGame(MultiAgentEnv):
    """
    A two-agent POMDP that wraps an underlying decision process (e.g. GuessingGame)
    and interleaves one round of directed communication before each environment step.

    Communication protocol
    ----------------------
    Each logical timestep consists of two JAX ``step_env`` calls:

    1. **Utterance phase** (``message_status == 0``): The designated sender emits an
       utterance.  Both agents also submit their current belief updates (ignored this
       phase — the environment records nulls in their place to avoid confusion).  The
       state records the utterance, flips ``message_status`` to 1, and returns zero
       rewards.  No underlying-env action occurs.

    2. **Action phase** (``message_status == 1``): The receiver has processed the
       utterance and returns its updated belief.  Both agents now act in the underlying
       environment via their optimal policies.  The underlying env steps, Bayesian
       beliefs are updated for both agents (own belief and estimate of other's belief),
       and ``message_status`` resets to 0 for the next round.

    Assumptions
    -----------
    * Communication is strictly unidirectional (sender → receiver) each round.
    * The underlying environment exposes ``_agent_0_optimal_policy``,
      ``_agent_1_optimal_policy``, and ``_joint_action_constructor``.
    * ``belief_factory`` implements ``update_with_observation_only`` and
      ``update_other_belief_estimate_with_observation_only``.

    Parameters
    ----------
    underlying_env:
        The underlying ``MultiAgentEnv`` (e.g. ``GuessingGame``) to wrap.
    initial_belief_distribution:
        A probability array (shape matching the state space) used as the prior belief
        at the start of each episode.
    belief_factory:
        Object responsible for Bayesian belief updates (e.g. ``CategoricalBeliefState``).
    """

    def __init__(
        self, underlying_env, initial_belief_distribution, belief_factory
    ) -> None:
        super().__init__(num_agents=2)
        self.underlying_env = underlying_env
        self.initial_belief_distribution = initial_belief_distribution
        self.belief_factory = belief_factory

        self.null_belief_distribution = distrax.Categorical(probs=jnp.zeros_like(self.initial_belief_distribution.probs))
        self.null_utterance = jnp.zeros(5)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: AugmentedState):
        """
        Construct observations for both agents from the current ``AugmentedState``.

        Each agent receives a 4-tuple:

        * **own belief state** — the agent's current posterior over world states.
        * **estimate of other's belief** — the agent's model of what the other agent
          believes.
        * **other agent's last utterance** — the utterance the other agent produced
          (null if that agent was not the sender or no utterance has been produced yet).
        * **is-sender bit** — ``True`` if this agent is the designated sender this round.

        Returns
        -------
        tuple[tuple, tuple]
            ``(agent_0_obs, agent_1_obs)``, each a 4-tuple as described above.
        """

        return (
            state.agent_0_belief_state,
            state.agent_0s_estimate_of_agent_1s_belief_state,
            state.agent_1_utterance_action,
            state.sender_agent == jnp.array(0),
        ), (
            state.agent_1_belief_state,
            state.agent_1s_estimate_of_agent_0s_belief_state,
            state.agent_0_utterance_action,
            state.sender_agent == jnp.array(1),
        )

    def step_env(self, key: chex.PRNGKey, state: AugmentedState, actions: chex.Array):
        """
        Advance the environment by one micro-step, branching on ``state.message_status``.

        Each call to ``step_env`` handles one of two phases (see class docstring):

        **Phase 0 — utterance (message_status == 0)**
            The sender's utterance action is stored in the state.  Both agents' belief
            actions are accepted but discarded (set to null) because the receiver has not
            yet seen the utterance.  Returns zero rewards and ``done=0``.

        **Phase 1 — action (message_status == 1)**
            The receiver's post-utterance belief is read.  Both agents act in the
            underlying environment via their optimal policies, observations are drawn,
            rewards are computed, and all beliefs (own + estimate of other) are updated
            by ``belief_factory``.  ``message_status`` resets to 0 for the next round.

        Parameters
        ----------
        key:
            JAX PRNG key.
        state:
            Current ``AugmentedState``.
        actions:
            Pair ``(agent_0_actions, agent_1_actions)``, each a 3-tuple of
            ``(utterance_action, belief_action, estimate_of_other_belief_post_utterance)``.

        Returns
        -------
        tuple
            ``(next_state, observations, rewards, done_flag)``
        """
        agent_0_actions, agent_1_actions = actions
        (
            agent_0_utterance_action,
            agent_0_belief_action,
            agent_0s_estimate_of_agent_1s_belief_post_utterance,
        ) = agent_0_actions
        (
            agent_1_utterance_action,
            agent_1_belief_action,
            agent_1s_estimate_of_agent_0s_belief_post_utterance,
        ) = agent_1_actions

        key, key_obs = jax.random.split(key)

        def message_unsent(_):
            # The sender has generated an utterance to send based on its belief state and its estimate of the receiver's belief state
            # We need to add that into the state, flip the bit, and move on.
            # There cannot possibly be a belief returned by the agents, so they will be null to avoid confusion.

            next_environment_state = AugmentedState(
                underlying_state=state.underlying_state,
                sender_agent=state.sender_agent,
                agent_0_utterance_action=jax.lax.cond(
                    state.sender_agent == jnp.array(0),
                    lambda _: agent_0_utterance_action,
                    lambda _: self.null_utterance,
                    None,
                ),
                agent_0_belief_action_post_utterance_from_previous_state=self.null_belief_distribution,
                agent_1_utterance_action=jax.lax.cond(
                    state.sender_agent == jnp.array(1),
                    lambda _: agent_1_utterance_action,
                    lambda _: self.null_utterance,
                    None,
                ),
                agent_1_belief_action_post_utterance_from_previous_state=self.null_belief_distribution,
                agent_0_belief_state=agent_0_belief_action,
                agent_1_belief_state=agent_1_belief_action,
                agent_0s_estimate_of_agent_1s_belief_state=state.agent_0s_estimate_of_agent_1s_belief_state,
                agent_1s_estimate_of_agent_0s_belief_state=state.agent_1s_estimate_of_agent_0s_belief_state,
                message_status=jnp.array(1),
                done=jnp.array(0),
            )

            # Now the message has been sent

            return (
                next_environment_state,
                self.get_obs(key_obs, next_environment_state),
                (0.0, 0.0),  # Rewards
                jnp.array(0),  # Done flag
            )

        def message_sent(_):
            # The message was sent. The receiver viewed it and returned its new belief state. We now act in the environment
            # So we take the sender's previous belief state and the receivers new belief state and construct a joint action given their optimal policies

            agent_0_present_belief = jax.lax.cond(
                state.sender_agent == jnp.array(0),
                lambda _: state.agent_0_belief_state,
                lambda _: agent_0_belief_action,
                None,
            )
            agent_0_underlying_action = jnp.argmax(
                self.underlying_env._agent_0_optimal_policy(agent_0_present_belief).probs
            )

            agent_1_present_belief = jax.lax.cond(
                state.sender_agent == jnp.array(1),
                lambda _: state.agent_1_belief_state,
                lambda _: agent_1_belief_action,
                None,
            )
            agent_1_underlying_action = jnp.argmax(
                self.underlying_env._agent_1_optimal_policy(agent_1_present_belief).probs
            )

            joint_action = self.underlying_env._joint_action_constructor(
                agent_id=0,
                ego_action=agent_0_underlying_action,
                other_action=agent_1_underlying_action,
            )

            # Now we act in the world

            (
                next_underlying_state,
                (agent_0_world_observation, agent_1_world_observation),
                rewards,
                done_flag,
            ) = self.underlying_env.step_env(key, state.underlying_state, joint_action)

            # Both agents must have their beliefs updated

            agent_0_next_belief = self.belief_factory.update_with_observation_only(
                agent_0_present_belief,
                state.agent_0s_estimate_of_agent_1s_belief_state,
                agent_0_world_observation,
                agent_0_underlying_action,
                self.underlying_env._agent_1_optimal_policy,
                agent_id=0,
            )
            agent_1_next_belief = self.belief_factory.update_with_observation_only(
                agent_1_present_belief,
                state.agent_1s_estimate_of_agent_0s_belief_state,
                agent_1_world_observation,
                agent_1_underlying_action,
                self.underlying_env._agent_0_optimal_policy,
                agent_id=1,
            )

            # And update their beliefs about the other agents (this is somewhat tied to the specific structure of the communication
            # e.g. this class expects strict sender -> receiver communication, and that changes how the belief estimate is calculated)

            agent_0s_post_utterance_estimate_of_agent_1s_belief_state = jax.lax.cond(
                state.sender_agent == jnp.array(0),
                lambda _: agent_0s_estimate_of_agent_1s_belief_post_utterance,
                lambda _: state.agent_0s_estimate_of_agent_1s_belief_state,  # It's unchanged if agent_0 was the receiver (in that case, agent_0 never emitted an utterance to agent_1, so it's estimate of agent_1's belief is unchanged)
                None,
            )
            agent_1s_post_utterance_estimate_of_agent_0s_belief_state = jax.lax.cond(
                state.sender_agent == jnp.array(1),
                lambda _: agent_1s_estimate_of_agent_0s_belief_post_utterance,
                lambda _: state.agent_1s_estimate_of_agent_0s_belief_state,
                None,
            )

            agent_0s_next_estimate_of_agent_1s_belief_state = (
                self.belief_factory.update_other_belief_estimate_with_observation_only(
                    agent_0s_post_utterance_estimate_of_agent_1s_belief_state,
                    agent_0_world_observation,
                    agent_0_underlying_action,
                    self.underlying_env._agent_1_optimal_policy,
                    agent_id=0,
                )
            )
            agent_1s_next_estimate_of_agent_0s_belief_state = (
                self.belief_factory.update_other_belief_estimate_with_observation_only(
                    agent_1s_post_utterance_estimate_of_agent_0s_belief_state,
                    agent_1_world_observation,
                    agent_1_underlying_action,
                    self.underlying_env._agent_0_optimal_policy,
                    agent_id=1,
                )
            )

            # Construct a new state

            next_environment_state = AugmentedState(
                underlying_state=next_underlying_state,
                sender_agent=next_underlying_state.sender_agent,
                agent_0_utterance_action=self.null_utterance,
                agent_1_belief_action_post_utterance_from_previous_state=jax.lax.cond(
                    state.sender_agent == jnp.array(0),
                    lambda _: agent_1_belief_action,
                    lambda _: self.null_belief_distribution,
                    None,
                ),
                agent_1_utterance_action=self.null_utterance,
                agent_0_belief_action_post_utterance_from_previous_state=jax.lax.cond(
                    state.sender_agent == jnp.array(1),
                    lambda _: agent_0_belief_action,
                    lambda _: self.null_belief_distribution,
                    None,
                ),
                agent_0_belief_state=agent_0_next_belief,
                agent_1_belief_state=agent_1_next_belief,
                agent_0s_estimate_of_agent_1s_belief_state=agent_0s_next_estimate_of_agent_1s_belief_state,
                agent_1s_estimate_of_agent_0s_belief_state=agent_1s_next_estimate_of_agent_0s_belief_state,
                message_status=jnp.array(0),
                done=next_underlying_state.done,
            )

            return (
                next_environment_state,
                self.get_obs(key_obs, next_environment_state),
                rewards,
                done_flag,
            )

        return jax.lax.switch(state.message_status, [message_unsent, message_sent], None)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """
        Reset the environment to a fresh episode.

        Resets the underlying GuessingGame, then initialises all belief states by
        conditioning the prior (``initial_belief_distribution``) on the first world
        observation each agent receives via ``belief_factory``.  Both agents' estimates
        of the other's belief are similarly initialised from the prior.

        ``message_status`` starts at 0 (utterance phase) so the first ``step_env`` call
        will be an utterance step.

        Parameters
        ----------
        key:
            JAX PRNG key.

        Returns
        -------
        tuple[AugmentedState, tuple]
            ``(initial_state, initial_observations)``
        """
        (
            initial_underlying_state,
            (agent_0_world_observation, agent_1_world_observation),
        ) = self.underlying_env.reset(key)

        agent_0_belief_state = (
            self.belief_factory.update_with_observation_and_joint_action(
                self.initial_belief_distribution,
                agent_0_world_observation,
                previous_joint_action=(-1, -1),
                agent_id=0,
            )
        )
        agent_1_belief_state = (
            self.belief_factory.update_with_observation_and_joint_action(
                self.initial_belief_distribution,
                agent_1_world_observation,
                previous_joint_action=(-1, -1),
                agent_id=1,
            )
        )

        agent_0s_estimate_of_agent_1s_belief_state = (
            self.belief_factory.update_other_belief_estimate_with_observation_only(
                self.initial_belief_distribution,
                agent_0_world_observation,
                0,
                self.underlying_env._agent_1_optimal_policy,
                agent_id=0,
            )
        )
        agent_1s_estimate_of_agent_0s_belief_state = (
            self.belief_factory.update_other_belief_estimate_with_observation_only(
                self.initial_belief_distribution,
                agent_1_world_observation,
                0,
                self.underlying_env._agent_0_optimal_policy,
                agent_id=1,
            )
        )

        initial_environment_state = AugmentedState(
            underlying_state=initial_underlying_state,
            sender_agent=initial_underlying_state.sender_agent,
            agent_0_utterance_action=self.null_utterance,
            agent_1_belief_action_post_utterance_from_previous_state=self.null_belief_distribution,
            agent_1_utterance_action=self.null_utterance,
            agent_0_belief_action_post_utterance_from_previous_state=self.null_belief_distribution,
            agent_0_belief_state=agent_0_belief_state,
            agent_1_belief_state=agent_1_belief_state,
            agent_0s_estimate_of_agent_1s_belief_state=agent_0s_estimate_of_agent_1s_belief_state,
            agent_1s_estimate_of_agent_0s_belief_state=agent_1s_estimate_of_agent_0s_belief_state,
            message_status=jnp.array(0),
            done=jnp.array(0),
        )

        return (initial_environment_state, self.get_obs(key, initial_environment_state))


if __name__ == "__main__":
    from guessing_game import GuessingGame
    from belief_representations import CategoricalBeliefState

    key = jax.random.key(0)

    # --- Build underlying env and belief factory ---
    underlying_env = GuessingGame()
    initial_belief = distrax.Categorical(probs=jnp.ones(3) / 3)
    belief_factory = CategoricalBeliefState(
        num_unique_states=3,
        num_unique_observations=3,
        num_unique_actions=4,
        joint_transition_function=underlying_env._joint_transition_function,
        joint_observation_function=underlying_env._joint_observation_function,
        joint_action_constructor=underlying_env._joint_action_constructor,
    )

    env = SignificationPOMDPGuessingGame(underlying_env, initial_belief, belief_factory)

    # --- Reset ---
    key, reset_key = jax.random.split(key)
    state, (obs_0, obs_1) = env.reset(reset_key)

    print("=== Initial state ===")
    print(f"  sender_agent:       {state.sender_agent}")
    print(f"  message_status:     {state.message_status}")
    print(f"  agent_0_belief:     {state.agent_0_belief_state.probs}")
    print(f"  agent_1_belief:     {state.agent_1_belief_state.probs}")
    print(f"  agent_0s_est_of_agent_1s_belief:  {state.agent_0s_estimate_of_agent_1s_belief_state.probs}")
    print(f"  agent_1s_est_of_agent_0s_belief:  {state.agent_1s_estimate_of_agent_0s_belief_state.probs}")
    print(f"  underlying obs:     {state.underlying_state.agent_0_world_observation}, {state.underlying_state.agent_1_world_observation}")

    # --- Run one full round (utterance phase + action phase) ---
    for phase in range(2):
        key, step_key = jax.random.split(key)

        # Agents pass through their current beliefs unchanged (no policy yet)
        # and send a zero utterance.
        agent_0_actions = (
            jnp.zeros(5),                                             # utterance
            state.agent_0_belief_state,                               # belief update
            state.agent_0s_estimate_of_agent_1s_belief_state,        # estimate of other's belief post-utterance
        )
        agent_1_actions = (
            jnp.zeros(5),
            state.agent_1_belief_state,
            state.agent_1s_estimate_of_agent_0s_belief_state,
        )

        state, (obs_0, obs_1), rewards, done = env.step_env(
            step_key, state, (agent_0_actions, agent_1_actions)
        )

        label = "Utterance phase" if phase == 0 else "Action phase"
        print(f"\n=== After step {phase + 1} ({label}) ===")
        print(f"  message_status:     {state.message_status}")
        print(f"  done:               {done}")
        print(f"  rewards:            {rewards}")
        print(f"  agent_0_belief:     {state.agent_0_belief_state.probs}")
        print(f"  agent_1_belief:     {state.agent_1_belief_state.probs}")
        print(f"  agent_0_belief_action_post_utterance:  {state.agent_0_belief_action_post_utterance_from_previous_state.probs}")
        print(f"  agent_1_belief_action_post_utterance:  {state.agent_1_belief_action_post_utterance_from_previous_state.probs}")
        print(f"  agent_0s_est_of_agent_1s_belief:  {state.agent_0s_estimate_of_agent_1s_belief_state.probs}")
        print(f"  agent_1s_est_of_agent_0s_belief:  {state.agent_1s_estimate_of_agent_0s_belief_state.probs}")
