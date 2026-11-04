import dataclasses
import jax, chex
import jax.numpy as jnp
import distrax
from flax import struct
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from communication.game_role_assignment import AssignmentFn
from communication.communication_scheme import CommunicationSchemeFn

# FlexibleEnvParams / OptimalPolicyBundle / NUM_ROLES are defined in envs.flexible_env (the leaf of
# the env-definition dependency graph) and consumed here.
from envs.flexible_env import NUM_ROLES, FlexibleEnvParams, OptimalPolicyBundle
from tools.belief_representations import CategoricalBeliefState


# A communicative round runs in two stages, one per step_env call; the state's
# communicative_round_stage records which stage the current step is in.
UTTERANCE_STAGE = 0  # the speaker emits an utterance / message action
BELIEF_STAGE = 1  # the listener emits a belief action, completing the round


# Terminology
# -----------
# ROUND:  One speaker->listener exchange, spanning TWO step_env calls (UTTERANCE_STAGE
#         then BELIEF_STAGE). Each round entails speaking AND listening.
# BLOCK:  The full CommunicationScheme between two underlying-env steps: a sequence of
#         rounds. The block's last round's belief stage also steps the DecPOMDP ("act").
# EPISODE: A complete sequence of blocks; ends when underlying_env_timestep hits episode_horizon,
#          at which point assignment_fn re-assigns agents to games/roles.
#
# Substates of StackedSignificationState
# ---------------------------------------
# dialog (DialogState):       The communication round state machine -- which round/stage
#                              we're in and the active block's who_speaks schedule.
# game_roles (GameRoleState): Agent-to-game/role assignment, set per episode.
# game_counters (GameCountersState): Four counters tracking position in the episode
#                              (underlying_env_timestep, episode_index) and across
#                              training (cumulative_env_timestep, episode_horizon).
#
# Beliefs (top-level, subject-indexed)
# ------------------------------------
# true_agent_belief_states[i]:      Agent i's OWN belief over world states [S].
# estimated_agent_belief_states[i]: The estimate ABOUT agent i's belief (held by i's
#                                   partner). Subject-indexed so it aligns row-for-row
#                                   with true beliefs.
#
# To get agent i's estimate OF its partner: estimated_agent_belief_states[partner(i)]
# (the partner map is its own inverse in a dyad).
#
# The estimate matters because each agent marginalizes its partner's unobserved action
# via the partner's optimal policy applied to this estimate.
#
# belief_estimate_post_utterance (step_env input) is also subject-indexed: row i is the
# refreshed estimate ABOUT agent i after i's partner (the speaker) has uttered.

@struct.dataclass
class DialogState:
    """Communication round state machine: which round/stage we're in and what the active block looks like."""

    active_who_speaks: chex.Array  # [num_rounds (padded), num_speakers]
    active_total_num_rounds: chex.Array  # scalar int: real round count of the active block (excludes padding)
    communication_round_iterator: chex.Array  # scalar int: round within the active block (0..total_num_rounds-1)
    cumulative_communication_round_iterator: chex.Array  # scalar int: total step_env calls; never resets
    communicative_round_stage: chex.Array  # scalar int: UTTERANCE_STAGE or BELIEF_STAGE


@struct.dataclass
class GameRoleState:
    """Agent-to-game/role assignment. Set at episode boundaries, read-only during steps."""

    agent_game_assignment: chex.Array  # [num_agents] -> game index
    agent_role_assignment: chex.Array  # [num_agents] -> role index
    game_types: chex.Array  # [num_games]  -> game-type index


@struct.dataclass
class GameCountersState:
    """Episode and training-run progress counters."""

    underlying_env_timestep: chex.Array  # scalar int; resets to 0 at each episode boundary
    cumulative_env_timestep: chex.Array  # scalar int; never resets; the key passed to communication_scheme_fn
    episode_index: chex.Array  # scalar int; which episode we are in; +1 at each boundary
    episode_horizon: chex.Array  # scalar int; this episode's underlying_env_steps_per_episode (from the route)


@struct.dataclass
class StackedSignificationState:
    """State carried by the StackedSignificationDecPOMDP across step_env calls.

    Games are dyadic (num_roles == 2). Agents are indexed ``[num_agents]``; games
    are indexed ``[num_games] = num_agents // num_roles``. The routing fields say
    which game/role each agent occupies, so per-agent quantities can be gathered
    to/from per-game quantities.

    Utterance fields are carried here. reset initializes the unrendered field to a
    zero vector per agent (the utterance stage of step_env writes into it); the
    rendered field is unused for now (None).
    """

    # Substates: bookkeeping grouped by concern.
    dialog: DialogState
    game_roles: GameRoleState
    game_counters: GameCountersState

    # Communication: the agents' utterance actions (raw and rendered). Populated by
    # the communication phase of step_env; None after reset.    NOTE: May want to get rid of rendered utterances
    agent_utterance_actions_unrendered: chex.Array
    agent_utterance_actions_rendered: chex.Array

    # World: the true DecPOMDP state of each game.
    game_states: chex.Array  # [num_games] (each state is a number)

    # Beliefs: both rows are indexed by the SUBJECT agent (same layout). Row i of
    # true_agent_belief_states is agent i's own belief; row i of
    # estimated_agent_belief_states is what agent i's dyadic partner estimates i believes.
    true_agent_belief_states: chex.Array  # [num_agents, S]
    estimated_agent_belief_states: chex.Array  # [num_agents, S]  (estimate ABOUT agent i)

    # Reward: each agent's underlying-DecPOMDP reward from the most recent act (the
    # stacked game's reward signal). NaN on non-act (communication-only) steps.
    last_agent_rewards: chex.Array  # [num_agents]

    # Debug: each agent's underlying-DecPOMDP action from the most recent act (the
    # action it actually took in its game). -1 on non-act (communication-only) steps.
    # Kept purely for inspection/verification; not consumed by the dynamics.
    last_agent_actions: chex.Array  # [num_agents]

    # Debug: each agent's underlying-DecPOMDP observation from the most recent act (the
    # env observation it sampled and updated its belief from). -1 on non-act steps.
    # Kept purely for inspection/verification; not consumed by the dynamics.
    last_agent_observations: chex.Array  # [num_agents]

    global_rng_key: chex.Array


class StackedSignificationDecPOMDP:
    """ """

    def __init__(
        self,
        num_agents: int,
        all_env_parameters: FlexibleEnvParams,
        optimal_policy_bundle: OptimalPolicyBundle,
        assignment_fn: AssignmentFn,
        communication_scheme_fn: CommunicationSchemeFn,
        utterance_action_dim: int,
        skip_first_communication_step: bool,
    ) -> None:
        """
        Args:
            num_agents
            all_env_parameters: Stacked FlexibleEnvParams, indexed by game type along the leading axis.
            optimal_policy_bundle: optimal_policy_bundle(game_type, role, belief) -> Categorical
                over actions; game_type and role may be traced (see
                envs.env_assembly.assemble_environments).
            assignment_fn
            communication_scheme_fn: Maps the cumulative in-game timestep to the
                CommunicationScheme in force at that timestep (see communication_scheme.py).
            utterance_action_dim: Constant length of each agent's utterance-action vector.
        """
        self.num_agents = num_agents
        self.all_env_parameters = all_env_parameters
        self.assignment_fn = assignment_fn
        self.communication_scheme_fn = communication_scheme_fn
        self.utterance_action_dim = utterance_action_dim
        self.act_on_reset_before_communicating = skip_first_communication_step

        # The (padded) world-state cardinality: the length of every belief vector agents
        # read and emit, shared by all game types (padding makes it uniform).
        self.belief_dim = all_env_parameters.initial_state_distribution.shape[-1]

        self.optimal_policy_bundle = optimal_policy_bundle
        self.num_game_types = all_env_parameters.transition.shape[0]

        # One belief-update engine per game type, built from that type's (unstacked)
        # params. Cardinalities are uniform across types (padded), so each is valid.
        # Currently the reset belief updates use only type 0 (see note in reset).
        self._belief_factories = tuple(
            CategoricalBeliefState(jax.tree.map(lambda leaf, t=game_type: leaf[t], all_env_parameters)) for game_type in range(self.num_game_types)
        )

    def _agent_belief_updates(self, game_type, role, ego_belief, other_belief_estimate, observation, action):
        """True-belief + other-belief-estimate update for one agent.

        Conditioned on the agent's environment (``game_type``, via lax.switch over
        the per-type belief factories) and its ``role`` (passed as the traced
        ``agent_id`` — the belief methods now accept a traced agent_id). The other
        agent's action is unobserved, so both updates marginalize over it via the
        other role's optimal policy. Assumes dyadic games (other role = 1 - role).

        Returns (new_true_belief_probs, new_other_estimate_probs).
        """
        other_policy = lambda belief: self.optimal_policy_bundle(game_type, 1 - role, belief)

        def make_branch(belief_factory):
            def branch(ego_belief, other_belief_estimate, observation, action):
                new_true = belief_factory.update_with_observation_only(
                    ego_belief,
                    other_belief_estimate,
                    observation,
                    action,
                    other_policy,
                    agent_id=role,
                )
                # ``ego_belief`` here is the PRE-update belief, and it must be: the estimate
                # predicts forward through this step's transition itself, so passing
                # ``new_true`` (which has already absorbed this observation) would apply the
                # step twice. It is also the only channel by which the ego's private
                # information reaches the estimate -- it says which states the EGO may have
                # entered, and hence which observations its partner plausibly received.
                new_estimate = belief_factory.update_other_belief_estimate_with_observation_only(
                    other_belief_estimate,
                    ego_belief,
                    observation,
                    action,
                    other_policy,
                    agent_id=role,
                )
                return new_true.probs, new_estimate.probs

            return branch

        branches = [make_branch(factory) for factory in self._belief_factories]
        return jax.lax.switch(game_type, branches, ego_belief, other_belief_estimate, observation, action)

    def _partner_agent(self, agent_game_assignment, agent_role_assignment, num_games):
        """Each agent's dyadic partner, plus the (game, role) -> agent map it is read from.

        Returns (partner_agent, game_role_to_agent):
          partner_agent:      [num_agents] partner_agent[i] is the agent i is paired with
                              (the agent in the OTHER role of i's game). The map is its
                              own inverse in a dyad.
          game_role_to_agent: [num_games, NUM_ROLES] the agent occupying each (game, role).

        Read and verified on 9/29/26
        """
        game_role_to_agent = (
            jnp.zeros((num_games, NUM_ROLES), dtype=jnp.int32).at[agent_game_assignment, agent_role_assignment].set(jnp.arange(self.num_agents))
        )
        partner_agent = game_role_to_agent[agent_game_assignment, 1 - agent_role_assignment]
        return partner_agent, game_role_to_agent

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: StackedSignificationState):
        """Per-agent observation: (beliefs, estimated_beliefs, utterances).

        ``key`` is currently unused (the masking is deterministic) but kept in the
        signature for future stochastic observations.

        Each is a [num_agents, ...] array, but they are NOT all indexed the same way --
        each is indexed so that row i is what agent i should CONSUME:
          - beliefs[i] is agent i's OWN belief (subject-indexed straight from
            true_agent_belief_states).
          - estimated_beliefs[i] is the estimate of the belief of the RECEIVER agent i is
            ENGAGED with -- i.e. what i (a sender) thinks its partner believes. Storage is
            subject-indexed (row k = estimate ABOUT agent k), so this is the partner's row,
            gathered by partner_agent.
          - utterances[i] is the utterance agent i should HEAR, namely the one its partner
            (the speaker) emitted -- again the partner's row of the (speaker-indexed)
            utterance store, gathered by partner_agent.

        Depending on communicative_round_stage the group that is irrelevant to the action
        the agent is about to take is filled with NaN (so the caller can mask on it); the
        shapes are unchanged either way:
          - UTTERANCE_STAGE: the agent speaks from its (own + estimated) beliefs, so the
            incoming utterance is meaningless -> utterances are NaN.
          - BELIEF_STAGE: the agent forms a new belief from the received utterance, so the
            belief inputs are meaningless -> beliefs and estimated_beliefs are NaN.

        """
        num_games = state.game_states.shape[0]
        partner_agent, _ = self._partner_agent(state.game_roles.agent_game_assignment, state.game_roles.agent_role_assignment, num_games)

        on_utterance_stage = state.dialog.communicative_round_stage == UTTERANCE_STAGE
        # beliefs[i] = agent i's own belief (subject-indexed as stored).
        beliefs = jnp.where(on_utterance_stage, state.true_agent_belief_states, jnp.nan)
        # estimated_beliefs[i] = estimate of agent i's partner (the receiver it is engaged
        # with). estimated_agent_belief_states is subject-indexed, so the estimate ABOUT
        # the partner lives at partner_agent[i].
        partner_estimates = state.estimated_agent_belief_states[partner_agent]
        estimated_beliefs = jnp.where(on_utterance_stage, partner_estimates, jnp.nan)
        # utterances[i] = the utterance agent i's partner (the speaker) emitted, stored
        # under that speaker's own row.
        partner_utterances = state.agent_utterance_actions_unrendered[partner_agent]
        utterances = jnp.where(on_utterance_stage, jnp.nan, partner_utterances)
        return beliefs, estimated_beliefs, utterances

    def _step_underlying_env(
        self,
        key,
        game_states,
        game_types_per_game,
        agent_game_assignment,
        agent_role_assignment,
        true_agent_belief_states,
        estimated_agent_belief_states,
    ):
        """Step every game's DecPOMDP once and update beliefs from the observation.

        NOTE: We may want policies to be a function of both the speaker's belief and the listener's belief...

        Each agent samples an action from its optimal policy given its true belief; the
        per-game joint actions drive the transition and joint observation; each agent
        then updates its true belief and its estimate of the partner from its own
        observation (the other agent's action is unobserved, so the updates marginalize
        over it).

        ``estimated_agent_belief_states`` is subject-indexed (row k = estimate ABOUT agent
        k), so we gather each agent's estimate-of-its-partner from the partner's row before
        the update and scatter the refreshed estimates back the same way afterward.

        Returns (next_game_states, next_true_beliefs, next_estimated_beliefs,
        agent_rewards, agent_actions, agent_observations), where agent_rewards[i] is
        agent i's underlying-DecPOMDP reward R(s, a0, a1, s') for this step -- the reward
        signal for the stacked game -- agent_actions[i] is the action agent i sampled and
        actually took in its game, and agent_observations[i] is the observation it drew
        from the step and updated its belief from (both kept for debugging/verification).
        """
        num_games = game_states.shape[0]
        agent_game_types = game_types_per_game[agent_game_assignment]  # [num_agents]

        # game_role_to_agent [num_games, NUM_ROLES] is used to assemble per-game joint actions.
        partner_agent, game_role_to_agent = self._partner_agent(agent_game_assignment, agent_role_assignment, num_games)
        # Agent i's estimate of its partner is the estimate ABOUT the partner (partner's row).
        estimate_of_partner = estimated_agent_belief_states[partner_agent]  # [num_agents, S]

        action_key, transition_key, obs_key = jax.random.split(key, 3)

        # 1. Each agent samples an action from its optimal policy given its true belief.
        #    vmap because the bundle dispatches on a scalar (game_type, role).
        agent_actions = jax.vmap(
            lambda game_type, role, belief_probs, k: self.optimal_policy_bundle(game_type, role, distrax.Categorical(probs=belief_probs)).sample(seed=k)
        )(
            agent_game_types, agent_role_assignment, true_agent_belief_states, jax.random.split(action_key, self.num_agents)
        )  # [num_agents]

        # 2. Assemble per-game joint actions in canonical (role 0, role 1) order.
        game_actions_by_role = agent_actions[game_role_to_agent]  # [num_games, num_roles]
        joint_a0, joint_a1 = game_actions_by_role[:, 0], game_actions_by_role[:, 1]

        # 3. Transition each game (gathers index the stacked params by per-game type).
        next_state_probs = self.all_env_parameters.transition[game_types_per_game, game_states, joint_a0, joint_a1]  # [num_games, S]
        game_next_states = jax.vmap(lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k))(
            next_state_probs, jax.random.split(transition_key, num_games)
        )  # [num_games]

        # 3b. Gather each game's per-role reward R_i(s, a0, a1, s') from the stacked
        # reward tensor [num_game_types, num_roles, S, A, A, S], then scatter to agents.
        game_rewards = self.all_env_parameters.reward[
            game_types_per_game[:, None],  # [num_games, 1]
            jnp.arange(NUM_ROLES)[None, :],  # [1, num_roles]
            game_states[:, None],  # [num_games, 1]
            joint_a0[:, None],  # [num_games, 1]
            joint_a1[:, None],  # [num_games, 1]
            game_next_states[:, None],  # [num_games, 1]
        ]  # [num_games, num_roles]
        agent_rewards = game_rewards[agent_game_assignment, agent_role_assignment]  # [num_agents]

        # 4. Sample the joint observation and split it into the two roles' observations.
        obs_probs = self.all_env_parameters.observation[game_types_per_game, game_next_states, joint_a0, joint_a1]  # [num_games, O, O]
        num_obs = obs_probs.shape[-1]
        flat_obs = jax.vmap(lambda probs, k: distrax.Categorical(probs=probs.reshape(-1)).sample(seed=k))(
            obs_probs, jax.random.split(obs_key, num_games)
        )  # [num_games]
        game_obs_by_role = jnp.stack([flat_obs // num_obs, flat_obs % num_obs], axis=1)  # [num_games, num_roles]
        agent_observations = game_obs_by_role[agent_game_assignment, agent_role_assignment]  # [num_agents]

        # 5. Per-agent belief + estimate-of-partner update from the observation.
        def per_agent_belief_updates(game_type, role, ego_belief_probs, partner_est_probs, ego_obs, ego_action):
            return self._agent_belief_updates(
                game_type,
                role,
                distrax.Categorical(probs=ego_belief_probs),
                distrax.Categorical(probs=partner_est_probs),
                ego_obs,
                ego_action,
            )

        next_true, next_estimate_of_partner = jax.vmap(per_agent_belief_updates)(
            agent_game_types,
            agent_role_assignment,
            true_agent_belief_states,
            estimate_of_partner,
            agent_observations,
            agent_actions,
        )
        # next_estimate_of_partner[i] is agent i's refreshed estimate of its partner; store
        # it subject-indexed -- the estimate ABOUT agent k lives at k, held by k's partner,
        # which is again a gather by partner_agent (the map is its own inverse in a dyad).
        next_estimated = next_estimate_of_partner[partner_agent]  # [num_agents, S]
        return (
            game_next_states,
            next_true,
            next_estimated,
            agent_rewards,
            agent_actions,
            agent_observations,
        )

    def step_env(
        self,
        key: chex.PRNGKey,
        state: StackedSignificationState,
        utterance_actions: chex.Array,
        belief_estimate_post_utterance: chex.Array,
        belief_actions: chex.Array,
    ):
        """
        This method has many important jobs, but the main thing it does is abstract away the complexity
        of the underlying DecPOMDPs the agents are engaged in. This method is a transition function for
        the abstract communication game.

        There are three main circumstances under which this function is called:
            1. Agents have already observed their belief state and have just emitted utterances.
                In this case, this function's job is to propagate those utterances to the correct listener via the state variable.
            2. Agents have already observed utterances and have just emitted belief states.
                In this case, this function's job is to... I don't know actually.
            3. Agents have already observed utterances and have just emitted belief states, and the communication round is over.
                In this case, the underlying environment must step forward, updating agent beliefs, and new beliefs must be passed to agents.

        At each time step, agents are either 1) generating utterances to send to interlocutors
        based on their belief about the state of the DecPOMDP and their estimate of their
        partner's belief state. Or 2) updating their belief given their prior belief and the
        utterance their partner sent to them.

        In order to abstract away the underlying machinery of the DecPOMDP, additional things need
        to happen after belief stages. Specifically, 1) the underlying DecPOMDP needs to actually advance
        according to the new belief states of the agents (by executing a policy conditioned on a belief 
        state), and 2) the agents beliefs need to update in accordance with their perceived env observation
        from env interaction, and 3) the agents' estimates about their partners beliefs need to update in 
        accordance with their perceived env observation.
        
        After all this happens, a new utterance stage begins again.

        Args:
            key: a PRNGKey.
            state: a StackedSignificationState.
            utterance_actions: [num_agents, utterance_action_dim], AGENT-indexed (row i is
                agent i's utterance).
            belief_estimate_post_utterance: [num_agents, S], SUBJECT-indexed (row i is the
                estimate ABOUT agent i's belief state). The new beliefs that agents are believed
                to have after hearing the utterances that speakers have generated for them.
            belief_actions: [num_agents, S], AGENT-indexed (row i is agent i's proposed new
                belief, a distribution over its game's S world states).

        Returns (state, observation, rewards, post_comm_true_beliefs,
        post_comm_estimated_beliefs). The last two are debug outputs: the true / estimated
        belief states after communication resolves but before the world steps forward. They are
        NaN-valued unless this step is an act phase (the belief stage of a block's last
        round). Callers that don't need them discard them with ``_``.
        """

        act_key, boundary_key, obs_key = jax.random.split(key, 3)

        # This round's speakers, over ROLES, mapped to a per-agent mask.
        who_speaks_this_round = state.dialog.active_who_speaks[state.dialog.communication_round_iterator]  # [num_roles]
        agent_speaks_binary_mask = who_speaks_this_round[state.game_roles.agent_role_assignment].astype(bool)  # [num_agents]
        # ...and this round's listeners:
        agent_listens_binary_mask = who_speaks_this_round[1 - state.game_roles.agent_role_assignment].astype(bool)  # [num_agents]

        # Useful predicates
        on_utterance_stage = state.dialog.communicative_round_stage == UTTERANCE_STAGE
        on_belief_stage = state.dialog.communicative_round_stage == BELIEF_STAGE

        # === 1. Schedule decisions ==========================================
        # A communication round has an utterance stage and a belief stage and completes on its belief stage.
        # When it's the belief stage of the final round of communication, the underlying environment must transition forward with agent actions.
        is_last_round = state.dialog.communication_round_iterator == state.dialog.active_total_num_rounds - 1
        ends_in_act_phase = on_belief_stage & is_last_round
        
        # Env timesteps advance only on an act phase. An act phase that increments underlying_env_timestep
        # to the episode horizon results in the end of the episode
        act_increment = ends_in_act_phase.astype(jnp.int32)
        incremented_cumulative_env_timestep = state.game_counters.cumulative_env_timestep + act_increment
        incremented_underlying_env_timestep = state.game_counters.underlying_env_timestep + act_increment
        is_boundary = ends_in_act_phase & (incremented_underlying_env_timestep >= state.game_counters.episode_horizon)

        # This keeps track of which communication round we are in within a block. It
        # advances only when a round completes (belief stage), wrapping to 0 after the
        # block's last round -- exactly the act.
        next_round_cursor = (state.dialog.communication_round_iterator + on_belief_stage) % state.dialog.active_total_num_rounds

        # === 2. Communication ===============================================
        # One way to think of this code subsection is that it's plumbing for passing masked inputs (e.g. utterance actions)
        # into the next state variable. That masking depends on what phase it is.

        # 2a. Utterance stage: if we are in an utterance stage, utterance_actions have been FRESHLY POPULATED.
        # What this means is that some of the utterance_actions given as input to step_env need to be
        # delivered to listener agents.
        deliver_utterance = on_utterance_stage & agent_speaks_binary_mask  # [num_agents]
        utterances_to_deliver = jnp.where(deliver_utterance[:, None], utterance_actions, jnp.nan)

        # 2b. Speaker belief estimate of listener after uttering: a speaker refreshes its estimate of its dyadic
        # partner (the listener). state.estimated_agent_belief_states is SUBJECT-indexed (row k =
        # the estimate ABOUT agent k), and belief_estimate_post_utterance shares that
        # indexing, so the rows that change are the LISTENERS' and we write straight across.
        # The estimate is formed right after the utterance, so it applies on the utterance stage.
        update_listener_belief_estimate = on_utterance_stage & agent_listens_binary_mask  # [num_agents]
        post_comm_estimated_beliefs = jnp.where(
            update_listener_belief_estimate[:, None],
            belief_estimate_post_utterance,
            state.estimated_agent_belief_states,  # Otherwise keep the existing estimate of their belief state
        )

        # 2c. Belief stage: belief_actions is freshly populated by listeners, who adopt them
        # as their new ground-truth belief states.
        update_listener_true_belief = on_belief_stage & agent_listens_binary_mask  # [num_agents]
        post_comm_true_beliefs = jnp.where(update_listener_true_belief[:, None], belief_actions, state.true_agent_belief_states)

        # === 3. World step (the act) ========================================
        # On an act phase, every game takes one DecPOMDP step: each agent samples an action from
        # its optimal policy given its post-communication belief, the games transition
        # and emit observations, and beliefs update from those observations. The step also
        # yields each agent's real reward from the environment, which becomes the communication game's
        # reward. Without an act phase, nothing world-side moves and the outputs are the no-act
        # sentinels (NaN reward).
        def step_world(_):
            return self._step_underlying_env(
                act_key,
                state.game_states,
                state.game_roles.game_types,
                state.game_roles.agent_game_assignment,
                state.game_roles.agent_role_assignment,
                post_comm_true_beliefs,
                post_comm_estimated_beliefs,
            )

        def hold_world(_):
            # No-act sentinels: NaN reward (a placeholder to be filled in retroactively once
            # communication finishes); actions/observations are int32, so they use -1.
            return (
                state.game_states,
                post_comm_true_beliefs,
                post_comm_estimated_beliefs,
                jnp.full((self.num_agents,), jnp.nan, dtype=jnp.float32),   # Rewards (nan)
                -jnp.ones((self.num_agents,), dtype=jnp.int32),             # Actions (-1)
                -jnp.ones((self.num_agents,), dtype=jnp.int32),             # Observations (-1)
            )

        (
            post_world_game_states,
            post_world_true_beliefs,
            post_world_estimated_beliefs,
            next_agent_rewards,
            next_agent_actions,
            next_agent_observations,
        ) = jax.lax.cond(ends_in_act_phase, step_world, hold_world, operand=None)

        # === 4. Episode boundary ============================================
        # On a boundary (decided in section 1), re-route a fresh assignment (keyed by the
        # next episode index), resample initial world states + beliefs, reset
        # underlying_env_timestep (to 0, or to 1 if act_on_reset_before_communicating has
        # the new episode act first, mirroring reset) and bump episode_index.
        # cumulative_env_timestep keeps counting every real env step, so it also absorbs the
        # new episode's act-first step (init["underlying_env_timestep"] is 0 or 1); this keeps
        # it in step with underlying_env_timestep. This step's reported
        # reward/action/observation stays the ENDING episode's final act (section 3); the
        # new episode's act-first only primes its beliefs/state.
        def begin_new_episode(_):
            new_episode_index = state.game_counters.episode_index + 1
            init = self._begin_episode(boundary_key, new_episode_index)
            return (
                GameRoleState(
                    agent_game_assignment=init["agent_game_assignment"],
                    agent_role_assignment=init["agent_role_assignment"],
                    game_types=init["game_types"],
                ),
                GameCountersState(
                    underlying_env_timestep=init["underlying_env_timestep"],
                    cumulative_env_timestep=incremented_cumulative_env_timestep + init["underlying_env_timestep"],
                    episode_index=new_episode_index,
                    episode_horizon=init["episode_horizon"],
                ),
                init["game_states"],
                init["true_agent_belief_states"],
                init["estimated_agent_belief_states"],
            )

        def continue_episode(_):
            return (
                state.game_roles,
                state.game_counters.replace(
                    underlying_env_timestep=incremented_underlying_env_timestep,
                    cumulative_env_timestep=incremented_cumulative_env_timestep,
                ),
                post_world_game_states,
                post_world_true_beliefs,
                post_world_estimated_beliefs,
            )

        (
            next_game_roles,
            next_game_counters,
            next_game_states,
            next_true_beliefs,
            next_estimated_beliefs,
        ) = jax.lax.cond(is_boundary, begin_new_episode, continue_episode, operand=None)

        # === 5. Advance the communication schedule ==========================
        # On an act, fetch the NEXT block, keyed by the final cumulative_env_timestep (which
        # includes a new episode's act-first step, as in reset); otherwise keep walking the
        # current one. communication_scheme_fn is a pure, shape-stable gather, so compute it
        # unconditionally and select with where.
        block_after_act = self.communication_scheme_fn(next_game_counters.cumulative_env_timestep)
        next_dialog = DialogState(
            active_who_speaks=jnp.where(ends_in_act_phase, block_after_act.who_speaks, state.dialog.active_who_speaks),
            active_total_num_rounds=jnp.where(ends_in_act_phase, block_after_act.total_num_rounds, state.dialog.active_total_num_rounds),
            communication_round_iterator=next_round_cursor,
            cumulative_communication_round_iterator=state.dialog.cumulative_communication_round_iterator + 1,
            communicative_round_stage=jnp.where(on_belief_stage, UTTERANCE_STAGE, BELIEF_STAGE),
        )

        # === 6. Assemble the new state and observation ======================
        state = state.replace(
            dialog=next_dialog,
            game_roles=next_game_roles,
            game_counters=next_game_counters,
            agent_utterance_actions_unrendered=utterances_to_deliver,
            game_states=next_game_states,
            true_agent_belief_states=next_true_beliefs,
            estimated_agent_belief_states=next_estimated_beliefs,
            last_agent_rewards=next_agent_rewards,
            last_agent_actions=next_agent_actions,
            last_agent_observations=next_agent_observations,
        )
        
        obs = self.get_obs(obs_key, state)
        return (
            state,
            obs,
            state.last_agent_rewards,
            jnp.where(ends_in_act_phase, post_comm_true_beliefs, jnp.nan),
            jnp.where(ends_in_act_phase, post_comm_estimated_beliefs, jnp.nan),
        )

    def _begin_episode(self, key, episode_index):
        """Route a fresh episode and produce its initial fields.

        Assigns agents to games/roles via assignment_fn (keyed by episode_index), samples
        each game's initial world state, and sets each agent's initial belief (and the
        initial estimate about it, which equals its own prior) from the env params.

        Then, if ``act_on_reset_before_communicating`` is set, takes one joint DecPOMDP
        step before any communication (beliefs update from the resulting observation) so
        the episode starts at in-game timestep 1; otherwise it stays communicate-first at
        timestep 0. Returns a dict of the episode-initial fields -- including the initial
        ``underlying_env_timestep`` and the act's ``last_agent_rewards`` /
        ``last_agent_actions`` / ``last_agent_observations`` (NaN / -1 / -1 when
        communicate-first) -- plus a fresh leftover
        ``key``. Shared by reset (episode 0) and the step_env episode boundary (later
        episodes), so the act-first behavior applies to every episode.
        """
        routing_key, state_key, key = jax.random.split(key, 3)
        route = self.assignment_fn(key=routing_key, episode_index=episode_index)

        agent_role_assignment = route.agent_role_assignment  # [num_agents]
        game_types_per_game = route.game_set  # [num_games]
        agent_game_types = game_types_per_game[route.agent_game_assignment]  # [num_agents]

        # Each agent's initial belief IS its game's world prior -- every agent, every role.
        # (There is no per-role initial belief; see FlexibleEnvParams.)
        # NOTE: This is subject to change in the future.
        agent_initial_belief_states = self.all_env_parameters.initial_state_distribution[agent_game_types]

        # Sample each game's true initial world state from its initial-state dist.
        num_games = game_types_per_game.shape[0]
        init_state_dists = self.all_env_parameters.initial_state_distribution[game_types_per_game]
        game_states = jax.vmap(lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k))(
            init_state_dists, jax.random.split(state_key, num_games)
        )

        true_agent_belief_states = agent_initial_belief_states
        estimated_agent_belief_states = agent_initial_belief_states
        # Default (communicate-first) path: no act yet, so the no-act sentinels (NaN reward,
        # -1 action/observation).
        last_agent_rewards = jnp.full((self.num_agents,), jnp.nan, dtype=jnp.float32)
        last_agent_actions = -jnp.ones((self.num_agents,), dtype=jnp.int32)
        last_agent_observations = -jnp.ones((self.num_agents,), dtype=jnp.int32)

        if self.act_on_reset_before_communicating:
            # Take one joint DecPOMDP step before any communication; beliefs update from
            # the resulting observation (no utterance has been exchanged yet).
            step_key, key = jax.random.split(key)
            (
                game_states,
                true_agent_belief_states,
                estimated_agent_belief_states,
                last_agent_rewards,
                last_agent_actions,
                last_agent_observations,
            ) = self._step_underlying_env(
                step_key,
                game_states,
                game_types_per_game,
                route.agent_game_assignment,
                agent_role_assignment,
                true_agent_belief_states,
                estimated_agent_belief_states,
            )

        # The act-first path advanced the underlying DecPOMDP once, so the episode starts
        # at in-game timestep 1; the communicate-first path is still at 0.
        underlying_env_timestep = jnp.asarray(1 if self.act_on_reset_before_communicating else 0, dtype=jnp.int32)

        return dict(
            agent_game_assignment=route.agent_game_assignment,
            agent_role_assignment=agent_role_assignment,
            game_types=game_types_per_game,
            game_states=game_states,
            true_agent_belief_states=true_agent_belief_states,
            estimated_agent_belief_states=estimated_agent_belief_states,
            episode_horizon=route.underlying_env_steps_per_episode,
            underlying_env_timestep=underlying_env_timestep,
            last_agent_rewards=last_agent_rewards,
            last_agent_actions=last_agent_actions,
            last_agent_observations=last_agent_observations,
            key=key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """Returns (state, observation); the observation is get_obs(state).
        Resetting only happens once. The env continues on forever according to the routing function.

        """

        # _begin_episode routes the episode and (if act_on_reset_before_communicating)
        # takes the initial act, returning the post-act fields, timestep, and rewards.
        init = self._begin_episode(key, jnp.asarray(0, dtype=jnp.int32))
        key = init["key"]
        game_states = init["game_states"]
        true_agent_belief_states = init["true_agent_belief_states"]
        estimated_agent_belief_states = init["estimated_agent_belief_states"]
        last_agent_rewards = init["last_agent_rewards"]
        last_agent_actions = init["last_agent_actions"]
        last_agent_observations = init["last_agent_observations"]

        # At reset the per-episode and cumulative env timesteps coincide (0, or 1 if the
        # episode acted first).
        underlying_timestep = init["underlying_env_timestep"]

        # Fetch the first active communication block, keyed by cumulative_env_timestep
        # (== underlying_timestep at reset), and store it flat. The round cursors start
        # at the block's first round; no communication rounds have run yet.
        active_scheme = self.communication_scheme_fn(underlying_timestep)

        state = StackedSignificationState(
            dialog=DialogState(
                active_who_speaks=active_scheme.who_speaks,
                active_total_num_rounds=active_scheme.total_num_rounds,
                communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
                cumulative_communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
                # Reset starts on the utterance stage (the speaker emits first).
                communicative_round_stage=jnp.asarray(UTTERANCE_STAGE, dtype=jnp.int32),
            ),
            game_roles=GameRoleState(
                agent_game_assignment=init["agent_game_assignment"],
                agent_role_assignment=init["agent_role_assignment"],
                game_types=init["game_types"],
            ),
            game_counters=GameCountersState(
                underlying_env_timestep=underlying_timestep,
                cumulative_env_timestep=underlying_timestep,
                episode_index=jnp.asarray(0, dtype=jnp.int32),
                episode_horizon=init["episode_horizon"],
            ),
            # No utterances yet at reset: unrendered starts as a zero vector per agent
            # (sized by utterance_action_dim) so the utterance stage can write into it;
            # rendered is unused for now.
            agent_utterance_actions_unrendered=jnp.zeros((self.num_agents, self.utterance_action_dim), dtype=jnp.float32),
            agent_utterance_actions_rendered=None,
            game_states=game_states,
            true_agent_belief_states=true_agent_belief_states,
            estimated_agent_belief_states=estimated_agent_belief_states,
            last_agent_rewards=last_agent_rewards,
            last_agent_actions=last_agent_actions,
            last_agent_observations=last_agent_observations,
            global_rng_key=key,
        )
        # Reset starts on the utterance stage, so the observation has utterances NaN'd.
        obs_key = jax.random.fold_in(key, 0)
        return state, self.get_obs(obs_key, state)


@dataclasses.dataclass
class EnvironmentConfig:
    """Tyro-facing knobs for building the StackedSignificationDecPOMDP.

    Fields:
        utterance_action_dim: Length of each agent's (flat, unrendered) utterance-action
            vector. Shared with the utterance agents (whose output must match), so
            make_train reads it from here and feeds it to both the env and the agent build.
            Interpreted as spline control points (6 per quadratic Bezier spline) by
            ``tools.utterance_rendering``, so it must be a multiple of 6 to be renderable.
        utterance_image_dim: Side length of the square canvas a flat utterance is rendered
            onto before a belief agent's conv encoder consumes it (see
            ``tools.utterance_rendering``); the belief agents' ``input_utterance_shape`` is
            ``(utterance_image_dim, utterance_image_dim)``.
        skip_first_communication_step: If set, every episode takes one underlying-env
            step (act) before any communication (``act_on_reset_before_communicating``).
        only_games: Names of the game types to assemble, in order -- game type index i is
            ``only_games[i]``. Empty means "every registered game" (in ``GAME_SPECS`` order).
            Names must be keys of ``envs.env_assembly.GAME_SPECS``.
    """

    utterance_action_dim: int = 12  # 2 splines x 6 control points; multiple of 6 to render
    utterance_image_dim: int = 32
    skip_first_communication_step: bool = False
    only_games: Tuple[str, ...] = ()    # defaults to all games when left empty

    def build(
        self,
        num_agents: int,
        assignment_fn: AssignmentFn,
        communication_scheme_fn: CommunicationSchemeFn,
    ) -> "StackedSignificationDecPOMDP":
        # Imported here (not at module top) to keep the env-definition module free of a
        # dependency on the assembly layer, matching the local imports in the examples.
        from envs.env_assembly import assemble_environments, GAME_SPECS

        game_names = self.only_games or tuple(GAME_SPECS.keys())
        specs = [GAME_SPECS[name] for name in game_names]
        stacked_env_parameters, optimal_policy_bundle = assemble_environments(specs)

        return StackedSignificationDecPOMDP(
            num_agents=num_agents,
            all_env_parameters=stacked_env_parameters,
            optimal_policy_bundle=optimal_policy_bundle,
            assignment_fn=assignment_fn,
            communication_scheme_fn=communication_scheme_fn,
            utterance_action_dim=self.utterance_action_dim,
            skip_first_communication_step=self.skip_first_communication_step,
        )


def a_to_b_thrice_example(key: chex.PRNGKey = jax.random.key(0)):
    """Build a small stacked env driven by ``a_to_b_thrice_scheme_fn`` and walk one block.

    ``a_to_b_thrice`` is a single 3-round block in which role A speaks to role B on
    each of the three rounds; the third round's belief stage is the act that steps
    the underlying DecPOMDP. Each round is two ``step_env`` calls (utterance then
    belief), so one block == 6 ``step_env`` calls == one underlying-env step. The walk
    prints the schedule as it advances so the stage / round / env-timestep cadence is
    visible. Returns ``(env, final_state)``.
    """
    from communication.game_role_assignment import simple_assignment_fn
    from communication.communication_scheme import a_to_b_thrice_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    stacked_params, optimal_policy_bundle = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=a_to_b_thrice_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )

    reset_key, walk_key = jax.random.split(key)
    state, _ = env.reset(reset_key)

    # Per-agent actions fed to step_env each call: an all-ones utterance, a one-hot
    # belief_action (must be a valid distribution -- the act samples an action from it),
    # and a valid post-utterance estimate (skew the reset belief so it stays full-support;
    # a padding-state one-hot would make the act's observation update degenerate).
    utterance_actions = jnp.ones((env.num_agents, env.utterance_action_dim))
    num_states = state.true_agent_belief_states.shape[-1]
    belief_actions = jnp.zeros((env.num_agents, num_states)).at[:, 0].set(1.0)
    belief_estimate = state.true_agent_belief_states.at[:, 0].add(0.5)
    belief_estimate = belief_estimate / belief_estimate.sum(axis=-1, keepdims=True)

    def fmt(st):
        return (
            f"stage={int(st.dialog.communicative_round_stage)} "
            f"round={int(st.dialog.communication_round_iterator)} "
            f"env_iter={int(st.game_counters.underlying_env_timestep)} "
            f"comm={int(st.dialog.cumulative_communication_round_iterator)}"
        )

    print("=== a_to_b_thrice example (1 block = 3 rounds = 6 step_env calls) ===")
    print("  reset: ", fmt(state))
    for t in range(6):
        walk_key, step_key = jax.random.split(walk_key)
        state, _, rewards, _, _ = env.step_env(step_key, state, utterance_actions, belief_estimate, belief_actions)
        print(f"  step {t}:", fmt(state), "reward_sum=", float(rewards.sum()))
    print("ok: a_to_b_thrice walks 3 rounds and acts once on the last belief stage")
    return env, state


if __name__ == "__main__":
    a_to_b_thrice_example(jax.random.key(0))

    from communication.game_role_assignment import simple_assignment_fn
    from communication.communication_scheme import b_to_a_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    # Build the stacked params + policy table from one game type (the guessing game).
    stacked_params, optimal_policy_bundle = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )

    # reset (communicate-first path): beliefs stay at initial.
    state, obs = env.reset(jax.random.key(0))
    obs_beliefs, obs_estimated, obs_utterances = obs
    # Reset is on the utterance stage: beliefs are present, utterances NaN'd.
    assert jnp.all(jnp.isnan(obs_utterances)), "utterance-stage obs NaNs the utterances"
    assert not jnp.any(jnp.isnan(obs_beliefs)), "utterance-stage obs keeps beliefs"
    assert not jnp.any(jnp.isnan(obs_estimated)), "utterance-stage obs keeps estimated beliefs"
    print("=== communicate-first reset ===")
    print("game_states:        ", state.game_states)
    print("underlying_iter:    ", state.game_counters.underlying_env_timestep)
    print("active who_speaks:  ", state.dialog.active_who_speaks.tolist())
    print("active rounds:      ", state.dialog.active_total_num_rounds)
    print(
        "round iters (u,c):  ",
        state.dialog.communication_round_iterator,
        state.dialog.cumulative_communication_round_iterator,
    )
    print("round stage:        ", state.dialog.communicative_round_stage)
    print("true beliefs[0]:    ", state.true_agent_belief_states[0])
    assert state.game_counters.underlying_env_timestep == 0, "communicate-first takes no env step at reset"
    assert state.dialog.communication_round_iterator == 0, "round cursor starts at 0"
    assert state.dialog.communicative_round_stage == UTTERANCE_STAGE, "reset starts on the utterance stage"

    # reset (act-first path): one joint action + belief updates before communicating.
    act_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=True,
    )
    act_state, _ = act_env.reset(jax.random.key(1))
    print("=== act-first reset ===")
    print("game_states:        ", act_state.game_states)
    print("underlying_iter:    ", act_state.game_counters.underlying_env_timestep)
    print("true beliefs[0]:    ", act_state.true_agent_belief_states[0])
    print("estimate about agent 0:", act_state.estimated_agent_belief_states[0])
    assert act_state.game_counters.underlying_env_timestep == 1, "act-first advances the env once at reset"

    # Direct dispatch check: a belief is routed through the (game_type, role)
    # policy. Both guessing-game roles are identity here, so each call should
    # return its own belief back through the correct branch.
    belief_role_0 = jnp.array([0.6, 0.3, 0.1, 0.0])
    belief_role_1 = jnp.array([0.2, 0.2, 0.6, 0.0])
    role_0_probs = env.optimal_policy_bundle(0, 0, distrax.Categorical(probs=belief_role_0)).probs
    role_1_probs = env.optimal_policy_bundle(0, 1, distrax.Categorical(probs=belief_role_1)).probs
    print("role 0 action probs:", role_0_probs)
    print("role 1 action probs:", role_1_probs)
    assert jnp.allclose(role_0_probs, belief_role_0)
    assert jnp.allclose(role_1_probs, belief_role_1)
    print("ok: env built via factory; policies dispatch by (game_type, role)")

    # step_env section 1: schedule decisions. Use a 3-round block (a_to_b_thrice) so
    # the round cursor visibly walks 0,0,1,1,2,2 (two stages each) and the env timestep
    # advances exactly once per completed block (every 6 step_env calls).
    from communication.communication_scheme import a_to_b_thrice_scheme_fn

    sched_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=a_to_b_thrice_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )
    s0, _ = sched_env.reset(jax.random.key(2))
    utt = jnp.ones((10, sched_env.utterance_action_dim))  # every agent "says" all-ones
    num_states = s0.true_agent_belief_states.shape[-1]
    # belief_actions are beliefs over states; the act samples actions from them, so they
    # must be valid distributions. Use a one-hot at state 0.
    valid_belief = jnp.zeros((10, num_states)).at[:, 0].set(1.0)
    # Post-utterance belief estimate, subject-indexed like the state field. Must be a
    # *valid* belief (full support over real states) because the act marginalizes the
    # partner's action through it; a one-hot on the padding state would make the
    # observation update degenerate. Skew the reset belief so it is distinct enough for
    # section 2b's wiring to be visible.
    other_est = s0.true_agent_belief_states.at[:, 0].add(0.5)
    other_est = other_est / other_est.sum(axis=-1, keepdims=True)
    role0 = s0.game_roles.agent_role_assignment == 0
    role1 = ~role0

    # Section 2a: the utterance stage stashes ONLY the speaking role's (role 0 in a_to_b)
    # utterances; listeners (role 1) are NaN'd.
    after_utt, after_utt_obs, after_utt_rewards, _, _ = sched_env.step_env(jax.random.key(0), s0, utt, other_est, valid_belief)
    # A communication-only step (the utterance stage) yields a NaN (placeholder) reward.
    assert jnp.all(jnp.isnan(after_utt_rewards)), "utterance-stage step returns NaN reward"
    assert jnp.array_equal(after_utt_rewards, after_utt.last_agent_rewards, equal_nan=True), "returned reward matches state"
    assert jnp.all(after_utt.agent_utterance_actions_unrendered[role0] == 1.0)
    assert jnp.all(jnp.isnan(after_utt.agent_utterance_actions_unrendered[role1]))
    print("ok: utterance stage stashes speakers' (role 0) utterances, NaNs listeners")

    # get_obs: after one step we are on the belief stage, so the observation NaNs the
    # belief groups and keeps the utterances (the opposite of the reset observation).
    ou_beliefs, ou_estimated, ou_utterances = after_utt_obs
    assert jnp.all(jnp.isnan(ou_beliefs)), "belief-stage obs NaNs the beliefs"
    assert jnp.all(jnp.isnan(ou_estimated)), "belief-stage obs NaNs the estimated beliefs"
    assert not jnp.any(jnp.isnan(ou_utterances[role1])), "belief-stage obs keeps the listeners' utterances"
    print("ok: belief-stage observation NaNs beliefs, keeps utterances")

    # get_obs indexing: utterances are remapped to the PARTNER (speaker), not the agent's
    # own row. role 0 spoke ones, role 1 was NaN'd; so each listener (role 1) HEARS its
    # role-0 partner's ones, and each speaker (role 0) hears its role-1 partner's NaNs.
    assert jnp.all(ou_utterances[role1] == 1.0), "listeners hear their partner-speaker's utterance"
    assert jnp.all(jnp.isnan(ou_utterances[role0])), "speakers hear their (silent listener) partner"
    print("ok: get_obs routes each agent its partner's utterance, not its own")

    # get_obs indexing: on the utterance stage, estimated_beliefs is the estimate of the
    # RECEIVER (partner) the sender is engaged with -- the partner's subject-indexed row.
    us_beliefs, us_estimated, us_utterances = sched_env.get_obs(jax.random.key(0), s0)
    partner, _ = sched_env._partner_agent(s0.game_roles.agent_game_assignment, s0.game_roles.agent_role_assignment, s0.game_states.shape[0])
    assert jnp.all(us_beliefs == s0.true_agent_belief_states), "beliefs[i] is agent i's own belief"
    assert jnp.all(
        us_estimated == s0.estimated_agent_belief_states[partner]
    ), "estimated_beliefs[i] is the estimate of agent i's partner (the receiver)"
    print("ok: get_obs routes each sender the estimate of its receiver partner")

    # Section 2b: the estimate is subject-indexed, so the LISTENERS' rows (role 1, the agents
    # a speaker just re-estimated) take the supplied value; the speakers' rows (role 0)
    # keep their prior estimate.
    assert jnp.all(after_utt.estimated_agent_belief_states[role1] == other_est[role1])
    assert jnp.all(after_utt.estimated_agent_belief_states[role0] == s0.estimated_agent_belief_states[role0])
    print("ok: utterance stage records the listener-subject (role 1) post-utterance estimate")

    # Section 2c: the following belief stage has the listeners (role 1) adopt their proposed
    # belief_actions; the speakers (role 0) keep their belief.
    after_belief, _, _, _, _ = sched_env.step_env(jax.random.key(1), after_utt, utt, other_est, valid_belief)
    assert jnp.all(after_belief.true_agent_belief_states[role1] == valid_belief[role1])
    assert jnp.all(after_belief.true_agent_belief_states[role0] == s0.true_agent_belief_states[role0])
    print("ok: belief stage applies listeners' (role 1) belief_actions, speakers unchanged")

    def fmt(st):
        return (
            f"stage={int(st.dialog.communicative_round_stage)} "
            f"round={int(st.dialog.communication_round_iterator)} "
            f"env_iter={int(st.game_counters.underlying_env_timestep)} "
            f"comm={int(st.dialog.cumulative_communication_round_iterator)}"
        )

    print("=== step_env scheduling walk (a_to_b_thrice: 3 rounds/block) ===")
    s = s0
    print("  reset:  ", fmt(s))
    for t in range(6):
        s, _, step_rewards, _, _ = sched_env.step_env(jax.random.key(t), s, utt, other_est, valid_belief)
        assert jnp.array_equal(step_rewards, s.last_agent_rewards, equal_nan=True), "step_env returns this step's reward"
        print(
            f"  step {t}:",
            fmt(s),
            "reward_sum=",
            float(step_rewards.sum()),
            "actions=",
            s.last_agent_actions.tolist(),
            "obs=",
            s.last_agent_observations.tolist(),
        )
        # The act is the last belief stage (step 5); every earlier (communication-only)
        # step leaves the reward signal at NaN and the debug actions/observations at the
        # -1 sentinel.
        if t < 5:
            assert jnp.all(jnp.isnan(s.last_agent_rewards)), "no reward on communication-only steps"
            assert jnp.all(s.last_agent_actions == -1), "no action on communication-only steps"
            assert jnp.all(s.last_agent_observations == -1), "no observation on communication-only steps"
        else:
            assert jnp.all(s.last_agent_actions >= 0), "act step records real actions"
            assert jnp.all(s.last_agent_observations >= 0), "act step records real observations"
    assert s.last_agent_rewards.shape == (10,), "reward is per-agent"
    assert s.last_agent_actions.shape == (10,), "action is per-agent"
    assert s.last_agent_observations.shape == (10,), "observation is per-agent"
    assert int(s.game_counters.underlying_env_timestep) == 1, "one block (6 stages) == one env step"
    assert int(s.dialog.communication_round_iterator) == 0, "cursor wrapped after the act"
    assert int(s.dialog.communicative_round_stage) == UTTERANCE_STAGE, "back to utterance after the act"
    # Section 3: the act ran a real DecPOMDP step -> beliefs stay normalized after the obs update.
    assert jnp.allclose(s.true_agent_belief_states.sum(-1), 1.0), "act keeps beliefs normalized"
    print("  game_states reset->now:", s0.game_states.tolist(), "->", s.game_states.tolist())
    print("ok: scheduler cycles stages/rounds and the act steps the underlying env once")

    # Section 4: episode boundary. Horizon = 2 underlying steps; b_to_a is 1 round/block, so
    # an env step every 2 step_env calls and a boundary every 4 calls.
    boundary_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, underlying_env_steps_per_episode=2),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=False,
    )
    b, _ = boundary_env.reset(jax.random.key(7))
    utt3 = jnp.ones((10, 3))
    valid3 = jnp.zeros((10, num_states)).at[:, 0].set(1.0)
    # Valid (full-support) estimate; a padding-state one-hot would break the act.
    other_est3 = b.true_agent_belief_states.at[:, 0].add(0.5)
    other_est3 = other_est3 / other_est3.sum(axis=-1, keepdims=True)
    assert int(b.game_counters.episode_index) == 0 and int(b.game_counters.episode_horizon) == 2
    print("=== episode boundary walk (horizon=2, b_to_a) ===")
    for t in range(4):
        b, _, _, _, _ = boundary_env.step_env(jax.random.key(100 + t), b, utt3, other_est3, valid3)
        print(
            f"  step {t}: env_iter={int(b.game_counters.underlying_env_timestep)} "
            f"cum_env={int(b.game_counters.cumulative_env_timestep)} episode={int(b.game_counters.episode_index)}"
        )
    assert int(b.game_counters.episode_index) == 1, "a new episode began at the horizon"
    assert int(b.game_counters.underlying_env_timestep) == 0, "per-episode env iter reset at the boundary"
    assert int(b.game_counters.cumulative_env_timestep) == 2, "cumulative env iter keeps counting"
    print("ok: episode boundary re-routes and resets the per-episode counters")

    # Act-first boundary: every episode (not just the first) takes an env step before
    # communicating, and cumulative_env_timestep counts it too, staying in step with
    # underlying_env_timestep. Reset: (underlying, cumulative) = (1, 1); the block's act
    # hits the horizon (2 -> boundary, cumulative 2); the new episode's act-first step
    # makes it (1, 3).
    act_first_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policy_bundle=optimal_policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, underlying_env_steps_per_episode=2),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=True,
    )
    af, _ = act_first_env.reset(jax.random.key(8))
    assert (int(af.game_counters.underlying_env_timestep), int(af.game_counters.cumulative_env_timestep)) == (1, 1)
    for t in range(2):
        af, _, _, _, _ = act_first_env.step_env(jax.random.key(200 + t), af, utt3, other_est3, valid3)
    assert int(af.game_counters.episode_index) == 1, "a new episode began at the horizon"
    assert int(af.game_counters.underlying_env_timestep) == 1, "the new episode acted first"
    assert int(af.game_counters.cumulative_env_timestep) == 3, "cumulative counts the new episode's act-first step"
    print("ok: act-first episodes keep cumulative_env_timestep in step with underlying_env_timestep")
