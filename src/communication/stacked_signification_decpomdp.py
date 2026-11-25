import dataclasses
import jax, chex
import jax.numpy as jnp
import distrax
from flax import struct
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from communication.game_role_assignment import AssignmentFn
from communication.communication_scheme import CommunicationSchemeFn

# FlexibleEnvParams / PolicyBundle / NUM_ROLES are defined in envs.flexible_env (the leaf of
# the env-definition dependency graph) and consumed here.
from envs.flexible_env import NUM_ROLES, FlexibleEnvParams, PolicyBundle
from tools.belief_representations import (
    update_other_belief_estimate_with_observation_only,
    update_second_order_ego_estimate_with_observation_only,
    update_with_observation_only,
)


# A communicative round runs in two stages, one per step_env call; the state's
# communicative_round_stage records which stage the current step is in.
UTTERANCE_STAGE = 0  # the speaker emits an utterance / message action
BELIEF_STAGE = 1  # the listener emits a belief action, completing the round


# Terminology (units of time, smallest to largest)
# ------------------------------------------------
# STAGE:  One step_env call -- either an UTTERANCE_STAGE or a BELIEF_STAGE. The unit the
#         training rollout scans over (ExperimentConfig.rollout_length counts stages).
# ROUND:  One speaker->listener utterance exchange, spanning TWO stages
#         (UTTERANCE_STAGE then BELIEF_STAGE). Each round entails speaking AND listening.
# BLOCK:  The full CommunicationScheme between two underlying-env steps: a sequence of
#         rounds. The block's last round's belief stage also steps the DecPOMDP ("act").
# ACT:    One step of the underlying DecPOMDP, taken on the final belief stage of a block.
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
# true_agent_belief_states[i]:      Agent i's OWN belief over world states [S].   (b_e)
# estimated_agent_belief_states[i]: The estimate ABOUT agent i's belief (held by i's
#                                   partner). Subject-indexed so it aligns row-for-row
#                                   with true beliefs.                            (b̄_o)
# second_order_agent_belief_estimates[i]: Agent i's estimate of its partner's estimate
#                                   of agent i's belief. It is ABOUT agent i and HELD by
#                                   agent i, so row i is both.                    (b̿_e)
#
# To get agent i's estimate OF its partner: estimated_agent_belief_states[partner(i)]
# (the partner map is its own inverse in a dyad).
#
# Policies are Boltzmann over Q(b_own, b_estimate_of_partner). Agent i acts on
# (b_e, b̄_o), and predicts its partner's unobserved action from its estimates of the
# partner's inputs, (b̄_o, b̿_e) -- which is why the env tracks b̿_e.
#
# step_env's estimate inputs use the same conventions as the state they overwrite:
# *_other_belief_estimate_actions are subject-indexed (row i = the new estimate ABOUT agent
# i, written by i's partner), *_second_order_ego_estimate_actions are by holder (row i =
# agent i's new b̿_e). See step_env for which rows each stage reads.

@struct.dataclass
class DialogState:
    """Communication round state machine: which round/stage we're in and what the active block looks like."""

    active_who_speaks: chex.Array  # [num_rounds (padded), num_speakers]
    active_total_num_rounds: chex.Array  # scalar int: real round count of the active block (excludes padding)
    communication_round_iterator: chex.Array  # scalar int: round within the active block (0..total_num_rounds-1)
    cumulative_stage_count: chex.Array  # scalar int: total step_env calls; never resets
    communicative_round_stage: chex.Array  # scalar int: UTTERANCE_STAGE or BELIEF_STAGE


@struct.dataclass
class GameRoleState:
    """Agent-to-game/role assignment. Set at episode boundaries, read-only during steps."""

    agent_game_assignment: chex.Array  # [num_agents] -> game index
    agent_role_assignment: chex.Array  # [num_agents] -> role index
    game_types: chex.Array  # [num_games]  -> game-type index

    # Derived from the assignment above; build via from_assignment so they stay consistent.
    game_role_to_agent: chex.Array  # [num_games, NUM_ROLES] -> agent occupying that (game, role)
    partner_agent: chex.Array  # [num_agents] -> agent i's dyadic partner (its own inverse)

    @classmethod
    def from_assignment(cls, agent_game_assignment, agent_role_assignment, game_types):
        """Build the state from an assignment, deriving game_role_to_agent and partner_agent."""
        num_agents = agent_game_assignment.shape[0]
        num_games = game_types.shape[0]
        game_role_to_agent = jnp.zeros((num_games, NUM_ROLES), dtype=jnp.int32).at[agent_game_assignment, agent_role_assignment].set(jnp.arange(num_agents))
        partner_agent = game_role_to_agent[agent_game_assignment, 1 - agent_role_assignment]
        return cls(
            agent_game_assignment=agent_game_assignment,
            agent_role_assignment=agent_role_assignment,
            game_types=game_types,
            game_role_to_agent=game_role_to_agent,
            partner_agent=partner_agent,
        )


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
    # Row i is agent i's estimate of its partner's estimate of agent i's belief: about i
    # AND held by i, so no partner gather is needed in either direction.
    second_order_agent_belief_estimates: chex.Array  # [num_agents, S]

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
        policy_bundle: PolicyBundle,
        assignment_fn: AssignmentFn,
        communication_scheme_fn: CommunicationSchemeFn,
        utterance_action_dim: int,
    ) -> None:
        """
        Args:
            num_agents
            all_env_parameters: Stacked FlexibleEnvParams, indexed by game type along the leading axis.
            policy_bundle: policy_bundle(game_type, role, ego_belief, other_belief_estimate)
                -> Categorical over actions, a Boltzmann policy over that (game type,
                role)'s Q function; game_type and role may be traced (see
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

        # The (padded) world-state cardinality: the length of every belief vector agents
        # read and emit, shared by all game types (padding makes it uniform).
        self.belief_dim = all_env_parameters.initial_state_distribution.shape[-1]

        self.policy_bundle = policy_bundle
        self.num_game_types = all_env_parameters.transition.shape[0]

    def _untracked_level_predictions(self, game_type, role, game_params, second_order_ego_estimate, is_first_act):
        """The two action predictions that reach belief levels above b̿_e, which are not
        tracked. Returns ``(ego_action_as_they_predict, other_action_as_they_think_we_predict)``:

            ego_action_as_they_predict            = π_e(b̿_e, b⃛_o)   our action, as the partner predicts it
            other_action_as_they_think_we_predict = π_o(b⃛_o, b⁗_e)   their action, as they think we predict it

        A lower level is never substituted for an untracked one. Both are Boltzmann policies
        from the bundle; the tracked b̿_e goes in its slot, and every untracked slot (b⃛_o,
        b⁗_e) gets the same stand-in belief:

          - on the episode's first act (``is_first_act``), the game's
            initial_state_distribution -- what every level is known to start from, even
            after the communication that precedes that act;
          - on every later act, a uniform belief over ALL the game's real states (terminal
            ones included), i.e. assuming almost no knowledge of the dynamics.

        Padded states get zero probability in either stand-in.
        """
        num_states = game_params.transition.shape[0]
        real_states = (jnp.arange(num_states) < game_params.num_states).astype(jnp.float32)
        untracked_belief = distrax.Categorical(
            probs=jnp.where(is_first_act, game_params.initial_state_distribution, real_states / real_states.sum())
        )
        ego_action_as_they_predict = self.policy_bundle(game_type, role, second_order_ego_estimate, untracked_belief)
        other_action_as_they_think_we_predict = self.policy_bundle(game_type, 1 - role, untracked_belief, untracked_belief)
        return ego_action_as_they_predict, other_action_as_they_think_we_predict

    def _agent_belief_updates(self, game_type, role, ego_belief_probs, other_belief_estimate_probs, second_order_ego_estimate_probs, observation, action, is_first_act):
        """One agent's update of all three tracked levels (b_e, b̄_o, b̿_e) after an act.

        Takes and returns raw [S] probs (the env's storage format); wraps them in
        Categoricals here for the tools.belief_representations functions. Every update
        reads the PREVIOUS values of all three levels.

        Conditioned on the agent's environment (``game_type``, by gathering that game
        type's params out of the stacked params -- the belief functions accept traced
        params) and its ``role`` (passed as the traced ``ego_role``). The other agent's
        action is unobserved, so the updates marginalize over it, predicting it with the
        other role's Boltzmann policy at our estimates of its inputs, π_o(b̄_o, b̿_e).
        Assumes dyadic games (other role = 1 - role). ``is_first_act`` (this act is the
        episode's first) picks the stand-in for the untracked levels; see
        _untracked_level_predictions.

        Returns (new_true_belief_probs, new_other_estimate_probs, new_second_order_probs).
        """
        # This game type's (unstacked) params; game_type may be traced.
        game_params = jax.tree.map(lambda leaf: leaf[game_type], self.all_env_parameters)
        ego_belief = distrax.Categorical(probs=ego_belief_probs)
        other_belief_estimate = distrax.Categorical(probs=other_belief_estimate_probs)
        second_order_ego_estimate = distrax.Categorical(probs=second_order_ego_estimate_probs)

        # Their action at the act, as we predict it: their policy at their inputs as we
        # estimate them -- their belief (b̄_o) and their estimate of ours (b̿_e).
        other_action_as_we_predict = self.policy_bundle(game_type, 1 - role, other_belief_estimate, second_order_ego_estimate)
        ego_action_as_they_predict, other_action_as_they_think_we_predict = self._untracked_level_predictions(
            game_type, role, game_params, second_order_ego_estimate, is_first_act
        )

        new_true = update_with_observation_only(
            game_params,
            ego_belief=ego_belief,
            ego_observation=observation,
            ego_action_tminus1=action,
            other_action_as_we_predict=other_action_as_we_predict,
            ego_role=role,
        )
        new_estimate = update_other_belief_estimate_with_observation_only(
            game_params,
            ego_belief=ego_belief,
            other_belief_estimate=other_belief_estimate,
            ego_observation=observation,
            ego_action=action,
            other_action_as_we_predict=other_action_as_we_predict,
            ego_role=role,
            ego_action_as_they_predict=ego_action_as_they_predict,
        )
        new_second_order = update_second_order_ego_estimate_with_observation_only(
            game_params,
            ego_belief=ego_belief,
            other_belief_estimate=other_belief_estimate,
            second_order_ego_estimate=second_order_ego_estimate,
            ego_observation=observation,
            ego_action=action,
            other_action_as_we_predict=other_action_as_we_predict,
            ego_role=role,
            ego_action_as_they_predict=ego_action_as_they_predict,
            other_action_as_they_think_we_predict=other_action_as_they_think_we_predict,
        )
        return new_true.probs, new_estimate.probs, new_second_order.probs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, key: chex.PRNGKey, state: StackedSignificationState):
        """Per-agent observation: (beliefs, estimated_beliefs, second_order_estimates, utterances).

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
          - second_order_estimates[i] is what agent i thinks its partner thinks agent i
            believes (b̿_e). Stored by holder, which is also the subject, so row i as is.
          - utterances[i] is the utterance agent i should HEAR, namely the one its partner
            (the speaker) emitted -- again the partner's row of the (speaker-indexed)
            utterance store, gathered by partner_agent.

        beliefs, estimated_beliefs and second_order_estimates are always the current values,
        on both stages: the speaker utters from them, and the listener updates them.

        utterances are NaN on the UTTERANCE_STAGE, because there is no utterance to hear
        yet (the speakers are about to emit them). On the BELIEF_STAGE, listeners hear
        their speaker's utterance; a speaker's row is its silent listener's NaN row.
        """
        partner_agent = state.game_roles.partner_agent

        on_utterance_stage = state.dialog.communicative_round_stage == UTTERANCE_STAGE
        # beliefs[i] = agent i's own belief (subject-indexed as stored).
        beliefs = state.true_agent_belief_states
        # estimated_beliefs[i] = estimate of agent i's partner (the agent it is engaged
        # with). estimated_agent_belief_states is subject-indexed, so the estimate ABOUT
        # the partner lives at partner_agent[i].
        estimated_beliefs = state.estimated_agent_belief_states[partner_agent]
        # utterances[i] = the utterance agent i's partner (the speaker) emitted, stored
        # under that speaker's own row.
        partner_utterances = state.agent_utterance_actions_unrendered[partner_agent]
        utterances = jnp.where(on_utterance_stage, jnp.nan, partner_utterances)
        second_order_estimates = state.second_order_agent_belief_estimates
        return beliefs, estimated_beliefs, second_order_estimates, utterances

    def _step_underlying_env(
        self,
        key,
        game_states,
        game_roles: GameRoleState,
        true_agent_belief_states,
        estimated_agent_belief_states,
        second_order_agent_belief_estimates,
        underlying_env_timestep,
    ):
        """Step every game's DecPOMDP once and update beliefs from the observation.

        Each agent samples an action from its Boltzmann policy at (its own belief, its
        estimate of its partner); the per-game joint actions drive the transition and
        joint observation; each agent then updates all three tracked levels from its own
        observation (the other agent's action is unobserved, so the updates marginalize
        over it).

        ``estimated_agent_belief_states`` is subject-indexed (row k = estimate ABOUT agent
        k), so we gather each agent's estimate-of-its-partner from the partner's row before
        the update and scatter the refreshed estimates back the same way afterward.

        ``underlying_env_timestep`` is the per-episode env timestep BEFORE this act (0 on the
        episode's first act); it selects the stand-in for the untracked belief levels.

        Returns (next_game_states, next_true_beliefs, next_estimated_beliefs,
        next_second_order_estimates, agent_rewards, agent_actions, agent_observations), where agent_rewards[i] is
        agent i's underlying-DecPOMDP reward R(s, a0, a1, s') for this step -- the reward
        signal for the stacked game -- agent_actions[i] is the action agent i sampled and
        actually took in its game, and agent_observations[i] is the observation it drew
        from the step and updated its belief from (both kept for debugging/verification).
        """
        num_games = game_states.shape[0]
        game_types_per_game = game_roles.game_types  # [num_games]
        agent_game_assignment = game_roles.agent_game_assignment  # [num_agents]
        agent_role_assignment = game_roles.agent_role_assignment  # [num_agents]
        game_role_to_agent = game_roles.game_role_to_agent  # [num_games, NUM_ROLES]
        partner_agent = game_roles.partner_agent  # [num_agents]
        agent_game_types = game_types_per_game[agent_game_assignment]  # [num_agents]
        # Agent i's estimate of its partner is the estimate ABOUT the partner (partner's row).
        estimate_of_partner = estimated_agent_belief_states[partner_agent]  # [num_agents, S]

        action_key, transition_key, obs_key = jax.random.split(key, 3)

        # 1. Each agent samples an action from its Boltzmann policy at (b_e, b̄_o).
        #    vmap because the bundle dispatches on a scalar (game_type, role).
        def sample_action(game_type, role, belief_probs, estimate_probs, k):
            policy = self.policy_bundle(game_type, role, distrax.Categorical(probs=belief_probs), distrax.Categorical(probs=estimate_probs))
            return policy.sample(seed=k)

        agent_actions = jax.vmap(sample_action)(
            agent_game_types, agent_role_assignment, true_agent_belief_states, estimate_of_partner, jax.random.split(action_key, self.num_agents)
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

        # 5. Per-agent update of all three tracked levels from the observation.
        next_true, next_estimate_of_partner, next_second_order = jax.vmap(
            self._agent_belief_updates, in_axes=(0, 0, 0, 0, 0, 0, 0, None)
        )(
            agent_game_types,
            agent_role_assignment,
            true_agent_belief_states,
            estimate_of_partner,
            second_order_agent_belief_estimates,  # row i is held by agent i: no gather
            agent_observations,
            agent_actions,
            underlying_env_timestep == 0,  # shared by every agent: all games start together
        )
        # next_estimate_of_partner[i] is agent i's refreshed estimate of its partner; store
        # it subject-indexed -- the estimate ABOUT agent k lives at k, held by k's partner,
        # which is again a gather by partner_agent (the map is its own inverse in a dyad).
        next_estimated = next_estimate_of_partner[partner_agent]  # [num_agents, S]
        return (
            game_next_states,
            next_true,
            next_estimated,
            next_second_order,
            agent_rewards,
            agent_actions,
            agent_observations,
        )

    def step_env(
        self,
        key: chex.PRNGKey,
        state: StackedSignificationState,
        utterance_actions: chex.Array,
        speaker_other_belief_estimate_actions: chex.Array,
        speaker_second_order_ego_estimate_actions: chex.Array,
        listener_ego_belief_actions: chex.Array,
        listener_other_belief_estimate_actions: chex.Array,
        listener_second_order_ego_estimate_actions: chex.Array,
    ):
        """
        This method has many important jobs, but the main thing it does is abstract away the complexity
        of the underlying DecPOMDPs the agents are engaged in. This method is a transition function for
        the abstract communication game.

        At each time step, agents are either 1) generating utterances to send to interlocutors
        based on their belief about the state of the DecPOMDP and their estimate of their
        partner's belief state. Or 2) updating their belief given their prior belief and the
        utterance their partner sent to them.

        In order to abstract away the underlying machinery of the DecPOMDP, additional things need
        to happen after belief stages. Specifically, 1) the underlying DecPOMDP needs to actually advance
        according to the new belief states of the agents (by executing a policy conditioned on a belief
        state), and 2) the agents beliefs need to update in accordance with their perceived env observation
        from env interaction, and 3) the agents' estimates about their partners' beliefs (and of their
        partners' estimates of them) need to update in accordance with their perceived env observation.

        After all this happens, a new utterance stage begins again.

        The stage-specific work lives in two methods; this one dispatches to whichever stage
        ``state`` is in, then does what both stages share (advancing the communication
        schedule, building the observation):

            _step_env_utterance_stage  speakers have just emitted utterances: deliver them to
                                       the listeners, and record each speaker's refreshed
                                       estimate of its listener (b̄_o) and of what its
                                       listener believes about it (b̿_e). The speaker's own
                                       belief (b_e) does not change from uttering.
            _step_env_belief_stage     listeners have just emitted belief states: adopt them
                                       as their true beliefs (b_e), and record each
                                       listener's refreshed estimate of its speaker (b̄_o)
                                       and of what its speaker believes about it (b̿_e).
                                       If the round is the block's last, also step the
                                       underlying DecPOMDPs (the act) and, at the horizon,
                                       begin a new episode.

        The stage is traced state (the rollout scans one stage per call), so the dispatch is a
        lax.cond, and both stages return the same structure. Each call therefore takes all
        six action inputs; the stage that is not running ignores the other stage's inputs.

        Args:
            key: a PRNGKey.
            state: a StackedSignificationState.
            utterance_actions: [num_agents, utterance_action_dim], AGENT-indexed (row i is
                agent i's utterance). Read on the utterance stage only.
            speaker_other_belief_estimate_actions: [num_agents, S], SUBJECT-indexed (row i is
                the estimate ABOUT agent i's belief state). Each speaker's new b̄_o: the belief
                it thinks its listener will have after hearing its utterance, so the rows read
                are the LISTENERS'. Read on the utterance stage only.
            speaker_second_order_ego_estimate_actions: [num_agents, S], AGENT-indexed, like
                the stored b̿_e (row i is held by and about agent i). Each speaker's new b̿_e:
                what it thinks its listener will believe about the speaker after hearing the
                utterance. Only the speakers' rows are read. Read on the utterance stage only.
            listener_ego_belief_actions: [num_agents, S], AGENT-indexed (row i is agent i's proposed new
                belief b_e after hearing the utterance, a distribution over its game's S world
                states). Only the listeners' rows are read. Read on the belief stage only.
            listener_other_belief_estimate_actions: [num_agents, S], SUBJECT-indexed (row i is the
                estimate ABOUT agent i's belief state). Each listener's new b̄_o, its estimate
                of its speaker's belief after hearing the utterance, so the rows read are the
                SPEAKERS'. Read on the belief stage only.
            listener_second_order_ego_estimate_actions: [num_agents, S], AGENT-indexed, like the stored
                b̿_e (row i is held by and about agent i). Each listener's new b̿_e: its
                estimate of what its speaker now believes about the listener, after hearing
                the utterance. Only the listeners' rows are read. Read on the belief stage only.

        Returns (state, observation, rewards, post_comm_true_beliefs,
        post_comm_estimated_beliefs). The last two are debug outputs: the true / estimated
        belief states after communication resolves but before the world steps forward. They are
        NaN-valued unless this step is an act phase (the belief stage of a block's last
        round). Callers that don't need them discard them with ``_``.
        """
        act_key, boundary_key, obs_key = jax.random.split(key, 3)
        on_belief_stage = state.dialog.communicative_round_stage == BELIEF_STAGE

        next_state, acted, post_comm_true_beliefs, post_comm_estimated_beliefs = jax.lax.cond(
            on_belief_stage,
            lambda _: self._step_env_belief_stage(
                act_key,
                boundary_key,
                state,
                listener_ego_belief_actions,
                listener_other_belief_estimate_actions,
                listener_second_order_ego_estimate_actions,
            ),
            lambda _: self._step_env_utterance_stage(
                state, utterance_actions, speaker_other_belief_estimate_actions, speaker_second_order_ego_estimate_actions
            ),
            operand=None,
        )

        # === Advance the communication schedule (both stages) ===============
        # A round completes on its belief stage, so the round cursor advances only then,
        # wrapping to 0 after the block's last round -- exactly the act. On an act, fetch the
        # NEXT block, keyed by the updated cumulative_env_timestep; otherwise keep walking the
        # current one. communication_scheme_fn is a pure, shape-stable gather, so compute it
        # unconditionally and select with where.
        dialog = state.dialog
        next_round_cursor = (dialog.communication_round_iterator + on_belief_stage) % dialog.active_total_num_rounds
        block_after_act = self.communication_scheme_fn(next_state.game_counters.cumulative_env_timestep)
        next_state = next_state.replace(
            dialog=DialogState(
                active_who_speaks=jnp.where(acted, block_after_act.who_speaks, dialog.active_who_speaks),
                active_total_num_rounds=jnp.where(acted, block_after_act.total_num_rounds, dialog.active_total_num_rounds),
                communication_round_iterator=next_round_cursor,
                cumulative_stage_count=dialog.cumulative_stage_count + 1,
                communicative_round_stage=jnp.where(on_belief_stage, UTTERANCE_STAGE, BELIEF_STAGE),
            )
        )

        obs = self.get_obs(obs_key, next_state)
        return next_state, obs, next_state.last_agent_rewards, post_comm_true_beliefs, post_comm_estimated_beliefs

    def _step_env_utterance_stage(self, state, utterance_actions, other_belief_estimate_actions, second_order_ego_estimate_actions):
        """The utterance stage: speakers have just emitted utterances.

        Delivers each speaker's utterance to its listener, and stores each speaker's
        refreshed estimate of its listener's belief (b̄_o) and of what its listener believes
        about it (b̿_e). True beliefs (b_e) are unchanged -- uttering is not evidence to the
        speaker about the world -- nothing world-side moves, and there is never an act.

        Returns (state, acted=False, post_comm_true_beliefs, post_comm_estimated_beliefs),
        the debug beliefs being NaN (no act); the dialog is advanced by step_env.
        """
        # This round's speakers and listeners, per agent. who_speaks is over ROLES; in a dyad
        # the listener is whoever holds the other role.
        who_speaks_this_round = state.dialog.active_who_speaks[state.dialog.communication_round_iterator]  # [num_roles]
        roles = state.game_roles.agent_role_assignment  # [num_agents]
        speaks = who_speaks_this_round[roles].astype(bool)
        listens = who_speaks_this_round[1 - roles].astype(bool)

        # Utterances are FRESHLY POPULATED: the speakers' rows are delivered to the listeners
        # (via state; get_obs routes each to the partner). Non-speakers' rows are NaN.
        utterances_to_deliver = jnp.where(speaks[:, None], utterance_actions, jnp.nan)

        # Speaker's estimate of its listener after uttering. state.estimated_agent_belief_states
        # is SUBJECT-indexed (row k = the estimate ABOUT agent k), and
        # other_belief_estimate_actions shares that indexing, so the rows that change are the
        # LISTENERS' and we write straight across.
        estimated_beliefs = jnp.where(listens[:, None], other_belief_estimate_actions, state.estimated_agent_belief_states)

        # Speaker's second-order estimate: what it thinks its listener now believes about it.
        # b̿_e is stored by holder (row i is held by and about agent i), so the rows that
        # change are the SPEAKERS'.
        second_order_estimates = jnp.where(
            speaks[:, None], second_order_ego_estimate_actions, state.second_order_agent_belief_estimates
        )

        # No act, so the act's outputs are sentinels: NaN rewards (placeholders, to be filled in
        # retroactively once communication finishes) and -1 actions/observations (int32).
        next_state = state.replace(
            agent_utterance_actions_unrendered=utterances_to_deliver,
            estimated_agent_belief_states=estimated_beliefs,
            second_order_agent_belief_estimates=second_order_estimates,
            last_agent_rewards=jnp.full((self.num_agents,), jnp.nan, dtype=jnp.float32),
            last_agent_actions=-jnp.ones((self.num_agents,), dtype=jnp.int32),
            last_agent_observations=-jnp.ones((self.num_agents,), dtype=jnp.int32),
        )
        nan_beliefs = jnp.full_like(state.true_agent_belief_states, jnp.nan)
        return next_state, jnp.asarray(False), nan_beliefs, nan_beliefs

    def _step_env_belief_stage(
        self, act_key, boundary_key, state, ego_belief_actions, other_belief_estimate_actions, second_order_ego_estimate_actions
    ):
        """The belief stage: listeners have just emitted belief states.

        Listeners adopt their emitted beliefs as their true beliefs (b_e), their emitted
        estimates of their speakers' beliefs as their b̄_o, and their emitted estimates of
        what their speakers believe about them as their b̿_e. If this round is the block's
        last, every game then takes one DecPOMDP step (the act), and if that act reaches the
        episode horizon a new episode begins.

        Returns (state, acted, post_comm_true_beliefs, post_comm_estimated_beliefs); the
        debug beliefs are the beliefs the act ran on, NaN when there was no act. The dialog
        is advanced by step_env.
        """
        # This round's speakers and listeners, per agent. who_speaks is over ROLES; in a dyad
        # the listener is whoever holds the other role.
        who_speaks_this_round = state.dialog.active_who_speaks[state.dialog.communication_round_iterator]  # [num_roles]
        roles = state.game_roles.agent_role_assignment  # [num_agents]
        speaks = who_speaks_this_round[roles].astype(bool)
        listens = who_speaks_this_round[1 - roles].astype(bool)

        # === 1. Schedule decisions ==========================================
        # When it's the belief stage of the final round of communication, the underlying
        # environment must transition forward with agent actions. Env timesteps advance only
        # on an act; an act that brings underlying_env_timestep to the episode horizon ends
        # the episode.
        acted = state.dialog.communication_round_iterator == state.dialog.active_total_num_rounds - 1
        act_increment = acted.astype(jnp.int32)
        incremented_cumulative_env_timestep = state.game_counters.cumulative_env_timestep + act_increment
        incremented_underlying_env_timestep = state.game_counters.underlying_env_timestep + act_increment
        is_boundary = acted & (incremented_underlying_env_timestep >= state.game_counters.episode_horizon)

        # === 2. Communication ===============================================
        # Listeners adopt ego_belief_actions as their new ground-truth belief states.
        # NOTE: Adopted verbatim, so exact zeros survive. If a listener's belief has zero mass on every
        # state consistent with its next observation, the observation update normalizes 0/0 and the
        # true belief becomes all-NaN (the tripwire).
        post_comm_true_beliefs = jnp.where(listens[:, None], ego_belief_actions, state.true_agent_belief_states)

        # Listener's estimate of its speaker after listening. state.estimated_agent_belief_states
        # is SUBJECT-indexed (row k = the estimate ABOUT agent k), and
        # other_belief_estimate_actions shares that indexing, so the rows that change are the
        # SPEAKERS' and we write straight across. (The listeners' rows, the speakers' estimates
        # of them, were written on the utterance stage.)
        post_comm_estimated_beliefs = jnp.where(speaks[:, None], other_belief_estimate_actions, state.estimated_agent_belief_states)

        # Listener's second-order estimate: what it now thinks its speaker believes about it.
        # b̿_e is stored by holder (row i is held by and about agent i), so the rows that
        # change are the LISTENERS'. (The speakers' rows were written on the utterance stage.)
        post_comm_second_order_estimates = jnp.where(
            listens[:, None], second_order_ego_estimate_actions, state.second_order_agent_belief_estimates
        )

        # === 3. World step (the act) ========================================
        # On an act, every game takes one DecPOMDP step: each agent samples an action from
        # its Boltzmann policy at its post-communication (b_e, b̄_o), the games transition
        # and emit observations, and beliefs update from those observations. The step also
        # yields each agent's real reward from the environment, which becomes the communication game's
        # reward. Without an act, nothing world-side moves and the outputs are the no-act
        # sentinels.
        def step_world(_):
            return self._step_underlying_env(
                act_key,
                state.game_states,
                state.game_roles,
                post_comm_true_beliefs,
                post_comm_estimated_beliefs,
                post_comm_second_order_estimates,
                state.game_counters.underlying_env_timestep,
            )

        def hold_world(_):
            # No-act sentinels: NaN reward (a placeholder to be filled in retroactively once
            # communication finishes); actions/observations are int32, so they use -1.
            return (
                state.game_states,
                post_comm_true_beliefs,
                post_comm_estimated_beliefs,
                post_comm_second_order_estimates,
                jnp.full((self.num_agents,), jnp.nan, dtype=jnp.float32),   # Rewards (nan)
                -jnp.ones((self.num_agents,), dtype=jnp.int32),             # Actions (-1)
                -jnp.ones((self.num_agents,), dtype=jnp.int32),             # Observations (-1)
            )

        (
            post_world_game_states,
            post_world_true_beliefs,
            post_world_estimated_beliefs,
            post_world_second_order_estimates,
            agent_rewards,
            agent_actions,
            agent_observations,
        ) = jax.lax.cond(acted, step_world, hold_world, operand=None)

        # === 4. Episode boundary ============================================
        # On a boundary (decided in section 1), re-route a fresh assignment (keyed by the
        # next episode index), resample initial world states + beliefs, reset
        # underlying_env_timestep to 0 and bump episode_index. cumulative_env_timestep keeps
        # counting every real env step. This step's reported reward/action/observation stays
        # the ENDING episode's final act (section 3).
        def begin_new_episode(_):
            new_episode_index = state.game_counters.episode_index + 1
            init = self._begin_episode(boundary_key, new_episode_index)
            return (
                init["game_roles"],
                GameCountersState(
                    underlying_env_timestep=jnp.asarray(0, dtype=jnp.int32),
                    cumulative_env_timestep=incremented_cumulative_env_timestep,
                    episode_index=new_episode_index,
                    episode_horizon=init["episode_horizon"],
                ),
                init["game_states"],
                init["true_agent_belief_states"],
                init["estimated_agent_belief_states"],
                init["second_order_agent_belief_estimates"],
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
                post_world_second_order_estimates,
            )

        (
            next_game_roles,
            next_game_counters,
            next_game_states,
            next_true_beliefs,
            next_estimated_beliefs,
            next_second_order_estimates,
        ) = jax.lax.cond(is_boundary, begin_new_episode, continue_episode, operand=None)

        next_state = state.replace(
            game_roles=next_game_roles,
            game_counters=next_game_counters,
            # No speaker on a belief stage: every utterance slot is NaN.
            agent_utterance_actions_unrendered=jnp.full_like(state.agent_utterance_actions_unrendered, jnp.nan),
            game_states=next_game_states,
            true_agent_belief_states=next_true_beliefs,
            estimated_agent_belief_states=next_estimated_beliefs,
            second_order_agent_belief_estimates=next_second_order_estimates,
            last_agent_rewards=agent_rewards,
            last_agent_actions=agent_actions,
            last_agent_observations=agent_observations,
        )
        return (
            next_state,
            acted,
            jnp.where(acted, post_comm_true_beliefs, jnp.nan),
            jnp.where(acted, post_comm_estimated_beliefs, jnp.nan),
        )

    def _begin_episode(self, key, episode_index):
        """Route a fresh episode and produce its initial fields.

        Assigns agents to games/roles via assignment_fn, samples each game's
        initial world state, and sets each agent's initial belief, the initial estimate
        about it, and its initial second-order estimate -- all equal to its game's
        prior -- from the env params.
        """
        routing_key, state_key, key = jax.random.split(key, 3)
        route = self.assignment_fn(key=routing_key, episode_index=episode_index)

        game_roles = GameRoleState.from_assignment(route.agent_game_assignment, route.agent_role_assignment, route.game_set)
        game_types_per_game = game_roles.game_types  # [num_games]
        agent_game_types = game_types_per_game[game_roles.agent_game_assignment]  # [num_agents]

        # Each agent's initial belief IS its game's world prior -- every agent, every role.
        # (There is no per-role initial belief; see FlexibleEnvParams.)
        # NOTE: This is subject to change in the future, we may want to seed agent priors.
        agent_initial_belief_states = self.all_env_parameters.initial_state_distribution[agent_game_types]

        # Sample each game's true initial world state from its initial-state dist.
        num_games = game_types_per_game.shape[0]
        init_state_dists = self.all_env_parameters.initial_state_distribution[game_types_per_game]
        game_states = jax.vmap(lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k))(
            init_state_dists, jax.random.split(state_key, num_games)
        )

        true_agent_belief_states = agent_initial_belief_states
        estimated_agent_belief_states = agent_initial_belief_states     # NOTE: This is also subject to change, we may want to seed priors on estimates of belief states
        second_order_agent_belief_estimates = agent_initial_belief_states

        return dict(
            game_roles=game_roles,
            game_states=game_states,
            true_agent_belief_states=true_agent_belief_states,
            estimated_agent_belief_states=estimated_agent_belief_states,
            second_order_agent_belief_estimates=second_order_agent_belief_estimates,
            episode_horizon=route.underlying_env_steps_per_episode,
            key=key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """Returns (state, observation); the observation is get_obs(state).
        Resetting only happens once. The env continues on forever according to the routing function.

        """

        # _begin_episode routes the episode and returns its initial world states and beliefs.
        init = self._begin_episode(key, jnp.asarray(0, dtype=jnp.int32))
        key = init["key"]
        game_states = init["game_states"]
        true_agent_belief_states = init["true_agent_belief_states"]
        estimated_agent_belief_states = init["estimated_agent_belief_states"]
        # No act yet, so the no-act sentinels (NaN reward, -1 action/observation).
        last_agent_rewards = jnp.full((self.num_agents,), jnp.nan, dtype=jnp.float32)
        last_agent_actions = -jnp.ones((self.num_agents,), dtype=jnp.int32)
        last_agent_observations = -jnp.ones((self.num_agents,), dtype=jnp.int32)

        # At reset no env step has happened: the per-episode and cumulative env timesteps
        # are both 0.
        underlying_timestep = jnp.asarray(0, dtype=jnp.int32)

        # Fetch the first active communication block, keyed by cumulative_env_timestep
        # (== underlying_timestep at reset), and store it flat. The round cursors start
        # at the block's first round; no communication rounds have run yet.
        active_scheme = self.communication_scheme_fn(underlying_timestep)

        state = StackedSignificationState(
            dialog=DialogState(
                active_who_speaks=active_scheme.who_speaks,
                active_total_num_rounds=active_scheme.total_num_rounds,
                communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
                cumulative_stage_count=jnp.asarray(0, dtype=jnp.int32),
                # Reset starts on the utterance stage (the speaker emits first).
                communicative_round_stage=jnp.asarray(UTTERANCE_STAGE, dtype=jnp.int32),
            ),
            game_roles=init["game_roles"],
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
            second_order_agent_belief_estimates=init["second_order_agent_belief_estimates"],
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
        only_games: Names of the game types to assemble, in order -- game type index i is
            ``only_games[i]``. Empty means "every registered game" (in ``GAME_SPECS`` order).
            Names must be keys of ``envs.env_assembly.GAME_SPECS``.
    """

    utterance_action_dim: int = 12  # 2 splines x 6 control points; multiple of 6 to render
    utterance_image_dim: int = 32
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
        stacked_env_parameters, policy_bundle = assemble_environments(specs)

        return StackedSignificationDecPOMDP(
            num_agents=num_agents,
            all_env_parameters=stacked_env_parameters,
            policy_bundle=policy_bundle,
            assignment_fn=assignment_fn,
            communication_scheme_fn=communication_scheme_fn,
            utterance_action_dim=self.utterance_action_dim,
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

    stacked_params, policy_bundle = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        policy_bundle=policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=a_to_b_thrice_scheme_fn,
        utterance_action_dim=3,
    )

    reset_key, walk_key = jax.random.split(key)
    state, _ = env.reset(reset_key)

    # Per-agent actions fed to step_env each call: an all-ones utterance, a one-hot
    # belief_action (must be a valid distribution -- the act samples an action from it),
    # and a valid post-utterance estimate (skew the reset belief so it stays full-support;
    # a padding-state one-hot would make the act's observation update degenerate).
    utterance_actions = jnp.ones((env.num_agents, env.utterance_action_dim))
    num_states = state.true_agent_belief_states.shape[-1]
    listener_ego_belief_actions = jnp.zeros((env.num_agents, num_states)).at[:, 0].set(1.0)
    belief_estimate = state.true_agent_belief_states.at[:, 0].add(0.5)
    belief_estimate = belief_estimate / belief_estimate.sum(axis=-1, keepdims=True)

    def fmt(st):
        return (
            f"stage={int(st.dialog.communicative_round_stage)} "
            f"round={int(st.dialog.communication_round_iterator)} "
            f"env_iter={int(st.game_counters.underlying_env_timestep)} "
            f"stages={int(st.dialog.cumulative_stage_count)}"
        )

    print("=== a_to_b_thrice example (1 block = 3 rounds = 6 step_env calls) ===")
    print("  reset: ", fmt(state))
    for t in range(6):
        walk_key, step_key = jax.random.split(walk_key)
        state, _, rewards, _, _ = env.step_env(
            step_key, state, utterance_actions, belief_estimate, belief_estimate, listener_ego_belief_actions, belief_estimate, belief_estimate
        )
        print(f"  step {t}:", fmt(state), "reward_sum=", float(rewards.sum()))
    print("ok: a_to_b_thrice walks 3 rounds and acts once on the last belief stage")
    return env, state


if __name__ == "__main__":
    a_to_b_thrice_example(jax.random.key(0))

    from communication.game_role_assignment import simple_assignment_fn
    from communication.communication_scheme import b_to_a_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    # Build the stacked params + policy table from one game type (the guessing game).
    stacked_params, policy_bundle = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        policy_bundle=policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
    )

    # reset: no env step yet, so beliefs stay at initial.
    state, obs = env.reset(jax.random.key(0))
    obs_beliefs, obs_estimated, obs_second_order, obs_utterances = obs
    # Reset is on the utterance stage: beliefs are present, utterances NaN'd.
    assert jnp.all(jnp.isnan(obs_utterances)), "utterance-stage obs NaNs the utterances"
    assert not jnp.any(jnp.isnan(obs_beliefs)), "utterance-stage obs keeps beliefs"
    assert not jnp.any(jnp.isnan(obs_estimated)), "utterance-stage obs keeps estimated beliefs"
    assert not jnp.any(jnp.isnan(obs_second_order)), "obs always keeps second-order estimates"
    print("=== reset ===")
    print("game_states:        ", state.game_states)
    print("underlying_iter:    ", state.game_counters.underlying_env_timestep)
    print("active who_speaks:  ", state.dialog.active_who_speaks.tolist())
    print("active rounds:      ", state.dialog.active_total_num_rounds)
    print(
        "round iter, stages: ",
        state.dialog.communication_round_iterator,
        state.dialog.cumulative_stage_count,
    )
    print("round stage:        ", state.dialog.communicative_round_stage)
    print("true beliefs[0]:    ", state.true_agent_belief_states[0])
    assert state.game_counters.underlying_env_timestep == 0, "reset takes no env step"
    assert state.dialog.communication_round_iterator == 0, "round cursor starts at 0"
    assert state.dialog.communicative_round_stage == UTTERANCE_STAGE, "reset starts on the utterance stage"

    # Direct dispatch check: beliefs are routed through the (game_type, role) Q function
    # and its Boltzmann policy. The presser's Q is the guessing game's presser Q; the
    # observer's action is inert, so its policy is uniform.
    from envs.flexible_env import boltzmann_policy

    guessing_game = guessing_game_spec()
    ego_belief = distrax.Categorical(probs=jnp.array([0.6, 0.3, 0.1, 0.0]))
    other_belief_estimate = distrax.Categorical(probs=jnp.array([0.2, 0.2, 0.6, 0.0]))
    role_0_probs = env.policy_bundle(0, 0, ego_belief, other_belief_estimate).probs
    role_1_probs = env.policy_bundle(0, 1, ego_belief, other_belief_estimate).probs
    print("role 0 action probs:", role_0_probs)
    print("role 1 action probs:", role_1_probs)
    expected_role_0 = boltzmann_policy(guessing_game.q_functions[0](ego_belief, other_belief_estimate), guessing_game.inverse_temperature).probs
    assert jnp.allclose(role_0_probs, expected_role_0)
    assert jnp.allclose(role_1_probs, 0.25)
    print("ok: env built via factory; policies dispatch by (game_type, role)")

    # Schedule decisions (step_env + _step_env_belief_stage section 1). Use a 3-round block (a_to_b_thrice) so
    # the round cursor visibly walks 0,0,1,1,2,2 (two stages each) and the env timestep
    # advances exactly once per completed block (every 6 step_env calls).
    from communication.communication_scheme import a_to_b_thrice_scheme_fn

    sched_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        policy_bundle=policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, game_type_id=0),
        communication_scheme_fn=a_to_b_thrice_scheme_fn,
        utterance_action_dim=3,
    )
    s0, _ = sched_env.reset(jax.random.key(2))
    utt = jnp.ones((10, sched_env.utterance_action_dim))  # every agent "says" all-ones
    num_states = s0.true_agent_belief_states.shape[-1]
    # listener_ego_belief_actions are beliefs over states the listeners adopt, so they must be valid
    # distributions. Give them full support over the referents: a certain-but-wrong belief
    # (e.g. a one-hot) makes the next observation impossible, and the observation update
    # NaNs by design.
    valid_belief = jnp.tile(jnp.array([0.5, 0.25, 0.25, 0.0]), (10, 1))
    # Post-utterance belief estimate, subject-indexed like the state field. Must be a
    # *valid* belief (full support over real states) because the act marginalizes the
    # partner's action through it; a one-hot on the padding state would make the
    # observation update degenerate. Skew the reset belief so it is distinct enough for
    # _step_env_utterance_stage's estimate wiring to be visible.
    other_est = s0.true_agent_belief_states.at[:, 0].add(0.5)
    other_est = other_est / other_est.sum(axis=-1, keepdims=True)
    # A second, different valid estimate for the listeners' estimates of their speakers.
    speaker_est = s0.true_agent_belief_states.at[:, 1].add(0.5)
    speaker_est = speaker_est / speaker_est.sum(axis=-1, keepdims=True)
    # A third, for the listeners' estimates of what their speakers believe about them (b̿_e).
    self_est = s0.true_agent_belief_states.at[:, 2].add(0.5)
    self_est = self_est / self_est.sum(axis=-1, keepdims=True)
    # A fourth, for the speakers' estimates of what their listeners believe about them (b̿_e).
    speaker_self_est = s0.true_agent_belief_states.at[:, 0].add(1.5)
    speaker_self_est = speaker_self_est / speaker_self_est.sum(axis=-1, keepdims=True)
    role0 = s0.game_roles.agent_role_assignment == 0
    role1 = ~role0

    # Section 2a: the utterance stage stashes ONLY the speaking role's (role 0 in a_to_b)
    # utterances; listeners (role 1) are NaN'd.
    after_utt, after_utt_obs, after_utt_rewards, _, _ = sched_env.step_env(
        jax.random.key(0), s0, utt, other_est, speaker_self_est, valid_belief, speaker_est, self_est
    )
    # A communication-only step (the utterance stage) yields a NaN (placeholder) reward.
    assert jnp.all(jnp.isnan(after_utt_rewards)), "utterance-stage step returns NaN reward"
    assert jnp.array_equal(after_utt_rewards, after_utt.last_agent_rewards, equal_nan=True), "returned reward matches state"
    assert jnp.all(after_utt.agent_utterance_actions_unrendered[role0] == 1.0)
    assert jnp.all(jnp.isnan(after_utt.agent_utterance_actions_unrendered[role1]))
    print("ok: utterance stage stashes speakers' (role 0) utterances, NaNs listeners")

    # get_obs: after one step we are on the belief stage. The belief groups are the current
    # values (the listener updates them), and listeners now hear their speaker's utterance.
    ou_beliefs, ou_estimated, ou_second_order, ou_utterances = after_utt_obs
    partner = s0.game_roles.partner_agent
    assert jnp.all(ou_beliefs == after_utt.true_agent_belief_states), "belief-stage obs has each agent's own belief"
    assert jnp.all(ou_estimated == after_utt.estimated_agent_belief_states[partner]), "...its estimate of its partner"
    assert jnp.all(ou_second_order == after_utt.second_order_agent_belief_estimates), "...and its second-order estimate"
    assert not jnp.any(jnp.isnan(ou_utterances[role1])), "belief-stage obs has the listeners' utterances"
    print("ok: belief-stage observation has all belief levels and the listeners' utterances")

    # get_obs indexing: utterances are remapped to the PARTNER (speaker), not the agent's
    # own row. role 0 spoke ones, role 1 was NaN'd; so each listener (role 1) HEARS its
    # role-0 partner's ones, and each speaker (role 0) hears its role-1 partner's NaNs.
    assert jnp.all(ou_utterances[role1] == 1.0), "listeners hear their partner-speaker's utterance"
    assert jnp.all(jnp.isnan(ou_utterances[role0])), "speakers hear their (silent listener) partner"
    print("ok: get_obs routes each agent its partner's utterance, not its own")

    # get_obs indexing: on the utterance stage, estimated_beliefs is the estimate of the
    # RECEIVER (partner) the sender is engaged with -- the partner's subject-indexed row.
    us_beliefs, us_estimated, us_second_order, us_utterances = sched_env.get_obs(jax.random.key(0), s0)
    assert jnp.all(jnp.isnan(us_utterances)), "utterance-stage obs has no utterance to hear"
    assert jnp.all(us_beliefs == s0.true_agent_belief_states), "beliefs[i] is agent i's own belief"
    assert jnp.all(
        us_estimated == s0.estimated_agent_belief_states[partner]
    ), "estimated_beliefs[i] is the estimate of agent i's partner (the receiver)"
    assert jnp.all(us_second_order == s0.second_order_agent_belief_estimates), "second_order_estimates[i] is agent i's own row"
    print("ok: get_obs routes each sender the estimate of its receiver partner")

    # Section 2b: the estimate is subject-indexed, so the LISTENERS' rows (role 1, the agents
    # a speaker just re-estimated) take the supplied value; the speakers' rows (role 0)
    # keep their prior estimate.
    assert jnp.all(after_utt.estimated_agent_belief_states[role1] == other_est[role1])
    assert jnp.all(after_utt.estimated_agent_belief_states[role0] == s0.estimated_agent_belief_states[role0])
    print("ok: utterance stage records the listener-subject (role 1) post-utterance estimate")

    # ...and each speaker's new b̿_e, stored by holder: the SPEAKERS' rows (role 0) take it,
    # the listeners' rows are unchanged, and nobody's true belief moves.
    assert jnp.all(after_utt.second_order_agent_belief_estimates[role0] == speaker_self_est[role0])
    assert jnp.all(after_utt.second_order_agent_belief_estimates[role1] == s0.second_order_agent_belief_estimates[role1])
    assert jnp.all(after_utt.true_agent_belief_states == s0.true_agent_belief_states)
    print("ok: utterance stage records the speakers' (role 0) second-order estimates; true beliefs unchanged")

    # The following belief stage has the listeners (role 1) adopt their proposed
    # listener_ego_belief_actions; the speakers (role 0) keep their belief.
    after_belief, _, _, _, _ = sched_env.step_env(
        jax.random.key(1), after_utt, utt, other_est, speaker_self_est, valid_belief, speaker_est, self_est
    )
    assert jnp.all(after_belief.true_agent_belief_states[role1] == valid_belief[role1])
    assert jnp.all(after_belief.true_agent_belief_states[role0] == s0.true_agent_belief_states[role0])
    print("ok: belief stage applies listeners' (role 1) listener_ego_belief_actions, speakers unchanged")

    # ...and records each listener's estimate of its speaker: subject-indexed, so the
    # SPEAKERS' rows (role 0) take listener_other_belief_estimate_actions, while the listeners' rows
    # keep what the utterance stage wrote.
    assert jnp.all(after_belief.estimated_agent_belief_states[role0] == speaker_est[role0])
    assert jnp.all(after_belief.estimated_agent_belief_states[role1] == other_est[role1])
    print("ok: belief stage records the speaker-subject (role 0) post-listening estimate")

    # ...and each listener's new b̿_e. b̿_e is stored by holder, so the LISTENERS' rows
    # (role 1) take listener_second_order_ego_estimate_actions; the speakers' (role 0) keep what the
    # utterance stage wrote.
    assert jnp.all(after_belief.second_order_agent_belief_estimates[role1] == self_est[role1])
    assert jnp.all(after_belief.second_order_agent_belief_estimates[role0] == speaker_self_est[role0])
    print("ok: belief stage records the listeners' (role 1) second-order estimates, speakers' unchanged")

    def fmt(st):
        return (
            f"stage={int(st.dialog.communicative_round_stage)} "
            f"round={int(st.dialog.communication_round_iterator)} "
            f"env_iter={int(st.game_counters.underlying_env_timestep)} "
            f"stages={int(st.dialog.cumulative_stage_count)}"
        )

    print("=== step_env scheduling walk (a_to_b_thrice: 3 rounds/block) ===")
    s = s0
    print("  reset:  ", fmt(s))
    for t in range(6):
        s, _, step_rewards, _, _ = sched_env.step_env(
            jax.random.key(t), s, utt, other_est, speaker_self_est, valid_belief, speaker_est, self_est
        )
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
        policy_bundle=policy_bundle,
        assignment_fn=simple_assignment_fn(num_agents=10, underlying_env_steps_per_episode=2),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
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
        b, _, _, _, _ = boundary_env.step_env(jax.random.key(100 + t), b, utt3, other_est3, other_est3, valid3, other_est3, other_est3)
        print(
            f"  step {t}: env_iter={int(b.game_counters.underlying_env_timestep)} "
            f"cum_env={int(b.game_counters.cumulative_env_timestep)} episode={int(b.game_counters.episode_index)}"
        )
    assert int(b.game_counters.episode_index) == 1, "a new episode began at the horizon"
    assert int(b.game_counters.underlying_env_timestep) == 0, "per-episode env iter reset at the boundary"
    assert int(b.game_counters.cumulative_env_timestep) == 2, "cumulative env iter keeps counting"
    print("ok: episode boundary re-routes and resets the per-episode counters")
