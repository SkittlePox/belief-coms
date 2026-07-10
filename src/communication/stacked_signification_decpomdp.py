import jax, chex
import jax.numpy as jnp
import distrax
from flax import struct
from typing import Any, Callable, Sequence
from functools import partial
from training.routing import RouteFn
from training.communication_scheme import CommunicationSchemeFn

# FlexibleEnvParams / OptimalPolicy are defined in envs.flexible_env (the leaf of
# the env-definition dependency graph) and consumed here.
from envs.flexible_env import FlexibleEnvParams, OptimalPolicy
from tools.belief_representations import CategoricalBeliefState


# A communicative round runs in two stages, one per step_env call; the state's
# communicative_round_stage records which stage the current step is in.
UTTERANCE_STAGE = 0  # the speaker emits an utterance / message action
BELIEF_STAGE = 1     # the listener emits a belief action, completing the round


# Iterator Q&A (the four counters in StackedSignificationState)
# -------------------------------------------------------------
# Q: What do "round" and "block" mean?
# A: A communication ROUND is one entry of a communication BLOCK -- a single
#    speaker(s)->listener(s) exchange. It plays out over TWO step_env calls (the
#    UTTERANCE stage then the BELIEF stage; see communicative_round_stage). A BLOCK is
#    the whole CommunicationScheme active that happens *between* two underlying-env steps: a sequence
#    of total_num_rounds rounds. The block's LAST round is the one whose belief stage
#    also steps the underlying DecPOMDP (the "act"). communication_scheme_fn hands back
#    one block per underlying-env step (keyed by cumulative_env_iteration).
#
# Q: Why four iterators?
# A: Two clocks (underlying-DecPOMDP steps vs communication rounds) x two scopes
#    (this episode vs cumulative over the whole run).
#
# Q: underlying_env_iteration vs cumulative_env_iteration?
# A: Both count underlying-DecPOMDP steps (acts). underlying_env_iteration resets to 0
#    at each episode boundary and is compared to the route horizon to detect the end of
#    an episode. cumulative_env_iteration never resets and is the key passed to
#    communication_scheme_fn, so the scheme can change over training.
#
# Q: communication_round_iterator vs cumulative_communication_round_iterator?
# A: communication_round_iterator is the cursor WITHIN the active block
#    (0..total_num_rounds-1); it advances once per COMPLETED round and wraps to 0 when
#    the block's last round acts. cumulative_communication_round_iterator counts
#    step_env calls and never resets. Each round spans TWO step_env calls (UTTERANCE
#    then BELIEF -- see communicative_round_stage), so it increments twice per round.
#
# Q: What changes on a single step_env call?
# A: Always: cumulative_communication_round_iterator += 1 and the stage toggles. On a
#    BELIEF stage the round completes, so communication_round_iterator
#    advances (or wraps); and if that was the block's last round, both env iterations
#    += 1 and the next block is fetched.
#
# Q: And episode_index / episode_horizon (not iterators, but related)?
# A: episode_horizon is this episode's length (the route's underlying_env_steps_per_episode);
#    when underlying_env_iteration reaches it after an act, the episode ends. episode_index
#    counts episodes and keys the re-route (routing_fn(key, episode_index)).
#
# Belief & estimate Q&A (true_agent_belief_states / estimated_agent_belief_states)
# --------------------------------------------------------------------------------
# Q: What is true_agent_belief_states[i]?
# A: Agent i's OWN belief -- a distribution [S] over the world states of the game i is in.
#    Indexed by the agent whose belief it is. Exactly one row per agent.
#
# Q: What is estimated_agent_belief_states[i]?
# A: The estimate of AGENT i's belief -- i.e. what agent i's dyadic partner thinks i
#    believes -- again a distribution [S]. It is indexed the SAME way as
#    true_agent_belief_states: by the SUBJECT (the agent being estimated). So the two rows
#    for agent i are "what i actually believes" and "what i's partner thinks i believes".
#
# Q: Subject-indexed vs estimator-indexed -- why call this out?
# A: The same table ("X's estimate of Y") can be laid out two ways:
#      - subject-indexed:   row i = the estimate ABOUT agent i (this codebase).
#      - estimator-indexed: row i = agent i's estimate of its partner.
#    In a dyad these are just the pair swapped, so they are easy to conflate but put a
#    given number in different rows. We keep estimated_agent_belief_states subject-indexed
#    so it lines up row-for-row with true_agent_belief_states.
#
# Q: If storage is subject-indexed, how does an agent get its OWN estimate of its partner
#    (which the belief update needs)?
# A: By a partner lookup. Agent i's estimate of its partner is the estimate ABOUT the
#    partner, which lives at the PARTNER's row: estimated_agent_belief_states[partner(i)].
#    _step_underlying_env does exactly this gather (via game_role_to_agent) before the
#    update, and scatters the refreshed estimates back by the same partner map afterward.
#
# Q: Why does the estimate feed the underlying act at all?
# A: In the act each agent updates its true belief from ONLY its own observation; the
#    partner's action is unobserved. To marginalize it out, the agent runs the partner's
#    optimal policy on its estimate of the partner's belief (the partner-row lookup
#    above). So a bad estimate (e.g. mass on a padding state) corrupts the true-belief
#    update.
#
# Q: How does the step_env input belief_estimate_post_utterance relate?
# A: It is subject-indexed to match this field: belief_estimate_post_utterance[i] is
#    the refreshed estimate ABOUT agent i once i's partner (the speaker) has uttered. On
#    the utterance stage the LISTENERS' rows are overwritten with it (bit 1b) -- the
#    listener is the subject a speaker just re-estimated -- and speakers' rows are
#    untouched. No remap is needed because both objects are subject-indexed.
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

    # Communication: the agents' utterance actions (raw and rendered). Populated by
    # the communication phase of step_env; None after reset.
    agent_utterance_actions_unrendered: chex.Array
    agent_utterance_actions_rendered: chex.Array

    # Routing: which game / role each agent occupies, and each game's type.
    agent_game_assignment: chex.Array  # [num_agents] -> game index
    agent_role_assignment: chex.Array  # [num_agents] -> role index
    game_types: chex.Array  # [num_games]  -> game-type index

    # Progress: how many underlying-DecPOMDP steps have elapsed this episode (the
    # in-game iteration ``g``). All games advance in lockstep, so this is a single
    # scalar shared across games. Used to index a time-varying communication scheme
    # and to detect the episode horizon. Set at reset (0, or 1 if reset acts first).
    underlying_env_iteration: chex.Array  # scalar int; resets to 0 at each episode boundary
    cumulative_env_iteration: chex.Array  # scalar int; never resets; the key passed to communication_scheme_fn
    episode_index: chex.Array             # scalar int; which episode we are in; +1 at each boundary; keys the re-route
    episode_horizon: chex.Array           # scalar int; this episode's underlying_env_steps_per_episode (from the route)

    # Communication schedule: the active block, stored flat as its who_speaks /
    # total_num_rounds (re-fetched from communication_scheme_fn on each underlying-env
    # step, keyed by cumulative_env_iteration), plus the cursors that walk it. The
    # instantaneous who-speaks for the current round is
    # active_who_speaks[communication_round_iterator].
    active_who_speaks: chex.Array                         # [num_rounds (padded), num_speakers]
    active_total_num_rounds: chex.Array                   # scalar int: real round count of the active block (excludes padding)
    communication_round_iterator: chex.Array   # scalar int: round within the active block (0..total_num_rounds-1); advances once per completed (two-step) round
    cumulative_communication_round_iterator: chex.Array   # scalar int: total step_env calls (two per who_speaks round); never resets

    # Two-stage rounds: each who_speaks round spans TWO step_env iterations. The
    # speaker emits an utterance on UTTERANCE_STAGE, then the listener emits a belief
    # action on BELIEF_STAGE, which completes the round and advances
    # communication_round_iterator.
    communicative_round_stage: chex.Array                 # scalar int: UTTERANCE_STAGE or BELIEF_STAGE

    # World: the true DecPOMDP state of each game.
    game_states: chex.Array  # [num_games]

    # Beliefs: both rows are indexed by the SUBJECT agent (same layout). Row i of
    # true_agent_belief_states is agent i's own belief; row i of
    # estimated_agent_belief_states is what agent i's dyadic partner estimates i believes.
    true_agent_belief_states: chex.Array  # [num_agents, S]
    estimated_agent_belief_states: chex.Array  # [num_agents, S]  (estimate ABOUT agent i)

    # Reward: each agent's underlying-DecPOMDP reward from the most recent act (the
    # stacked game's reward signal). Zero on non-act (communication-only) steps.
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
        optimal_policies: Sequence[Sequence[OptimalPolicy]],
        routing_fn: RouteFn,
        communication_scheme_fn: CommunicationSchemeFn,
        utterance_action_dim: int,
        skip_first_communication_step: bool,
    ) -> None:
        """
        Args:
            num_agents
            all_env_parameters: Stacked FlexibleEnvParams, indexed by game type along the leading axis.
            optimal_policies: Nested table of optimal policies indexed [game_type][role].
                Each entry is an arbitrary callable Categorical(belief) -> Categorical(action).
                Stored separately from all_env_parameters because callables are not
                traceable pytree data and cannot be gathered by a traced index.
            routing_fn
            communication_scheme_fn: Maps the cumulative in-game iteration to the
                CommunicationScheme in force at that iteration (see communication_scheme.py).
            utterance_action_dim: Constant length of each agent's utterance-action vector.
        """
        self.num_agents = num_agents
        self.all_env_parameters = all_env_parameters
        self.routing_fn = routing_fn
        self.communication_scheme_fn = communication_scheme_fn
        self.utterance_action_dim = utterance_action_dim
        self.act_on_reset_before_communicating = skip_first_communication_step

        # Policy table indexed [game_type][role]. Flatten once for lax.switch
        # dispatch: the table is static (Python-level), so the flat list of
        # callables is available at trace time even though the (game_type, role)
        # selection index is traced.
        self.optimal_policies = optimal_policies
        self.num_game_types = len(optimal_policies)
        self.num_roles = len(optimal_policies[0])
        self._flat_policies = tuple(
            optimal_policies[game_type][role]
            for game_type in range(self.num_game_types)
            for role in range(self.num_roles)
        )

        # One belief-update engine per game type, built from that type's (unstacked)
        # params. Cardinalities are uniform across types (padded), so each is valid.
        # Currently the reset belief updates use only type 0 (see note in reset).
        self._belief_factories = tuple(
            CategoricalBeliefState(
                jax.tree.map(lambda leaf, t=game_type: leaf[t], all_env_parameters)
            )
            for game_type in range(self.num_game_types)
        )

    def _agent_policy(
        self, game_type, role, belief_distribution: distrax.Categorical
    ) -> distrax.Categorical:
        """Select and apply the optimal policy for a single (game_type, role).

        `game_type` and `role` may be traced; dispatch goes through lax.switch over
        the flattened [game_type * num_roles + role] index. All policies must accept
        the same belief shape and return the same action-distribution shape (pad /
        mask actions across game types if they differ).
        """
        flat_index = game_type * self.num_roles + role
        return jax.lax.switch(flat_index, self._flat_policies, belief_distribution)

    def agent_action_distributions(
        self, agent_game_types, agent_roles, agent_belief_probs
    ):
        """Vectorized optimal action distribution for every agent.

        Args:
            agent_game_types: [num_agents] game-type index for each agent.
            agent_roles:      [num_agents] role index for each agent.
            agent_belief_probs: [num_agents, num_states] each agent's belief as raw probs.

        Returns:
            [num_agents, num_actions] each agent's action-distribution probs.
        """

        def one_agent(game_type, role, belief_probs):
            belief = distrax.Categorical(probs=belief_probs)
            return self._agent_policy(game_type, role, belief).probs

        return jax.vmap(one_agent)(agent_game_types, agent_roles, agent_belief_probs)

    def _agent_belief_updates(
        self, game_type, role, ego_belief, other_belief_estimate, observation, action
    ):
        """True-belief + other-belief-estimate update for one agent.

        Conditioned on the agent's environment (``game_type``, via lax.switch over
        the per-type belief factories) and its ``role`` (passed as the traced
        ``agent_id`` — the belief methods now accept a traced agent_id). The other
        agent's action is unobserved, so both updates marginalize over it via the
        other role's optimal policy. Assumes dyadic games (other role = 1 - role).

        Returns (new_true_belief_probs, new_other_estimate_probs).
        """
        other_policy = lambda belief: self._agent_policy(game_type, 1 - role, belief)

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
                new_estimate = (
                    belief_factory.update_other_belief_estimate_with_observation_only(
                        other_belief_estimate,
                        observation,
                        action,
                        other_policy,
                        agent_id=role,
                    )
                )
                return new_true.probs, new_estimate.probs

            return branch

        branches = [make_branch(factory) for factory in self._belief_factories]
        return jax.lax.switch(
            game_type, branches, ego_belief, other_belief_estimate, observation, action
        )
    
    def _partner_agent(self, agent_game_assignment, agent_role_assignment, num_games):
        """Each agent's dyadic partner (the agent in the other role of its game).

        Builds the (game, role) -> agent map and reads off the OTHER role, giving a
        [num_agents] array where partner[i] is the agent i is paired with. The map is
        its own inverse in a dyad. Same gather used in ``_step_underlying_env``.
        """
        game_role_to_agent = (
            jnp.zeros((num_games, self.num_roles), dtype=jnp.int32)
            .at[agent_game_assignment, agent_role_assignment]
            .set(jnp.arange(self.num_agents))
        )
        return game_role_to_agent[agent_game_assignment, 1 - agent_role_assignment]

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
            engaged with -- i.e. what i (a sender) thinks its partner believes. Storage is
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
        partner_agent = self._partner_agent(
            state.agent_game_assignment, state.agent_role_assignment, num_games
        )

        on_utterance_stage = state.communicative_round_stage == UTTERANCE_STAGE
        # beliefs[i] = agent i's own belief (subject-indexed as stored).
        beliefs = jnp.where(
            on_utterance_stage, state.true_agent_belief_states, jnp.nan
        )
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

        # Map (game, role) -> agent index so we can assemble per-game joint actions.
        game_role_to_agent = (
            jnp.zeros((num_games, self.num_roles), dtype=jnp.int32)
            .at[agent_game_assignment, agent_role_assignment]
            .set(jnp.arange(self.num_agents))
        )

        # Each agent's dyadic partner (the agent in the other role of its game). Used to
        # translate between subject-indexed storage and the per-agent update, whose math
        # is in terms of "this agent's estimate of its partner".
        partner_agent = game_role_to_agent[
            agent_game_assignment, 1 - agent_role_assignment
        ]  # [num_agents]
        # Agent i's estimate of its partner is the estimate ABOUT the partner (partner's row).
        estimate_of_partner = estimated_agent_belief_states[partner_agent]  # [num_agents, S]

        action_key, transition_key, obs_key = jax.random.split(key, 3)

        # 1. Each agent samples an action from its optimal policy given its true belief.
        agent_action_distributions = self.agent_action_distributions(
            agent_game_types, agent_role_assignment, true_agent_belief_states
        )
        agent_actions = jax.vmap(
            lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
        )(agent_action_distributions, jax.random.split(action_key, self.num_agents))  # [num_agents]

        # 2. Assemble per-game joint actions in canonical (role 0, role 1) order.
        game_actions_by_role = agent_actions[game_role_to_agent]  # [num_games, num_roles]
        joint_a0, joint_a1 = game_actions_by_role[:, 0], game_actions_by_role[:, 1]

        # 3. Transition each game (gathers index the stacked params by per-game type).
        next_state_probs = self.all_env_parameters.transition[
            game_types_per_game, game_states, joint_a0, joint_a1
        ]  # [num_games, S]
        game_next_states = jax.vmap(
            lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
        )(next_state_probs, jax.random.split(transition_key, num_games))  # [num_games]

        # 3b. Gather each game's per-role reward R_i(s, a0, a1, s') from the stacked
        # reward tensor [num_game_types, num_roles, S, A, A, S], then scatter to agents.
        game_rewards = self.all_env_parameters.reward[
            game_types_per_game[:, None],           # [num_games, 1]
            jnp.arange(self.num_roles)[None, :],    # [1, num_roles]
            game_states[:, None],                   # [num_games, 1]
            joint_a0[:, None],                      # [num_games, 1]
            joint_a1[:, None],                      # [num_games, 1]
            game_next_states[:, None],              # [num_games, 1]
        ]  # [num_games, num_roles]
        agent_rewards = game_rewards[
            agent_game_assignment, agent_role_assignment
        ]  # [num_agents]

        # 4. Sample the joint observation and split it into the two roles' observations.
        obs_probs = self.all_env_parameters.observation[
            game_types_per_game, game_next_states, joint_a0, joint_a1
        ]  # [num_games, O, O]
        num_obs = obs_probs.shape[-1]
        flat_obs = jax.vmap(
            lambda probs, k: distrax.Categorical(probs=probs.reshape(-1)).sample(seed=k)
        )(obs_probs, jax.random.split(obs_key, num_games))  # [num_games]
        game_obs_by_role = jnp.stack(
            [flat_obs // num_obs, flat_obs % num_obs], axis=1
        )  # [num_games, num_roles]
        agent_observations = game_obs_by_role[
            agent_game_assignment, agent_role_assignment
        ]  # [num_agents]

        # 5. Per-agent belief + estimate-of-partner update from the observation.
        def per_agent_belief_updates(
            game_type, role, ego_belief_probs, partner_est_probs, ego_obs, ego_action
        ):
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
        return game_next_states, next_true, next_estimated, agent_rewards, agent_actions, agent_observations

    def _step_env_impl(self, key: chex.PRNGKey, state: StackedSignificationState, utterance_actions: chex.Array, belief_estimate_post_utterance: chex.Array, belief_actions: chex.Array):
        """
        This method has many important jobs, but the main thing it does is abstract away the complexity
        of the underlying DecPOMDPs the agents are engaged in. This method is a transition function for
        the overall communication game.

        At each time step, agents are either 1) generating utterances to send to interlocutors
        based on their belief about the state of the DecPOMDP and their estimate of their
        partner's belief state. Or 2) updating their belief given their prior belief and the
        utterance their partner sent to them. Both of these updates are made by neural networks
        and are learned -- this does not happen analytically.

        In order to abstract away the underlying machinery of the DecPOMDP, additional things need
        to happen after belief stages. Specifically, the underlying DecPOMDP needs to actually advance
        according to the new belief states of the agents. After this, an utterance stage begins again,
        after the DecPOMDP observations update the ground-truth belief states of both agents.

        Args:
            key: a PRNGKey.
            state: a StackedSignificationState.
            utterance_actions: [num_agents, utterance_action_dim], AGENT-indexed (row i is
                agent i's utterance).
            belief_estimate_post_utterance: [num_agents, S], SUBJECT-indexed (row i is the
                estimate ABOUT agent i's belief state).
            belief_actions: [num_agents, S], AGENT-indexed (row i is agent i's proposed new
                belief, a distribution over its game's S world states).

        Returns (pre_act_state, state, observation, rewards). pre_act_state is the substate
        after communication has resolved but before the underlying world steps -- only
        meaningful on act steps. The public ``step_env`` wrapper drops it (returning the
        usual (state, observation, rewards)); ``step_env_with_substate`` exposes it.
        """

        act_key, boundary_key, obs_key = jax.random.split(key, 3)

        # This round's speakers, over ROLES, mapped to a per-agent mask.
        who_speaks_this_round = state.active_who_speaks[state.communication_round_iterator]  # [num_roles]
        agent_speaks_binary_mask = who_speaks_this_round[state.agent_role_assignment].astype(bool)  # [num_agents]
        # ...and this round's listeners:
        agent_listens_binary_mask = who_speaks_this_round[1 - state.agent_role_assignment].astype(bool)  # [num_agents]

        # Useful predicates
        on_utterance_stage = state.communicative_round_stage == UTTERANCE_STAGE
        on_belief_stage = state.communicative_round_stage == BELIEF_STAGE

        # === Bit 1: utterance stage =========================================
        # On the UTTERANCE_STAGE the utterance_actions have been freshly populated.
        # They need to be stored in the state so that get_obs can return them to the
        # corresponding listener agent next. NOTE: There is double-filtering going on here from two jnp.wheres, not sure it's necessary
        speakers_utterances = jnp.where(
            agent_speaks_binary_mask[:, None], utterance_actions, jnp.zeros_like(utterance_actions)
        )
        # Later, we may care about keeping track of the previous utterances in future states
        # (using the field as persistent memory) so information can be aggregated across rounds:
        # next_agent_utterance_actions_unrendered = jnp.where(
        #     on_utterance_stage,
        #     speakers_utterances,
        #     state.agent_utterance_actions_unrendered,
        # )
        # For now, keep it a pure per-step input (no memory):
        next_agent_utterance_actions_unrendered = jnp.where(
            on_utterance_stage,
            speakers_utterances,
            jnp.ones_like(state.agent_utterance_actions_unrendered),
        )

        # === Bit 1b: belief estimate after uttering ==========================
        # After a speaker utters, it refreshes its estimate of its dyadic partner (the
        # listener). estimated_agent_belief_states is SUBJECT-indexed (row k = the estimate
        # ABOUT agent k), and the incoming belief_estimate_post_utterance shares that
        # indexing, so the rows that change are the LISTENERS' and we write straight across for them.
        # The estimate is formed right after the utterance, so we apply it on the utterance stage.
        update_listener_belief_estimate = on_utterance_stage & agent_listens_binary_mask  # [num_agents]
        next_estimated_agent_belief_states = jnp.where(
            update_listener_belief_estimate[:, None],
            belief_estimate_post_utterance,
            state.estimated_agent_belief_states, # Otherwise keep the existing estimate of their belief state
        )

        # === Bit 2: belief stage ============================================
        # On the BELIEF_STAGE, belief_actions is freshly populated by listeners.
        # We will write those beliefs into the next ground truth belief states.
        update_listener_true_belief = on_belief_stage & agent_listens_binary_mask  # [num_agents]
        next_true_agent_belief_states = jnp.where(
            update_listener_true_belief[:, None], belief_actions, state.true_agent_belief_states
        )

        # === Substate: communication resolved, world not yet stepped ========
        # A plottable snapshot of the moment BETWEEN communication and the world step:
        # beliefs have been adopted (next_true / next_estimated) but the underlying DecPOMDP
        # has NOT transitioned yet, so world states, routing and the scheduler are still the
        # incoming `state`'s, and the act outputs sit at their no-act sentinels. It reuses
        # this same dataclass -- no new fields -- and is only meaningful on act steps (where
        # the world is about to move). Callers that want it use step_env_with_substate.
        pre_act_state = state.replace(
            true_agent_belief_states=next_true_agent_belief_states,
            estimated_agent_belief_states=next_estimated_agent_belief_states,
            last_agent_rewards=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            last_agent_actions=-jnp.ones((self.num_agents,), dtype=jnp.int32),
            last_agent_observations=-jnp.ones((self.num_agents,), dtype=jnp.int32),
        )

        # === Bit 3: scheduling state machine ================================
        # This round's utterance and belief effects are now recorded, so advance the
        # stage, the round cursor, the env iterations and the active block. World
        # transitions and the episode boundary are still NOT applied yet (later bits) --
        # this only moves the schedule forward.
        #
        # A round completes on its belief stage; that completion is an "act" (the env
        # steps) when it is the block's last REAL round (total_num_rounds drops padding).
        is_last_round = (
            state.communication_round_iterator
            == state.active_total_num_rounds - 1
        )
        is_act = on_belief_stage & is_last_round

        # The stage toggles every call. The round cursor only moves when a round
        # completes (belief stage): wrap to 0 on an act, else advance within the block.
        next_stage = jnp.where(on_belief_stage, UTTERANCE_STAGE, BELIEF_STAGE)
        # This keeps track of which communication round we are in within a block.
        next_round_cursor = jnp.where(
            on_belief_stage,
            jnp.where(is_act, 0, state.communication_round_iterator + 1),
            state.communication_round_iterator,
        )

        # Env iterations advance only on an act.
        act_increment = is_act.astype(jnp.int32)
        next_cumulative_env_iteration = state.cumulative_env_iteration + act_increment

        # On an act, fetch the NEXT block, keyed by the new cumulative_env_iteration;
        # otherwise keep walking the current one. communication_scheme_fn is a pure,
        # shape-stable gather, so compute it unconditionally and select with where.
        next_block = self.communication_scheme_fn(next_cumulative_env_iteration)
        next_active_who_speaks = jnp.where(
            is_act, next_block.who_speaks, state.active_who_speaks
        )
        next_active_total_num_rounds = jnp.where(
            is_act, next_block.total_num_rounds, state.active_total_num_rounds
        )

        # === Bit 4: act (the underlying world step) =========================
        # On the block's last belief stage (is_act) every game takes one DecPOMDP step:
        # each agent samples an action from its optimal policy given its post-
        # communication true belief, the games transition + emit observations, and
        # beliefs update from those observations. The step also yields each agent's
        # underlying reward, which becomes the stacked game's reward signal. Off the act,
        # nothing world-side moves and the reward is zero.
        def do_act(_):
            return self._step_underlying_env(
                act_key,
                state.game_states,
                state.game_types,
                state.agent_game_assignment,
                state.agent_role_assignment,
                next_true_agent_belief_states,
                next_estimated_agent_belief_states,
            )

        def skip_act(_):
            return (
                state.game_states,
                next_true_agent_belief_states,
                next_estimated_agent_belief_states,
                jnp.zeros((self.num_agents,), dtype=jnp.float32),   # Until we take an environment step, we don't have rewards to supply to speakers or listeners.
                -jnp.ones((self.num_agents,), dtype=jnp.int32),
                -jnp.ones((self.num_agents,), dtype=jnp.int32),   # no observation off the act
            )

        (
            game_states_after_act,
            true_beliefs_after_act,
            estimated_beliefs_after_act,
            agent_rewards_after_act,
            agent_actions_after_act,
            agent_observations_after_act,
        ) = jax.lax.cond(is_act, do_act, skip_act, operand=None)

        # === Bit 5: episode boundary ========================================
        # When an act takes underlying_env_iteration to the episode horizon, end the
        # episode: re-route a fresh assignment (keyed by the next episode index), resample
        # initial world states + beliefs, reset underlying_env_iteration (to 0, or to 1 if
        # act_on_reset_before_communicating has the new episode act first, mirroring reset),
        # bump episode_index, and clear utterances. cumulative_env_iteration and the comm
        # schedule (already advanced to the next block in bit 3) keep going. This step's
        # reported reward/action stays the ENDING episode's final act (below); the new
        # episode's act-first only primes its beliefs/state.
        next_underlying_env_iteration = state.underlying_env_iteration + act_increment
        is_boundary = is_act & (next_underlying_env_iteration >= state.episode_horizon)

        def begin_new_episode(_):
            new_episode_index = state.episode_index + 1
            init = self._begin_episode(boundary_key, new_episode_index)
            return (
                init["agent_game_assignment"],
                init["agent_role_assignment"],
                init["game_types"],
                init["game_states"],
                init["true_agent_belief_states"],
                init["estimated_agent_belief_states"],
                init["episode_horizon"],
                new_episode_index,
                init["underlying_env_iteration"],
                jnp.zeros_like(next_agent_utterance_actions_unrendered),
            )

        def continue_episode(_):
            return (
                state.agent_game_assignment,
                state.agent_role_assignment,
                state.game_types,
                game_states_after_act,
                true_beliefs_after_act,
                estimated_beliefs_after_act,
                state.episode_horizon,
                state.episode_index,
                next_underlying_env_iteration,
                next_agent_utterance_actions_unrendered,
            )

        (
            ep_agent_game_assignment,
            ep_agent_role_assignment,
            ep_game_types,
            ep_game_states,
            ep_true_beliefs,
            ep_estimated_beliefs,
            ep_episode_horizon,
            ep_episode_index,
            ep_underlying_env_iteration,
            ep_utterances,
        ) = jax.lax.cond(is_boundary, begin_new_episode, continue_episode, operand=None)

        state = state.replace(
            communicative_round_stage=next_stage,
            communication_round_iterator=next_round_cursor,
            cumulative_communication_round_iterator=(
                state.cumulative_communication_round_iterator + 1
            ),
            underlying_env_iteration=ep_underlying_env_iteration,
            cumulative_env_iteration=next_cumulative_env_iteration,
            episode_index=ep_episode_index,
            episode_horizon=ep_episode_horizon,
            active_who_speaks=next_active_who_speaks,
            active_total_num_rounds=next_active_total_num_rounds,
            agent_utterance_actions_unrendered=ep_utterances,
            agent_game_assignment=ep_agent_game_assignment,
            agent_role_assignment=ep_agent_role_assignment,
            game_types=ep_game_types,
            game_states=ep_game_states,
            true_agent_belief_states=ep_true_beliefs,
            estimated_agent_belief_states=ep_estimated_beliefs,
            # The act's reward is this step's signal even when it also ends the episode;
            # the boundary reset does not clear it. Same for the debug actions.
            last_agent_rewards=agent_rewards_after_act,
            last_agent_actions=agent_actions_after_act,
            last_agent_observations=agent_observations_after_act,
        )

        # Observation reflects the NEW stage (the action the agent will take next); its
        # beliefs/estimated-beliefs or utterances are NaN'd per that stage. The per-agent
        # environment reward for this step is returned alongside (also readable off
        # state.last_agent_rewards): it is the act's reward on an act step and zero on a
        # communication-only step.
        obs = self.get_obs(obs_key, state)
        return pre_act_state, state, obs, state.last_agent_rewards

    def step_env(self, key: chex.PRNGKey, state: StackedSignificationState, utterance_actions: chex.Array, belief_estimate_post_utterance: chex.Array, belief_actions: chex.Array):
        """Public transition function: returns (state, observation, rewards).

        Thin wrapper over ``_step_env_impl`` that drops the pre-act substate, preserving the
        original (state, observation, rewards) contract for training and rollouts.
        """
        _pre_act, state, obs, rewards = self._step_env_impl(
            key, state, utterance_actions, belief_estimate_post_utterance, belief_actions
        )
        return state, obs, rewards

    def step_env_with_substate(self, key: chex.PRNGKey, state: StackedSignificationState, utterance_actions: chex.Array, belief_estimate_post_utterance: chex.Array, belief_actions: chex.Array):
        """Like ``step_env`` but also returns the pre-act substate for inspection/plotting.

        Returns (pre_act_state, state, observation, rewards). On an act step, pre_act_state
        is the snapshot after communication resolves but before the world steps (post-
        communication beliefs, old world/routing, no reward/action/observation yet); on a
        communication-only step it carries the same beliefs as the returned state and is not
        typically plotted as a separate frame.
        """
        return self._step_env_impl(
            key, state, utterance_actions, belief_estimate_post_utterance, belief_actions
        )

    def _begin_episode(self, key, episode_index):
        """Route a fresh episode and produce its initial fields.

        Assigns agents to games/roles via routing_fn (keyed by episode_index), samples
        each game's initial world state, and sets each agent's initial belief (and the
        initial estimate about it, which equals its own prior) from the env params.

        Then, if ``act_on_reset_before_communicating`` is set, takes one joint DecPOMDP
        step before any communication (beliefs update from the resulting observation) so
        the episode starts at in-game iteration 1; otherwise it stays communicate-first at
        iteration 0. Returns a dict of the episode-initial fields -- including the initial
        ``underlying_env_iteration`` and the act's ``last_agent_rewards`` /
        ``last_agent_actions`` / ``last_agent_observations`` (zeros / -1 / -1 when
        communicate-first) -- plus a fresh leftover
        ``key``. Shared by reset (episode 0) and the step_env episode boundary (later
        episodes), so the act-first behavior applies to every episode.
        """
        routing_key, state_key, key = jax.random.split(key, 3)
        route = self.routing_fn(key=routing_key, iteration=episode_index)

        agent_role_assignment = route.agent_role_assignment  # [num_agents]
        game_types_per_game = route.game_set  # [num_games]
        agent_game_types = game_types_per_game[route.agent_game_assignment]  # [num_agents]

        # Each agent's initial belief. estimated_agent_belief_states is subject-indexed
        # (row i = the estimate ABOUT agent i); before any communication the partner's
        # estimate of agent i is just agent i's own prior, so the two rows coincide at
        # the start of an episode.
        agent_initial_belief_states = self.all_env_parameters.initial_belief_states[
            agent_game_types, agent_role_assignment
        ]

        # Sample each game's true initial world state from its initial-state dist.
        num_games = game_types_per_game.shape[0]
        init_state_dists = self.all_env_parameters.initial_state_distribution[
            game_types_per_game
        ]
        game_states = jax.vmap(
            lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
        )(init_state_dists, jax.random.split(state_key, num_games))

        true_agent_belief_states = agent_initial_belief_states
        estimated_agent_belief_states = agent_initial_belief_states
        # Default (communicate-first) path: no act yet, so no reward and no action.
        last_agent_rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
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
        # at in-game iteration 1; the communicate-first path is still at 0.
        underlying_env_iteration = jnp.asarray(
            1 if self.act_on_reset_before_communicating else 0, dtype=jnp.int32
        )

        return dict(
            agent_game_assignment=route.agent_game_assignment,
            agent_role_assignment=agent_role_assignment,
            game_types=game_types_per_game,
            game_states=game_states,
            true_agent_belief_states=true_agent_belief_states,
            estimated_agent_belief_states=estimated_agent_belief_states,
            episode_horizon=route.underlying_env_steps_per_episode,
            underlying_env_iteration=underlying_env_iteration,
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
        # takes the initial act, returning the post-act fields, iteration, and rewards.
        init = self._begin_episode(key, jnp.asarray(0, dtype=jnp.int32))
        key = init["key"]
        game_states = init["game_states"]
        true_agent_belief_states = init["true_agent_belief_states"]
        estimated_agent_belief_states = init["estimated_agent_belief_states"]
        last_agent_rewards = init["last_agent_rewards"]
        last_agent_actions = init["last_agent_actions"]
        last_agent_observations = init["last_agent_observations"]

        # At reset the per-episode and cumulative env iterations coincide (0, or 1 if the
        # episode acted first).
        underlying_iteration = init["underlying_env_iteration"]

        # Fetch the first active communication block, keyed by cumulative_env_iteration
        # (== underlying_iteration at reset), and store it flat. The round cursors start
        # at the block's first round; no communication rounds have run yet.
        active_scheme = self.communication_scheme_fn(underlying_iteration)

        state = StackedSignificationState(
            # No utterances yet at reset: unrendered starts as a zero vector per agent
            # (sized by utterance_action_dim) so the utterance stage can write into it;
            # rendered is unused for now.
            agent_utterance_actions_unrendered=jnp.zeros(
                (self.num_agents, self.utterance_action_dim), dtype=jnp.float32
            ),
            agent_utterance_actions_rendered=None,
            agent_game_assignment=init["agent_game_assignment"],
            agent_role_assignment=init["agent_role_assignment"],
            game_types=init["game_types"],
            underlying_env_iteration=underlying_iteration,
            cumulative_env_iteration=underlying_iteration,
            episode_index=jnp.asarray(0, dtype=jnp.int32),
            episode_horizon=init["episode_horizon"],
            active_who_speaks=active_scheme.who_speaks,
            active_total_num_rounds=active_scheme.total_num_rounds,
            communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
            cumulative_communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
            # Reset starts on the utterance stage (the speaker emits first).
            communicative_round_stage=jnp.asarray(UTTERANCE_STAGE, dtype=jnp.int32),
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


def a_to_b_thrice_example(key: chex.PRNGKey = jax.random.key(0)):
    """Build a small stacked env driven by ``a_to_b_thrice_scheme_fn`` and walk one block.

    ``a_to_b_thrice`` is a single 3-round block in which role A speaks to role B on
    each of the three rounds; the third round's belief stage is the act that steps
    the underlying DecPOMDP. Each round is two ``step_env`` calls (utterance then
    belief), so one block == 6 ``step_env`` calls == one underlying-env step. The walk
    prints the schedule as it advances so the stage / round / env-iteration cadence is
    visible. Returns ``(env, final_state)``.
    """
    from training.routing import simple_routing_fn
    from training.communication_scheme import a_to_b_thrice_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
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
        return (f"stage={int(st.communicative_round_stage)} "
                f"round={int(st.communication_round_iterator)} "
                f"env_iter={int(st.underlying_env_iteration)} "
                f"comm={int(st.cumulative_communication_round_iterator)}")

    print("=== a_to_b_thrice example (1 block = 3 rounds = 6 step_env calls) ===")
    print("  reset: ", fmt(state))
    for t in range(6):
        walk_key, step_key = jax.random.split(walk_key)
        state, _, rewards = env.step_env(
            step_key, state, utterance_actions, belief_estimate, belief_actions
        )
        print(f"  step {t}:", fmt(state), "reward_sum=", float(rewards.sum()))
    print("ok: a_to_b_thrice walks 3 rounds and acts once on the last belief stage")
    return env, state


if __name__ == "__main__":
    a_to_b_thrice_example(jax.random.key(0))

    from training.routing import simple_routing_fn
    from training.communication_scheme import b_to_a_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    # Build the stacked params + policy table from one game type (the guessing game).
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
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
    print("underlying_iter:    ", state.underlying_env_iteration)
    print("active who_speaks:  ", state.active_who_speaks.tolist())
    print("active rounds:      ", state.active_total_num_rounds)
    print("round iters (u,c):  ", state.communication_round_iterator,
          state.cumulative_communication_round_iterator)
    print("round stage:        ", state.communicative_round_stage)
    print("true beliefs[0]:    ", state.true_agent_belief_states[0])
    assert state.underlying_env_iteration == 0, "communicate-first takes no env step at reset"
    assert state.communication_round_iterator == 0, "round cursor starts at 0"
    assert state.communicative_round_stage == UTTERANCE_STAGE, "reset starts on the utterance stage"

    # reset (act-first path): one joint action + belief updates before communicating.
    act_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
        communication_scheme_fn=b_to_a_scheme_fn,
        utterance_action_dim=3,
        skip_first_communication_step=True,
    )
    act_state, _ = act_env.reset(jax.random.key(1))
    print("=== act-first reset ===")
    print("game_states:        ", act_state.game_states)
    print("underlying_iter:    ", act_state.underlying_env_iteration)
    print("true beliefs[0]:    ", act_state.true_agent_belief_states[0])
    print("estimate about agent 0:", act_state.estimated_agent_belief_states[0])
    assert act_state.underlying_env_iteration == 1, "act-first advances the env once at reset"

    # Direct dispatch check: each agent's belief is routed through its
    # (game_type, role) policy. Both guessing-game roles are identity here, so we
    # confirm each agent gets its OWN belief back through the correct branch.
    belief_role_0 = jnp.array([0.6, 0.3, 0.1, 0.0])
    belief_role_1 = jnp.array([0.2, 0.2, 0.6, 0.0])
    dists = env.agent_action_distributions(
        agent_game_types=jnp.array([0, 0]),
        agent_roles=jnp.array([0, 1]),
        agent_belief_probs=jnp.stack([belief_role_0, belief_role_1]),
    )
    print("role 0 action probs:", dists[0])
    print("role 1 action probs:", dists[1])
    assert jnp.allclose(dists[0], belief_role_0)
    assert jnp.allclose(dists[1], belief_role_1)
    print("ok: env built via factory; policies dispatch by (game_type, role)")

    # step_env bit 3: scheduling state machine. Use a 3-round block (a_to_b_thrice) so
    # the round cursor visibly walks 0,0,1,1,2,2 (two stages each) and the env iteration
    # advances exactly once per completed block (every 6 step_env calls).
    from training.communication_scheme import a_to_b_thrice_scheme_fn

    sched_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
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
    # bit 1b's wiring to be visible.
    other_est = s0.true_agent_belief_states.at[:, 0].add(0.5)
    other_est = other_est / other_est.sum(axis=-1, keepdims=True)
    role0 = s0.agent_role_assignment == 0
    role1 = ~role0

    # Bit 1: the utterance stage stashes ONLY the speaking role's (role 0 in a_to_b)
    # utterances; listeners (role 1) are zeroed.
    after_utt, after_utt_obs, after_utt_rewards = sched_env.step_env(jax.random.key(0), s0, utt, other_est, valid_belief)
    # A communication-only step (the utterance stage) yields zero environment reward.
    assert jnp.all(after_utt_rewards == 0.0), "utterance-stage step returns zero reward"
    assert jnp.all(after_utt_rewards == after_utt.last_agent_rewards), "returned reward matches state"
    assert jnp.all(after_utt.agent_utterance_actions_unrendered[role0] == 1.0)
    assert jnp.all(after_utt.agent_utterance_actions_unrendered[role1] == 0.0)
    print("ok: utterance stage stashes speakers' (role 0) utterances, zeros listeners")

    # get_obs: after one step we are on the belief stage, so the observation NaNs the
    # belief groups and keeps the utterances (the opposite of the reset observation).
    ou_beliefs, ou_estimated, ou_utterances = after_utt_obs
    assert jnp.all(jnp.isnan(ou_beliefs)), "belief-stage obs NaNs the beliefs"
    assert jnp.all(jnp.isnan(ou_estimated)), "belief-stage obs NaNs the estimated beliefs"
    assert not jnp.any(jnp.isnan(ou_utterances)), "belief-stage obs keeps the utterances"
    print("ok: belief-stage observation NaNs beliefs, keeps utterances")

    # get_obs indexing: utterances are remapped to the PARTNER (speaker), not the agent's
    # own row. role 0 spoke ones, role 1 was zeroed; so each listener (role 1) HEARS its
    # role-0 partner's ones, and each speaker (role 0) hears its role-1 partner's zeros.
    assert jnp.all(ou_utterances[role1] == 1.0), "listeners hear their partner-speaker's utterance"
    assert jnp.all(ou_utterances[role0] == 0.0), "speakers hear their (silent listener) partner"
    print("ok: get_obs routes each agent its partner's utterance, not its own")

    # get_obs indexing: on the utterance stage, estimated_beliefs is the estimate of the
    # RECEIVER (partner) the sender is engaged with -- the partner's subject-indexed row.
    us_beliefs, us_estimated, us_utterances = sched_env.get_obs(jax.random.key(0), s0)
    partner = sched_env._partner_agent(
        s0.agent_game_assignment, s0.agent_role_assignment, s0.game_states.shape[0]
    )
    assert jnp.all(us_beliefs == s0.true_agent_belief_states), "beliefs[i] is agent i's own belief"
    assert jnp.all(
        us_estimated == s0.estimated_agent_belief_states[partner]
    ), "estimated_beliefs[i] is the estimate of agent i's partner (the receiver)"
    print("ok: get_obs routes each sender the estimate of its receiver partner")

    # Bit 1b: the estimate is subject-indexed, so the LISTENERS' rows (role 1, the agents
    # a speaker just re-estimated) take the supplied value; the speakers' rows (role 0)
    # keep their prior estimate.
    assert jnp.all(after_utt.estimated_agent_belief_states[role1] == other_est[role1])
    assert jnp.all(
        after_utt.estimated_agent_belief_states[role0]
        == s0.estimated_agent_belief_states[role0]
    )
    print("ok: utterance stage records the listener-subject (role 1) post-utterance estimate")

    # Bit 2: the following belief stage has the listeners (role 1) adopt their proposed
    # belief_actions; the speakers (role 0) keep their belief.
    after_belief, _, _ = sched_env.step_env(jax.random.key(1), after_utt, utt, other_est, valid_belief)
    assert jnp.all(after_belief.true_agent_belief_states[role1] == valid_belief[role1])
    assert jnp.all(
        after_belief.true_agent_belief_states[role0] == s0.true_agent_belief_states[role0]
    )
    print("ok: belief stage applies listeners' (role 1) belief_actions, speakers unchanged")

    def fmt(st):
        return (f"stage={int(st.communicative_round_stage)} "
                f"round={int(st.communication_round_iterator)} "
                f"env_iter={int(st.underlying_env_iteration)} "
                f"comm={int(st.cumulative_communication_round_iterator)}")

    print("=== step_env scheduling walk (a_to_b_thrice: 3 rounds/block) ===")
    s = s0
    print("  reset:  ", fmt(s))
    for t in range(6):
        s, _, step_rewards = sched_env.step_env(jax.random.key(t), s, utt, other_est, valid_belief)
        assert jnp.all(step_rewards == s.last_agent_rewards), "step_env returns this step's reward"
        print(f"  step {t}:", fmt(s),
              "reward_sum=", float(step_rewards.sum()),
              "actions=", s.last_agent_actions.tolist(),
              "obs=", s.last_agent_observations.tolist())
        # The act is the last belief stage (step 5); every earlier (communication-only)
        # step leaves the reward signal at zero and the debug actions/observations at the
        # -1 sentinel.
        if t < 5:
            assert jnp.all(s.last_agent_rewards == 0.0), "no reward on communication-only steps"
            assert jnp.all(s.last_agent_actions == -1), "no action on communication-only steps"
            assert jnp.all(s.last_agent_observations == -1), "no observation on communication-only steps"
        else:
            assert jnp.all(s.last_agent_actions >= 0), "act step records real actions"
            assert jnp.all(s.last_agent_observations >= 0), "act step records real observations"
    assert s.last_agent_rewards.shape == (10,), "reward is per-agent"
    assert s.last_agent_actions.shape == (10,), "action is per-agent"
    assert s.last_agent_observations.shape == (10,), "observation is per-agent"
    assert int(s.underlying_env_iteration) == 1, "one block (6 stages) == one env step"
    assert int(s.communication_round_iterator) == 0, "cursor wrapped after the act"
    assert int(s.communicative_round_stage) == UTTERANCE_STAGE, "back to utterance after the act"
    # Bit 4: the act ran a real DecPOMDP step -> beliefs stay normalized after the obs update.
    assert jnp.allclose(s.true_agent_belief_states.sum(-1), 1.0), "act keeps beliefs normalized"
    print("  game_states reset->now:", s0.game_states.tolist(), "->", s.game_states.tolist())
    print("ok: scheduler cycles stages/rounds and the act steps the underlying env once")

    # Bit 5: episode boundary. Horizon = 2 underlying steps; b_to_a is 1 round/block, so
    # an env step every 2 step_env calls and a boundary every 4 calls.
    boundary_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(
            num_agents=10, agents_per_game=2, underlying_env_steps_per_episode=2
        ),
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
    assert int(b.episode_index) == 0 and int(b.episode_horizon) == 2
    print("=== episode boundary walk (horizon=2, b_to_a) ===")
    for t in range(4):
        b, _, _ = boundary_env.step_env(jax.random.key(100 + t), b, utt3, other_est3, valid3)
        print(f"  step {t}: env_iter={int(b.underlying_env_iteration)} "
              f"cum_env={int(b.cumulative_env_iteration)} episode={int(b.episode_index)}")
    assert int(b.episode_index) == 1, "a new episode began at the horizon"
    assert int(b.underlying_env_iteration) == 0, "per-episode env iter reset at the boundary"
    assert int(b.cumulative_env_iteration) == 2, "cumulative env iter keeps counting"
    print("ok: episode boundary re-routes and resets the per-episode counters")

    
