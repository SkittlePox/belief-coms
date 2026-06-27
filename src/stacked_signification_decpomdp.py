import jax, chex
import jax.numpy as jnp
import distrax
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from flax import struct
from typing import Any, Callable, Sequence
from functools import partial
from routing import RouteFn
from communication_scheme import CommunicationSchemeFn

# FlexibleEnvParams / OptimalPolicy are defined in envs.flexible_env (the leaf of
# the env-definition dependency graph) and consumed here.
from envs.flexible_env import FlexibleEnvParams, OptimalPolicy
from tools.belief_representations import CategoricalBeliefState


# A communicative round runs in two stages, one per step_env call; the state's
# communicative_round_stage records which stage the current step is in.
UTTERANCE_STAGE = 0  # the speaker emits an utterance / message action
BELIEF_STAGE = 1     # the listener emits a belief action, completing the round


@struct.dataclass
class StackedSignificationState:
    """State carried by the StackedSignificationDecPOMDP across step_env calls.

    Games are dyadic (num_roles == 2). Agents are indexed ``[num_agents]``; games
    are indexed ``[num_games] = num_agents // num_roles``. The routing fields say
    which game/role each agent occupies, so per-agent quantities can be gathered
    to/from per-game quantities.

    Utterance fields are carried here but are NOT produced by reset (only by the
    communication phase of step_env); reset leaves them as None.
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

    # Communication schedule: the active block, stored flat as its who_speaks /
    # total_num_rounds (re-fetched from communication_scheme_fn on each underlying-env
    # step, keyed by cumulative_env_iteration), plus the cursors that walk it. The
    # instantaneous who-speaks for the current round is
    # active_who_speaks[underlying_communication_round_iterator].
    active_who_speaks: chex.Array                         # [num_rounds (padded), num_speakers]
    active_total_num_rounds: chex.Array                   # scalar int: real round count of the active block (excludes padding)
    underlying_communication_round_iterator: chex.Array   # scalar int: round within the active block (0..total_num_rounds-1); advances once per completed (two-step) round
    cumulative_communication_round_iterator: chex.Array   # scalar int: total step_env calls (two per who_speaks round); never resets

    # Two-stage rounds: each who_speaks round spans TWO step_env iterations. The
    # speaker emits an utterance on UTTERANCE_STAGE, then the listener emits a belief
    # action on BELIEF_STAGE, which completes the round and advances
    # underlying_communication_round_iterator.
    communicative_round_stage: chex.Array                 # scalar int: UTTERANCE_STAGE or BELIEF_STAGE

    # World: the true DecPOMDP state of each game.
    game_states: chex.Array  # [num_games]

    # Beliefs: each agent's belief over states, and its estimate of the OTHER
    # role's belief in the same game.
    true_agent_belief_states: chex.Array  # [num_agents, S]
    other_estimated_agent_belief_states: chex.Array  # [num_agents, S]

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
        """
        self.num_agents = num_agents
        self.all_env_parameters = all_env_parameters
        self.routing_fn = routing_fn
        self.communication_scheme_fn = communication_scheme_fn
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

    def step_env(self, key: chex.PRNGKey, state, utterance_actions: chex.Array, belief_actions: chex.Array):
        """
        This function's job is basically just to listen to the routing function and handle all the communication processes and belief updates, etc. It actually does a lot.
        """
        # We assume that utterance_actions and belief_actions are size [num_agents, ?].
        #
        # ------------------------------------------------------------------ TODO
        # IMPORTANT: each who_speaks round spans TWO step_env iterations / stages,
        # tracked by state.communicative_round_stage -- the speaker emits an utterance
        # action on UTTERANCE_STAGE, the listener emits a belief action on BELIEF_STAGE,
        # which COMPLETES the round. So the round cursor advances (and the env may step)
        # only on BELIEF_STAGE. Nothing below is implemented.
        #
        # 1. Read this round's speakers from the active block:
        #        who_speaks_now = state.active_who_speaks[
        #            state.underlying_communication_round_iterator]            # [num_speakers]
        #    We only consume utterance_actions for agents whose role speaks this round;
        #    the rest contribute belief_actions / nothing.
        #
        # 2. Branch on state.communicative_round_stage (which stage of the round):
        #    a. UTTERANCE_STAGE: speaking agents emit utterances from their current
        #       belief (consume utterance_actions for agents whose role speaks this
        #       round); stash them in agent_utterance_actions_*. Advance the stage to
        #       BELIEF_STAGE. The round cursor does NOT move and the env does NOT step
        #       -> return here.
        #    b. BELIEF_STAGE: listening agents update true belief + other-belief-estimate
        #       from the utterance stashed last step (consume belief_actions). Reset the
        #       stage to UTTERANCE_STAGE. The round is now complete; continue to steps 3-4.
        #
        # 3. (BELIEF_STAGE only) the act coincides with the block's last REAL round
        #    (total_num_rounds excludes padding):
        #        is_act = (state.underlying_communication_round_iterator
        #                  == state.active_total_num_rounds - 1)
        #
        # 4. If is_act: step every game's DecPOMDP (actions, transition, observation,
        #    belief-from-observation update), advance game_states, then
        #        underlying_env_iteration               += 1
        #        cumulative_env_iteration               += 1
        #        underlying_communication_round_iterator  = 0
        #        (active_who_speaks, active_total_num_rounds) <- unpack
        #            self.communication_scheme_fn(cumulative_env_iteration)   # the NEXT block
        #    Else (still mid-block):
        #        underlying_communication_round_iterator += 1
        #
        # 5. Always: cumulative_communication_round_iterator += 1  (counts step_env
        #    calls -- two per who_speaks round).
        #
        # 6. Episode boundary (only reachable on an act step): if
        #    underlying_env_iteration == the episode horizon (the route's
        #    underlying_env_steps_per_episode), end the episode -- re-route (new
        #    assignment + horizon, keyed by an episode index), resample initial world
        #    states, reset beliefs, and set underlying_env_iteration = 0
        #    (cumulative_env_iteration keeps counting). Games that terminate early are
        #    masked until the boundary rather than re-routed individually.
        #
        # 7. Assemble and return the next StackedSignificationState (plus whatever
        #    observations / rewards the caller needs).
        # --------------------------------------------------------------------------
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        """
        NOTE: Resetting only happens once I think. The env basically continues on forever according to the routing function.
        """

        # So agents are either seeing beliefs and utterances and returning beliefs or they are seeing beliefs and something else and returning utterances.

        routing_key, key = jax.random.split(key)

        initial_route = self.routing_fn(key=routing_key, iteration=0)

        # Each agent's initial belief comes from the env parameters of the game
        agent_game_types = initial_route.game_set[
            initial_route.agent_game_assignment
        ]  # [num_agents]
        agent_roles = initial_route.agent_role_assignment  # [num_agents]
        agent_initial_belief_states = self.all_env_parameters.initial_belief_states[
            agent_game_types, agent_roles
        ]  # [num_agents, *belief_shape]

        # Each agent's initial estimate of the OTHER role's belief in the same game.
        # (Assumes 2 roles per game, so "the other role" is 1 - role.)
        other_roles = 1 - agent_roles
        est_other_initial_belief_states = self.all_env_parameters.initial_belief_states[
            agent_game_types, other_roles
        ]  # [num_agents, *belief_shape]

        # Sample each game's true initial world state from its initial-state dist.
        num_games = initial_route.game_set.shape[0]
        game_types_per_game = initial_route.game_set  # [num_games]
        state_key, key = jax.random.split(key)
        init_state_dists = self.all_env_parameters.initial_state_distribution[
            game_types_per_game
        ]  # [num_games, S]
        game_states = jax.vmap(
            lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
        )(
            init_state_dists, jax.random.split(state_key, num_games)
        )  # [num_games]

        # Map (game, role) -> agent index so we can assemble per-game joint actions.
        game_role_to_agent = (
            jnp.zeros((num_games, self.num_roles), dtype=jnp.int32)
            .at[
                initial_route.agent_game_assignment, initial_route.agent_role_assignment
            ]
            .set(jnp.arange(self.num_agents))
        )

        if self.act_on_reset_before_communicating:
            # Agents take one joint action (no communication yet), then everyone
            # updates beliefs. Nothing has been communicated, so agents do NOT observe
            # each other's actions -> belief updates marginalize over the other agent's
            # action via its policy (the *_with_observation_only belief updates).
            agent_initial_action_distributions = self.agent_action_distributions(
                agent_game_types, agent_roles, agent_initial_belief_states
            )

            action_key, transition_key, obs_key, key = jax.random.split(key, 4)

            # 1. Each agent samples an action from its action distribution.
            agent_actions = jax.vmap(
                lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
            )(
                agent_initial_action_distributions,
                jax.random.split(action_key, self.num_agents),
            )  # [num_agents]

            # 2. Assemble per-game joint actions in canonical (role 0, role 1) order.
            game_actions_by_role = agent_actions[
                game_role_to_agent
            ]  # [num_games, num_roles]
            joint_a0, joint_a1 = game_actions_by_role[:, 0], game_actions_by_role[:, 1]

            # 3. Step each game's DecPOMDP: sample next state, then joint observation.
            #    These gathers index the stacked params by per-game game type, so
            #    stepping is already correct for multiple game types.
            next_state_probs = self.all_env_parameters.transition[
                game_types_per_game, game_states, joint_a0, joint_a1
            ]  # [num_games, S]
            game_next_states = jax.vmap(
                lambda probs, k: distrax.Categorical(probs=probs).sample(seed=k)
            )(
                next_state_probs, jax.random.split(transition_key, num_games)
            )  # [num_games]

            obs_probs = self.all_env_parameters.observation[
                game_types_per_game, game_next_states, joint_a0, joint_a1
            ]  # [num_games, O, O]
            num_obs = obs_probs.shape[-1]
            flat_obs = jax.vmap(
                lambda probs, k: distrax.Categorical(probs=probs.reshape(-1)).sample(
                    seed=k
                )
            )(
                obs_probs, jax.random.split(obs_key, num_games)
            )  # [num_games]
            game_obs_by_role = jnp.stack(
                [flat_obs // num_obs, flat_obs % num_obs], axis=1
            )  # [num_games, num_roles]

            # 4. Each agent receives its own role's observation in its game.
            agent_observations = game_obs_by_role[
                initial_route.agent_game_assignment, initial_route.agent_role_assignment
            ]  # [num_agents]

            # 5. Per-agent belief + belief-estimate updates, dispatched by the
            #    agent's (game_type, role). agent_id (role) is now traced, so a single
            #    update per agent suffices (no role-hypothesis double compute).
            def per_agent_belief_updates(
                game_type, role, ego_belief_probs, other_est_probs, ego_obs, ego_action
            ):
                return self._agent_belief_updates(
                    game_type,
                    role,
                    distrax.Categorical(probs=ego_belief_probs),
                    distrax.Categorical(probs=other_est_probs),
                    ego_obs,
                    ego_action,
                )

            true_agent_belief_states, other_estimated_agent_belief_states = jax.vmap(
                per_agent_belief_updates
            )(
                agent_game_types,
                agent_roles,
                agent_initial_belief_states,
                est_other_initial_belief_states,
                agent_observations,
                agent_actions,
            )

            game_states = game_next_states
        else:
            # Communicate first: no environment step yet, beliefs stay at initial.
            true_agent_belief_states = agent_initial_belief_states
            other_estimated_agent_belief_states = est_other_initial_belief_states

        # The act-first path advanced the underlying DecPOMDP once, so it starts the
        # episode at in-game iteration 1; the communicate-first path is still at 0. At
        # reset the per-episode and cumulative env iterations coincide.
        underlying_iteration = jnp.asarray(
            1 if self.act_on_reset_before_communicating else 0, dtype=jnp.int32
        )

        # Fetch the first active communication block, keyed by cumulative_env_iteration
        # (== underlying_iteration at reset), and store it flat. The round cursors start
        # at the block's first round; no communication rounds have run yet.
        active_scheme = self.communication_scheme_fn(underlying_iteration)

        return StackedSignificationState(
            # Utterances are produced only by the communication phase, not reset.
            agent_utterance_actions_unrendered=None,
            agent_utterance_actions_rendered=None,
            agent_game_assignment=initial_route.agent_game_assignment,
            agent_role_assignment=agent_roles,
            game_types=game_types_per_game,
            underlying_env_iteration=underlying_iteration,
            cumulative_env_iteration=underlying_iteration,
            active_who_speaks=active_scheme.who_speaks,
            active_total_num_rounds=active_scheme.total_num_rounds,
            underlying_communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
            cumulative_communication_round_iterator=jnp.asarray(0, dtype=jnp.int32),
            # Reset starts on the utterance stage (the speaker emits first).
            communicative_round_stage=jnp.asarray(UTTERANCE_STAGE, dtype=jnp.int32),
            game_states=game_states,
            true_agent_belief_states=true_agent_belief_states,
            other_estimated_agent_belief_states=other_estimated_agent_belief_states,
            global_rng_key=key,
        )


if __name__ == "__main__":
    from routing import simple_routing_fn
    from communication_scheme import a_to_b_scheme_fn
    from envs.env_assembly import assemble_environments, guessing_game_spec

    # Build the stacked params + policy table from one game type (the guessing game).
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])

    env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
        communication_scheme_fn=a_to_b_scheme_fn,
        skip_first_communication_step=False,
    )

    # reset (communicate-first path): beliefs stay at initial.
    state = env.reset(jax.random.key(0))
    print("=== communicate-first reset ===")
    print("game_states:        ", state.game_states)
    print("underlying_iter:    ", state.underlying_env_iteration)
    print("active who_speaks:  ", state.active_who_speaks.tolist())
    print("active rounds:      ", state.active_total_num_rounds)
    print("round iters (u,c):  ", state.underlying_communication_round_iterator,
          state.cumulative_communication_round_iterator)
    print("round stage:        ", state.communicative_round_stage)
    print("true beliefs[0]:    ", state.true_agent_belief_states[0])
    assert state.underlying_env_iteration == 0, "communicate-first takes no env step at reset"
    assert state.underlying_communication_round_iterator == 0, "round cursor starts at 0"
    assert state.communicative_round_stage == UTTERANCE_STAGE, "reset starts on the utterance stage"

    # reset (act-first path): one joint action + belief updates before communicating.
    act_env = StackedSignificationDecPOMDP(
        num_agents=10,
        all_env_parameters=stacked_params,
        optimal_policies=optimal_policies,
        routing_fn=simple_routing_fn(num_agents=10, game_type_id=0, agents_per_game=2),
        communication_scheme_fn=a_to_b_scheme_fn,
        skip_first_communication_step=True,
    )
    act_state = act_env.reset(jax.random.key(1))
    print("=== act-first reset ===")
    print("game_states:        ", act_state.game_states)
    print("underlying_iter:    ", act_state.underlying_env_iteration)
    print("true beliefs[0]:    ", act_state.true_agent_belief_states[0])
    print("est other belief[0]:", act_state.other_estimated_agent_belief_states[0])
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
