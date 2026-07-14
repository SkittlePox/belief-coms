from stringprep import in_table_c11
from jax.random import categorical
import distrax, chex, jax
import jax.numpy as jnp
from functools import partial
from .distributions import *

class CategoricalBeliefState:
    """Represents a belief over a set of possible underlying states. States are assumed to be categorical, so a belief can be represented by a single distrax categorical distribution.

    """
    def __init__(self, env_params):
        """Build a belief-update engine for a single DecPOMDP.

        Args:
            env_params: A FlexibleEnvParams (or any object exposing ``transition``
                [S, A, A, S], ``observation`` [S, A, A, O, O], ``num_states`` and
                ``num_actions``). The dense dynamics tensors are read directly via
                gathers, replacing the per-env callables this class used to take.
                Must be concrete (not a tracer) so the cardinalities are static.

        Note: pass a SINGLE game's params here, not the stacked (leading
        game-type axis) FlexibleEnvParams produced by ``assemble_environments``.
        """
        self.env_params = env_params
        self.num_unique_states = int(env_params.num_states)
        self.num_unique_actions = int(env_params.num_actions)
        # A single observation alphabet shared by all agents, read off the
        # observation tensor's trailing axis.
        self.num_unique_observations = env_params.observation.shape[-1]
        # The joint observation model O(o0, o1 | s', a) is still correlated, but
        # both agents use the same cardinality, so the two per-agent marginals
        # are the same size (this is what removes the earlier lax.cond mismatch).
        self.joint_factory = JointCategoricalPair((self.num_unique_observations, self.num_unique_observations))

    def joint_transition_function(self, state, joint_action) -> distrax.Categorical:
        """T(s' | s, a0, a1) as a gather into env_params.transition -> [S']."""
        agent_0_action, agent_1_action = joint_action
        return distrax.Categorical(probs=self.env_params.transition[state, agent_0_action, agent_1_action])

    def joint_observation_function(self, next_state, joint_action) -> distrax.Categorical:
        """O(o0, o1 | s', a0, a1) flattened to [O * O] (JointCategoricalPair order)."""
        agent_0_action, agent_1_action = joint_action
        probs = self.env_params.observation[next_state, agent_0_action, agent_1_action].reshape(-1)
        return distrax.Categorical(probs=probs)

    def joint_action_constructor(self, agent_id, ego_action, other_action):
        """Order (ego, other) actions into the (agent_0, agent_1) joint action."""
        return jax.lax.cond(
            agent_id == 0,
            lambda _: (ego_action, other_action),
            lambda _: (other_action, ego_action),
            None,
        )
    
    def update_with_observation_and_joint_action(
        self, 
        belief_distribution: distrax.Categorical, 
        observation, 
        previous_joint_action,
        agent_id = 0
    ):
        """Perform a Bayesian belief update given a new observation.

        Implements the standard POMDP belief update rule:

            b'(s') ∝ O(o | a, s') ∑_s T(s' | s, a) b(s)

        where:
          - b(s)  is the prior belief (probability of being in state s)
          - T(s' | s, a) is the transition model (probability of moving to s' from s under joint action a)
          - O(o | a, s') is the observation model (probability of observing o in state s' after action a)
          - b'(s') is the unnormalized posterior belief over next states s'

        The result is renormalized by distrax.Categorical to form a valid distribution.

        Because the observation model is joint over all agents — O(o1, o2 | s) — the agent's
        own marginal observation likelihood O(oi | s') is computed by marginalizing out the
        other agent's observation before evaluating the likelihood of the received observation.

        Args:
            belief_distribution: Current belief b(s) as a distrax.Categorical over states.
            observation: The observation received by this agent at the current timestep.
            previous_joint_action: The joint action taken by all agents at the previous timestep,
                used to condition the transition and observation models.

        Returns:
            A new distrax.Categorical representing the updated belief b'(s') (automatically
            normalized by distrax).
        """

        def state_likelihood(next_state):
            # ∑_s T(s' | s, a) b(s)
            def transition_contrib(state):
                return self.joint_transition_function(state, previous_joint_action).prob(next_state) * belief_distribution.prob(state)

            predicted_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))

            # O(oi | s') — marginalize the joint observation model down to this agent's view
            joint_obs = self.joint_observation_function(next_state, previous_joint_action)
            marginal_obs = jax.lax.cond(
                agent_id == 0,
                lambda _: self.joint_factory.marginalize_var2(joint_obs),
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                None
            )

            return marginal_obs.prob(observation) * predicted_prior

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def update_with_observation_only(
        self, 
        ego_belief_distribution: distrax.Categorical, 
        other_belief_distribution_estimate: distrax.Categorical, 
        ego_observation, 
        previous_ego_action, 
        other_optimal_policy,
        agent_id = 0
    ):
        """Perform a Bayesian belief update when the other agent's action is unobserved.

        Because the joint action is not directly observed, we marginalize over the other
        agent's possible actions weighted by their policy, giving:

            b'(s') ∝ ∑_a [ O(o | a, s') ∑_s T(s' | s, a) b(s) · π*(b̄_S)(a) ]

        where:
          - b(s)       is the ego agent's prior belief over states
          - T(s' | s, a) is the transition model conditioned on the joint action a
          - O(o | a, s') is the ego agent's marginal observation likelihood in state s'
          - π*(b̄_S)(a_other) is the probability the other agent takes action a_other
                              under their optimal policy given their belief b̄_S
          - The sum over a reduces to a sum over a_other since previous_ego_action is known;
            the joint action is reconstructed as a flat index from (ego_action, other_action)

        Args:
            ego_belief_distribution: The ego agent's current belief b(s) as a distrax.Categorical.
            other_belief_distribution_estimate: An estimate of other agent's current belief b̄_S, passed to
                other_optimal_policy to obtain a distribution over their actions....
            ego_observation: The observation received by the ego agent at this timestep.
            previous_ego_action: The ego agent's own action at the previous timestep (known).
            other_optimal_policy: A callable π* that takes a belief distribution and returns a
                distrax.Categorical over the other agent's actions.

        Returns:
            A new distrax.Categorical representing the updated belief b'(s').
        """
        # π*(b̄_S) — other agent's action distribution under their optimal policy
        other_action_dist = other_optimal_policy(other_belief_distribution_estimate)

        def state_likelihood(next_state):
            def contribution_for_other_action(other_action):
                joint_action = self.joint_action_constructor(agent_id, previous_ego_action, other_action)

                # O(oi | s') — marginalize joint obs model down to this agent's view
                joint_obs = self.joint_observation_function(next_state, joint_action)
                marginal_obs = jax.lax.cond(
                    agent_id == 0,
                    lambda _: self.joint_factory.marginalize_var2(joint_obs),
                    lambda _: self.joint_factory.marginalize_var1(joint_obs),
                    None
                )
                obs_likelihood = marginal_obs.prob(ego_observation)

                # ∑_s T(s' | s, a) b(s)
                def transition_contrib(state):
                    return self.joint_transition_function(state, joint_action).prob(next_state) * ego_belief_distribution.prob(state)

                transition_prior = jnp.sum(jax.vmap(transition_contrib)(jnp.arange(self.num_unique_states)))

                return obs_likelihood * transition_prior * other_action_dist.prob(other_action)

            return jnp.sum(jax.vmap(contribution_for_other_action)(jnp.arange(self.num_unique_actions)))

        probs = jax.vmap(state_likelihood)(jnp.arange(self.num_unique_states))
        return distrax.Categorical(probs=probs)

    def initial_belief(
        self,
        ego_observation,
        agent_id = 0,       # This is the ego agent's id!
        reset_action = 0,
    ):
        """The ego agent's own belief at t=0, from the reset observation.

            b_0(s) ∝ P0(s) · O_ego(o_0 | s, reset_action)

        There is no action before the reset observation, so the other update methods do
        not apply: passing a placeholder joint action to
        ``update_with_observation_and_joint_action`` would silently run a TRANSITION as
        well, i.e. treat the placeholder as an action really taken. In the guessing game
        that means "agent 0 pressed button 0", which quietly corrupts the belief from the
        first step. Use this at reset instead.

        `reset_action` must match what the environment queries the observation tensor with
        at reset (FlexibleEnv.get_obs uses 0).
        """
        states = jnp.arange(self.num_unique_states)
        joint_action = (reset_action, reset_action)

        def state_likelihood(state):
            joint_obs = self.joint_observation_function(state, joint_action)
            marginal_obs = jax.lax.cond(
                agent_id == 0,
                lambda _: self.joint_factory.marginalize_var2(joint_obs),
                lambda _: self.joint_factory.marginalize_var1(joint_obs),
                None,
            )
            return (marginal_obs.prob(ego_observation)
                    * self.env_params.initial_state_distribution[state])

        probs = jax.vmap(state_likelihood)(states)
        return distrax.Categorical(probs=probs / jnp.sum(probs))

    def initial_other_belief_estimate(
        self,
        ego_observation,
        agent_id = 0,       # This is the ego agent's id!
        reset_action = 0,
    ):
        """The ego's estimate of the other agent's belief at t=0, from the reset observation.

        The reset observation has no action before it, so it cannot be fed to
        update_other_belief_estimate_with_observation_only -- but it is still informative
        about the other agent, because the two agents' observations are correlated through
        the state. Dropping it (starting the estimate at the prior and only updating from
        t=1) throws away a real step of evidence.

        Same shape as the main update, with the transition removed: weight each hypothetical
        o_other by how well it explains the o_ego we actually got, and mix the posteriors it
        would induce.

            b̄_0(s) = ∑_{o_other} w(o_other) · b_{o_other}(s)
            w(o_other) ∝ ∑_s P0(s) · O(o_ego, o_other | s, reset_action)
            b_{o_other}(s) ∝ P0(s) · O_other(o_other | s, reset_action)

        `reset_action` is the no-op joint action the environment queries the observation
        tensor with at reset (FlexibleEnv.get_obs uses 0).
        """
        states = jnp.arange(self.num_unique_states)
        prior = self.env_params.initial_state_distribution
        joint_action = (reset_action, reset_action)

        def observation_rows(state):
            joint_obs = self.joint_observation_function(state, joint_action)
            grid = joint_obs.probs.reshape(
                self.num_unique_observations, self.num_unique_observations
            )
            joint_with_ego = jax.lax.cond(
                agent_id == 0,
                lambda _: grid[ego_observation, :],
                lambda _: grid[:, ego_observation],
                None,
            )
            other_marginal = jax.lax.cond(
                agent_id == 0,
                lambda _: self.joint_factory.marginalize_var1(joint_obs).probs,
                lambda _: self.joint_factory.marginalize_var2(joint_obs).probs,
                None,
            )
            return joint_with_ego, other_marginal

        joint_with_ego, other_marginal = jax.vmap(observation_rows)(states)   # (S, O) each

        weights = prior @ joint_with_ego                                      # (O,)

        posteriors = (prior[:, None] * other_marginal).T                      # (O, S)
        mass = jnp.sum(posteriors, axis=1, keepdims=True)
        posteriors = posteriors / jnp.where(mass > 0, mass, 1.0)

        unnormalized = weights @ posteriors                                   # (S,)
        total = jnp.sum(unnormalized)
        probs = jnp.where(total > 0, unnormalized / jnp.where(total > 0, total, 1.0), prior)
        return distrax.Categorical(probs=probs)

    def update_other_belief_estimate_with_observation_only(
        self,
        other_belief_distribution_estimate: distrax.Categorical,
        ego_belief_distribution: distrax.Categorical,
        ego_observation,
        previous_ego_action,
        other_optimal_policy,
        agent_id = 0,   # This is the ego agent's id!
        ego_action_prior = None,
        mode = "mixture",
    ):
        """Update the ego agent's *mean-belief estimate* of the other agent's belief.

        An in-place, history-free, deliberately approximate update. It maps

            (b̄, a_ego, o_ego)  ->  b̄'

        carrying no state between calls beyond b̄ itself. The ego observes neither a_other
        nor o_other, so both are marginalized out; the ego's own observation enters ONLY
        as evidence about what the other agent is likely to have seen and done.

        Per candidate a_other (a = the joint action, ordered by role):

          1. Their action distribution comes from their belief, since their behaviour last
             timestep was a function of that belief alone:   π(a_other) = π*(b̄)(a_other)

          2. OUR side. We know our own action, so we condition on it: predict forward from
             our own belief under the joint action (a_ego, a_other).

                 pred_ego(s') = ∑_s T(s' | s, a_ego, a_other) b_ego(s)

          3. Weight each (a_other, o_other) by how well it explains what WE saw. o_ego and
             o_other are correlated -- through s' and through the joint observation model
             -- so our own observation is evidence about theirs:

                 w(a_other, o_other) ∝ π(a_other) · ∑_{s'} pred_ego(s') · O(o_ego, o_other | s', a)

          4. THEIR side. They never saw our action, so their prior must NOT condition on
             it; it marginalizes over what we might have done:

                 pred_other_h(s') = ∑_s T(s' | s, a_ego', a_other) b̄(s)
                 b_{a,o}(s') ∝ ∑_{a_ego'} P(a_ego') · pred_other_h(s') · O_other(o_other | s', ·)

          5. Average over everything we cannot see:

                 b̄'(s') = ∑_{a_other, o_other} w(a_other, o_other) · b_{a,o}(s')

        Two asymmetries hold this together, and both are easy to get wrong:

          - Step 4 never applies O_ego(o_ego | s') to b̄. Our observation is OUR private
            information; the other agent did not see it, and folding it in would credit
            them with knowledge they do not have. It only weights hypotheses, in step 3.

          - Step 4 also does not use our actual action, though step 2 does. If we press
            button 0, WE learn that state 0 is now impossible -- they do not. Building
            their posterior under our real action would rule out a state they still fully
            believe in. (This is precisely the bug that
            test_other_belief_estimate_keeps_states_the_partner_cannot_rule_out pins down.)

        mode
        ----
        "mixture" (default)
            Steps 4-5 above: average their POSTERIORS. This is the right shape for the
            quantity being approximated -- the exact E[b_other] is itself a mixture of the
            posteriors they would hold under each possible history -- so each component
            stays sharp and only the mixing blurs.

        "soft_evidence"
            Accumulate the same components UNnormalized and apply Bayes once (Jeffrey's
            rule) -- i.e. average their likelihood rather than their posterior. In
            principle this pulls less hard, because an averaged likelihood is flatter in s'
            than any single one.

        In practice the two land in the same place on the guessing game (worst error 0.165
        vs 0.169), so do not agonize over the choice. The two things that DO matter, by an
        order of magnitude more than `mode`, are passing `ego_belief_distribution` and
        seeding the estimate with `initial_other_belief_estimate` rather than the prior.

        Measured against the exact model (memo-decpomdp, guessing game, ego waits twice and
        sees symbols 1, 2, 2; exact E[b_other] = [0.875, 0.06, 0.06, 0]):

            mixture                              [0.71, 0.13, 0.16, 0]   worst err 0.17
            soft_evidence                        [0.71, 0.12, 0.18, 0]   worst err 0.17
            mixture, no ego_belief_distribution  [0.50, 0.29, 0.21, 0]   worst err 0.37

        The approximation, stated plainly
        ---------------------------------
        b̄ is a MEAN belief, and a mean belief is not a sufficient statistic in a DecPOMDP:
        the exact object is a joint distribution over (world state, other's belief), and
        their belief is a function of their whole history. Collapsing that joint to its
        mean every step -- which carrying a single Categorical forces -- discards the
        correlation between "what the state is" and "what they think it is", and the error
        compounds. Two consequences worth knowing about:

          - It cannot represent "they are certain, but I don't know of what". That looks
            like a spread-out b̄, indistinguishable from "they are confused".
          - Step 1 feeds b̄ back into π*, so an error in b̄ biases the action likelihood,
            which biases the next b̄.

        For the exact treatment see memo-decpomdp's build_recursive_belief_model, which
        keeps the joint and does the recursion properly. This is the cheap version; it is
        fine as long as you know which corner you cut.

        Args:
            other_belief_distribution_estimate: The ego's current estimate of the other
                agent's belief b̄(s), as a distrax.Categorical over states.
            ego_observation: The observation the ego received at this timestep.
            previous_ego_action: The ego's OWN action at the previous timestep (not joint).
            other_optimal_policy: A callable π* mapping a belief to a distrax.Categorical
                over the other agent's actions.
            agent_id: The EGO agent's id (0 or 1). May be traced.
            ego_belief_distribution: REQUIRED. The EGO's OWN belief at the previous
                timestep (from update_with_observation_only, or initial_belief at reset).
                It is the only channel by which the ego's private knowledge reaches the
                estimate: the ego uses it to work out which states IT may have entered, and
                hence which observations the other agent plausibly received. This used to
                default to b̄, which silently roughly doubled the error (0.17 -> 0.37), so
                it is now mandatory rather than a trap.
            ego_action_prior: What the OTHER agent assumes about OUR action, as a length-A
                array -- they never observed it. Defaults to uniform, the level-0
                assumption, which is also what memo-decpomdp's exact model uses, so the two
                remain comparable. This is where the I-POMDP regress is cut: modelling it
                properly would require their estimate of OUR belief, and then ours of
                theirs, forever.
            mode: "mixture" (default) or "soft_evidence"; a static Python string.

        Returns:
            A new distrax.Categorical representing the updated estimate b̄'(s').
        """
        states = jnp.arange(self.num_unique_states)
        actions = jnp.arange(self.num_unique_actions)

        # What the OTHER agent assumes about OUR action -- they never saw it. Uniform is
        # the level-0 cop-out, and it is only the DEFAULT, not the answer. What they
        # actually do is infer our action from THEIR estimate of OUR belief, i.e.
        #
        #     ego_action_prior = pi_ego(bel[2])
        #
        # where bel[2] is the ego's estimate of the other agent's estimate of the ego's
        # belief. That is level 2 of a tower, and it does not stop there. Rather than
        # hand-roll the recursion -- the conditioning rules are subtle and easy to get
        # wrong -- use tools.nested_belief.build_nested_belief_step, which carries the whole
        # tower at arbitrary depth and derives the rules from memo's nested `thinks[...]`.
        # Feed its bel[2] in here, or just use it instead of this method.
        if ego_action_prior is None:
            ego_action_prior = jnp.ones(self.num_unique_actions) / self.num_unique_actions

        if ego_belief_distribution is None:
            # This used to fall back to b̄, and that silently roughly doubled the error
            # (worst case 0.17 -> 0.37): b̄ is the OTHER agent's view of the world, so
            # standing it in here asks them to guess their own observation from their own
            # belief, which is nearly uninformative. The ego's own belief is the only
            # channel by which the ego's private knowledge reaches the estimate, so it is
            # required rather than quietly defaulted.
            raise ValueError(
                "ego_belief_distribution is required: it is what tells the estimate which "
                "states the EGO may have entered, and hence which observations the other "
                "agent plausibly received. Pass the ego's own belief (from "
                "update_with_observation_only, or initial_belief at reset)."
            )

        # 1. Their action distribution, from the belief we think they hold. Their
        #    behaviour last timestep was a function of that belief alone.
        policy_probs = other_optimal_policy(other_belief_distribution_estimate).probs  # (A,)

        def transition_prior(belief, joint_action):
            """∑_s T(s' | s, a) belief(s), as a vector over s'."""
            return jax.vmap(lambda next_state: jnp.sum(jax.vmap(
                lambda state: self.joint_transition_function(state, joint_action).prob(next_state)
                * belief.prob(state)
            )(states)))(states)

        def observation_rows(next_state, joint_action):
            joint_obs = self.joint_observation_function(next_state, joint_action)
            # Row-major (var1 = agent 0's obs, var2 = agent 1's obs), per
            # JointCategoricalPair and FlexibleEnv.get_obs.
            grid = joint_obs.probs.reshape(
                self.num_unique_observations, self.num_unique_observations
            )
            # P(o_ego, o_other | s', a) with the ego's ACTUAL observation pinned, as a
            # function of o_other. This term carries the ego's private information into
            # its guess about what the other agent saw.
            joint_with_ego = jax.lax.cond(
                agent_id == 0,
                lambda _: grid[ego_observation, :],   # ego is var1, so o_other is var2
                lambda _: grid[:, ego_observation],   # ego is var2, so o_other is var1
                None,
            )
            # O_other(o_other | s', a): the other agent's own marginal likelihood.
            # marginalize_var1 sums out agent 0, leaving agent 1's marginal.
            other_marginal = jax.lax.cond(
                agent_id == 0,
                lambda _: self.joint_factory.marginalize_var1(joint_obs).probs,
                lambda _: self.joint_factory.marginalize_var2(joint_obs).probs,
                None,
            )
            return joint_with_ego, other_marginal

        def as_if_other_took_action(other_action):
            # --- OUR side. We know our own action, so we condition on it. ---
            actual_joint = self.joint_action_constructor(
                agent_id, previous_ego_action, other_action)

            predicted_ego = transition_prior(ego_belief_distribution, actual_joint)  # (S',)
            joint_with_ego, _ = jax.vmap(
                lambda s_: observation_rows(s_, actual_joint))(states)               # (S', O)

            # 3. How likely each o_other is, jointly with the o_ego we actually saw:
            #       ∑_{s'} pred_ego(s') · O(o_ego, o_other | s', a)
            #    pred_EGO, not pred_other: this is the ego reasoning about which states IT
            #    may have entered and what the other agent would have seen there. It is the
            #    only channel by which the ego's private knowledge reaches the estimate.
            #    Unnormalized -- normalization happens once, jointly over (a_other,
            #    o_other), below, so o_ego is also allowed to be evidence about their ACTION.
            observation_weights = predicted_ego @ joint_with_ego                     # (O,)

            # --- THEIR side. They never saw our action, so they must NOT condition on it. ---
            # This is the asymmetry that makes the whole thing subtle. If we press button 0,
            # WE know state 0 is now impossible -- but they do not, and modelling their
            # posterior under our actual action would rule out a state they still believe
            # in. So their prior marginalizes over what we might have done.
            def under_hypothetical_ego_action(hypothetical_ego_action):
                joint = self.joint_action_constructor(
                    agent_id, hypothetical_ego_action, other_action)
                predicted = transition_prior(other_belief_distribution_estimate, joint)  # (S',)
                _, other_marginal = jax.vmap(
                    lambda s_: observation_rows(s_, joint))(states)                      # (S', O)
                # Their unnormalized posterior mass, per (o_other, s'), for this hypothesis.
                return (predicted[:, None] * other_marginal).T                           # (O, S')

            # 4. Their posterior had they seen o_other, marginalizing our unseen action:
            #       b_{a,o}(s') ∝ ∑_{a_ego'} P(a_ego') · pred_{a_ego'}(s') · O_other(o | s', ·)
            per_hypothesis = jax.vmap(under_hypothetical_ego_action)(actions)   # (A_ego, O, S')
            posteriors = jnp.einsum("h,hos->os", ego_action_prior, per_hypothesis)   # (O, S')
            mass = jnp.sum(posteriors, axis=1, keepdims=True)
            normalized = posteriors / jnp.where(mass > 0, mass, 1.0)

            return observation_weights, normalized, posteriors

        # (A, O), (A, O, S), (A, O, S)
        observation_weights, posteriors, unnormalized_posteriors = \
            jax.vmap(as_if_other_took_action)(actions)

        # P(a_other, o_other | b̄, a_ego, o_ego), unnormalized.
        weights = policy_probs[:, None] * observation_weights          # (A, O)

        if mode == "mixture":
            # 5a. Average their POSTERIORS, each normalized first. This is the right shape
            #     for the thing being approximated: the exact E[b_other] is itself a mixture
            #     of the posteriors the other agent would hold under each history, so mixing
            #     normalized posteriors keeps every component sharp and blurs only across
            #     components.
            unnormalized = jnp.einsum("ao,aos->s", weights, posteriors)
        elif mode == "soft_evidence":
            # 5b. Do not normalize the components first -- just accumulate the unnormalized
            #     mass and Bayes once (Jeffrey's rule). Equivalent to averaging their
            #     LIKELIHOOD rather than their posterior, which pulls less hard, because an
            #     averaged likelihood is flatter in s' than any single one. Kept for
            #     comparison; see the mode note in the docstring.
            unnormalized = jnp.einsum("ao,aos->s", weights, unnormalized_posteriors)
        else:
            raise ValueError(f"mode must be 'mixture' or 'soft_evidence', got {mode!r}")

        # If every branch died -- an (a_ego, o_ego) this estimate says is impossible --
        # leave the estimate untouched rather than emitting NaNs.
        total = jnp.sum(unnormalized)
        probs = jnp.where(
            total > 0,
            unnormalized / jnp.where(total > 0, total, 1.0),
            other_belief_distribution_estimate.probs,
        )
        return distrax.Categorical(probs=probs)
