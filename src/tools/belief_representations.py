"""Categorical belief updates for a dyadic DecPOMDP, as pure functions of its env_params.

Every function takes a single game's ``FlexibleEnvParams`` as its first argument. They
read only array shapes and contents -- never ``int(env_params.num_states)`` -- so ``env_params``
may itself be traced, e.g. one game type gathered out of the stacked env_params with a
traced index. Cardinalities come from the (possibly padded) tensor shapes; padded
states and actions carry zero probability, so they drop out of every sum.

Beliefs go in and come out as ``distrax.Categorical`` over states; policies are
``Categorical -> Categorical`` callables (see envs.flexible_env.OptimalPolicy).

The ego frame
-------------
The dynamics tensors are indexed by agent position, (agent 0, agent 1). Almost every
update is naturally written from one agent's point of view -- "my action, your action,
my observation, your observation" -- so ``ego_frame`` swaps the agent axes once, when
the ego is agent 1. Everything downstream is then written for the ego alone, with no
per-agent branching:

    T[s, a_ego, a_other, s']                    from env_params.transition  [S, A, A, S]
    O[s', a_ego, a_other, o_ego, o_other]       from env_params.observation [S, A, A, O, O]

``ego_role`` is always the EGO's id (0 or 1) and may be traced.

Levels of belief
----------------
The ego tracks three distributions over states, each updated from the previous step's
values of all three:

    b_e   the ego's own belief                    update_with_observation_only
    b̄_o   the ego's estimate of the other's belief update_other_belief_estimate_with_observation_only
    b̿_e   the ego's estimate of the other's        update_second_order_ego_estimate_with_observation_only
          estimate of the ego's belief

b̄_o and b̿_e are mean estimates (see the "approximation" section of the b̄_o update).

Impossible observations
-----------------------
There are no fallbacks. If an observation has zero likelihood under the belief, the
final normalization divides by zero and the returned belief is NaN. The environment
should never produce such an observation, so a NaN belief deliberately points at a
problem elsewhere rather than being papered over.
"""

import distrax
import jax
import jax.numpy as jnp


def ego_frame(env_params, ego_role):
    """This function reframes Transition and observation tensors so that they are also ego-centric.

    Returns ``(T, O)`` with ``T[s, a_ego, a_other, s']`` and
    ``O[s', a_ego, a_other, o_ego, o_other]``. ``ego_role`` may be traced: both
    orderings are built and one is selected, which is cheap for tensors this size.
    """
    T, O = env_params.transition, env_params.observation
    is_agent_0 = ego_role == 0
    T = jnp.where(is_agent_0, T, T.swapaxes(1, 2))
    O = jnp.where(is_agent_0, O, O.swapaxes(1, 2).swapaxes(3, 4))
    return T, O


def update_with_observation_and_joint_action(env_params, belief, observation, joint_action, ego_role):
    """Bayesian belief update when the full joint action is known.

        b'(s') ∝ O_ego(o | s', a) · ∑_s T(s' | s, a) b(s)

    where O_ego is the ego's marginal of the joint observation model (the other
    agent's observation summed out).

    NOTE: This is mostly for learning purposes, the model doesn't actually use this because it doesn't see the joint action.

    Args:
        env_params: A single game's FlexibleEnvParams.
        belief: The ego's current belief b(s), a distrax.Categorical over states.
        observation: The ego's observation at this timestep.
        joint_action: ``(agent_0_action, agent_1_action)`` -- agent order, not ego order.
        ego_role: The ego's id; selects whose observation marginal is used.

    Returns:
        b'(s') as a distrax.Categorical. All NaN if the observation is impossible
        under the model (see the module docstring).
    """
    T, O = ego_frame(env_params, ego_role)
    agent_0_action, agent_1_action = joint_action
    ego_action = jnp.where(ego_role == 0, agent_0_action, agent_1_action)
    other_action = jnp.where(ego_role == 0, agent_1_action, agent_0_action)

    predicted = belief.probs @ T[:, ego_action, other_action]  # [S']
    likelihood = O[:, ego_action, other_action, observation, :].sum(-1)  # [S']
    unnormalized = predicted * likelihood
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def update_with_observation_only(
    env_params,
    *,
    ego_belief,
    other_belief_estimate,
    ego_observation,
    ego_action_tminus1,
    other_policy,
    ego_role,
):
    """Bayesian belief update when the other agent's action is unobserved.

    The other agent's action is marginalized under the policy we attribute to them,
    evaluated at our estimate of their belief:

        b'(s') ∝ ∑_{a_other} π*(b̄)(a_other) · O_ego(o | s', a) · ∑_s T(s' | s, a) b(s)

    with a = (ego_action, a_other).

    Args:
        env_params: A single game's FlexibleEnvParams.
        ego_belief: The ego's current belief b(s), a distrax.Categorical.
        other_belief_estimate: The ego's estimate of the other agent's belief b̄,
            fed to ``other_policy``.
        ego_observation: The ego's observation at this timestep.
        ego_action: The ego's own action at the previous timestep.
        other_policy: π*, mapping a belief Categorical to a Categorical over the other
            agent's actions.
        ego_role: The ego's id.

    Returns:
        b'(s') as a distrax.Categorical. All NaN if the observation is impossible
        under the model (see the module docstring).
    """
    T, O = ego_frame(env_params, ego_role)
    other_action_probs = other_policy(other_belief_estimate).probs  # [A_other]

    predicted = jnp.einsum("s,sxt->xt", ego_belief.probs, T[:, ego_action_tminus1])  # [A_other, S']
    likelihood = O[:, ego_action_tminus1, :, ego_observation, :].sum(-1).T  # [A_other, S']
    unnormalized = other_action_probs @ (predicted * likelihood)
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def initial_belief(env_params, ego_observation, ego_role, reset_action=0):
    """The ego's own belief at t=0, from the reset observation.

        b_0(s) ∝ P0(s) · O_ego(o_0 | s, reset_action)

    There is no action before the reset observation, so the update functions do not
    apply: passing a placeholder joint action to
    ``update_with_observation_and_joint_action`` would also run a TRANSITION, i.e.
    treat the placeholder as an action really taken. In the guessing game that means
    "agent 0 pressed button 0", which quietly corrupts the belief from the first step.

    ``reset_action`` must match the action the environment queries the observation
    tensor with at reset (FlexibleEnv.get_obs uses 0).

    Returns:
        b_0 as a distrax.Categorical. All NaN if the observation is impossible under
        the prior (see the module docstring).
    """
    _, O = ego_frame(env_params, ego_role)
    prior = env_params.initial_state_distribution
    likelihood = O[:, reset_action, reset_action, ego_observation, :].sum(-1)  # [S]
    unnormalized = prior * likelihood
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def initial_other_belief_estimate(env_params, ego_observation, ego_role, reset_action=0):
    """The ego's estimate of the other agent's belief at t=0, from the reset observation.

    The reset observation has no action before it, so it cannot go through
    ``update_other_belief_estimate_with_observation_only`` -- but it is still
    informative about the other agent, because the two agents' observations are
    correlated through the state. Starting the estimate at the prior throws away a
    real step of evidence.

    Same shape as the main estimate update, with the transition removed: weight each
    hypothetical o_other by how well it explains the o_ego we actually got, and mix
    the posteriors it would induce.

        b̄_0(s)          = ∑_{o_other} w(o_other) · b_{o_other}(s)
        w(o_other)      ∝ ∑_s P0(s) · O(o_ego, o_other | s, reset_action)
        b_{o_other}(s)  ∝ P0(s) · O_other(o_other | s, reset_action)

    ``reset_action`` is as in ``initial_belief``.

    Returns:
        b̄_0 as a distrax.Categorical. All NaN if the observation is impossible
        under the prior (see the module docstring).
    """
    _, O = ego_frame(env_params, ego_role)
    prior = env_params.initial_state_distribution
    O_reset = O[:, reset_action, reset_action]  # [S, O_ego, O_other]

    weights = prior @ O_reset[:, ego_observation, :]  # [O_other]
    posteriors = (prior[:, None] * O_reset.sum(1)).T  # [O_other, S]
    posteriors = posteriors / posteriors.sum(1, keepdims=True)  # NaN for an o_other they could never see
    # Not a fallback: zeroing only ZERO-weight branches keeps 0 · NaN from poisoning a
    # healthy estimate. A NaN branch with positive weight still comes through.
    posteriors = jnp.where(weights[:, None] > 0, posteriors, 0.0)

    unnormalized = weights @ posteriors
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def update_other_belief_estimate_with_observation_only(
    env_params,
    *,
    ego_belief,
    other_belief_estimate,
    ego_observation,
    ego_action,
    other_action_distribution,
    ego_role,
    ego_action_prior,
):
    """Update the ego's *mean-belief estimate* of the other agent's belief.

    An in-place, history-free, deliberately approximate update

        (b̄, a_ego, o_ego)  ->  b̄'.

    The ego observes neither a_other nor o_other, so both are marginalized out;
    the ego's own observation enters ONLY as evidence about what the other agent is likely to have seen and done.

    For each action the other agent may have taken, a_other (a = (a_ego, a_other)):

      1. Their action distribution is given by the caller:
         π(a_other) = other_action_distribution(a_other). Typically it is their policy
         applied to b̄, since their behaviour last timestep was a function of their belief.

      2. OUR side. We know our own action, so we condition on it: predict forward from
         our own belief under the joint action (a_ego, a_other).

             pred_ego(s') = ∑_s T(s' | s, a_ego, a_other) b_ego(s)

      3. Weight each (a_other, o_other) by how well it explains what WE saw. o_ego and
         o_other are correlated -- through s' and through the joint observation model
         -- so our own observation is evidence about theirs:

             w(a_other, o_other) ∝ π(a_other) · ∑_{s'} pred_ego(s') · O(o_ego, o_other | s', a)

         Normalization happens once, jointly over (a_other, o_other), so o_ego is also
         evidence about their ACTION.

      4. THEIR side. They never saw our action, so their prior must NOT condition on
         it; it marginalizes over what we might have done:

             pred_h(s')   = ∑_s T(s' | s, h, a_other) b̄(s)
             b_{a,o}(s')  ∝ ∑_h P(h) · pred_h(s') · O_other(o_other | s', h, a_other)

         P(h) is ``ego_action_prior``: what THEY think we did. It comes from a
         second-order estimate (what we estimate they believe we believe), which is
         calculated outside this function.

      5. Average over everything we cannot see:

             b̄'(s') = ∑_{a_other, o_other} w(a_other, o_other) · b_{a,o}(s')

    Two asymmetries hold this together, and both are easy to get wrong:

      - Step 4 never applies O_ego(o_ego | s') to b̄. Our observation is OUR private
        information; folding it into their posterior would credit them with knowledge
        they do not have. It only weights hypotheses, in step 3.

      - Step 4 also does not use our actual action, though step 2 does. If we press
        button 0, WE learn that state 0 is now impossible -- they do not. (Pinned down
        by test_other_belief_estimate_keeps_states_the_partner_cannot_rule_out.)

    The approximation, stated plainly
    ---------------------------------
    b̄ is a MEAN belief, and a mean belief is not a sufficient statistic in a DecPOMDP:
    the exact object is a joint distribution over (world state, other's belief).
    Collapsing that joint to its mean every step discards the correlation between "what
    the state is" and "what they think it is", and the error compounds:

      - It cannot represent "they are certain, but I don't know of what". That looks
        like a spread-out b̄, indistinguishable from "they are confused".
      - When the caller derives the step-1 action distribution from b̄, an error in b̄
        biases the action likelihood, which biases the next b̄.

    Args:
        env_params: A single game's FlexibleEnvParams.
        ego_belief: The EGO's OWN belief at the previous timestep (from
            ``update_with_observation_only``, or ``initial_belief`` at reset). It is the
            only channel by which the ego's private knowledge reaches the estimate (step
            2). Substituting b̄ here roughly doubles the error, so it has no default.
        other_belief_estimate: The ego's current estimate b̄ of the other's belief.
        ego_observation: The ego's observation at this timestep.
        ego_action: The ego's OWN action at the previous timestep.
        other_action_distribution: A Categorical over the OTHER agent's action at the
            previous timestep (the unobserved a_other in step 1).
        ego_role: The ego's id.
        ego_action_prior: P(h), a Categorical over OUR action at the previous timestep,
            as the OTHER agent predicts it (step 4). Required, with no default. It should
            come from a second-order belief estimate (what we estimate they believe we
            believe), calculated outside this function, e.g. our policy applied to the
            b̿_e from ``update_second_order_ego_estimate_with_observation_only``.

    Returns:
        b̄' as a distrax.Categorical. All NaN if every branch has zero weight, i.e.
        o_ego is impossible under pred_ego and other_action_distribution, or if a
        branch with positive weight is impossible from their seat, i.e. our estimates
        contradict each other (see the module docstring).
    """
    if ego_belief is None:
        raise ValueError("ego_belief is required; pass the ego's own belief, not b̄ (see docstring).")

    T, O = ego_frame(env_params, ego_role)

    # 1-3. OUR side, under our actual action: weight each (a_other, o_other).
    #   s = state, t = next state, x = a_other, j = o_other
    weights = _branch_weights(T, O, ego_belief, ego_observation, ego_action, other_action_distribution)  # [X, O_other]

    # 4. THEIR side, marginalizing our unseen action h.
    predicted_other = jnp.einsum("s,shxt->hxt", other_belief_estimate.probs, T)  # [H, X, S']
    other_obs_marginal = O.sum(3)  # [S', H, X, O_other]
    posteriors = jnp.einsum("h,hxt,thxj->xjt", ego_action_prior.probs, predicted_other, other_obs_marginal)  # [X, O_other, S']
    posteriors = posteriors / posteriors.sum(-1, keepdims=True)  # NaN where o_other is impossible from their seat
    # Not a fallback: zeroing only ZERO-weight branches keeps 0 · NaN from poisoning a
    # healthy estimate. A branch WE find possible but THEY find impossible means our
    # estimates contradict each other; its NaN comes through.
    posteriors = jnp.where(weights[..., None] > 0, posteriors, 0.0)

    # 5. Average over everything we cannot see.
    unnormalized = jnp.einsum("xj,xjt->t", weights, posteriors)
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def _branch_weights(T, O, ego_belief, ego_observation, ego_action, other_action_distribution):
    """Steps 1-3 of ``update_other_belief_estimate_with_observation_only``: how well each
    hypothesis (a_other, o_other) explains what the ego saw.

        pred_ego(s')         = ∑_s T(s' | s, a_ego, a_other) b_ego(s)
        w(a_other, o_other)  ∝ π(a_other) · ∑_{s'} pred_ego(s') · O(o_ego, o_other | s', a)

    ``T`` and ``O`` are in the ego frame. Returned unnormalized, shape [A_other, O_other];
    callers normalize once, after mixing.
    """
    #   s = state, t = next state, x = a_other, j = o_other
    predicted_ego = jnp.einsum("s,sxt->xt", ego_belief.probs, T[:, ego_action])  # [X, S']
    obs_weights = jnp.einsum("xt,txj->xj", predicted_ego, O[:, ego_action, :, ego_observation, :])  # [X, O_other]
    return other_action_distribution.probs[:, None] * obs_weights


def initial_second_order_ego_estimate(env_params, ego_observation, ego_role, reset_action=0):
    """The ego's estimate b̿_e of the other agent's estimate of the ego's belief, at t=0.

    Same shape as ``initial_other_belief_estimate``, one level up. For each reset
    observation o_other the other agent might have received, weight it by how well it
    explains the o_ego we actually got, and mix the estimate of US that the other agent
    would form from it:

        b̿_0(s)  = ∑_{o_other} w(o_other) · [initial_other_belief_estimate from THEIR seat, given o_other](s)
        w(o_other) ∝ ∑_s P0(s) · O(o_ego, o_other | s, reset_action)

    Our own reset observation enters only through w: they never saw it. No plug-in
    estimate is involved at reset, so this is the exact mean.

    ``reset_action`` is as in ``initial_belief``.

    Returns:
        b̿_0 as a distrax.Categorical. All NaN if the observation is impossible under
        the prior (see the module docstring).
    """
    _, O = ego_frame(env_params, ego_role)
    prior = env_params.initial_state_distribution
    weights = prior @ O[:, reset_action, reset_action, ego_observation, :]  # [O_other]

    def their_estimate_of_us(other_observation):
        return initial_other_belief_estimate(env_params, other_observation, 1 - ego_role, reset_action).probs

    branches = jax.vmap(their_estimate_of_us)(jnp.arange(weights.shape[0]))  # [O_other, S]
    # Not a fallback: an o_other they could never see has zero weight, and their estimate
    # given it is NaN. Zeroing those branches keeps 0 · NaN from poisoning a healthy estimate.
    branches = jnp.where(weights[:, None] > 0, branches, 0.0)

    unnormalized = weights @ branches
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())


def update_second_order_ego_estimate_with_observation_only(
    env_params,
    *,
    ego_belief,
    other_belief_estimate,
    second_order_ego_estimate,
    ego_observation,
    ego_action,
    other_action_distribution,
    ego_role,
    ego_action_prior,
    other_action_prior,
):
    """Update b̿_e, the ego's *mean* estimate of the other agent's estimate of the ego's belief.

        (b_e, b̄_o, b̿_e, a_ego, o_ego)  ->  b̿_e'

    Same structure as ``update_other_belief_estimate_with_observation_only``: weight
    every hypothesis (a_other, o_other) by how well it explains what we saw, compute
    what the other agent would do under that hypothesis, and average. The only
    difference is WHAT they compute. There it was their belief update; here it is their
    update of their estimate of us -- which is ``update_other_belief_estimate_with_observation_only``
    itself, run from their seat:

        w(a_other, o_other)  as in steps 1-3 of the first-order update (OUR side)

        U(a_other, o_other) = update_other_belief_estimate_with_observation_only(
            ego_belief                = b̄_o          their belief       (plug-in mean)
            other_belief_estimate     = b̿_e          their estimate of us (plug-in mean)
            ego_observation           = o_other
            ego_action                = a_other
            other_action_distribution = ego_action_prior    our action, as they predict it
            ego_role                  = the other agent's role
            ego_action_prior          = other_action_prior  their action, as they think we predict it
        )

        b̿_e'(s) = ∑_{a_other, o_other} w(a_other, o_other) · U(a_other, o_other)(s)

    Our private information stays private one level up as well: our actual a_ego and
    o_ego appear only in w. Inside U our action is always a hypothesis, drawn from
    ``ego_action_prior``.

    The approximation
    -----------------
    Every plug-in is a mean: b̄_o stands in for their belief, b̿_e for their estimate of
    us, and the action distributions are policies applied to means. The errors of the
    first-order update (see its docstring) therefore apply twice, and b̿_e is cruder
    than b̄_o.

    Args:
        env_params: A single game's FlexibleEnvParams.
        ego_belief: b_e, the ego's own belief at the previous timestep.
        other_belief_estimate: b̄_o, the ego's estimate of the other's belief at the
            previous timestep.
        second_order_ego_estimate: b̿_e, the ego's estimate of the other's estimate of the
            ego's belief at the previous timestep.
        ego_observation: The ego's observation at this timestep.
        ego_action: The ego's own action at the previous timestep.
        other_action_distribution: A Categorical over the OTHER agent's action at the
            previous timestep, as WE predict it (e.g. their policy applied to b̄_o).
        ego_role: The ego's id.
        ego_action_prior: A Categorical over OUR action at the previous timestep, as THEY
            predict it (e.g. our policy applied to b̿_e). The same quantity is the
            first-order update's ``ego_action_prior``.
        other_action_prior: A Categorical over THEIR action at the previous timestep, as
            they think WE predict it. It would come from a third-order estimate; until
            one exists, callers pass a uniform distribution explicitly.

    Returns:
        b̿_e' as a distrax.Categorical. All NaN if every branch has zero weight (o_ego
        is impossible), or if a branch with positive weight is impossible from their
        seat (see the module docstring).
    """
    T, O = ego_frame(env_params, ego_role)
    weights = _branch_weights(T, O, ego_belief, ego_observation, ego_action, other_action_distribution)  # [X, O_other]
    num_other_actions, num_other_observations = weights.shape

    def their_estimate_of_us(other_action, other_observation):
        return update_other_belief_estimate_with_observation_only(
            env_params,
            ego_belief=other_belief_estimate,
            other_belief_estimate=second_order_ego_estimate,
            ego_observation=other_observation,
            ego_action=other_action,
            other_action_distribution=ego_action_prior,
            ego_role=1 - ego_role,
            ego_action_prior=other_action_prior,
        ).probs

    over_observations = jax.vmap(their_estimate_of_us, in_axes=(None, 0))
    over_both = jax.vmap(over_observations, in_axes=(0, None))
    branches = over_both(jnp.arange(num_other_actions), jnp.arange(num_other_observations))  # [X, O_other, S]
    # Not a fallback: a hypothesis with zero weight may be impossible from their seat too,
    # which makes its branch NaN. Zeroing only ZERO-weight branches keeps 0 · NaN from
    # poisoning a healthy estimate; a NaN branch with positive weight still comes through.
    branches = jnp.where(weights[..., None] > 0, branches, 0.0)

    unnormalized = jnp.einsum("xj,xjt->t", weights, branches)
    return distrax.Categorical(probs=unnormalized / unnormalized.sum())
