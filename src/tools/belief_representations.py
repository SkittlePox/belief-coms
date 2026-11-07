"""Categorical belief updates for a dyadic DecPOMDP, as pure functions of its params.

Every function takes a single game's ``FlexibleEnvParams`` as its first argument. They
read only array shapes and contents -- never ``int(params.num_states)`` -- so ``params``
may itself be traced, e.g. one game type gathered out of the stacked params with a
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

    T[s, a_ego, a_other, s']                    from params.transition  [S, A, A, S]
    O[s', a_ego, a_other, o_ego, o_other]       from params.observation [S, A, A, O, O]

``agent_id`` is always the EGO's id (0 or 1) and may be traced.
"""

import distrax
import jax.numpy as jnp


def ego_frame(params, agent_id):
    """Transition and observation tensors with the agent axes ordered (ego, other).

    Returns ``(T, O)`` with ``T[s, a_ego, a_other, s']`` and
    ``O[s', a_ego, a_other, o_ego, o_other]``. ``agent_id`` may be traced: both
    orderings are built and one is selected, which is cheap for tensors this size.
    """
    T, O = params.transition, params.observation
    is_agent_0 = agent_id == 0
    T = jnp.where(is_agent_0, T, T.swapaxes(1, 2))
    O = jnp.where(is_agent_0, O, O.swapaxes(1, 2).swapaxes(3, 4))
    return T, O


def _normalize(unnormalized, fallback=None):
    """``unnormalized / sum``.

    A zero sum means the evidence was impossible under the model (e.g. an
    observation that no state reachable from the belief can emit). With no
    ``fallback`` this deliberately returns NaN: the environment should never
    produce such an observation, so a NaN belief points at a bug upstream rather
    than being papered over. With a ``fallback``, that is returned instead (the
    inner ``where`` keeps the gradient finite too).
    """
    total = jnp.sum(unnormalized)
    if fallback is None:
        return unnormalized / total
    safe_total = jnp.where(total > 0, total, 1.0)
    return jnp.where(total > 0, unnormalized / safe_total, fallback)


def update_with_observation_and_joint_action(params, belief, observation, joint_action, agent_id):
    """Bayesian belief update when the full joint action is known.

        b'(s') ∝ O_ego(o | s', a) · ∑_s T(s' | s, a) b(s)

    where O_ego is the ego's marginal of the joint observation model (the other
    agent's observation summed out).

    Args:
        params: A single game's FlexibleEnvParams.
        belief: The ego's current belief b(s), a distrax.Categorical over states.
        observation: The ego's observation at this timestep.
        joint_action: ``(agent_0_action, agent_1_action)`` -- agent order, not ego order.
        agent_id: The ego's id; selects whose observation marginal is used.

    Returns:
        b'(s') as a distrax.Categorical. All NaN if the observation is impossible
        under the model (see ``_normalize``).
    """
    T, O = ego_frame(params, agent_id)
    agent_0_action, agent_1_action = joint_action
    ego_action = jnp.where(agent_id == 0, agent_0_action, agent_1_action)
    other_action = jnp.where(agent_id == 0, agent_1_action, agent_0_action)

    predicted = belief.probs @ T[:, ego_action, other_action]  # [S']
    likelihood = O[:, ego_action, other_action, observation, :].sum(-1)  # [S']
    return distrax.Categorical(probs=_normalize(predicted * likelihood))


def update_with_observation_only(
    params,
    *,
    ego_belief,
    other_belief_estimate,
    ego_observation,
    ego_action,
    other_policy,
    agent_id,
):
    """Bayesian belief update when the other agent's action is unobserved.

    The other agent's action is marginalized under the policy we attribute to them,
    evaluated at our estimate of their belief:

        b'(s') ∝ ∑_{a_other} π*(b̄)(a_other) · O_ego(o | s', a) · ∑_s T(s' | s, a) b(s)

    with a = (ego_action, a_other).

    Args:
        params: A single game's FlexibleEnvParams.
        ego_belief: The ego's current belief b(s), a distrax.Categorical.
        other_belief_estimate: The ego's estimate of the other agent's belief b̄,
            fed to ``other_policy``.
        ego_observation: The ego's observation at this timestep.
        ego_action: The ego's own action at the previous timestep.
        other_policy: π*, mapping a belief Categorical to a Categorical over the other
            agent's actions.
        agent_id: The ego's id.

    Returns:
        b'(s') as a distrax.Categorical. All NaN if the observation is impossible
        under the model (see ``_normalize``).
    """
    T, O = ego_frame(params, agent_id)
    other_action_probs = other_policy(other_belief_estimate).probs  # [A_other]

    predicted = jnp.einsum("s,sxt->xt", ego_belief.probs, T[:, ego_action])  # [A_other, S']
    likelihood = O[:, ego_action, :, ego_observation, :].sum(-1).T  # [A_other, S']
    unnormalized = other_action_probs @ (predicted * likelihood)
    return distrax.Categorical(probs=_normalize(unnormalized))


def initial_belief(params, ego_observation, agent_id, reset_action=0):
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
        the prior (see ``_normalize``).
    """
    _, O = ego_frame(params, agent_id)
    prior = params.initial_state_distribution
    likelihood = O[:, reset_action, reset_action, ego_observation, :].sum(-1)  # [S]
    return distrax.Categorical(probs=_normalize(prior * likelihood))


def initial_other_belief_estimate(params, ego_observation, agent_id, reset_action=0):
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
        b̄_0 as a distrax.Categorical; the prior if the observation is impossible.
    """
    _, O = ego_frame(params, agent_id)
    prior = params.initial_state_distribution
    O_reset = O[:, reset_action, reset_action]  # [S, O_ego, O_other]

    weights = prior @ O_reset[:, ego_observation, :]  # [O_other]
    posteriors = (prior[:, None] * O_reset.sum(1)).T  # [O_other, S]
    mass = posteriors.sum(1, keepdims=True)
    posteriors = posteriors / jnp.where(mass > 0, mass, 1.0)

    # NOTE: Unlike initial_belief, this falls back to the prior (rather than NaN) when the
    # ego's reset observation is impossible under the prior. Consider dropping the fallback
    # so an impossible reset observation surfaces as NaN, as it does in initial_belief.
    return distrax.Categorical(probs=_normalize(weights @ posteriors, prior))


def update_other_belief_estimate_with_observation_only(
    params,
    *,
    ego_belief,
    other_belief_estimate,
    ego_observation,
    ego_action,
    other_policy,
    agent_id,
    ego_action_prior=None,
    mode="mixture",
):
    """Update the ego's *mean-belief estimate* of the other agent's belief.

    An in-place, history-free, deliberately approximate update

        (b̄, a_ego, o_ego)  ->  b̄'

    carrying no state between calls beyond b̄ itself. The ego observes neither a_other
    nor o_other, so both are marginalized out; the ego's own observation enters ONLY
    as evidence about what the other agent is likely to have seen and done.

    Per candidate a_other (a = (a_ego, a_other)):

      1. Their action distribution comes from their belief, since their behaviour last
         timestep was a function of that belief alone:   π(a_other) = π*(b̄)(a_other)

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

      5. Average over everything we cannot see:

             b̄'(s') = ∑_{a_other, o_other} w(a_other, o_other) · b_{a,o}(s')

    Two asymmetries hold this together, and both are easy to get wrong:

      - Step 4 never applies O_ego(o_ego | s') to b̄. Our observation is OUR private
        information; folding it into their posterior would credit them with knowledge
        they do not have. It only weights hypotheses, in step 3.

      - Step 4 also does not use our actual action, though step 2 does. If we press
        button 0, WE learn that state 0 is now impossible -- they do not. (Pinned down
        by test_other_belief_estimate_keeps_states_the_partner_cannot_rule_out.)

    mode
    ----
    "mixture" (default)
        Steps 4-5 as written: average their normalized POSTERIORS. The exact E[b_other]
        is itself a mixture of the posteriors they would hold under each possible
        history, so each component stays sharp and only the mixing blurs.

    "soft_evidence"
        Accumulate the same components UNnormalized and apply Bayes once (Jeffrey's
        rule): average their likelihood rather than their posterior, which pulls less
        hard. Kept for comparison; on the guessing game the two land close together.
        What matters far more than ``mode`` is passing the ego's own belief (not b̄) and
        seeding the estimate with ``initial_other_belief_estimate`` rather than the prior.

    The approximation, stated plainly
    ---------------------------------
    b̄ is a MEAN belief, and a mean belief is not a sufficient statistic in a DecPOMDP:
    the exact object is a joint distribution over (world state, other's belief).
    Collapsing that joint to its mean every step discards the correlation between "what
    the state is" and "what they think it is", and the error compounds:

      - It cannot represent "they are certain, but I don't know of what". That looks
        like a spread-out b̄, indistinguishable from "they are confused".
      - Step 1 feeds b̄ back into π*, so an error in b̄ biases the action likelihood,
        which biases the next b̄.

    For the exact treatment see memo-decpomdp's build_recursive_belief_model.

    Args:
        params: A single game's FlexibleEnvParams.
        ego_belief: The EGO's OWN belief at the previous timestep (from
            ``update_with_observation_only``, or ``initial_belief`` at reset). It is the
            only channel by which the ego's private knowledge reaches the estimate (step
            2). Substituting b̄ here roughly doubles the error, so it has no default.
        other_belief_estimate: The ego's current estimate b̄ of the other's belief.
        ego_observation: The ego's observation at this timestep.
        ego_action: The ego's OWN action at the previous timestep.
        other_policy: π*, mapping a belief Categorical to a Categorical over the other
            agent's actions.
        agent_id: The ego's id.
        ego_action_prior: P(h), what the OTHER agent assumes about OUR action, as a
            length-A array. Defaults to uniform -- the level-0 assumption, which is also
            what memo-decpomdp's exact model uses. Properly it is π_ego(bel[2]), the
            ego's estimate of the other's estimate of the ego's belief, and so on up a
            tower; tools.nested_belief.build_nested_belief_step carries that tower at
            arbitrary depth and supersedes this knob.
        mode: "mixture" (default) or "soft_evidence"; a static Python string.

    Returns:
        b̄' as a distrax.Categorical. If every branch has zero weight -- an
        (a_ego, o_ego) this estimate says is impossible -- b̄ is returned unchanged.
    """
    if ego_belief is None:
        raise ValueError("ego_belief is required; pass the ego's own belief, not b̄ (see docstring).")
    if mode not in ("mixture", "soft_evidence"):
        raise ValueError(f"mode must be 'mixture' or 'soft_evidence', got {mode!r}")

    T, O = ego_frame(params, agent_id)
    num_actions = T.shape[1]
    if ego_action_prior is None:
        ego_action_prior = jnp.ones(num_actions) / num_actions

    # 1. π(a_other), from the belief we think they hold.
    other_action_probs = other_policy(other_belief_estimate).probs  # [X]

    # 2-3. OUR side, under our actual action: weight each (a_other, o_other).
    #   s = state, t = next state, x = a_other, j = o_other
    predicted_ego = jnp.einsum("s,sxt->xt", ego_belief.probs, T[:, ego_action])  # [X, S']
    obs_weights = jnp.einsum("xt,txj->xj", predicted_ego, O[:, ego_action, :, ego_observation, :])  # [X, O_other]
    weights = other_action_probs[:, None] * obs_weights  # [X, O_other]

    # 4. THEIR side, marginalizing our unseen action h.
    predicted_other = jnp.einsum("s,shxt->hxt", other_belief_estimate.probs, T)  # [H, X, S']
    other_obs_marginal = O.sum(3)  # [S', H, X, O_other]
    posteriors = jnp.einsum("h,hxt,thxj->xjt", ego_action_prior, predicted_other, other_obs_marginal)  # [X, O_other, S']
    if mode == "mixture":
        mass = posteriors.sum(-1, keepdims=True)
        posteriors = posteriors / jnp.where(mass > 0, mass, 1.0)

    # 5. Average over everything we cannot see.
    unnormalized = jnp.einsum("xj,xjt->t", weights, posteriors)
    # NOTE: Unlike update_with_observation_only, this falls back to b̄ unchanged (rather
    # than NaN) when every (a_other, o_other) branch has zero weight, i.e. o_ego is
    # impossible under pred_ego and π*(b̄). Consider dropping the fallback so this surfaces
    # as NaN too. Caveat: it can also fire when b̄ is merely inaccurate -- π*(b̄) giving zero
    # probability to the partner's real action -- not only on an environment bug (the same
    # is true of update_with_observation_only, which already NaNs).
    return distrax.Categorical(probs=_normalize(unnormalized, other_belief_estimate.probs))
