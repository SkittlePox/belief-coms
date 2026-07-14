"""Expected-return estimation for a DecPOMDP under given policies and beliefs.

Two estimators of the *same* quantity — the expected discounted return of a
policy pair, rooted at the env's ``initial_state_distribution``:

  * :func:`exact_expected_returns` — exact, via a depth-limited Bellman lookahead
    that enumerates all actions / next states / observation pairs
    (``ExactReturnEvaluator``). Deterministic; cost grows fast with depth/size.
  * :func:`monte_carlo_returns` — sampled, by simulating episodes and averaging.
    Stochastic; converges to the exact value as ``num_rollouts -> inf``; scales to
    long horizons.

Both share the signature ``(env_spec, policies, initial_beliefs, ...)`` and return
a per-agent tuple of expected discounted returns, so they are drop-in swappable.
"""

import distrax
import jax
import jax.numpy as jnp

from envs.flexible_env import FlexibleEnv
from tools.belief_representations import CategoricalBeliefState


class ExactReturnEvaluator:
    """Exact depth-limited DecPOMDP lookahead from a single root state.

    The *model* is the FlexibleEnvParams; this class evaluates a policy pair
    against it. All dynamics come from the ``belief_state_factory``
    (CategoricalBeliefState); reward is gathered from its ``env_params.reward``.
    """

    def __init__(self, belief_state_factory):
        self.belief_state_factory = belief_state_factory
        self.env_params = belief_state_factory.env_params
        self.num_unique_states = belief_state_factory.num_unique_states
        self.num_unique_observations = belief_state_factory.num_unique_observations
        self.num_unique_actions = belief_state_factory.num_unique_actions
        self.joint_transition_function = belief_state_factory.joint_transition_function
        self.joint_observation_function = belief_state_factory.joint_observation_function
        self.joint_action_constructor = belief_state_factory.joint_action_constructor

    def evaluate_expected_returns(
        self,
        state,
        ego_policy,
        other_policy,
        ego_belief,
        other_belief,
        ego_agent_id=0,
        evaluation_depth=2,
        discount_factor=0.9,
    ):
        """Ego agent's expected discounted return from a single root ``state``.

        Depth-limited expansion of the Bellman equation:

            V(s, b_ego, b_other, d) =
                ∑_{a_ego} π_ego(a_ego | b_ego)
                ∑_{a_other} π_other(a_other | b_other)
                [ ∑_{s'} T(s'|s,a) R_ego(s,a,s')
                  + γ ∑_{s'} T(s'|s,a) ∑_{o_ego,o_other} O(o|s',a)
                    · V(s', b'_ego(o_ego,a), b'_other(o_other,a), d-1) ]

        R_ego is gathered from ``env_params.reward[ego_agent_id]``. The inner ∑_o is
        a Python loop over O² obs pairs so belief updates (which produce new
        distrax.Categorical distributions and cannot be vmapped) are computed once
        per pair, outside the vmap over next states.
        """
        belief_state_factory = self.belief_state_factory
        ego_action_dist = ego_policy(ego_belief)
        other_action_dist = other_policy(other_belief)

        def as_if_ego_acts(ego_action):
            def as_if_other_acts(other_action):
                joint_action = self.joint_action_constructor(ego_agent_id, ego_action, other_action)
                next_state_dist = self.joint_transition_function(state, joint_action)

                # ∑_{s'} T(s'|s,a) R_ego(s,a,s'): gather the ego agent's reward row
                # over next states and dot it with the transition distribution.
                ego_rewards_for_each_state = self.env_params.reward[ego_agent_id, state, joint_action[0], joint_action[1], :]
                immediate = jnp.sum(ego_rewards_for_each_state * next_state_dist.probs)

                if evaluation_depth <= 1:
                    return immediate * other_action_dist.prob(other_action)

                # This is the combinatorial explosion: this node spawns one
                # recursive call per (ego_action, other_action, ego_obs, other_obs,
                # next_state) = A²·O²·S children, all Python-unrolled at trace time.
                # Total compiled graph ≈ (A²·O²·S)^(depth-1). See the function-level
                # warning on exact_expected_returns; prefer monte_carlo_returns.
                future = jnp.zeros(())
                for ego_obs in range(self.num_unique_observations):
                    for other_obs in range(self.num_unique_observations):
                        new_ego_belief = belief_state_factory.update_with_observation_and_joint_action(
                            ego_belief, ego_obs, joint_action, agent_id=ego_agent_id
                        )
                        new_other_belief = belief_state_factory.update_with_observation_and_joint_action(
                            other_belief,
                            other_obs,
                            joint_action,
                            agent_id=int(not ego_agent_id),
                        )

                        def per_next_state(
                            next_state,
                            _ego_obs=ego_obs,
                            _other_obs=other_obs,
                            _new_ego_belief=new_ego_belief,
                            _new_other_belief=new_other_belief,
                        ):
                            joint_obs_dist = self.joint_observation_function(next_state, joint_action)
                            obs_prob = jax.lax.cond(
                                ego_agent_id == 0,
                                lambda _: belief_state_factory.joint_factory.prob(joint_obs_dist, _ego_obs, _other_obs),
                                lambda _: belief_state_factory.joint_factory.prob(joint_obs_dist, _other_obs, _ego_obs),
                                None,
                            )
                            future_v = self.evaluate_expected_returns(
                                next_state,
                                ego_policy,
                                other_policy,
                                _new_ego_belief,
                                _new_other_belief,
                                ego_agent_id,
                                evaluation_depth - 1,
                                discount_factor,
                            )
                            return next_state_dist.prob(next_state) * obs_prob * future_v

                        future = future + jnp.sum(jnp.nan_to_num(jax.vmap(per_next_state)(jnp.arange(self.num_unique_states))))

                return (immediate + discount_factor * future) * other_action_dist.prob(other_action)

            return jnp.sum(jax.vmap(as_if_other_acts)(jnp.arange(self.num_unique_actions))) * ego_action_dist.prob(ego_action)

        return jnp.sum(jax.vmap(as_if_ego_acts)(jnp.arange(self.num_unique_actions)))


def exact_expected_returns(
    env_spec,
    policies,
    initial_beliefs,
    evaluation_depth: int = 2,
    discount_factor: float = 0.9,
):
    """Exact per-agent expected discounted returns, rooted at the initial state dist.

    ⚠️  COST EXPLODES WITH DEPTH. The lookahead branches over every
    (ego action, other action, ego obs, other obs, next state) at each level, i.e.
    ``A² · O² · S`` children per node (guessing game: 4²·3²·4 = 576). Because the
    recursion and the observation-pair loop are pure Python, JAX *statically
    unrolls the entire tree at trace time*, so the COMPILED GRAPH — not just the
    runtime — grows like ``(A²·O²·S)^(evaluation_depth - 1)``:

        depth 2 -> ~576 nodes      depth 3 -> ~3.3e5      depth 4 -> ~1.9e8 (hangs)

    The blow-up is in *compilation* (graph size / memory), so it does not get
    faster on a bigger machine. Use this only for shallow depth (≤ ~3) on small
    envs as a reference / validation tool; for real horizons use
    :func:`monte_carlo_returns` (cost is linear in ``num_rollouts × max_steps`` and
    independent of depth — both functions share this signature, so swapping is a
    one-line change).

    Args:
        env_spec: An EnvSpec ``() -> (FlexibleEnvParams, policies)``. Only its params
            are used; ``policies`` below takes precedence.
        policies: Per-role policies indexed [role], each Categorical(belief) -> Categorical(action).
        initial_beliefs: Per-role initial beliefs indexed [role] (distrax.Categorical).
        evaluation_depth: Bellman lookahead depth.
        discount_factor: γ.

    Returns:
        Per-agent tuple of scalar expected discounted returns, each averaged over
        the env's ``initial_state_distribution``.
    """
    params, _ = env_spec()
    belief_factory = CategoricalBeliefState(params)
    evaluator = ExactReturnEvaluator(belief_factory)
    initial_state_distribution = params.initial_state_distribution

    def return_for_agent(ego_agent_id):
        ego_policy, other_policy = policies[ego_agent_id], policies[1 - ego_agent_id]
        ego_belief, other_belief = initial_beliefs[ego_agent_id], initial_beliefs[1 - ego_agent_id]
        total = jnp.zeros(())
        for s in range(evaluator.num_unique_states):
            total = total + initial_state_distribution[s] * evaluator.evaluate_expected_returns(
                s,
                ego_policy,
                other_policy,
                ego_belief,
                other_belief,
                ego_agent_id=ego_agent_id,
                evaluation_depth=evaluation_depth,
                discount_factor=discount_factor,
            )
        return total

    return tuple(return_for_agent(agent_id) for agent_id in range(len(policies)))


def monte_carlo_returns(
    env_spec,
    policies,
    initial_beliefs,
    num_rollouts: int = 1000,
    max_steps: int = 20,
    discount_factor: float = 0.9,
    rng_key=None,
):
    """Monte-Carlo per-agent expected discounted returns over sampled episodes.

    Plays full episodes (up to ``max_steps``) from the env's reset distribution,
    accumulating discounted per-agent reward until termination, and averages over
    ``num_rollouts`` episodes.

    Args / returns: same ``(env_spec, policies, initial_beliefs)`` contract as
    :func:`exact_expected_returns`; returns a per-agent tuple of scalar mean
    discounted returns.
    """
    params, _ = env_spec()
    env = FlexibleEnv(params)
    belief_factory = CategoricalBeliefState(params)

    agent_0_policy, agent_1_policy = policies
    initial_belief_agent_0, initial_belief_agent_1 = initial_beliefs

    def play_episode(rng):
        reset_rng, episode_rng, next_rng = jax.random.split(rng, 3)
        # The reset observation is unused: like the exact evaluator, agents act on
        # the supplied initial beliefs at step 0 and only update from the
        # observations produced by their own actions.
        env_state, _initial_obs = env.reset(reset_rng)

        def step_body(carry, step_rng):
            env_state, b0, b1, return_0, return_1, discount, done_before = carry
            a0_rng, a1_rng, env_rng = jax.random.split(step_rng, 3)

            agent_0_action = agent_0_policy(b0).sample(seed=a0_rng)
            agent_1_action = agent_1_policy(b1).sample(seed=a1_rng)
            joint_action = (agent_0_action, agent_1_action)

            next_state, (next_o0, next_o1), (r0, r1), done = env.step_env(env_rng, env_state, joint_action)

            active = 1.0 - done_before
            return_0 = return_0 + discount * r0 * active
            return_1 = return_1 + discount * r1 * active

            b0 = belief_factory.update_with_observation_and_joint_action(b0, next_o0, joint_action, agent_id=0)
            b1 = belief_factory.update_with_observation_and_joint_action(b1, next_o1, joint_action, agent_id=1)

            done_after = jnp.maximum(done_before, done.astype(jnp.float32))
            carry = (next_state, b0, b1, return_0, return_1, discount * discount_factor, done_after)
            return carry, None

        init_carry = (
            env_state,
            initial_belief_agent_0,
            initial_belief_agent_1,
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(1.0),
            jnp.array(0.0),
        )
        step_rngs = jax.random.split(episode_rng, max_steps)
        (_s, _b0, _b1, return_0, return_1, _disc, _done), _ = jax.lax.scan(step_body, init_carry, step_rngs)
        return (return_0, return_1), next_rng

    def scan_body(rng, _):
        returns, next_rng = play_episode(rng)
        return next_rng, returns

    init_rng = jax.random.key(0) if rng_key is None else rng_key
    _final_rng, all_returns = jax.lax.scan(scan_body, init_rng, None, length=num_rollouts)
    return tuple(jnp.mean(agent_returns) for agent_returns in all_returns)


if __name__ == "__main__":
    from envs.guessing_game import guessing_game_spec

    params, policies = guessing_game_spec()
    # Both agents start from the world prior -- there is no per-role initial belief.
    beliefs = (
        distrax.Categorical(probs=params.initial_state_distribution),
        distrax.Categorical(probs=params.initial_state_distribution),
    )

    exact = exact_expected_returns(guessing_game_spec, policies, beliefs, evaluation_depth=2)
    mc = monte_carlo_returns(guessing_game_spec, policies, beliefs, num_rollouts=2000)

    print("exact expected returns (depth 2):", [round(float(x), 4) for x in exact])
    print("monte-carlo returns (2000 eps):  ", [round(float(x), 4) for x in mc])
