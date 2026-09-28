"""Independent PPO (IPPO)."""

from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState

from training.config import ExperimentConfig
from training.optimizer import OptimizerConfig
from agents.belief_agents import BeliefAgentConfig
from agents.utterance_agents import UtteranceAgentConfig
from tools.utterance_rendering import paint_multiple_splines
from tools.visualization import plot_belief_states, plot_utterances
from training.rollout import collect_rollout
from training.advantages import compute_advantages
from training.ppo_update import ppo_update


def _batched_train_states(network: nn.Module, tx, rng, num_agents, *init_inputs) -> TrainState:
    """Init ``num_agents`` independent parameter sets and wrap each in a TrainState.

    A single network *definition* is shared by every agent, but each agent gets its own
    randomly initialized params (one per split of ``rng``) and its own optimizer state.
    Rather than a Python list of TrainStates, we ``vmap`` the per-agent init so the result
    is one TrainState pytree with a leading ``num_agents`` axis on the array leaves
    (params, opt_state, step); the static fields (``apply_fn``, ``tx``) are shared. That
    batched layout is what lets the downstream update ``vmap``/scan over agents.

    ``init_inputs`` are the dummy forward-pass inputs (batch dimension of 1) whose shapes
    seed the parameter shapes; they are identical across agents, so they stay outside the
    vmap.
    """

    def init_one(key):
        params = network.init(key, *init_inputs)
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    return jax.vmap(init_one)(jax.random.split(rng, num_agents))


def initialize_belief_agents(
    agent_config: BeliefAgentConfig,
    optimizer_config: OptimizerConfig,
    rng,
    num_agents: int,
    input_utterance_shape: Sequence[int],
    belief_dim: int,
) -> TrainState:
    """Build ``num_agents`` initialized belief-agent train states.

    Returns the vmapped TrainState (leading ``num_agents`` axis on params/opt_state). The
    network itself isn't returned: the TrainState already carries ``apply_fn`` (the shared
    ``network.apply``, a static field) and ``params``, which is all callers need to run the
    model.
    """
    network = agent_config.build(input_utterance_shape=input_utterance_shape, belief_dim=belief_dim)
    tx = optimizer_config.build()

    # Dummy inputs to shape-init the params: a valid previous belief (uniform over the
    # simplex -- zeros would send BeliefActor's log-prior to -inf) and a rendered
    # utterance image. Batch dim of 1; shared across agents.
    dummy_belief = jnp.full((1, belief_dim), 1.0 / belief_dim)
    dummy_utterance = jnp.zeros((1, *input_utterance_shape))

    return _batched_train_states(network, tx, rng, num_agents, dummy_belief, dummy_utterance)


def initialize_utterance_agents(
    agent_config: UtteranceAgentConfig,
    optimizer_config: OptimizerConfig,
    rng,
    num_agents: int,
    utterance_action_dim: int,
    belief_dim: int,
) -> TrainState:
    """Build ``num_agents`` initialized utterance-agent train states.

    Same contract as ``initialize_belief_agents``; the utterance agent reads the sender's
    own belief and its estimate of the receiver's belief (both length ``belief_dim``).
    """
    network = agent_config.build(utterance_action_dim=utterance_action_dim, belief_dim=belief_dim)
    tx = optimizer_config.build()

    # Dummy inputs: the sender's own belief and its estimate of the receiver's belief.
    dummy_own_belief = jnp.full((1, belief_dim), 1.0 / belief_dim)
    dummy_estimate = jnp.full((1, belief_dim), 1.0 / belief_dim)

    return _batched_train_states(network, tx, rng, num_agents, dummy_own_belief, dummy_estimate)


def make_train(config: ExperimentConfig):

    # Each sub-config knows how to build its runtime function, so resolution is
    # uniform: assignment -> AssignmentFn, communication -> CommunicationSchemeFn.
    assignment_fn = config.role_assignment.build()
    scheme_fn = config.communication.build()
    num_agents = config.role_assignment.num_agents

    # The env config assembles the stacked DecPOMDP params + optimal-policy table from
    # the selected games and wires in the two runtime functions above.
    env = config.environment.build(
        num_agents=num_agents,
        assignment_fn=assignment_fn,
        communication_scheme_fn=scheme_fn,
    )

    # Env-derived shapes the agents need. belief_dim is the (padded) world-state
    # cardinality (length of every belief vector). A belief agent consumes a rendered
    # utterance image of side utterance_image_dim; the utterance agent emits a flat
    # utterance of length utterance_action_dim (which must match the env's).
    belief_dim = env.belief_dim
    utterance_action_dim = config.environment.utterance_action_dim
    image_dim = config.environment.utterance_image_dim
    input_utterance_shape = (image_dim, image_dim)

    # Belief agents consume a rendered utterance image; the rollout paints each raw
    # utterance vector onto an (image_dim x image_dim) canvas via this closure. The leading
    # rng is unused today but kept in the signature so stochastic rendering (e.g. jitter /
    # sampled ink) can be added later without touching every call site.
    render_utterance_fn = lambda rng, utterances: paint_multiple_splines(utterances, image_dim)

    # How agents reason during rollouts (first-order vs. ToM); resolved once here and
    # threaded into every env_step by collect_rollout.
    inference_strategy = config.inference.build()

    def train(rng):
        # One independent population per role: num_agents belief agents and num_agents
        # utterance agents, each a distinct parameter set + optimizer state batched under
        # a leading num_agents axis.
        belief_rng, utterance_rng, env_rng, loop_rng = jax.random.split(rng, 4)
        belief_train_states = initialize_belief_agents(
            config.belief_agents,
            config.belief_optimizer,
            belief_rng,
            num_agents,
            input_utterance_shape=input_utterance_shape,
            belief_dim=belief_dim,
        )
        utterance_train_states = initialize_utterance_agents(
            config.utterance_agents,
            config.utterance_optimizer,
            utterance_rng,
            num_agents,
            utterance_action_dim=utterance_action_dim,
            belief_dim=belief_dim,
        )

        def run_debug():
            """DEBUG-only: push random inputs through every agent and collect their outputs.

            Kept in its own function so it stays clearly separable from the real training
            code. Closes over the two populations and the env-derived shapes; returns a dict
            of inspection arrays + matplotlib Figures (merged into train's return below).

            Each population's params carry a leading num_agents axis, so we vmap apply over
            that axis (in_axes=0 for params) while broadcasting ONE shared random input batch
            to all agents (in_axes=None). Because every agent sees identical inputs, any
            difference in their outputs is attributable purely to parameter-init randomness --
            which is what makes this a sanity check on the init. Outputs have a leading
            [num_agents, batch, ...] shape. We pull the concrete distribution parameters + a
            sample out INSIDE the vmap (where each distribution is un-batched) rather than
            returning the distribution objects -- a vmapped distrax distribution's batch_shape
            metadata doesn't track the agent axis, which breaks its .loc/.mean accessors.
            """
            debug_batch = 8
            db_belief, db_image, db_bs, db_us = jax.random.split(jax.random.fold_in(rng, 0xDEB), 4)

            # One shared belief batch (valid simplex points -- BeliefActor takes its log) fed
            # to BOTH populations: as the belief agents' previous_belief and as the utterance
            # agents' own_belief and receiver estimate. Same distributions across every agent.
            debug_belief = jax.random.dirichlet(db_belief, jnp.ones(belief_dim), shape=(debug_batch,))
            debug_utterance_image = jax.random.uniform(db_image, (debug_batch, *input_utterance_shape))
            print(debug_belief)

            def belief_forward(params, prev_belief, utterance_image, sample_key):
                dist, value = belief_train_states.apply_fn(params, prev_belief, utterance_image)
                # concentration [batch, belief_dim]; sample is a next-belief on the simplex.
                return dist.concentration, dist.sample(seed=sample_key), value

            # -> concentration [num_agents, batch, belief_dim], sample [num_agents, batch, belief_dim],
            #    value [num_agents, batch]
            belief_debug_concentration, belief_debug_sample, belief_debug_value = jax.vmap(belief_forward, in_axes=(0, None, None, None))(
                belief_train_states.params, debug_belief, debug_utterance_image, db_bs
            )

            def utterance_forward(params, own_belief, estimate_belief, sample_key):
                dist, value = utterance_train_states.apply_fn(params, own_belief, estimate_belief)
                # loc/scale_diag [batch, utterance_action_dim]; sample is an utterance vector.
                return dist.loc, dist.scale_diag, dist.sample(seed=sample_key), value

            # -> loc/scale/sample [num_agents, batch, utterance_action_dim], value [num_agents, batch]
            utterance_debug_loc, utterance_debug_scale, utterance_debug_sample, utterance_debug_value = jax.vmap(
                utterance_forward, in_axes=(0, None, None, None)
            )(utterance_train_states.params, debug_belief, debug_belief, db_us)

            # Render each sampled utterance to a canvas so it can be viewed as an image. The
            # utterance vector is spline control points; paint_multiple_splines maps over its
            # leading (batch) axis, so we vmap it once more over the agent axis. Requires
            # utterance_action_dim to be a multiple of 6 (6 params per spline).
            # -> [num_agents, batch, utterance_image_dim, utterance_image_dim]
            utterance_debug_render = jax.vmap(lambda utterances: paint_multiple_splines(utterances, image_dim))(utterance_debug_sample)

            # Plot one shared-input batch element (index 0) across every agent, so the spread
            # between panels reflects parameter-init randomness alone. This is debug output,
            # so it saves itself to disk here rather than pushing figures out to the caller.
            agent_titles = [f"agent {i}" for i in range(num_agents)]
            belief_debug_fig = plot_belief_states(belief_debug_sample[:, 0], titles=agent_titles, fig_title="Sampled beliefs (debug, batch 0)")
            utterance_debug_fig = plot_utterances(
                utterance_debug_render[:, 0], image_dim=image_dim, titles=agent_titles, fig_title="Sampled utterances (debug, batch 0)"
            )
            belief_debug_fig.savefig("belief_debug.png", dpi=150)
            utterance_debug_fig.savefig("utterance_debug.png", dpi=150)

            return dict(
                belief_debug_concentration=belief_debug_concentration,
                belief_debug_sample=belief_debug_sample,
                belief_debug_value=belief_debug_value,
                utterance_debug_loc=utterance_debug_loc,
                utterance_debug_scale=utterance_debug_scale,
                utterance_debug_sample=utterance_debug_sample,
                utterance_debug_value=utterance_debug_value,
                utterance_debug_render=utterance_debug_render,
                belief_debug_fig=belief_debug_fig,
                utterance_debug_fig=utterance_debug_fig,
            )

        debug_outputs = run_debug()

        # --- Training loop -----------------------------------------------------------
        # Reset the env ONCE up front; the resulting env_state is threaded through the scan
        # carry so the rollout continues across update steps rather than restarting each
        # iteration. (The stacked env never terminates -- it re-routes at episode boundaries
        # internally -- so a single reset is all it needs; see StackedSignificationDecPOMDP.)
        env_state, init_obs = env.reset(env_rng)

        # One _update_step is a single training iteration; we scan it num_epochs times.
        # The carry is both populations' train states, the env state, and the iteration
        # counter -- each TrainState already holds its own optimizer state (opt_state) and
        # step, so nothing else is needed. Per-iteration randomness is derived by folding the
        # iteration index into the base loop key rather than pre-splitting num_epochs keys via
        # xs: with potentially very many iterations, materializing that whole key array up
        # front is wasteful, whereas fold_in costs nothing to carry. The scanned output is
        # per-iteration metrics, stacked along the leading (epoch) axis.
        def _update_step(carry, _):
            belief_train_states, utterance_train_states, env_state, last_obs, iteration = carry
            step_rng = jax.random.fold_in(loop_rng, iteration)
            rollout_rng, update_rng = jax.random.split(step_rng)

            # 1) Roll out a trajectory: scan env_step num_steps times, running both agent
            #    populations each stage and stepping the env. The final env_state/last_obs
            #    are threaded back into the carry so the rollout continues across iterations;
            #    transitions carries a leading [num_steps] axis (see training.rollout).
            env_state, last_obs, _rollout_rng, transitions = collect_rollout(
                belief_train_states,
                utterance_train_states,
                env,
                env_state,
                last_obs,
                rollout_rng,
                config.num_steps_per_epoch,
                render_utterance_fn,
                image_dim,
                inference_strategy,
            )

            # 2) Turn the rollout into per-population advantages + value targets. The env is
            #    staged (utterance/belief alternate; reward only on act steps), so this is a
            #    per-population masked GAE -- see training.advantages. Bootstrap values
            #    default to zeros for now (window treated as episode end); a continuing
            #    rollout would run each critic on the final carried obs instead.
            advantages = compute_advantages(transitions, config.gamma, config.gae_lambda)

            # 3) PPO update: form the clipped losses for each population (masked to its own
            #    stages) and take config.ppo.update_epochs gradient steps, vmapped over the
            #    num_agents axis. Returns the updated (batched) train states + loss metrics.
            belief_train_states, utterance_train_states, ppo_metrics = ppo_update(
                update_rng,
                belief_train_states,
                utterance_train_states,
                transitions,
                advantages,
                config.ppo,
            )

            # 4) Metrics for this iteration.
            metrics = {
                "utterance_advantage_mean": advantages.utterance_advantages.mean(),
                "belief_advantage_mean": advantages.belief_advantages.mean(),
                **ppo_metrics,
            }

            carry = (belief_train_states, utterance_train_states, env_state, last_obs, iteration + 1)
            return carry, metrics

        initial_carry = (belief_train_states, utterance_train_states, env_state, init_obs, jnp.asarray(0, dtype=jnp.int32))
        (belief_train_states, utterance_train_states, env_state, _, _), metrics = jax.lax.scan(
            _update_step,
            initial_carry,
            xs=None,
            length=config.num_epochs,
        )

        return dict(
            belief_train_states=belief_train_states,
            utterance_train_states=utterance_train_states,
            metrics=metrics,
            **debug_outputs,
        )

    return train  #
