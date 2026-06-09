"""
Simple PPO training for LargeGuessingGame via SignificationPOMDP.

Four agents: agent_0 and agent_1 each have their own utterance network and
belief network.  Each episode the environment assigns one agent as the sender
and the other as the receiver.

Each logical step consists of two micro-steps (the SignificationPOMDP protocol):
  Phase 0 — sender produces an utterance; receiver passes current belief through.
  Phase 1 — receiver processes the rendered utterance and returns an updated belief.
"""
import distrax
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

from envs.large_guessing_game import LargeGuessingGame
from signification_pomdp import SignificationPOMDP
from agents.belief_agents import ActorCriticBeliefAgent
from agents.utterance_agents import ActorCriticUtteranceAgent
from tools.belief_representations import CategoricalBeliefState
from tools.utterance_rendering import paint_multiple_splines

# ── Hyperparameters ───────────────────────────────────────────────────────────
N             = 4
IMAGE_DIM     = 32
NUM_SPLINES   = 2
UTTERANCE_DIM = NUM_SPLINES * 6   # = 12

NUM_STEPS       = 128
NUM_MINIBATCHES = 4
MINIBATCH_SIZE  = NUM_STEPS // NUM_MINIBATCHES
NUM_PPO_EPOCHS  = 4
NUM_UPDATES     = 500

LR            = 3e-5
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.01
MAX_GRAD_NORM = 0.5


# ── Transition ────────────────────────────────────────────────────────────────
class Transition(NamedTuple):
    # Cooperative game: both agents share the reward.
    reward: jnp.ndarray   # ()

    # Utterance agents — index 0 = agent_0, index 1 = agent_1.
    # Only the sender's utterance agent is "alive" each step.
    utt_obs_own_belief:  jnp.ndarray  # (2, N)
    utt_obs_other_est:   jnp.ndarray  # (2, N)
    utt_action:          jnp.ndarray  # (2, UTTERANCE_DIM)
    utt_log_prob:        jnp.ndarray  # (2,)
    utt_value:           jnp.ndarray  # (2,)
    utt_alive:           jnp.ndarray  # (2,)  1 for sender, 0 for the other

    # Belief agents — index 0 = agent_0, index 1 = agent_1.
    # Only the receiver's belief agent is "alive" each step.
    bel_obs_prior:       jnp.ndarray  # (2, N)
    bel_obs_utterance:   jnp.ndarray  # (2, IMAGE_DIM, IMAGE_DIM)
    bel_action:          jnp.ndarray  # (2, N)   sampled belief state (from Dirichlet)
    bel_log_prob:        jnp.ndarray  # (2,)
    bel_value:           jnp.ndarray  # (2,)
    bel_alive:           jnp.ndarray  # (2,)  1 for receiver, 0 for the other


# ── GAE ───────────────────────────────────────────────────────────────────────
def calculate_gae(values, rewards, last_value):
    """
    values:     (NUM_STEPS, 2) — one value per agent
    rewards:    (NUM_STEPS,)   — shared reward, broadcast to both agents
    last_value: (2,)           — bootstrap value (zeros since every episode ends done=True)
    Returns advantages and targets, both (NUM_STEPS, 2).
    """
    def _step(carry, t):
        gae, next_val = carry
        # done is always 1 in this env, so the (1 - done) terms vanish.
        # Writing the full formula for generality.
        done  = jnp.ones(2)
        delta = rewards[t] + GAMMA * next_val * (1 - done) - values[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - done) * gae
        return (gae, values[t]), gae

    _, advantages = jax.lax.scan(
        _step, (jnp.zeros(2), last_value), jnp.arange(NUM_STEPS), reverse=True
    )
    return advantages, advantages + values


# ── make_train ────────────────────────────────────────────────────────────────
def make_train():
    # Underlying guessing game, wrapped in the signification POMDP
    underlying_env = LargeGuessingGame(num_referents=N)

    initial_belief = distrax.Categorical(probs=jnp.ones(N) / N)
    belief_factory = CategoricalBeliefState(
        num_unique_states=N,
        num_unique_observations=N,
        num_unique_actions=N + 1,   # N possible guesses + 1 null action for the sender
        joint_transition_function=underlying_env._joint_transition_function,
        joint_observation_function=underlying_env._joint_observation_function,
        joint_action_constructor=underlying_env._joint_action_constructor,
    )

    env = SignificationPOMDP(
        underlying_env,
        utterance_size=UTTERANCE_DIM,
        initial_belief_distribution=initial_belief,
        belief_factory=belief_factory,
    )

    # One network class per role; each agent has its own independent parameters.
    utterance_network = ActorCriticUtteranceAgent(utterance_action_dim=UTTERANCE_DIM, belief_dim=N)
    belief_network    = ActorCriticBeliefAgent(input_utterance_shape=(IMAGE_DIM, IMAGE_DIM), belief_dim=N)

    def train(rng):

        # ── Initialize 4 agents ───────────────────────────────────────────────
        rng, k0u, k1u, k0b, k1b = jax.random.split(rng, 5)
        dummy_belief    = jnp.ones((1, N)) / N
        dummy_utterance = jnp.zeros((1, IMAGE_DIM, IMAGE_DIM))

        a0_utt_params = utterance_network.init(k0u, dummy_belief, dummy_belief)
        a1_utt_params = utterance_network.init(k1u, dummy_belief, dummy_belief)
        a0_bel_params = belief_network.init(k0b, dummy_belief, dummy_utterance)
        a1_bel_params = belief_network.init(k1b, dummy_belief, dummy_utterance)

        tx        = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
        a0_utt_opt = tx.init(a0_utt_params)
        a1_utt_opt = tx.init(a1_utt_params)
        a0_bel_opt = tx.init(a0_bel_params)
        a1_bel_opt = tx.init(a1_bel_params)

        # ── env_step: one full episode (phase 0 + phase 1) ───────────────────
        def env_step(carry, _):
            a0_utt_p, a1_utt_p, a0_bel_p, a1_bel_p, rng = carry
            rng, reset_key, p0_key, p1_key = jax.random.split(rng, 4)
            rng, k0u, k1u, k0b, k1b       = jax.random.split(rng, 5)

            # Reset → AugmentedState with Bayesian-initialised beliefs
            state, (obs_0, obs_1) = env.reset(reset_key)
            # obs_X = (own_belief, est_of_other_belief, other_utterance, is_sender)
            a0_belief, a0_est, _, _ = obs_0
            a1_belief, a1_est, _, _ = obs_1
            a0_bel_probs = a0_belief.probs   # (N,)
            a1_bel_probs = a1_belief.probs
            a0_est_probs = a0_est.probs
            a1_est_probs = a1_est.probs

            # Which agent is the sender this episode?  (does not change during the episode)
            sender_is_0 = (state.sender_agent == jnp.array(0)).astype(jnp.float32)

            # ── Phase 0: utterance ────────────────────────────────────────────
            # Both utterance agents produce an utterance from their current belief.
            a0_utt_pi, a0_utt_val = utterance_network.apply(a0_utt_p, a0_bel_probs[None], a0_est_probs[None])
            a1_utt_pi, a1_utt_val = utterance_network.apply(a1_utt_p, a1_bel_probs[None], a1_est_probs[None])

            a0_utt_action = a0_utt_pi.sample(seed=k0u)   # (1, UTTERANCE_DIM)
            a1_utt_action = a1_utt_pi.sample(seed=k1u)

            a0_utt_log_prob = a0_utt_pi.log_prob(a0_utt_action).squeeze()
            a1_utt_log_prob = a1_utt_pi.log_prob(a1_utt_action).squeeze()
            a0_utt_val = a0_utt_val.squeeze()
            a1_utt_val = a1_utt_val.squeeze()

            # Step through phase 0: records sender's utterance, flips message_status.
            # The receiver's belief_action is ignored here; pass current belief unchanged.
            phase0_a0 = (a0_utt_action.squeeze(), p0_key, a0_belief, a0_est)
            phase0_a1 = (a1_utt_action.squeeze(), p0_key, a1_belief, a1_est)
            state, _, _, _ = env.step_env(p0_key, state, (phase0_a0, phase0_a1))

            # Render the sender's utterance to an image for the receiver.
            sender_utt_params = jax.lax.cond(
                state.sender_agent == jnp.array(0),
                lambda _: a0_utt_action,
                lambda _: a1_utt_action,
                None,
            )
            sender_utt_img = paint_multiple_splines(sender_utt_params, IMAGE_DIM)  # (1, H, W)

            # Each agent receives the image only when the OTHER agent is the sender.
            null_img  = jnp.zeros((1, IMAGE_DIM, IMAGE_DIM))
            a0_recv_img = jax.lax.cond(  # agent_0 receives when agent_1 is sender
                state.sender_agent == jnp.array(1),
                lambda _: sender_utt_img, lambda _: null_img, None,
            )
            a1_recv_img = jax.lax.cond(  # agent_1 receives when agent_0 is sender
                state.sender_agent == jnp.array(0),
                lambda _: sender_utt_img, lambda _: null_img, None,
            )

            # ── Phase 1: belief update ────────────────────────────────────────
            # Both belief agents update their belief given the received utterance.
            a0_bel_pi, a0_bel_val = belief_network.apply(a0_bel_p, a0_bel_probs[None], a0_recv_img)
            a1_bel_pi, a1_bel_val = belief_network.apply(a1_bel_p, a1_bel_probs[None], a1_recv_img)

            a0_bel_sample = a0_bel_pi.sample(seed=k0b)   # (1, N) — sampled belief state
            a1_bel_sample = a1_bel_pi.sample(seed=k1b)

            a0_bel_log_prob = a0_bel_pi.log_prob(a0_bel_sample).squeeze()
            a1_bel_log_prob = a1_bel_pi.log_prob(a1_bel_sample).squeeze()
            a0_bel_val = a0_bel_val.squeeze()
            a1_bel_val = a1_bel_val.squeeze()

            # Wrap sampled beliefs back into Categoricals so the env can act on them.
            a0_bel_cat = distrax.Categorical(probs=a0_bel_sample.squeeze())
            a1_bel_cat = distrax.Categorical(probs=a1_bel_sample.squeeze())

            # Step through phase 1: receiver submits updated belief; underlying env steps.
            phase1_a0 = (env.null_utterance, p1_key, a0_bel_cat, a0_est)
            phase1_a1 = (env.null_utterance, p1_key, a1_bel_cat, a1_est)
            _, _, (reward, _), _ = env.step_env(p1_key, state, (phase1_a0, phase1_a1))

            # Alive masks: sender's utterance agent and receiver's belief agent are active.
            utt_alive = jnp.stack([sender_is_0,       1.0 - sender_is_0])   # [a0_is_sender, a1_is_sender]
            bel_alive = jnp.stack([1.0 - sender_is_0, sender_is_0])          # [a0_is_receiver, a1_is_receiver]

            transition = Transition(
                reward=reward,
                utt_obs_own_belief  = jnp.stack([a0_bel_probs, a1_bel_probs]),
                utt_obs_other_est   = jnp.stack([a0_est_probs, a1_est_probs]),
                utt_action          = jnp.concatenate([a0_utt_action, a1_utt_action], axis=0),
                utt_log_prob        = jnp.stack([a0_utt_log_prob, a1_utt_log_prob]),
                utt_value           = jnp.stack([a0_utt_val, a1_utt_val]),
                utt_alive           = utt_alive,
                bel_obs_prior       = jnp.stack([a0_bel_probs, a1_bel_probs]),
                bel_obs_utterance   = jnp.concatenate([a0_recv_img, a1_recv_img], axis=0),
                bel_action          = jnp.concatenate([a0_bel_sample, a1_bel_sample], axis=0),
                bel_log_prob        = jnp.stack([a0_bel_log_prob, a1_bel_log_prob]),
                bel_value           = jnp.stack([a0_bel_val, a1_bel_val]),
                bel_alive           = bel_alive,
            )
            return (a0_utt_p, a1_utt_p, a0_bel_p, a1_bel_p, rng), transition

        # ── _update_step: one full PPO iteration ──────────────────────────────
        def _update_step(runner_state, update_idx):
            a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o, \
            a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng = runner_state
            rng, collect_rng = jax.random.split(rng)

            # Collect rollout
            (a0_utt_p, a1_utt_p, a0_bel_p, a1_bel_p, _), traj = jax.lax.scan(
                env_step,
                (a0_utt_p, a1_utt_p, a0_bel_p, a1_bel_p, collect_rng),
                None, NUM_STEPS,
            )

            # GAE — last_value = 0 since every episode ends with done=True
            utt_adv, utt_tgt = calculate_gae(traj.utt_value, traj.reward, jnp.zeros(2))
            bel_adv, bel_tgt = calculate_gae(traj.bel_value, traj.reward, jnp.zeros(2))

            # ── PPO epochs ────────────────────────────────────────────────────
            def ppo_epoch(carry, _):
                a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o, \
                a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng = carry
                rng, perm_key = jax.random.split(rng)
                perm = jax.random.permutation(perm_key, NUM_STEPS)

                t       = jax.tree.map(lambda x: x[perm], traj)
                ua, ut  = utt_adv[perm], utt_tgt[perm]
                ba, bt  = bel_adv[perm], bel_tgt[perm]

                # Reshape into (NUM_MINIBATCHES, MINIBATCH_SIZE, ...)
                mb     = lambda x: x.reshape((NUM_MINIBATCHES, MINIBATCH_SIZE) + x.shape[1:])
                t_mb   = jax.tree.map(mb, t)
                ua_mb, ut_mb = mb(ua), mb(ut)
                ba_mb, bt_mb = mb(ba), mb(bt)

                def update_minibatch(carry, i):
                    a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o, \
                    a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o = carry
                    sl = jax.tree.map(lambda x: x[i], t_mb)

                    # Masked PPO loss for one utterance agent (agent_idx ∈ {0, 1})
                    def utt_loss(params, agent_idx):
                        pi, value = utterance_network.apply(
                            params,
                            sl.utt_obs_own_belief[:, agent_idx, :],
                            sl.utt_obs_other_est[:, agent_idx, :],
                        )
                        alive  = sl.utt_alive[:, agent_idx]
                        adv    = ua_mb[i, :, agent_idx]
                        tgt    = ut_mb[i, :, agent_idx]
                        ratio  = jnp.exp(pi.log_prob(sl.utt_action[:, agent_idx, :]) - sl.utt_log_prob[:, agent_idx])
                        n_adv  = (adv - adv.mean()) / (adv.std() + 1e-8)
                        actor  = (-jnp.minimum(ratio * n_adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * n_adv) * alive).sum() / (alive.sum() + 1e-8)
                        v_loss = (0.5 * jnp.square(value - tgt) * alive).sum() / (alive.sum() + 1e-8)
                        ent    = (pi.entropy() * alive).sum() / (alive.sum() + 1e-8)
                        return actor + VF_COEF * v_loss - ENT_COEF * ent

                    # Masked PPO loss for one belief agent (agent_idx ∈ {0, 1})
                    def bel_loss(params, agent_idx):
                        pi, value = belief_network.apply(
                            params,
                            sl.bel_obs_prior[:, agent_idx, :],
                            sl.bel_obs_utterance[:, agent_idx, :, :],
                        )
                        alive  = sl.bel_alive[:, agent_idx]
                        adv    = ba_mb[i, :, agent_idx]
                        tgt    = bt_mb[i, :, agent_idx]
                        ratio  = jnp.exp(pi.log_prob(sl.bel_action[:, agent_idx, :]) - sl.bel_log_prob[:, agent_idx])
                        n_adv  = (adv - adv.mean()) / (adv.std() + 1e-8)
                        actor  = (-jnp.minimum(ratio * n_adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * n_adv) * alive).sum() / (alive.sum() + 1e-8)
                        v_loss = (0.5 * jnp.square(value - tgt) * alive).sum() / (alive.sum() + 1e-8)
                        ent    = (pi.entropy() * alive).sum() / (alive.sum() + 1e-8)
                        return actor + VF_COEF * v_loss - ENT_COEF * ent

                    # Update all 4 agents
                    g = jax.grad(utt_loss)(a0_utt_p, 0)
                    u, a0_utt_o = tx.update(g, a0_utt_o);  a0_utt_p = optax.apply_updates(a0_utt_p, u)

                    g = jax.grad(utt_loss)(a1_utt_p, 1)
                    u, a1_utt_o = tx.update(g, a1_utt_o);  a1_utt_p = optax.apply_updates(a1_utt_p, u)

                    g = jax.grad(bel_loss)(a0_bel_p, 0)
                    u, a0_bel_o = tx.update(g, a0_bel_o);  a0_bel_p = optax.apply_updates(a0_bel_p, u)

                    g = jax.grad(bel_loss)(a1_bel_p, 1)
                    u, a1_bel_o = tx.update(g, a1_bel_o);  a1_bel_p = optax.apply_updates(a1_bel_p, u)

                    return (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                            a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o), None

                (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                 a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o), _ = jax.lax.scan(
                    update_minibatch,
                    (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                     a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o),
                    jnp.arange(NUM_MINIBATCHES),
                )
                return (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                        a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng), None

            (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
             a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng), _ = jax.lax.scan(
                ppo_epoch,
                (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                 a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng),
                None, NUM_PPO_EPOCHS,
            )

            mean_reward  = traj.reward.mean()
            mean_success = (traj.reward > 0).mean()
            jax.debug.print(
                "Update {i:4d} | reward {r:.3f} | success {s:.1%}",
                i=update_idx, r=mean_reward, s=mean_success,
            )

            runner_state = (a0_utt_p, a0_utt_o, a1_utt_p, a1_utt_o,
                            a0_bel_p, a0_bel_o, a1_bel_p, a1_bel_o, rng)
            return runner_state, (mean_reward, mean_success)

        # ── Outer scan over update steps ──────────────────────────────────────
        runner_state = (a0_utt_params, a0_utt_opt, a1_utt_params, a1_utt_opt,
                        a0_bel_params, a0_bel_opt, a1_bel_params, a1_bel_opt, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(NUM_UPDATES)
        )
        return runner_state, metrics

    return train


if __name__ == "__main__":
    rng = jax.random.key(42)
    train = make_train()
    runner_state, (rewards, successes) = train(rng)
    print(f"\nFinal mean reward:  {rewards[-1]:.3f}")
    print(f"Final success rate: {successes[-1]:.1%}")
