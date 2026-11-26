# Codebase overview & open work

What's broken, missing or fat in `src/`, package by package. Revised 2026-11-25. Items have
IDs so they can be referenced in discussion; delete them when resolved.

- **B** — bugs (confirmed by running code unless noted)
- **G** — gaps: features that are missing or stubbed
- **D** — design questions that need a decision before code
- **T** — trimming: fat, dead code, duplication, hygiene

---

## Headline

- Tests pass (168: 119 belief-function, 7 stacked env, 30 Q policies, 3 GAE, 9 agents).
  Nothing covers the rollout or the PPO update.
- Training **runs end to end but does not train**: belief log-probs are `+inf` (B1), env
  beliefs go NaN after a few acts (B5), multi-round schemes lose their learning signal
  (B2), and the loss masks credit agents for discarded actions (B3).
- The env takes communication updates to every belief level, but the e2e agents only
  produce utterances and the listener's b_e; the rollout passes the four b̄_o / b̿_e inputs
  through unchanged. What should produce them is open in
  [src/agents/NOTES.md](../src/agents/NOTES.md).
- The only environment is the guessing game, a weak test of communication (D4).

### Suggested order

1. Trimming: T1, T5, T6, T9–T11.
2. B1–B3 and B5, plus a smoke test of `make_train` that asserts finite metrics. B1 and B5
   share a root cause in the belief action, so settle D3 first.
3. Agent outputs for the new estimate inputs ([src/agents/NOTES.md](../src/agents/NOTES.md)).
4. wandb logging (G5).
5. An env where communication is necessary (D4), to prove learning works.

---

## Current design (short)

- **Belief levels.** Each agent tracks b_e (own belief), b̄_o (estimate of its partner's
  belief) and b̿_e (estimate of its partner's estimate of it). All start at the game's
  initial state distribution and are updated from observations at every act
  (`tools/belief_representations.py`). Storage conventions are in the comment block above
  `DialogState` in the stacked env.
- **Policies** are Boltzmann over hand-specified Q functions, π(a) ∝ exp(β·Q(b_e, b̄_o, a)).
  `envs.flexible_env` defines `QFunction` and `GameSpec`; `envs.env_assembly` builds the
  `BoltzmannPolicyBundle`. The guessing game's presser Q is the exact infinite-horizon
  optimum (checked against brute-force lookahead in `tests/test_q_policies.py`).
- **Untracked levels.** Predictions that need b⃛_o or b⁗_e use the Boltzmann policies with
  a stand-in belief U in each untracked slot: U is the initial state distribution on an
  episode's first act and uniform over all real states afterwards. A lower level is never
  substituted for an untracked one. See `_untracked_level_predictions`.
- **Stages.** `step_env` dispatches to `_step_env_utterance_stage` (speakers' utterances,
  b̄_o and b̿_e) or `_step_env_belief_stage` (listeners' b_e, b̄_o and b̿_e, then the act
  and episode boundary on a block's last round). Its docstring lists the six action
  inputs and which rows each writes. `get_obs` returns every belief level on both stages;
  only the utterances are NaN on the utterance stage, when there is nothing to hear.

---

## `training`

**B1. Belief log-prob is `+inf`, so belief PPO is NaN from iteration 0.** The guessing-game
prior has zero mass on the done state. With `multiply_by_ego_belief` on (the default), the
e2e belief agent's logits are `log b + log L`, so `log b = -inf` there, the Dirichlet
concentration is `1e-4`, sampled coordinates come out as exactly `0.0`, and `log_prob` is
`+inf` on every belief stage (still visible in `inspect_rollout`). The same exact zeros feed
B5. Fix options are D3.

**B2. Multi-round schemes have no learning signal.** In e.g. `a_to_b_thrice`, non-final
belief stages carry a NaN reward placeholder that is never filled in. In `_masked_gae`
([advantages.py](../src/training/advantages.py)) the NaN propagates backward to almost
every decision step of both populations, and `_ppo_loss` then `nan_to_num`s those
advantages to 0. Single-round schemes are unaffected.

**B3. Loss masks are per-step, not per-agent.** `stage_mask` has no agent axis
([ppo_update.py](../src/training/ppo_update.py)). In `a_to_b`, role-1 agents' utterances are
discarded by the env but still PPO-trained against the shared reward; likewise role-0
agents' belief actions. `Transition` needs per-agent speaks/listens masks, recorded per step
(they vary per episode under `simple_assignment_fn`).

**B4. Truncation treated as termination.** Bootstrap values default to 0 at the end of each
rollout window, but the env continues across windows. Fix: run each critic on the final
carried obs.

**B6 (unconfirmed). Utterance PPO ratios drift fast.** In a short smoke run,
`utterance_clip_frac` is ~0.68 from the first iteration and `utterance_approx_kl` reaches ~4
by the second. Could be the high-dimensional unbounded Gaussian, or a mismatch between the
stored and recomputed log-prob. Check that the ratio is exactly 1 on the first minibatch of
the first epoch.

**G4. ToM executors are placeholders.** `execute_tom_*` in
[inference_agents.py](../src/agents/inference_agents.py) behave like the e2e agents, and they
vmap the partner population index-aligned instead of gathering the real partner via
`game_roles.partner_agent`.

**G5. No wandb logging.** [wandb_log.py](../src/training/wandb_log.py) is empty and
`launch.py` discards what `train()` returns. Simplest option: a `jax.debug.callback` in
`_update_step` that calls `wandb.log`.

**T1. `run_debug` inside `train`.** Pushes random inputs through every agent, prints, and
saves two PNGs into the CWD on every `train` call, and is why `train` can't be jitted (T2).
Delete it or move it to a standalone script.

**T2. `train` is not jitted.** Follows from T1.

**Minor.** Two comments in `_update_step` ([ippo.py](../src/training/ippo.py)) still say
`num_steps` where they mean `rollout_length`; `inspect_rollout`'s `--num-steps` flag also
counts stages and could become `--rollout-length`.

---

## `agents`

See also [src/agents/NOTES.md](../src/agents/NOTES.md) for the current agents and the open
question of what produces b̄_o and b̿_e.

**D3. What is a belief action?** The e2e belief agent uses `softplus(logits)` as a Dirichlet
concentration, with `logits = log b + log L` when `multiply_by_ego_belief` is on (the
default) and `log L` alone when it is off. Besides B1, the "on" form is not a Bayesian
update: it is not invariant to adding a constant to `log L`, and the Dirichlet's mean is
`∝ softplus(log posterior)`, not the posterior. The "off" form avoids B1's `-inf` but has
no support guarantee at all. The listener also adopts its sample
verbatim as its true belief, so it can drop support it never ruled out (the true-belief
half of B5). Options:
- a Dirichlet restricted to the support of `b` (zero-support coordinates fixed at 0 and
  excluded from `log_prob`);
- make the action a log-likelihood vector and compute the belief as `normalize(b · L)` (a
  "likelihood action"), which makes the support guarantee exact.

**Perf.** The conv trunk has no pooling: 32×32×64 → Dense(128) is ~8.4M params per trunk,
and the actor and critic each have one, so ~17M per belief agent, times `num_agents`.

**Minor.** Utterance samples are unbounded Gaussians around a sigmoid mean; the renderer
clips them to [0, 1].

---

## `communication`

**B5. Env beliefs go NaN within a few acts.** Measured 2026-11-16 with untrained agents:
most seeds NaN by the first or second act. The NaN tripwire is reporting a real problem: an
untrained listener adopts a Dirichlet sample with exact zeros (B1, D3), so it often rules
out the true state, and the next observation is impossible under its belief. NaN beliefs
then flow into agent inputs and the advantages. (A second trigger, in the b̄_o update, came
from the old probability-matching policies; Boltzmann policies give every real action
positive probability, so it should be gone. Re-measure once D3 is settled.)

**D1. Reset observations.** The stacked env emits no reset observation; every belief level
starts at the prior. `belief_representations` argues this is a significant modelling error
and provides `initial_belief`, `initial_other_belief_estimate` and
`initial_second_order_ego_estimate`, which only the tests use. Either emit reset
observations or drop the functions.

**D5. Terminated games keep communicating.** A game in the guessing game's absorbing done
state keeps running comm rounds until the episode horizon. Rewards are 0 there, but those
stages still enter training as data.

**T5. Self-tests in `__main__`.** The stacked env ends in ~250 lines of `__main__` checks
plus `a_to_b_thrice_example`, overlapping `tests/`. Move what isn't covered into `tests/`
and delete the rest.

**T6. Dead state and arguments.** `agent_utterance_actions_rendered` is always `None`;
`global_rng_key` is written at reset and never read; `get_obs(key, ...)` never uses `key`.

---

## `envs`

**D4. The guessing game is a weak test of communication.** Role 1's action is inert, and
the presser can solve it alone by waiting (~3 waits on average; the Q-based presser waits on
the prior with P ≈ 0.89 at β = 10). Candidates:
- **Lewis signaling game:** the observer sees the state, the presser sees nothing.
  Communication is necessary; the cleanest "can this train at all" check.
- **Joint action:** both agents' actions matter, each holding half the information.
- The existing game with a wait cost high enough that guessing early dominates.

**T7. Three readers of the same tensors.** `FlexibleEnv._joint_*`, `ego_frame` in
`belief_representations`, and the stacked env's `_step_underlying_env` index
`transition` / `observation` / `reward` independently. `FlexibleEnv` is only used by
`tools/returns.py` and `guessing_game.__main__`, and its `get_obs` always queries with action
`(0, 0)`, which is wrong for action-dependent observations.

**T8. Brittle builders.** `build_reward_tensor` hardcodes four rewards, and
`guessing_game_spec` refuses any size other than 3 referents, since its Q functions are
derived for that. Stale `# [num_roles, S]` comment in `_pad_params`
([env_assembly.py](../src/envs/env_assembly.py)).

---

## `tools`

**T9. `belief_representations` leftovers.**
- `if ego_belief is None: raise` is near-dead: `ego_belief` is keyword-only with no default.
- `update_with_observation_only` takes `ego_action_tminus1` while the estimate updates take
  `ego_action` for the same quantity. Pick one.
- Still `distrax.Categorical` in and out; the planned move to raw probs arrays (so
  `ExactReturnEvaluator` can vmap over observation pairs) hasn't happened.
- `initial_belief`'s docstring refers to `FlexibleEnv.get_obs`, a runtime only `returns.py`
  uses (T7, T10).

**T10. `returns.py` is unused** outside its own `__main__`. Its evaluators give each agent
its partner's true belief as b̄_o (an oracle), and its Monte-Carlo estimator observes the
joint action, a different information structure from the stacked env. Rework it with the
planned Monte-Carlo Q estimator, move it to `analysis/`, or delete it (which would leave
`FlexibleEnv` without a real user).

**T11. `distributions.py` is unused.** Nothing imports `JointCategoricalPair`; only a
docstring reference in `flexible_env.py` remains.

**Minor.** `utterance_rendering` clips indices to `image_dim` rather than `image_dim - 1`
(harmless: out-of-bounds scatters are dropped), its brush is an asymmetric 2×2, and its
`__main__` writes `test_renders.png` to the CWD. `visualization.py` is used only by
`run_debug` (T1).
