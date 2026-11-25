# Codebase overview & open work

A bird's-eye review of `src/`, package by package: what's broken, what's missing, and
what's fat. Written 2026-11-07; last revised 2026-11-16 against `530f933` (clean tree).
That includes Claude's refactor of `tools/belief_representations.py` from the
`CategoricalBeliefState` class to pure functions, and the follow-up commits that removed
every fallback from it — see [Belief-state refactor](#belief-state-refactor-done) under
`tools`.

**Since 2026-11-07:** T3, T4 and T12 are done; the belief functions now NaN everywhere
(no fallbacks) and use `env_params` / `ego_role` naming; the test suite grew from 49 to
162 tests, almost all reference tests for the belief functions. The training, agents and
envs code is otherwise unchanged, so B1–B4, G1–G5 and D1–D5 all still stand. Removing the
fallbacks made one existing problem visible: rollouts now hit NaN beliefs within a few
acts (B5).

Each item has an ID (`B1`, `T3`, ...) so it can be referenced in discussion. Strike items
through or delete them as they're resolved.

- **B** — bugs (confirmed by running code unless noted)
- **G** — gaps: features that are missing or stubbed
- **D** — design questions that need a decision before code
- **T** — trimming: fat, dead code, duplication, hygiene

---

## Headline

- Tests pass (159 as of 2026-11-25: 119 belief-function, 7 stacked env, 30 Q policies,
  3 GAE).
- Training **runs end to end but does not train**:
  - the belief population's losses are NaN from the first iteration (B1);
  - env beliefs go NaN within a few acts at initialization, so from the second iteration
    both populations' advantage means are NaN, even under single-round `a_to_b` (B5);
  - multi-round schemes produce NaN advantages (B2);
  - the loss masks credit agents for actions the env discarded (B3).

  Nothing in the test suite covers the agents, the rollout or the PPO update, and the
  GAE tests use no NaN rewards. That is why none of this was caught.
- Policies are Boltzmann over hand-specified Q functions of two beliefs, Q(b_e, b̄_o, a)
  (see "Q-function policies" under `envs`). The env tracks three levels per agent, b_e,
  b̄_o and b̿_e (G1, G2), but communication moves none of the estimates yet (G3).
- No wandb logging (G5). The only environment is the guessing game, which is a weak test
  of communication (D4).

### Suggested order

1. Trimming (T-items): low risk, makes everything after it easier to read. T3, T4 and T12
   are done; T1, T5, T6 and T9–T11 remain.
2. Training bugs B1–B3 and B5, plus a smoke test of `make_train` that asserts finite
   metrics, so they stay fixed. B1 and B5 share a root cause in the belief action, so
   settle D3 (and the D2 question of overwriting) first.
3. wandb logging (G5).
4. A Lewis-style env where communication is *necessary* (D4), to prove learning works.
5. Belief-estimate tower (G1–G3, D1–D2), whose effect is only measurable once 4 works.

---

## `training`

**B1. Belief log-prob is `+inf` → belief PPO is NaN from iteration 0.** The guessing-game
prior has zero mass on the done state, so `BeliefActor`'s `log b = -inf` there, the
Dirichlet concentration is `1e-4`, sampled coordinates come out as exactly `0.0`, and
`log_prob` is `+inf` on every belief stage. Predicted by the NOTE in
[belief_agents.py](../src/agents/belief_agents.py) (`BeliefActor.__call__`); it is live in
the default config. Fix options are a design question (see D3). Re-confirmed 2026-11-16:
623–639 of 640 belief-stage log-probs are `+inf` in a 64-stage rollout. The same exact
zeros also feed B5.

**B2. Multi-round schemes have no learning signal.** In e.g. `a_to_b_thrice`, non-final
belief stages carry a NaN reward (the "fill in retroactively" placeholder that was never
filled). In `_masked_gae` ([advantages.py](../src/training/advantages.py)) the NaN enters
`new_gae`, which is selected into the carry on active steps, so it propagates backward to
almost every decision step of both populations. `_ppo_loss` then `nan_to_num`s advantages
to 0, so those steps silently contribute nothing. Single-round schemes (`a_to_b`, `b_to_a`,
`both_speak`) are unaffected.

**B3. Loss masks are per-step, not per-agent.** `stage_mask` has no agent axis
(`_update_all_agents` in [ppo_update.py](../src/training/ppo_update.py), `in_axes=None`).
In `a_to_b`, role-1 agents' utterances are discarded by the env but those agents are still
PPO-trained on them against the shared reward; likewise role-0 agents' belief actions.
`Transition` needs per-agent `speaks` / `listens` masks — and they vary per episode under
`simple_assignment_fn`, so they must be recorded per step, not derived from the scheme.

**B4. Truncation treated as termination.** Bootstrap values default to 0 at the end of each
rollout window, but the env continues across windows. Documented in `compute_advantages`;
fix is to run each critic on the final carried obs.

**B6 (unconfirmed). Utterance PPO ratios drift fast.** In a 3-iteration smoke run
(`a_to_b`, `rollout_length=32`, default `PPOConfig`), `utterance_clip_frac` is ~0.68 from
the first iteration and `utterance_approx_kl` reaches ~4 by the second. The utterance
action is a high-dimensional unbounded Gaussian (entropy ~13), so a single minibatch step
can move log-probs a lot, which would explain it. But a mismatch between the log-prob
stored in the rollout and the one recomputed in `_ppo_loss` would look the same. Check
before trusting any utterance-side learning curve: on the first minibatch of the first
epoch, the ratio should be exactly 1.

**G4. ToM executors are placeholders.** `execute_tom_*` in
[agent_execution.py](../src/training/agent_execution.py) behave first-order. They also vmap
the partner population index-aligned (`partner_params[i]` for agent `i`) rather than
gathering the real partner via `game_roles.partner_agent`.

**G5. No wandb logging.** [wandb_log.py](../src/training/wandb_log.py) is an empty
skeleton, and `launch.py` discards what `train()` returns. Metrics are already produced per
iteration inside the scan, so the simplest live option is a `jax.debug.callback` in
`_update_step` that calls `wandb.log`.

**T1. `run_debug` inside `train`.** Pushes random inputs through every agent, prints, and
`savefig`s two PNGs into the CWD on every `train` call (this is where the committed
`src/belief_debug.png` / `src/utterance_debug.png` came from). It is also why `train` can't
be `jax.jit`-ed. Delete it, or move it to a standalone script.

**T2. `train` is not jitted.** Follows from T1.

~~**T3. Duplicated setup.**~~ Done 2026-11-07: [ippo.py](../src/training/ippo.py) now has
plain functions `build_env(config)`, `initialize_agents(config, env, rng)` (both
populations from one key; shapes read from `env` and `config`), and
`make_render_utterance_fn(image_dim)`. `make_train` and `inspect_rollout.run_rollout` both
use them, with the same `(agent, env, loop/rollout)` key split, so a given `jax_seed`
inspects exactly the agents and reset that training starts from. (`ippo.py` is kept to
functions only — no runtime container class.)

~~**T4. Time vocabulary.**~~ "Epoch" and "step" are both overloaded: `num_epochs` = training
iterations, `PPOConfig.update_epochs` = passes over a rollout, `num_steps_per_epoch` =
`step_env` calls per rollout (stages, not env steps), and
`cumulative_communication_round_iterator` counts stages, not rounds.

Decided 2026-11-07:
- **Env time, smallest to largest:** **stage** (one `step_env` call) → **round** (utterance
  stage + belief stage) → **block** (the rounds between two acts) → **act** (one
  underlying DecPOMDP step) → **episode**. "Stage" gets added to the Terminology comment in
  the stacked env.
- **Training time:** `num_epochs` → `num_iterations`; `num_steps_per_epoch` →
  `rollout_length`, **counted in stages** (fixed-length scan); `update_epochs` and
  `num_minibatches` keep their standard PPO names, so "epoch" means only a PPO pass.
- `cumulative_communication_round_iterator` → `cumulative_stage_count` (overlaps T6).

~~Renames~~ applied 2026-11-07 in `training/config.py`, `ippo.py`, `ppo_update.py`, the
stacked env (field, debug prints, and the Terminology comment, which now lists STAGE and ACT)
and `tests/stacked_harness.py`. CLI flags are now `--num-iterations` / `--rollout-length`.
Leftovers: two comments in `_update_step` ([ippo.py](../src/training/ippo.py)) still say
"scan env_step num_steps times" / "`[num_steps]` axis". `inspect_rollout`'s `--num-steps`
flag also counts stages; it could become `--rollout-length` to match.
(`num_steps` as a local name for the leading stage axis in `ppo_update.py` is fine.)

---

## `agents`

**D3. What is a belief action?** `BeliefActor` uses `softplus(log b + log L)` as a Dirichlet
concentration. Besides B1, this is not the Bayesian update the docstring claims: it is not
invariant to adding a constant to `log L` (softmax would be), and the Dirichlet's mean is
`∝ softplus(log posterior)`, not the posterior. Options:
- a Dirichlet restricted to the support of `b` (zero-support coords fixed at 0 and excluded
  from `log_prob`);
- make the *action* the log-likelihood vector (e.g. Gaussian in logit space) and compute the
  belief deterministically as `normalize(b · L)` — a "likelihood action" rather than a
  "belief action", which also makes the support guarantee exact.

**Perf.** The conv trunk has no pooling: 32×32×64 → Dense(128) is ~8.4M params per head, so
~17M per belief agent (actor and critic have separate trunks) before Adam state, times
`num_agents`.

**Minor.** `UtteranceCritic.belief_dim` is unused. Utterance samples are unbounded
Gaussians around a sigmoid mean; the renderer clips them to [0, 1].

---

## `communication`

[stacked_signification_decpomdp.py](../src/communication/stacked_signification_decpomdp.py)
is the core of the project. `communication_scheme.py` and `game_role_assignment.py` are
small and clean.

**`step_env` is split by stage (2026-11-25).** `step_env` dispatches with a `lax.cond` on
the (traced) stage to `_step_env_utterance_stage` (deliver utterances, store the speaker's
estimate of its listener) or `_step_env_belief_stage` (listeners adopt their beliefs and
store their estimates of their speakers and their new b̿_e, then the act and the episode
boundary on a block's last round), and then advances the schedule and builds the
observation for both. The rollout still makes one `step_env` call per stage with all five
action inputs, and both stages return the same structure, so the no-act
placeholders (NaN rewards, -1 actions and observations, NaN debug beliefs) remain.

**B5. Env beliefs go NaN within a few acts at initialization.** Found 2026-11-16 in
256-stage rollouts with untrained agents (5 seeds). Under `a_to_b`, 4 of 5 seeds go NaN,
the earliest at stage 3, i.e. the first act after one communication round. Under
`a_to_b_thrice`, all 5 do, the earliest at stage 5. This is the NaN tripwire working as
intended: it is reporting two real problems, not a belief-function bug.
- **True beliefs (old problem, newly visible).** The listener adopts its `belief_action`
  verbatim as its true belief (D2), and the Dirichlet samples contain exact zeros (B1). An
  untrained listener therefore routinely adopts a belief that rules out the true state.
  The next real observation is then impossible under that belief, and the update NaNs.
  Traced example (seed 2, game 4, true state 0): the observer adopted
  `[0.007, 0.993, 0, 0]`, then saw an observation only state 0 could emit. These true-belief
  updates have NaN'd since 2026-11-07, so this predates the latest commits.
- **Estimates (new since `f05287b`).** `update_other_belief_estimate_with_observation_only`
  used to fall back to the unchanged b̄ and now NaNs. It fires whenever π*(b̄) gives zero
  probability to every partner action consistent with `o_ego`. That is a wrong estimate,
  not an env bug — exactly the caveat the deleted NOTE recorded. With probability-matching
  π* on a sharp, wrong b̄, this is easy to hit. (2026-11-25: policies are now Boltzmann,
  which gives every real action positive probability, so this particular trigger is gone.
  The B5 measurements above predate that and have not been re-run.)
- **Consequence.** NaN beliefs flow into agent inputs: in a 64-stage seed-2 rollout, 8 of
  320 agent-stage rows already have NaN inputs, log-probs and values. In training, both
  populations' `advantage_mean` is NaN from the second iteration on, under `a_to_b`.

Fixing it is a decision, not a patch:
- For the true belief, D3's likelihood action (belief = `normalize(b · L)`, so it can never
  drop support the agent hasn't ruled out) or a support-restricted Dirichlet removes the
  cause.
- For the estimate, decide whether "my model of you was wrong" should be a NaN (tripwire)
  or a modelled event. If the latter, it calls for a model change rather than a fallback,
  e.g. mixing π*(b̄) with ε-uniform as a noise model of the partner.

**G1. ~~No second-order estimate in the env~~** — done 2026-11-25. State now also holds
`second_order_agent_belief_estimates` (row i: agent i's estimate of its partner's estimate
of agent i, held by and about agent i), seeded from the prior, updated at every act, and
returned by `get_obs` as the third of `(beliefs, estimated_beliefs, second_order_estimates,
utterances)`. It is never NaN'd by stage; the rollout unpacks it but no agent consumes it
yet.

**G3. Communication never moves the estimates.**
- The speaker's post-utterance estimate of the listener is a passthrough in the rollout
  (`belief_estimate_post_utterance = env_state.estimated_agent_belief_states` in
  [rollout.py](../src/training/rollout.py)), so it never changes during communication.
- ~~The listener never updates its estimate of the *speaker* after hearing an utterance~~ —
  the env side is done (2026-11-25): the belief stage takes `other_belief_estimate_actions`
  (subject-indexed; the speakers' rows are written). The rollout still passes the existing
  estimate through unchanged, as it does for `belief_estimate_post_utterance`; no model
  produces either yet.
- b̿_e: the listener's side is wired (2026-11-25). The belief stage takes
  `second_order_ego_estimate_actions`, the listener's new estimate of what its speaker
  believes about it, and writes the listeners' rows (b̿_e is stored by holder). The rollout
  passes the existing b̿_e through unchanged. The speaker's side is still a TODO in
  `_step_env_belief_stage` section 2: after uttering, the speaker should revise what it
  thinks the listener thinks of it. A stale b̿_e biases the partner prediction
  π_o(b̄_o, b̿_e) at the act.

**D1. Reset observations.** The stacked env emits no reset observation; beliefs and
estimates start at the prior. `belief_representations` argues this is a significant
modelling error and provides `initial_belief` / `initial_other_belief_estimate`, which
nothing in the env uses (only tests). Either emit reset observations or drop the argument.

**D2. Who owns "what an agent sees"?** `get_obs` NaNs own beliefs on the belief stage, but
the rollout reads own beliefs straight from `env_state` anyway, so the masking is half-used.
(Second-order estimates are deliberately never NaN'd: the listener needs them on the belief
stage.)
Relatedly, the listener adopts `belief_action` *verbatim* as its new true belief, so it can
overwrite private information it had; whether that's intended (the network must learn to
preserve it) or should be structural (D3's likelihood action) is a decision. Now concrete:
this overwrite is the true-belief half of B5.

**D5. Terminated games keep communicating.** A game that reaches the guessing game's
absorbing done state keeps running comm rounds until the episode horizon. Rewards are 0
there, but those stages still enter training as data.

**T5. ~310 lines of self-tests in `__main__`** plus `a_to_b_thrice_example`, which
duplicates part of `__main__`. Overlaps `tests/stacked_harness.py`; move what isn't already
covered into `tests/` and delete the rest.

**T6. Dead / redundant state and arguments.**
- `agent_utterance_actions_rendered` is always `None`.
- `global_rng_key` is written at reset and never read.
- `get_obs(key, ...)` never uses `key`.
- ~~One flag, two names: `skip_first_communication_step` vs
  `act_on_reset_before_communicating`~~ — the act-before-communicating option was removed
  2026-11-25; every episode starts with communication at env timestep 0.
- ~~`cumulative_communication_round_iterator` counts `step_env` calls, not rounds~~ —
  renamed `cumulative_stage_count` (T4).

~~`_belief_factories` + `lax.switch` over game types~~ — done by the belief-state refactor
(params are gathered with a traced `game_type`).

---

## `envs`

**D4. The guessing game is a weak test of communication.** Role 1's action is inert, and
the presser can solve it alone by waiting (~3 waits on average). (The old
probability-matching policies never waited; the Q-based presser now waits on the prior,
P(wait) ≈ 0.89 at β = 10.) Candidate envs:
- **Lewis signaling game:** the observer sees the state, the presser sees nothing.
  Communication is necessary; the cleanest "can this train at all" check.
- **Joint action:** both agents' actions matter (e.g. both must press the same button, each
  holding half the information).
- The existing game with a wait cost high enough that guessing early dominates.

**T7. Three readers of the same tensors.** `FlexibleEnv._joint_*`, the gathers in
`belief_representations` (`ego_frame`), and the stacked env's `_step_underlying_env` all
index `transition` / `observation` / `reward` independently. `FlexibleEnv` claims to be
"the single reader" but is only used by `tools/returns.py` and `guessing_game.__main__`. Its
`get_obs` also always queries with action `(0, 0)`, so it would be wrong for any env with
action-dependent observations (the stacked env does pass the real action).

**T8. Brittle builders.** `build_reward_tensor` hardcodes four rewards (now from the
`CORRECT_PRESS_REWARD` / `WRONG_PRESS_REWARD` / `WAIT_REWARD` constants the presser Q also
reads), so it only works for `num_actions == 4`; `guessing_game_spec` now refuses any other
size, since the Q functions are derived for 3 referents. Stale `# [num_roles, S]` comment in `_pad_params`
([env_assembly.py](../src/envs/env_assembly.py)).

**Q-function policies (2026-11-25).** Policies used to be `Categorical -> Categorical`
functions of the actor's own belief. They are now Boltzmann over Q functions of two beliefs:

- `envs.flexible_env`: `QFunction = (b_e, b̄_o) -> [A]` values (-inf = unavailable),
  `boltzmann_policy(q, β)` (padded -inf actions get probability 0 at every β), and
  `GameSpec(params, q_functions, inverse_temperature)`, which every EnvSpec now returns.
- `envs.env_assembly`: `BoltzmannPolicyBundle`, called as
  `policy_bundle(game_type, role, b_e, b̄_o) -> Categorical`, with `.q_values(...)` for the raw
  values. It pads each game's Q to the common action count with -inf. β is per game type.
- `envs.guessing_game`: `presser_q_function(γ)` is the exact infinite-horizon optimal Q
  (closed form, derived in its docstring, checked against a brute-force Bellman lookahead
  over the real tensors in `tests/test_q_policies.py`). It ignores b̄_o, because the
  observer is inert in T, O and R. `observer_q_function` is constant, so the observer acts
  uniformly. Defaults: γ = 0.9, β = 10. Both Q functions return NaN for a NaN belief.
- Stacked env: an agent acts on π(b_e, b̄_o) and predicts its partner's unseen action as
  π_o(b̄_o, b̿_e). Two predictions need levels above b̿_e: `ego_action_as_they_predict` = π_e(b̿_e, b⃛_o)
  and `other_action_as_they_think_we_predict` = π_o(b⃛_o, b⁗_e). Both are evaluated with the Boltzmann
  policies, with a stand-in belief U in every untracked slot: π_e(b̿_e, U) and π_o(U, U). U is the
  game's initial state distribution on an episode's first act (even after that act's
  communication), and a uniform belief over all real states, terminal ones included, on
  later acts. A lower level is never substituted for an untracked one (a `stationary` option
  that did so was removed the same day). In the guessing game U changes nothing: the presser's Q
  ignores b̄_o, the observer's is constant, and U is symmetric over the referents, so a
  symmetric P(press) cancels out of the observer's b̿_e update. A test injects an asymmetric
  prediction to check that the first-act flag reaches the updates.

---

## `tools`

### Belief-state refactor (done)

Claude refactored [belief_representations.py](../src/tools/belief_representations.py)
(committed in `58ffda1`, 2026-11-07), with follow-ups in `f05287b` (2026-11-12) and
`530f933` (2026-11-16). What changed:

- **Class → pure functions.** `CategoricalBeliefState` is gone. Every update is a function
  taking a single game's `FlexibleEnvParams` (`env_params`) first:
  `update_with_observation_and_joint_action`, `update_with_observation_only`,
  `initial_belief`, `initial_other_belief_estimate`,
  `update_other_belief_estimate_with_observation_only`. The two observation-only updates
  take keyword-only arguments (`ego_belief=`, `other_belief_estimate=`, `ego_observation=`,
  `ego_role=`, plus `ego_action_tminus1=` / `ego_action=` — see T9). Since 2026-11-25 all
  three observation-only updates take `other_action_as_we_predict=`, a Categorical over the
  other agent's action; none of them calls a policy (the ego update used to take
  `other_policy=`).
- **Traceable params.** The functions read cardinalities from tensor shapes, never
  `int(env_params.num_states)`, so `env_params` can be traced. The stacked env gathers one
  game type's params with a traced `game_type` index; the per-type `_belief_factories` tuple
  and its `lax.switch` are gone.
- **`ego_frame(env_params, ego_role)`** swaps the agent axes of `T` / `O` once when the
  ego is agent 1, so every update is written from the ego's point of view. This replaced
  the five repeated `lax.cond` marginal selections and `joint_action_constructor`.
- **Vectorized math.** The nested `vmap`s over states/actions became `einsum`s.
- **No fallbacks anywhere (since `f05287b`).** `_normalize` is gone; every function ends in
  a plain `unnormalized / unnormalized.sum()`, so all five return all-NaN when the evidence
  is impossible under the model. The module docstring's "Impossible observations" section
  states the policy. The estimate functions mix per-hypothesis branches. Their only
  guard zeroes branches with zero OUR-side weight, so 0 · NaN can't poison a healthy
  estimate; it is commented as "not a fallback". Since 2026-11-16 the guard is on weight,
  not on the partner-side posterior mass. A branch we find possible but they find
  impossible means the estimates contradict each other, and it now yields NaN
  (`test_contradictory_estimates_produce_nan`). See B5 for what the stricter policy
  surfaced in the env.
- **Renames (`530f933`).** `params` → `env_params`, `agent_id` → `agent_role` (since renamed `ego_role`, 2026-11-16) (all call
  sites updated: the stacked env, `returns.py`, `guessing_game.__main__`).
  `update_with_observation_and_joint_action` now carries a NOTE that the model never uses
  it (agents don't see the joint action); its only caller is `returns.py` (T10).
- **No more `JointCategoricalPair`** in the belief code (see T11).
- **Call sites updated:** `stacked_signification_decpomdp._agent_belief_updates`,
  `tools/returns.py` (`ExactReturnEvaluator` takes `params` directly and gathers
  transition/observation/reward itself), and `guessing_game.__main__`.
- **Tests:** [test_belief_representations.py](../tests/test_belief_representations.py)
  has 119 tests: worked examples, `test_impossible_observations_produce_nan`, and a
  `*_matches_reference` test per function against brute-force references. These are
  parametrized over seed, `ego_role` and traced vs concrete inputs, and the estimate
  update also with a random `ego_action_as_they_predict`.

What's left over from it is in T9.

**G2. Second-order estimate: functions done, wired in 2026-11-25** (see G1 and "Q-function
policies"). (2026-11-16) The memo-based
`nested_belief.py` was deleted (memo is not to be used) and replaced by hand-written JAX in
`belief_representations`: `initial_second_order_ego_estimate` and
`update_second_order_ego_estimate_with_observation_only`, which updates b̿_e (our estimate of
their estimate of our belief). It reuses the first-order branch weights
(`_branch_weights`) and, in each branch, runs the first-order estimate update from the
other agent's seat. With two-belief policies, `ego_action_as_they_predict` is π_e(b̿_e, b⃛_o) and
`other_action_as_they_think_we_predict` is π_o(b⃛_o, b⁗_e); both reach above the tracked levels. The env
evaluates them as π_e(b̿_e, U) and π_o(U, U), where U is the initial state distribution on the first act and uniform over all real states afterwards. The reset-observation initializers
(`initial_second_order_ego_estimate` etc.) are still unused by the env (D1).

**T9. `belief_representations` leftovers** (after the refactor above).
- `if ego_belief is None: raise` is near-dead now that `ego_belief` is keyword-only with no
  default.
- ~~**Stale docstring:** `update_with_observation_only`'s Args documented `ego_action`~~ —
  fixed 2026-11-25.
- **Naming asymmetry:** `update_with_observation_only` takes `ego_action_tminus1`, while
  `update_other_belief_estimate_with_observation_only` takes `ego_action` for the same
  quantity (the ego's previous action). `_agent_belief_updates` passes the same `action` to
  both under different names. Pick one.
- Still `distrax.Categorical` in and out; the planned move to raw probs arrays (so
  `ExactReturnEvaluator` can vmap over observation pairs instead of its Python double loop)
  hasn't happened.
- Docstring references to `FlexibleEnv.get_obs` (in `initial_belief`) tie these functions to
  a runtime that only `returns.py` uses (T7, T10).
- ~~Stray imports, repeated `lax.cond` marginal selection, per-type `lax.switch`~~ — done.
- ~~NaN-vs-fallback in the estimate functions~~ — decided 2026-11-12: NaN everywhere (but
  see B5).

**T10. `returns.py` is unused.** Only its own `__main__` calls it. Since 2026-11-25 its
policies take `(own belief, estimate of partner)`, and both evaluators feed each agent its
partner's TRUE belief as that estimate (an oracle b̄_o; NOTE in the module docstring);
`policies_from_spec` builds Boltzmann policies from a `GameSpec`. To be reworked with the
Monte-Carlo Q estimator. Its Monte-Carlo estimator
also uses `update_with_observation_and_joint_action` (joint action observed) and ignores the
reset observation — a different information structure from the stacked env. Move to an
`analysis/` folder or delete. If deleted, `FlexibleEnv` (T7) loses its last real user.

**T11. `distributions.py` is unused.** Nothing imports `JointCategoricalPair` any more (the
belief-state refactor dropped it; its last other user was a deleted visualization). Only a docstring
reference in `flexible_env.py` remains.

**Minor.** `utterance_rendering` clips indices to `image_dim` rather than `image_dim - 1`
(out-of-bounds scatters are dropped, so harmless) and its brush is an asymmetric 2×2; its
`__main__` writes `test_renders.png` to the CWD. `visualization.py` is used only by
`run_debug` (T1).

---

## Repo hygiene

**T12.**
- ~~Committed junk: `src/.DS_Store`, `src/belief_debug.png`, `src/utterance_debug.png`~~ —
  removed 2026-11-07. The PNGs come back whenever `train` runs from `src/` until `run_debug`
  goes (T1).
- ~~Dependencies~~ — done 2026-11-07: removed `jaxmarl` (never imported; also dropped
  `jaxopt` and a pile of transitive packages from the lock), `dash` and `plotly`; declared
  `matplotlib`. The core JAX stack versions in `uv.lock` are unchanged. `chex` is still
  imported directly but undeclared; it is a hard dependency of distrax/flax/optax, so it
  cannot go missing.
- README is half personal notes, half install/run instructions (keeping as is for now).
- ~~`useful_visualizations/`~~ — deleted 2026-11-07 (stale; to be rebuilt from scratch when
  needed).
