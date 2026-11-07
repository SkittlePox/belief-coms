# Codebase overview & open work

A bird's-eye review of `src/`, package by package: what's broken, what's missing, and
what's fat. Written 2026-11-07 against the working tree, which includes Claude's
(uncommitted) refactor of `tools/belief_representations.py` from the
`CategoricalBeliefState` class to pure functions — see
[Belief-state refactor](#belief-state-refactor-done) under `tools`.

Each item has an ID (`B1`, `T3`, ...) so it can be referenced in discussion. Strike items
through or delete them as they're resolved.

- **B** — bugs (confirmed by running code unless noted)
- **G** — gaps: features that are missing or stubbed
- **D** — design questions that need a decision before code
- **T** — trimming: fat, dead code, duplication, hygiene

---

## Headline

- Tests pass (49 as of this writing).
- Training **runs end to end but does not train**: the belief population's losses are NaN
  from the first iteration (B1), multi-round schemes produce NaN advantages (B2), and the
  loss masks credit agents for actions the env discarded (B3). Nothing in the test suite
  covers agents or PPO, which is why none of this was caught.
- The belief machinery has one estimate slot per agent; there is no second-order estimate
  in the env, and the tool that would provide one (`nested_belief`) can't run inside it
  (G1, G2).
- No wandb logging (G5). The only environment is the guessing game, which is a weak test
  of communication (D4).

### Suggested order

1. Trimming (T-items): low risk, makes everything after it easier to read.
2. Training bugs B1–B3, plus a smoke test of `make_train` so they stay fixed.
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
the default config. Fix options are a design question (see D3).

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

**T4. Time vocabulary.** "Epoch" and "step" are both overloaded: `num_epochs` = training
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

Renames not yet applied (touch `training/config.py`, `ippo.py`, `ppo_update.py`, the stacked
env, and `tests/stacked_harness.py`). Note the `--num-epochs` / `--num-steps-per-epoch` CLI
flags change with them.

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

**G1. No second-order estimate in the env.** State holds `true_agent_belief_states` and one
`estimated_agent_belief_states` per agent. There is no slot for "what my partner thinks I
believe" (or deeper). See D1 for what it would feed and G2 for how to compute it.

**G3. Communication never moves the estimates.**
- The speaker's post-utterance estimate of the listener is a passthrough in the rollout
  (`belief_estimate_post_utterance = env_state.estimated_agent_belief_states` in
  [rollout.py](../src/training/rollout.py)), so it never changes during communication.
- The listener never updates its estimate of the *speaker* after hearing an utterance, even
  though the utterance is evidence about the speaker's belief.

**D1. Reset observations.** The stacked env emits no reset observation; beliefs and
estimates start at the prior. `belief_representations` argues this is a significant
modelling error and provides `initial_belief` / `initial_other_belief_estimate`, which
nothing in the env uses (only tests). Either emit reset observations or drop the argument.

**D2. Who owns "what an agent sees"?** `get_obs` NaNs own beliefs on the belief stage, but
the rollout reads own beliefs straight from `env_state` anyway, so the masking is half-used.
Relatedly, the listener adopts `belief_action` *verbatim* as its new true belief, so it can
overwrite private information it had; whether that's intended (the network must learn to
preserve it) or should be structural (D3's likelihood action) is a decision.

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
- One flag, two names: `skip_first_communication_step` (constructor / config) vs
  `act_on_reset_before_communicating` (attribute).
- `cumulative_communication_round_iterator` counts `step_env` calls, not rounds.

~~`_belief_factories` + `lax.switch` over game types~~ — done by the belief-state refactor
(params are gathered with a traced `game_type`).

---

## `envs`

**D4. The guessing game is a weak test of communication.** Role 1's action is inert; the
presser can solve it alone by waiting (~3 waits on average); and the "optimal" policies are
probability matching, so the presser never waits (`P(wait) = b(done) = 0`) — the off-policy
problem `nested_belief`'s docstring warns about. Candidate envs:
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

**T8. Brittle builders.** `build_reward_tensor` hardcodes `[-1, -1, -1, -0.1]`, so it only
works for `num_actions == 4`. Stale `# [num_roles, S]` comment in `_pad_params`
([env_assembly.py](../src/envs/env_assembly.py)).

---

## `tools`

### Belief-state refactor (done)

Claude refactored [belief_representations.py](../src/tools/belief_representations.py)
(uncommitted as of 2026-11-07). What changed:

- **Class → pure functions.** `CategoricalBeliefState` is gone. Every update is a function
  taking a single game's `FlexibleEnvParams` first: `update_with_observation_and_joint_action`,
  `update_with_observation_only`, `initial_belief`, `initial_other_belief_estimate`,
  `update_other_belief_estimate_with_observation_only`. The two observation-only updates
  take keyword-only arguments (`ego_belief=`, `other_belief_estimate=`, `ego_observation=`,
  `ego_action=`, `other_policy=`, `agent_id=`).
- **Traceable params.** The functions read cardinalities from tensor shapes, never
  `int(params.num_states)`, so `params` can be traced. The stacked env now gathers one game
  type's params with a traced `game_type` index; the per-type `_belief_factories` tuple and
  its `lax.switch` are gone.
- **`ego_frame(params, agent_id)`** swaps the agent axes of `T` / `O` once when the ego is
  agent 1, so every update is written from the ego's point of view. This replaced the five
  repeated `lax.cond` marginal selections and `joint_action_constructor`.
- **Vectorized math.** The nested `vmap`s over states/actions became `einsum`s.
- **Explicit NaN policy (`_normalize`).** The true-belief updates (`update_with_*`,
  `initial_belief`) deliberately return all-NaN when the observation is impossible under the
  model, so upstream bugs surface. The estimate updates fall back instead
  (`initial_other_belief_estimate` → prior, `update_other_belief_estimate_*` → the unchanged
  estimate). An in-code NOTE flags the `initial_other_belief_estimate` fallback as a
  candidate for switching to NaN too.
- **No more `JointCategoricalPair`** in the belief code (see T11).
- **Call sites updated:** `stacked_signification_decpomdp._agent_belief_updates`,
  `tools/returns.py` (`ExactReturnEvaluator` now takes `params` directly and gathers
  transition/observation/reward itself), and `guessing_game.__main__`.
- **Tests:** [test_belief_representations.py](../tests/test_belief_representations.py)
  gained `test_impossible_observations_produce_nan` and
  `test_matches_brute_force_reference` (parametrized over seed, `agent_id`, and traced vs
  concrete inputs).

What's left over from it is in T9.

**G2. The second-order tool can't run in the env.**
[nested_belief.py](../src/tools/nested_belief.py) carries the full tower (`bel[0..K]`) and
is the principled answer to `ego_action_prior`. But it generates memo source into a temp
file at runtime and its `NestedBeliefStep.__call__` uses `int()` and `np.asarray`, so it is
host-side only — not jittable or vmappable inside `step_env`. Options:
- a hand-written jittable tower step: the first-order estimate update applied recursively,
  with level `k+1` supplying level `k`'s `ego_action_prior`, stored as `[num_agents, K+1, S]`;
  `nested_belief` then serves as its reference in tests;
- check whether memo's generated `module.step` itself vmaps once the wrapper's `int()` /
  `np.asarray` are removed.

**T9. `belief_representations` leftovers** (after the refactor above).
- `mode="soft_evidence"` is kept only "for comparison".
- `if ego_belief is None: raise` is near-dead now that `ego_belief` is keyword-only with no
  default.
- Decide the NaN-vs-fallback question the NOTE in `initial_other_belief_estimate` raises.
- Still `distrax.Categorical` in and out; the planned move to raw probs arrays (so
  `ExactReturnEvaluator` can vmap over observation pairs instead of its Python double loop)
  hasn't happened.
- Docstring references to `FlexibleEnv.get_obs` (in `initial_belief`) tie these functions to
  a runtime that only `returns.py` uses (T7, T10).
- ~~Stray imports, repeated `lax.cond` marginal selection, per-type `lax.switch`~~ — done.

**T10. `returns.py` is unused.** Only its own `__main__` calls it. Its Monte-Carlo estimator
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
