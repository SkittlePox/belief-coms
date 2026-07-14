# useful_visualizations

Small, interactive Plotly views for double-checking what the belief-coms environment
is doing under the hood. Each `render()` writes a **self-contained** interactive HTML
file into `renders/` (open it in any browser — no server, works offline).

Everything shares one visual vocabulary defined in `_figures.py` (one template, one
probability colorscale, one diverging reward scale, one temporal-slider driver), so
the three views read as one system.

## Run

From the repo root (this package is dev-only — it is not part of the installed
library, so it is imported from the working directory rather than the environment):

```bash
# all three
uv run python -m useful_visualizations

# or one at a time
uv run python -m useful_visualizations.viz_distributions
uv run python -m useful_visualizations.viz_guessing_game
uv run python -m useful_visualizations.viz_stacked
```

## The views

| File | Renders | What to look for |
| --- | --- | --- |
| `viz_distributions.py` | `JointCategoricalPair` (`tools/distributions.py`) | **Interactive.** Edit the joint `P(var1, var2)` grid and everything recomputes live in-browser: the flat vector, its `(v1, v2)` reshape, both marginals, a conditional (pick which `var1`), and an empirical sample grid. `reset example` / `uniform` / `clear` presets. |
| `viz_guessing_game.py` | The guessing game's `transition` / `observation` / `reward` tensors + a `FlexibleEnv` rollout | The three dense tensors from `guessing_game_spec`, then a **slider** over the "learning by waiting" rollout — watch the presser's belief collapse to the truth as each observed symbol rules out a state. |
| `viz_stacked.py` | `StackedSignificationDecPOMDP` state + `step_env` trace | The star. A per-step dashboard (routing, world states, scheduler counters, true vs estimated beliefs, **actions taken**, rewards) with a **slider** over many `step_env` calls. The actions panel (`[agent × action]`, read from `state.last_agent_actions`) lights up only on **ACT** steps — showing which button each agent pressed / whether it waited. The title reads out stage / round / episode and flags **ACT** and **EPISODE BOUNDARY** steps. |

## Notes

- `renders/` is git-ignored; regenerate any time by re-running.
- The stacked view drives `step_env` with scripted stand-in inputs (agents aren't
  learned — this is inspection), mirroring the harness in
  `stacked_signification_decpomdp.py`'s `__main__`.
- Tune the stacked trace via the constants at the top of `viz_stacked.py`
  (`NUM_AGENTS`, `NUM_STEPS`) or by swapping the routing / communication-scheme fns.
