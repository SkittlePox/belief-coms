# Agent notes (revised 2026-11-25)

Where the agents stand and what is still open.

## Current agents

| Agent | File | Inputs | Outputs |
|---|---|---|---|
| e2e utterance (speaker) | `e2e_utterance_agents.py` | b_e, b̄_o, b̿_e | MultivariateNormalDiag over utterances, value |
| e2e belief (listener) | `e2e_belief_agents.py` | b_e, b̄_o, b̿_e, rendered utterance | Dirichlet over the new b_e, value |

The belief agent's Dirichlet concentration is `softplus(logits) + 1e-4`. With
`multiply_by_ego_belief` on (the default), the logits are `log b_e + log L`, which never
gains support b_e didn't have but gives `+inf` log-probs where b_e is zero (B1 in
`docs/codebase_overview.md`). With it off, the logits are `log L` alone.

`--inference.mode` (`training/agent_execution.py`) picks how the agents run:
- `e2e`: each agent runs its own network.
- `tom`: the executors in `inference_agents.py` reason through the partner population's
  network. These are skeletons that currently behave like `e2e`.

The e2e networks are the basis of inferential communication: the ToM executors use them as
each agent's own policy and as its model of its partner. Names may change as the
inferential agents take shape.

## What `step_env` asks of the agents

| Stage | Who | Argument | Meaning | Rows written | Produced by |
|---|---|---|---|---|---|
| utterance | speaker | `utterance_actions` | the utterance | speakers' | utterance agent |
| utterance | speaker | `speaker_other_belief_estimate_actions` | b̄_o: what it thinks the listener now believes | listeners' (subject-indexed) | passed through |
| utterance | speaker | `speaker_second_order_ego_estimate_actions` | b̿_e: what it thinks the listener now believes about it | speakers' (by holder) | passed through |
| belief | listener | `listener_ego_belief_actions` | b_e: its new belief | listeners' | belief agent |
| belief | listener | `listener_other_belief_estimate_actions` | b̄_o: what it thinks the speaker believes | speakers' (subject-indexed) | passed through |
| belief | listener | `listener_second_order_ego_estimate_actions` | b̿_e: what it thinks the speaker now believes about it | listeners' (by holder) | passed through |

The speaker's own b_e does not change from uttering. "Passed through" means
`training/rollout.py` hands back the env's current values (a marked SKELETON block), so
communication never moves b̄_o or b̿_e. Observations at each act still update them.

## Open: what produces the new b̄_o and b̿_e?

1. **Sampled PPO actions.** Extra Dirichlet heads, with the action's log-prob summed over
   heads. Tried on 2026-11-25 and backed out in favour of single-output agents.
2. **Deterministic heads.** Not sampled and not in the PPO ratio; trained by a separate
   loss, or untrained at first.
3. **Inference through the partner's network.** E.g. the speaker runs the listener's belief
   network on its own rendered utterance to get the listener's b̄_o. The ToM skeletons in
   `inference_agents.py` are where this would go.

## Code that changes with the decision

- `training/rollout.py`: the SKELETON block that passes the four estimates through, and
  `Transition` (one action array per population).
- `training/ppo_update.py`: one log-prob per action (only if the estimates become sampled
  actions).
- `e2e_*_agents.py` and/or `inference_agents.py`: whatever produces the estimates.
