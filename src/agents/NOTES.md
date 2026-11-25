# Agent redesign notes (tabled 2026-11-25)

Open questions about the belief and utterance agents, written down when `step_env`'s
arguments changed. Nothing here is decided.

## What `step_env` now asks of the agents

Each stage takes the communicating agent's updated beliefs:

| Stage | Who | Argument | Meaning | Rows written |
|---|---|---|---|---|
| utterance | speaker | `utterance_actions` | the utterance | speakers' |
| utterance | speaker | `speaker_other_belief_estimate_actions` | b̄_o: what it thinks the listener now believes | listeners' (subject-indexed) |
| utterance | speaker | `speaker_second_order_ego_estimate_actions` | b̿_e: what it thinks the listener now believes about it | speakers' (by holder) |
| belief | listener | `listener_ego_belief_actions` | b_e: its new belief | listeners' |
| belief | listener | `listener_other_belief_estimate_actions` | b̄_o: what it thinks the speaker believes | speakers' (subject-indexed) |
| belief | listener | `listener_second_order_ego_estimate_actions` | b̿_e: what it thinks the speaker now believes about it | listeners' (by holder) |

The speaker's own b_e does not change from uttering.

Today the agents produce only `utterance_actions` and `listener_ego_belief_actions`.
`training/rollout.py` passes the four estimate arguments through unchanged.

## What each agent has vs. needs

- **Utterance agent** (`utterance_agents.py`): input `(own_belief, estimate_of_receiver_belief)`,
  output a Gaussian over utterances. Needs to also output b̄_o of the listener and b̿_e,
  and probably to take b̿_e as an input (`get_obs` already returns it; nothing reads it).
- **Belief agent** (`belief_agents.py`): input `(previous_belief, utterance_image)`, output a
  Dirichlet over b_e. Needs to also output b̄_o of the speaker and b̿_e, and to take their
  current values as inputs to update them. `get_obs` now provides all three belief levels
  on both stages.

## The main decision: how are the new estimates produced?

1. **Sampled PPO actions.** Extra Dirichlet heads; the action's log-prob is the sum over
   heads.
2. **Deterministic heads.** Not sampled, not in the PPO ratio; trained by a separate loss,
   or untrained at first.
3. **Theory-of-mind inference from the partner model.** E.g. the speaker runs the listener
   model on its own rendered utterance to get the listener's b̄_o. The `execute_tom_*`
   skeletons in `training/agent_execution.py` already sketch this.

## Code that changes with the decision

- `training/rollout.py`: the four placeholder estimates, and `Transition` (which stores
  exactly two inputs per population).
- `training/ppo_update.py`: assumes two network inputs (`net_input_0`, `net_input_1`) and
  one log-prob per action.
- `training/agent_execution.py`: the `execute_*` functions, the ToM variants, and the
  `InferenceStrategy` signatures.
- `training/ippo.py` `initialize_agents` and the agents' `initialize_*`: dummy inputs.
