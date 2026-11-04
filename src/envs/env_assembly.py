"""Environment assembly for the stacked signification setting.

A single game type is described by an :class:`EnvSpec` — a builder returning one
(unstacked) :class:`FlexibleEnvParams` together with its per-role optimal
policies (e.g. ``envs.guessing_game.guessing_game_spec``).
:func:`assemble_environments` turns a list of specs (one per game type) into the
two things ``StackedSignificationDecPOMDP`` consumes:

  * one ``FlexibleEnvParams`` whose array fields are padded to common
    ``(S, A, O)`` maxima and stacked along a leading game-type axis, and
  * an optimal policy bundle: a single function
    ``optimal_policy_bundle(game_type, role, belief) -> Categorical`` that
    dispatches to the right per-role policy via ``lax.switch``, so
    ``game_type`` and ``role`` may be traced.

``FlexibleEnvParams`` / ``OptimalPolicy`` and the generic runtime live in
``envs.flexible_env`` (the leaf of the dependency graph); this module imports
them and the per-env specs, never the reverse.

Padding note: padded states/actions/observations are zero-probability (and
non-terminal), so they are never reached or selected as long as consumers
respect the recorded ``num_states`` / ``num_actions``. When mixing game types
with *different action counts*, the per-role policies must also return action
distributions of the common (padded) size so ``lax.switch`` dispatch sees a
uniform output shape.
"""

import jax
import distrax
import jax.numpy as jnp
from typing import Callable, Sequence, Tuple

from envs.flexible_env import NUM_ROLES, FlexibleEnvParams, OptimalPolicy, OptimalPolicyBundle
from envs.guessing_game import guessing_game_spec


# An EnvSpec builds a single (unstacked) env's params + its per-role policies.
EnvSpec = Callable[[], Tuple[FlexibleEnvParams, Sequence[OptimalPolicy]]]


# Registry of every game type available to a run, keyed by name. A game's TYPE INDEX
# (the value stored in ``game_set`` / read by ``game_type_id``) is its position in the
# list handed to ``assemble_environments`` -- for a run that assembles every registered
# game, that is this dict's insertion order. Add a new game type by writing its EnvSpec
# and registering it here; it then becomes selectable from the config.
GAME_SPECS: dict[str, EnvSpec] = {
    "guessing_game": guessing_game_spec,
}


def _pad_params(p: FlexibleEnvParams, target_states, target_actions, target_obs) -> FlexibleEnvParams:
    """Zero-pad a single env's tensors up to common (S, A, O) maxima.

    Cardinalities are read from tensor shapes (not the num_* fields). num_states /
    num_actions are left untouched so they keep recording the true, unpadded sizes.
    """
    num_states = p.transition.shape[0]
    num_actions = p.transition.shape[1]
    num_obs = p.observation.shape[-1]

    pad_states = target_states - num_states
    pad_actions = target_actions - num_actions
    pad_obs = target_obs - num_obs

    return p.replace(
        # [S, A, A, S]
        transition=jnp.pad(
            p.transition,
            ((0, pad_states), (0, pad_actions), (0, pad_actions), (0, pad_states)),
        ),
        # [S, A, A, O, O]
        observation=jnp.pad(
            p.observation,
            ((0, pad_states), (0, pad_actions), (0, pad_actions), (0, pad_obs), (0, pad_obs)),
        ),
        # [N, S, A, A, S]
        reward=jnp.pad(
            p.reward,
            ((0, 0), (0, pad_states), (0, pad_actions), (0, pad_actions), (0, pad_states)),
        ),
        # [num_roles, S]
        # [S]
        initial_state_distribution=jnp.pad(p.initial_state_distribution, (0, pad_states)),
        # [S] — padded states are non-terminal (0)
        terminal_mask=jnp.pad(p.terminal_mask, (0, pad_states)),
    )


def assemble_environments(specs: Sequence[EnvSpec]) -> Tuple[FlexibleEnvParams, OptimalPolicyBundle]:
    """Build the stacked params and optimal policy bundle from a list of EnvSpecs.

    Args:
        specs: One EnvSpec per game type. Spec ``i`` becomes game type ``i``.

    Returns:
        (stacked_params, optimal_policy_bundle) where stacked_params is a
        FlexibleEnvParams with a leading game-type axis on every field, and
        optimal_policy_bundle(game_type, role, belief) applies that game type's
        optimal policy for that role (indices may be traced).
    """
    built = [spec() for spec in specs]
    params_list = [params for params, _ in built]
    # Flatten row-major [game_type][role] for lax.switch dispatch.
    flat_policies = []
    for _, policies in built:
        assert len(policies) == NUM_ROLES, f"each EnvSpec must provide {NUM_ROLES} role policies, got {len(policies)}"
        flat_policies.extend(policies)
    flat_policies = tuple(flat_policies)

    target_states = max(p.transition.shape[0] for p in params_list)
    target_actions = max(p.transition.shape[1] for p in params_list)
    target_obs = max(p.observation.shape[-1] for p in params_list)

    padded = [_pad_params(p, target_states, target_actions, target_obs) for p in params_list]

    # Stack matching leaves across game types -> leading game-type axis.
    stacked_params = jax.tree.map(lambda *leaves: jnp.stack(leaves), *padded)

    def optimal_policy_bundle(game_type, role, belief: distrax.Categorical) -> distrax.Categorical:
        return jax.lax.switch(game_type * NUM_ROLES + role, flat_policies, belief)

    return stacked_params, optimal_policy_bundle


if __name__ == "__main__":
    # Assemble a single-game-type environment and show the stacked shapes.
    stacked_params, optimal_policy_bundle = assemble_environments([guessing_game_spec])

    print("stacked transition:           ", stacked_params.transition.shape)
    print("stacked observation:          ", stacked_params.observation.shape)
    print("stacked reward:               ", stacked_params.reward.shape)
    print("stacked initial_state_dist:   ", stacked_params.initial_state_distribution.shape)
    print("stacked terminal_mask:        ", stacked_params.terminal_mask.shape)
    belief = distrax.Categorical(probs=stacked_params.initial_state_distribution[0])
    print("role 0 policy on prior:       ", optimal_policy_bundle(0, 0, belief).probs)
