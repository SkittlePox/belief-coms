"""Environment assembly for the stacked signification setting.

A single game type is described by an :class:`EnvSpec` — a builder returning one
(unstacked) :class:`FlexibleEnvParams` together with its per-role optimal
policies (e.g. ``envs.guessing_game.guessing_game_spec``).
:func:`assemble_environments` turns a list of specs (one per game type) into the
two things ``StackedSignificationDecPOMDP`` consumes:

  * one ``FlexibleEnvParams`` whose array fields are padded to common
    ``(S, A, O)`` maxima and stacked along a leading game-type axis, and
  * a nested ``[game_type][role]`` table of optimal policies, kept as plain
    Python callables (not traceable array data).

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
import jax.numpy as jnp
from typing import Callable, Sequence, Tuple

from envs.flexible_env import FlexibleEnvParams, OptimalPolicy
from envs.guessing_game import guessing_game_spec


# An EnvSpec builds a single (unstacked) env's params + its per-role policies.
EnvSpec = Callable[[], Tuple[FlexibleEnvParams, Sequence[OptimalPolicy]]]


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


def assemble_environments(specs: Sequence[EnvSpec]):
    """Build the stacked params and policy table from a list of EnvSpecs.

    Args:
        specs: One EnvSpec per game type. Spec ``i`` becomes game type ``i``.

    Returns:
        (stacked_params, optimal_policies) where stacked_params is a
        FlexibleEnvParams with a leading game-type axis on every field, and
        optimal_policies is the nested [game_type][role] tuple of callables.
    """
    built = [spec() for spec in specs]
    params_list = [params for params, _ in built]
    policy_lists = [tuple(policies) for _, policies in built]

    target_states = max(p.transition.shape[0] for p in params_list)
    target_actions = max(p.transition.shape[1] for p in params_list)
    target_obs = max(p.observation.shape[-1] for p in params_list)

    padded = [
        _pad_params(p, target_states, target_actions, target_obs) for p in params_list
    ]

    # Stack matching leaves across game types -> leading game-type axis.
    stacked_params = jax.tree.map(lambda *leaves: jnp.stack(leaves), *padded)

    optimal_policies = tuple(policy_lists)  # [game_type][role]
    return stacked_params, optimal_policies


if __name__ == "__main__":
    # Assemble a single-game-type environment and show the stacked shapes.
    stacked_params, optimal_policies = assemble_environments([guessing_game_spec])

    print("stacked transition:           ", stacked_params.transition.shape)
    print("stacked observation:          ", stacked_params.observation.shape)
    print("stacked reward:               ", stacked_params.reward.shape)
    print("stacked initial_state_dist:   ", stacked_params.initial_state_distribution.shape)
    print("stacked terminal_mask:        ", stacked_params.terminal_mask.shape)
    print("num game types:               ", len(optimal_policies))
    print("num roles:                    ", len(optimal_policies[0]))
