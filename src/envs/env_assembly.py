"""Environment assembly for the stacked signification setting.

A single game type is described by an :class:`EnvSpec` — a builder returning a
:class:`GameSpec`: one (unstacked) :class:`FlexibleEnvParams`, its per-role Q
functions and the inverse temperature of their Boltzmann policies (e.g.
``envs.guessing_game.guessing_game_spec``). :func:`assemble_environments` turns a
list of specs (one per game type) into the two things
``StackedSignificationDecPOMDP`` consumes:

  * one ``FlexibleEnvParams`` whose array fields are padded to common
    ``(S, A, O)`` maxima and stacked along a leading game-type axis, and
  * a :class:`BoltzmannPolicyBundle`, called as
    ``policy_bundle(game_type, role, ego_belief, other_belief_estimate) -> Categorical``.
    It dispatches to the right per-role Q function via ``lax.switch``, so
    ``game_type`` and ``role`` may be traced.

``FlexibleEnvParams`` / ``GameSpec`` / ``QFunction`` and the generic runtime live in
``envs.flexible_env`` (the leaf of the dependency graph); this module imports
them and the per-env specs, never the reverse.

Padding note: padded states/actions/observations are zero-probability (and
non-terminal), so they are never reached or selected as long as consumers
respect the recorded ``num_states`` / ``num_actions``. Each Q function returns
values for its own game's actions; the bundle pads them to the common action count
with -inf, so ``lax.switch`` sees one output shape and padded actions get zero
probability. Q functions receive beliefs over the common (padded) state space, whose
padded tail carries zero mass.
"""

import jax
import distrax
import jax.numpy as jnp
from typing import Callable, Sequence, Tuple

from envs.flexible_env import NUM_ROLES, FlexibleEnvParams, GameSpec, QFunction, boltzmann_policy
from envs.guessing_game import guessing_game_spec


# An EnvSpec builds a single (unstacked) game type: its params, per-role Q functions and β.
EnvSpec = Callable[[], GameSpec]


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


def _pad_q_function(q_function: QFunction, target_actions: int) -> QFunction:
    """Wrap a game's Q function so its values are padded to ``target_actions`` with -inf."""

    def padded(ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical) -> jnp.ndarray:
        q_values = q_function(ego_belief, other_belief_estimate)
        return jnp.pad(q_values, (0, target_actions - q_values.shape[0]), constant_values=-jnp.inf)

    return padded


class BoltzmannPolicyBundle:
    """Per-(game_type, role) Q functions and the Boltzmann policies built on them.

    ``game_type`` and ``role`` may be traced. Calling the bundle gives the policy:

        policy_bundle(game_type, role, ego_belief, other_belief_estimate)
            = boltzmann_policy(Q_{game_type, role}(b_e, b̄_o), β_{game_type})

    "ego" is whoever is acting. To predict the OTHER agent's action, call it with the
    other's role and the ego's estimates of the other's inputs, (b̄_o, b̿_e).
    """

    def __init__(self, flat_q_functions: Sequence[QFunction], inverse_temperatures: jnp.ndarray):
        self._flat_q_functions = tuple(flat_q_functions)  # row-major [game_type][role]
        self.inverse_temperatures = inverse_temperatures  # [num_game_types]

    def q_values(self, game_type, role, ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical) -> jnp.ndarray:
        """Q_{game_type, role}(b_e, b̄_o), shape [A] (padded actions are -inf)."""
        return jax.lax.switch(game_type * NUM_ROLES + role, self._flat_q_functions, ego_belief, other_belief_estimate)

    def __call__(self, game_type, role, ego_belief: distrax.Categorical, other_belief_estimate: distrax.Categorical) -> distrax.Categorical:
        q_values = self.q_values(game_type, role, ego_belief, other_belief_estimate)
        return boltzmann_policy(q_values, self.inverse_temperatures[game_type])


def assemble_environments(specs: Sequence[EnvSpec]) -> Tuple[FlexibleEnvParams, BoltzmannPolicyBundle]:
    """Build the stacked params and Boltzmann policy bundle from a list of EnvSpecs.

    Args:
        specs: One EnvSpec per game type. Spec ``i`` becomes game type ``i``.

    Returns:
        (stacked_params, policy_bundle) where stacked_params is a FlexibleEnvParams with
        a leading game-type axis on every field, and policy_bundle is a
        BoltzmannPolicyBundle over every (game type, role).
    """
    built = [spec() for spec in specs]
    params_list = [game.params for game in built]

    target_states = max(p.transition.shape[0] for p in params_list)
    target_actions = max(p.transition.shape[1] for p in params_list)
    target_obs = max(p.observation.shape[-1] for p in params_list)

    # Flatten row-major [game_type][role] for lax.switch dispatch.
    flat_q_functions = []
    for game in built:
        assert len(game.q_functions) == NUM_ROLES, f"each EnvSpec must provide {NUM_ROLES} role Q functions, got {len(game.q_functions)}"
        flat_q_functions.extend(_pad_q_function(q, target_actions) for q in game.q_functions)
    inverse_temperatures = jnp.array([game.inverse_temperature for game in built], dtype=jnp.float32)

    padded = [_pad_params(p, target_states, target_actions, target_obs) for p in params_list]

    # Stack matching leaves across game types -> leading game-type axis.
    stacked_params = jax.tree.map(lambda *leaves: jnp.stack(leaves), *padded)

    return stacked_params, BoltzmannPolicyBundle(flat_q_functions, inverse_temperatures)

if __name__ == "__main__":
    # Assemble a single-game-type environment and show the stacked shapes.
    stacked_params, policy_bundle = assemble_environments([guessing_game_spec])

    print("stacked transition:           ", stacked_params.transition.shape)
    print("stacked observation:          ", stacked_params.observation.shape)
    print("stacked reward:               ", stacked_params.reward.shape)
    print("stacked initial_state_dist:   ", stacked_params.initial_state_distribution.shape)
    print("stacked terminal_mask:        ", stacked_params.terminal_mask.shape)
    belief = distrax.Categorical(probs=stacked_params.initial_state_distribution[0])
    print("role 0 Q on prior:            ", policy_bundle.q_values(0, 0, belief, belief))
    print("role 0 policy on prior:       ", policy_bundle(0, 0, belief, belief).probs)
    print("role 1 policy on prior:       ", policy_bundle(0, 1, belief, belief).probs)
