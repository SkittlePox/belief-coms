import jax, chex
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from flax import struct
from typing import Any
from functools import partial

from routing import AgentGameRoleRoute, RouteFn

# =============================================================================
# DESIGN NOTE: representing a variable, time-changing mix of heterogeneous games
# =============================================================================
#
# Scenario: many *kinds* of games run concurrently (e.g. 3 sig1 + 2 sig2 now,
# 4 sig2 + 1 sig3 later), each type with possibly different pomdp state. The TOTAL
# number of concurrent games (G) is constant across timesteps.
#
# Core constraint: under jit/scan the state pytree structure (which arrays exist,
# and their shapes) must be STATIC across timesteps. So the mix can't change by
# adding/removing arrays -- the array set is fixed for all time; only CONTENTS and
# LIVENESS change. Rules out anything that literally grows/shrinks per type.
#
# -----------------------------------------------------------------------------
# Pattern A -- typed pools + active masks
# -----------------------------------------------------------------------------
# One field per game TYPE: that type's state dataclass padded to a fixed CAPACITY,
# plus a bool `active` mask. Step: vmap each type's OWN step over its OWN pool
# (homogeneous -> no dispatch, each type keeps its own clean dataclass). Mix change
# = flip `active` bits + scatter new state into freed slots.
#
#   @struct.dataclass
#   class StackedState:
#       sig1_states: Sig1State   # leading axis [cap_1]; likewise sig2/sig3
#       sig1_active: chex.Array  # bool [cap_1]
#       iteration: int
#
# Cost: worst case any type can fill all G slots -> cap_t = G each -> num_types * G
# slots held AND stepped per timestep, only G ever active (e.g. 10 types, G=5 -> 50
# vs 5). Tighter per-type bounds shrink this.
#
# -----------------------------------------------------------------------------
# Pattern B -- tagged union + lax.switch
# -----------------------------------------------------------------------------
# A single length-G batch whose dataclass is the UNION of all types' fields, plus a
# game_type[G] tag. Step: vmap over G, inside lax.switch(game_type[i], branch_fns).
# Exactly G slots of work; mix change is just RETAGGING a slot. Downside: every slot
# carries all types' fields (wasteful only if states are large and disjoint).
#
#   @struct.dataclass
#   class StackedState:
#       decpomdp_state: UnionState  # all types' fields, leading axis [G]
#       game_type: chex.Array       # int [G]
#       iteration: int
#
# lax.switch needs every branch to share inputs + output pytree structure (the
# UnionState slice), but each game's step wants its own SigKState -- so wrap each
# step in adapters: from_union (slice -> SigKState) in, to_union (SigKState -> slice)
# out. to_union re-embeds k's fields and carries other types' fields unchanged.
#
#   def make_branch(sigk, from_u, to_u):
#       return lambda u, key, act: to_u(sigk.step(key, from_u(u), act), u)
#   branch_fns = [make_branch(games[k], from_union[k], to_union[k]) for k in types]
#   new_slice  = jax.lax.switch(game_type[i], branch_fns, union_slice, key, actions)
#
# -----------------------------------------------------------------------------
# Decision tree (settle the actual game set first -- it picks the row)
# -----------------------------------------------------------------------------
#   States differ a LOT             -> Pattern A; eat the num_types*G over-allocation.
#   States similar, dynamics differ -> Pattern B (union + lax.switch + adapters).
#     as CODE                          [currently leaning here]
#   States similar, dynamics differ -> Neither: one vmapped step over per-game param
#     only as DATA                       arrays ([G] axis), indexing per slot. Cleanest.
#
# Routing knock-on: with either typed pools or a union, a game is addressed by
# (type, slot). AgentGameRoleRoute.global_to_local_game currently encodes a flat id
# -- extend it to a (type, slot) pair so routing and StackedState stay consistent.
# =============================================================================

@struct.dataclass
class StackedState:
    """
    Full state for stacked SignificationPOMDP
    """

    decpomdp_state: Any  # An array of states, each one corresponding to a DecPOMDP

    iteration: int

class StackedSignificationPOMDP():
    """
    This class has multiple agents that are passed to a series of SignificationPOMDPs
    This class assigns agents to games according to an assignment scheduler
        I think it should be agent to global role number, and then global role number to game + game role number
    We also need to pass a set of 
    """

    def __init__(self, num_agents: int, route_fn: RouteFn) -> None:
        """
        Args:
            num_agents, always divisible by 2
            env_schedule_function
        """
        self.num_agents = num_agents
        self.route_fn = route_fn
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):

        return (initial_environment_state, self.get_obs(key, initial_environment_state))


