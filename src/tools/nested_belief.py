"""In-place nested belief updating: a tower of mean beliefs, one memo step at a time.

The problem this solves
-----------------------
A mean-belief estimator has to answer a question it cannot dodge: when I update my
estimate of YOUR belief, what do you assume about MY action? You never saw it. Defaulting
to uniform is a level-0 cop-out; what you actually do is infer my action from YOUR
estimate of MY belief. That is a second level of the hierarchy, and it does not stop
there.

So the state is not one belief but a TOWER, alternating roles as it goes up:

    bel[0]  my belief over states                                  (me)
    bel[1]  my estimate of your belief                             (you)
    bel[2]  my estimate of your estimate of my belief              (me)
    bel[3]  my estimate of your estimate of my estimate of yours   (you)
    ...
    bel[K]  bottoms out: at this level the opponent is modeled as acting uniformly

`ego_action_prior` in tools/belief_representations.py is then not a free parameter at all
-- it is pi_me(bel[2]). At depth K the uniform assumption is banished to level K, where you
can put it as far away as you can afford. This module supersedes that method's
`ego_action_prior=` knob: use the tower instead of picking a prior by hand.

Every level is still a single Categorical -- a MEAN belief, not a sufficient statistic --
so this is the imperfect estimator, deliberately. What it is not is imprecise about the
recursion: the conditioning rules that make it correct are fiddly and easy to get wrong by
hand (whose action is known to whom, whose observation may be conditioned on), so they are
not written by hand. They are what memo's nested `thinks[...]` means, and memo derives
them.

Off-policy warning
------------------
The nested levels never condition on the ego's ACTUAL action -- they only ever marginalize
it under the policy they attribute to the ego. So if the ego acts off-policy, no
contradiction is ever raised and the deep levels drift silently rather than failing loudly.

This is not hypothetical in the guessing game: role_0_optimal_policy is probability
matching, so P(wait) = b(terminal state) = 0 and the optimal presser NEVER waits. Any
history in which the presser waits is off-policy, and the other agent's model of it is
simply wrong. (Exact, history-based inference reports this honestly, as a degenerate
all-zero estimate; this estimator does not.) If you intend to explore, or to run a policy
that has not converged, give the policy a floor so no action has zero likelihood.

What one step does
------------------
Input: the tower bel[0..K], the ego's action a_ego, the ego's observation o_ego.
Output: the updated tower bel'[0..K].

    step(bel, a_ego, o_ego) -> bel'

No history is carried. Feed the output back in next timestep.

Each level k of the tower is the prior of one nested memo frame. Inside frame k the agent
acts on its own belief, and its opponent is frame k+1 -- so the action likelihood at every
level is derived from that level's belief, exactly as you would want. Only the ego's own
action and observation are known facts; everything else is marginalized. The updated
level-k belief is read back out as

    ego[ E[ opp[ E[ opp[ ... Pr[w.s_ == st] ... ] ] ] ] ]

with k nested `E[opp[...]]` wrappers.

Policies
--------
Arbitrary. Each is an OptimalPolicy -- a jittable Categorical(belief) -> Categorical(actions),
the type flexible_env already defines -- and NOTHING has to be linear in the belief. memo
cannot hand a black-box callable an array, but a belief over S states just IS the S
expectations E[s == 0] .. E[s == S-1], so it is passed as S scalars and reassembled:

    opp: chooses(a in Ac, wpp=POL1(E[w.s == 0], ..., E[w.s == S-1], a))

which means a softmax over Q, an argmax, or a neural policy all work as-is. What
`guessing_game_spec()` returns plugs straight in:

    params, policies = guessing_game_spec()
    build_nested_belief_step(params, policies=policies)

Policies are resolved per LEVEL, not once per program. Passing a 2-tuple gives each role the
same policy wherever it appears -- the agents model each other correctly. Passing a list of
depth+1 policies lets level 2 (their model of YOUR policy) differ from level 0 (your actual
policy), which is how you model an opponent who is wrong about you.

Cost
----
The tower is K+1 beliefs and one step, so this is far cheaper than exact history-based
inference -- there is no |H_other| to enumerate, only the single step's latents. Depth
costs, but a step is a step.

Accuracy
--------
Benchmarked against exact recursive inference (the ../../memo-decpomdp scratch repo, which
enumerates the other agent's whole history and is validated against a brute-force
enumeration): on-policy guessing-game histories, this tracks the exact E[b_other] to within
~0.2 max abs error, and depth does not move that number. The residual is the mean-belief
collapse -- one Categorical per level instead of a joint over (state, their belief) -- and
depth cannot fix it. Depth fixes the action-prior question; nothing short of carrying the
joint fixes the other.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

import distrax
import jax
import jax.numpy as jnp
import numpy as np


_REGISTRY: dict[str, dict] = {}
_GENERATED_DIR: Optional[str] = None
_BUILD_COUNTER = itertools.count()


def _write_and_import(src: str, build_id: str):
    global _GENERATED_DIR
    if _GENERATED_DIR is None:
        _GENERATED_DIR = tempfile.mkdtemp(prefix="memo_nested_")
    path = os.path.join(_GENERATED_DIR, f"{build_id}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(build_id, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[build_id] = module
    spec.loader.exec_module(module)
    return module, path


# --------------------------------------------------------------------------- #
# Source generation
#
# One step has to be emitted in two phases, because the minds are interleaved: at every
# level the world's transition needs the opponent's action, and the opponent can only
# choose it once its own mind has been given the prior it acts on. So phase A lays down
# every level's prior and every level's action, and phase B then runs every level's
# transition and observation.
# --------------------------------------------------------------------------- #

def _names(owner_is_ego: bool, top_level: bool):
    """Variable names in one frame. Always keyed to ROLE, never to whose mind we are in,
    so TR and OBS take the same argument order at every nesting level.

        w.s / w.s_   state before / after the step
        w.ox / w.oy  the ego-ROLE / other-ROLE agent's new observation
        pa / pb      the ego-role / other-role agent's action
        opp          this frame's model of its opponent
    """
    own_act = "pa" if owner_is_ego else "pb"
    opp_act = "pb" if owner_is_ego else "pa"
    own_action = "a_ego" if top_level else own_act   # the ego knows what it actually did
    return dict(
        own_action=own_action,
        opp_act=opp_act,
        opp_obs="oy" if owner_is_ego else "ox",
        # TR/OBS always take (ego-role thing, other-role thing) in that order.
        joint=(f"{own_action}, opp.{opp_act}" if owner_is_ego
               else f"opp.{opp_act}, {own_action}"),
    )


def _belief_args(num_states: int) -> str:
    """The agent's belief, as num_states scalars, for handing to a policy function.

    This is what makes ARBITRARY policies expressible. memo has no way to materialize an
    agent's belief as an array and pass it to a black-box callable -- but a belief over S
    states just IS the S expectations E[s == 0] .. E[s == S-1], each a perfectly ordinary
    scalar in the agent's own frame. Spell it that way and the policy can be any jittable
    function of the full belief vector: a softmax over Q, an argmax, a neural network.
    Nothing here has to be linear in the belief.
    """
    return ", ".join(f"E[w.s == {s}]" for s in range(num_states))


def _phase_a(level: int, depth: int, owner_is_ego: bool, indent: str, num_states: int,
             top_level: bool = False) -> list[str]:
    """Priors and actions, from this level all the way down."""
    n = _names(owner_is_ego, top_level)
    out = [
        # THE point of the whole module: this level's prior is the tower entry for this
        # level, not the environment's initial distribution.
        f"{indent}w: chooses(s in S, wpp=PRIOR(s, bel{level})),",
        f"{indent}opp: knows(st),",
    ]
    if level < depth:
        # The opponent has a mind, whose prior is the next level of the tower. It acts on
        # the belief that mind holds -- which is exactly the action likelihood we want,
        # instead of a uniform guess. The E[...] terms are evaluated inside the OPPONENT's
        # frame, so the belief handed to POL is the opponent's, i.e. level+1's.
        out.append(f"{indent}opp: thinks[")
        out.extend(_phase_a(level + 1, depth, not owner_is_ego, indent + "    ", num_states))
        out.append(f"{indent}],")
        out.append(f"{indent}opp: chooses({n['opp_act']} in Ac, "
                   f"wpp=POL{level + 1}({_belief_args(num_states)}, {n['opp_act']})),")
    else:
        # The bottom. Someone has to stop, and here the opponent gets no mind. Push this
        # further away by raising `depth`.
        out.append(f"{indent}opp: chooses({n['opp_act']} in Ac, uniformly),")
    return out


def _phase_b(level: int, depth: int, owner_is_ego: bool, indent: str,
             top_level: bool = False) -> list[str]:
    """Transitions and observations, from this level all the way down."""
    n = _names(owner_is_ego, top_level)
    out: list[str] = []

    if level < depth:
        # Advance the opponent's mind through the same step, so it ends up with a
        # posterior over the NEW state -- that posterior is its updated belief.
        out.append(f"{indent}opp: thinks[")
        out.extend(_phase_b(level + 1, depth, not owner_is_ego, indent + "    "))
        out.append(f"{indent}],")

    # The transition conditions on both actions, so both must be visible in w's frame. The
    # top-level ego's action is a memo parameter, already in scope everywhere.
    if not top_level:
        out.append(f"{indent}w: knows({n['own_action']}),")
    out.append(f"{indent}w: knows(opp.{n['opp_act']}),")
    out.append(f"{indent}w: chooses(s_ in S, wpp=TR(s, {n['joint']}, s_)),")
    out.append(f"{indent}w: chooses(ox in Ob, oy in Ob, "
               f"wpp=OBS(ox, oy, s_, {n['joint']})),")

    if level < depth:
        # The opponent sees its own observation -- and only its own. This is the asymmetry
        # that is so easy to fumble by hand: it must NOT see the ego's observation, and
        # its posterior must NOT condition on the ego's action.
        out.append(f"{indent}opp: observes [w.{n['opp_obs']}] is w.{n['opp_obs']},")

    return out


def _init_mind(level: int, depth: int, owner_is_ego: bool, indent: str,
               top_level: bool = False) -> list[str]:
    """The reset step: a prior and an observation, no action and no transition.

    Seeding the tower matters and is easy to botch. At t=0 the other agent has ALREADY
    seen its own reset observation, which is correlated with the ego's through the state
    -- so bel[1] at t=0 is not the environment prior, and neither is bel[2]. Starting the
    tower at the prior throws a full step of evidence away before the filter even begins.
    """
    n = _names(owner_is_ego, top_level)
    out = [
        # Every mind -- at every level, for either role -- starts from the world prior.
        # There is no per-role initial belief: before anyone observes anything, my estimate
        # of your belief, and of your estimate of mine, are all just P0.
        f"{indent}w: chooses(s in S, wpp=P0(s)),",
        f"{indent}opp: knows(st),",
    ]
    if level < depth:
        out.append(f"{indent}opp: thinks[")
        out.extend(_init_mind(level + 1, depth, not owner_is_ego, indent + "    "))
        out.append(f"{indent}],")
    out.append(f"{indent}w: chooses(ox in Ob, oy in Ob, wpp=OBS_RESET(ox, oy, s)),")
    if level < depth:
        out.append(f"{indent}opp: observes [w.{n['opp_obs']}] is w.{n['opp_obs']},")
    return out


def _model_src(depth: int, num_states: int, build_id: str) -> str:
    belief_params = "".join(f", bel{k}: ..." for k in range(depth + 1))

    lines = [
        "@memo",
        "def init[st: S](o_ego):",
        "    ego: knows(st)",
        "    ego: thinks[",
        *_init_mind(0, depth, owner_is_ego=True, indent=" " * 8, top_level=True),
        "    ]",
        "    ego: observes_that [w.ox == o_ego]",
    ]
    for k in range(depth + 1):
        expr = "Pr[w.s == st]"
        for _ in range(k):
            expr = f"E[ opp[ {expr} ] ]"
        lines.append(f"    return ego[ {expr} ]")

    lines += [
        "",
        "",
        "@memo",
        f"def step[st: S](a_ego, o_ego{belief_params}):",
        "    ego: knows(st)",
        "    ego: thinks[",
        *_phase_a(0, depth, owner_is_ego=True, indent=" " * 8, num_states=num_states,
                  top_level=True),
        *_phase_b(0, depth, owner_is_ego=True, indent=" " * 8, top_level=True),
        "    ]",
        # The ego knows what it saw. Its own action is already the constant a_ego, and
        # carries no information about s beyond the belief that produced it.
        "    ego: observes_that [w.ox == o_ego]",
    ]

    # Read each level of the updated tower back out: k nested E[opp[...]] wrappers walk
    # k frames down into the hierarchy.
    for k in range(depth + 1):
        expr = "Pr[w.s_ == st]"
        for _ in range(k):
            expr = f"E[ opp[ {expr} ] ]"
        lines.append(f"    return ego[ {expr} ]")

    preamble = [
        "from memo import memo",
        "from tools.nested_belief import _REGISTRY",
        "",
        f"_r = _REGISTRY[{build_id!r}]",
        "S, Ac, Ob = _r['S'], _r['Ac'], _r['Ob']",
        "P0, PRIOR = _r['P0'], _r['PRIOR']",
        "TR, OBS, OBS_RESET = _r['TR'], _r['OBS'], _r['OBS_RESET']",
        *[f"POL{k} = _r['POL{k}']" for k in range(1, depth + 1)],
        "",
        "",
    ]
    return "\n".join(preamble + lines) + "\n"


# --------------------------------------------------------------------------- #

@dataclass
class NestedBeliefStep:
    """One in-place step of the nested belief tower, compiled for one (env, role, depth)."""
    module: object
    ego_role: int
    depth: int
    num_states: int
    num_actions: int
    num_observations: int
    source: str
    path: str
    fn: Callable
    init_fn: Callable

    def initial_tower(self, ego_observation):
        """Seed the tower from the ego's reset observation, with the same nesting.

        Not the environment prior: at t=0 the other agent has already seen ITS reset
        observation, correlated with the ego's through the state, so bel[1] and bel[2] have
        already moved. Starting at the prior discards a whole step of evidence.
        """
        out = self.init_fn(int(ego_observation))
        return [np.asarray(b) for b in out]

    def __call__(self, tower, previous_ego_action, ego_observation):
        """Advance the whole tower one step.

        Args:
            tower: list of depth+1 length-S arrays, tower[k] as described in the module
                docstring.
            previous_ego_action: the ego's OWN action last step (not the joint action).
            ego_observation: the observation the ego just received.

        Returns:
            The updated tower, same shape. Feed it straight back in next step.
        """
        if len(tower) != self.depth + 1:
            raise ValueError(f"expected a tower of {self.depth + 1} beliefs, got {len(tower)}")
        out = self.fn(
            int(previous_ego_action),
            int(ego_observation),
            *[jnp.asarray(b, dtype=jnp.float32) for b in tower],
        )
        return [np.asarray(b) for b in out]


def _as_memo_policy(policy, num_states: int) -> Callable:
    """Wrap an OptimalPolicy (Categorical -> Categorical) into what memo can call.

    memo calls a jitted function with SCALAR arguments, so the belief arrives as
    num_states separate floats (see _belief_args). Reassemble it, hand it to the policy,
    and return the probability of the one action being asked about. The policy itself can
    be anything jittable -- softmax over Q, argmax, a neural net. It does not have to be
    linear in the belief, and it does not have to be differentiable in a nice way.
    """
    @jax.jit
    def memo_policy(*args):
        belief = jnp.stack(args[:num_states])
        action = args[num_states]
        belief = belief / jnp.clip(jnp.sum(belief), 1e-12)   # memo hands us E[...] terms
        return policy(distrax.Categorical(probs=belief)).probs[action]

    return memo_policy


def build_nested_belief_step(
    env_params,
    ego_role: int = 0,
    depth: int = 2,
    policies=None,
    policy_weight: Optional[Callable] = None,
) -> NestedBeliefStep:
    """Compile the in-place nested belief step for a DecPOMDP.

    Args:
        env_params: a FlexibleEnvParams (or the same fields).
        ego_role: which agent (0 or 1) holds the tower.
        depth: how many levels of the hierarchy to carry. The tower has depth+1 beliefs.
            depth=1 gives you an estimate of their belief, but their model of YOUR action
            is still uniform -- i.e. the thing you were unhappy about. depth=2 is the first
            level at which their estimate of your belief drives their guess at your action;
            depth>=3 keeps pushing the uniform assumption further away.

        policies: the agents' policies. ARBITRARY -- each is an OptimalPolicy, i.e. a
            jittable Categorical(belief) -> Categorical(actions), exactly the type
            flexible_env already defines and guessing_game_spec already returns. Nothing
            needs to be linear in the belief; a softmax over Q, an argmax, or a neural
            policy are all fine. Two accepted forms:

              by ROLE   a 2-tuple (pi_role0, pi_role1). Each role uses the same policy
                        wherever it appears in the hierarchy -- i.e. the agents model each
                        other's policies CORRECTLY. This is what you normally want, and it
                        is what `guessing_game_spec()` hands you:

                            params, policies = guessing_game_spec()
                            build_nested_belief_step(params, policies=policies)

              by LEVEL  a list of depth+1 policies, policies[k] being the policy of the
                        agent at level k of the tower. This is how you represent an
                        opponent who is WRONG about you: make policies[2] -- their model of
                        your policy -- differ from policies[0], your actual one. Level 0's
                        entry is never used (the ego's action is a known fact, not a
                        prediction), but keep the list aligned with the tower.

        policy_weight: a shortcut for belief-LINEAR policies only, jitted w(role, s, a)
            with pi_role(a|b) proportional to sum_s b(s) w(role, s, a). Defaults to
            1[s == a] (probability matching), which is the guessing game's optimal policy
            for both roles -- but that default only means anything when action indices ARE
            state indices, so it is refused unless num_actions == num_states. Ignored if
            `policies` is given.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1 to represent the other agent's belief at all")
    if ego_role not in (0, 1):
        raise ValueError("ego_role must be 0 or 1")
    if (policies is None and policy_weight is None
            and int(env_params.num_actions) != int(env_params.num_states)):
        # Left to itself, 1[s == a] would quietly give some states no action at all (and
        # some actions no state), and every level of the tower would be built on a policy
        # that means nothing. Fail instead of returning a confident wrong answer.
        raise ValueError(
            "the default policy_weight 1[s == a] assumes action indices are state indices, "
            f"but this environment has {int(env_params.num_states)} states and "
            f"{int(env_params.num_actions)} actions. Pass policy_weight=w(role, s, a)."
        )

    transition = jnp.asarray(env_params.transition)
    observation = jnp.asarray(env_params.observation)
    # One prior, shared by the world and by every agent at every level of the hierarchy.
    initial = jnp.asarray(env_params.initial_state_distribution)
    num_states = int(env_params.num_states)
    num_actions = int(env_params.num_actions)
    num_observations = int(observation.shape[-1])
    other_role = 1 - ego_role

    # Resolve `policies` down to one OptimalPolicy per nesting LEVEL. Levels alternate
    # role, so by default level k's policy is simply that role's policy -- the agents model
    # each other correctly. A per-level list overrides that, which is how you give the
    # opponent a mistaken model of you.
    if policies is None:
        if policy_weight is None:
            @jax.jit
            def policy_weight(role, s, a):  # noqa: F811 -- probability matching
                del role
                return (s == a) * 1.0

        def _from_weight(role):
            def pi(belief: "distrax.Categorical") -> "distrax.Categorical":
                weights = jax.vmap(
                    lambda a: jnp.sum(jax.vmap(
                        lambda s: belief.probs[s] * policy_weight(role, s, a)
                    )(jnp.arange(num_states)))
                )(jnp.arange(num_actions))
                return distrax.Categorical(probs=weights / jnp.sum(weights))
            return pi

        by_role = (_from_weight(0), _from_weight(1))
        policy_for_level = lambda k: by_role[ego_role if k % 2 == 0 else other_role]
    elif len(policies) == depth + 1:                   # by level
        policy_for_level = lambda k: policies[k]
    elif len(policies) == 2:                           # by role
        policy_for_level = lambda k: policies[ego_role if k % 2 == 0 else other_role]
    else:
        raise ValueError(
            f"policies must be a 2-tuple (by role) or a list of {depth + 1} (by level), "
            f"got {len(policies)}"
        )

    def _order(ego_thing, other_thing):
        return (ego_thing, other_thing) if ego_role == 0 else (other_thing, ego_thing)

    @jax.jit
    def P0(s):
        return initial[s]

    @jax.jit
    def PRIOR(s, belief):
        return belief[s]

    @jax.jit
    def TR(s, a_ego, a_other, s_):
        a0, a1 = _order(a_ego, a_other)
        return transition[s, a0, a1, s_]

    @jax.jit
    def OBS(o_ego, o_other, s_, a_ego, a_other):
        a0, a1 = _order(a_ego, a_other)
        o0, o1 = _order(o_ego, o_other)
        return observation[s_, a0, a1, o0, o1]

    @jax.jit
    def OBS_RESET(o_ego, o_other, s):
        # FlexibleEnv.get_obs queries the observation tensor with a no-op joint action at
        # reset; mirror it so the model agrees with the environment it is modeling.
        return OBS(o_ego, o_other, s, 0, 0)

    # One policy function per NESTING LEVEL, not per role. Levels alternate role, and by
    # default a role uses the same policy wherever it appears -- i.e. the agents model each
    # other's policies correctly. Passing `policies` as a per-level list breaks that: it
    # lets level 2 (their model of MY policy) differ from level 0 (my actual policy), which
    # is how you represent an opponent who is WRONG about how you behave.
    level_policies = {}
    for k in range(1, depth + 1):
        pol = policy_for_level(k)
        level_policies[k] = _as_memo_policy(pol, num_states)

    build_id = f"memo_nested_g{next(_BUILD_COUNTER)}"
    _REGISTRY[build_id] = dict(
        S=jnp.arange(num_states),
        Ac=jnp.arange(num_actions),
        Ob=jnp.arange(num_observations),
        P0=P0, PRIOR=PRIOR, TR=TR, OBS=OBS, OBS_RESET=OBS_RESET,
        **{f"POL{k}": level_policies[k] for k in range(1, depth + 1)},
    )

    src = _model_src(depth, num_states, build_id)
    module, path = _write_and_import(src, build_id)

    return NestedBeliefStep(
        module=module,
        ego_role=ego_role,
        depth=depth,
        num_states=num_states,
        num_actions=num_actions,
        num_observations=num_observations,
        source=src,
        path=path,
        fn=jax.jit(module.step),
        init_fn=jax.jit(module.init),
    )
