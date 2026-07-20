"""Optimizer config for the agent train states.

``OptimizerConfig`` follows the same tyro-facing ``build()`` idiom as the assignment /
communication / agent configs: a dataclass of knobs whose ``build()`` returns the
configured object -- here an ``optax.GradientTransformation`` -- keeping the training
code free of optimizer-parsing. Belief and utterance agents each get their own instance
(they are separate populations trained independently), so their learning rates and
clipping can diverge without touching the other.
"""

from __future__ import annotations
import dataclasses
import optax


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Knobs for the Adam optimizer wrapping each agent's train state.

    Fields:
        learning_rate: Adam step size.
        max_grad_norm: Global-norm gradient clip applied before the Adam update.
        b1, b2: Adam moment-decay rates.
        eps: Adam numerical-stability epsilon.
    """

    learning_rate: float = 1e-3
    max_grad_norm: float = 0.5
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-5

    def build(self) -> optax.GradientTransformation:
        # Clip first, then Adam -- the standard PPO ordering (bound the raw gradient's
        # global norm before the adaptive rescaling sees it).
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate, b1=self.b1, b2=self.b2, eps=self.eps),
        )
