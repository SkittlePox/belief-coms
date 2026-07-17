"""Independent PPO (IPPO)."""

from __future__ import annotations
from training.config import ExperimentConfig


def make_train(config: ExperimentConfig):

    # Each sub-config knows how to build its runtime function, so resolution is
    # uniform: assignment -> AssignmentFn, communication -> CommunicationSchemeFn.
    assignment_fn = config.role_assignment.build()
    scheme_fn = config.communication.build()

    # Create stacked signification decpomdp
    # env = ...

    def train(rng):
        # Now build agents and initialize them, and get optimizer states (batch them, apply_fns, etc.)

        pass

    return train
