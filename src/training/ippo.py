"""Independent PPO (IPPO)."""

from __future__ import annotations
from training.config import ExperimentConfig


def make_train(config: ExperimentConfig):

    # Each sub-config knows how to build its runtime function, so resolution is
    # uniform: routing -> RouteFn, communication -> CommunicationSchemeFn.
    route_fn = config.routing.build()
    scheme_fn = config.communication.build()

    # Create stacked signification decpomdp
    # env = ...

    def train(rng):
        pass

    return train
