"""The launch<->ippo shared contract: ExperimentConfig.

``ExperimentConfig`` is the one config imported by *both* the launcher
(``launch.py``) and the algorithm (``ippo.py``), so it must live in a module that
imports neither -- otherwise launch (imports ippo) and ippo (would import launch)
form a cycle. That is the sole reason this module exists. Every other config lives
with the code that consumes it: domain configs in their domain modules,
``WandbConfig`` in ``launch.py``.
"""

from __future__ import annotations
import dataclasses

# Each domain's config type is owned by its library module (colocated with the code
# it configures, so options and code can't drift); this module just assembles them
# into ExperimentConfig. CommunicationConfig is a flat scheme selector; RoutingConfig
# is a Union of per-family configs (disjoint params, each with build()) that tyro
# renders as subcommands.
from communication.communication_scheme import CommunicationConfig
from communication.routing import RoutingConfig, SimpleRoutingConfig


@dataclasses.dataclass
class ExperimentConfig:
    """Full run config."""

    seed: int = 42
    learning_rate: float = 1e-3
    routing: RoutingConfig = dataclasses.field(default_factory=SimpleRoutingConfig)
    communication: CommunicationConfig = dataclasses.field(default_factory=CommunicationConfig)
