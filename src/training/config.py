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
# into ExperimentConfig. CommunicationConfig is a flat scheme selector; AssignmentConfig
# is a Union of per-family configs (disjoint params, each with build()) that tyro
# renders as subcommands.
from communication.communication_scheme import CommunicationConfig
from communication.game_role_assignment import AssignmentConfig, SimpleAssignmentConfig
from communication.stacked_signification_decpomdp import EnvironmentConfig
from agents.belief_agents import BeliefAgentConfig
from agents.utterance_agents import UtteranceAgentConfig
from training.optimizer import OptimizerConfig


@dataclasses.dataclass
class ExperimentConfig:
    """Full run config."""

    jax_seed: int = 42
    role_assignment: AssignmentConfig = dataclasses.field(default_factory=SimpleAssignmentConfig)
    communication: CommunicationConfig = dataclasses.field(default_factory=CommunicationConfig)
    environment: EnvironmentConfig = dataclasses.field(default_factory=EnvironmentConfig)
    belief_agents: BeliefAgentConfig = dataclasses.field(default_factory=BeliefAgentConfig)
    utterance_agents: UtteranceAgentConfig = dataclasses.field(default_factory=UtteranceAgentConfig)
    # Separate optimizers: belief and utterance agents are independent populations.
    belief_optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    utterance_optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
