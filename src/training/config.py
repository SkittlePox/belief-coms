"""Training configuration.

Shared data contract for the launcher (``launch.py``) and the algorithm
(``ippo.py``). This module imports no other training code so it can be
depended on freely without circular imports.
"""

from __future__ import annotations
import dataclasses


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    """wandb config"""

    entity: str = "signification-team"
    project: str = "belief-coms"
    mode: str = "disabled"
    notes: str = ""
    save_code: bool = True


@dataclasses.dataclass
class ExperimentConfig:
    """Full run config."""
    seed: int = 42
    learning_rate: float = 1e-3
