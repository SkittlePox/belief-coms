"""Training entry point.

Run with:
    uv run python -m training.launch
"""

from __future__ import annotations
from typing import Annotated
import wandb
import dataclasses
import tyro
import jax
from training.config import ExperimentConfig
from training.ippo import make_train


# WandbConfig lives here, not in config.py: only main() consumes it (wandb.init), so
# it's a launcher concern rather than part of the launch<->ippo shared contract.
@dataclasses.dataclass(frozen=True)
class WandbConfig:
    """wandb config"""

    entity: str = "signification-team"
    project: str = "belief-coms"
    mode: str = "disabled"
    notes: str = ""
    save_code: bool = True


_DEFAULT_WANDB_CONFIG = WandbConfig()


def main(
    # name="" drops only the top-level `experiment-config.` prefix while keeping the
    # meaningful sub-config prefixes (e.g. --belief-agents.trunk-dims vs
    # --utterance-agents.trunk-dims), which disambiguates fields that share a leaf name
    # across sub-configs. (OmitArgPrefixes flattened everything and collided on trunk_dims.)
    experiment_config: Annotated[ExperimentConfig, tyro.conf.arg(name="")],
    wandb_config: Annotated[WandbConfig, tyro.conf.arg(name="wandb")] = _DEFAULT_WANDB_CONFIG,
):
    run = wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        config=dataclasses.asdict(experiment_config),
        mode=wandb_config.mode,
        notes=wandb_config.notes,
        save_code=wandb_config.save_code,
    )
    print(experiment_config, wandb_config)

    # Build the training function from the config, then run it under the seeded RNG.
    rng = jax.random.key(experiment_config.jax_seed)
    train = make_train(experiment_config)
    train(rng)

    run.finish()

    # Perhaps do some saving of agents in here.

    print("Done")


if __name__ == "__main__":
    tyro.cli(
        main,
        config=(
            tyro.conf.UsePythonSyntaxForLiteralCollections,
            tyro.conf.FlagConversionOff,
        ),
    )
