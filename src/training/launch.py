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
from training.config import ExperimentConfig, WandbConfig
from training.ippo import make_train

_DEFAULT_WANDB_CONFIG = WandbConfig()


def main(
    experiment_config: Annotated[ExperimentConfig, tyro.conf.OmitArgPrefixes],
    wandb_config: Annotated[
        WandbConfig, tyro.conf.arg(name="wandb")
    ] = _DEFAULT_WANDB_CONFIG,
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

    # Execute training code in here
    rng = jax.random.PRNGKey(experiment_config.seed)
    train = make_train(experiment_config)
    results = train(rng)

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
