#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from pathlib import Path
from typing import TypeAlias, Literal

import click

ExperimentName: TypeAlias = Literal["rxn", "additions"]


@click.command()
@click.option(
    "--experiment", "-x", type=click.Choice(["rxn", "additions"]), required=True
)
@click.option(
    "--out_path", "-o", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--config", "-c", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--sharding", type=click.INT, required=False, default=250)
def sub_beam_cli(
    experiment: ExperimentName, out_path: Path, config: Path, sharding: int = 250
):
    """Sub-beam experiment entry point"""
    # Import in command to speed up startup (for --help)
    from confbeam_experiments.sub_beams.cli import sub_beam_experiment

    sub_beam_experiment(experiment, out_path, config_path=config, sharding=sharding)


if __name__ == "__main__":
    sub_beam_cli()
