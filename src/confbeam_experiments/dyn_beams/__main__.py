#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from pathlib import Path
from typing import Optional, Sequence, TypeAlias, Literal

import click

ExperimentName: TypeAlias = Literal["rxn", "additions"]


@click.group
def dyn_beam_cli():
    pass


@dyn_beam_cli.command()
@click.option("--experiment", "-x", type=click.Choice(["rxn", "additions"]), required=True)
@click.option("--out_path", "-o", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--n_rep", "-n", type=int, required=False)
@click.option("--alphas", "-a", type=float, required=False, multiple=True)
@click.option("--seed", "-s", type=int, required=False)
@click.option("--pb/--no-pb", default=True)
def predict(
    experiment: ExperimentName,
    out_path: Path,
    config: Path,
    n_rep: Optional[int] = None,
    alphas: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    pb: bool = True,
):
    """Experiment entry point
    Usage: python -m confbeam_experiment.dyn_beams predict -x additions -o <output_path> -c config.yaml
    """
    # Importing in command to avoid long startup due to imports
    from confbeam_experiments.dyn_beams.cli import run_dynamic_beam_repetitions

    run_dynamic_beam_repetitions(
        experiment=experiment,
        out_path=out_path,
        config_path=config,
        n_rep=n_rep,
        alphas=alphas,
        seed=seed,
        progress_bar=pb,
    )


@dyn_beam_cli.command()
@click.option("--exp_dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--out_dir",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
def analyze(exp_dir: Path, out_dir: Optional[Path] = None):
    """Aggregation entry point
    Usage: python -m confbeam_experiment.dyn_beams analze --exp_dir <experiment output path>
    """
    # Importing in command to avoid long startup due to imports
    from confbeam_experiments.dyn_beams.analysis import collect_result_from_dir
    from confbeam_experiments.dyn_beams.cli import dyn_beam_aggregate_results

    all_results = collect_result_from_dir(exp_dir)
    if out_dir is None:
        out_dir = exp_dir
    dyn_beam_aggregate_results(experiment_results=all_results, out_dir=out_dir)


if __name__ == "__main__":
    dyn_beam_cli()
