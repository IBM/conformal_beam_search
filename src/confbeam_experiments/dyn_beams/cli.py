#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
import pickle
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

import pandas as pd
from omegaconf import OmegaConf
from tqdm.notebook import tqdm

from confbeam_experiments.dyn_beams.__main__ import ExperimentName
from confbeam_experiments.dyn_beams.analysis import (
    aggregate_global_results,
    format_aggregated_results_latex,
)
from confbeam_experiments.dyn_beams.experiment import dynamic_beam_repeated_experiments
from confbeam_experiments.dyn_beams.utils import DEFAULT_SEED
from confbeam_experiments.setup import setup_rxn, setup_additions


def run_dynamic_beam_repetitions(
    experiment: ExperimentName,
    out_path: Path,
    config_path: Path,
    n_rep: Optional[int] = None,
    alphas: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    progress_bar: bool = False,
):
    """Main experiment script for dynamic beams"""
    config = OmegaConf.load(config_path)
    if seed is None:
        seed = DEFAULT_SEED
    if experiment == "rxn":
        manager, test_set = setup_rxn(config)
        max_len = 50
        if alphas is None:
            alphas = [0.005, 0.01]
        if n_rep is None:
            n_rep = 50

    elif experiment == "additions":
        manager, test_set = setup_additions(config=config)
        max_len = 5
        if alphas is None:
            alphas = [0.01, 0.02, 0.05]
        if n_rep is None:
            n_rep = 120
    else:
        raise ValueError("Available experiments: 'rxn' and 'additions'.")

    for alpha in tqdm(alphas, leave=True):
        experiment_results = dynamic_beam_repeated_experiments(
            data=test_set,
            manager=manager,
            alpha=alpha,
            max_len=max_len,
            cal_size=len(test_set) // 3,
            test_size=5000,
            n_repetitions=n_rep,
            seed=seed,
            progress_bar=progress_bar,
        )
        with open(out_path / f"dynbeam_exp_result-{uuid4()}.pkl", "wb") as pf:
            pickle.dump(experiment_results, pf)


def dyn_beam_aggregate_results(experiment_results: pd.DataFrame, out_dir: Path):
    """Result aggregation to reproduce table 2"""
    agg_results = aggregate_global_results(experiment_results)
    agg_results.to_csv(out_dir / "aggregated_metrics.csv")
    format_aggregated_results_latex(
        agg_results, latex_out_buffer=out_dir / "aggregated_metrics_table.tex"
    )
