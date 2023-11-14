#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from pathlib import Path

from omegaconf import OmegaConf

from confbeam_experiments.setup import setup_rxn, setup_additions
from confbeam_experiments.sub_beams.__main__ import ExperimentName
from confbeam_experiments.sub_beams.beam_search import generate_beam_search_sharded
from confbeam_experiments.sub_beams.conformal import evaluate_sub_beam_coverage


def sub_beam_experiment(experiment: ExperimentName, out_path: Path, config_path: Path, sharding: int = 250):
    """Main experiment script for conformal sub-beam selection"""
    config = OmegaConf.load(config_path)
    if experiment == "rxn":
        manager, test_set = setup_rxn(config)
        max_decode = 170

    elif experiment == "additions":
        manager, test_set = setup_additions(config)
        max_decode = 8
    else:
        raise ValueError("Available experiments: 'rxn' and 'additions'.")

    delta = 0.05
    for beam_size in [5, 10]:
        beam_data = generate_beam_search_sharded(
            manager=manager, data=test_set, beam_size=beam_size, max_decode=max_decode, sharding=sharding
        )
        for alpha in [0.05, 0.01]:
            eval_result = evaluate_sub_beam_coverage(
                beam_data, manager, alpha=alpha, delta=delta, n_rep=1000
            )
            eval_result.to_csv(
                out_path / f"results_{experiment}_bs_{beam_size}_alpha_{alpha}.csv"
            )
