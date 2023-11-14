#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
"""Experiment code for dynamic conformal beams"""
from typing import Iterable, Tuple, List
from uuid import uuid4

import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import DataCollatorForSeq2Seq

from confbeam_experiments.models import ModelManager
from confbeam_experiments.utils import to
from conformal_beams.dynamic_beams import DynamicBeamConformal, DynamicBeamEvaluation


def setup_pbar(iterable: Iterable, use_pbar: bool, **tqdm_kwargs) -> Iterable:
    if use_pbar:
        return tqdm(iterable, **tqdm_kwargs)
    else:
        return iterable


def prepare_data(dataset: Dataset, manager: ModelManager):
    """Pre-tokenize the whole dataset"""

    def tokenize(data):
        return manager.tokenizer(data["source"], text_target=data["target"])

    return dataset.map(tokenize, batched=True)


def calibrate(
    calibration_dataset: Dataset,
    conformal_predictor: DynamicBeamConformal,
    manager: ModelManager,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """High level calibration function

    Args:
        calibration_dataset: Dataset
            HF datasets object with keys 'input_ids', 'attention_mask', 'labels'
        conformal_predictor: DynamicBeamConformal
        manager: Modelmanager
        batch_size: int

    Returns: (np.ndarray, np.ndarray)
        step-by-step confidence score thresholds ($t_\alpha^{(l)}$ in sec. 4.1),
        step-by-step removed calibration samples ($k_\alpha^{(l)}$ in sec. 4.1)
    """
    collator = DataCollatorForSeq2Seq(
        tokenizer=manager.tokenizer, model=manager.model, padding=True
    )
    calibration_dataset = calibration_dataset.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )
    calibration_dataset = (
        calibration_dataset.map(lambda r: dict(**r, input_len=len(r["input_ids"])))
        .sort("input_len")
        .remove_columns("input_len")
    )

    dl = DataLoader(
        calibration_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    calibration_scores = conformal_predictor.score_dataloader(dl)
    calibration_outputs = conformal_predictor.calibrate(calibration_scores)
    return calibration_outputs


def evaluate(
    test_dataset: Dataset,
    conformal_predictor: DynamicBeamConformal,
    progress_bar: bool = True,
) -> List[DynamicBeamEvaluation]:
    """Predict dynamic beams on a pre-calibrated conformal predictor and evaluate coverage and beam sizes"""
    all_results = []

    for idx, sample in setup_pbar(
        enumerate(test_dataset), use_pbar=progress_bar, total=len(test_dataset)
    ):
        evaluation_result: DynamicBeamEvaluation = to(
            conformal_predictor.evaluate_sample(
                input_seq=np.array(sample["input_ids"]),
                input_msk=np.array(sample["attention_mask"]),
                label=np.array(sample["labels"]),
            ),
            "cpu",
        )
        evaluation_result["sample_idx"] = idx
        all_results.append(evaluation_result)

    return all_results


def dynamic_beam_experiment(
    tokenized_dataset: Dataset,
    manager: ModelManager,
    rng: np.random.Generator,
    cal_size: int,
    test_size: int,
    alpha: float,
    max_len: int,
    progress_bar: bool = True,
):
    """End-to-end single dynamic beam experiment:
    - split data between calibration and test
    - calibrate
    - evaluate

    Returns: dict
        keys: calibration thresholds (t) and rejected sample sizes (k), risk level alpha,
        evaluation results (itself a dict, the output of DynamicBeamConformal.evaluate)
    """
    ds = tokenized_dataset.train_test_split(
        test_size=test_size, train_size=cal_size, shuffle=True, generator=rng
    )
    cal_set = ds["train"]
    test_set = ds["test"]

    conformal_predictor = DynamicBeamConformal(
        manager=manager, alpha=alpha, max_len=max_len
    )

    thresholds, calibration_ks = calibrate(
        calibration_dataset=cal_set,
        conformal_predictor=conformal_predictor,
        manager=manager,
    )

    evaluation_results = evaluate(
        test_dataset=test_set,
        conformal_predictor=conformal_predictor,
        progress_bar=progress_bar,
    )

    return {
        "thresholds": thresholds,
        "calibration_ks": calibration_ks,
        "evaluation_results": evaluation_results,
        "alpha": alpha,
    }


def explode_result_steps(evaluation_result):
    n_steps = len(evaluation_result["ranks"])
    per_step_rows = []
    for s in range(n_steps):
        data_row = dict(
            oracle_size=evaluation_result["ranks"][s] + 1,
            beam_size=evaluation_result["beam_sizes"][s],
            in_beam=evaluation_result["in_beam"][s],
            step=s,
            L=(evaluation_result["true_seq"] != -100).sum().item(),
            sample_idx=evaluation_result["sample_idx"],
        )
        per_step_rows.append(data_row)

    return per_step_rows


def format_experiment_result(experiment_results) -> pd.DataFrame:
    """Convert dictionary of results into a DataFrame"""
    all_rows = []
    for sample in experiment_results["evaluation_results"]:
        all_rows.extend(explode_result_steps(sample))
    exploded_df = pd.DataFrame(all_rows)
    exploded_df.attrs["rep"] = experiment_results.get("rep_id", None)
    exploded_df.attrs["alpha"] = experiment_results["alpha"]
    exploded_df.attrs["ks"] = experiment_results["calibration_ks"]
    exploded_df.attrs["thresholds"] = experiment_results["thresholds"]
    return exploded_df


def dynamic_beam_repeated_experiments(
    data: Dataset,
    manager: ModelManager,
    alpha: float,
    max_len: int,
    cal_size: int,
    test_size: int,
    n_repetitions: int,
    seed=None,
    progress_bar: bool = True,
):
    """Complete MC evaluation over repetitions of conformal dynamic beams over a given dataset with a fixed risk"""
    tokenized_dataset = prepare_data(dataset=data, manager=manager)
    tokenized_dataset = tokenized_dataset.filter(
        lambda row: len(row["labels"]) <= max_len
    )
    rng = np.random.default_rng(seed=seed)
    experiment_results = []
    for rep in range(n_repetitions):
        individual_result = dynamic_beam_experiment(
            tokenized_dataset,
            manager,
            rng,
            cal_size,
            test_size,
            alpha,
            max_len,
            progress_bar,
        )
        individual_result["rep_id"] = str(uuid4())
        experiment_results.append(format_experiment_result(individual_result))

    return experiment_results
