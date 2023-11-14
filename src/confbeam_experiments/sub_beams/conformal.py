#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
"""Evaluation code for sub-beam prediction sets"""
import numpy as np
import pandas as pd

from confbeam_experiments.models import ModelManager
from conformal_beams.sub_beams import SubBeamData, SubBeamConformal
from ..utils import DEFAULT_SEED


def evaluate_sub_beam_coverage(
    data: SubBeamData, man: ModelManager, alpha: float, delta: float, n_rep: int = 1000
):
    eval_results = []
    rng = np.random.default_rng(DEFAULT_SEED)
    for rep in range(n_rep):
        seed = rng.integers(0, 10000)
        cal, test = data.split(random_state=seed)
        conf_predictor = SubBeamConformal(
            alpha=alpha, delta=delta, padding_token_id=man.model.config.pad_token_id
        )
        conf_predictor.calibrate(cal)
        result = conf_predictor.evaluate(test)
        result["datasplit_seed"] = seed
        eval_results.append(result)

    return pd.DataFrame(eval_results)
