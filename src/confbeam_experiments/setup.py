#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from typing import Tuple

import numpy as np
from datasets import Dataset
from omegaconf import DictConfig

from confbeam_experiments.models.additions_t5 import (
    AdditionsT5Manager,
)
from confbeam_experiments.models.rxn_t5 import RXNT5Manager
from confbeam_experiments.utils import DEFAULT_SEED, DEVICE


def setup_rxn(config: DictConfig) -> Tuple[RXNT5Manager, Dataset]:
    manager = RXNT5Manager(config=config)
    test_set = manager.dataset["test"].shuffle(
        generator=np.random.default_rng(DEFAULT_SEED)
    )
    manager.model.to(DEVICE)
    # The model is intentionally set in training mode to degrade performance
    manager.model.train()
    return manager, test_set


def setup_additions(config: DictConfig) -> Tuple[AdditionsT5Manager, Dataset]:
    t5man = AdditionsT5Manager(model_spec_or_checkpoint=config.model_spec_or_checkpoint)
    t5man.create_data()
    t5man.model.to(DEVICE)
    # The model is intentionally set in training mode to allow enough errors
    t5man.model.train()

    test_set = t5man.data["test"].shuffle(generator=np.random.default_rng(DEFAULT_SEED))

    return t5man, test_set
