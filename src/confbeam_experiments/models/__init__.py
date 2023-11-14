#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import DatasetDict


@dataclass
class ModelManager:
    """Ensure the right model is used with the right data and the right tokenizer"""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    data: DatasetDict
