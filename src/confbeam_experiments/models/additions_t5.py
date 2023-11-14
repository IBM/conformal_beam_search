#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from datasets import DatasetDict

from . import ModelManager
from confbeam_experiments.data.math import split_dataset, tokenize_datasets


@dataclass(kw_only=True)
class AdditionsT5Manager(ModelManager):
    model: PreTrainedModel = field(init=False, repr=False)
    tokenizer: PreTrainedTokenizerBase = field(init=False, repr=False)
    data: DatasetDict = field(init=False, repr=False)
    model_spec_or_checkpoint: str

    def __post_init__(self) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_spec_or_checkpoint
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_spec_or_checkpoint)
        self.data = self.create_data().rename_columns(
            {"question": "source", "answer": "target"}
        )

    def create_data(self, tokenize=False) -> DatasetDict:
        dataset_dict = split_dataset()
        if tokenize:
            dataset_dict = tokenize_datasets(dataset_dict, self.tokenizer)

        return dataset_dict
