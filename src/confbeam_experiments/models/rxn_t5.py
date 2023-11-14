from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from datasets import Dataset, DatasetDict
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from . import ModelManager


@dataclass
class DataCollatorForRXN:
    tokenizer: PreTrainedTokenizerBase
    padding: str = "longest"
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None) -> dict:
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch = BatchEncoding(
            {k: [batch[i][k] for i in range(len(batch))] for k, v in batch[0].items()}
        )

        tokenized_input = self.tokenizer(
            text=batch["source"],
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=return_tensors,
            truncation=True,
        )

        tokenized_label = self.tokenizer(
            text=batch["target"],
            max_length=self.max_target_length,
            padding=self.padding,
            return_tensors=return_tensors,
            truncation=True,
        )

        return {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "labels": tokenized_label["input_ids"],
        }


@dataclass(kw_only=True)
class RXNT5Manager(ModelManager):
    config: DictConfig
    model: PreTrainedModel = field(init=False)
    tokenizer: PreTrainedTokenizerBase = field(init=False)
    data: DatasetDict = field(init=False)
    loader_dict: dict = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer, self.model = self.load_tokenizer_model(**self.config)
        self.dataset = self.load_data(self.config.data_dir_path)
        self.loader_dict = self.get_dataloaders(self.dataset, tokenizer=self.tokenizer)

    def load_tokenizer_model(
        self, tokenizer_path: str, model_path: str, checkpoint_path: str, **kwargs
    ) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenizer.model_max_length = int(1e9)

        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_config(config)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

        return tokenizer, model

    def get_dataloaders(
        self,
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 10,
        **kwargs,
    ) -> Dict[str, DataLoader]:
        dataloaders = {}

        for split in ["train", "test", "val"]:
            dataloaders[split] = self.get_dataloader(
                dataset[split], tokenizer, batch_size
            )

        return dataloaders

    def get_dataloader(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase, batch_size: int = 10
    ):
        data_collator = self.get_data_collator(tokenizer=tokenizer)

        loader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        return loader

    def load_data(self, data_dir_path: str) -> DatasetDict:
        data_path = Path(data_dir_path)

        dataset_train = Dataset.from_json(str(data_path.joinpath("train.json")))
        dataset_test = Dataset.from_json(str(data_path.joinpath("test.json")))
        dataset_val = Dataset.from_json(str(data_path.joinpath("val.json")))

        dataset_splits = DatasetDict(
            {"train": dataset_train, "test": dataset_test, "val": dataset_val}
        )

        return dataset_splits

    def get_data_collator(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> DataCollatorForRXN:
        data_collator = DataCollatorForRXN(
            tokenizer=tokenizer,
            padding="longest",
            max_source_length=270,
            max_target_length=200,
            label_pad_token_id=-100,
        )
        return data_collator
