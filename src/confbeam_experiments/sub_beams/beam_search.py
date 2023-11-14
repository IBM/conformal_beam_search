#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
"""Tools to calibrate the Clopper-Pearson Bound on Beam Coverage"""
import torch
from datasets import Dataset
from tqdm.autonotebook import tqdm

from confbeam_experiments.models import ModelManager
from confbeam_experiments.utils import to, DEVICE
from conformal_beams.sub_beams import SubBeamData, compute_ranks


def generate_beam_search_sharded(
    manager: ModelManager,
    data: Dataset,
    beam_size: int,
    max_decode: int,
    sharding: int = 250,
):
    """Batched beam search and conformal sub-beam prediction"""
    beams = []
    for i in tqdm(range(sharding)):
        shard_data = data.shard(sharding, i)
        beam_data = generate_beams(manager, shard_data, beam_size, max_decode)
        beams.append(beam_data)
    return SubBeamData.concatenate(
        *beams, padding_token=manager.model.config.pad_token_id
    )


def generate_beams(
    manager: ModelManager, data: Dataset, beam_size: int, max_decode: int
) -> SubBeamData:
    """Beam search + conformal sub-beam prediction on a batch fitting in memory"""
    N = len(data)
    tokenized_data = manager.tokenizer(
        data["source"], text_target=data["target"], padding=True, return_tensors="pt"
    )
    tokenized_labels = tokenized_data["labels"][:, :max_decode]
    try:
        del tokenized_data["labels"]
        del tokenized_data["token_type_ids"]
    except KeyError:
        pass
    tokenized_data = tokenized_data.to(DEVICE)

    generated_preds = manager.model.generate(
        **tokenized_data,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        do_sample=False,
        max_new_tokens=max_decode,
        return_dict_in_generate=True,
        output_scores=True,
        length_penalty=1,
    )
    generated_preds = to(generated_preds, "cpu")
    del tokenized_data
    torch.cuda.empty_cache()

    # Remove the bos token used by seq2seq
    generated_preds["sequences"] = generated_preds["sequences"][:, 1:]

    beam_data = SubBeamData(
        sequences=generated_preds["sequences"].reshape(N, beam_size, -1).numpy(),
        scores=generated_preds["sequences_scores"].reshape(N, beam_size).numpy(),
        labels=tokenized_labels.numpy(),
    )
    beam_data.pad_to_match(padding_token_id=manager.model.config.pad_token_id)
    beam_data = compute_ranks(beam_data, beam_size)

    return beam_data
