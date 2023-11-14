#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
import numpy as np
import torch


def pad_to_match(tokens1: np.ndarray, tokens2: np.ndarray, padding_token_id: int):
    l1 = tokens1.shape[-1]
    l2 = tokens2.shape[-1]
    if l1 == l2:
        return tokens1, tokens2
    elif l1 < l2:
        tokens2, tokens1 = pad_to_match(tokens2, tokens1, padding_token_id)
    else:  # l1 > l2
        delta = l1 - l2
        pad_blob = np.full(
            shape=(*(tokens2.shape[:-1]), delta),
            fill_value=padding_token_id,
            dtype=tokens2.dtype,
        )
        tokens2 = np.concatenate([tokens2, pad_blob], axis=-1)

    return tokens1, tokens2


def torch2numpy(tree_object):
    """Convert tensor leaves to numpy leaves in nested containers with native python types"""
    if isinstance(tree_object, torch.Tensor):
        return tree_object.detach().cpu().numpy()
    if isinstance(tree_object, dict):
        return {k: torch2numpy(v) for k, v in tree_object.items()}
    if isinstance(tree_object, (list, tuple)):
        return [torch2numpy(el) for el in tree_object]
    else:
        return tree_object
