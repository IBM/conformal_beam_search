#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from typing import Callable

import torch


def to(
    obj: torch.Tensor | dict | list | tuple, device: str
) -> torch.Tensor | dict | list | tuple:
    if hasattr(obj, "to") and isinstance(obj.to, Callable):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 3247
