#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
import datetime
import os
from pathlib import Path


def get_models_root():
    root_path = os.environ.get(
        "CONFBEAM_MODELS_STORE", Path.cwd() / "confbeam_models_assets"
    )
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def get_now_tag() -> str:
    now_tag = datetime.now().strftime("%Y-%m-%d_%H%S.%f")[:-3]
    return now_tag
