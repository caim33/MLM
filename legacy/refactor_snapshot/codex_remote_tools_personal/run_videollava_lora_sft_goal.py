#!/usr/bin/env python3
"""Launch Video-LLaVA LoRA SFT with the isolated legacy dependency stack."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Video-LLaVA"
LEGACY = ROOT / "codex_envs" / "legacy_torch211_cu128"
LEGACY_TF = ROOT / "codex_envs" / "legacy_tf431"
VIDEO_EXTRA = ROOT / "codex_envs" / "video_extra"

for path in reversed([SRC, LEGACY, LEGACY_TF, VIDEO_EXTRA]):
    sys.path.insert(0, str(path))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import huggingface_hub.utils as hub_utils  # noqa: E402

hub_utils.insecure_hashlib = hashlib

import videollava.train.train as videollava_train  # noqa: E402


def _safe_maybe_zero_3(param, ignore_status=False, name=None):
    """Video-LLaVA imports DeepSpeed even when ZeRO is not active; avoid that."""
    if hasattr(param, "ds_id"):
        try:
            from deepspeed import zero
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
                import logging

                logging.warning("%s: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: %s", name, param.ds_status)
            with zero.GatheredParameters([param]):
                return param.data.detach().cpu().clone()
        except Exception:
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


videollava_train.maybe_zero_3 = _safe_maybe_zero_3


if __name__ == "__main__":
    videollava_train.train()
