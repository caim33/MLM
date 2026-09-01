"""Small runtime guards that are safe before torch.distributed initialization."""

from __future__ import annotations

import importlib
from typing import Any


def _torch_or_none(torch_module: Any | None) -> Any | None:
    if torch_module is not None:
        return torch_module
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def distributed_rank(torch_module: Any | None = None) -> int:
    torch = _torch_or_none(torch_module)
    distributed = getattr(torch, "distributed", None) if torch is not None else None
    if distributed is None:
        return 0
    try:
        if not distributed.is_available() or not distributed.is_initialized():
            return 0
        rank = int(distributed.get_rank())
    except Exception:
        return 0
    return rank if rank >= 0 else 0


def is_primary_process(torch_module: Any | None = None) -> bool:
    return distributed_rank(torch_module) == 0
