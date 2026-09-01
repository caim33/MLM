"""Framework-free generation phase classification."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

from .errors import MotionInjectionError


def _first_cache_position(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MotionInjectionError("cache_position must contain integers, not bool")
    try:
        return operator.index(value)
    except TypeError:
        pass

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not value:
            raise MotionInjectionError("cache_position cannot be empty")
        return _first_cache_position(value[0])

    # Torch/NumPy scalars and one-dimensional tensors expose numel/item without
    # requiring either framework at this boundary.
    numel = getattr(value, "numel", None)
    if callable(numel):
        size = int(numel())
        if size == 0:
            raise MotionInjectionError("cache_position cannot be empty")
        flattened = getattr(value, "reshape", lambda *_: value)(-1)
        first = flattened[0] if size > 1 else value
        item = getattr(first, "item", None)
        if callable(item):
            return _first_cache_position(item())
    item = getattr(value, "item", None)
    if callable(item):
        return _first_cache_position(item())
    raise MotionInjectionError("cache_position must be an integer or integer sequence")


def is_generation_prefill(
    *, cache_position: Any = None, past_key_values: Any = None
) -> bool:
    """Return true exactly for the prompt/prefill generation phase.

    ``cache_position`` is authoritative when supplied.  For older transformers
    releases that omit it, the absence of ``past_key_values`` identifies
    prefill.  Negative positions are always invalid.
    """

    position = _first_cache_position(cache_position)
    if position is not None:
        if position < 0:
            raise MotionInjectionError("cache_position must be non-negative")
        return position == 0
    return past_key_values is None


def prefill_motion_payload(
    motion: Any,
    motion_lengths: Any,
    *,
    cache_position: Any = None,
    past_key_values: Any = None,
) -> tuple[Any, Any]:
    """Keep motion only during prefill; return explicit nulls during decode."""

    if is_generation_prefill(
        cache_position=cache_position, past_key_values=past_key_values
    ):
        return motion, motion_lengths
    return None, None

