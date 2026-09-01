"""Explicit contracts for temporal truncation, padding, and downsampling."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .errors import TemporalContractError
from .validation import validate_motion_array


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TemporalContractError(f"{name} must be a positive integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TemporalContractError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise TemporalContractError(f"{name} must be > 0, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class TemporalLengthContract:
    """Immutable accounting for one motion encoder input.

    Truncation happens first, then the retained motion is padded to a multiple
    of ``downsample_factor``.  Therefore ``encoded_length`` is exact rather
    than an estimate.
    """

    raw_length: int
    retained_length: int
    padded_length: int
    downsample_factor: int
    encoded_length: int

    def __post_init__(self) -> None:
        raw = _positive_int(self.raw_length, name="raw_length")
        retained = _positive_int(self.retained_length, name="retained_length")
        padded = _positive_int(self.padded_length, name="padded_length")
        factor = _positive_int(
            self.downsample_factor, name="downsample_factor"
        )
        encoded = _positive_int(self.encoded_length, name="encoded_length")
        if retained > raw:
            raise TemporalContractError("retained_length cannot exceed raw_length")
        if padded < retained:
            raise TemporalContractError("padded_length cannot be below retained_length")
        if padded % factor:
            raise TemporalContractError(
                "padded_length must be divisible by downsample_factor"
            )
        if encoded != padded // factor:
            raise TemporalContractError(
                "encoded_length must equal padded_length // downsample_factor"
            )

    @property
    def padding_length(self) -> int:
        return self.padded_length - self.retained_length

    @property
    def truncated_length(self) -> int:
        return self.raw_length - self.retained_length

    @property
    def placeholder_count(self) -> int:
        """Number of motion placeholders required by the legacy protocol."""

        return self.encoded_length


def plan_temporal_length(
    raw_length: int,
    *,
    downsample_factor: int = 4,
    max_encoded_steps: int | None = None,
) -> TemporalLengthContract:
    """Build an exact length contract without touching array data.

    ``max_encoded_steps`` is a post-downsampling cap.  Its corresponding raw
    frame limit is consequently aligned to ``downsample_factor`` and cannot
    create a partially encoded step.
    """

    raw = _positive_int(raw_length, name="raw_length")
    factor = _positive_int(downsample_factor, name="downsample_factor")
    retained = raw
    if max_encoded_steps is not None:
        maximum = _positive_int(max_encoded_steps, name="max_encoded_steps")
        retained = min(retained, maximum * factor)
    padded = retained + (-retained % factor)
    return TemporalLengthContract(
        raw_length=raw,
        retained_length=retained,
        padded_length=padded,
        downsample_factor=factor,
        encoded_length=padded // factor,
    )


def apply_temporal_contract(
    motion: Any,
    contract: TemporalLengthContract,
    *,
    pad_mode: Literal["edge", "zero"] = "edge",
) -> np.ndarray:
    """Truncate and pad a motion according to a verified contract."""

    motion_array = validate_motion_array(motion)
    if not isinstance(contract, TemporalLengthContract):
        raise TemporalContractError("contract must be TemporalLengthContract")
    if motion_array.shape[0] != contract.raw_length:
        raise TemporalContractError(
            f"motion time length {motion_array.shape[0]} does not match "
            f"contract raw_length {contract.raw_length}"
        )
    if pad_mode not in {"edge", "zero"}:
        raise TemporalContractError(
            f"pad_mode must be 'edge' or 'zero', got {pad_mode!r}"
        )

    retained = np.array(motion_array[: contract.retained_length], copy=True)
    if contract.padding_length == 0:
        return retained
    if pad_mode == "edge":
        padding = np.repeat(retained[-1:, :], contract.padding_length, axis=0)
    else:
        padding = np.zeros(
            (contract.padding_length, retained.shape[1]), dtype=retained.dtype
        )
    result = np.concatenate((retained, padding), axis=0)
    if result.shape[0] != contract.padded_length:  # defensive invariant
        raise TemporalContractError("temporal padding produced an invalid length")
    return result


def prepare_motion_temporal(
    motion: Any,
    *,
    downsample_factor: int = 4,
    max_encoded_steps: int | None = None,
    pad_mode: Literal["edge", "zero"] = "edge",
) -> tuple[np.ndarray, TemporalLengthContract]:
    """Validate, plan, truncate, and pad a motion in one pure operation."""

    motion_array = validate_motion_array(motion)
    contract = plan_temporal_length(
        motion_array.shape[0],
        downsample_factor=downsample_factor,
        max_encoded_steps=max_encoded_steps,
    )
    return apply_temporal_contract(motion_array, contract, pad_mode=pad_mode), contract


def downsample_motion(
    padded_motion: Any,
    contract: TemporalLengthContract,
    *,
    reduction: Literal["mean", "first", "last"] = "mean",
) -> np.ndarray:
    """Reference NumPy downsampling that obeys a length contract.

    Neural encoders will replace the reduction itself, but must preserve this
    exact input/output length relationship.
    """

    motion_array = validate_motion_array(padded_motion)
    if not isinstance(contract, TemporalLengthContract):
        raise TemporalContractError("contract must be TemporalLengthContract")
    if motion_array.shape[0] != contract.padded_length:
        raise TemporalContractError(
            f"padded motion length must be {contract.padded_length}; "
            f"got {motion_array.shape[0]}"
        )
    if reduction not in {"mean", "first", "last"}:
        raise TemporalContractError(
            f"unsupported reduction {reduction!r}; expected mean/first/last"
        )

    grouped = motion_array.reshape(
        contract.encoded_length,
        contract.downsample_factor,
        motion_array.shape[1],
    )
    if reduction == "mean":
        result = grouped.mean(axis=1)
    elif reduction == "first":
        result = grouped[:, 0, :]
    else:
        result = grouped[:, -1, :]
    result = np.array(result, copy=True)
    if result.shape[0] != contract.encoded_length:  # defensive invariant
        raise TemporalContractError("downsampling violated encoded_length")
    return result
