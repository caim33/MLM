"""Strict validation for motion arrays and normalization statistics.

The core convention is ``(time, features)``.  Singleton dimensions are not
silently squeezed: accepting a different layout would make sample provenance
depend on an implicit heuristic.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .errors import MotionValidationError


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MotionValidationError(f"{name} must be a positive integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise MotionValidationError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise MotionValidationError(f"{name} must be > 0, got {result}")
    return result


def _real_numeric_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise MotionValidationError(f"{name} cannot be converted to an array") from exc

    if array.dtype.kind not in "iuf":
        raise MotionValidationError(
            f"{name} must have a real numeric dtype, got {array.dtype}"
        )
    return array


def _require_finite(array: np.ndarray, *, name: str) -> None:
    if not bool(np.isfinite(array).all()):
        non_finite = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise MotionValidationError(
            f"{name} contains {non_finite} non-finite value(s)"
        )


def validate_motion_array(
    motion: Any,
    *,
    expected_feature_dim: int | None = None,
    name: str = "motion",
) -> np.ndarray:
    """Validate and return a motion array without changing its dtype or shape.

    A valid motion is a non-empty, finite, real numeric two-dimensional array
    laid out as ``(time, features)``.  A one-frame motion is valid; an empty
    time or feature axis is not.
    """

    array = _real_numeric_array(motion, name=name)
    if array.ndim != 2:
        raise MotionValidationError(
            f"{name} must have shape (time, features); got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise MotionValidationError(f"{name} must be non-empty; got {array.shape}")

    if expected_feature_dim is not None:
        feature_dim = _positive_int(
            expected_feature_dim, name="expected_feature_dim"
        )
        if array.shape[1] != feature_dim:
            raise MotionValidationError(
                f"{name} feature dimension must be {feature_dim}; "
                f"got {array.shape[1]}"
            )

    _require_finite(array, name=name)
    return array


def validate_normalization_stats(
    mean: Any,
    std: Any,
    *,
    expected_feature_dim: int | None = None,
    minimum_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate feature-wise mean/std arrays.

    ``std`` must be strictly greater than ``minimum_std`` for every feature.
    This deliberately rejects zero variance rather than silently clamping it.
    """

    mean_array = _real_numeric_array(mean, name="mean")
    std_array = _real_numeric_array(std, name="std")

    if mean_array.ndim != 1 or std_array.ndim != 1:
        raise MotionValidationError(
            "mean and std must both have shape (features,); "
            f"got {mean_array.shape} and {std_array.shape}"
        )
    if mean_array.size == 0 or std_array.size == 0:
        raise MotionValidationError("mean and std must be non-empty")
    if mean_array.shape != std_array.shape:
        raise MotionValidationError(
            f"mean/std shapes differ: {mean_array.shape} != {std_array.shape}"
        )

    try:
        threshold = float(minimum_std)
    except (TypeError, ValueError) as exc:
        raise MotionValidationError("minimum_std must be a finite number") from exc
    if not np.isfinite(threshold) or threshold < 0:
        raise MotionValidationError(
            f"minimum_std must be finite and >= 0, got {minimum_std!r}"
        )

    if expected_feature_dim is not None:
        feature_dim = _positive_int(
            expected_feature_dim, name="expected_feature_dim"
        )
        if mean_array.size != feature_dim:
            raise MotionValidationError(
                f"normalization feature dimension must be {feature_dim}; "
                f"got {mean_array.size}"
            )

    _require_finite(mean_array, name="mean")
    _require_finite(std_array, name="std")
    invalid = np.flatnonzero(std_array <= threshold)
    if invalid.size:
        preview = ", ".join(str(int(i)) for i in invalid[:8])
        suffix = "..." if invalid.size > 8 else ""
        raise MotionValidationError(
            f"std must be > {threshold}; invalid feature index(es): "
            f"{preview}{suffix}"
        )

    return mean_array, std_array
