"""Pure motion normalization operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MotionValidationError
from .validation import validate_motion_array, validate_normalization_stats


def normalize_motion(
    motion: Any,
    mean: Any,
    std: Any,
    *,
    minimum_std: float = 0.0,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Return ``(motion - mean) / std`` without mutating any input."""

    motion_array = validate_motion_array(motion)
    mean_array, std_array = validate_normalization_stats(
        mean,
        std,
        expected_feature_dim=motion_array.shape[1],
        minimum_std=minimum_std,
    )
    try:
        output_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise MotionValidationError(f"invalid output dtype: {dtype!r}") from exc
    if output_dtype.kind != "f":
        raise MotionValidationError(
            f"normalization output dtype must be floating point, got {output_dtype}"
        )

    # Perform arithmetic in at least float32 to avoid integer arithmetic and
    # low-precision intermediate overflow.
    compute_dtype = np.result_type(
        motion_array.dtype, mean_array.dtype, std_array.dtype, np.float32
    )
    result = (
        motion_array.astype(compute_dtype, copy=False)
        - mean_array.astype(compute_dtype, copy=False)
    ) / std_array.astype(compute_dtype, copy=False)
    result = result.astype(output_dtype, copy=True)
    validate_motion_array(result, expected_feature_dim=motion_array.shape[1])
    return result


def denormalize_motion(
    motion: Any,
    mean: Any,
    std: Any,
    *,
    minimum_std: float = 0.0,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Invert feature-wise normalization, primarily for round-trip tests."""

    motion_array = validate_motion_array(motion)
    mean_array, std_array = validate_normalization_stats(
        mean,
        std,
        expected_feature_dim=motion_array.shape[1],
        minimum_std=minimum_std,
    )
    try:
        output_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise MotionValidationError(f"invalid output dtype: {dtype!r}") from exc
    if output_dtype.kind != "f":
        raise MotionValidationError(
            f"denormalization output dtype must be floating point, got {output_dtype}"
        )

    compute_dtype = np.result_type(
        motion_array.dtype, mean_array.dtype, std_array.dtype, np.float32
    )
    result = (
        motion_array.astype(compute_dtype, copy=False)
        * std_array.astype(compute_dtype, copy=False)
        + mean_array.astype(compute_dtype, copy=False)
    ).astype(output_dtype, copy=True)
    validate_motion_array(result, expected_feature_dim=motion_array.shape[1])
    return result
