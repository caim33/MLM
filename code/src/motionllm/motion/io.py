"""Safe, deterministic NumPy motion-file readers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MotionIOError, MotionValidationError
from .validation import validate_motion_array, validate_normalization_stats


def _path(value: str | os.PathLike[str], *, name: str) -> Path:
    try:
        path = Path(value)
    except TypeError as exc:
        raise MotionIOError(f"{name} must be a filesystem path") from exc
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{name} is not a file: {path}")
    return path


def _load_numpy_array(
    path_like: str | os.PathLike[str],
    *,
    key: str | None,
    name: str,
) -> np.ndarray:
    path = _path(path_like, name=name)
    suffix = path.suffix.lower()
    if suffix not in {".npy", ".npz"}:
        raise MotionIOError(
            f"{name} must use .npy or .npz, got suffix {path.suffix!r}"
        )
    if suffix == ".npy" and key is not None:
        raise MotionIOError(f"{name}: key is only valid for .npz files")

    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise MotionIOError(f"failed to read {name}: {path}") from exc

    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys = tuple(loaded.files)
            if key is None:
                if len(keys) != 1:
                    raise MotionIOError(
                        f"{name} archive must contain exactly one array when no "
                        f"key is given; found {list(keys)!r}"
                    )
                selected = keys[0]
            else:
                if not isinstance(key, str) or not key:
                    raise MotionIOError(f"{name}: npz key must be a non-empty string")
                if key not in keys:
                    raise MotionIOError(
                        f"{name}: npz key {key!r} not found; available keys={list(keys)!r}"
                    )
                selected = key
            # Copy before closing the archive so callers never retain a lazy handle.
            return np.array(loaded[selected], copy=True)
        finally:
            loaded.close()

    if not isinstance(loaded, np.ndarray):
        raise MotionIOError(f"{name} did not decode to a NumPy array")
    return np.array(loaded, copy=True)


def _floating_dtype(dtype: Any) -> np.dtype[Any]:
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise MotionValidationError(f"invalid output dtype: {dtype!r}") from exc
    if resolved.kind != "f":
        raise MotionValidationError(
            f"motion output dtype must be floating point, got {resolved}"
        )
    return resolved


def load_motion_array(
    path: str | os.PathLike[str],
    *,
    npz_key: str | None = None,
    expected_feature_dim: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Load one strict ``(time, features)`` motion array.

    Multi-array ``.npz`` files require an explicit ``npz_key``.  Object,
    complex, boolean, empty, non-finite, or non-2D arrays are rejected.
    """

    raw = _load_numpy_array(path, key=npz_key, name="motion file")
    validate_motion_array(raw, expected_feature_dim=expected_feature_dim)
    result = raw.astype(_floating_dtype(dtype), copy=True)
    # Casting (for example float64 -> float32) can itself overflow.
    validate_motion_array(result, expected_feature_dim=expected_feature_dim)
    return result


def load_normalization_stats(
    mean_path: str | os.PathLike[str],
    std_path: str | os.PathLike[str],
    *,
    mean_npz_key: str | None = None,
    std_npz_key: str | None = None,
    expected_feature_dim: int | None = None,
    minimum_std: float = 0.0,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate feature-wise normalization statistics."""

    mean = _load_numpy_array(mean_path, key=mean_npz_key, name="mean file")
    std = _load_numpy_array(std_path, key=std_npz_key, name="std file")
    validate_normalization_stats(
        mean,
        std,
        expected_feature_dim=expected_feature_dim,
        minimum_std=minimum_std,
    )
    resolved_dtype = _floating_dtype(dtype)
    mean_result = mean.astype(resolved_dtype, copy=True)
    std_result = std.astype(resolved_dtype, copy=True)
    validate_normalization_stats(
        mean_result,
        std_result,
        expected_feature_dim=expected_feature_dim,
        minimum_std=minimum_std,
    )
    return mean_result, std_result
