from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motionllm.motion import (
    MotionIOError,
    MotionValidationError,
    denormalize_motion,
    load_motion_array,
    load_normalization_stats,
    normalize_motion,
    validate_motion_array,
    validate_normalization_stats,
)


def test_load_npy_returns_validated_float_copy(tmp_path: Path) -> None:
    path = tmp_path / "motion.npy"
    original = np.arange(12, dtype=np.float64).reshape(4, 3)
    np.save(path, original)

    loaded = load_motion_array(path, expected_feature_dim=3)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, original)
    loaded[0, 0] = 999
    assert original[0, 0] == 0


def test_load_single_array_npz_without_key(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(path, motion=np.ones((2, 5), dtype=np.float32))

    loaded = load_motion_array(path, expected_feature_dim=5)

    assert loaded.shape == (2, 5)


def test_multi_array_npz_requires_explicit_key(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(path, first=np.ones((2, 2)), second=np.zeros((3, 2)))

    with pytest.raises(MotionIOError, match="exactly one array"):
        load_motion_array(path)

    selected = load_motion_array(path, npz_key="second")
    assert selected.shape == (3, 2)


def test_npz_missing_key_is_not_silently_substituted(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(path, actual=np.ones((2, 2)))

    with pytest.raises(MotionIOError, match="not found"):
        load_motion_array(path, npz_key="requested")


@pytest.mark.parametrize(
    "array",
    [
        np.empty((0, 3), dtype=np.float32),
        np.empty((3, 0), dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        np.ones((1, 3, 2), dtype=np.float32),
    ],
)
def test_bad_or_empty_motion_shapes_are_rejected(array: np.ndarray) -> None:
    with pytest.raises(MotionValidationError, match="shape|non-empty"):
        validate_motion_array(array)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_motion_is_rejected(bad_value: float) -> None:
    motion = np.ones((2, 3), dtype=np.float32)
    motion[1, 2] = bad_value

    with pytest.raises(MotionValidationError, match="non-finite"):
        validate_motion_array(motion)


@pytest.mark.parametrize(
    "array",
    [
        np.array([[True, False]], dtype=np.bool_),
        np.array([[1 + 2j]], dtype=np.complex64),
        np.array([["x"]], dtype=object),
    ],
)
def test_non_real_motion_dtype_is_rejected(array: np.ndarray) -> None:
    with pytest.raises(MotionValidationError, match="real numeric"):
        validate_motion_array(array)


def test_feature_dimension_is_exact() -> None:
    with pytest.raises(MotionValidationError, match="feature dimension"):
        validate_motion_array(np.ones((2, 4)), expected_feature_dim=3)


@pytest.mark.parametrize(
    ("mean", "std", "message"),
    [
        (np.zeros(3), np.array([1.0, 0.0, 1.0]), "std must be"),
        (np.zeros(3), np.array([1.0, -1.0, 1.0]), "std must be"),
        (np.zeros(3), np.array([1.0, np.nan, 1.0]), "non-finite"),
        (np.zeros((1, 3)), np.ones(3), "shape"),
        (np.zeros(3), np.ones(4), "shapes differ"),
    ],
)
def test_invalid_normalization_stats_are_rejected(
    mean: np.ndarray, std: np.ndarray, message: str
) -> None:
    with pytest.raises(MotionValidationError, match=message):
        validate_normalization_stats(mean, std)


def test_minimum_std_is_strict() -> None:
    with pytest.raises(MotionValidationError, match="std must be"):
        validate_normalization_stats(
            np.zeros(2), np.array([0.1, 0.2]), minimum_std=0.1
        )


def test_load_normalization_stats_and_normalize_round_trip(tmp_path: Path) -> None:
    mean_path = tmp_path / "Mean.npy"
    std_path = tmp_path / "Std.npy"
    np.save(mean_path, np.array([1.0, 10.0], dtype=np.float64))
    np.save(std_path, np.array([2.0, 5.0], dtype=np.float64))
    mean, std = load_normalization_stats(
        mean_path, std_path, expected_feature_dim=2
    )
    motion = np.array([[1.0, 15.0], [5.0, 0.0]], dtype=np.float32)
    original = motion.copy()

    normalized = normalize_motion(motion, mean, std)
    restored = denormalize_motion(normalized, mean, std)

    np.testing.assert_allclose(normalized, [[0.0, 1.0], [2.0, -2.0]])
    np.testing.assert_allclose(restored, motion)
    np.testing.assert_array_equal(motion, original)


def test_load_rejects_unsupported_extension_and_missing_file(tmp_path: Path) -> None:
    text = tmp_path / "motion.txt"
    text.write_text("not numpy", encoding="utf-8")

    with pytest.raises(MotionIOError, match=".npy or .npz"):
        load_motion_array(text)
    with pytest.raises(FileNotFoundError):
        load_motion_array(tmp_path / "missing.npy")


def test_npy_does_not_accept_npz_key(tmp_path: Path) -> None:
    path = tmp_path / "motion.npy"
    np.save(path, np.ones((1, 2)))

    with pytest.raises(MotionIOError, match="only valid for .npz"):
        load_motion_array(path, npz_key="arr_0")
