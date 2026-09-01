from __future__ import annotations

import numpy as np
import pytest

from motionllm.motion import (
    MotionValidationError,
    TemporalContractError,
    TemporalLengthContract,
    apply_temporal_contract,
    downsample_motion,
    plan_temporal_length,
    prepare_motion_temporal,
)


@pytest.mark.parametrize(
    ("raw", "padded", "encoded", "padding"),
    [
        (1, 4, 1, 3),
        (4, 4, 1, 0),
        (5, 8, 2, 3),
        (8, 8, 2, 0),
    ],
)
def test_temporal_length_contract(
    raw: int, padded: int, encoded: int, padding: int
) -> None:
    contract = plan_temporal_length(raw, downsample_factor=4)

    assert contract.raw_length == raw
    assert contract.retained_length == raw
    assert contract.padded_length == padded
    assert contract.encoded_length == encoded
    assert contract.placeholder_count == encoded
    assert contract.padding_length == padding
    assert contract.truncated_length == 0


def test_single_frame_motion_repeats_last_frame() -> None:
    motion = np.array([[2.0, -1.0]], dtype=np.float32)

    padded, contract = prepare_motion_temporal(motion, downsample_factor=4)

    assert padded.shape == (4, 2)
    np.testing.assert_array_equal(padded, np.repeat(motion, 4, axis=0))
    assert contract.encoded_length == 1


def test_temporal_truncation_uses_post_downsample_cap() -> None:
    motion = np.arange(22, dtype=np.float32).reshape(11, 2)

    padded, contract = prepare_motion_temporal(
        motion, downsample_factor=4, max_encoded_steps=2
    )

    assert contract.retained_length == 8
    assert contract.truncated_length == 3
    assert contract.padded_length == 8
    assert contract.encoded_length == 2
    np.testing.assert_array_equal(padded, motion[:8])


def test_truncation_then_padding_is_fully_accounted() -> None:
    contract = plan_temporal_length(
        5, downsample_factor=4, max_encoded_steps=2
    )

    assert contract.retained_length == 5
    assert contract.padding_length == 3
    assert contract.encoded_length == 2


def test_zero_padding_does_not_modify_input() -> None:
    motion = np.arange(6, dtype=np.float32).reshape(3, 2)
    original = motion.copy()
    contract = plan_temporal_length(3, downsample_factor=4)

    padded = apply_temporal_contract(motion, contract, pad_mode="zero")

    np.testing.assert_array_equal(padded[-1], [0, 0])
    np.testing.assert_array_equal(motion, original)


def test_reference_downsampling_obeys_exact_length() -> None:
    motion = np.arange(10, dtype=np.float32).reshape(5, 2)
    padded, contract = prepare_motion_temporal(motion, downsample_factor=4)

    result = downsample_motion(padded, contract, reduction="mean")

    assert result.shape == (contract.encoded_length, 2)
    np.testing.assert_allclose(result[0], motion[:4].mean(axis=0))
    np.testing.assert_allclose(result[1], motion[-1])


@pytest.mark.parametrize("reduction", ["first", "last"])
def test_reference_downsampling_selection_modes(reduction: str) -> None:
    motion = np.arange(8, dtype=np.float32).reshape(4, 2)
    padded, contract = prepare_motion_temporal(motion, downsample_factor=2)

    result = downsample_motion(padded, contract, reduction=reduction)  # type: ignore[arg-type]

    expected = motion[::2] if reduction == "first" else motion[1::2]
    np.testing.assert_array_equal(result, expected)


def test_empty_motion_is_rejected_before_padding() -> None:
    with pytest.raises(MotionValidationError, match="non-empty"):
        prepare_motion_temporal(np.empty((0, 3), dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"raw_length": 0}, "raw_length"),
        ({"raw_length": 4, "downsample_factor": 0}, "downsample_factor"),
        ({"raw_length": 4, "max_encoded_steps": 0}, "max_encoded_steps"),
    ],
)
def test_invalid_temporal_plan_is_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(TemporalContractError, match=message):
        plan_temporal_length(**kwargs)


def test_contract_rejects_inconsistent_encoded_length() -> None:
    with pytest.raises(TemporalContractError, match="encoded_length must equal"):
        TemporalLengthContract(
            raw_length=4,
            retained_length=4,
            padded_length=4,
            downsample_factor=4,
            encoded_length=2,
        )


def test_apply_rejects_contract_for_another_motion() -> None:
    contract = plan_temporal_length(4)

    with pytest.raises(TemporalContractError, match="does not match"):
        apply_temporal_contract(np.ones((3, 2)), contract)


def test_downsample_rejects_unpadded_input() -> None:
    contract = plan_temporal_length(5)

    with pytest.raises(TemporalContractError, match="padded motion length"):
        downsample_motion(np.ones((5, 2)), contract)
