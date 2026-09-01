from __future__ import annotations

import pytest

from motionllm.contracts import Modality
from motionllm.models import (
    MotionInjectionError,
    MotionResizePolicy,
    normalize_modalities,
    required_feature_length,
    validate_motion_compute_contract,
    validate_motion_presence,
    validate_motion_segment_ownership,
    validate_preembedded_motion_inputs,
)


def test_all_four_modalities_normalize_without_case_guessing():
    assert normalize_modalities(["v", "m", "vm", "t"], batch_size=4) == (
        Modality.VIDEO,
        Modality.MOTION,
        Modality.VIDEO_MOTION,
        Modality.TEXT,
    )
    assert normalize_modalities(["V", "M", "VM", "T"], batch_size=4) == (
        Modality.VIDEO,
        Modality.MOTION,
        Modality.VIDEO_MOTION,
        Modality.TEXT,
    )
    with pytest.raises(MotionInjectionError):
        normalize_modalities("Vm", batch_size=1)


def test_mixed_batch_motion_presence_is_checked_per_row():
    modalities = (Modality.VIDEO, Modality.MOTION, Modality.VIDEO_MOTION, Modality.TEXT)
    assert validate_motion_presence(
        modalities, (False, True, True, False), prefill=True
    ) == (False, True, True, False)
    with pytest.raises(MotionInjectionError, match="requires motion"):
        validate_motion_presence(
            modalities, (False, False, True, False), prefill=True
        )
    with pytest.raises(MotionInjectionError, match="forbids motion"):
        validate_motion_presence(
            modalities, (True, True, True, False), prefill=True
        )


def test_decode_does_not_require_motion_for_motion_modalities():
    assert validate_motion_presence(
        (Modality.MOTION, Modality.VIDEO_MOTION),
        (False, False),
        prefill=False,
    ) == (False, False)
    with pytest.raises(MotionInjectionError, match="decode phase"):
        validate_motion_presence(
            (Modality.MOTION,), (True,), prefill=False
        )


def test_resize_is_explicit_and_error_is_default_capability():
    assert required_feature_length(5, 5, policy=MotionResizePolicy.ERROR) == 5
    with pytest.raises(MotionInjectionError, match="mismatch"):
        required_feature_length(4, 5, policy=MotionResizePolicy.ERROR)
    assert required_feature_length(4, 5, policy=MotionResizePolicy.LINEAR) == 5


def test_packed_segment_ownership_is_checked_per_span_not_only_in_aggregate():
    assert validate_motion_segment_ownership((2, 3), (2, 3)) == (2, 3)
    with pytest.raises(MotionInjectionError, match="segment 0"):
        validate_motion_segment_ownership((2, 3), (3, 2))
    assert validate_motion_segment_ownership(
        (2, 3), (3, 2), allow_per_segment_resize=True
    ) == (2, 3)
    with pytest.raises(MotionInjectionError, match="count mismatch"):
        validate_motion_segment_ownership((5,), (2, 3))


def test_preembedded_requests_cannot_silently_drop_raw_motion():
    validate_preembedded_motion_inputs(
        inputs_embeds_present=True,
        motion_present=False,
        motion_lengths_present=False,
    )
    with pytest.raises(MotionInjectionError, match="cannot be combined"):
        validate_preembedded_motion_inputs(
            inputs_embeds_present=True,
            motion_present=True,
            motion_lengths_present=True,
        )


def test_motion_compute_contract_checks_every_parameterized_module():
    placements = (
        ("encoder", "bf16", "cuda:0"),
        ("projector", "bf16", "cuda:0"),
    )
    validate_motion_compute_contract(
        expected_dtype="bf16",
        expected_device="cuda:0",
        module_placements=placements,
    )
    with pytest.raises(MotionInjectionError, match="projector dtype"):
        validate_motion_compute_contract(
            expected_dtype="bf16",
            expected_device="cuda:0",
            module_placements=(
                placements[0],
                ("projector", "float32", "cuda:0"),
            ),
        )
    with pytest.raises(MotionInjectionError, match="encoder device"):
        validate_motion_compute_contract(
            expected_dtype="bf16",
            expected_device="cuda:1",
            module_placements=placements,
        )
