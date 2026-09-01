from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.fusion import MotionTokenIds, parse_and_validate_motion_spans
from motionllm.models import (
    MotionInjectionError,
    MotionModelConfigError,
    enumerate_motion_compute_placements,
    resolve_motion_model_spec,
    validate_motion_compute_contract,
    validate_motion_encoder_downsample,
    validate_motion_segment_ownership,
)
from motionllm.motion import plan_temporal_length


def _modern_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "motion_input_dim": 251,
        "motion_encoder_output_dim": 384,
        "motion_downsample_factor": 8,
        "motion_placeholder_token_id": 900,
        "motion_start_token_id": 901,
        "motion_end_token_id": 902,
        "motion_projector_input_dim": 384,
        "motion_projector_hidden_dims": (512,),
        "motion_projector_output_dim": 1024,
        "motion_projector_activation": "gelu",
        "motion_projector_bias": True,
        "motion_projector_pre_norm": True,
        "motion_projector_post_norm": False,
        "motion_resize_policy": "error",
        "motion_compute_dtype": "model",
        "motion_device_policy": "encoder",
        "motion_allow_batch_broadcast": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _TensorPlacement:
    def __init__(self, dtype: str, device: str) -> None:
        self.dtype = dtype
        self.device = device


class _ModuleState:
    def __init__(
        self,
        parameters: list[tuple[str, _TensorPlacement]] | None = None,
        buffers: list[tuple[str, _TensorPlacement]] | None = None,
    ) -> None:
        self._parameters = parameters or []
        self._buffers = buffers or []

    def named_parameters(self, recurse: bool = True):
        assert recurse is True
        return iter(self._parameters)

    def named_buffers(self, recurse: bool = True):
        assert recurse is True
        return iter(self._buffers)


def _uniform_module(name: str = "weight") -> _ModuleState:
    return _ModuleState([(name, _TensorPlacement("bf16", "cuda:0"))])


def test_dtype_audit_reaches_the_last_parameter_in_a_deep_encoder() -> None:
    parameters = [
        (f"blocks.{index}.weight", _TensorPlacement("bf16", "cuda:0"))
        for index in range(2_048)
    ]
    parameters[-1] = (
        parameters[-1][0],
        _TensorPlacement("float32", "cuda:0"),
    )
    placements = enumerate_motion_compute_placements(
        encoder_name="motion_encoder",
        encoder=_ModuleState(parameters),
        motion_prenorm=_uniform_module(),
        motion_proj=_uniform_module(),
        motion_postnorm=_uniform_module(),
        apply_postnorm=False,
        motion_boundary_embed=_uniform_module(),
    )

    assert len(placements) == 2_051
    with pytest.raises(MotionInjectionError, match=r"blocks\.2047\.weight dtype"):
        validate_motion_compute_contract(
            expected_dtype="bf16",
            expected_device="cuda:0",
            module_placements=placements,
        )


def test_buffer_and_boundary_embedding_are_both_in_compute_audit() -> None:
    placements = enumerate_motion_compute_placements(
        encoder_name="motion_encoder",
        encoder=_ModuleState(
            parameters=[("weight", _TensorPlacement("bf16", "cuda:0"))],
            buffers=[("late_buffer", _TensorPlacement("bf16", "cuda:1"))],
        ),
        motion_prenorm=_uniform_module(),
        motion_proj=_uniform_module(),
        motion_postnorm=_uniform_module(),
        apply_postnorm=True,
        motion_boundary_embed=_uniform_module("embedding"),
    )
    names = {name for name, _, _ in placements}
    assert "motion_encoder.late_buffer" in names
    assert "motion_postnorm.weight" in names
    assert "motion_boundary_embed.embedding" in names
    with pytest.raises(MotionInjectionError, match="late_buffer device"):
        validate_motion_compute_contract(
            expected_dtype="bf16",
            expected_device="cuda:0",
            module_placements=placements,
        )


def test_duplicate_parameter_buffer_path_fails_closed() -> None:
    duplicated = _ModuleState(
        parameters=[("shared", _TensorPlacement("bf16", "cuda:0"))],
        buffers=[("shared", _TensorPlacement("bf16", "cuda:0"))],
    )
    with pytest.raises(MotionInjectionError, match="duplicated"):
        enumerate_motion_compute_placements(
            encoder_name="motion_encoder",
            encoder=duplicated,
            motion_prenorm=_uniform_module(),
            motion_proj=_uniform_module(),
            motion_postnorm=_uniform_module(),
            apply_postnorm=False,
            motion_boundary_embed=_uniform_module(),
        )


def test_twenty_thousand_packed_segments_preserve_per_segment_ownership() -> None:
    size = 20_000
    features = (1,) * size
    placeholders = (1,) * size
    assert validate_motion_segment_ownership(features, placeholders) == features

    mismatched = list(placeholders)
    mismatched[-1] = 2
    with pytest.raises(MotionInjectionError, match=f"segment {size - 1}"):
        validate_motion_segment_ownership(features, mismatched)


def test_thousands_of_placeholder_spans_are_parsed_without_cross_ownership() -> None:
    token_spec = MotionTokenIds(start=10, placeholder=11, end=12)
    span_count = 5_000
    tokens = [token for _ in range(span_count) for token in (10, 11, 12)]
    spans = parse_and_validate_motion_spans(tokens, token_spec, (1,) * span_count)

    assert len(spans) == span_count
    assert spans[0].placeholder_positions == (1,)
    assert spans[-1].placeholder_positions == (len(tokens) - 2,)


def test_huge_encoder_exponent_is_rejected_before_power_allocation() -> None:
    spec = resolve_motion_model_spec(_modern_config())
    enormous_even_stride = 1 << 1_000_000

    with pytest.raises(MotionModelConfigError, match="disagrees"):
        validate_motion_encoder_downsample(
            spec,
            vqvae_down_t=1_000_000,
            vqvae_stride_t=enormous_even_stride,
        )


def test_gigantic_declared_raw_length_can_be_planned_under_a_small_cap() -> None:
    raw_length = 10**100_000
    contract = plan_temporal_length(
        raw_length,
        downsample_factor=16,
        max_encoded_steps=32,
    )

    assert contract.raw_length == raw_length
    assert contract.retained_length == 512
    assert contract.padded_length == 512
    assert contract.encoded_length == 32
