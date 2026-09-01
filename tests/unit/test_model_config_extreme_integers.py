from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.models import (
    MotionModelConfigError,
    resolve_motion_model_spec,
    validate_motion_encoder_downsample,
)


def _spec():
    return resolve_motion_model_spec(
        SimpleNamespace(
            motion_input_dim=251,
            motion_encoder_output_dim=384,
            motion_downsample_factor=8,
            motion_placeholder_token_id=900,
            motion_start_token_id=901,
            motion_end_token_id=902,
            motion_projector_input_dim=384,
            motion_projector_hidden_dims=(512,),
            motion_projector_output_dim=1024,
            motion_projector_activation="gelu",
            motion_projector_bias=True,
            motion_projector_pre_norm=True,
            motion_projector_post_norm=False,
            motion_resize_policy="error",
            motion_compute_dtype="model",
            motion_device_policy="encoder",
            motion_allow_batch_broadcast=False,
        )
    )


@pytest.mark.parametrize(
    "down_t,stride_t",
    [
        (1_000_000, 1 << 1_000_000),
        (1, 1 << 1_000_000),
        (1 << 1_000_000, 2),
    ],
    ids=("huge-base-and-exponent", "huge-base", "huge-exponent"),
)
def test_enormous_encoder_integers_have_bounded_safe_diagnostics(down_t, stride_t):
    with pytest.raises(MotionModelConfigError) as captured:
        validate_motion_encoder_downsample(
            _spec(), vqvae_down_t=down_t, vqvae_stride_t=stride_t
        )

    message = str(captured.value)
    assert len(message) < 512
    assert "bit_length=1000001" in message
    assert "low16=0x0000" in message


@pytest.mark.parametrize(
    "down_t,stride_t",
    [
        (True, 2),
        (3, False),
        (-(1 << 1_000_000), 2),
        (3, -(1 << 1_000_000)),
        (0, 2),
        (3, 0),
    ],
    ids=(
        "bool-exponent",
        "bool-base",
        "negative-huge-exponent",
        "negative-huge-base",
        "zero-exponent",
        "zero-base",
    ),
)
def test_invalid_encoder_integers_only_raise_model_config_error(down_t, stride_t):
    with pytest.raises(MotionModelConfigError) as captured:
        validate_motion_encoder_downsample(
            _spec(), vqvae_down_t=down_t, vqvae_stride_t=stride_t
        )

    message = str(captured.value)
    assert len(message) < 512
    if isinstance(down_t, int) and not isinstance(down_t, bool) and down_t < 0:
        assert "sign=-" in message
    if isinstance(stride_t, int) and not isinstance(stride_t, bool) and stride_t < 0:
        assert "sign=-" in message


def test_ordinary_mismatch_keeps_readable_decimal_diagnostics():
    with pytest.raises(
        MotionModelConfigError,
        match=r"motion_downsample_factor=8.*vqvae_stride_t \*\* vqvae_down_t=4",
    ):
        validate_motion_encoder_downsample(
            _spec(), vqvae_down_t=1, vqvae_stride_t=4
        )
