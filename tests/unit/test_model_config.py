from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.models import (
    MotionDevicePolicy,
    MotionDTypePolicy,
    MotionModelConfigError,
    MotionResizePolicy,
    migrate_legacy_motion_config,
    resolve_motion_model_spec,
    validate_motion_encoder_downsample,
)


def modern_config(**overrides):
    values = {
        "motion_input_dim": 251,
        "motion_encoder_output_dim": 384,
        "motion_downsample_factor": 8,
        "motion_placeholder_token_id": 900,
        "motion_start_token_id": 901,
        "motion_end_token_id": 902,
        "motion_projector_input_dim": 384,
        "motion_projector_hidden_dims": (512, 768),
        "motion_projector_output_dim": 1024,
        "motion_projector_activation": "silu",
        "motion_projector_bias": False,
        "motion_projector_pre_norm": True,
        "motion_projector_post_norm": True,
        "motion_resize_policy": "error",
        "motion_compute_dtype": "model",
        "motion_device_policy": "encoder",
        "motion_allow_batch_broadcast": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_spec_uses_only_explicit_dimensions_and_policies():
    spec = resolve_motion_model_spec(modern_config())
    assert spec.input_dim == 251
    assert spec.encoder_output_dim == 384
    assert spec.projector.dimensions == (384, 512, 768, 1024)
    assert spec.resize_policy is MotionResizePolicy.ERROR
    assert spec.dtype_policy is MotionDTypePolicy.MODEL
    assert spec.device_policy is MotionDevicePolicy.ENCODER


def test_missing_explicit_field_fails_closed():
    config = modern_config()
    del config.motion_placeholder_token_id
    with pytest.raises(MotionModelConfigError, match="motion_placeholder_token_id"):
        resolve_motion_model_spec(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("motion_resize_policy", "sometimes"),
        ("motion_compute_dtype", "auto"),
        ("motion_device_policy", "cuda:0"),
    ],
)
def test_unknown_policy_is_never_guessed(field, value):
    with pytest.raises(MotionModelConfigError):
        resolve_motion_model_spec(modern_config(**{field: value}))


def test_projector_encoder_dimension_disagreement_is_rejected():
    with pytest.raises(MotionModelConfigError, match="projector input"):
        resolve_motion_model_spec(
            modern_config(motion_projector_input_dim=385)
        )


def test_boundary_ids_must_be_both_present_and_distinct():
    with pytest.raises(MotionModelConfigError, match="set together"):
        resolve_motion_model_spec(modern_config(motion_end_token_id=None))
    with pytest.raises(MotionModelConfigError, match="distinct"):
        resolve_motion_model_spec(
            modern_config(motion_end_token_id=900)
        )


def test_legacy_migration_persists_auditable_defaults_once():
    config = SimpleNamespace(
        dataname="kit",
        vqvae_output_emb_width=512,
        text_config=SimpleNamespace(hidden_size=2560),
    )
    migrated = migrate_legacy_motion_config(config)
    assert "motion_input_dim" in migrated
    assert "motion_projector_hidden_dims" in migrated
    assert config.motion_input_dim == 251
    assert config.motion_projector_hidden_dims == (4096,)
    assert config.motion_projector_output_dim == 2560
    assert config.motion_placeholder_token_id == 160001
    assert config.motion_projector_post_norm is False
    assert config.motion_legacy_postnorm_state_compat is True
    assert migrate_legacy_motion_config(config) == ()
    spec = resolve_motion_model_spec(config)
    assert spec.projector.dimensions == (512, 4096, 2560)
    assert spec.projector.post_norm is False


def test_legacy_migration_does_not_override_explicit_values():
    config = modern_config()
    assert migrate_legacy_motion_config(config) == ()
    assert resolve_motion_model_spec(config).projector.hidden_dims == (512, 768)


def test_normalization_paths_must_be_absolute_and_paired(tmp_path):
    with pytest.raises(MotionModelConfigError, match="absolute"):
        resolve_motion_model_spec(
            modern_config(motion_normalization_mean_path="Mean.npy")
        )
    with pytest.raises(MotionModelConfigError, match="set together"):
        resolve_motion_model_spec(
            modern_config(motion_normalization_mean_path=str(tmp_path / "Mean.npy"))
        )


def test_encoder_downsample_must_match_placeholder_contract():
    spec = resolve_motion_model_spec(
        modern_config(motion_downsample_factor=8)
    )
    assert validate_motion_encoder_downsample(
        spec, vqvae_down_t=3, vqvae_stride_t=2
    ) == 8
    with pytest.raises(MotionModelConfigError, match="disagrees"):
        validate_motion_encoder_downsample(
            spec, vqvae_down_t=2, vqvae_stride_t=2
        )
    with pytest.raises(MotionModelConfigError, match="disagrees"):
        validate_motion_encoder_downsample(
            spec, vqvae_down_t=1_000_000, vqvae_stride_t=2
        )
    with pytest.raises(MotionModelConfigError, match="must be even"):
        validate_motion_encoder_downsample(
            spec, vqvae_down_t=1, vqvae_stride_t=3
        )
