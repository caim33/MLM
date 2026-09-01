from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.fusion import (
    ProjectorSpec,
    ProjectorSpecError,
    build_projector_spec,
    infer_projector_output_shape,
)


def test_projector_spec_is_derived_from_explicit_mapping() -> None:
    config = {
        "motion_projector_input_dim": 512,
        "motion_projector_hidden_dims": [4096],
        "motion_projector_output_dim": 2560,
        "motion_projector_activation": "gelu",
    }

    spec = build_projector_spec(config)

    assert spec.dimensions == (512, 4096, 2560)
    assert [(layer.module_index, layer.in_features, layer.out_features) for layer in spec.linear_layers] == [
        (0, 512, 4096),
        (2, 4096, 2560),
    ]
    expected_linear = 512 * 4096 + 4096 + 4096 * 2560 + 2560
    expected_norm = 2 * 512 + 2 * 2560
    assert spec.parameter_count == expected_linear + expected_norm


def test_legacy_dimension_aliases_are_config_driven() -> None:
    config = SimpleNamespace(
        vqvae_output_emb_width=384,
        motion_projector_hidden_dim=768,
        text_config=SimpleNamespace(hidden_size=1024),
    )

    spec = ProjectorSpec.from_config(config)

    assert spec.dimensions == (384, 768, 1024)
    assert spec.state_dict_prefix == "motion_proj"
    assert spec.pre_norm_prefix == "motion_prenorm"
    assert spec.post_norm_prefix == "motion_postnorm"


def test_explicit_empty_hidden_dims_builds_direct_projection() -> None:
    spec = ProjectorSpec.from_config(
        {
            "motion_projector_input_dim": 4,
            "motion_projector_hidden_dims": [],
            "motion_projector_output_dim": 8,
        }
    )

    assert spec.dimensions == (4, 8)
    assert len(spec.linear_layers) == 1


def test_projector_shape_preserves_all_leading_dimensions() -> None:
    spec = ProjectorSpec(4, (8,), 16)

    assert spec.infer_output_shape((2, 7, 4)) == (2, 7, 16)
    assert infer_projector_output_shape((0, 4), spec) == (0, 16)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((), "at least one"),
        ((2, 5), "expected final dimension 4"),
        ((2, -1, 4), "must be >= 0"),
        ((True, 4), "non-negative integer"),
    ],
)
def test_projector_shape_rejects_bad_or_truncated_shape(
    shape: tuple[object, ...], message: str
) -> None:
    spec = ProjectorSpec(4, (8,), 16)

    with pytest.raises(ProjectorSpecError, match=message):
        spec.infer_output_shape(shape)  # type: ignore[arg-type]


def test_hidden_dimensions_must_be_explicit() -> None:
    with pytest.raises(ProjectorSpecError, match="hidden_dims"):
        ProjectorSpec.from_config(
            {
                "motion_projector_input_dim": 4,
                "motion_projector_output_dim": 8,
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        {
            "motion_projector_input_dim": 0,
            "motion_projector_hidden_dims": [8],
            "motion_projector_output_dim": 16,
        },
        {
            "motion_projector_input_dim": 4,
            "motion_projector_hidden_dims": [0],
            "motion_projector_output_dim": 16,
        },
        {
            "motion_projector_input_dim": 4,
            "motion_projector_hidden_dims": [8],
            "motion_projector_output_dim": -1,
        },
    ],
)
def test_projector_rejects_non_positive_dimensions(config: dict[str, object]) -> None:
    with pytest.raises(ProjectorSpecError, match="must be > 0"):
        ProjectorSpec.from_config(config)


def test_projector_rejects_unsupported_activation() -> None:
    with pytest.raises(ProjectorSpecError, match="unsupported activation"):
        ProjectorSpec(4, (8,), 16, activation="made_up")


def test_projector_flags_are_real_booleans() -> None:
    with pytest.raises(ProjectorSpecError, match="must be bool"):
        ProjectorSpec.from_config(
            {
                "motion_projector_input_dim": 4,
                "motion_projector_hidden_dims": [8],
                "motion_projector_output_dim": 16,
                "motion_projector_bias": "false",
            }
        )


def test_projector_has_no_torch_dependency() -> None:
    import motionllm.fusion.projector as projector

    assert "torch" not in projector.__dict__
