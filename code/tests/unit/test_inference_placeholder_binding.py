from __future__ import annotations

import pytest

from qwenvl.infer.infer_qwen3_vl_motion import validate_motion_placeholder_binding


def test_accepts_shared_vocabulary_placeholder() -> None:
    assert (
        validate_motion_placeholder_binding(
            42,
            tokenizer_size=100,
            embedding_size=100,
        )
        == "vocabulary_token"
    )


def test_accepts_explicit_external_legacy_sentinel() -> None:
    assert (
        validate_motion_placeholder_binding(
            160001,
            tokenizer_size=151671,
            embedding_size=151671,
            boundary_token_ids=(151669, 151670),
        )
        == "external_sentinel"
    )


@pytest.mark.parametrize("value", [-1, True, "160001", None])
def test_rejects_invalid_placeholder_types_and_values(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_motion_placeholder_binding(
            value,
            tokenizer_size=100,
            embedding_size=100,
        )


def test_rejects_half_in_half_out_binding() -> None:
    with pytest.raises(ValueError, match="only one"):
        validate_motion_placeholder_binding(
            105,
            tokenizer_size=100,
            embedding_size=110,
        )


def test_rejects_boundary_collision() -> None:
    with pytest.raises(ValueError, match="differ"):
        validate_motion_placeholder_binding(
            90,
            tokenizer_size=100,
            embedding_size=100,
            boundary_token_ids=(90, 91),
        )
