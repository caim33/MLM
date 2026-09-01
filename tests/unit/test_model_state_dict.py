from __future__ import annotations

import numpy as np
import pytest

from motionllm.models import (
    StateDictAuditError,
    audit_state_dict,
    extract_state_dict,
    normalize_state_dict_keys,
    select_vq_checkpoint_state,
)


def test_extracts_one_known_wrapper_only():
    state = {"encoder.weight": np.zeros((2, 3))}
    assert extract_state_dict({"state_dict": state}) is state
    with pytest.raises(StateDictAuditError, match="ambiguous"):
        extract_state_dict({"state_dict": state, "net": state})
    with pytest.raises(StateDictAuditError, match="must not be empty"):
        extract_state_dict({"state_dict": {}})


def test_prefix_normalization_is_leading_only_and_collision_safe():
    value = np.zeros((1,))
    normalized = normalize_state_dict_keys(
        {"module.vqvae.encoder.weight": value, "block.vqvae.bias": value}
    )
    assert set(normalized) == {"encoder.weight", "block.vqvae.bias"}
    with pytest.raises(StateDictAuditError, match="collision"):
        normalize_state_dict_keys(
            {"module.encoder.weight": value, "encoder.weight": value}
        )


def test_clean_state_audit_records_matched_keys():
    expected = {"a": np.zeros((2, 3)), "b": np.zeros((4,))}
    candidate = {"a": np.ones((2, 3)), "b": np.ones((4,))}
    report = audit_state_dict(expected, candidate).require_clean()
    assert report.ok
    assert report.matched_keys == ("a", "b")


def test_missing_unexpected_and_shape_are_all_fatal():
    expected = {"a": np.zeros((2, 3)), "missing": np.zeros((1,))}
    candidate = {"a": np.ones((3, 2)), "extra": np.ones((1,))}
    report = audit_state_dict(expected, candidate)
    assert report.missing_keys == ("missing",)
    assert report.unexpected_keys == ("extra",)
    assert report.shape_mismatches == (("a", (2, 3), (3, 2)),)
    with pytest.raises(StateDictAuditError, match="checkpoint state audit failed"):
        report.require_clean()


def test_vq_checkpoint_selects_exact_full_or_bare_encoder_state():
    full = {
        "encoder.block.weight": np.zeros((2, 3)),
        "decoder.block.weight": np.zeros((3, 2)),
    }
    encoder = {"block.weight": np.zeros((2, 3))}

    selected_full = select_vq_checkpoint_state(
        full_expected=full,
        encoder_expected=encoder,
        candidate={key: np.ones_like(value) for key, value in full.items()},
    )
    assert selected_full.target == "full_vqvae"

    selected_bare = select_vq_checkpoint_state(
        full_expected=full,
        encoder_expected=encoder,
        candidate={"block.weight": np.ones((2, 3))},
    )
    assert selected_bare.target == "encoder_only"
    assert set(selected_bare.state_dict) == {"block.weight"}


def test_vq_checkpoint_accepts_exact_encoder_prefix_and_rejects_partial_mixed():
    full = {
        "encoder.block.weight": np.zeros((2, 3)),
        "decoder.block.weight": np.zeros((3, 2)),
    }
    encoder = {"block.weight": np.zeros((2, 3))}
    selected = select_vq_checkpoint_state(
        full_expected=full,
        encoder_expected=encoder,
        candidate={"encoder.block.weight": np.ones((2, 3))},
    )
    assert selected.target == "encoder_only"
    assert set(selected.state_dict) == {"block.weight"}

    for bad in (
        {"block.weight": np.ones((3, 2))},
        {
            "encoder.block.weight": np.ones((2, 3)),
            "metadata": np.ones((1,)),
        },
    ):
        with pytest.raises(StateDictAuditError, match="neither"):
            select_vq_checkpoint_state(
                full_expected=full,
                encoder_expected=encoder,
                candidate=bad,
            )
