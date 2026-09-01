from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration
from motionllm.models import MotionResizePolicy


def bare_model(*, allow_broadcast=False, input_dim=3, placeholder_token_id=11):
    model = object.__new__(Qwen3VlMotionForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.motion_spec = SimpleNamespace(
        allow_batch_broadcast=allow_broadcast,
        input_dim=input_dim,
        placeholder_token_id=placeholder_token_id,
        resize_policy=MotionResizePolicy.ERROR,
    )
    return model


def test_legacy_class_name_and_four_branch_normalization_remain_callable():
    normalized = Qwen3VlMotionForConditionalGeneration._normalize_branch_per_sample(
        ["v", "m", "vm", "t"], 4, strict=True
    )
    assert normalized == ["v", "m", "vm", "t"]


def test_legacy_batch_normalizer_keeps_none_and_motion_per_row():
    model = bare_model()
    first = torch.zeros((4, 3))
    second = torch.zeros((8, 3))
    motions, lengths = model._normalize_batch_motion_inputs(
        [None, first, second, None],
        [None, (4,), (8,), None],
        4,
    )
    assert motions[0] is None and motions[3] is None
    assert motions[1] is first and motions[2] is second
    assert lengths == [None, [4], [8], None]


def test_legacy_batch_normalizer_rejects_implicit_motion_broadcast():
    model = bare_model(allow_broadcast=False)
    with pytest.raises(ValueError, match="broadcast"):
        model._normalize_batch_motion_inputs(
            [torch.zeros((4, 3))], [(4,)], 2
        )


def test_legacy_batch_normalizer_rejects_bad_shape_and_nonfinite_motion():
    model = bare_model()
    with pytest.raises(ValueError, match="2D"):
        model._normalize_batch_motion_inputs(
            [torch.zeros((1, 4, 3))], [(4,)], 1
        )
    bad = torch.zeros((4, 3))
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model._normalize_batch_motion_inputs([bad], [(4,)], 1)


def test_packed_motion_spans_reject_per_segment_ownership_swap():
    model = bare_model()
    model.config = SimpleNamespace(
        motion_start_token_id=10,
        motion_end_token_id=12,
        motion_allowed_interstitial_token_ids=(),
    )
    swapped = torch.tensor([10, 11, 11, 11, 12, 99, 10, 11, 11, 12])
    with pytest.raises(Exception, match="segment 0"):
        model._strict_motion_placeholder_positions(
            swapped,
            expected_segment_feature_counts=(2, 3),
        )
