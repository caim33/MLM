from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.fusion import (
    MotionPlaceholderError,
    MotionTextProtocol,
    MotionTokenIds,
    find_motion_anchors,
    parse_and_validate_motion_spans,
    parse_motion_spans,
    render_motion_span,
    replace_motion_anchors,
    validate_placeholder_counts,
)


TOKEN_SPEC = MotionTokenIds(start=10, placeholder=11, end=12)


def test_find_and_replace_multiple_motion_anchors() -> None:
    prompt = "first <motion>, then <motion>."

    anchors = find_motion_anchors(prompt)
    expanded = replace_motion_anchors(prompt, [1, 3])

    assert len(anchors) == 2
    assert expanded == (
        "first <motion_start><motion><motion_end>, then "
        "<motion_start><motion><motion><motion><motion_end>."
    )


def test_no_anchor_with_no_counts_is_identity() -> None:
    assert replace_motion_anchors("text only", ()) == "text only"


@pytest.mark.parametrize("counts", [1, [1], [1, 2, 3]])
def test_anchor_count_must_match_motion_count(counts: object) -> None:
    with pytest.raises(MotionPlaceholderError, match="anchor/count mismatch"):
        replace_motion_anchors("<motion> and <motion>", counts)  # type: ignore[arg-type]


def test_already_expanded_prompt_is_rejected() -> None:
    with pytest.raises(MotionPlaceholderError, match="re-expansion"):
        replace_motion_anchors(
            "<motion_start><motion><motion_end>",
            1,
        )


def test_render_requires_positive_placeholder_count() -> None:
    with pytest.raises(MotionPlaceholderError, match="must be > 0"):
        render_motion_span(0)


def test_protocol_boundaries_must_be_distinct() -> None:
    with pytest.raises(MotionPlaceholderError, match="must be distinct"):
        MotionTextProtocol(start="x", placeholder="x", end="y")


def test_parse_multiple_spans_and_validate_exact_counts() -> None:
    tokens = [99, 10, 11, 12, 88, 10, 11, 11, 12, 77]

    spans = parse_and_validate_motion_spans(tokens, TOKEN_SPEC, [1, 2])

    assert [(span.start_index, span.end_index) for span in spans] == [(1, 3), (5, 8)]
    assert [span.placeholder_positions for span in spans] == [(2,), (6, 7)]


def test_allowed_interstitial_tokens_are_explicit() -> None:
    tokens = [10, 90, 11, 90, 11, 12]

    with pytest.raises(MotionPlaceholderError, match="unexpected token"):
        parse_motion_spans(tokens, TOKEN_SPEC)
    spans = parse_motion_spans(
        tokens, TOKEN_SPEC, allowed_interstitial_token_ids=[90]
    )
    validate_placeholder_counts(spans, 2)


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        ([11], "stray"),
        ([12], "no open span"),
        ([10, 10, 11, 12], "nested"),
        ([10, 12], "no placeholders"),
        ([10, 11, 99, 12], "unexpected token"),
        ([10, 11], "truncated"),
    ],
)
def test_malformed_or_truncated_spans_fail_closed(
    tokens: list[int], message: str
) -> None:
    with pytest.raises(MotionPlaceholderError, match=message):
        parse_motion_spans(tokens, TOKEN_SPEC)


def test_wrong_placeholder_quantity_is_rejected() -> None:
    spans = parse_motion_spans([10, 11, 11, 12], TOKEN_SPEC)

    with pytest.raises(MotionPlaceholderError, match="expected 3"):
        validate_placeholder_counts(spans, 3)


def test_wrong_number_of_motion_spans_is_rejected() -> None:
    spans = parse_motion_spans([10, 11, 12], TOKEN_SPEC)

    with pytest.raises(MotionPlaceholderError, match="span/count mismatch"):
        validate_placeholder_counts(spans, [1, 1])


def test_token_ids_are_loaded_from_mapping_or_object_config() -> None:
    mapping = {
        "motion_start_token_id": 1,
        "motion_placeholder_token_id": 2,
        "motion_end_token_id": 3,
    }
    obj = SimpleNamespace(**mapping)

    assert MotionTokenIds.from_config(mapping) == MotionTokenIds(1, 2, 3)
    assert MotionTokenIds.from_config(obj) == MotionTokenIds(1, 2, 3)


def test_token_protocol_ids_must_be_distinct() -> None:
    with pytest.raises(MotionPlaceholderError, match="must be distinct"):
        MotionTokenIds(start=1, placeholder=1, end=2)


def test_allowed_interstitial_ids_cannot_mask_protocol_tokens() -> None:
    with pytest.raises(MotionPlaceholderError, match="cannot include"):
        parse_motion_spans(
            [10, 11, 12], TOKEN_SPEC, allowed_interstitial_token_ids=[11]
        )
