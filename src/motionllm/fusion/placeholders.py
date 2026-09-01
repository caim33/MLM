"""Strict text-anchor expansion and token-level motion span parsing."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import MotionPlaceholderError


_MISSING = object()


def _config_value(config: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(config, Mapping):
        if key in config:
            return config[key]
    elif hasattr(config, key):
        return getattr(config, key)
    if default is _MISSING:
        raise MotionPlaceholderError(f"missing required config field {key!r}")
    return default


def _token_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise MotionPlaceholderError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise MotionPlaceholderError(
            f"{name} must be a non-negative integer"
        ) from exc
    if result < 0:
        raise MotionPlaceholderError(f"{name} must be >= 0, got {result}")
    return result


def _positive_count(value: Any, *, name: str) -> int:
    result = _token_id(value, name=name)
    if result == 0:
        raise MotionPlaceholderError(f"{name} must be > 0")
    return result


@dataclass(frozen=True, slots=True)
class MotionTextProtocol:
    """Text form of the ``<motion_start><motion><motion_end>`` protocol."""

    anchor: str = "<motion>"
    start: str = "<motion_start>"
    placeholder: str = "<motion>"
    end: str = "<motion_end>"

    def __post_init__(self) -> None:
        values = {
            "anchor": self.anchor,
            "start": self.start,
            "placeholder": self.placeholder,
            "end": self.end,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value:
                raise MotionPlaceholderError(f"{name} token must be a non-empty string")
        if len({self.start, self.placeholder, self.end}) != 3:
            raise MotionPlaceholderError(
                "motion start, placeholder, and end strings must be distinct"
            )


@dataclass(frozen=True, slots=True)
class MotionTokenIds:
    """Configured token IDs used by the encoded motion protocol."""

    start: int
    placeholder: int
    end: int

    def __post_init__(self) -> None:
        start = _token_id(self.start, name="motion start token ID")
        placeholder = _token_id(
            self.placeholder, name="motion placeholder token ID"
        )
        end = _token_id(self.end, name="motion end token ID")
        if len({start, placeholder, end}) != 3:
            raise MotionPlaceholderError(
                "motion start, placeholder, and end token IDs must be distinct"
            )

    @classmethod
    def from_config(cls, config: Any) -> "MotionTokenIds":
        """Build IDs from explicit model config fields."""

        return cls(
            start=_config_value(config, "motion_start_token_id"),
            placeholder=_config_value(config, "motion_placeholder_token_id"),
            end=_config_value(config, "motion_end_token_id"),
        )


@dataclass(frozen=True, slots=True)
class TextAnchor:
    start_index: int
    end_index: int


@dataclass(frozen=True, slots=True)
class MotionSpan:
    """One fully closed token span; indices include both boundary tokens."""

    start_index: int
    end_index: int
    placeholder_positions: tuple[int, ...]

    @property
    def placeholder_count(self) -> int:
        return len(self.placeholder_positions)


def find_motion_anchors(
    text: str,
    *,
    protocol: MotionTextProtocol | None = None,
) -> tuple[TextAnchor, ...]:
    """Locate literal, non-overlapping motion anchors in prompt text."""

    if not isinstance(text, str):
        raise MotionPlaceholderError("motion prompt must be a string")
    selected = protocol or MotionTextProtocol()
    anchors: list[TextAnchor] = []
    cursor = 0
    while True:
        start = text.find(selected.anchor, cursor)
        if start < 0:
            break
        end = start + len(selected.anchor)
        anchors.append(TextAnchor(start, end))
        cursor = end
    return tuple(anchors)


def render_motion_span(
    placeholder_count: int,
    *,
    protocol: MotionTextProtocol | None = None,
) -> str:
    """Render one fully bounded motion placeholder span."""

    count = _positive_count(placeholder_count, name="placeholder_count")
    selected = protocol or MotionTextProtocol()
    return selected.start + selected.placeholder * count + selected.end


def _counts(value: int | Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise MotionPlaceholderError(f"{name} must be an integer or integer sequence")
    try:
        single = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        try:
            items = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise MotionPlaceholderError(
                f"{name} must be an integer or integer sequence"
            ) from exc
    else:
        items = (single,)
    return tuple(
        _positive_count(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    )


def replace_motion_anchors(
    text: str,
    placeholder_counts: int | Sequence[int],
    *,
    protocol: MotionTextProtocol | None = None,
) -> str:
    """Replace each raw ``<motion>`` anchor with one exact bounded span.

    Existing boundary tokens are rejected to prevent accidentally expanding an
    already-expanded prompt a second time.
    """

    selected = protocol or MotionTextProtocol()
    if not isinstance(text, str):
        raise MotionPlaceholderError("motion prompt must be a string")
    if selected.start in text or selected.end in text:
        raise MotionPlaceholderError(
            "prompt already contains a motion boundary token; refusing re-expansion"
        )
    anchors = find_motion_anchors(text, protocol=selected)
    counts = _counts(placeholder_counts, name="placeholder_counts")
    if len(anchors) != len(counts):
        raise MotionPlaceholderError(
            f"motion anchor/count mismatch: {len(anchors)} anchor(s), "
            f"{len(counts)} count(s)"
        )

    parts: list[str] = []
    cursor = 0
    for anchor, count in zip(anchors, counts, strict=True):
        parts.append(text[cursor : anchor.start_index])
        parts.append(render_motion_span(count, protocol=selected))
        cursor = anchor.end_index
    parts.append(text[cursor:])
    return "".join(parts)


def parse_motion_spans(
    token_ids: Sequence[int],
    token_spec: MotionTokenIds,
    *,
    allowed_interstitial_token_ids: Sequence[int] = (),
) -> tuple[MotionSpan, ...]:
    """Parse all motion spans with a strict single-pass state machine.

    Outside a span, ordinary tokens are ignored, while stray placeholder/end
    tokens are rejected.  Inside a span only placeholders and explicitly
    whitelisted interstitial IDs are accepted.  Nested or truncated spans are
    always invalid.
    """

    if not isinstance(token_spec, MotionTokenIds):
        raise MotionPlaceholderError("token_spec must be MotionTokenIds")
    if isinstance(token_ids, (str, bytes, bytearray)):
        raise MotionPlaceholderError("token_ids must be an integer sequence")
    try:
        tokens = tuple(
            _token_id(value, name=f"token_ids[{index}]")
            for index, value in enumerate(token_ids)
        )
    except TypeError as exc:
        raise MotionPlaceholderError("token_ids must be an integer sequence") from exc
    allowed = {
        _token_id(value, name=f"allowed_interstitial_token_ids[{index}]")
        for index, value in enumerate(allowed_interstitial_token_ids)
    }
    reserved = {token_spec.start, token_spec.placeholder, token_spec.end}
    if allowed & reserved:
        raise MotionPlaceholderError(
            "allowed interstitial IDs cannot include motion protocol IDs"
        )

    spans: list[MotionSpan] = []
    open_start: int | None = None
    positions: list[int] = []
    for index, token in enumerate(tokens):
        if open_start is None:
            if token == token_spec.start:
                open_start = index
                positions = []
            elif token == token_spec.placeholder:
                raise MotionPlaceholderError(
                    f"stray motion placeholder at token index {index}"
                )
            elif token == token_spec.end:
                raise MotionPlaceholderError(
                    f"motion end token at index {index} has no open span"
                )
            continue

        if token == token_spec.start:
            raise MotionPlaceholderError(
                f"nested motion start token at index {index}"
            )
        if token == token_spec.placeholder:
            positions.append(index)
            continue
        if token == token_spec.end:
            if not positions:
                raise MotionPlaceholderError(
                    f"motion span starting at {open_start} has no placeholders"
                )
            spans.append(
                MotionSpan(
                    start_index=open_start,
                    end_index=index,
                    placeholder_positions=tuple(positions),
                )
            )
            open_start = None
            positions = []
            continue
        if token not in allowed:
            raise MotionPlaceholderError(
                f"unexpected token ID {token} inside motion span at index {index}"
            )

    if open_start is not None:
        raise MotionPlaceholderError(
            f"truncated motion span starting at token index {open_start}: "
            "missing end token"
        )
    return tuple(spans)


def validate_placeholder_counts(
    spans: Sequence[MotionSpan],
    expected_counts: int | Sequence[int],
) -> None:
    """Require one exact positive placeholder count for every motion span."""

    try:
        parsed_spans = tuple(spans)
    except TypeError as exc:
        raise MotionPlaceholderError("spans must be a MotionSpan sequence") from exc
    if any(not isinstance(span, MotionSpan) for span in parsed_spans):
        raise MotionPlaceholderError("spans must contain only MotionSpan values")
    expected = _counts(expected_counts, name="expected_counts")
    if len(parsed_spans) != len(expected):
        raise MotionPlaceholderError(
            f"motion span/count mismatch: {len(parsed_spans)} span(s), "
            f"{len(expected)} expected count(s)"
        )
    for index, (span, count) in enumerate(zip(parsed_spans, expected, strict=True)):
        if span.placeholder_count != count:
            raise MotionPlaceholderError(
                f"motion span {index} has {span.placeholder_count} placeholder(s); "
                f"expected {count}"
            )


def parse_and_validate_motion_spans(
    token_ids: Sequence[int],
    token_spec: MotionTokenIds,
    expected_counts: int | Sequence[int],
    *,
    allowed_interstitial_token_ids: Sequence[int] = (),
) -> tuple[MotionSpan, ...]:
    """Parse spans and enforce counts as one fail-closed operation."""

    spans = parse_motion_spans(
        token_ids,
        token_spec,
        allowed_interstitial_token_ids=allowed_interstitial_token_ids,
    )
    validate_placeholder_counts(spans, expected_counts)
    return spans
