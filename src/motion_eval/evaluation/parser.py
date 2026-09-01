"""Fail-closed parser for main-table generative answers."""

from __future__ import annotations

from dataclasses import dataclass

from motion_eval.contracts.prediction import ParseStatus, _parse_strict_answer_parts


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    answer: str | None
    complete_tag_count: int

    @property
    def is_valid(self) -> bool:
        return self.status is ParseStatus.VALID


def parse_strict_answer(output: str) -> ParseResult:
    """Parse exactly one literal ``<answer>[A-D]</answer>`` tag.

    Text outside the answer tag is permitted for models that emit visible
    reasoning.  Lowercase choices, whitespace inside the tag, attributes,
    partial tags, nested tags, and a second answer-like tag all fail closed.
    Isolated A/B/C/D characters are never considered predictions.
    """

    if not isinstance(output, str):
        raise TypeError("model output must be a string")

    status, answer, count = _parse_strict_answer_parts(output)
    return ParseResult(status, answer, count)
