"""Typed multiple-choice option and gold-answer contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .errors import GoldSyntaxError, OptionContractError


class OptionLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

    @classmethod
    def parse(cls, value: object) -> "OptionLabel":
        if not isinstance(value, str):
            raise OptionContractError("option label must be exactly A, B, C, or D")
        try:
            return cls(value)
        except ValueError as exc:
            raise OptionContractError(
                "option label must be exactly A, B, C, or D"
            ) from exc

    def __str__(self) -> str:
        return self.value


OPTION_LABELS: tuple[OptionLabel, ...] = tuple(OptionLabel)
STRICT_GOLD_PATTERN = re.compile(r"\A<answer>([A-D])</answer>\Z", re.ASCII)


def _validate_canonical_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise OptionContractError(f"{field_name} must be a string")
    if not value or not value.strip():
        raise OptionContractError(f"{field_name} must not be empty")
    if value != value.strip():
        raise OptionContractError(
            f"{field_name} must not contain leading or trailing whitespace"
        )
    if any(
        (ord(character) < 32 and character not in "\n\t")
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise OptionContractError(f"{field_name} contains a control character")
    return value


@dataclass(frozen=True, slots=True)
class Option:
    """One explicitly labelled canonical answer option."""

    label: OptionLabel
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, OptionLabel):
            raise OptionContractError("Option.label must be an OptionLabel")
        _validate_canonical_text(self.text, field_name=f"option {self.label.value}")


@dataclass(frozen=True, slots=True)
class GoldAnswer:
    """A gold answer represented without any loose parsing.

    Use :meth:`parse` at external-data boundaries.  It accepts exactly one
    case-sensitive ``<answer>[A-D]</answer>`` tag and rejects surrounding text,
    whitespace, lower-case variants, and multiple tags.
    """

    label: OptionLabel

    def __post_init__(self) -> None:
        if not isinstance(self.label, OptionLabel):
            raise GoldSyntaxError("GoldAnswer.label must be an OptionLabel")

    @classmethod
    def parse(cls, value: object) -> "GoldAnswer":
        if not isinstance(value, str):
            raise GoldSyntaxError(
                "gold must be exactly <answer>A</answer> through <answer>D</answer>"
            )
        match = STRICT_GOLD_PATTERN.fullmatch(value)
        if match is None:
            raise GoldSyntaxError(
                "gold must be exactly one case-sensitive <answer>[A-D]</answer> tag"
            )
        return cls(OptionLabel(match.group(1)))

    @classmethod
    def from_label(cls, label: OptionLabel | str) -> "GoldAnswer":
        parsed = label if isinstance(label, OptionLabel) else OptionLabel.parse(label)
        return cls(parsed)

    @property
    def tag(self) -> str:
        return f"<answer>{self.label.value}</answer>"

    def __str__(self) -> str:
        return self.tag


def parse_gold_answer(value: object) -> OptionLabel:
    """Parse an exact gold tag and return its typed label."""

    return GoldAnswer.parse(value).label


def format_gold_answer(label: OptionLabel | str) -> str:
    """Format a typed (or exact canonical) label as a gold tag."""

    return GoldAnswer.from_label(label).tag
