"""Prediction row contracts used by all fifteen model adapters.

These dataclasses deliberately contain no torch/numpy/model dependencies.
They validate the invariants required for fixed-denominator evaluation before
a row can be serialized.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .errors import EvaluationErrorCode

ANSWER_CHOICES: tuple[str, ...] = ("A", "B", "C", "D")
PREDICTION_SCHEMA_VERSION = "1.0"


class ContractValidationError(ValueError):
    """Raised when a candidate prediction row violates the frozen contract."""


class InputModality(str, Enum):
    VIDEO = "V"
    MOTION = "M"
    VIDEO_MOTION = "VM"
    TEXT = "T"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "InputModality":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ContractValidationError("modality must be one of V, M, VM, or T")
        try:
            return cls(value)
        except ValueError as exc:
            raise ContractValidationError(
                f"unsupported modality {value!r}; expected V, M, VM, or T"
            ) from exc


class ParseStatus(str, Enum):
    VALID = "valid"
    MISSING_ANSWER_TAG = "missing_answer_tag"
    MULTIPLE_ANSWER_TAGS = "multiple_answer_tags"
    MALFORMED_ANSWER_TAG = "malformed_answer_tag"
    NOT_ATTEMPTED = "not_attempted"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "ParseStatus":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ContractValidationError("parse_status must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            raise ContractValidationError(f"unknown parse status {value!r}") from exc


_COMPLETE_ANSWER_TAG = re.compile(r"<answer>([A-D])</answer>")
_ANSWER_LIKE_MARKER = re.compile(r"<\s*/?\s*answer\b", re.IGNORECASE)


def _parse_strict_answer_parts(output: str) -> tuple[ParseStatus, str | None, int]:
    """Lower-level strict parser shared by the row contract and public API."""

    if not isinstance(output, str):
        raise TypeError("model output must be a string")
    matches = list(_COMPLETE_ANSWER_TAG.finditer(output))
    if len(matches) > 1:
        return ParseStatus.MULTIPLE_ANSWER_TAGS, None, len(matches)
    if not matches:
        status = (
            ParseStatus.MALFORMED_ANSWER_TAG
            if _ANSWER_LIKE_MARKER.search(output)
            else ParseStatus.MISSING_ANSWER_TAG
        )
        return status, None, 0
    match = matches[0]
    residual = output[: match.start()] + output[match.end() :]
    if _ANSWER_LIKE_MARKER.search(residual):
        return ParseStatus.MALFORMED_ANSWER_TAG, None, 1
    return ParseStatus.VALID, match.group(1), 1


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _nonempty_string(value, field_name)
    if value != value.strip():
        raise ContractValidationError(
            f"{field_name} must not contain leading or trailing whitespace"
        )
    if any(
        ord(character) < 32
        or ord(character) == 127
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise ContractValidationError(f"{field_name} contains an invalid character")
    return value


def _unicode_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractValidationError(f"{field_name} contains an invalid Unicode surrogate")
    return value


def _answer_choice(value: Any, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or value not in ANSWER_CHOICES:
        suffix = " or null" if optional else ""
        raise ContractValidationError(
            f"{field_name} must be exactly one of A, B, C, D{suffix}"
        )
    return value


def _freeze_json(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{field_name} must contain only finite JSON values")
        return value
    if isinstance(value, str):
        return _unicode_string(value, field_name)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field_name} keys must all be strings")
            frozen[key] = _freeze_json(child, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(child, f"{field_name}[{index}]")
            for index, child in enumerate(value)
        )
    raise ContractValidationError(f"{field_name} must contain only finite JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _json_mapping_copy(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be a JSON object")
    frozen = _freeze_json(value, field_name)
    assert isinstance(frozen, Mapping)
    return frozen


def _normalized_scores(scores: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(scores, Mapping):
        raise ContractValidationError("scores must be an object with A/B/C/D keys")
    actual = set(scores)
    expected = set(ANSWER_CHOICES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=str)
        raise ContractValidationError(
            f"scores must contain exactly A/B/C/D; missing={missing}, extra={extra}"
        )

    normalized: dict[str, float] = {}
    for choice in ANSWER_CHOICES:
        raw = scores[choice]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ContractValidationError(f"score {choice} must be a real number")
        try:
            value = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ContractValidationError(f"score {choice} must be a finite real number") from exc
        if not math.isfinite(value):
            raise ContractValidationError(f"score {choice} must be finite")
        normalized[choice] = value
    return MappingProxyType(normalized)


def argmax_choice(scores: Mapping[str, Any]) -> str:
    """Return deterministic A/B/C/D argmax after strict score validation.

    Exact ties use canonical option order, matching Python/numpy first-argmax
    semantics without importing numpy.
    """

    normalized = _normalized_scores(scores)
    return max(ANSWER_CHOICES, key=lambda choice: normalized[choice])


@dataclass(frozen=True, kw_only=True)
class PredictionRow:
    """Fields common to generative and discriminative result rows."""

    KIND: ClassVar[str] = "base"

    batch_id: str
    model_id: str
    sample_id: str
    group_id: str
    modality: InputModality | str
    gold: str
    prediction: str | None
    correct: bool
    error_code: EvaluationErrorCode | str = EvaluationErrorCode.NONE
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported prediction schema_version {self.schema_version!r}"
            )
        for name in ("batch_id", "model_id", "sample_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        object.__setattr__(self, "modality", InputModality.coerce(self.modality))
        object.__setattr__(self, "gold", _answer_choice(self.gold, "gold"))
        object.__setattr__(
            self,
            "prediction",
            _answer_choice(self.prediction, "prediction", optional=True),
        )
        try:
            error_code = EvaluationErrorCode.coerce(self.error_code)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(str(exc)) from exc
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "metadata", _json_mapping_copy(self.metadata, "metadata"))

        if not isinstance(self.correct, bool):
            raise ContractValidationError("correct must be a boolean")

        if error_code is EvaluationErrorCode.NONE:
            if self.prediction is None:
                raise ContractValidationError("successful rows require a prediction")
            if self.error_message not in (None, ""):
                raise ContractValidationError("successful rows cannot have an error_message")
            object.__setattr__(self, "error_message", None)
            expected_correct = self.prediction == self.gold
            if self.correct is not expected_correct:
                raise ContractValidationError(
                    "correct must equal (prediction == gold) for successful rows"
                )
        else:
            if self.prediction is not None:
                raise ContractValidationError("failed rows cannot publish a prediction")
            if self.correct:
                raise ContractValidationError("failed rows must remain incorrect")
            object.__setattr__(
                self,
                "error_message",
                _unicode_string(
                    _nonempty_string(self.error_message, "error_message"),
                    "error_message",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.KIND}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Mapping):
                value = _thaw_json(value)
            result[item.name] = value
        return result


@dataclass(frozen=True, kw_only=True)
class GenerativePredictionRow(PredictionRow):
    KIND: ClassVar[str] = "generative"

    raw_output: str | None
    parse_status: ParseStatus | str

    def __post_init__(self) -> None:
        super().__post_init__()
        status = ParseStatus.coerce(self.parse_status)
        object.__setattr__(self, "parse_status", status)

        if self.raw_output is not None:
            object.__setattr__(
                self, "raw_output", _unicode_string(self.raw_output, "raw_output")
            )

        if self.raw_output is not None:
            actual_status, actual_answer, _ = _parse_strict_answer_parts(self.raw_output)
            if status is not actual_status:
                raise ContractValidationError(
                    "parse_status does not match strict re-parse of raw_output"
                )
            if status is ParseStatus.VALID and self.prediction != actual_answer:
                raise ContractValidationError(
                    "prediction does not match strict answer parsed from raw_output"
                )

        if self.error_code is EvaluationErrorCode.NONE:
            if status is not ParseStatus.VALID:
                raise ContractValidationError("successful generative rows must parse as valid")
            if self.raw_output is None:
                raise ContractValidationError("successful generative rows require raw_output")
        elif self.error_code is EvaluationErrorCode.INVALID_OUTPUT:
            if status not in {
                ParseStatus.MISSING_ANSWER_TAG,
                ParseStatus.MULTIPLE_ANSWER_TAGS,
                ParseStatus.MALFORMED_ANSWER_TAG,
            }:
                raise ContractValidationError(
                    "invalid_output rows require a failed strict parse status"
                )
            if self.raw_output is None:
                raise ContractValidationError("invalid_output rows require the raw output")
        elif status is not ParseStatus.NOT_ATTEMPTED:
            raise ContractValidationError(
                "operational errors must use parse_status=not_attempted"
            )


@dataclass(frozen=True, kw_only=True)
class DiscriminativePredictionRow(PredictionRow):
    KIND: ClassVar[str] = "discriminative"

    scores: Mapping[str, float] | None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.error_code is EvaluationErrorCode.NONE:
            if self.scores is None:
                raise ContractValidationError("successful discriminative rows require scores")
            normalized = _normalized_scores(self.scores)
            object.__setattr__(self, "scores", normalized)
            expected = argmax_choice(normalized)
            if self.prediction != expected:
                raise ContractValidationError(
                    f"prediction {self.prediction!r} does not match score argmax {expected!r}"
                )
        elif self.scores is not None:
            raise ContractValidationError("failed discriminative rows cannot publish scores")


_COMMON_FIELDS = {
    "batch_id",
    "model_id",
    "sample_id",
    "group_id",
    "modality",
    "gold",
    "prediction",
    "correct",
    "error_code",
    "error_message",
    "metadata",
    "schema_version",
}


def prediction_row_from_dict(value: Mapping[str, Any]) -> PredictionRow:
    """Validate and materialize a serialized prediction row.

    Unknown fields are rejected so misspelled provenance/error fields cannot
    silently disappear during release validation.
    """

    if not isinstance(value, Mapping):
        raise ContractValidationError("prediction row must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ContractValidationError("prediction row field names must be strings")
    candidate = dict(value)
    kind = candidate.pop("kind", None)
    if kind == GenerativePredictionRow.KIND:
        row_type: type[PredictionRow] = GenerativePredictionRow
        expected = _COMMON_FIELDS | {"raw_output", "parse_status"}
    elif kind == DiscriminativePredictionRow.KIND:
        row_type = DiscriminativePredictionRow
        expected = _COMMON_FIELDS | {"scores"}
    else:
        raise ContractValidationError(
            "prediction row kind must be 'generative' or 'discriminative'"
        )

    unknown = sorted(set(candidate) - expected)
    if unknown:
        raise ContractValidationError(f"unknown prediction row fields: {unknown}")
    missing = sorted(expected - set(candidate))
    if missing:
        raise ContractValidationError(f"missing prediction row fields: {missing}")
    try:
        return row_type(**candidate)
    except TypeError as exc:
        raise ContractValidationError(f"invalid {kind} prediction row fields: {exc}") from exc


def validate_prediction_row(value: PredictionRow | Mapping[str, Any]) -> PredictionRow:
    """Return a validated immutable row, revalidating serialized mappings."""

    if isinstance(value, PredictionRow):
        return prediction_row_from_dict(value.to_dict())
    return prediction_row_from_dict(value)
