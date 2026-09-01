"""Canonical, framework-free MotionLLM sample contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import MediaContractError, OptionContractError, SampleContractError
from .modality import Modality
from .options import GoldAnswer, OPTION_LABELS, Option, OptionLabel


def _validate_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SampleContractError(f"{field_name} must be a string")
    if not value or not value.strip():
        raise SampleContractError(f"{field_name} must not be empty")
    if value != value.strip():
        raise SampleContractError(
            f"{field_name} must not contain leading or trailing whitespace"
        )
    if any(
        ord(character) < 32
        or ord(character) == 127
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise SampleContractError(f"{field_name} contains a control character")
    return value


def _validate_question(value: object) -> str:
    if not isinstance(value, str):
        raise SampleContractError("question must be a string")
    if not value or not value.strip():
        raise SampleContractError("question must not be empty")
    if value != value.strip():
        raise SampleContractError(
            "question must not contain leading or trailing whitespace"
        )
    if any(
        (ord(character) < 32 and character not in "\n\t")
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise SampleContractError("question contains a control character")
    return value


def _freeze_json(value: Any, *, location: str = "metadata") -> Any:
    """Validate JSON-compatible metadata and recursively make it immutable."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise SampleContractError(f"{location} contains an invalid Unicode surrogate")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleContractError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise SampleContractError(f"{location} keys must be strings")
            frozen[key] = _freeze_json(child, location=f"{location}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        )
    raise SampleContractError(f"{location} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class MediaReferences:
    """Resolved media paths for one sample.

    Paths are required to be absolute because containment, symlink resolution,
    and existence checks belong to the data boundary before construction.
    """

    video: Path | None = None
    motion: Path | None = None

    def __post_init__(self) -> None:
        for field_name, value in (("video", self.video), ("motion", self.motion)):
            if value is None:
                continue
            if not isinstance(value, Path):
                raise MediaContractError(f"{field_name} reference must be a pathlib.Path")
            if not value.is_absolute():
                raise MediaContractError(f"{field_name} reference must be resolved and absolute")
            try:
                normalized = value.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise MediaContractError(
                    f"{field_name} reference cannot be resolved"
                ) from exc
            if value != normalized:
                raise MediaContractError(
                    f"{field_name} reference must be normalized and symlink-resolved"
                )


def validate_media_for_modality(
    modality: Modality, media: MediaReferences
) -> None:
    """Enforce the V/M/VM/T media matrix without fallback behavior."""

    if not isinstance(modality, Modality):
        raise MediaContractError("media validation requires a typed Modality")
    if not isinstance(media, MediaReferences):
        raise MediaContractError("media must be a MediaReferences value")

    has_video = media.video is not None
    has_motion = media.motion is not None
    expected = (modality.requires_video, modality.requires_motion)
    actual = (has_video, has_motion)
    if actual != expected:
        required = {
            Modality.VIDEO: "video only",
            Modality.MOTION: "motion only",
            Modality.VIDEO_MOTION: "both video and motion",
            Modality.TEXT: "neither video nor motion",
        }[modality]
        raise MediaContractError(
            f"modality {modality.value} requires {required}; media references do not match"
        )


@dataclass(frozen=True, slots=True)
class Sample:
    """One immutable canonical multiple-choice sample."""

    sample_id: str
    group_id: str
    modality: Modality
    question: str
    options: tuple[Option, ...]
    gold: GoldAnswer
    media: MediaReferences = field(default_factory=MediaReferences)
    rollout_id: int | str | None = None
    request_id: str | None = None
    motion_lengths: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id, field_name="sample_id")
        _validate_identifier(self.group_id, field_name="group_id")
        if not isinstance(self.modality, Modality):
            raise SampleContractError("modality must be a typed Modality")
        _validate_question(self.question)

        if not isinstance(self.options, tuple):
            if isinstance(self.options, Sequence) and not isinstance(
                self.options, (str, bytes)
            ):
                object.__setattr__(self, "options", tuple(self.options))
            else:
                raise OptionContractError("options must be a sequence of Option values")
        if len(self.options) != len(OPTION_LABELS):
            raise OptionContractError("a canonical sample must contain exactly four options")
        if any(not isinstance(option, Option) for option in self.options):
            raise OptionContractError("every option must be a typed Option")
        labels = tuple(option.label for option in self.options)
        if labels != OPTION_LABELS:
            raise OptionContractError("options must appear exactly once in A, B, C, D order")
        normalized_texts = {
            " ".join(option.text.split()).casefold() for option in self.options
        }
        if len(normalized_texts) != len(OPTION_LABELS):
            raise OptionContractError("option texts must be distinct")

        if not isinstance(self.gold, GoldAnswer):
            raise SampleContractError("gold must be a typed GoldAnswer")
        if not isinstance(self.media, MediaReferences):
            raise MediaContractError("media must be a MediaReferences value")
        validate_media_for_modality(self.modality, self.media)

        if isinstance(self.rollout_id, bool) or not (
            self.rollout_id is None or isinstance(self.rollout_id, (int, str))
        ):
            raise SampleContractError("rollout_id must be an integer, string, or null")
        if isinstance(self.rollout_id, int) and self.rollout_id < 0:
            raise SampleContractError("integer rollout_id must be non-negative")
        if isinstance(self.rollout_id, str):
            _validate_identifier(self.rollout_id, field_name="rollout_id")
        if self.request_id is not None:
            _validate_identifier(self.request_id, field_name="request_id")

        if self.motion_lengths is not None:
            if not isinstance(self.motion_lengths, tuple):
                if isinstance(self.motion_lengths, Sequence) and not isinstance(
                    self.motion_lengths, (str, bytes)
                ):
                    object.__setattr__(
                        self, "motion_lengths", tuple(self.motion_lengths)
                    )
                else:
                    raise SampleContractError(
                        "motion_lengths must be a sequence of positive integers"
                    )
            if not self.modality.requires_motion:
                raise MediaContractError(
                    "motion_lengths is forbidden when the modality has no motion input"
                )
            if not self.motion_lengths or any(
                isinstance(length, bool) or not isinstance(length, int) or length <= 0
                for length in self.motion_lengths
            ):
                raise SampleContractError(
                    "motion_lengths must contain only positive integers"
                )

        if not isinstance(self.metadata, Mapping):
            raise SampleContractError("metadata must be an object")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @property
    def branch(self) -> str:
        return self.modality.branch

    @property
    def video(self) -> Path | None:
        return self.media.video

    @property
    def motion(self) -> Path | None:
        return self.media.motion

    @property
    def gold_label(self) -> OptionLabel:
        return self.gold.label

    def option(self, label: OptionLabel | str) -> Option:
        parsed = label if isinstance(label, OptionLabel) else OptionLabel.parse(label)
        return self.options[OPTION_LABELS.index(parsed)]

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "modality": self.modality.value,
            "branch": self.branch,
            "question": self.question,
            "options": {option.label.value: option.text for option in self.options},
            "gold": self.gold.tag,
            "video": str(self.video) if self.video is not None else None,
            "motion": str(self.motion) if self.motion is not None else None,
            "rollout_id": self.rollout_id,
            "request_id": self.request_id,
            "motion_lengths": (
                list(self.motion_lengths) if self.motion_lengths is not None else None
            ),
            "metadata": _thaw_json(self.metadata),
        }
        return payload
