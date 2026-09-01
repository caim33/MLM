"""Canonical sample construction from strict JSONL objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from motionllm.contracts import (
    ContractError,
    GoldAnswer,
    MediaReferences,
    Modality,
    OPTION_LABELS,
    Option,
    OptionLabel,
    Sample,
)

from .errors import (
    DataContractError,
    DuplicateSampleIdError,
    JsonlLineError,
)
from .jsonl import read_jsonl
from .paths import resolve_media_path


_ALLOWED_FIELDS = {
    "sample_id",
    "group_id",
    "modality",
    "branch",
    "question",
    "options",
    "gold",
    "video",
    "motion",
    "rollout_id",
    "request_id",
    "motion_lengths",
    "metadata",
}
_REQUIRED_FIELDS = {"sample_id", "group_id", "question", "options", "gold"}


def _sample_identity(row: Mapping[str, Any]) -> str | None:
    value = row.get("sample_id")
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def _parse_modality(row: Mapping[str, Any]) -> Modality:
    has_modality = "modality" in row
    has_branch = "branch" in row
    if not has_modality and not has_branch:
        raise DataContractError("one of modality or branch is required")
    canonical = Modality.parse(row["modality"]) if has_modality else None
    legacy = Modality.from_branch(row["branch"]) if has_branch else None
    if canonical is not None and legacy is not None and canonical is not legacy:
        raise DataContractError("modality and branch disagree")
    return canonical or legacy  # type: ignore[return-value]


def _parse_options(value: Any) -> tuple[Option, ...]:
    by_label: dict[OptionLabel, Option] = {}
    if isinstance(value, Mapping):
        expected = {label.value for label in OPTION_LABELS}
        actual = set(value)
        if actual != expected:
            raise DataContractError("options object must contain exactly A, B, C, and D")
        for label in OPTION_LABELS:
            by_label[label] = Option(label, value[label.value])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise DataContractError(f"options[{index}] must be an object")
            if set(entry) != {"label", "text"}:
                raise DataContractError(
                    f"options[{index}] must contain exactly label and text"
                )
            label = OptionLabel.parse(entry["label"])
            if label in by_label:
                raise DataContractError(f"duplicate option label: {label.value}")
            by_label[label] = Option(label, entry["text"])
        if set(by_label) != set(OPTION_LABELS):
            raise DataContractError("option array must contain exactly A, B, C, and D")
    else:
        raise DataContractError("options must be an object or explicitly labelled array")
    return tuple(by_label[label] for label in OPTION_LABELS)


def _parse_media_reference(
    root: Path,
    value: Any,
    *,
    field_name: str,
    check_media_exists: bool,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
        raise DataContractError(f"{field_name} must be a path string or null")
    return resolve_media_path(root, value, must_exist=check_media_exists)


def _sample_from_row(
    row: Mapping[str, Any],
    *,
    media_root: Path,
    check_media_exists: bool,
) -> Sample:
    unknown = sorted(set(row) - _ALLOWED_FIELDS)
    if unknown:
        raise DataContractError(f"unknown fields: {', '.join(unknown)}")
    missing = sorted(_REQUIRED_FIELDS - set(row))
    if missing:
        raise DataContractError(f"missing required fields: {', '.join(missing)}")

    modality = _parse_modality(row)
    media = MediaReferences(
        video=_parse_media_reference(
            media_root,
            row.get("video"),
            field_name="video",
            check_media_exists=check_media_exists,
        ),
        motion=_parse_media_reference(
            media_root,
            row.get("motion"),
            field_name="motion",
            check_media_exists=check_media_exists,
        ),
    )
    return Sample(
        sample_id=row["sample_id"],
        group_id=row["group_id"],
        modality=modality,
        question=row["question"],
        options=_parse_options(row["options"]),
        gold=GoldAnswer.parse(row["gold"]),
        media=media,
        rollout_id=row.get("rollout_id"),
        request_id=row.get("request_id"),
        motion_lengths=row.get("motion_lengths"),
        metadata=row.get("metadata", {}),
    )


def read_samples_jsonl(
    source: str | os.PathLike[str],
    *,
    media_root: str | os.PathLike[str] | None = None,
    check_media_exists: bool = True,
) -> list[Sample]:
    """Read canonical samples, failing at the exact malformed row identity."""

    source_path = Path(source).resolve(strict=False)
    rows = read_jsonl(source_path)
    root = (
        Path(media_root).resolve(strict=False)
        if media_root is not None
        else source_path.parent
    )
    samples: list[Sample] = []
    first_lines: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = _sample_identity(row)
        try:
            sample = _sample_from_row(
                row,
                media_root=root,
                check_media_exists=check_media_exists,
            )
        except (ContractError, DataContractError, TypeError) as exc:
            raise JsonlLineError(
                source_path,
                line_number,
                str(exc),
                sample_id=sample_id,
            ) from exc
        if sample.sample_id in first_lines:
            raise DuplicateSampleIdError(
                source_path,
                line_number,
                sample.sample_id,
                first_lines[sample.sample_id],
            )
        first_lines[sample.sample_id] = line_number
        samples.append(sample)
    return samples
