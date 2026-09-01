"""Framework-neutral logical ownership planning for model collators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

from motionllm.contracts import Modality

from .errors import CollationContractError


@dataclass(frozen=True, slots=True)
class CollationPlan:
    motions: tuple[Any | None, ...]
    motion_lengths: tuple[tuple[int, ...] | None, ...]
    physical_branches: tuple[str, ...]
    packed_sample_ids: tuple[str, ...]
    packed_group_ids: tuple[str, ...]
    packed_branches: tuple[str, ...]
    motion_owner_indices: tuple[int, ...]


def _identity(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CollationContractError(f"{field_name} must be a non-empty canonical string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CollationContractError(f"{field_name} contains a control character")
    return value


def _positive_length(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise CollationContractError(f"{field_name} must be a positive integer")
    return int(value)


def _modality(value: Mapping[str, Any]) -> Modality:
    canonical: Modality | None = None
    legacy: Modality | None = None
    try:
        if "modality" in value:
            raw = value["modality"]
            canonical = raw if isinstance(raw, Modality) else Modality.parse(raw)
        if "branch" in value:
            raw = value["branch"]
            legacy = raw if isinstance(raw, Modality) else Modality.from_branch(raw)
    except ValueError as exc:
        raise CollationContractError(str(exc)) from exc
    if canonical is None and legacy is None:
        raise CollationContractError("logical sample requires modality or branch")
    if canonical is not None and legacy is not None and canonical is not legacy:
        raise CollationContractError("modality and branch disagree")
    return canonical or legacy  # type: ignore[return-value]


def logical_sample_payload(
    *,
    sample_id: str,
    group_id: str,
    modality: Modality,
    motion_length: int | None,
) -> dict[str, Any]:
    """Create validated logical identity metadata for one packed sample."""

    if not isinstance(modality, Modality):
        raise CollationContractError("modality must be a typed Modality")
    parsed_length = None
    if modality.requires_motion:
        parsed_length = _positive_length(motion_length, "motion_length")
    elif motion_length is not None:
        raise CollationContractError("motion presence and modality disagree")
    return {
        "sample_id": _identity(sample_id, "sample_id"),
        "group_id": _identity(group_id, "group_id"),
        "branch": modality.branch,
        "motion_length": parsed_length,
    }


def _motion_extent(motion: Any) -> int:
    shape = getattr(motion, "shape", None)
    try:
        extent = shape[0] if shape is not None else len(motion)
    except (TypeError, IndexError) as exc:
        raise CollationContractError("motion must expose a leading frame dimension") from exc
    return _positive_length(extent, "motion frame dimension")


def _lengths(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return (_positive_length(value, "motion_lengths"),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CollationContractError("motion_lengths must be a sequence")
    if not value:
        raise CollationContractError("motion_lengths must not be empty")
    return tuple(
        _positive_length(length, f"motion_lengths[{index}]")
        for index, length in enumerate(value)
    )


def _physical_modality(logical_modalities: Sequence[Modality]) -> Modality:
    has_video = any(value.requires_video for value in logical_modalities)
    has_motion = any(value.requires_motion for value in logical_modalities)
    return {
        (True, False): Modality.VIDEO,
        (False, True): Modality.MOTION,
        (True, True): Modality.VIDEO_MOTION,
        (False, False): Modality.TEXT,
    }[(has_video, has_motion)]


def _logical_rows(instance: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = instance.get("logical_samples")
    if raw is None:
        return (instance,)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise CollationContractError("logical_samples must be a non-empty sequence")
    if any(not isinstance(row, Mapping) for row in raw):
        raise CollationContractError("every logical_samples entry must be an object")
    return tuple(raw)


def plan_collation(instances: Sequence[Mapping[str, Any]]) -> CollationPlan:
    """Validate a physical batch and preserve every logical sample owner."""

    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
        raise CollationContractError("instances must be a sequence")
    if not instances:
        raise CollationContractError("instances must not be empty")

    motions: list[Any | None] = []
    physical_lengths: list[tuple[int, ...] | None] = []
    physical_branches: list[str] = []
    sample_ids: list[str] = []
    group_ids: list[str] = []
    branches: list[str] = []
    owners: list[int] = []
    seen_ids: set[str] = set()

    for physical_index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise CollationContractError(f"instance {physical_index} must be an object")
        logical_rows = _logical_rows(instance)
        logical_modalities: list[Modality] = []
        expected_lengths: list[int] = []
        local_identities: list[tuple[str, str]] = []

        for row in logical_rows:
            modality = _modality(row)
            sample_id = _identity(row.get("sample_id"), "sample_id")
            group_id = _identity(row.get("group_id"), "group_id")
            if sample_id in seen_ids:
                raise CollationContractError(f"duplicate sample_id in batch: {sample_id}")
            seen_ids.add(sample_id)
            logical_modalities.append(modality)
            local_identities.append((sample_id, group_id))
            motion_length = row.get("motion_length")
            if modality.requires_motion:
                if motion_length is None and row is instance:
                    direct_lengths = _lengths(instance.get("motion_lengths"))
                    if direct_lengths is not None and len(direct_lengths) == 1:
                        motion_length = direct_lengths[0]
                    else:
                        raise CollationContractError(
                            "motion presence and modality disagree"
                        )
                expected_lengths.append(_positive_length(motion_length, "motion_length"))
                owners.append(len(sample_ids))
            elif motion_length is not None:
                raise CollationContractError("motion presence and modality disagree")
            sample_ids.append(sample_id)
            group_ids.append(group_id)
            branches.append(modality.branch)

        derived_physical = _physical_modality(logical_modalities)
        if "branch" in instance or "modality" in instance:
            declared_physical = _modality(instance)
            if declared_physical is not derived_physical:
                raise CollationContractError("physical and logical modality disagree")
        physical_branches.append(derived_physical.branch)

        motion = instance.get("motion")
        has_motion = motion is not None
        if has_motion != bool(expected_lengths):
            raise CollationContractError("motion presence and modality disagree")
        declared_lengths = _lengths(instance.get("motion_lengths"))
        if expected_lengths:
            expected_tuple = tuple(expected_lengths)
            if declared_lengths != expected_tuple:
                raise CollationContractError(
                    "motion_lengths disagree with the logical motion-owned rows"
                )
            if _motion_extent(motion) != sum(expected_tuple):
                raise CollationContractError(
                    "motion tensor extent does not equal lengths for motion-owned rows"
                )
            physical_lengths.append(expected_tuple)
        else:
            if declared_lengths is not None:
                raise CollationContractError("motion lengths and modality disagree")
            physical_lengths.append(None)
        motions.append(motion)

    return CollationPlan(
        motions=tuple(motions),
        motion_lengths=tuple(physical_lengths),
        physical_branches=tuple(physical_branches),
        packed_sample_ids=tuple(sample_ids),
        packed_group_ids=tuple(group_ids),
        packed_branches=tuple(branches),
        motion_owner_indices=tuple(owners),
    )
