"""LoRA save-policy contracts independent of PEFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .freeze import FreezePolicy

MOTION_STATE_MODULES = (
    "motion_prenorm",
    "motion_proj",
    "motion_postnorm",
    "motion_boundary_embed",
)


@dataclass(frozen=True)
class LoraSavePolicy:
    preserve_motion_modules: bool = True
    require_motion_modules: bool = False
    requested_modules: tuple[str, ...] = ()


def _resolve_attr(root: Any, dotted_name: str) -> Any | None:
    current = root
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
        if current is None:
            return None
    return current


def _normalize_requested(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("modules_to_save entries must be non-empty strings")
        name = value.strip()
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _requested_is_trainable(name: str, freeze: FreezePolicy) -> bool:
    if name == "visual":
        return freeze.visual_encoder
    if name == "visual.merger":
        return freeze.visual_merger
    if name == "motion_encoder":
        return freeze.motion_encoder
    if name in {"language_model", "lm_head"}:
        return freeze.language_model if name == "language_model" else freeze.lm_head
    if name in {"motion_prenorm", "motion_proj", "motion_postnorm"}:
        return freeze.motion_adapters
    if name == "motion_boundary_embed":
        return freeze.motion_boundary
    return False


def resolve_modules_to_save(
    model: Any,
    policy: LoraSavePolicy | None = None,
    *,
    freeze_policy: FreezePolicy | None = None,
) -> tuple[str, ...]:
    """Return existing custom modules that PEFT must save beside LoRA weights."""

    policy = policy or LoraSavePolicy()
    requested = _normalize_requested(policy.requested_modules)
    if "motion_embed" in requested:
        raise ValueError(
            "motion_embed no longer exists; migrate to motion_boundary_embed and motion_proj"
        )
    if requested and freeze_policy is None:
        raise ValueError("freeze_policy is required when specifying modules_to_save")
    blocked = [
        name
        for name in requested
        if freeze_policy is not None and not _requested_is_trainable(name, freeze_policy)
    ]
    if blocked:
        raise ValueError(
            "modules_to_save would make freeze-policy blocks trainable: " + ", ".join(blocked)
        )
    missing_requested = [name for name in requested if _resolve_attr(model, name) is None]
    if missing_requested:
        raise ValueError(f"requested modules_to_save are absent: {missing_requested}")

    result = list(requested)
    found_motion: list[str] = []
    if policy.preserve_motion_modules:
        for name in MOTION_STATE_MODULES:
            permitted = (
                freeze_policy is None
                or _requested_is_trainable(name, freeze_policy)
            )
            if permitted and _resolve_attr(model, name) is not None:
                found_motion.append(name)
                if name not in result:
                    result.append(name)
    if policy.require_motion_modules and not found_motion:
        raise ValueError("motion modules were required but none are present on the model")
    return tuple(result)
