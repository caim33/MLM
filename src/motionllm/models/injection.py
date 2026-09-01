"""Pure validation used before a framework adapter injects motion features."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

from motionllm.contracts import Modality

from .config import MotionResizePolicy
from .errors import MotionInjectionError


def _module_state_placements(module_name: str, module: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Enumerate every recursive parameter and buffer without exposing values."""

    if not isinstance(module_name, str) or not module_name or any(
        ord(character) < 32 for character in module_name
    ):
        raise MotionInjectionError("motion module name must be a non-empty safe path")
    if module is None:
        raise MotionInjectionError(f"motion module {module_name} is missing")

    placements: list[tuple[str, Any, Any]] = []
    seen_paths: set[str] = set()
    for accessor_name in ("named_parameters", "named_buffers"):
        accessor = getattr(module, accessor_name, None)
        if not callable(accessor):
            raise MotionInjectionError(
                f"motion module {module_name} does not expose {accessor_name}()"
            )
        try:
            values = accessor(recurse=True)
        except TypeError:
            # Framework-light test doubles may expose the same recursive view
            # without accepting PyTorch's keyword argument.
            values = accessor()
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise MotionInjectionError(
                f"motion module {module_name} returned invalid {accessor_name}() data"
            ) from exc
        for index, item in enumerate(iterator):
            if not isinstance(item, tuple) or len(item) != 2:
                raise MotionInjectionError(
                    f"motion module {module_name} {accessor_name} entry {index} is invalid"
                )
            relative_name, tensor = item
            if not isinstance(relative_name, str) or any(
                ord(character) < 32 for character in relative_name
            ):
                raise MotionInjectionError(
                    f"motion module {module_name} {accessor_name} entry {index} has an invalid path"
                )
            path = module_name if not relative_name else f"{module_name}.{relative_name}"
            if path in seen_paths:
                raise MotionInjectionError(f"motion state path is duplicated: {path}")
            seen_paths.add(path)
            if tensor is None or not hasattr(tensor, "dtype") or not hasattr(tensor, "device"):
                raise MotionInjectionError(
                    f"motion state {path} does not expose dtype and device"
                )
            placements.append((path, tensor.dtype, tensor.device))
    return tuple(placements)


def enumerate_motion_compute_placements(
    *,
    encoder_name: str,
    encoder: Any,
    motion_prenorm: Any,
    motion_proj: Any,
    motion_postnorm: Any,
    apply_postnorm: bool,
    motion_boundary_embed: Any,
) -> tuple[tuple[str, Any, Any], ...]:
    """Return all state placements used by the motion compute path.

    Parameterless modules are valid and contribute no placement. Buffer-only
    modules are fully represented. The fixed arguments prevent an adapter from
    accidentally omitting the boundary embedding or an enabled post-normalizer.
    """

    if not isinstance(apply_postnorm, bool):
        raise MotionInjectionError("apply_postnorm must be bool")
    modules: list[tuple[str, Any]] = [
        (encoder_name, encoder),
        ("motion_prenorm", motion_prenorm),
        ("motion_proj", motion_proj),
    ]
    if apply_postnorm:
        modules.append(("motion_postnorm", motion_postnorm))
    modules.append(("motion_boundary_embed", motion_boundary_embed))

    placements: list[tuple[str, Any, Any]] = []
    for module_name, module in modules:
        placements.extend(_module_state_placements(module_name, module))
    return tuple(placements)


def normalize_modalities(value: Any, *, batch_size: int) -> tuple[Modality, ...] | None:
    """Normalize explicit canonical modalities or legacy branches per row."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise MotionInjectionError("batch_size must be a positive integer")
    if value is None:
        return None
    if isinstance(value, str):
        items = (value,) * batch_size
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = tuple(value)
        if len(items) == 1 and batch_size > 1:
            items *= batch_size
        if len(items) != batch_size:
            raise MotionInjectionError(
                f"modality metadata length mismatch: expected {batch_size}, got {len(items)}"
            )
    else:
        raise MotionInjectionError("modality metadata must be a string or string sequence")

    parsed: list[Modality] = []
    for item in items:
        try:
            parsed.append(Modality.parse(item))
        except Exception:
            try:
                parsed.append(Modality.from_branch(item))
            except Exception as exc:
                raise MotionInjectionError(
                    f"invalid modality/branch value {item!r}"
                ) from exc
    return tuple(parsed)


def validate_motion_presence(
    modalities: Sequence[Modality],
    motion_present: Sequence[bool],
    *,
    prefill: bool,
) -> tuple[bool, ...]:
    """Enforce the motion half of the V/M/VM/T matrix per physical row."""

    typed = tuple(modalities)
    present = tuple(motion_present)
    if len(typed) != len(present):
        raise MotionInjectionError("modality and motion presence lengths differ")
    inject: list[bool] = []
    for index, (modality, has_motion) in enumerate(zip(typed, present, strict=True)):
        if not isinstance(modality, Modality) or not isinstance(has_motion, bool):
            raise MotionInjectionError("invalid modality or motion presence value")
        if not prefill and has_motion:
            raise MotionInjectionError(
                f"row {index} decode phase must not receive motion again"
            )
        if modality.requires_motion:
            if prefill and not has_motion:
                raise MotionInjectionError(
                    f"row {index} modality {modality.value} requires motion during prefill"
                )
            inject.append(prefill and has_motion)
        else:
            if has_motion:
                raise MotionInjectionError(
                    f"row {index} modality {modality.value} forbids motion"
                )
            inject.append(False)
    return tuple(inject)


def required_feature_length(
    actual_features: Any,
    placeholder_count: Any,
    *,
    policy: MotionResizePolicy,
) -> int:
    """Validate a feature/span length match and return the target length."""

    try:
        actual = operator.index(actual_features)
        expected = operator.index(placeholder_count)
    except TypeError as exc:
        raise MotionInjectionError("feature and placeholder counts must be integers") from exc
    if actual <= 0 or expected <= 0:
        raise MotionInjectionError("feature and placeholder counts must be positive")
    if not isinstance(policy, MotionResizePolicy):
        raise MotionInjectionError("policy must be MotionResizePolicy")
    if actual != expected and policy is MotionResizePolicy.ERROR:
        raise MotionInjectionError(
            f"motion feature/placeholder mismatch: {actual} feature(s), "
            f"{expected} placeholder(s)"
        )
    return expected


def validate_motion_segment_ownership(
    feature_counts: Sequence[Any],
    placeholder_counts: Sequence[Any],
    *,
    allow_per_segment_resize: bool = False,
) -> tuple[int, ...]:
    """Require every packed motion segment to own its exact prompt span."""

    if not isinstance(allow_per_segment_resize, bool):
        raise MotionInjectionError("allow_per_segment_resize must be bool")
    try:
        raw_features = tuple(feature_counts)
        raw_placeholders = tuple(placeholder_counts)
        if any(
            isinstance(value, bool)
            for value in raw_features + raw_placeholders
        ):
            raise TypeError("bool is not an integer count")
        features = tuple(operator.index(value) for value in raw_features)
        placeholders = tuple(operator.index(value) for value in raw_placeholders)
    except (TypeError, ValueError) as exc:
        raise MotionInjectionError(
            "motion segment feature/placeholder counts must be integer sequences"
        ) from exc
    if not features or len(features) != len(placeholders):
        raise MotionInjectionError(
            "motion segment/span count mismatch: "
            f"{len(features)} encoded segment(s), {len(placeholders)} span(s)"
        )
    if any(value <= 0 for value in features + placeholders):
        raise MotionInjectionError(
            "motion segment feature/placeholder counts must be positive"
        )
    mismatches = tuple(
        (index, feature, placeholder)
        for index, (feature, placeholder) in enumerate(
            zip(features, placeholders, strict=True)
        )
        if feature != placeholder
    )
    if mismatches and not allow_per_segment_resize:
        raise MotionInjectionError(
            "packed motion segment ownership mismatch: "
            + ", ".join(
                f"segment {index} has {feature} feature(s) but span has {placeholder} placeholder(s)"
                for index, feature, placeholder in mismatches
            )
        )
    return features


def validate_preembedded_motion_inputs(
    *,
    inputs_embeds_present: bool,
    motion_present: bool,
    motion_lengths_present: bool,
) -> None:
    """Reject ambiguous requests that would otherwise silently drop motion."""

    values = (inputs_embeds_present, motion_present, motion_lengths_present)
    if any(not isinstance(value, bool) for value in values):
        raise MotionInjectionError("input presence flags must be bool")
    if inputs_embeds_present and (motion_present or motion_lengths_present):
        raise MotionInjectionError(
            "inputs_embeds cannot be combined with raw motion or motion_lengths; "
            "supply input_ids for model-owned motion injection"
        )


def validate_motion_compute_contract(
    *,
    expected_dtype: Any,
    expected_device: Any,
    module_placements: Sequence[tuple[str, Any, Any]],
) -> None:
    """Fail before compute when a motion module violates dtype/device policy."""

    try:
        placements = tuple(module_placements)
    except TypeError as exc:
        raise MotionInjectionError("module placements must be a sequence") from exc
    if not placements:
        raise MotionInjectionError("motion compute contract has no parameters or buffers")
    for index, placement in enumerate(placements):
        if not isinstance(placement, tuple) or len(placement) != 3:
            raise MotionInjectionError(
                f"module placement {index} must be a (name, dtype, device) tuple"
            )
        name, dtype, device = placement
        if not isinstance(name, str) or not name:
            raise MotionInjectionError(f"module placement {index} has an invalid name")
        if dtype != expected_dtype:
            raise MotionInjectionError(
                f"motion module {name} dtype {dtype!r} violates expected "
                f"dtype {expected_dtype!r}"
            )
        if device != expected_device:
            raise MotionInjectionError(
                f"motion module {name} device {device!r} violates expected "
                f"device {expected_device!r}"
            )
