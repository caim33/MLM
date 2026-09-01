"""Parameter-level freeze policy with auditable receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FreezePolicy:
    language_model: bool = False
    lm_head: bool = False
    visual_encoder: bool = False
    visual_merger: bool = False
    motion_encoder: bool = False
    motion_adapters: bool = True
    motion_boundary: bool = True

    @classmethod
    def from_legacy_arguments(cls, arguments: Any) -> "FreezePolicy":
        return cls(
            language_model=bool(getattr(arguments, "tune_mm_llm", False)),
            lm_head=bool(getattr(arguments, "tune_mm_llm", False)),
            visual_encoder=bool(getattr(arguments, "tune_mm_vision", False)),
            visual_merger=bool(getattr(arguments, "tune_mm_mlp", False)),
            motion_encoder=bool(getattr(arguments, "tune_mm_motion", False)),
            motion_adapters=True,
            motion_boundary=True,
        )


@dataclass(frozen=True)
class ComponentFreezeReceipt:
    component: str
    present: bool
    parameter_tensors: int
    parameter_elements: int
    trainable: bool


@dataclass(frozen=True)
class FreezeReceipt:
    components: tuple[ComponentFreezeReceipt, ...]

    @property
    def trainable_parameter_tensors(self) -> int:
        return sum(item.parameter_tensors for item in self.components if item.trainable)

    def by_component(self, name: str) -> ComponentFreezeReceipt:
        for item in self.components:
            if item.component == name:
                return item
        raise KeyError(name)


def _resolve_attr(root: Any, dotted_name: str) -> Any | None:
    current = root
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
        if current is None:
            return None
    return current


def _first_present(root: Any, *dotted_names: str) -> Any | None:
    for name in dotted_names:
        value = _resolve_attr(root, name)
        if value is not None:
            return value
    return None


def _set_module(component: str, module: Any | None, trainable: bool) -> ComponentFreezeReceipt:
    if module is None:
        return ComponentFreezeReceipt(component, False, 0, 0, trainable)
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise TypeError(f"component {component} does not expose parameters()")
    tensors = list(parameters())
    elements = 0
    for parameter in tensors:
        # Deliberately update every Parameter. Setting module.requires_grad is
        # not recursive and is therefore forbidden here.
        parameter.requires_grad = bool(trainable)
        numel = getattr(parameter, "numel", None)
        elements += int(numel()) if callable(numel) else 0
    return ComponentFreezeReceipt(component, True, len(tensors), elements, trainable)


def apply_freeze_policy(model: Any, policy: FreezePolicy) -> FreezeReceipt:
    """Freeze everything first, then enable explicit components.

    The fail-closed baseline prevents a renamed or newly introduced framework
    module from becoming trainable merely because the adapter did not classify
    it yet. Specific submodules intentionally override their parents.
    """

    receipts: list[ComponentFreezeReceipt] = []
    receipts.append(_set_module("__all__", model, False))
    receipts.append(
        _set_module(
            "language_model",
            _first_present(model, "language_model", "model.language_model"),
            policy.language_model,
        )
    )
    receipts.append(_set_module("lm_head", _resolve_attr(model, "lm_head"), policy.lm_head))
    visual = _first_present(model, "visual", "model.visual")
    receipts.append(_set_module("visual", visual, policy.visual_encoder))
    visual_merger = getattr(visual, "merger", None) if visual is not None else None
    receipts.append(
        _set_module("visual.merger", visual_merger, policy.visual_merger)
    )
    receipts.append(
        _set_module("motion_encoder", _resolve_attr(model, "motion_encoder"), policy.motion_encoder)
    )
    for name in ("motion_prenorm", "motion_proj", "motion_postnorm"):
        receipts.append(_set_module(name, _resolve_attr(model, name), policy.motion_adapters))
    receipts.append(
        _set_module(
            "motion_boundary_embed",
            _resolve_attr(model, "motion_boundary_embed"),
            policy.motion_boundary,
        )
    )
    return FreezeReceipt(tuple(receipts))
