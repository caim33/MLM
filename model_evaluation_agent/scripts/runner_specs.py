"""Runner-facing views of the controller's immutable adapter contract.

This module deliberately owns no model or backend mapping.  Both controller
receipts and runner facades derive from ``FROZEN_ADAPTER_SPECS`` so a backend
cannot silently be renamed or reassigned on only one side.  A ``None`` backend
is intentionally unimplemented and must fail closed without producing output.
"""

from __future__ import annotations

from types import MappingProxyType

from motion_eval.adapters.catalog import FROZEN_ADAPTER_SPECS


MODEL_SPECS = MappingProxyType(
    {
        model_id: (spec.modality, spec.evaluation_mode, spec.initialization)
        for model_id, spec in FROZEN_ADAPTER_SPECS.items()
    }
)

BACKENDS = MappingProxyType(
    {
        model_id: MappingProxyType(
            {
                role: spec.backend_import_for(role)
                for role in ("finetune", "evaluation", "verifier")
            }
        )
        for model_id, spec in FROZEN_ADAPTER_SPECS.items()
    }
)

DEPENDENCIES = MappingProxyType(
    {
        model_id: spec.dependencies
        for model_id, spec in FROZEN_ADAPTER_SPECS.items()
        if spec.dependencies
    }
)


def backend_for(model_id: str, role: str) -> str | None:
    try:
        return FROZEN_ADAPTER_SPECS[model_id].backend_import_for(role)
    except KeyError as exc:
        raise ValueError(f"unknown catalog model: {model_id}") from exc


def dependencies_for(model_id: str) -> tuple[str, ...]:
    try:
        return FROZEN_ADAPTER_SPECS[model_id].dependencies
    except KeyError as exc:
        raise ValueError(f"unknown catalog model: {model_id}") from exc
