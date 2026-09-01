"""Strict canonical registry loading for the single 15-model controller."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from motion_eval.contracts import InputModality
from motion_eval.data.jsonio import load_json_strict

EXPECTED_MODEL_MATRIX: tuple[tuple[str, str, str], ...] = (
    ("qwen36_27b_lora", "V", "generative"),
    ("motionr1_vm_lora", "VM", "generative"),
    ("qwen3vl_8b_lora", "V", "generative"),
    ("qwen3vl_4b_lora", "V", "generative"),
    ("qwen35_4b_lora", "V", "generative"),
    ("videollava_7b_lora", "V", "generative"),
    ("videochatgpt_lora", "V", "generative"),
    ("videochat2_lora", "V", "generative"),
    ("videollama_trainables", "V", "generative"),
    ("videollama_lora", "V", "generative"),
    ("mplug_owl_video_lora", "V", "generative"),
    ("otter_video_lora", "V", "generative"),
    ("agcn_official", "M", "discriminative_abcd_scores"),
    ("motionclip_official", "M", "discriminative_abcd_scores"),
    ("motionllm_official", "V", "generative"),
)
EXPECTED_MODEL_IDS = tuple(item[0] for item in EXPECTED_MODEL_MATRIX)


class RegistryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    modality: InputModality
    finetune_kind: str
    evaluation_mode: str
    asset_state: str

    @property
    def prediction_kind(self) -> str:
        return "discriminative" if self.evaluation_mode == "discriminative_abcd_scores" else "generative"


@dataclass(frozen=True)
class PretrainedArtifactSpec:
    role: str
    path: str
    kind: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class CanonicalRegistry:
    registry_path: Path
    pretrained_registry_path: Path
    models: tuple[ModelSpec, ...]
    pretrained_root: str
    pretrained_artifacts: Mapping[str, tuple[PretrainedArtifactSpec, ...]]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models)

    def model(self, model_id: str) -> ModelSpec:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise RegistryValidationError(f"unknown canonical model_id: {model_id!r}")


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{name} must be a JSON object")
    return value


def load_canonical_registry(
    registry_path: str | Path, pretrained_registry_path: str | Path
) -> CanonicalRegistry:
    registry_file = Path(registry_path).resolve(strict=True)
    pretrained_file = Path(pretrained_registry_path).resolve(strict=True)
    registry = _object(load_json_strict(registry_file), "model registry")
    pretrained = _object(load_json_strict(pretrained_file), "pretrained registry")

    if registry.get("schema_version") != "1.0" or pretrained.get("schema_version") != "1.0":
        raise RegistryValidationError("both registry schema versions must be 1.0")
    if registry.get("fresh_finetune_required_per_batch") is not True:
        raise RegistryValidationError("registry must require fresh finetune for every batch")
    if registry.get("global_finetune_barrier_before_eval") is not True:
        raise RegistryValidationError("registry must require the global finetune barrier")
    policy = _object(pretrained.get("policy"), "pretrained policy")
    if policy.get("fresh_finetune_required_per_batch") is not True:
        raise RegistryValidationError("pretrained policy cannot waive fresh finetune")
    if policy.get("historical_finetune_artifacts_are_pretrain") is not False:
        raise RegistryValidationError("historical finetune artifacts cannot be pretrained inputs")

    rows = registry.get("models")
    pre_rows = pretrained.get("models")
    if not isinstance(rows, list) or not isinstance(pre_rows, list):
        raise RegistryValidationError("registry model lists are required")
    if len(rows) != 15 or len(pre_rows) != 15:
        raise RegistryValidationError("canonical coverage must contain exactly 15 models")
    row_ids = [row.get("id") if isinstance(row, Mapping) else None for row in rows]
    pre_ids = [row.get("id") if isinstance(row, Mapping) else None for row in pre_rows]
    if tuple(row_ids) != EXPECTED_MODEL_IDS:
        raise RegistryValidationError("model registry IDs/order differ from canonical 15-model matrix")
    if tuple(pre_ids) != EXPECTED_MODEL_IDS:
        raise RegistryValidationError("pretrained registry IDs/order differ from canonical matrix")
    if len(set(row_ids)) != 15 or len(set(pre_ids)) != 15:
        raise RegistryValidationError("registry model IDs must be unique")

    models: list[ModelSpec] = []
    for row, expected in zip(rows, EXPECTED_MODEL_MATRIX, strict=True):
        row = _object(row, f"model {expected[0]}")
        model_id, modality, mode = expected
        if row.get("main_modality") != modality:
            raise RegistryValidationError(f"{model_id} modality must remain {modality}")
        if row.get("evaluation_mode") != mode:
            raise RegistryValidationError(f"{model_id} evaluation_mode must remain {mode}")
        for field in ("display_name", "finetune_kind", "current_asset_state"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise RegistryValidationError(f"{model_id}.{field} must be non-empty")
        models.append(
            ModelSpec(
                model_id=model_id,
                display_name=row["display_name"],
                modality=InputModality.coerce(modality),
                finetune_kind=row["finetune_kind"],
                evaluation_mode=mode,
                asset_state=row["current_asset_state"],
            )
        )

    by_pretrain = {row["id"]: _object(row, f"pretrained {row['id']}") for row in pre_rows}
    agcn_roles = {item.get("role") for item in by_pretrain["agcn_official"].get("artifacts", [])}
    if agcn_roles != {"official_source"}:
        raise RegistryValidationError("AGCN must use only its pinned official source and fresh initialization")
    motionclip_roles = {
        item.get("role") for item in by_pretrain["motionclip_official"].get("artifacts", [])
    }
    if motionclip_roles != {"official_source", "pretrained_motionclip"}:
        raise RegistryValidationError("MotionCLIP must use official source and paper checkpoint")
    artifact_matrix: dict[str, tuple[PretrainedArtifactSpec, ...]] = {}
    for model_id in EXPECTED_MODEL_IDS:
        artifacts = by_pretrain[model_id].get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RegistryValidationError(f"{model_id} has no registered pretrained artifacts")
        parsed: list[PretrainedArtifactSpec] = []
        seen_roles: set[str] = set()
        for artifact in artifacts:
            artifact = _object(artifact, f"{model_id} pretrained artifact")
            role, path, kind = artifact.get("role"), artifact.get("path"), artifact.get("kind")
            if not all(isinstance(item, str) and item.strip() for item in (role, path, kind)):
                raise RegistryValidationError(f"{model_id} has an invalid pretrained artifact")
            normalized_path = Path(path.replace("/", os.sep))
            if normalized_path.is_absolute() or ".." in normalized_path.parts:
                raise RegistryValidationError(
                    f"{model_id}.{role} pretrained path must be relative and traversal-free"
                )
            expected_sha256 = artifact.get("sha256")
            if expected_sha256 is not None and (
                not isinstance(expected_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            ):
                raise RegistryValidationError(
                    f"{model_id}.{role} has an invalid expected SHA-256"
                )
            if role in seen_roles:
                raise RegistryValidationError(f"{model_id} repeats pretrained role {role!r}")
            seen_roles.add(role)
            parsed.append(PretrainedArtifactSpec(role, path, kind, expected_sha256))
        artifact_matrix[model_id] = tuple(parsed)
    return CanonicalRegistry(
        registry_path=registry_file,
        pretrained_registry_path=pretrained_file,
        models=tuple(models),
        pretrained_root=str(pretrained.get("remote_root", "")),
        pretrained_artifacts=artifact_matrix,
    )
