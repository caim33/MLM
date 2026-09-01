"""Verified fresh-finetune provenance, artifacts and reload receipts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from motion_eval.core import (
    atomic_write_json,
    hash_path,
    resolve_within_root,
    sha256_file,
    sha256_json,
)
from motion_eval.training_receipt import (
    load_and_validate_formal_provenance_snapshot,
    load_and_validate_training_receipt,
)

from .tokens import verify_motion_tokens

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED_PROVENANCE_ROLES = (
    "base_artifact",
    "train_data",
    "validation_data",
    "benchmark",
    "leakage_audit",
    "config",
    "code",
    "environment",
)
_OPTIONAL_PROVENANCE_ROLES = ("runner_code", "motion_vqvae")
_PROVENANCE_ROLES = _REQUIRED_PROVENANCE_ROLES + _OPTIONAL_PROVENANCE_ROLES
_EVIDENCE_KEYS = frozenset(
    {"path", "algorithm", "kind", "digest", "file_count", "total_bytes"}
)
_MANIFEST_V2_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at",
        "batch_id",
        "model_id",
        "training_mode",
        "provenance",
        "artifact",
        "reload_verification",
    }
)
_MANIFEST_V3_KEYS = frozenset(
    set(_MANIFEST_V2_KEYS)
    | {"training_verification", "manifest_sha256"}
)
_RELOAD_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "batch_id",
        "model_id",
        "artifact_hash",
        "expected_modules",
        "reloaded_modules",
        "motion_start_token_id",
        "motion_end_token_id",
        "state_hash_before",
        "state_hash_after",
        "processor_state_hash_before",
        "processor_state_hash_after",
        "processor_assets_hash",
    }
)


class ArtifactValidationError(ValueError):
    pass


def _validate_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ArtifactValidationError(f"{name} is not a safe non-empty identifier")
    return value


def _validate_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_training_mode(value: str) -> str:
    if value not in {"full_sft", "lora_sft", "grpo"}:
        raise ArtifactValidationError("training_mode must be full_sft, lora_sft, or grpo")
    return value


@dataclass(frozen=True)
class ArtifactProvenancePaths:
    base_artifact: str | Path
    train_data: str | Path
    validation_data: str | Path
    benchmark: str | Path
    leakage_audit: str | Path
    config: str | Path
    code: str | Path
    environment: str | Path
    runner_code: str | Path | None = None
    motion_vqvae: str | Path | None = None

    def to_dict(self) -> dict[str, str | Path]:
        result: dict[str, str | Path] = {
            role: getattr(self, role) for role in _REQUIRED_PROVENANCE_ROLES
        }
        for role in _OPTIONAL_PROVENANCE_ROLES:
            value = getattr(self, role)
            if value not in (None, ""):
                result[role] = value
        return result


@dataclass(frozen=True)
class ArtifactBinding:
    batch_id: str
    model_id: str
    training_mode: str
    base_artifact_hash: str
    train_data_hash: str
    validation_data_hash: str
    benchmark_hash: str
    leakage_audit_hash: str
    config_hash: str
    code_hash: str
    environment_hash: str
    runner_code_hash: str | None = None
    motion_vqvae_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier("batch_id", self.batch_id)
        _validate_identifier("model_id", self.model_id)
        _validate_training_mode(self.training_mode)
        for name in (
            "base_artifact_hash",
            "train_data_hash",
            "validation_data_hash",
            "benchmark_hash",
            "leakage_audit_hash",
            "config_hash",
            "code_hash",
            "environment_hash",
        ):
            _validate_digest(name, getattr(self, name))
        if self.motion_vqvae_hash is not None:
            _validate_digest("motion_vqvae_hash", self.motion_vqvae_hash)
        if self.runner_code_hash is not None:
            _validate_digest("runner_code_hash", self.runner_code_hash)


@dataclass(frozen=True)
class FinetuneArtifactReceipt:
    manifest_path: Path
    artifact_path: Path
    artifact_digest: str
    artifact_kind: str
    manifest_digest: str
    training_receipt_digest: str | None = None


@dataclass(frozen=True)
class ReloadVerificationReceipt:
    batch_id: str
    model_id: str
    artifact_hash: str
    expected_modules: tuple[str, ...]
    reloaded_modules: tuple[str, ...]
    motion_start_token_id: int | None
    motion_end_token_id: int | None
    state_hash_before: str
    state_hash_after: str
    processor_state_hash_before: str
    processor_state_hash_after: str
    processor_assets_hash: str

    def __post_init__(self) -> None:
        _validate_identifier("batch_id", self.batch_id)
        _validate_identifier("model_id", self.model_id)
        for name in (
            "artifact_hash",
            "state_hash_before",
            "state_hash_after",
            "processor_state_hash_before",
            "processor_state_hash_after",
            "processor_assets_hash",
        ):
            _validate_digest(name, getattr(self, name))
        if (
            len(set(self.expected_modules)) != len(self.expected_modules)
            or any(not isinstance(name, str) or not name for name in self.expected_modules)
        ):
            raise ArtifactValidationError("expected reload modules must be unique safe names")
        if tuple(self.expected_modules) != tuple(self.reloaded_modules):
            raise ArtifactValidationError(
                "reloaded modules must exactly equal the expected modules_to_save/state scope"
            )
        if self.state_hash_before != self.state_hash_after:
            raise ArtifactValidationError("save/reload state hashes differ")
        if self.processor_state_hash_before != self.processor_state_hash_after:
            raise ArtifactValidationError("save/reload processor state hashes differ")
        token_ids = (self.motion_start_token_id, self.motion_end_token_id)
        if (token_ids[0] is None) != (token_ids[1] is None):
            raise ArtifactValidationError(
                "motion token ids must both be present or both be null"
            )
        if token_ids[0] is not None:
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            ):
                raise ArtifactValidationError(
                    "motion token ids must be non-negative integers"
                )
            if not self.expected_modules:
                raise ArtifactValidationError(
                    "motion reload receipts require non-empty expected modules"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "status": "reload_verified",
            "batch_id": self.batch_id,
            "model_id": self.model_id,
            "artifact_hash": self.artifact_hash,
            "expected_modules": list(self.expected_modules),
            "reloaded_modules": list(self.reloaded_modules),
            "motion_start_token_id": self.motion_start_token_id,
            "motion_end_token_id": self.motion_end_token_id,
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "processor_state_hash_before": self.processor_state_hash_before,
            "processor_state_hash_after": self.processor_state_hash_after,
            "processor_assets_hash": self.processor_assets_hash,
        }


@dataclass(frozen=True)
class NamedStateSnapshot:
    """Compact exact hashes for a named parameter/state selection."""

    sha256: str
    tensor_count: int
    parameter_count: int
    tensor_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_digest("state snapshot sha256", self.sha256)
        if self.tensor_count <= 0 or self.parameter_count <= 0:
            raise ArtifactValidationError("state snapshot must be non-empty")
        if len(self.tensor_sha256) != self.tensor_count:
            raise ArtifactValidationError("state snapshot tensor count is inconsistent")
        for name, digest in self.tensor_sha256.items():
            if not isinstance(name, str) or not name:
                raise ArtifactValidationError("state snapshot names must be non-empty")
            _validate_digest(f"state snapshot {name}", digest)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ArtifactValidationError(f"non-finite JSON constant: {token}")
            ),
        )
    except UnicodeError as exc:
        raise ArtifactValidationError(f"JSON is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON root must be an object: {path}")
    return value


def _hash_nonempty(path: str | Path, *, role: str) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    digest = hash_path(candidate, symlink_policy="follow")
    if digest.file_count <= 0 or digest.total_bytes <= 0:
        raise ArtifactValidationError(
            f"{role} must contain at least one non-empty regular file; "
            f"file_count={digest.file_count}, total_bytes={digest.total_bytes}"
        )
    return {"path": str(candidate), **digest.to_dict()}


def compute_verified_provenance(
    paths: ArtifactProvenancePaths,
    *,
    batch_id: str,
    model_id: str,
    training_mode: str,
) -> tuple[ArtifactBinding, dict[str, dict[str, Any]]]:
    """Recompute every formal provenance digest from an explicit existing path."""

    _validate_identifier("batch_id", batch_id)
    _validate_identifier("model_id", model_id)
    _validate_training_mode(training_mode)
    evidence = {
        role: _hash_nonempty(path, role=role) for role, path in paths.to_dict().items()
    }
    binding = ArtifactBinding(
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        base_artifact_hash=evidence["base_artifact"]["digest"],
        train_data_hash=evidence["train_data"]["digest"],
        validation_data_hash=evidence["validation_data"]["digest"],
        benchmark_hash=evidence["benchmark"]["digest"],
        leakage_audit_hash=evidence["leakage_audit"]["digest"],
        config_hash=evidence["config"]["digest"],
        code_hash=evidence["code"]["digest"],
        environment_hash=evidence["environment"]["digest"],
        runner_code_hash=(
            evidence["runner_code"]["digest"] if "runner_code" in evidence else None
        ),
        motion_vqvae_hash=(
            evidence["motion_vqvae"]["digest"] if "motion_vqvae" in evidence else None
        ),
    )
    return binding, evidence


def binding_from_provenance_evidence(
    paths: ArtifactProvenancePaths,
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    batch_id: str,
    model_id: str,
    training_mode: str,
) -> ArtifactBinding:
    """Validate a previously captured immutable provenance snapshot.

    This deliberately does not claim to re-read the underlying content.  The
    caller must perform its own pre/post content verification before passing a
    snapshot here.  It only proves that the exact evidence roles and canonical
    paths are the ones being published.
    """

    _validate_identifier("batch_id", batch_id)
    _validate_identifier("model_id", model_id)
    _validate_training_mode(training_mode)
    expected_paths = paths.to_dict()
    if not isinstance(evidence, Mapping) or set(evidence) != set(expected_paths):
        raise ArtifactValidationError("precomputed provenance roles are incomplete")
    checked: dict[str, Mapping[str, Any]] = {}
    for role, raw_path in expected_paths.items():
        value = _validate_evidence_schema(evidence[role], name=role)
        canonical_path = Path(raw_path).resolve(strict=True)
        if value["path"] != str(canonical_path):
            raise ArtifactValidationError(
                f"precomputed provenance path differs for {role}"
            )
        checked[role] = value
    return ArtifactBinding(
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        base_artifact_hash=checked["base_artifact"]["digest"],
        train_data_hash=checked["train_data"]["digest"],
        validation_data_hash=checked["validation_data"]["digest"],
        benchmark_hash=checked["benchmark"]["digest"],
        leakage_audit_hash=checked["leakage_audit"]["digest"],
        config_hash=checked["config"]["digest"],
        code_hash=checked["code"]["digest"],
        environment_hash=checked["environment"]["digest"],
        runner_code_hash=(
            checked["runner_code"]["digest"] if "runner_code" in checked else None
        ),
        motion_vqvae_hash=(
            checked["motion_vqvae"]["digest"]
            if "motion_vqvae" in checked
            else None
        ),
    )


def _validate_evidence_schema(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _EVIDENCE_KEYS:
        raise ArtifactValidationError(f"{name} evidence has an invalid schema")
    _validate_digest(f"{name}.digest", value.get("digest"))
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ArtifactValidationError(f"{name}.path must be non-empty")
    for count_name in ("file_count", "total_bytes"):
        count = value.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ArtifactValidationError(f"{name}.{count_name} must be positive")
    return value


def _validate_reload_receipt(
    path: Path,
    *,
    batch_id: str,
    model_id: str,
    artifact_hash: str,
) -> Mapping[str, Any]:
    value = _read_json(path)
    if set(value) != _RELOAD_KEYS:
        raise ArtifactValidationError("reload receipt schema mismatch")
    if value.get("schema_version") != "2.0" or value.get("status") != "reload_verified":
        raise ArtifactValidationError("reload receipt is not verified v2 evidence")
    if value.get("batch_id") != batch_id or value.get("model_id") != model_id:
        raise ArtifactValidationError("reload receipt batch/model binding mismatch")
    if value.get("artifact_hash") != artifact_hash:
        raise ArtifactValidationError("reload receipt artifact hash mismatch")
    ReloadVerificationReceipt(
        batch_id=value["batch_id"],
        model_id=value["model_id"],
        artifact_hash=value["artifact_hash"],
        expected_modules=tuple(value["expected_modules"]),
        reloaded_modules=tuple(value["reloaded_modules"]),
        motion_start_token_id=value["motion_start_token_id"],
        motion_end_token_id=value["motion_end_token_id"],
        state_hash_before=value["state_hash_before"],
        state_hash_after=value["state_hash_after"],
        processor_state_hash_before=value["processor_state_hash_before"],
        processor_state_hash_after=value["processor_state_hash_after"],
        processor_assets_hash=value["processor_assets_hash"],
    )
    return value


def _validate_provenance_snapshot_reference(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    artifact: Path,
    provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    if receipt.get("schema_version") != "2.0":
        # Legacy schema-1 receipts have no snapshot reference to parse.  The
        # production Qwen entrypoints are independently fail-closed and cannot
        # emit/promote these as current formal artifacts.
        return
    snapshot = resolve_within_root(
        receipt.get("provenance_snapshot_path"),
        root,
        must_exist=True,
        allow_root=False,
    )
    if not snapshot.is_file() or snapshot == artifact or artifact in snapshot.parents:
        raise ArtifactValidationError(
            "formal provenance snapshot must be a file outside the artifact"
        )
    try:
        payload = load_and_validate_formal_provenance_snapshot(
            snapshot,
            expected_file_sha256=receipt.get("provenance_snapshot_file_sha256"),
            expected={
                "batch_id": receipt.get("batch_id"),
                "model_id": receipt.get("model_id"),
                "training_mode": receipt.get("training_mode"),
                "snapshot_sha256": receipt.get("provenance_pre_sha256"),
            },
        )
    except ValueError as exc:
        raise ArtifactValidationError(
            f"formal provenance snapshot is invalid: {exc}"
        ) from exc
    if (
        payload["snapshot_sha256"] != receipt.get("provenance_post_sha256")
        or receipt.get("provenance_unchanged") is not True
    ):
        raise ArtifactValidationError(
            "formal provenance snapshot does not prove identical pre/post generations"
        )
    snapshot_provenance = payload["provenance"]
    if dict(snapshot_provenance) != {
        role: dict(value) for role, value in provenance.items()
    }:
        raise ArtifactValidationError(
            "formal provenance snapshot roles differ from the artifact manifest"
        )
    digest_bindings = {
        "base_artifact": "base_artifact_sha256",
        "train_data": "train_sha256",
        "validation_data": "validation_sha256",
        "leakage_audit": "leakage_audit_sha256",
        "config": "config_sha256",
        "code": "code_sha256",
        "runner_code": "runner_code_sha256",
        "environment": "environment_sha256",
    }
    mismatches = [
        role
        for role, receipt_field in digest_bindings.items()
        if snapshot_provenance.get(role, {}).get("digest")
        != receipt.get(receipt_field)
    ]
    if mismatches:
        raise ArtifactValidationError(
            f"formal provenance snapshot differs from training receipt: {sorted(mismatches)}"
        )


def write_finetune_artifact_manifest(
    manifest_path: str | Path,
    *,
    artifact_path: str | Path,
    provenance_paths: ArtifactProvenancePaths,
    batch_id: str,
    model_id: str,
    training_mode: str,
    allowed_root: str | Path,
    reload_receipt_path: str | Path | None = None,
    training_receipt_path: str | Path | None = None,
    provenance_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> FinetuneArtifactReceipt:
    """Recompute all evidence and atomically publish a formal manifest."""

    root = Path(allowed_root).resolve(strict=True)
    artifact = resolve_within_root(artifact_path, root, must_exist=True, allow_root=False)
    manifest = resolve_within_root(manifest_path, root, must_exist=False, allow_root=False)
    if manifest == artifact or artifact in manifest.parents:
        raise ArtifactValidationError("manifest cannot be inside the artifact being hashed")
    if provenance_evidence is None:
        _, provenance = compute_verified_provenance(
            provenance_paths,
            batch_id=batch_id,
            model_id=model_id,
            training_mode=training_mode,
        )
    else:
        binding_from_provenance_evidence(
            provenance_paths,
            provenance_evidence,
            batch_id=batch_id,
            model_id=model_id,
            training_mode=training_mode,
        )
        provenance = {
            role: dict(value) for role, value in provenance_evidence.items()
        }
    if training_mode in {"full_sft", "lora_sft"} and "runner_code" not in provenance:
        raise ArtifactValidationError(
            f"{training_mode} formal artifacts require independent runner_code provenance"
        )
    artifact_info = _hash_nonempty(artifact, role="artifact")

    reload_info: dict[str, Any] | None = None
    requires_reload = training_mode in {"full_sft", "lora_sft", "grpo"}
    if reload_receipt_path in (None, ""):
        if requires_reload:
            raise ArtifactValidationError(
                f"{training_mode} formal artifacts require a verified reload receipt"
            )
    else:
        reload_path = resolve_within_root(
            reload_receipt_path, root, must_exist=True, allow_root=False
        )
        if reload_path == artifact or artifact in reload_path.parents:
            raise ArtifactValidationError(
                "reload receipt cannot be inside the artifact whose digest it attests"
            )
        if not reload_path.is_file():
            raise ArtifactValidationError("reload receipt must be a regular file")
        _validate_reload_receipt(
            reload_path,
            batch_id=batch_id,
            model_id=model_id,
            artifact_hash=artifact_info["digest"],
        )
        reload_info = {
            "path": str(reload_path),
            "digest": sha256_file(reload_path),
        }

    training_info: dict[str, Any] | None = None
    training_receipt_digest: str | None = None
    requires_training_receipt = training_mode in {"full_sft", "lora_sft"}
    if training_receipt_path in (None, ""):
        if requires_training_receipt:
            raise ArtifactValidationError(
                f"{training_mode} formal artifacts require a training receipt"
            )
    else:
        training_path = resolve_within_root(
            training_receipt_path, root, must_exist=True, allow_root=False
        )
        if training_path == artifact or artifact in training_path.parents:
            raise ArtifactValidationError(
                "training receipt cannot be inside the artifact whose digest it attests"
            )
        if not training_path.is_file():
            raise ArtifactValidationError("training receipt must be a regular file")
        try:
            training_receipt = load_and_validate_training_receipt(
                training_path,
                expected={
                    "batch_id": batch_id,
                    "model_id": model_id,
                    "training_mode": training_mode,
                    "base_artifact_sha256": provenance["base_artifact"]["digest"],
                    "train_sha256": provenance["train_data"]["digest"],
                    "validation_sha256": provenance["validation_data"]["digest"],
                    "leakage_audit_sha256": provenance["leakage_audit"]["digest"],
                    "config_sha256": provenance["config"]["digest"],
                    "code_sha256": provenance["code"]["digest"],
                    "runner_code_sha256": provenance["runner_code"]["digest"],
                    "environment_sha256": provenance["environment"]["digest"],
                    "artifact_sha256": artifact_info["digest"],
                },
            )
        except ValueError as exc:
            raise ArtifactValidationError(f"invalid training receipt: {exc}") from exc
        _validate_provenance_snapshot_reference(
            training_receipt,
            root=root,
            artifact=artifact,
            provenance=provenance,
        )
        training_receipt_digest = training_receipt["receipt_sha256"]
        training_info = {
            "path": str(training_path),
            "digest": sha256_file(training_path),
            "content_sha256": training_receipt_digest,
        }

    payload: dict[str, Any] = {
        "schema_version": "3.0" if requires_training_receipt else "2.0",
        "status": "finetune_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "model_id": model_id,
        "training_mode": training_mode,
        "provenance": provenance,
        "artifact": artifact_info,
        "reload_verification": reload_info,
    }
    if requires_training_receipt:
        payload["training_verification"] = training_info
        payload["manifest_sha256"] = sha256_json(payload)
    atomic_write_json(manifest, payload, root=root, overwrite=overwrite)
    return FinetuneArtifactReceipt(
        manifest_path=manifest,
        artifact_path=artifact,
        artifact_digest=artifact_info["digest"],
        artifact_kind=artifact_info["kind"],
        manifest_digest=sha256_file(manifest),
        training_receipt_digest=training_receipt_digest,
    )


def validate_resume_artifact(
    manifest_path: str | Path,
    *,
    provenance_paths: ArtifactProvenancePaths,
    batch_id: str,
    model_id: str,
    training_mode: str,
    allowed_root: str | Path,
    provenance_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> FinetuneArtifactReceipt:
    """Recompute provenance and artifact evidence before accepting a resume."""

    root = Path(allowed_root).resolve(strict=True)
    manifest = resolve_within_root(manifest_path, root, must_exist=True, allow_root=False)
    if not manifest.is_file():
        raise ArtifactValidationError("resume manifest must be a regular file")
    value = _read_json(manifest)
    requires_training_receipt = training_mode in {"full_sft", "lora_sft"}
    expected_keys = _MANIFEST_V3_KEYS if requires_training_receipt else _MANIFEST_V2_KEYS
    expected_version = "3.0" if requires_training_receipt else "2.0"
    if set(value) != expected_keys:
        raise ArtifactValidationError("artifact manifest schema mismatch")
    if (
        value.get("schema_version") != expected_version
        or value.get("status") != "finetune_complete"
    ):
        raise ArtifactValidationError(
            f"resume manifest is not a completed v{expected_version} finetune artifact"
        )
    if requires_training_receipt:
        manifest_body = {
            key: item for key, item in value.items() if key != "manifest_sha256"
        }
        if value.get("manifest_sha256") != sha256_json(manifest_body):
            raise ArtifactValidationError("artifact manifest self-hash mismatch")
    if (
        value.get("batch_id") != batch_id
        or value.get("model_id") != model_id
        or value.get("training_mode") != training_mode
    ):
        raise ArtifactValidationError("resume manifest batch/model/mode binding mismatch")
    if provenance_evidence is None:
        _, actual_provenance = compute_verified_provenance(
            provenance_paths,
            batch_id=batch_id,
            model_id=model_id,
            training_mode=training_mode,
        )
    else:
        binding_from_provenance_evidence(
            provenance_paths,
            provenance_evidence,
            batch_id=batch_id,
            model_id=model_id,
            training_mode=training_mode,
        )
        actual_provenance = {
            role: dict(item) for role, item in provenance_evidence.items()
        }
    if requires_training_receipt and "runner_code" not in actual_provenance:
        raise ArtifactValidationError(
            f"{training_mode} resume requires independent runner_code provenance"
        )
    manifest_provenance = value.get("provenance")
    expected_roles = set(actual_provenance)
    if not isinstance(manifest_provenance, dict) or set(manifest_provenance) != expected_roles:
        raise ArtifactValidationError("resume provenance schema mismatch")
    for role in actual_provenance:
        recorded = _validate_evidence_schema(manifest_provenance[role], name=role)
        if recorded != actual_provenance[role]:
            raise ArtifactValidationError(f"resume provenance changed: {role}")

    artifact_info = _validate_evidence_schema(value.get("artifact"), name="artifact")
    artifact = resolve_within_root(
        artifact_info["path"], root, must_exist=True, allow_root=False
    )
    actual_artifact = _hash_nonempty(artifact, role="artifact")
    if actual_artifact != artifact_info:
        raise ArtifactValidationError("resume artifact content or filesystem evidence changed")

    reload_info = value.get("reload_verification")
    if training_mode in {"full_sft", "lora_sft", "grpo"}:
        if not isinstance(reload_info, dict) or set(reload_info) != {"path", "digest"}:
            raise ArtifactValidationError("resume manifest is missing reload verification")
        reload_path = resolve_within_root(
            reload_info["path"], root, must_exist=True, allow_root=False
        )
        if reload_path == artifact or artifact in reload_path.parents:
            raise ArtifactValidationError(
                "reload receipt cannot be inside the artifact whose digest it attests"
            )
        if sha256_file(reload_path) != reload_info["digest"]:
            raise ArtifactValidationError("reload receipt changed after publication")
        _validate_reload_receipt(
            reload_path,
            batch_id=batch_id,
            model_id=model_id,
            artifact_hash=actual_artifact["digest"],
        )
    elif reload_info is not None:
        if not isinstance(reload_info, dict) or set(reload_info) != {"path", "digest"}:
            raise ArtifactValidationError("invalid reload verification evidence")

    training_receipt_digest: str | None = None
    if requires_training_receipt:
        training_info = value.get("training_verification")
        if not isinstance(training_info, dict) or set(training_info) != {
            "path", "digest", "content_sha256"
        }:
            raise ArtifactValidationError("resume manifest lacks training verification")
        training_path = resolve_within_root(
            training_info["path"], root, must_exist=True, allow_root=False
        )
        if training_path == artifact or artifact in training_path.parents:
            raise ArtifactValidationError(
                "training receipt cannot be inside the artifact whose digest it attests"
            )
        if sha256_file(training_path) != training_info["digest"]:
            raise ArtifactValidationError("training receipt changed after publication")
        try:
            training_receipt = load_and_validate_training_receipt(
                training_path,
                expected={
                    "batch_id": batch_id,
                    "model_id": model_id,
                    "training_mode": training_mode,
                    "base_artifact_sha256": actual_provenance["base_artifact"]["digest"],
                    "train_sha256": actual_provenance["train_data"]["digest"],
                    "validation_sha256": actual_provenance["validation_data"]["digest"],
                    "leakage_audit_sha256": actual_provenance["leakage_audit"]["digest"],
                    "config_sha256": actual_provenance["config"]["digest"],
                    "code_sha256": actual_provenance["code"]["digest"],
                    "runner_code_sha256": actual_provenance["runner_code"]["digest"],
                    "environment_sha256": actual_provenance["environment"]["digest"],
                    "artifact_sha256": actual_artifact["digest"],
                },
            )
        except ValueError as exc:
            raise ArtifactValidationError(f"invalid training receipt: {exc}") from exc
        _validate_provenance_snapshot_reference(
            training_receipt,
            root=root,
            artifact=artifact,
            provenance=actual_provenance,
        )
        training_receipt_digest = training_receipt["receipt_sha256"]
        if training_receipt_digest != training_info["content_sha256"]:
            raise ArtifactValidationError(
                "training receipt content hash differs from manifest"
            )

    return FinetuneArtifactReceipt(
        manifest_path=manifest,
        artifact_path=artifact,
        artifact_digest=actual_artifact["digest"],
        artifact_kind=actual_artifact["kind"],
        manifest_digest=sha256_file(manifest),
        training_receipt_digest=training_receipt_digest,
    )


def write_reload_verification_receipt(
    path: str | Path,
    receipt: ReloadVerificationReceipt,
    *,
    allowed_root: str | Path,
    overwrite: bool = False,
) -> Path:
    root = Path(allowed_root).resolve(strict=True)
    destination = resolve_within_root(path, root, must_exist=False, allow_root=False)
    return atomic_write_json(destination, receipt.to_dict(), root=root, overwrite=overwrite)


def _resolve_module(root: Any, dotted_name: str) -> Any:
    current = root
    for part in dotted_name.split("."):
        if not hasattr(current, part) or getattr(current, part) is None:
            raise ArtifactValidationError(f"state module is absent: {dotted_name}")
        current = getattr(current, part)
    return current


def _state_payload(value: Any, *, name: str) -> tuple[str, tuple[int, ...], bytes]:
    current = value
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    try:
        import torch
    except Exception:  # pragma: no cover - torch-free control environment
        torch = None
    if torch is not None and isinstance(current, torch.Tensor):
        if (current.is_floating_point() or current.is_complex()) and not bool(
            torch.isfinite(current).all().item()
        ):
            raise ArtifactValidationError(f"state value is non-finite: {name}")
        raw = current.view(torch.uint8).numpy().tobytes(order="C")
        return str(current.dtype), tuple(int(item) for item in current.shape), raw
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        array = np.asarray(current)
    except Exception as exc:
        raise ArtifactValidationError(f"state value cannot be serialized safely: {name}") from exc
    if array.dtype.hasobject:
        raise ArtifactValidationError(f"object dtype is forbidden in state value: {name}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ArtifactValidationError(f"state value is non-finite: {name}")
    array = np.ascontiguousarray(array)
    return array.dtype.str, tuple(int(item) for item in array.shape), array.tobytes(order="C")


def _compact_named_snapshot(
    values: Mapping[str, Any], *, domain: bytes
) -> NamedStateSnapshot:
    if not isinstance(values, Mapping) or not values:
        raise ArtifactValidationError("state snapshot selection must be non-empty")
    aggregate = hashlib.sha256(domain)
    entry_hashes: dict[str, str] = {}
    parameter_count = 0
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("state snapshot names must be non-empty strings")
        dtype_name, shape, raw = _state_payload(values[name], name=name)
        shape_json = json.dumps(shape, separators=(",", ":")).encode("ascii")
        metadata = b"\0".join(
            (name.encode("utf-8"), dtype_name.encode("ascii"), shape_json)
        )
        entry_digest = hashlib.sha256(
            b"motionllm-state-entry-v1\0" + metadata + b"\0" + raw
        ).hexdigest()
        entry_hashes[name] = entry_digest
        for part in (metadata, bytes.fromhex(entry_digest)):
            aggregate.update(len(part).to_bytes(8, "big"))
            aggregate.update(part)
        size = 1
        for dimension in shape:
            size *= dimension
        parameter_count += size
    return NamedStateSnapshot(
        sha256=aggregate.hexdigest(),
        tensor_count=len(entry_hashes),
        parameter_count=parameter_count,
        tensor_sha256=entry_hashes,
    )


def snapshot_trainable_state(
    model: Any,
    *,
    parameter_names: tuple[str, ...] | None = None,
) -> NamedStateSnapshot:
    """Hash trainable tensors without retaining a second full model copy."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise ArtifactValidationError("model must expose named_parameters()")
    entries = list(named_parameters())
    if len({name for name, _ in entries}) != len(entries):
        raise ArtifactValidationError("model.named_parameters() contains duplicate names")
    by_name = dict(entries)
    if parameter_names is None:
        selected_names = tuple(
            name
            for name, parameter in entries
            if bool(getattr(parameter, "requires_grad", False))
        )
    else:
        if not parameter_names or len(set(parameter_names)) != len(parameter_names):
            raise ArtifactValidationError(
                "parameter_names must be a non-empty unique tuple"
            )
        missing = sorted(set(parameter_names) - set(by_name))
        if missing:
            raise ArtifactValidationError(
                f"snapshot parameters are missing after reload: {missing}"
            )
        selected_names = parameter_names
    return _compact_named_snapshot(
        {name: by_name[name] for name in selected_names},
        domain=b"motionllm-trainable-snapshot-v1\0",
    )


def changed_trainable_tensor_count(
    before: NamedStateSnapshot, after: NamedStateSnapshot
) -> int:
    before_names = set(before.tensor_sha256)
    after_names = set(after.tensor_sha256)
    if before_names != after_names:
        raise ArtifactValidationError(
            "trainable parameter names changed during training/reload"
        )
    changed = sum(
        before.tensor_sha256[name] != after.tensor_sha256[name]
        for name in before_names
    )
    if changed <= 0 or before.sha256 == after.sha256:
        raise ArtifactValidationError("optimizer training did not change trainable state")
    return changed


def model_state_snapshot(model: Any) -> NamedStateSnapshot:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise ArtifactValidationError("model must expose state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise ArtifactValidationError("model state_dict() must return a mapping")
    return _compact_named_snapshot(
        state, domain=b"motionllm-full-model-state-v1\0"
    )


def module_state_sha256(model: Any, module_names: tuple[str, ...]) -> str:
    if not module_names or len(set(module_names)) != len(module_names):
        raise ArtifactValidationError("module_names must be non-empty and unique")
    digest = hashlib.sha256(b"motionllm-module-state-v1\0")
    for module_name in sorted(module_names):
        module = _resolve_module(model, module_name)
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise ArtifactValidationError(f"module has no state_dict(): {module_name}")
        state = state_dict()
        if not isinstance(state, Mapping) or not state:
            raise ArtifactValidationError(f"module state_dict is empty/invalid: {module_name}")
        if not all(isinstance(key, str) and key for key in state):
            raise ArtifactValidationError(f"invalid state key in module {module_name}")
        for key in sorted(state):
            dtype_name, shape, raw_bytes = _state_payload(
                state[key], name=f"{module_name}.{key}"
            )
            for part in (
                module_name.encode("utf-8"),
                key.encode("utf-8"),
                dtype_name.encode("ascii"),
                json.dumps(shape, separators=(",", ":")).encode("ascii"),
                raw_bytes,
            ):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
    return digest.hexdigest()


def _state_entries(state: Mapping[str, Any], *, prefix: str = "") -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    entries: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for key in sorted(state):
        if not isinstance(key, str) or not key:
            raise ArtifactValidationError("state keys must be non-empty strings")
        name = f"{prefix}{key}"
        if name in entries:
            raise ArtifactValidationError(f"duplicate state entry: {name}")
        entries[name] = _state_payload(state[key], name=name)
    return entries


def _entries_sha256(entries: Mapping[str, tuple[str, tuple[int, ...], bytes]]) -> str:
    if not entries:
        raise ArtifactValidationError("verified reload state must be non-empty")
    digest = hashlib.sha256(b"motionllm-named-trainable-state-v1\0")
    for name in sorted(entries):
        dtype_name, shape, raw_bytes = entries[name]
        for part in (
            name.encode("utf-8"),
            dtype_name.encode("ascii"),
            json.dumps(shape, separators=(",", ":")).encode("ascii"),
            raw_bytes,
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def state_mapping_sha256(state: Mapping[str, Any]) -> str:
    """Hash a named state mapping including names, dtype, shape and exact bytes."""

    if not isinstance(state, Mapping):
        raise ArtifactValidationError("state must be a mapping")
    return _entries_sha256(_state_entries(state))


def verify_state_mapping_reload(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> str:
    """Compare exact named state metadata/content and return their shared hash."""

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ArtifactValidationError("reload states must be mappings")
    before_entries = _state_entries(before)
    after_entries = _state_entries(after)
    _compare_state_entries(before_entries, after_entries)
    before_hash = _entries_sha256(before_entries)
    after_hash = _entries_sha256(after_entries)
    if before_hash != after_hash:  # defensive; exact comparison above should imply this
        raise ArtifactValidationError("save/reload state hashes differ")
    return before_hash


_LORA_MARKERS = (
    (".lora_a.", "a"),
    (".lora_b.", "b"),
    (".lora_embedding_a.", "a"),
    (".lora_embedding_b.", "b"),
)


def validate_lora_adapter_pairs(names: Any) -> tuple[str, ...]:
    """Require an exact A/B pair for every LoRA adapter prefix."""

    if isinstance(names, Mapping):
        values = list(names)
    else:
        try:
            values = list(names)
        except TypeError as exc:
            raise ArtifactValidationError("LoRA state names must be iterable") from exc
    pairs: dict[str, dict[str, int]] = {}
    for name in values:
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("LoRA state names must be non-empty strings")
        lowered = name.casefold()
        matches = [
            (marker, side, lowered.find(marker))
            for marker, side in _LORA_MARKERS
            if marker in lowered
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise ArtifactValidationError(f"ambiguous LoRA adapter state name: {name}")
        marker, side, index = matches[0]
        canonical = lowered[:index] + ".lora_pair." + lowered[index + len(marker) :]
        counts = pairs.setdefault(canonical, {"a": 0, "b": 0})
        counts[side] += 1
    if not pairs:
        raise ArtifactValidationError("reload verification requires at least one LoRA A/B adapter")
    incomplete = {
        prefix: counts
        for prefix, counts in pairs.items()
        if counts != {"a": 1, "b": 1}
    }
    if incomplete:
        raise ArtifactValidationError(
            f"every LoRA adapter prefix requires exactly one A and one B tensor: {incomplete}"
        )
    return tuple(sorted(pairs))


def _json_safe_token(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_token(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_token(child) for child in value]
    return str(value)


def processor_state_sha256(processor: Any) -> str:
    """Hash tokenizer/processor semantics without trusting pickle state."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ArtifactValidationError("processor must expose tokenizer")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ArtifactValidationError("processor tokenizer must expose get_vocab()")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ArtifactValidationError("processor tokenizer vocabulary is empty/invalid")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if (
            not isinstance(token, str)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise ArtifactValidationError("processor tokenizer vocabulary is invalid")
        normalized_vocab[token] = token_id
    def class_name(value: Any) -> str:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    def component_state(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        to_dict = getattr(value, "to_dict", None)
        raw = to_dict() if callable(to_dict) else {}
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(
                f"processor component {class_name(value)} returned invalid to_dict() state"
            )
        return {"class": class_name(value), "config": _json_safe_token(raw)}

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    backend_state = backend_to_str() if callable(backend_to_str) else None
    if backend_state is not None and not isinstance(backend_state, str):
        raise ArtifactValidationError("tokenizer backend serialization must be text")
    payload = {
        "processor_class": class_name(processor),
        "processor_config": component_state(processor),
        "image_processor": component_state(getattr(processor, "image_processor", None)),
        "video_processor": component_state(getattr(processor, "video_processor", None)),
        "feature_extractor": component_state(getattr(processor, "feature_extractor", None)),
        "processor_chat_template": _json_safe_token(
            getattr(processor, "chat_template", None)
        ),
        "tokenizer_class": class_name(tokenizer),
        "vocab": normalized_vocab,
        "added_vocab": _json_safe_token(
            getattr(tokenizer, "get_added_vocab", lambda: {})()
        ),
        "backend_state": backend_state,
        "tokenizer_chat_template": _json_safe_token(
            getattr(tokenizer, "chat_template", None)
        ),
        "additional_special_tokens": _json_safe_token(
            getattr(tokenizer, "additional_special_tokens", ())
        ),
        "special_tokens_map": _json_safe_token(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "model_max_length": _json_safe_token(
            getattr(tokenizer, "model_max_length", None)
        ),
        "padding_side": _json_safe_token(getattr(tokenizer, "padding_side", None)),
        "truncation_side": _json_safe_token(
            getattr(tokenizer, "truncation_side", None)
        ),
        "clean_up_tokenization_spaces": _json_safe_token(
            getattr(tokenizer, "clean_up_tokenization_spaces", None)
        ),
        "split_special_tokens": _json_safe_token(
            getattr(tokenizer, "split_special_tokens", None)
        ),
    }
    return sha256_json(payload)


_PROCESSOR_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def processor_assets_sha256(artifact_path: str | Path) -> str:
    root = Path(artifact_path).resolve(strict=True)
    if not root.is_dir():
        raise ArtifactValidationError("processor artifact path must be a directory")
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.name in _PROCESSOR_ASSET_NAMES
                or path.name.startswith(("tokenizer", "processor", "preprocessor"))
            )
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ArtifactValidationError("saved artifact contains no processor/tokenizer assets")
    names = {path.name for path in files}
    if not any(name.startswith(("tokenizer", "vocab")) for name in names):
        raise ArtifactValidationError("saved artifact contains no tokenizer assets")
    if not any(name.startswith(("processor", "preprocessor")) for name in names):
        raise ArtifactValidationError("saved artifact contains no processor assets")
    evidence = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    if any(item["bytes"] <= 0 for item in evidence):
        raise ArtifactValidationError("processor/tokenizer asset files must be non-empty")
    return sha256_json(evidence)


def verify_processor_save_reload(
    original_processor: Any,
    reloaded_processor: Any,
    *,
    artifact_path: str | Path,
) -> tuple[str, str, str]:
    before = processor_state_sha256(original_processor)
    after = processor_state_sha256(reloaded_processor)
    if before != after:
        raise ArtifactValidationError("save/reload processor or tokenizer state differs")
    return before, after, processor_assets_sha256(artifact_path)


def _module_entries(model: Any, module_names: tuple[str, ...]) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    entries: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for module_name in sorted(module_names):
        module = _resolve_module(model, module_name)
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise ArtifactValidationError(f"module has no state_dict(): {module_name}")
        state = state_dict()
        if not isinstance(state, Mapping) or not state:
            raise ArtifactValidationError(f"module state_dict is empty/invalid: {module_name}")
        for name, payload in _state_entries(state, prefix=f"module:{module_name}.").items():
            if name in entries:
                raise ArtifactValidationError(f"duplicate module state entry: {name}")
            entries[name] = payload
    return entries


def _original_trainable_entries(
    model: Any, module_names: tuple[str, ...]
) -> tuple[dict[str, tuple[str, tuple[int, ...], bytes]], tuple[str, ...]]:
    entries = _module_entries(model, module_names)
    selected_names: list[str] = []
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise ArtifactValidationError("LoRA model must expose named_parameters()")
    all_parameters = list(named_parameters())
    if len({name for name, _ in all_parameters}) != len(all_parameters):
        raise ArtifactValidationError("model.named_parameters() contains duplicate names")
    lora_names = [
        name
        for name, _ in all_parameters
        if any(marker in name.casefold() for marker, _ in _LORA_MARKERS)
    ]
    validate_lora_adapter_pairs(lora_names)
    selected_names = sorted(
        {
            name
            for name, parameter in all_parameters
            if bool(getattr(parameter, "requires_grad", False)) or name in lora_names
        }
    )
    parameter_map = dict(all_parameters)
    for name in selected_names:
        entry_name = f"parameter:{name}"
        entries[entry_name] = _state_payload(parameter_map[name], name=entry_name)
    if not entries:
        raise ArtifactValidationError("no trainable or modules_to_save state was selected")
    return entries, tuple(selected_names)


def _reloaded_trainable_entries(
    model: Any,
    module_names: tuple[str, ...],
    parameter_names: tuple[str, ...],
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    entries = _module_entries(model, module_names)
    if parameter_names:
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise ArtifactValidationError("reloaded model has no named_parameters()")
        parameters = dict(named_parameters())
        reloaded_lora_names = {
            name
            for name in parameters
            if any(marker in name.casefold() for marker, _ in _LORA_MARKERS)
        }
        validate_lora_adapter_pairs(reloaded_lora_names)
        expected_lora_names = {
            name
            for name in parameter_names
            if any(marker in name.casefold() for marker, _ in _LORA_MARKERS)
        }
        if reloaded_lora_names != expected_lora_names:
            raise ArtifactValidationError(
                "reloaded LoRA adapter parameter names differ: "
                f"missing={sorted(expected_lora_names - reloaded_lora_names)}, "
                f"unexpected={sorted(reloaded_lora_names - expected_lora_names)}"
            )
        missing = sorted(set(parameter_names) - set(parameters))
        if missing:
            raise ArtifactValidationError(f"reloaded trainable parameters are missing: {missing}")
        for name in parameter_names:
            entry_name = f"parameter:{name}"
            entries[entry_name] = _state_payload(parameters[name], name=entry_name)
    return entries


def _compare_state_entries(
    before: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    after: Mapping[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    before_names = set(before)
    after_names = set(after)
    if before_names != after_names:
        raise ArtifactValidationError(
            "save/reload trainable state names differ: "
            f"missing={sorted(before_names - after_names)}, "
            f"unexpected={sorted(after_names - before_names)}"
        )
    for name in sorted(before):
        before_dtype, before_shape, before_bytes = before[name]
        after_dtype, after_shape, after_bytes = after[name]
        if before_dtype != after_dtype:
            raise ArtifactValidationError(f"save/reload dtype differs for {name}")
        if before_shape != after_shape:
            raise ArtifactValidationError(f"save/reload shape differs for {name}")
        if before_bytes != after_bytes:
            raise ArtifactValidationError(f"save/reload content differs for {name}")


def verify_full_save_reload(
    original_model: Any,
    reloaded_model: Any,
    *,
    tokenizer: Any,
    reloaded_tokenizer: Any,
    processor: Any,
    reloaded_processor: Any,
    processor_artifact_path: str | Path,
    batch_id: str,
    model_id: str,
    artifact_hash: str,
    supports_motion: bool = True,
) -> ReloadVerificationReceipt:
    """Prove a fresh full-model and processor reload matches the saved state."""

    if not isinstance(supports_motion, bool):
        raise ArtifactValidationError("supports_motion must be bool")
    token_ids: tuple[int | None, int | None] = (None, None)
    if supports_motion:
        token_ids = verify_motion_tokens(tokenizer, original_model)
        if token_ids != verify_motion_tokens(reloaded_tokenizer, reloaded_model):
            raise ArtifactValidationError("motion token ids changed after full reload")
    before = model_state_snapshot(original_model)
    after = model_state_snapshot(reloaded_model)
    if before.tensor_sha256 != after.tensor_sha256 or before.sha256 != after.sha256:
        raise ArtifactValidationError("full model content differs after save/reload")
    processor_before, processor_after, processor_assets_hash = verify_processor_save_reload(
        processor,
        reloaded_processor,
        artifact_path=processor_artifact_path,
    )
    return ReloadVerificationReceipt(
        batch_id=batch_id,
        model_id=model_id,
        artifact_hash=artifact_hash,
        expected_modules=("__full_model__",),
        reloaded_modules=("__full_model__",),
        motion_start_token_id=token_ids[0],
        motion_end_token_id=token_ids[1],
        state_hash_before=before.sha256,
        state_hash_after=after.sha256,
        processor_state_hash_before=processor_before,
        processor_state_hash_after=processor_after,
        processor_assets_hash=processor_assets_hash,
    )


def verify_lora_save_reload(
    original_model: Any,
    reloaded_model: Any,
    *,
    tokenizer: Any,
    reloaded_tokenizer: Any,
    processor: Any,
    reloaded_processor: Any,
    processor_artifact_path: str | Path,
    module_names: tuple[str, ...],
    batch_id: str,
    model_id: str,
    artifact_hash: str,
    supports_motion: bool = True,
) -> ReloadVerificationReceipt:
    if not isinstance(supports_motion, bool):
        raise ArtifactValidationError("supports_motion must be bool")
    if supports_motion and not module_names:
        raise ArtifactValidationError(
            "motion LoRA reload verification requires modules_to_save"
        )
    original_ids: tuple[int | None, int | None] = (None, None)
    if supports_motion:
        original_ids = verify_motion_tokens(tokenizer, original_model)
        reloaded_ids = verify_motion_tokens(reloaded_tokenizer, reloaded_model)
        if original_ids != reloaded_ids:
            raise ArtifactValidationError(
                f"motion token ids changed after reload: {original_ids} -> {reloaded_ids}"
            )
    before_entries, parameter_names = _original_trainable_entries(
        original_model, module_names
    )
    after_entries = _reloaded_trainable_entries(
        reloaded_model, module_names, parameter_names
    )
    _compare_state_entries(before_entries, after_entries)
    before = _entries_sha256(before_entries)
    after = _entries_sha256(after_entries)
    processor_before, processor_after, processor_assets_hash = verify_processor_save_reload(
        processor,
        reloaded_processor,
        artifact_path=processor_artifact_path,
    )
    return ReloadVerificationReceipt(
        batch_id=batch_id,
        model_id=model_id,
        artifact_hash=artifact_hash,
        expected_modules=module_names,
        reloaded_modules=module_names,
        motion_start_token_id=original_ids[0],
        motion_end_token_id=original_ids[1],
        state_hash_before=before,
        state_hash_after=after,
        processor_state_hash_before=processor_before,
        processor_state_hash_after=processor_after,
        processor_assets_hash=processor_assets_hash,
    )
