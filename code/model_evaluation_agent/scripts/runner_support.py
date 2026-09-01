"""Strict shared CLI and provenance layer for all catalog runner facades."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_VERIFIED_MULTI_ROOT_BOOTSTRAP_BLOCKER = (
    "blocker=verified-multi-root-bootstrap: production catalog execution is "
    "disabled until the controller provides an isolated -I -S -B verified "
    "multi-root source/environment bootstrap"
)


def _raw_option(name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name:
            following = index + 2
            return sys.argv[following] if following < len(sys.argv) else None
    return None


def _fail_closed_before_project_imports() -> None:
    """Stop production facades before importing mutable project modules.

    This is defense in depth only: Python may execute a system
    ``sitecustomize`` before this file.  The trusted controller's pre-spawn
    refusal is the authoritative boundary until the multi-root bootstrap is
    implemented.
    """

    # CLI discovery is non-executing and must remain usable.  argparse exits
    # immediately after rendering help, so this does not open a production
    # finetune/evaluation/verifier path.
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return
    facade = Path(sys.argv[0]).name
    if facade.startswith("eval_") or facade == "verify_artifact_reload.py":
        raise RuntimeError(_VERIFIED_MULTI_ROOT_BOOTSTRAP_BLOCKER)
    if facade.startswith("finetune_") and _raw_option("--purpose") != "preflight":
        raise RuntimeError(_VERIFIED_MULTI_ROOT_BOOTSTRAP_BLOCKER)


_fail_closed_before_project_imports()

import motion_eval.training_receipt as training_receipt_module
import motion_eval.core.source_inventory as source_inventory_module
from motion_eval.core import atomic_write_json, hash_path, sha256_file, sha256_json
from motion_eval.data import load_and_validate_batch_receipt, load_json_strict
from motion_eval.training_receipt import (
    load_and_validate_formal_provenance_snapshot,
    load_and_validate_training_receipt,
)

from runner_specs import MODEL_SPECS, backend_for, dependencies_for


_SAFE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_ATTEMPT_KEYS = frozenset(
    {
        "schema_version", "attempt_id", "batch_id", "batch_receipt_sha256",
        "model_id", "stage", "purpose", "expected_training_steps", "sample_limit",
        "gpu_uuid", "gpu_index", "keepalive_root", "keepalive_owner", "lease_nonce",
        "leased_revision", "created_at", "command", "command_sha256", "attempt_sha256",
    }
)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version", "batch_id", "model_id", "attempt_id", "purpose", "status",
        "exit_code", "started_at", "finished_at", "training_steps", "bindings", "artifact",
        "training_receipt", "manifest_sha256",
    }
)
_ARTIFACT_KEYS = frozenset(
    {"path", "algorithm", "kind", "digest", "file_count", "total_bytes"}
)
_TRAINING_REFERENCE_KEYS = frozenset({"path", "file_sha256", "content_sha256"})
_FINETUNE_EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "evidence_type", "batch_id", "model_id", "attempt_id",
        "batch_receipt_sha256", "attempt_sha256", "execution_sha256",
        "verification_sha256", "run_manifest_path", "run_manifest_sha256", "artifact",
        "training_receipt", "fresh_current_batch", "finished_at", "evidence_sha256",
    }
)
_FORMAL_QWEN_MODEL_IDS = frozenset(
    {
        "motionr1_vm_lora",
        "qwen35_4b_lora",
        "qwen36_27b_lora",
        "qwen3vl_4b_lora",
        "qwen3vl_8b_lora",
    }
)
_FORMAL_QWEN_ROLE_SPLIT_ERROR = (
    "formal Qwen catalog publication is blocked: the receipt contract does not "
    "separately represent controller_code/catalog_runner_code and "
    "training_code/training_runner_code; incompatible same-named roles cannot "
    "be aliased or cross-bound"
)


class BackendUnavailable(RuntimeError):
    """A catalog facade has no reviewed implementation behind it."""


def _verify_project_code_origins(receipt: Mapping[str, Any]) -> None:
    """Bind imported project modules to the two recursively hashed code trees."""

    import motion_eval

    expected_runner_root = Path(str(receipt.get("runner_code", {}).get("path", ""))).resolve(
        strict=True
    )
    expected_code_root = Path(str(receipt.get("code", {}).get("path", ""))).resolve(
        strict=True
    )
    actual_runner_root = Path(__file__).resolve(strict=True).parent
    actual_code_root = Path(motion_eval.__file__).resolve(strict=True).parent
    actual_training_receipt = Path(training_receipt_module.__file__).resolve(strict=True)
    actual_source_inventory = Path(source_inventory_module.__file__).resolve(strict=True)
    if actual_runner_root != expected_runner_root:
        raise RuntimeError("runner_support was not imported from frozen runner_code")
    if actual_code_root != expected_code_root:
        raise RuntimeError("motion_eval was not imported from frozen controller code")
    if actual_training_receipt != expected_code_root / "training_receipt.py":
        raise RuntimeError(
            "formal snapshot validator was not imported from frozen controller code"
        )
    if actual_source_inventory != expected_code_root / "core" / "source_inventory.py":
        raise RuntimeError(
            "formal source inventory verifier was not imported from frozen controller code"
        )


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _load_object(
    path: Path,
    label: str,
    *,
    expected_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    try:
        value = load_json_strict(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is not readable strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    result = dict(value)
    if expected_keys is not None and set(result) != expected_keys:
        raise ValueError(f"{label} schema is invalid")
    return result


def _load_attempt(path: Path, label: str) -> dict[str, Any]:
    attempt = _load_object(path, label, expected_keys=_ATTEMPT_KEYS)
    body = {key: item for key, item in attempt.items() if key != "attempt_sha256"}
    if (
        attempt.get("schema_version") != "2.0"
        or attempt.get("attempt_sha256") != sha256_json(body)
        or attempt.get("command_sha256") != sha256_json(attempt.get("command"))
        or not isinstance(attempt.get("command"), dict)
        or attempt["command"].get("shell") is not False
    ):
        raise ValueError(f"{label} self-hash/command binding is invalid")
    return attempt


def _load_batch(path: Path, label: str) -> dict[str, Any]:
    try:
        return load_and_validate_batch_receipt(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid or its frozen inputs changed") from exc


def _existing_file(value: str, label: str) -> Path:
    supplied = Path(value)
    if supplied.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    path = supplied.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a non-symlink regular file")
    return path


def _selected_model(model_id: str, expected_model_ids: Iterable[str]) -> tuple[str, str, str]:
    allowed = frozenset(expected_model_ids)
    if model_id not in allowed:
        raise ValueError(
            f"runner facade is bound to {sorted(allowed)}, not model {model_id!r}"
        )
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown catalog model: {model_id}") from exc


def _require_backend(model_id: str, role: str):
    module_name = backend_for(model_id, role)
    if module_name is None:
        raise BackendUnavailable(
            f"{model_id} has no reviewed {role} backend; no output was produced. "
            f"Use the controller missing_code blocker component=backend:{role}."
        )
    return importlib.import_module(module_name)


def _production_preflight(model_id: str, pretrained_root: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"{model_id} production backend requires Linux")
    if model_id == "motionllm_official":
        runtime_deps = pretrained_root / "runtime_deps" / "motionllm"
        if not runtime_deps.is_dir():
            raise RuntimeError(
                "motionllm_official pinned runtime dependency tree is missing"
            )
        sys.path.insert(0, str(runtime_deps))
    missing: list[str] = []
    for name in dependencies_for(model_id):
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"{model_id} production dependencies are missing: {', '.join(missing)}"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{model_id} production backend requires CUDA")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be controller-bound")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _batch_context(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output = Path(args.output_dir).resolve(strict=False)
    attempt_root = output.parent
    if output != attempt_root / "artifact":
        raise ValueError("--output-dir must be the controller attempt artifact directory")
    attempt = _load_attempt(attempt_root / "attempt_receipt.json", "attempt receipt")
    batch_root = attempt_root.parents[3]
    receipt = _load_batch(
        batch_root / "00_inputs" / "batch_receipt.json", "batch receipt"
    )
    _verify_project_code_origins(receipt)
    if (
        attempt.get("batch_id") != args.batch_id
        or attempt.get("model_id") != args.model_id
        or attempt.get("stage") != "finetune"
        or attempt.get("purpose") != args.purpose
        or attempt.get("expected_training_steps") != args.training_steps
        or attempt.get("sample_limit") != args.limit
        or receipt.get("batch_id") != args.batch_id
        or receipt.get("receipt_sha256") != attempt.get("batch_receipt_sha256")
    ):
        raise ValueError("runner arguments differ from frozen batch/attempt identity")
    roots = receipt.get("runtime_roots", {})
    if Path(str(roots.get("pretrained_root", ""))).resolve(strict=False) != Path(
        args.pretrained_root
    ).resolve(strict=False):
        raise ValueError("--pretrained-root differs from frozen batch receipt")
    expected_inputs = {
        "train": args.train_manifest,
        "validation": args.validation_manifest,
        "leakage_audit": args.leakage_audit,
    }
    for role, supplied in expected_inputs.items():
        path = _existing_file(supplied, f"--{role.replace('_', '-')}")
        frozen = receipt.get("inputs", {}).get(role, {})
        if str(path) != frozen.get("path") or sha256_file(path) != frozen.get("digest"):
            raise ValueError(f"{role} input differs from its frozen receipt")
    if receipt.get("inputs", {}).get("train", {}).get("digest") == receipt.get(
        "inputs", {}
    ).get("validation", {}).get("digest"):
        raise ValueError("train and validation inputs must be distinct")
    if not isinstance(receipt.get("leakage_verification"), dict):
        raise ValueError("batch receipt lacks controller-recomputed leakage evidence")
    return attempt_root, attempt, receipt


def _backend_training_identity(backend: Any, *, modality: str) -> dict[str, str]:
    identity = {
        "backend_id": getattr(backend, "BACKEND_ID", None),
        "model_family": getattr(backend, "MODEL_FAMILY", None),
        "modality": modality,
        "training_mode": getattr(backend, "TRAINING_MODE", None),
    }
    for field, value in identity.items():
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"backend does not publish a fixed {field}")
    if identity["modality"] not in {"V", "M", "VM"}:
        raise RuntimeError("backend modality contract is invalid")
    if identity["training_mode"] not in {"full_sft", "lora_sft", "official_finetune"}:
        raise RuntimeError("backend training_mode contract is invalid")
    return identity


def _training_bindings(
    *,
    args: argparse.Namespace,
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
    backend: Any,
) -> dict[str, Any]:
    identity = _backend_training_identity(backend, modality=args.modality)
    return {
        "batch_id": args.batch_id,
        "model_id": args.model_id,
        **identity,
        "batch_receipt_sha256": receipt["receipt_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "train_sha256": receipt["inputs"]["train"]["digest"],
        "validation_sha256": receipt["inputs"]["validation"]["digest"],
        "leakage_audit_sha256": receipt["inputs"]["leakage_audit"]["digest"],
        "base_artifact_sha256": sha256_json(receipt["pretrained_assets"][args.model_id]),
        "config_sha256": receipt["config_sha256"],
        "code_sha256": receipt["code"]["digest"],
        "runner_code_sha256": receipt["runner_code"]["digest"],
        "environment_sha256": receipt["environment_sha256"],
    }


def _manifest_bindings(
    receipt: Mapping[str, Any], attempt: Mapping[str, Any], model_id: str
) -> dict[str, str]:
    training_config = receipt["config"]["model_training"][model_id]
    return {
        "batch_receipt_sha256": receipt["receipt_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "command_sha256": attempt["command_sha256"],
        "registry_sha256": receipt["registry"]["sha256"],
        "pretrained_registry_sha256": receipt["pretrained_registry"]["sha256"],
        "pretrained_assets_sha256": receipt["pretrained_assets_sha256"],
        "model_pretrained_assets_sha256": sha256_json(
            receipt["pretrained_assets"][model_id]
        ),
        "model_training_config_sha256": sha256_json(training_config),
        "train_sha256": receipt["inputs"]["train"]["digest"],
        "validation_sha256": receipt["inputs"]["validation"]["digest"],
        "leakage_audit_sha256": receipt["inputs"]["leakage_audit"]["digest"],
        "code_sha256": receipt["code"]["digest"],
        "runner_code_sha256": receipt["runner_code"]["digest"],
        "config_sha256": receipt["config_sha256"],
        "environment_sha256": receipt["environment_sha256"],
    }


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _formal_snapshot_file(
    value: Any,
    *,
    attempt_root: Path,
    artifact: Path,
    training_path: Path,
) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("formal provenance snapshot path must be absolute")
    lexical = Path(os.path.abspath(value))
    if os.path.normcase(str(lexical)) != os.path.normcase(value):
        raise ValueError("formal provenance snapshot path must be normalized")
    try:
        relative = lexical.relative_to(attempt_root)
    except ValueError as exc:
        raise ValueError("formal provenance snapshot escapes the finetune attempt") from exc
    if not relative.parts:
        raise ValueError("formal provenance snapshot cannot be the attempt directory")
    current = attempt_root
    for part in relative.parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise ValueError("formal provenance snapshot path cannot contain links/reparse points")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("formal provenance snapshot is not an existing regular file") from exc
    if (
        os.path.normcase(str(resolved)) != os.path.normcase(str(lexical))
        or not resolved.is_file()
        or resolved in {training_path, attempt_root / "run_manifest.json"}
        or resolved == artifact
        or artifact in resolved.parents
    ):
        raise ValueError(
            "formal provenance snapshot must be a non-linked attempt file outside the artifact"
        )
    return resolved


def _matching_frozen_asset_rows(
    evidence: Mapping[str, Any], rows: Any
) -> list[int]:
    if not isinstance(rows, list):
        return []
    expected_content = {
        key: evidence.get(key)
        for key in ("algorithm", "kind", "digest", "file_count", "total_bytes")
    }
    matches: list[int] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("state") != "present"
            or row.get("content") != expected_content
            or not isinstance(row.get("path"), str)
        ):
            continue
        try:
            frozen_path = Path(row["path"]).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if str(frozen_path) == evidence.get("path"):
            matches.append(index)
    return matches


def _validate_catalog_formal_snapshot(
    *,
    attempt_root: Path,
    artifact: Path,
    training_path: Path,
    batch_receipt: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    """Independently follow and bind a formal Qwen snapshot at reload time."""

    schema_version = training_receipt.get("schema_version")
    if schema_version != "2.0":
        if model_id in _FORMAL_QWEN_MODEL_IDS:
            raise ValueError("formal Qwen artifacts require a schema-2 provenance snapshot")
        return None

    snapshot_path = _formal_snapshot_file(
        training_receipt.get("provenance_snapshot_path"),
        attempt_root=attempt_root,
        artifact=artifact,
        training_path=training_path,
    )
    pre_sha256 = training_receipt.get("provenance_pre_sha256")
    try:
        snapshot = load_and_validate_formal_provenance_snapshot(
            snapshot_path,
            expected_file_sha256=training_receipt.get(
                "provenance_snapshot_file_sha256"
            ),
            expected={
                "batch_id": training_receipt.get("batch_id"),
                "model_id": model_id,
                "training_mode": training_receipt.get("training_mode"),
                "snapshot_sha256": pre_sha256,
            },
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("formal Qwen provenance snapshot is invalid") from exc
    if (
        snapshot.get("snapshot_sha256") != pre_sha256
        or training_receipt.get("provenance_post_sha256") != pre_sha256
        or training_receipt.get("provenance_unchanged") is not True
    ):
        raise ValueError("formal Qwen provenance pre/post binding is invalid")

    provenance = snapshot.get("provenance")
    identity = snapshot.get("canonical_identity")
    bindings = artifact_manifest.get("bindings")
    if not all(isinstance(item, Mapping) for item in (provenance, identity, bindings)):
        raise ValueError("formal Qwen snapshot/manifest provenance is malformed")
    expected_modality = MODEL_SPECS[model_id][0]
    if identity.get("modality") != expected_modality:
        raise ValueError("formal Qwen snapshot modality differs from the catalog")
    batch_code = batch_receipt.get("code")
    batch_runner_code = batch_receipt.get("runner_code")
    training_code = provenance.get("code")
    training_runner_code = provenance.get("runner_code")
    if model_id in _FORMAL_QWEN_MODEL_IDS and (
        not all(
            isinstance(item, Mapping)
            for item in (
                batch_code,
                batch_runner_code,
                training_code,
                training_runner_code,
            )
        )
        or batch_code.get("path") != training_code.get("path")
        or batch_runner_code.get("path") != training_runner_code.get("path")
    ):
        raise ValueError(_FORMAL_QWEN_ROLE_SPLIT_ERROR)

    digest_bindings = {
        "base_artifact": ("base_artifact_sha256", "model_pretrained_assets_sha256"),
        "train_data": ("train_sha256", "train_sha256"),
        "validation_data": ("validation_sha256", "validation_sha256"),
        "leakage_audit": ("leakage_audit_sha256", "leakage_audit_sha256"),
        "config": ("config_sha256", "config_sha256"),
        "code": ("code_sha256", "code_sha256"),
        "runner_code": ("runner_code_sha256", "runner_code_sha256"),
        "environment": ("environment_sha256", "environment_sha256"),
    }
    mismatches = []
    for role, (training_field, manifest_field) in digest_bindings.items():
        evidence = provenance.get(role)
        digest = evidence.get("digest") if isinstance(evidence, Mapping) else None
        if (
            digest != training_receipt.get(training_field)
            or digest != bindings.get(manifest_field)
        ):
            mismatches.append(role)
    if mismatches:
        raise ValueError(
            "formal Qwen snapshot differs from training/artifact manifest provenance: "
            f"{sorted(mismatches)}"
        )

    input_roles = {
        "train_data": "train",
        "validation_data": "validation",
        "benchmark": "benchmark",
        "leakage_audit": "leakage_audit",
    }
    frozen_inputs = batch_receipt.get("inputs")
    if not isinstance(frozen_inputs, Mapping):
        raise ValueError("batch receipt inputs are unavailable for formal provenance")
    input_mismatches = [
        role
        for role, input_role in input_roles.items()
        if provenance.get(role) != frozen_inputs.get(input_role)
    ]
    if input_mismatches:
        raise ValueError(
            "formal Qwen snapshot differs from frozen batch inputs: "
            f"{sorted(input_mismatches)}"
        )

    for role, batch_role in (("code", "code"), ("runner_code", "runner_code")):
        evidence = provenance[role]
        frozen = batch_receipt.get(batch_role)
        if (
            not isinstance(frozen, Mapping)
            or evidence.get("path") != frozen.get("path")
            or evidence.get("digest") != frozen.get("digest")
        ):
            raise ValueError(f"formal Qwen snapshot {role} differs from the batch receipt")
    if (
        provenance["config"].get("digest") != batch_receipt.get("config_sha256")
        or provenance["environment"].get("digest")
        != batch_receipt.get("environment_sha256")
    ):
        raise ValueError("formal Qwen snapshot config/environment differs from the batch receipt")

    frozen_assets = batch_receipt.get("pretrained_assets")
    asset_rows = frozen_assets.get(model_id) if isinstance(frozen_assets, Mapping) else None
    used_asset_rows: set[int] = set()
    for role in ("base_artifact", "motion_vqvae"):
        evidence = provenance.get(role)
        if evidence is None:
            continue
        matches = _matching_frozen_asset_rows(evidence, asset_rows)
        if len(matches) != 1 or matches[0] in used_asset_rows:
            raise ValueError(
                f"formal Qwen snapshot {role} is not uniquely bound to a frozen asset"
            )
        used_asset_rows.add(matches[0])
    if model_id in _FORMAL_QWEN_MODEL_IDS:
        raise ValueError(_FORMAL_QWEN_ROLE_SPLIT_ERROR)
    return snapshot


def _load_bound_finetune_artifact(
    artifact: Path,
    *,
    batch_id: str,
    model_id: str,
    require_evidence: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reverify the complete current-batch training chain from immutable files."""

    attempt_root = artifact.parent
    if artifact != attempt_root / "artifact":
        raise ValueError("artifact must be the controller finetune attempt artifact")
    attempt = _load_attempt(
        attempt_root / "attempt_receipt.json", "finetune attempt receipt"
    )
    receipt = _load_batch(
        attempt_root.parents[3] / "00_inputs" / "batch_receipt.json",
        "batch receipt",
    )
    if (
        attempt.get("batch_id") != batch_id
        or attempt.get("model_id") != model_id
        or attempt.get("stage") != "finetune"
        or attempt.get("purpose") != "production"
        or attempt.get("sample_limit") is not None
        or attempt.get("batch_receipt_sha256") != receipt.get("receipt_sha256")
    ):
        raise ValueError("finetune attempt is not a production attempt for this batch/model")

    manifest_path = attempt_root / "run_manifest.json"
    manifest = _load_object(
        manifest_path, "finetune run manifest", expected_keys=_RUN_MANIFEST_KEYS
    )
    manifest_body = {
        key: item for key, item in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_sha256") != sha256_json(manifest_body)
        or manifest.get("status") != "success"
        or manifest.get("exit_code") != 0
        or manifest.get("batch_id") != batch_id
        or manifest.get("model_id") != model_id
        or manifest.get("attempt_id") != attempt.get("attempt_id")
        or manifest.get("purpose") != "production"
        or manifest.get("training_steps") != attempt.get("expected_training_steps")
        or manifest.get("bindings") != _manifest_bindings(receipt, attempt, model_id)
    ):
        raise ValueError("finetune run manifest identity/provenance is invalid")

    artifact_info = manifest.get("artifact")
    if not isinstance(artifact_info, dict) or set(artifact_info) != _ARTIFACT_KEYS:
        raise ValueError("finetune manifest artifact schema is invalid")
    actual_artifact = {
        "path": str(artifact),
        **hash_path(
            artifact, symlink_policy="reject", allowed_root=attempt_root
        ).to_dict(),
    }
    if artifact_info != actual_artifact:
        raise ValueError("finetune artifact differs from its run manifest")

    reference = manifest.get("training_receipt")
    if not isinstance(reference, dict) or set(reference) != _TRAINING_REFERENCE_KEYS:
        raise ValueError("finetune manifest lacks strict training evidence")
    training_path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if training_path != attempt_root / "training_receipt.json":
        raise ValueError("training receipt path is outside the finetune attempt")
    if sha256_file(training_path) != reference.get("file_sha256"):
        raise ValueError("training receipt file changed after publication")
    finetune_backend = _require_backend(model_id, "finetune")
    modality, _mode, _initialization = _selected_model(model_id, MODEL_SPECS)
    expected_training = {
        **_training_bindings(
            args=argparse.Namespace(
                batch_id=batch_id, model_id=model_id, modality=modality
            ),
            attempt=attempt,
            receipt=receipt,
            backend=finetune_backend,
        ),
        "planned_global_steps": manifest["training_steps"],
        "actual_global_steps": manifest["training_steps"],
        "planned_optimizer_steps": manifest["training_steps"],
        "actual_optimizer_steps": manifest["training_steps"],
        "artifact_sha256": actual_artifact["digest"],
    }
    training = load_and_validate_training_receipt(
        training_path, expected=expected_training
    )
    if training["receipt_sha256"] != reference.get("content_sha256"):
        raise ValueError("training receipt content hash differs from the manifest")
    _validate_catalog_formal_snapshot(
        attempt_root=attempt_root,
        artifact=artifact,
        training_path=training_path,
        batch_receipt=receipt,
        artifact_manifest=manifest,
        training_receipt=training,
        model_id=model_id,
    )

    if require_evidence:
        evidence_path = attempt_root / "finetune_evidence.json"
        evidence = _load_object(
            evidence_path,
            "finetune evidence",
            expected_keys=_FINETUNE_EVIDENCE_KEYS,
        )
        evidence_body = {
            key: item for key, item in evidence.items() if key != "evidence_sha256"
        }
        if (
            evidence.get("schema_version") != "1.0"
            or evidence.get("evidence_type") != "fresh_finetune"
            or evidence.get("evidence_sha256") != sha256_json(evidence_body)
            or evidence.get("batch_id") != batch_id
            or evidence.get("model_id") != model_id
            or evidence.get("attempt_id") != attempt["attempt_id"]
            or evidence.get("batch_receipt_sha256") != receipt["receipt_sha256"]
            or evidence.get("attempt_sha256") != attempt["attempt_sha256"]
            or evidence.get("run_manifest_path") != str(manifest_path)
            or evidence.get("run_manifest_sha256") != sha256_file(manifest_path)
            or evidence.get("artifact") != artifact_info
            or evidence.get("training_receipt") != reference
            or evidence.get("fresh_current_batch") is not True
        ):
            raise ValueError("finetune evidence is not bound to this manifest/artifact")
    return Path(receipt["runtime_roots"]["pretrained_root"]).resolve(strict=True), receipt, manifest, training


def _finetune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict catalog finetune facade; never substitutes a fake backend."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--leakage-audit", required=True)
    parser.add_argument("--pretrained-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modality", required=True, choices=("V", "M", "VM"))
    parser.add_argument("--initialization", required=True, choices=("pretrained", "random"))
    parser.add_argument("--purpose", required=True, choices=("production", "preflight"))
    parser.add_argument("--training-steps", required=True, type=int)
    parser.add_argument("--limit", type=int)
    return parser


def finetune_main(expected_model_ids: Iterable[str]) -> int:
    args = _finetune_parser().parse_args()
    _safe_id(args.batch_id, "batch id")
    _safe_id(args.model_id, "model id")
    modality, _mode, initialization = _selected_model(args.model_id, expected_model_ids)
    if args.modality != modality or args.initialization != initialization:
        raise ValueError("modality/initialization differs from the frozen model spec")
    if args.training_steps <= 0:
        raise ValueError("--training-steps must be positive")
    if args.purpose == "production" and args.limit is not None:
        raise ValueError("production finetune cannot use --limit")
    if args.purpose == "preflight" and (args.limit is None or args.limit <= 0):
        raise ValueError("preflight finetune requires a positive --limit")
    attempt_root, attempt, receipt = _batch_context(args)
    manifest_path = attempt_root / "run_manifest.json"
    if Path(args.output_dir).exists() or manifest_path.exists():
        raise FileExistsError("finetune outputs must not predate this process")
    backend = _require_backend(args.model_id, "finetune")
    _production_preflight(args.model_id, Path(args.pretrained_root).resolve(strict=True))
    evidence_bindings = _training_bindings(
        args=args, attempt=attempt, receipt=receipt, backend=backend
    )
    training_receipt_path = attempt_root / "training_receipt.json"
    if training_receipt_path.exists():
        raise FileExistsError("training receipt must not predate this process")
    seed = receipt.get("config", {}).get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("batch config must freeze a non-negative integer seed")
    started_at = datetime.now(timezone.utc).isoformat()
    result = backend.run_finetune(
        train_manifest=Path(args.train_manifest).resolve(strict=True),
        validation_manifest=Path(args.validation_manifest).resolve(strict=True),
        pretrained_root=Path(args.pretrained_root).resolve(strict=True),
        output_dir=Path(args.output_dir).resolve(strict=False),
        work_dir=attempt_root / "work",
        training_steps=args.training_steps,
        limit=args.limit,
        seed=seed,
        training_receipt_path=training_receipt_path,
        evidence_bindings=evidence_bindings,
    )
    if result not in (None, 0):
        raise RuntimeError(f"backend returned a non-success result: {result!r}")
    # Re-run the full batch validator after training. This rehashes all frozen
    # inputs, code, runner code, pretrained assets, config, and environment.
    current_receipt = _load_batch(
        attempt_root.parents[3] / "00_inputs" / "batch_receipt.json",
        "post-training batch receipt",
    )
    current_attempt = _load_attempt(
        attempt_root / "attempt_receipt.json", "post-training attempt receipt"
    )
    if current_receipt != receipt or current_attempt != attempt:
        raise RuntimeError("frozen batch or attempt evidence changed during training")
    artifact_path = Path(args.output_dir).resolve(strict=True)
    artifact = {
        "path": str(artifact_path),
        **hash_path(
            artifact_path, symlink_policy="reject", allowed_root=attempt_root
        ).to_dict(),
    }
    if artifact["file_count"] <= 0 or artifact["total_bytes"] <= 0:
        raise RuntimeError("backend produced an empty artifact")
    training_receipt = load_and_validate_training_receipt(
        training_receipt_path,
        expected={
            **evidence_bindings,
            "planned_global_steps": args.training_steps,
            "actual_global_steps": args.training_steps,
            "planned_optimizer_steps": args.training_steps,
            "actual_optimizer_steps": args.training_steps,
            "artifact_sha256": artifact["digest"],
        },
    )
    bindings = _manifest_bindings(receipt, attempt, args.model_id)
    body = {
        "schema_version": "1.0",
        "batch_id": args.batch_id,
        "model_id": args.model_id,
        "attempt_id": attempt["attempt_id"],
        "purpose": args.purpose,
        "status": "success",
        "exit_code": 0,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "training_steps": args.training_steps,
        "bindings": bindings,
        "artifact": artifact,
        "training_receipt": {
            "path": str(training_receipt_path.resolve(strict=True)),
            "file_sha256": sha256_file(training_receipt_path),
            "content_sha256": training_receipt["receipt_sha256"],
        },
    }
    atomic_write_json(
        manifest_path,
        {**body, "manifest_sha256": sha256_json(body)},
        root=attempt_root,
        overwrite=False,
    )
    return 0


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict catalog evaluation facade; unavailable backends fail closed."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--media-manifest", required=True)
    parser.add_argument("--media-manifest-sha256", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--modality", required=True, choices=("V", "M", "VM"))
    parser.add_argument(
        "--evaluation-mode",
        required=True,
        choices=("generative", "discriminative_abcd_scores"),
    )
    parser.add_argument("--do-sample")
    parser.add_argument("--temperature")
    parser.add_argument("--strict-answer-tags", action="store_true")
    parser.add_argument("--score-order")
    parser.add_argument("--limit", type=int)
    return parser


def evaluation_main(expected_model_ids: Iterable[str]) -> int:
    args = _evaluation_parser().parse_args()
    modality, mode, _initialization = _selected_model(args.model_id, expected_model_ids)
    if args.modality != modality or args.evaluation_mode != mode:
        raise ValueError("evaluation modality/mode differs from the model spec")
    if mode == "generative":
        if (args.do_sample, args.temperature, args.strict_answer_tags) != ("false", "0", True):
            raise ValueError("generative evaluation must be deterministic and strict-tagged")
        if args.score_order is not None:
            raise ValueError("generative evaluation cannot accept --score-order")
    elif args.score_order != "A,B,C,D" or any(
        value is not None for value in (args.do_sample, args.temperature)
    ) or args.strict_answer_tags:
        raise ValueError("discriminative evaluation requires only --score-order A,B,C,D")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    for value, label in (
        (args.benchmark_manifest, "benchmark manifest"),
        (args.media_manifest, "media manifest"),
    ):
        _existing_file(value, label)
    if _SHA256.fullmatch(args.media_manifest_sha256) is None:
        raise ValueError("--media-manifest-sha256 must be lowercase SHA-256")
    if sha256_file(args.media_manifest) != args.media_manifest_sha256:
        raise ValueError("media manifest hash mismatch")
    artifact = Path(args.artifact).resolve(strict=True)
    predictions = Path(args.predictions).resolve(strict=False)
    if predictions.exists():
        raise FileExistsError("predictions path already exists")
    backend = _require_backend(args.model_id, "evaluation")
    pretrained_root, batch_receipt, _manifest, _training = (
        _load_bound_finetune_artifact(
            artifact,
            batch_id=args.batch_id,
            model_id=args.model_id,
            require_evidence=True,
        )
    )
    _production_preflight(args.model_id, pretrained_root)
    result = backend.run_evaluation(args=args, artifact=artifact, predictions=predictions)
    if result not in (None, 0) or not predictions.is_file():
        raise RuntimeError("evaluation backend did not produce predictions")
    return 0


def _verifier_context(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    artifact = Path(args.artifact).resolve(strict=True)
    pretrained_root, receipt, manifest, _training = _load_bound_finetune_artifact(
        artifact,
        batch_id=args.batch_id,
        model_id=args.model_id,
        require_evidence=False,
    )
    if (
        manifest.get("attempt_id") != args.attempt_id
        or manifest.get("artifact", {}).get("digest") != args.artifact_sha256
    ):
        raise ValueError("verifier arguments differ from frozen attempt identity")
    return pretrained_root, receipt


def verifier_main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent catalog artifact reload verifier."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    _safe_id(args.batch_id, "batch id")
    _safe_id(args.model_id, "model id")
    _safe_id(args.attempt_id, "attempt id")
    _selected_model(args.model_id, MODEL_SPECS)
    if _SHA256.fullmatch(args.artifact_sha256) is None:
        raise ValueError("--artifact-sha256 must be lowercase SHA-256")
    artifact = Path(args.artifact).resolve(strict=True)
    if hash_path(artifact).digest != args.artifact_sha256:
        raise ValueError("artifact hash mismatch before reload")
    report = Path(args.report).resolve(strict=False)
    if report.exists() or report.parent != artifact.parent:
        raise ValueError("fresh verifier report must be inside the attempt directory")
    pretrained_root, _receipt = _verifier_context(args)
    backend = _require_backend(args.model_id, "verifier")
    _production_preflight(args.model_id, pretrained_root)
    verified = backend.verify_reload(
        artifact=artifact,
        pretrained_root=pretrained_root,
    )
    if verified is not True:
        raise RuntimeError("backend did not prove a fresh strict artifact reload")
    payload = {
        "schema_version": "1.0",
        "status": "passed",
        "batch_id": args.batch_id,
        "model_id": args.model_id,
        "attempt_id": args.attempt_id,
        "artifact_digest": args.artifact_sha256,
        "checker": f"{args.model_id}:catalog-reload",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(report, payload, root=artifact.parent, overwrite=False)
    return 0
