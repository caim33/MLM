"""Shared SFT artifact policy and distributed publication."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import operator
import os
import re
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from motion_eval.core import (
    atomic_write_json,
    bytecode_source as _bytecode_source,
    formal_source_role_files,
    hash_path,
    is_link_or_reparse as _is_link_or_reparse,
    resolve_within_root,
    sha256_file,
    sha256_json,
    stable_source_files as _stable_source_files,
)
from motion_eval.data import load_json_strict
from motion_eval.training_receipt import make_training_receipt, write_training_receipt

from .artifact import (
    ArtifactBinding,
    ArtifactProvenancePaths,
    FinetuneArtifactReceipt,
    binding_from_provenance_evidence,
    compute_verified_provenance,
    validate_resume_artifact,
    write_finetune_artifact_manifest,
)
from .runtime import distributed_rank

_PROVENANCE_ARGUMENTS = {
    "base_artifact": "base_artifact_path",
    "train_data": "train_data_path",
    "validation_data": "validation_data_path",
    "benchmark": "benchmark_path",
    "leakage_audit": "leakage_audit_path",
    "config": "config_path",
    "code": "code_path",
    "environment": "environment_path",
}
_OPTIONAL_PROVENANCE_ARGUMENTS = {
    "runner_code": "runner_code_path",
    "motion_vqvae": "motion_vqvae_asset_path",
}


@dataclass(frozen=True)
class SupervisedDataBindingReceipt:
    train_dataset_use: str
    train_data_path: Path
    validation_dataset_use: str
    validation_data_path: Path


@dataclass(frozen=True)
class BaseArtifactBindingReceipt:
    model_path: Path
    provenance_path: Path
    digest: str


@dataclass(frozen=True)
class CanonicalSFTIdentity:
    model_id: str
    model_family: str
    modality: str
    base_artifact: Path
    code_root: Path
    runner_code_root: Path
    environment_root: Path
    interpreter: Path
    base_environment_root: Path | None = None


@dataclass(frozen=True)
class FormalProvenanceSnapshot:
    """Secondary in-process snapshot; never a pre-spawn execution attestation."""

    path: Path
    file_sha256: str
    snapshot_sha256: str
    binding: ArtifactBinding
    evidence: Mapping[str, Mapping[str, Any]]


_SOURCE_SNAPSHOT_SCHEMA = "motionllm-inprocess-provenance-v2"
_SOURCE_MANIFEST_SCHEMA = "motionllm-source-allowlist-v2"
_ENVIRONMENT_MANIFEST_SCHEMA = "motionllm-installed-environment-v2"
_DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")


_QWEN_REGISTRY_FAMILIES = {
    "motionr1_vm_lora": ("qwen3_vl_motion", "VM"),
    "qwen3vl_8b_lora": ("qwen3_vl", "V"),
    "qwen3vl_4b_lora": ("qwen3_vl", "V"),
    "qwen36_27b_lora": ("qwen3_vl_moe", "V"),
    "qwen35_4b_lora": ("qwen3_vl", "V"),
}


def bind_canonical_formal_identity(
    model_arguments: Any,
    artifact_arguments: Any,
    *,
    model_spec: Any,
    formal_artifact: bool,
) -> CanonicalSFTIdentity | None:
    """Bind a Qwen entrypoint to its checkout, runner, environment, and registries."""

    if not formal_artifact:
        return None
    project_root = Path(__file__).resolve().parents[3]
    code_root = Path(artifact_arguments.code_path).resolve(strict=True)
    if code_root != project_root:
        raise ValueError(
            "formal SFT code_path must be the actual checkout containing the imported code"
        )
    interpreter_entry = Path(os.path.abspath(sys.executable))
    interpreter = interpreter_entry.resolve(strict=True)
    environment_root = Path(sys.prefix).resolve(strict=True)
    base_environment_root = Path(sys.base_prefix).resolve(strict=True)
    if environment_root == base_environment_root:
        raise ValueError("formal SFT requires an isolated Python environment")
    environment = Path(artifact_arguments.environment_path).resolve(strict=True)
    if not environment.is_dir() or environment != environment_root:
        raise ValueError(
            "formal SFT environment_path must be the current isolated environment root"
        )
    lexical_environment_root = Path(os.path.abspath(sys.prefix))
    try:
        interpreter_entry.relative_to(lexical_environment_root)
    except ValueError as exc:
        raise ValueError(
            "formal SFT interpreter entry must be inside the frozen environment root"
        ) from exc

    runner_code_root = project_root / "qwenvl"
    runner_code = Path(artifact_arguments.runner_code_path)
    if not runner_code.is_absolute():
        raise ValueError("formal SFT runner_code_path must be absolute")
    runner_code = Path(os.path.abspath(runner_code))
    if not runner_code.is_dir() or runner_code != runner_code_root:
        raise ValueError(
            "formal SFT runner_code_path must be the actual checkout's qwenvl tree"
        )

    registry_path = project_root / "model_evaluation_agent" / "model_registry.json"
    pretrained_path = (
        project_root / "model_evaluation_agent" / "pretrained_registry.json"
    )
    registry = load_json_strict(registry_path)
    pretrained = load_json_strict(pretrained_path)
    if not isinstance(registry, Mapping) or set(registry) != {
        "schema_version", "updated_at", "fresh_finetune_required_per_batch",
        "global_finetune_barrier_before_eval", "models",
    }:
        raise ValueError("canonical model registry schema is invalid")
    if not isinstance(pretrained, Mapping) or not isinstance(
        pretrained.get("models"), list
    ):
        raise ValueError("canonical pretrained registry schema is invalid")

    model_id = getattr(artifact_arguments, "model_registry_id", None)
    try:
        expected_family, expected_modality = _QWEN_REGISTRY_FAMILIES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"model_registry_id {model_id!r} is not supported by the Qwen SFT entrypoint"
        ) from exc
    registry_matches = [
        item
        for item in registry.get("models", [])
        if isinstance(item, Mapping) and item.get("id") == model_id
    ]
    pretrained_matches = [
        item
        for item in pretrained["models"]
        if isinstance(item, Mapping) and item.get("id") == model_id
    ]
    if len(registry_matches) != 1 or len(pretrained_matches) != 1:
        raise ValueError("model identity is not unique in both canonical registries")
    if registry_matches[0].get("main_modality") != expected_modality:
        raise ValueError("canonical model modality disagrees with the Qwen binding")
    actual_family = getattr(model_arguments, "model_family", None)
    spec_family = getattr(getattr(model_spec, "family", None), "value", None)
    if actual_family != expected_family or spec_family != expected_family:
        raise ValueError("model_family differs from the canonical registry/model factory binding")
    supports_motion = bool(getattr(model_spec, "supports_motion", False))
    if supports_motion != ("M" in expected_modality):
        raise ValueError("loaded model capability disagrees with canonical modality")

    artifacts = pretrained_matches[0].get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("canonical pretrained entry has no artifact list")
    base_entries = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("role") in {"base_and_processor", "base_processor_and_towers"}
    ]
    if len(base_entries) != 1 or not isinstance(base_entries[0].get("path"), str):
        raise ValueError("canonical pretrained entry has no unique Qwen base artifact")
    remote_root_raw = pretrained.get("remote_root")
    if not isinstance(remote_root_raw, str) or not remote_root_raw:
        raise ValueError("canonical pretrained registry has no remote_root")
    remote_root = Path(remote_root_raw)
    registered_relative = Path(base_entries[0]["path"])
    if (
        not remote_root.is_absolute()
        or registered_relative.is_absolute()
        or any(part in {"", ".", ".."} for part in registered_relative.parts)
    ):
        raise ValueError("canonical pretrained base artifact path is unsafe")
    base_raw = Path(artifact_arguments.base_artifact_path)
    if not base_raw.is_absolute():
        raise ValueError("formal SFT base_artifact_path must be absolute")
    base = Path(os.path.abspath(base_raw))
    registered_base = Path(os.path.abspath(remote_root / registered_relative))
    if base != registered_base:
        raise ValueError(
            "base_artifact_path must exactly equal remote_root plus the registered artifact path"
        )
    try:
        registered_base.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("canonical pretrained base artifact does not exist") from exc
    return CanonicalSFTIdentity(
        model_id=model_id,
        model_family=expected_family,
        modality=expected_modality,
        base_artifact=base,
        code_root=code_root,
        runner_code_root=runner_code_root,
        environment_root=environment_root,
        interpreter=interpreter,
        base_environment_root=base_environment_root,
    )


def _source_manifest(
    identity: CanonicalSFTIdentity, *, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = identity.code_root
    manifest_root, records = formal_source_role_files(
        project_root, runner_root=identity.runner_code_root, role=role
    )
    body = {
        "schema_version": _SOURCE_MANIFEST_SCHEMA,
        "role": role,
        "root": str(manifest_root),
        "files": list(records),
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    evidence = {
        "path": str(manifest_root),
        "algorithm": _SOURCE_MANIFEST_SCHEMA,
        "kind": "source-allowlist",
        "digest": manifest["manifest_sha256"],
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
    }
    return manifest, evidence


def _decode_record_sha256(value: str, *, field: str) -> str:
    if not value.startswith("sha256="):
        raise ValueError(f"formal environment rejects non-SHA256 RECORD hash: {field}")
    encoded = value.split("=", 1)[1]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as exc:
        raise ValueError(f"invalid RECORD digest: {field}") from exc
    if len(raw) != 32:
        raise ValueError(f"invalid RECORD SHA-256 length: {field}")
    return raw.hex()


def _environment_file_digest(path: Path, environment_root: Path) -> tuple[str, int]:
    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(environment_root)
    except ValueError as exc:
        raise ValueError(f"environment file escapes sys.prefix: {path}") from exc
    if _is_link_or_reparse(lexical):
        raise ValueError(f"formal environment rejects linked package file: {path}")
    digest = hash_path(
        lexical, symlink_policy="reject", allowed_root=environment_root
    )
    if digest.kind != "file" or digest.file_count != 1:
        raise ValueError(f"environment entry is not a regular file: {path}")
    return digest.digest, digest.total_bytes


def _runtime_loading_environment() -> dict[str, Any]:
    forbidden = {
        "LD_PRELOAD",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONINSPECT",
        "PYTHONEXEC",
    }
    safe_python = {
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
    }
    for name, value in os.environ.items():
        if value and (name in forbidden or (name.startswith("PYTHON") and name not in safe_python)):
            raise ValueError(
                f"formal environment rejects Python/native-loader injection variable: {name}"
            )
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    normalized_library_paths: list[str] = []
    for raw in library_path.split(os.pathsep) if library_path else ():
        candidate = Path(raw)
        if not candidate.is_absolute() or not candidate.is_dir():
            raise ValueError("formal LD_LIBRARY_PATH entries must be existing absolute directories")
        if _is_link_or_reparse(candidate):
            raise ValueError(f"formal LD_LIBRARY_PATH rejects linked directory: {candidate}")
        normalized_library_paths.append(str(candidate.resolve(strict=True)))
    bound_names = {
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
    }
    bound_names.update(name for name in os.environ if name.startswith("NCCL_"))
    bound_names.update(name for name in safe_python if name in os.environ)
    return {
        "variables": {name: os.environ.get(name) for name in sorted(bound_names)},
        "normalized_ld_library_path": normalized_library_paths,
    }


def _validate_pth_files(site_roots: Sequence[Path], environment_root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for site_root in sorted(set(site_roots)):
        for egg_link in site_root.glob("*.egg-link"):
            raise ValueError(f"formal environment rejects legacy editable egg-link: {egg_link}")
        for pth in sorted(site_root.glob("*.pth")):
            if _is_link_or_reparse(pth) or not pth.is_file():
                raise ValueError(f"formal environment rejects linked/non-file .pth: {pth}")
            accepted: list[str] = []
            for number, raw in enumerate(
                pth.read_text(encoding="utf-8", errors="strict").splitlines(), 1
            ):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("import ") or line.startswith("import\t"):
                    raise ValueError(
                        f"formal environment rejects executable .pth line: {pth}:{number}"
                    )
                logical = Path(os.path.abspath(pth.parent / line))
                try:
                    logical.relative_to(environment_root)
                except ValueError as exc:
                    raise ValueError(
                        f"formal environment rejects external .pth path: {pth}:{number}"
                    ) from exc
                if not logical.is_dir() or _is_link_or_reparse(logical):
                    raise ValueError(
                        f"formal environment rejects invalid .pth path: {pth}:{number}"
                    )
                accepted.append(str(logical))
            digest = hash_path(pth, symlink_policy="reject", allowed_root=environment_root)
            receipts.append(
                {
                    "relative_path": pth.relative_to(environment_root).as_posix(),
                    "sha256": digest.digest,
                    "accepted_paths": accepted,
                }
            )
    return receipts


def _hash_base_stdlib(base_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    raw = sysconfig.get_path("stdlib")
    if not isinstance(raw, str) or not raw:
        raise ValueError("formal environment cannot locate the base stdlib")
    stdlib = Path(raw).resolve(strict=True)
    try:
        stdlib.relative_to(base_root)
    except ValueError as exc:
        raise ValueError("formal base stdlib escapes sys.base_prefix") from exc
    records: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(stdlib, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = base / name
            if name in {"site-packages", "dist-packages"}:
                continue
            if _is_link_or_reparse(child):
                raise ValueError(f"formal base stdlib rejects linked directory: {child}")
            kept.append(name)
        directory_names[:] = kept
        for name in sorted(file_names):
            child = base / name
            if child.suffix.lower() in {".pyc", ".pyo"}:
                source = _bytecode_source(child)
                if source is None or not source.is_file():
                    raise ValueError(f"formal base stdlib rejects sourceless bytecode: {child}")
            if _is_link_or_reparse(child) or not child.is_file():
                raise ValueError(f"formal base stdlib rejects non-regular file: {child}")
            digest = hash_path(child, symlink_policy="reject", allowed_root=base_root)
            records.append(
                {
                    "relative_path": child.relative_to(base_root).as_posix(),
                    "sha256": digest.digest,
                    "size": digest.total_bytes,
                }
            )
    if not records:
        raise ValueError("formal base stdlib inventory is empty")
    return stdlib, records


def _loaded_native_runtime_inventory(
    *, allowed_roots: Sequence[Path]
) -> list[dict[str, Any]]:
    canonical_roots = tuple(root.resolve(strict=True) for root in allowed_roots)
    if not canonical_roots:
        raise ValueError("formal native runtime allowlist is empty")
    candidates: set[Path] = set()
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="strict").splitlines():
            fields = line.split()
            if not fields:
                continue
            raw = fields[-1]
            if not raw.startswith("/") or raw.endswith(" (deleted)"):
                continue
            lowered = Path(raw).name.lower()
            if ".so" in lowered or lowered.endswith((".dylib", ".dll", ".pyd")):
                candidates.add(Path(raw))
    else:
        for module in tuple(sys.modules.values()):
            raw = getattr(module, "__file__", None)
            if isinstance(raw, str) and raw.lower().endswith((".so", ".dylib", ".dll", ".pyd")):
                candidates.add(Path(raw))
    records: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: str(item)):
        lexical = Path(os.path.abspath(candidate))
        if not lexical.exists():
            raise ValueError(f"formal native runtime is missing: {lexical}")
        link_sha256 = None
        if _is_link_or_reparse(lexical):
            link_sha256 = hash_path(lexical, symlink_policy="link").digest
        target = lexical.resolve(strict=True)
        if not target.is_file() or _is_link_or_reparse(target):
            raise ValueError(f"formal native runtime target is not regular: {target}")
        if not any(target == root or root in target.parents for root in canonical_roots):
            raise ValueError(
                f"formal native runtime target escapes frozen runtime roots: {target}"
            )
        digest = hash_path(target, symlink_policy="reject")
        records.append(
            {
                "logical_path": str(lexical),
                "link_sha256": link_sha256,
                "resolved_target": str(target),
                "sha256": digest.digest,
                "size": digest.total_bytes,
            }
        )
    if not records:
        raise ValueError("formal environment cannot inventory loaded native runtimes")
    return records


def _import_runtime_inventory(
    *, allowed_roots: Sequence[Path], allowed_files: Sequence[Path]
) -> tuple[list[str], list[dict[str, Any]]]:
    normalized_paths: list[str] = []
    canonical_roots = tuple(root.resolve(strict=True) for root in allowed_roots)
    canonical_files = {
        os.path.normcase(str(path.resolve(strict=True))) for path in allowed_files
    }

    def require_exact_root(candidate: Path) -> Path:
        resolved = candidate.resolve(strict=True)
        if resolved not in canonical_roots:
            raise ValueError(
                f"formal sys.path is not an exact inventoried import root: {resolved}"
            )
        return resolved

    def require_inventoried_file(candidate: Path) -> Path:
        resolved = candidate.resolve(strict=True)
        if os.path.normcase(str(resolved)) not in canonical_files:
            raise ValueError(
                "formal sys.meta_path origin is not an exact inventoried file: "
                f"{resolved}"
            )
        return resolved

    for raw in sys.path:
        candidate = Path(raw or os.getcwd())
        if candidate.suffix.lower() in {".zip", ".egg", ".whl"}:
            raise ValueError(f"formal environment rejects archive import path: {candidate}")
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"formal sys.path entry is missing/not a directory: {candidate}")
        normalized_paths.append(str(require_exact_root(candidate)))

    finders: list[dict[str, Any]] = []
    originless_finders = {
        ("_frozen_importlib", "BuiltinImporter"),
        ("_frozen_importlib", "FrozenImporter"),
    }
    for finder in sys.meta_path:
        finder_type = finder if isinstance(finder, type) else type(finder)
        module_name = getattr(finder_type, "__module__", None)
        qualname = getattr(finder_type, "__qualname__", None)
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            raise ValueError("formal sys.meta_path finder identity is invalid")
        module = sys.modules.get(module_name)
        origin_raw = getattr(module, "__file__", None) if module is not None else None
        origin = None
        origin_sha256 = None
        if isinstance(origin_raw, str):
            candidate = Path(origin_raw)
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                source = _bytecode_source(candidate)
                if source is not None and source.is_file():
                    candidate = source
            candidate = require_inventoried_file(candidate)
            digest = hash_path(candidate, symlink_policy="reject")
            origin = str(candidate)
            origin_sha256 = digest.digest
        elif (module_name, qualname) not in originless_finders:
            raise ValueError(
                f"formal sys.meta_path finder lacks a frozen origin: {module_name}.{qualname}"
            )
        finders.append(
            {
                "module": module_name,
                "qualname": qualname,
                "origin": origin,
                "origin_sha256": origin_sha256,
            }
        )
    return normalized_paths, finders


def _environment_manifest(
    identity: CanonicalSFTIdentity,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash actual installed files and verify every usable RECORD SHA-256."""

    root = identity.environment_root.resolve(strict=True)
    actual_prefix = Path(sys.prefix).resolve(strict=True)
    if root != actual_prefix:
        raise ValueError("formal environment root differs from actual worker sys.prefix")
    base_root = (
        identity.base_environment_root.resolve(strict=True)
        if identity.base_environment_root is not None
        else Path(sys.base_prefix).resolve(strict=True)
    )
    if base_root != Path(sys.base_prefix).resolve(strict=True):
        raise ValueError("formal base environment differs from actual worker sys.base_prefix")
    if root == base_root:
        raise ValueError("formal environment requires an isolated sys.prefix")
    loading_environment = _runtime_loading_environment()
    pyvenv_path = root / "pyvenv.cfg"
    if not pyvenv_path.is_file() or _is_link_or_reparse(pyvenv_path):
        raise ValueError("formal environment requires a regular pyvenv.cfg")
    pyvenv_values: dict[str, str] = {}
    for number, raw in enumerate(
        pyvenv_path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "=" not in raw:
            raise ValueError(f"formal pyvenv.cfg line is invalid: {number}")
        name, value = raw.split("=", 1)
        normalized = name.strip().lower()
        if not normalized or normalized in pyvenv_values:
            raise ValueError(f"formal pyvenv.cfg key is invalid/duplicated: {number}")
        pyvenv_values[normalized] = value.strip()
    if pyvenv_values.get("include-system-site-packages", "").lower() != "false":
        raise ValueError("formal environment requires include-system-site-packages=false")
    dist_infos: list[Path] = []
    site_roots: set[Path] = set()

    for directory, directory_names, _file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = base / name
            if name == "__pycache__":
                if _is_link_or_reparse(child) or not child.is_dir():
                    raise ValueError(
                        f"formal environment rejects linked/non-directory bytecode cache: {child}"
                    )
                continue
            if _is_link_or_reparse(child):
                raise ValueError(
                    f"formal environment rejects directory link/reparse: {child}"
                )
            if name.endswith(".dist-info"):
                dist_infos.append(child)
                site_roots.add(base)
            kept.append(name)
        directory_names[:] = kept
    if not dist_infos:
        raise ValueError("formal environment has no installed .dist-info distributions")
    pth_receipts = _validate_pth_files(tuple(site_roots), root)

    declared: dict[Path, tuple[str | None, int | None, str]] = {}
    distributions: list[dict[str, Any]] = []
    for dist_info in sorted(dist_infos, key=lambda item: item.as_posix()):
        metadata_path = dist_info / "METADATA"
        record_path = dist_info / "RECORD"
        if not metadata_path.is_file() or not record_path.is_file():
            raise ValueError(f"installed distribution lacks METADATA/RECORD: {dist_info}")
        metadata_text = metadata_path.read_text(encoding="utf-8", errors="strict")
        name = next(
            (line[6:].strip() for line in metadata_text.splitlines() if line.startswith("Name: ")),
            None,
        )
        version = next(
            (line[9:].strip() for line in metadata_text.splitlines() if line.startswith("Version: ")),
            None,
        )
        if not name or not version:
            raise ValueError(f"installed distribution metadata lacks name/version: {dist_info}")
        direct_url_path = dist_info / "direct_url.json"
        if direct_url_path.is_file():
            try:
                direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"installed distribution has invalid direct_url.json: {dist_info}"
                ) from exc
            if (
                isinstance(direct_url, Mapping)
                and isinstance(direct_url.get("dir_info"), Mapping)
                and direct_url["dir_info"].get("editable") is True
            ):
                raise ValueError(
                    f"formal environment rejects editable installation: {name}=={version}"
                )
        record_bytes = record_path.read_bytes()
        rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict"))))
        if not rows:
            raise ValueError(f"installed distribution RECORD is empty: {record_path}")
        for number, row in enumerate(rows, 1):
            if len(row) != 3 or not row[0]:
                raise ValueError(f"invalid RECORD row {number}: {record_path}")
            logical = Path(os.path.abspath(dist_info.parent / Path(row[0])))
            try:
                logical.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"installed distribution RECORD escapes sys.prefix: {record_path}:{number}"
                ) from exc
            if not logical.is_file():
                raise ValueError(
                    f"installed distribution RECORD file is missing: {logical}"
                )
            expected_digest = (
                _decode_record_sha256(row[1], field=f"{record_path}:{number}")
                if row[1]
                else None
            )
            try:
                expected_size = int(row[2]) if row[2] else None
            except ValueError as exc:
                raise ValueError(f"invalid RECORD size: {record_path}:{number}") from exc
            prior = declared.get(logical)
            current = (expected_digest, expected_size, f"{name}=={version}")
            if prior is not None and prior[:2] != current[:2]:
                raise ValueError(f"conflicting RECORD declarations for {logical}")
            declared[logical] = current
        distributions.append(
            {
                "name": name,
                "version": version,
                "dist_info": dist_info.relative_to(root).as_posix(),
                "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "record_file_count": len(rows),
            }
        )

    selected: set[Path] = set(declared)
    for site_root in sorted(site_roots):
        for directory, directory_names, file_names in os.walk(
            site_root, followlinks=False
        ):
            base = Path(directory)
            kept: list[str] = []
            for name in sorted(directory_names):
                child = base / name
                if name == "__pycache__":
                    if _is_link_or_reparse(child) or not child.is_dir():
                        raise ValueError(
                            f"formal environment rejects linked/non-directory bytecode cache: {child}"
                        )
                    for bytecode in sorted(child.iterdir()):
                        if (
                            _is_link_or_reparse(bytecode)
                            or not bytecode.is_file()
                            or bytecode.suffix.lower() not in {".pyc", ".pyo"}
                        ):
                            raise ValueError(
                                f"formal environment rejects invalid bytecode cache entry: {bytecode}"
                            )
                        source = _bytecode_source(bytecode)
                        if (
                            source is None
                            or not source.is_file()
                            or _is_link_or_reparse(source)
                        ):
                            raise ValueError(
                                f"formal environment rejects sourceless bytecode: {bytecode}"
                            )
                        selected.add(Path(os.path.abspath(bytecode)))
                    continue
                if _is_link_or_reparse(child):
                    raise ValueError(
                        f"formal environment rejects directory link/reparse: {child}"
                    )
                kept.append(name)
            directory_names[:] = kept
            for name in sorted(file_names):
                child = Path(os.path.abspath(base / name))
                if child.suffix.lower() in {".pyc", ".pyo"}:
                    source = _bytecode_source(child)
                    if (
                        source is None
                        or not source.is_file()
                        or _is_link_or_reparse(source)
                    ):
                        raise ValueError(
                            f"formal environment rejects sourceless bytecode: {child}"
                        )
                if _is_link_or_reparse(child):
                    raise ValueError(
                        f"formal environment rejects linked site-package file: {child}"
                    )
                selected.add(child)

    selected.add(Path(os.path.abspath(pyvenv_path)))
    scripts_root = root / ("Scripts" if os.name == "nt" else "bin")
    if scripts_root.is_dir():
        for child in scripts_root.iterdir():
            if child.is_file() and not _is_link_or_reparse(child):
                selected.add(Path(os.path.abspath(child)))

    files: list[dict[str, Any]] = []
    for child in sorted(selected, key=lambda item: item.as_posix()):
        digest, size = _environment_file_digest(child, root)
        expected_digest, expected_size, owner = declared.get(
            child, (None, None, "unowned-direct-content")
        )
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"installed file differs from RECORD SHA-256: {child}")
        if expected_size is not None and size != expected_size:
            raise ValueError(f"installed file differs from RECORD size: {child}")
        files.append(
            {
                "relative_path": child.relative_to(root).as_posix(),
                "sha256": digest,
                "size": size,
                "owner": owner,
                "record_sha256": expected_digest,
            }
        )

    interpreter_entry = Path(os.path.abspath(sys.executable))
    if interpreter_entry.resolve(strict=True) != identity.interpreter.resolve(strict=True):
        raise ValueError("formal interpreter differs from actual worker sys.executable")
    interpreter_link = None
    if _is_link_or_reparse(interpreter_entry):
        link_digest = hash_path(interpreter_entry, symlink_policy="link")
        interpreter_link = link_digest.digest
    interpreter_target = interpreter_entry.resolve(strict=True)
    interpreter_digest = hash_path(interpreter_target, symlink_policy="reject")
    stdlib_root, stdlib_files = _hash_base_stdlib(base_root)
    import_roots = [
        identity.code_root / "src",
        identity.code_root / "models",
        identity.runner_code_root,
        *sorted(site_roots, key=lambda item: str(item)),
        stdlib_root,
    ]
    for optional_import_root in (
        identity.runner_code_root / "train",
        stdlib_root / "lib-dynload",
    ):
        if optional_import_root.is_dir():
            import_roots.append(optional_import_root)
    inventoried_import_files = [
        root / row["relative_path"]
        for role in ("code", "runner_code")
        for root, rows in (
            formal_source_role_files(
                identity.code_root,
                runner_root=identity.runner_code_root,
                role=role,
            ),
        )
        for row in rows
    ]
    inventoried_import_files.extend(
        root / row["relative_path"] for row in files
    )
    inventoried_import_files.extend(
        base_root / row["relative_path"] for row in stdlib_files
    )
    sys_path_entries, meta_path_finders = _import_runtime_inventory(
        allowed_roots=import_roots,
        allowed_files=inventoried_import_files,
    )
    native_allowed_roots: set[Path] = {root, base_root}
    for raw in loading_environment["normalized_ld_library_path"]:
        native_allowed_roots.add(Path(raw).resolve(strict=True))
    for name in ("CUDA_HOME", "CUDA_PATH"):
        raw = loading_environment["variables"].get(name)
        if raw in (None, ""):
            continue
        candidate = Path(raw)
        if (
            not candidate.is_absolute()
            or not candidate.is_dir()
            or _is_link_or_reparse(candidate)
        ):
            raise ValueError(f"formal {name} must be an existing unlinked absolute directory")
        native_allowed_roots.add(candidate.resolve(strict=True))
    ordered_native_roots = sorted(
        (str(item) for item in native_allowed_roots), key=os.path.normcase
    )
    native_runtime_files = _loaded_native_runtime_inventory(
        allowed_roots=tuple(Path(item) for item in ordered_native_roots)
    )
    body = {
        "schema_version": _ENVIRONMENT_MANIFEST_SCHEMA,
        "environment_root": str(root),
        "base_environment_root": str(base_root),
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "interpreter_entry": str(interpreter_entry),
        "interpreter_entry_link_sha256": interpreter_link,
        "interpreter_target": str(interpreter_target),
        "interpreter_target_sha256": interpreter_digest.digest,
        "interpreter_target_size": interpreter_digest.total_bytes,
        "pyvenv_sha256": sha256_file(pyvenv_path),
        "pyvenv": pyvenv_values,
        "pth_files": pth_receipts,
        "loading_environment": loading_environment,
        "sys_path": sys_path_entries,
        "meta_path": meta_path_finders,
        "stdlib_root": str(stdlib_root),
        "stdlib_files": stdlib_files,
        "native_runtime_files": native_runtime_files,
        "native_allowed_roots": ordered_native_roots,
        "internal_links": [],
        "distributions": distributions,
        "files": files,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    evidence = {
        "path": str(root),
        "algorithm": _ENVIRONMENT_MANIFEST_SCHEMA,
        "kind": "installed-environment-manifest",
        "digest": manifest["manifest_sha256"],
        "file_count": (
            len(files)
            + len(stdlib_files)
            + len(native_runtime_files)
            + 1
        ),
        "total_bytes": sum(item["size"] for item in files)
        + sum(item["size"] for item in stdlib_files)
        + sum(item["size"] for item in native_runtime_files)
        + interpreter_digest.total_bytes,
    }
    return manifest, evidence


def _plain_provenance_evidence(path: str | Path, *, role: str) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    digest = hash_path(candidate, symlink_policy="follow")
    if digest.file_count <= 0 or digest.total_bytes <= 0:
        raise ValueError(f"formal provenance {role} is empty")
    return {"path": str(candidate), **digest.to_dict()}


def _snapshot_identity(identity: CanonicalSFTIdentity) -> dict[str, Any]:
    return {
        "model_id": identity.model_id,
        "model_family": identity.model_family,
        "modality": identity.modality,
        "base_artifact": str(identity.base_artifact.resolve(strict=True)),
        "code_root": str(identity.code_root),
        "runner_code_root": str(identity.runner_code_root),
        "environment_root": str(identity.environment_root),
        "interpreter": str(identity.interpreter),
        "base_environment_root": (
            str(identity.base_environment_root)
            if identity.base_environment_root is not None
            else str(Path(sys.base_prefix).resolve(strict=True))
        ),
    }


def _compute_formal_snapshot_body(
    arguments: Any,
    *,
    canonical_identity: CanonicalSFTIdentity,
    training_mode: str,
) -> dict[str, Any]:
    paths = provenance_paths_from_arguments(arguments)
    code_manifest, code_evidence = _source_manifest(
        canonical_identity, role="code"
    )
    runner_manifest, runner_evidence = _source_manifest(
        canonical_identity, role="runner_code"
    )
    environment_manifest, environment_evidence = _environment_manifest(
        canonical_identity
    )
    evidence: dict[str, dict[str, Any]] = {}
    for role, path in paths.to_dict().items():
        if role == "code":
            evidence[role] = code_evidence
        elif role == "runner_code":
            evidence[role] = runner_evidence
        elif role == "environment":
            evidence[role] = environment_evidence
        else:
            evidence[role] = _plain_provenance_evidence(path, role=role)
    batch_id, model_id = _identity_from_arguments(arguments)
    binding_from_provenance_evidence(
        paths,
        evidence,
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
    )
    return {
        "schema_version": _SOURCE_SNAPSHOT_SCHEMA,
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": batch_id,
        "model_id": model_id,
        "training_mode": training_mode,
        "canonical_identity": _snapshot_identity(canonical_identity),
        "provenance": evidence,
        "manifests": {
            "code": code_manifest,
            "runner_code": runner_manifest,
            "environment": environment_manifest,
        },
    }


def _formal_snapshot_path(arguments: Any) -> Path:
    root = getattr(arguments, "artifact_root", None)
    training_receipt = getattr(arguments, "training_receipt_path", None)
    if root in (None, "") or training_receipt in (None, ""):
        raise ValueError("formal provenance snapshot requires artifact/training receipt paths")
    receipt_path = resolve_within_root(
        training_receipt, root, must_exist=False, allow_root=False
    )
    return resolve_within_root(
        receipt_path.with_name("formal_provenance_snapshot.json"),
        root,
        must_exist=False,
        allow_root=False,
    )


def _ensure_formal_distributed(torch_module: Any) -> Any | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or not distributed.is_available():
        raise RuntimeError("formal multi-rank provenance requires torch.distributed")
    if not distributed.is_initialized():
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not cuda.is_available():
            raise RuntimeError("formal multi-rank Qwen SFT requires CUDA/NCCL")
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise RuntimeError("formal multi-rank provenance requires LOCAL_RANK")
        cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
    for method in ("get_rank", "get_world_size", "broadcast_object_list", "barrier"):
        if not callable(getattr(distributed, method, None)):
            raise RuntimeError(f"formal distributed provenance lacks {method}")
    if int(distributed.get_world_size()) != world_size:
        raise RuntimeError("formal distributed world size differs from launcher")
    return distributed


def _validate_snapshot_payload(
    payload: Any,
    *,
    path: Path,
    arguments: Any,
    canonical_identity: CanonicalSFTIdentity,
    training_mode: str,
) -> FormalProvenanceSnapshot:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "status", "batch_id", "model_id", "training_mode",
        "canonical_identity", "provenance", "manifests", "snapshot_sha256",
    }:
        raise ValueError("formal provenance snapshot schema is invalid")
    body = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    snapshot_sha256 = payload.get("snapshot_sha256")
    if (
        payload.get("schema_version") != _SOURCE_SNAPSHOT_SCHEMA
        or payload.get("status")
        != "captured_before_model_data_load_after_entrypoint_imports"
        or not isinstance(snapshot_sha256, str)
        or _DIGEST_RE.fullmatch(snapshot_sha256) is None
        or snapshot_sha256 != sha256_json(body)
    ):
        raise ValueError("formal provenance snapshot self-hash is invalid")
    if payload.get("canonical_identity") != _snapshot_identity(canonical_identity):
        raise ValueError("formal provenance snapshot canonical identity changed")
    batch_id, model_id = _identity_from_arguments(arguments)
    if (
        payload.get("batch_id") != batch_id
        or payload.get("model_id") != model_id
        or payload.get("training_mode") != training_mode
    ):
        raise ValueError("formal provenance snapshot run identity changed")
    evidence = payload.get("provenance")
    if not isinstance(evidence, Mapping):
        raise ValueError("formal provenance snapshot evidence is invalid")
    binding = binding_from_provenance_evidence(
        provenance_paths_from_arguments(arguments),
        evidence,
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
    )
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {
        "code", "runner_code", "environment"
    }:
        raise ValueError("formal provenance detailed manifests are incomplete")
    for role in manifests:
        manifest = manifests[role]
        if not isinstance(manifest, Mapping) or manifest.get("manifest_sha256") != sha256_json(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ):
            raise ValueError(f"formal provenance {role} manifest hash is invalid")
        if evidence[role]["digest"] != manifest["manifest_sha256"]:
            raise ValueError(f"formal provenance {role} evidence differs from manifest")
    return FormalProvenanceSnapshot(
        path=path,
        file_sha256=sha256_file(path),
        snapshot_sha256=snapshot_sha256,
        binding=binding,
        evidence={role: dict(value) for role, value in evidence.items()},
    )


def capture_formal_provenance_snapshot(
    arguments: Any,
    *,
    canonical_identity: CanonicalSFTIdentity | None,
    training_mode: str,
    formal_artifact: bool,
    torch_module: Any,
) -> FormalProvenanceSnapshot | None:
    """Capture a secondary pre-model snapshot after entrypoint imports.

    Formal publication is blocked by require_controller_verified_formal_bootstrap;
    this diagnostic is not evidence of the bytes that already executed.
    """

    if not formal_artifact:
        return None
    if canonical_identity is None:
        raise ValueError("formal provenance capture requires canonical identity")
    distributed = _ensure_formal_distributed(torch_module)
    rank = int(distributed.get_rank()) if distributed is not None else 0
    destination = _formal_snapshot_path(arguments)
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            if destination.exists():
                raise FileExistsError("formal provenance snapshot must be fresh")
            body = _compute_formal_snapshot_body(
                arguments,
                canonical_identity=canonical_identity,
                training_mode=training_mode,
            )
            payload = {**body, "snapshot_sha256": sha256_json(body)}
            atomic_write_json(
                destination,
                payload,
                root=arguments.artifact_root,
                overwrite=False,
            )
            envelope[0] = {
                "ok": True,
                "path": str(destination),
                "file_sha256": sha256_file(destination),
                "snapshot_sha256": payload["snapshot_sha256"],
            }
        except Exception as exc:
            envelope[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if distributed is not None:
        distributed.broadcast_object_list(envelope, src=0)
        distributed.barrier()
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"formal provenance pre-load capture failed: {detail}")
    if result.get("path") != str(destination) or sha256_file(destination) != result.get(
        "file_sha256"
    ):
        raise RuntimeError("formal provenance snapshot file differs across ranks")
    snapshot = _validate_snapshot_payload(
        load_json_strict(destination),
        path=destination,
        arguments=arguments,
        canonical_identity=canonical_identity,
        training_mode=training_mode,
    )
    if snapshot.snapshot_sha256 != result.get("snapshot_sha256"):
        raise RuntimeError("formal provenance snapshot digest differs across ranks")
    return snapshot


def verify_formal_provenance_unchanged(
    arguments: Any,
    *,
    snapshot: FormalProvenanceSnapshot | None,
    canonical_identity: CanonicalSFTIdentity | None,
    training_mode: str,
    formal_artifact: bool,
    torch_module: Any,
) -> str | None:
    """Re-read the same actual content after training and fail on any change."""

    if not formal_artifact:
        return None
    if snapshot is None or canonical_identity is None:
        raise RuntimeError("formal provenance post-training verification was not initialized")
    distributed = _ensure_formal_distributed(torch_module)
    rank = int(distributed.get_rank()) if distributed is not None else 0
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            persisted = _validate_snapshot_payload(
                load_json_strict(snapshot.path),
                path=snapshot.path,
                arguments=arguments,
                canonical_identity=canonical_identity,
                training_mode=training_mode,
            )
            if (
                persisted.file_sha256 != snapshot.file_sha256
                or persisted.snapshot_sha256 != snapshot.snapshot_sha256
            ):
                raise ValueError("formal provenance pre-load snapshot changed on disk")
            post_body = _compute_formal_snapshot_body(
                arguments,
                canonical_identity=canonical_identity,
                training_mode=training_mode,
            )
            post_sha256 = sha256_json(post_body)
            if post_sha256 != snapshot.snapshot_sha256:
                raise ValueError("formal provenance changed during training")
            envelope[0] = {"ok": True, "post_sha256": post_sha256}
        except Exception as exc:
            envelope[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if distributed is not None:
        distributed.broadcast_object_list(envelope, src=0)
        distributed.barrier()
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"formal provenance post-training verification failed: {detail}")
    post_sha256 = result.get("post_sha256")
    if post_sha256 != snapshot.snapshot_sha256:
        raise RuntimeError("formal provenance post-training digest differs across ranks")
    return post_sha256


def bind_model_base_provenance(
    model_arguments: Any,
    artifact_arguments: Any,
    *,
    formal_artifact: bool,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
) -> BaseArtifactBindingReceipt | None:
    """Bind the exact local model loader source to formal base provenance."""

    if not formal_artifact:
        return None
    actual_raw = getattr(model_arguments, "model_name_or_path", None)
    expected_raw = getattr(artifact_arguments, "base_artifact_path", None)
    if not isinstance(actual_raw, str) or not actual_raw.strip():
        raise ValueError("formal SFT requires a non-empty local model_name_or_path")
    if expected_raw in (None, ""):
        raise ValueError("formal SFT requires base_artifact_path provenance")
    try:
        actual_input = Path(actual_raw)
        expected_input = Path(expected_raw)
        if not actual_input.is_absolute() or not expected_input.is_absolute():
            raise ValueError("formal SFT base model/provenance paths must be absolute")
        actual = Path(os.path.abspath(actual_input))
        expected = Path(os.path.abspath(expected_input))
        actual.resolve(strict=True)
        expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("formal SFT base model/provenance path must exist locally") from exc
    if actual != expected:
        raise ValueError(
            "actual model_name_or_path must exactly equal base_artifact_path provenance"
        )
    if provenance_snapshot is not None:
        evidence = provenance_snapshot.evidence.get("base_artifact")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("path") != str(expected.resolve(strict=True))
            or evidence.get("digest")
            != provenance_snapshot.binding.base_artifact_hash
            or evidence.get("file_count", 0) <= 0
            or evidence.get("total_bytes", 0) <= 0
        ):
            raise ValueError("formal SFT base model differs from pre-load snapshot")
        digest = provenance_snapshot.binding.base_artifact_hash
    else:
        actual_evidence = hash_path(actual, symlink_policy="follow")
        expected_evidence = hash_path(expected, symlink_policy="follow")
        if actual_evidence != expected_evidence:
            raise ValueError("formal SFT base model identity/hash differs from provenance")
        if actual_evidence.file_count <= 0 or actual_evidence.total_bytes <= 0:
            raise ValueError("formal SFT base model must contain non-empty files")
        digest = actual_evidence.digest
    setattr(model_arguments, "model_name_or_path", str(actual))
    setattr(artifact_arguments, "base_artifact_path", str(expected))
    return BaseArtifactBindingReceipt(actual, expected, digest)


def require_explicit_formal_seed(
    arguments: Any,
    *,
    formal_artifact: bool,
    argv: Sequence[str] | None = None,
) -> int:
    """Reject an inherited framework default seed for formal direct entrypoints."""

    seed = getattr(arguments, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("training seed must be an integer in [0, 2**32-1]")
    if not formal_artifact:
        return seed
    tokens = list(sys.argv[1:] if argv is None else argv)
    occurrences = 0
    for index, token in enumerate(tokens):
        if token == "--seed":
            occurrences += 1
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError("formal SFT --seed requires an explicit value")
        elif token.startswith("--seed="):
            occurrences += 1
    if occurrences != 1:
        raise ValueError("formal SFT requires exactly one explicit --seed argument")
    return seed


def validate_formal_deepspeed_zero2(
    arguments: Any, *, formal_artifact: bool
) -> None:
    """Reject ZeRO-3 until all-rank gathered state proof is implemented."""

    if not formal_artifact:
        return
    canonical_path = Path(__file__).resolve().parents[3] / "scripts" / "zero2.json"
    canonical = load_json_strict(canonical_path)
    raw = getattr(arguments, "deepspeed", None)
    if isinstance(raw, str):
        supplied_path = Path(raw)
        if not supplied_path.is_absolute():
            supplied_path = Path.cwd() / supplied_path
        if supplied_path.resolve(strict=True) != canonical_path.resolve(strict=True):
            raise ValueError(
                "formal SFT requires the canonical scripts/zero2.json DeepSpeed config"
            )
        supplied = load_json_strict(supplied_path)
    elif isinstance(raw, Mapping):
        supplied = dict(raw)
    else:
        hf_config = getattr(arguments, "hf_deepspeed_config", None)
        supplied = getattr(hf_config, "config", None)
    if not isinstance(supplied, Mapping) or dict(supplied) != canonical:
        raise ValueError("formal SFT DeepSpeed config differs from canonical zero2.json")
    zero = supplied.get("zero_optimization")
    stage = zero.get("stage") if isinstance(zero, Mapping) else None
    if type(stage) is not int or stage != 2:
        raise ValueError(
            "formal SFT rejects ZeRO-3; only ZeRO-2 is compatible with current state proof"
        )


def require_controller_verified_formal_bootstrap(
    *, formal_artifact: bool
) -> None:
    """Fail closed until the controller can execute frozen Qwen source bytes.

    A snapshot taken inside ``full_sft.py``/``lora_sft.py`` is necessarily too
    late to prove the bytes already imported by Python, site hooks, torch, or
    the Qwen entrypoint.  Formal publication therefore remains disabled until
    the external HMAC-bound controller supplies a pre-spawn source/environment
    attestation and starts every worker from a verified in-memory source bundle.
    Legacy smoke runs are unaffected because they cannot publish a manifest.
    """

    if not formal_artifact:
        return
    topology_fields = {
        "NNODES": os.environ.get("NNODES", "1"),
        "GROUP_WORLD_SIZE": os.environ.get("GROUP_WORLD_SIZE", "1"),
        "NODE_RANK": os.environ.get("NODE_RANK", "0"),
        "GROUP_RANK": os.environ.get("GROUP_RANK", "0"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE", "1"),
        "LOCAL_WORLD_SIZE": os.environ.get("LOCAL_WORLD_SIZE", "1"),
        "RANK": os.environ.get("RANK", "0"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK", "0"),
        "TORCHELASTIC_RESTART_COUNT": os.environ.get(
            "TORCHELASTIC_RESTART_COUNT", "0"
        ),
    }
    parsed: dict[str, int] = {}
    for name, raw in topology_fields.items():
        if not isinstance(raw, str) or re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
            raise RuntimeError(f"formal Qwen SFT rejects invalid {name}")
        parsed[name] = int(raw)
    if (
        parsed["NNODES"] != 1
        or parsed["GROUP_WORLD_SIZE"] != 1
        or parsed["NODE_RANK"] != 0
        or parsed["GROUP_RANK"] != 0
    ):
        raise RuntimeError(
            "formal Qwen SFT supports one node only; cross-node provenance is not implemented"
        )
    if (
        parsed["WORLD_SIZE"] != parsed["LOCAL_WORLD_SIZE"]
        or not 0 <= parsed["RANK"] < parsed["WORLD_SIZE"]
        or not 0 <= parsed["LOCAL_RANK"] < parsed["LOCAL_WORLD_SIZE"]
        or parsed["TORCHELASTIC_RESTART_COUNT"] != 0
    ):
        raise RuntimeError(
            "formal Qwen SFT requires a single-node, zero-restart local-rank topology"
        )
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1").strip().lower()
    if master_addr not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("formal Qwen SFT requires a loopback MASTER_ADDR")
    raise RuntimeError(
        "formal Qwen SFT is blocked: the controller does not yet provide an "
        "external-HMAC-bound pre-spawn source/environment snapshot plus verified "
        "in-memory worker source bootstrap; use --unsafe_legacy_no_manifest only "
        "for ineligible non-release smoke runs"
    )


def validate_fresh_formal_output_directory(
    arguments: Any,
    *,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Require a fresh output isolated from every immutable input/resume source."""

    root_raw = getattr(arguments, "artifact_root", None)
    output_raw = getattr(arguments, "output_dir", None)
    if root_raw in (None, "") or output_raw in (None, ""):
        raise ValueError("formal SFT requires artifact_root and output_dir")
    root = Path(root_raw).resolve(strict=True)
    output = resolve_within_root(output_raw, root, must_exist=False, allow_root=False)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("formal SFT output_dir must be fresh and empty")

    source_paths: list[Path] = []
    for raw in provenance_paths_from_arguments(arguments).to_dict().values():
        source_paths.append(Path(raw).resolve(strict=True))
    if resume_checkpoint is not None:
        source_paths.append(Path(resume_checkpoint).resolve(strict=True))
    for source in source_paths:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                f"formal SFT output_dir overlaps immutable base/resume/provenance source: {source}"
            )
    destinations: dict[str, Path] = {}
    for name in (
        "artifact_manifest_path",
        "reload_receipt_path",
        "training_receipt_path",
        "resume_manifest",
    ):
        raw = getattr(arguments, name, None)
        if raw in (None, ""):
            continue
        destination = resolve_within_root(raw, root, must_exist=False, allow_root=False)
        if destination == output or output in destination.parents:
            raise ValueError(f"{name} must be outside formal SFT output_dir")
        destinations[name] = destination
    # Standalone callers may use this overlap validator before configuring the
    # receipt destination.  Formal entrypoints require training_receipt_path in
    # validate_artifact_policy and therefore always validate the derived
    # pre-load snapshot destination here.
    if getattr(arguments, "training_receipt_path", None) not in (None, ""):
        snapshot_path = _formal_snapshot_path(arguments)
        if snapshot_path == output or output in snapshot_path.parents:
            raise ValueError("formal provenance snapshot must be outside output_dir")
        destinations["formal_provenance_snapshot"] = snapshot_path
    values = list(destinations.values())
    if len(values) != len(set(values)):
        raise ValueError("formal manifest/receipt/snapshot paths must be distinct")
    return output


def resolve_motion_length_divisor(model: Any) -> int:
    """Resolve the exact VQ encoder temporal factor for the data pipeline."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("motion model must expose config for temporal bridging")

    def positive_index(value: Any, *, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            parsed = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    stride = positive_index(
        getattr(config, "vqvae_stride_t", None), name="vqvae_stride_t"
    )
    depth = positive_index(
        getattr(config, "vqvae_down_t", None), name="vqvae_down_t"
    )
    motion_spec = getattr(model, "motion_spec", None)
    configured = getattr(config, "motion_downsample_factor", None)
    authoritative = (
        positive_index(
            motion_spec.downsample_factor,
            name="motion_spec.downsample_factor",
        )
        if motion_spec is not None
        else (
            positive_index(configured, name="motion_downsample_factor")
            if configured is not None
            else None
        )
    )
    if authoritative is not None and stride >= 2 and depth > authoritative.bit_length():
        raise ValueError(
            "VQ stride/depth disagree with the configured motion downsample factor"
        )
    computed = stride**depth
    if authoritative is not None and computed != authoritative:
        raise ValueError(
            f"VQ temporal factor {stride} ** {depth}={computed} disagrees with "
            f"configured motion downsample factor {authoritative}"
        )
    if motion_spec is not None and configured is not None:
        config_factor = positive_index(configured, name="motion_downsample_factor")
        if config_factor != authoritative:
            raise ValueError(
                "model.motion_spec.downsample_factor disagrees with model.config."
                "motion_downsample_factor"
            )
    return computed


def bind_motion_length_divisor(data_arguments: Any, model: Any) -> int:
    """Bind data placeholder planning to the verified encoder factor."""

    expected = resolve_motion_length_divisor(model)
    current = getattr(data_arguments, "motion_length_divisor", None)
    if current is not None:
        if isinstance(current, bool):
            raise ValueError("motion_length_divisor must be a positive integer")
        try:
            supplied = operator.index(current)
        except TypeError as exc:
            raise ValueError("motion_length_divisor must be a positive integer") from exc
        if supplied != expected:
            raise ValueError(
                f"data motion_length_divisor={supplied} disagrees with verified "
                f"VQ encoder factor {expected}"
            )
    setattr(data_arguments, "motion_length_divisor", expected)
    return expected


def resolve_eval_enabled(training_arguments: Any, eval_dataset: Any | None) -> bool:
    do_eval = bool(getattr(training_arguments, "do_eval", False))
    strategy = getattr(training_arguments, "eval_strategy", None)
    if strategy is None:
        strategy = getattr(training_arguments, "evaluation_strategy", None)
    strategy_text = str(getattr(strategy, "value", strategy) or "no").strip().lower()
    return bool(eval_dataset is not None and (do_eval or strategy_text not in {"", "no", "none"}))


def provenance_paths_from_arguments(arguments: Any) -> ArtifactProvenancePaths:
    missing = [
        argument_name
        for argument_name in _PROVENANCE_ARGUMENTS.values()
        if getattr(arguments, argument_name, None) in (None, "")
    ]
    if missing:
        raise ValueError(f"verified provenance paths are required: {missing}")
    values = {
        role: getattr(arguments, argument_name)
        for role, argument_name in _PROVENANCE_ARGUMENTS.items()
    }
    values.update(
        {
            role: getattr(arguments, argument_name, None)
            for role, argument_name in _OPTIONAL_PROVENANCE_ARGUMENTS.items()
        }
    )
    return ArtifactProvenancePaths(**values)


def bind_motion_vqvae_provenance(
    motion_arguments: Any,
    artifact_arguments: Any,
    *,
    formal_artifact: bool,
    supports_motion: bool,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
) -> Path | None:
    """Bind the exact VQ-VAE path passed to the model loader to provenance."""

    actual_raw = getattr(motion_arguments, "motion_vqvae_path", None)
    provenance_raw = getattr(artifact_arguments, "motion_vqvae_asset_path", None)
    if not supports_motion:
        if formal_artifact and provenance_raw not in (None, ""):
            raise ValueError("non-motion training must not declare motion_vqvae_asset_path")
        return None
    if actual_raw in (None, ""):
        raise ValueError("motion training requires motion_vqvae_path")
    actual = Path(actual_raw).resolve(strict=True)
    if not actual.is_file():
        raise ValueError("motion_vqvae_path must be a regular checkpoint file")
    if not formal_artifact:
        return actual
    if provenance_raw in (None, ""):
        raise ValueError("formal motion training requires motion_vqvae_asset_path provenance")
    provenance = Path(provenance_raw).resolve(strict=True)
    if provenance != actual:
        raise ValueError(
            "model motion_vqvae_path must exactly equal provenance motion_vqvae_asset_path"
        )
    if provenance_snapshot is not None:
        evidence = provenance_snapshot.evidence.get("motion_vqvae")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("path") != str(actual)
            or evidence.get("digest")
            != provenance_snapshot.binding.motion_vqvae_hash
            or evidence.get("file_count") != 1
            or evidence.get("total_bytes", 0) <= 0
        ):
            raise ValueError("motion VQ-VAE differs from pre-load provenance snapshot")
    else:
        actual_digest = hash_path(actual, symlink_policy="follow")
        provenance_digest = hash_path(provenance, symlink_policy="follow")
        if actual_digest != provenance_digest or actual_digest.total_bytes <= 0:
            raise ValueError("motion VQ-VAE identity/hash differs from formal provenance")
    setattr(motion_arguments, "motion_vqvae_path", str(actual))
    setattr(artifact_arguments, "motion_vqvae_asset_path", str(actual))
    return actual


def _resolve_single_dataset_use(
    dataset_use: Any,
    *,
    label: str,
    dataset_resolver: Callable[[list[str]], list[Mapping[str, Any]]],
) -> tuple[str, Path]:
    if not isinstance(dataset_use, str) or not dataset_use.strip():
        raise ValueError(f"{label} dataset_use must be one explicit dataset name")
    name = dataset_use.strip()
    if "," in name or "%" in name:
        raise ValueError(f"{label} dataset_use cannot combine or sample aliases")
    resolved = dataset_resolver([name])
    if not isinstance(resolved, list) or len(resolved) != 1:
        raise ValueError(f"{label} dataset_use must resolve to exactly one dataset")
    entry = resolved[0]
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label} dataset resolver returned an invalid entry")
    if entry.get("sampling_rate", 1.0) != 1.0:
        raise ValueError(f"{label} dataset sampling is forbidden for formal provenance")
    annotation = entry.get("annotation_path")
    if not isinstance(annotation, str) or not annotation:
        raise ValueError(f"{label} dataset has no annotation_path")
    path = Path(annotation).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} annotation_path must be a regular file")
    return name, path


def bind_supervised_data_provenance(
    data_arguments: Any,
    artifact_arguments: Any,
    *,
    dataset_resolver: Callable[[list[str]], list[Mapping[str, Any]]],
) -> SupervisedDataBindingReceipt:
    """Prove the legacy loader's one train/validation path equals provenance."""

    train_name, actual_train = _resolve_single_dataset_use(
        getattr(data_arguments, "dataset_use", None),
        label="train",
        dataset_resolver=dataset_resolver,
    )
    validation_name, actual_validation = _resolve_single_dataset_use(
        getattr(data_arguments, "eval_dataset_use", None),
        label="validation",
        dataset_resolver=dataset_resolver,
    )
    expected_train = Path(artifact_arguments.train_data_path).resolve(strict=True)
    expected_validation = Path(artifact_arguments.validation_data_path).resolve(strict=True)
    if actual_train != expected_train:
        raise ValueError("actual SFT train loader path does not equal train_data_path provenance")
    if actual_validation != expected_validation:
        raise ValueError(
            "actual SFT validation loader path does not equal validation_data_path provenance"
        )
    if actual_train == actual_validation:
        raise ValueError("formal SFT train and validation loader paths must be distinct")
    return SupervisedDataBindingReceipt(
        train_dataset_use=train_name,
        train_data_path=actual_train,
        validation_dataset_use=validation_name,
        validation_data_path=actual_validation,
    )


def make_provenance_bound_supervised_data_module(
    processor: Any,
    data_arguments: Any,
    artifact_arguments: Any,
    *,
    dataset_resolver: Callable[[list[str]], list[Mapping[str, Any]]],
    data_module_factory: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Construct the real loader between two immutable path-binding checks."""

    before = bind_supervised_data_provenance(
        data_arguments, artifact_arguments, dataset_resolver=dataset_resolver
    )
    result = data_module_factory(processor, data_args=data_arguments)
    after = bind_supervised_data_provenance(
        data_arguments, artifact_arguments, dataset_resolver=dataset_resolver
    )
    if before != after:
        raise ValueError("dataset resolver changed while the supervised loader was constructed")
    if not isinstance(result, Mapping) or result.get("train_dataset") is None:
        raise ValueError("supervised data module did not return a train_dataset")
    if result.get("eval_dataset") is None:
        raise ValueError("formal supervised data module did not return a validation dataset")
    for name, dataset in (
        ("train_dataset", result["train_dataset"]),
        ("eval_dataset", result["eval_dataset"]),
    ):
        try:
            size = len(dataset)
        except Exception as exc:
            raise ValueError(f"formal {name} must expose a deterministic non-empty length") from exc
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"formal {name} must contain at least one sample")
    return result


def _identity_from_arguments(arguments: Any) -> tuple[str, str]:
    batch_id = getattr(arguments, "batch_id", None)
    model_id = getattr(arguments, "model_registry_id", None)
    missing = [
        name
        for name, value in (("batch_id", batch_id), ("model_registry_id", model_id))
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"formal artifact identity is required: {missing}")
    for name, value in (("batch_id", batch_id), ("model_registry_id", model_id)):
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or any(not (character.isalnum() or character in "._-") for character in value)
        ):
            raise ValueError(f"formal {name} must be a safe identifier")
    return batch_id, model_id


def validate_artifact_policy(arguments: Any, *, training_mode: str) -> bool:
    """Return True for formal publication or require an explicit unsafe escape hatch."""

    manifest = getattr(arguments, "artifact_manifest_path", None)
    unsafe = bool(getattr(arguments, "unsafe_legacy_no_manifest", False))
    if manifest in (None, ""):
        if not unsafe:
            raise ValueError(
                "formal training requires --artifact_manifest_path and verified provenance; "
                "use --unsafe_legacy_no_manifest only for non-release legacy smoke runs"
            )
        if getattr(arguments, "resume_manifest", None) not in (None, ""):
            raise ValueError("unsafe no-manifest mode cannot resume a formal artifact")
        return False
    if unsafe:
        raise ValueError("--unsafe_legacy_no_manifest cannot be combined with formal manifest output")
    if getattr(arguments, "artifact_root", None) in (None, ""):
        raise ValueError("--artifact_root is required for formal artifact publication")
    if training_mode == "grpo" and getattr(arguments, "resume_manifest", None) not in (
        None,
        "",
    ):
        raise ValueError(
            "formal GRPO resume is not implemented with complete optimizer/state proof"
        )
    _identity_from_arguments(arguments)
    provenance_paths_from_arguments(arguments)
    if training_mode in {"full_sft", "lora_sft", "grpo"} and getattr(
        arguments, "reload_receipt_path", None
    ) in (None, ""):
        raise ValueError(f"{training_mode} requires --reload_receipt_path")
    if training_mode in {"full_sft", "lora_sft"}:
        if getattr(arguments, "runner_code_path", None) in (None, ""):
            raise ValueError(f"{training_mode} requires --runner_code_path")
        if getattr(arguments, "training_receipt_path", None) in (None, ""):
            raise ValueError(f"{training_mode} requires --training_receipt_path")
        for name in ("batch_receipt_sha256", "attempt_sha256"):
            value = getattr(arguments, name, None)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"formal SFT requires --{name} as lowercase SHA-256")
    return True


def artifact_binding_from_arguments(arguments: Any, *, training_mode: str) -> ArtifactBinding:
    batch_id, model_id = _identity_from_arguments(arguments)
    binding, _ = compute_verified_provenance(
        provenance_paths_from_arguments(arguments),
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
    )
    return binding


def write_training_proof_from_arguments(
    arguments: Any,
    *,
    training_mode: str,
    backend_id: str,
    canonical_identity: CanonicalSFTIdentity,
    artifact_path: str | Path,
    planned_steps: int,
    actual_steps: int,
    finite_losses: Sequence[float],
    nonzero_finite_gradient_steps: int,
    max_gradient: float,
    initial_snapshot: Any,
    final_snapshot: Any,
    changed_tensor_count: int,
    max_parameter_update: float | None = None,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
    provenance_post_sha256: str | None = None,
) -> Path:
    """Bind optimizer evidence to the unchanged in-process diagnostic snapshot.

    This helper does not make Qwen output formally eligible; the production
    entrypoints are blocked until a controller pre-spawn attestation exists.
    """

    if not validate_artifact_policy(arguments, training_mode=training_mode):
        raise ValueError("training proof is only available for formal SFT")
    if Path(arguments.code_path).resolve(strict=True) != canonical_identity.code_root:
        raise ValueError("formal SFT code_path changed after canonical identity binding")
    runner_code = Path(arguments.runner_code_path)
    if (
        not runner_code.is_absolute()
        or Path(os.path.abspath(runner_code)) != canonical_identity.runner_code_root
    ):
        raise ValueError(
            "formal SFT runner_code_path changed after canonical identity binding"
        )
    if (
        Path(arguments.environment_path).resolve(strict=True)
        != canonical_identity.environment_root
    ):
        raise ValueError(
            "formal SFT environment_path changed after canonical identity binding"
        )
    base_artifact = Path(arguments.base_artifact_path)
    if (
        not base_artifact.is_absolute()
        or Path(os.path.abspath(base_artifact)) != canonical_identity.base_artifact
    ):
        raise ValueError(
            "formal SFT base_artifact_path changed after canonical identity binding"
        )
    if provenance_snapshot is None:
        raise ValueError("formal SFT training proof requires a pre-load provenance snapshot")
    if provenance_snapshot.path != _formal_snapshot_path(arguments):
        raise ValueError("formal provenance snapshot path changed before receipt")
    if sha256_file(provenance_snapshot.path) != provenance_snapshot.file_sha256:
        raise ValueError("formal provenance snapshot file changed before receipt")
    if provenance_post_sha256 != provenance_snapshot.snapshot_sha256:
        raise ValueError("formal provenance pre/post digests differ")
    binding = binding_from_provenance_evidence(
        provenance_paths_from_arguments(arguments),
        provenance_snapshot.evidence,
        batch_id=provenance_snapshot.binding.batch_id,
        model_id=provenance_snapshot.binding.model_id,
        training_mode=training_mode,
    )
    if binding != provenance_snapshot.binding:
        raise ValueError("formal provenance binding differs from pre-load snapshot")
    if binding.runner_code_hash is None:
        raise ValueError("formal SFT requires independently hashed runner_code")
    artifact = resolve_within_root(
        artifact_path,
        arguments.artifact_root,
        must_exist=True,
        allow_root=False,
    )
    destination = resolve_within_root(
        arguments.training_receipt_path,
        arguments.artifact_root,
        must_exist=False,
        allow_root=False,
    )
    if destination == artifact or artifact in destination.parents:
        raise ValueError("training receipt must be outside the artifact directory")
    artifact_digest = hash_path(
        artifact, symlink_policy="reject", allowed_root=arguments.artifact_root
    ).digest
    receipt = make_training_receipt(
        batch_id=binding.batch_id,
        model_id=binding.model_id,
        backend_id=backend_id,
        model_family=canonical_identity.model_family,
        modality=canonical_identity.modality,
        training_mode=training_mode,
        planned_global_steps=planned_steps,
        actual_global_steps=actual_steps,
        planned_optimizer_steps=planned_steps,
        actual_optimizer_steps=actual_steps,
        finite_losses=finite_losses,
        nonzero_finite_gradient_steps=nonzero_finite_gradient_steps,
        max_gradient=max_gradient,
        trainable_tensor_count=initial_snapshot.tensor_count,
        trainable_parameter_count=initial_snapshot.parameter_count,
        changed_trainable_tensor_count=changed_tensor_count,
        initial_trainable_sha256=initial_snapshot.sha256,
        final_trainable_sha256=final_snapshot.sha256,
        max_parameter_update=max_parameter_update,
        batch_receipt_sha256=arguments.batch_receipt_sha256,
        attempt_sha256=arguments.attempt_sha256,
        train_sha256=binding.train_data_hash,
        validation_sha256=binding.validation_data_hash,
        leakage_audit_sha256=binding.leakage_audit_hash,
        base_artifact_sha256=binding.base_artifact_hash,
        config_sha256=binding.config_hash,
        code_sha256=binding.code_hash,
        runner_code_sha256=binding.runner_code_hash,
        environment_sha256=binding.environment_hash,
        artifact_sha256=artifact_digest,
        provenance_snapshot_path=str(provenance_snapshot.path),
        provenance_snapshot_file_sha256=provenance_snapshot.file_sha256,
        provenance_pre_sha256=provenance_snapshot.snapshot_sha256,
        provenance_post_sha256=provenance_post_sha256,
        provenance_unchanged=True,
    )
    return write_training_receipt(
        destination,
        receipt,
        root=arguments.artifact_root,
        overwrite=False,
    )


def resolve_validated_resume_checkpoint(
    resume_manifest: str | Path | None,
    *,
    provenance_paths: ArtifactProvenancePaths,
    batch_id: str,
    model_id: str,
    training_mode: str,
    allowed_root: str | Path,
    provenance_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path | None:
    if resume_manifest in (None, ""):
        return None
    if training_mode == "grpo":
        raise ValueError(
            "formal GRPO resume is not implemented with complete optimizer/state proof"
        )
    receipt = validate_resume_artifact(
        resume_manifest,
        provenance_paths=provenance_paths,
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        allowed_root=allowed_root,
        provenance_evidence=provenance_evidence,
    )
    artifact = receipt.artifact_path

    def nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def complete_checkpoint(candidate: Path) -> bool:
        common = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin")
        if not candidate.is_dir() or any(
            not nonempty_file(candidate / name) for name in common
        ):
            return False
        if not any(nonempty_file(path) for path in candidate.glob("rng_state*.pth")):
            return False
        if training_mode == "lora_sft":
            return nonempty_file(candidate / "adapter_config.json") and any(
                nonempty_file(candidate / name)
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            )
        if training_mode != "full_sft":
            return False
        if any(
            nonempty_file(candidate / name)
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            return True
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            index_path = candidate / index_name
            if not nonempty_file(index_path):
                continue
            try:
                import json

                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
            except (OSError, UnicodeError, ValueError):
                return False
            if not isinstance(weight_map, Mapping) or not weight_map:
                return False
            shards = set(weight_map.values())
            if not all(isinstance(name, str) and nonempty_file(candidate / name) for name in shards):
                return False
            return True
        return False

    candidates = [artifact]
    if artifact.is_dir():
        candidates.extend(sorted(path for path in artifact.glob("checkpoint-*") if path.is_dir()))
    complete = [path for path in candidates if complete_checkpoint(path)]
    if len(complete) != 1:
        raise ValueError(
            "formal SFT resume requires exactly one complete checkpoint containing weights, "
            "optimizer, scheduler, trainer_state, training_args and RNG state"
        )
    return complete[0]


def resolve_resume_from_arguments(
    arguments: Any,
    *,
    training_mode: str,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
) -> Path | None:
    manifest = getattr(arguments, "resume_manifest", None)
    if manifest in (None, ""):
        return None
    root = getattr(arguments, "artifact_root", None)
    if root in (None, ""):
        raise ValueError("--artifact_root is required with --resume_manifest")
    batch_id, model_id = _identity_from_arguments(arguments)
    return resolve_validated_resume_checkpoint(
        manifest,
        provenance_paths=provenance_paths_from_arguments(arguments),
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        allowed_root=root,
        provenance_evidence=(
            provenance_snapshot.evidence
            if provenance_snapshot is not None
            else None
        ),
    )


def write_artifact_from_arguments(
    arguments: Any,
    *,
    training_mode: str,
    artifact_path: str | Path,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
) -> FinetuneArtifactReceipt | None:
    if not validate_artifact_policy(arguments, training_mode=training_mode):
        return None
    batch_id, model_id = _identity_from_arguments(arguments)
    return write_finetune_artifact_manifest(
        arguments.artifact_manifest_path,
        artifact_path=artifact_path,
        provenance_paths=provenance_paths_from_arguments(arguments),
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        allowed_root=arguments.artifact_root,
        reload_receipt_path=getattr(arguments, "reload_receipt_path", None),
        training_receipt_path=getattr(arguments, "training_receipt_path", None),
        provenance_evidence=(
            provenance_snapshot.evidence
            if provenance_snapshot is not None
            else None
        ),
        overwrite=False,
    )


def _distributed(torch_module: Any) -> Any | None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None:
        return None
    try:
        if not distributed.is_available() or not distributed.is_initialized():
            return None
    except Exception:
        return None
    return distributed


def publish_artifact_distributed(
    arguments: Any,
    *,
    training_mode: str,
    artifact_path: str | Path,
    torch_module: Any,
    provenance_snapshot: FormalProvenanceSnapshot | None = None,
) -> FinetuneArtifactReceipt | None:
    """Only rank zero publishes; every rank synchronizes and revalidates evidence."""

    if not validate_artifact_policy(arguments, training_mode=training_mode):
        return None
    distributed = _distributed(torch_module)
    rank = distributed_rank(torch_module)
    failure: list[str | None] = [None]
    receipt: FinetuneArtifactReceipt | None = None
    if rank == 0:
        try:
            receipt = write_artifact_from_arguments(
                arguments,
                training_mode=training_mode,
                artifact_path=artifact_path,
                provenance_snapshot=provenance_snapshot,
            )
        except Exception as exc:  # synchronize a readable failure instead of racing writers
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed is not None:
        broadcaster = getattr(distributed, "broadcast_object_list", None)
        if not callable(broadcaster):
            raise RuntimeError("distributed artifact publication requires broadcast_object_list")
        broadcaster(failure, src=0)
        if failure[0] is not None:
            raise RuntimeError(f"primary artifact publication failed: {failure[0]}")
        distributed.barrier()
    elif failure[0] is not None:
        raise RuntimeError(f"artifact publication failed: {failure[0]}")

    batch_id, model_id = _identity_from_arguments(arguments)
    confirmed: FinetuneArtifactReceipt | None = None
    confirmation_error: str | None = None
    try:
        confirmed = validate_resume_artifact(
            arguments.artifact_manifest_path,
            provenance_paths=provenance_paths_from_arguments(arguments),
            batch_id=batch_id,
            model_id=model_id,
            training_mode=training_mode,
            allowed_root=arguments.artifact_root,
            provenance_evidence=(
                provenance_snapshot.evidence
                if provenance_snapshot is not None
                else None
            ),
        )
    except Exception as exc:
        confirmation_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    if distributed is not None:
        gather = getattr(distributed, "all_gather_object", None)
        world_size = getattr(distributed, "get_world_size", None)
        if not callable(gather) or not callable(world_size):
            raise RuntimeError(
                "distributed artifact validation requires all_gather_object/get_world_size"
            )
        errors: list[str | None] = [None] * int(world_size())
        gather(errors, confirmation_error)
        failures = [item for item in errors if item is not None]
        if failures:
            raise RuntimeError("distributed artifact validation failed: " + " | ".join(failures))
        distributed.barrier()
    elif confirmation_error is not None:
        raise RuntimeError(f"artifact validation failed: {confirmation_error}")
    if confirmed is None:  # defensive: all error paths above have raised
        raise RuntimeError("artifact validation produced no receipt")
    return confirmed if receipt is None else receipt
