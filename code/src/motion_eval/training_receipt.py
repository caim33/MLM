"""Strict, self-hashed proof that a fresh finetune actually updated weights."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from motion_eval.core import (
    atomic_write_json,
    formal_source_role_files,
    hash_path,
    sha256_file,
    sha256_json,
)
from motion_eval.data import load_json_strict


class TrainingReceiptError(ValueError):
    """A training receipt is malformed, unbound, or does not prove training."""


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_TRAINING_MODES = frozenset({"full_sft", "lora_sft", "official_finetune"})
TRAINING_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at",
        "batch_id",
        "model_id",
        "backend_id",
        "model_family",
        "modality",
        "training_mode",
        "planned_global_steps",
        "actual_global_steps",
        "planned_optimizer_steps",
        "actual_optimizer_steps",
        "finite_losses",
        "nonzero_finite_gradient_steps",
        "max_gradient",
        "trainable_tensor_count",
        "trainable_parameter_count",
        "changed_trainable_tensor_count",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "max_parameter_update",
        "batch_receipt_sha256",
        "attempt_sha256",
        "train_sha256",
        "validation_sha256",
        "leakage_audit_sha256",
        "base_artifact_sha256",
        "config_sha256",
        "code_sha256",
        "runner_code_sha256",
        "environment_sha256",
        "artifact_sha256",
        "receipt_sha256",
    }
)
TRAINING_RECEIPT_V2_KEYS = frozenset(
    set(TRAINING_RECEIPT_KEYS)
    | {
        "provenance_snapshot_path",
        "provenance_snapshot_file_sha256",
        "provenance_pre_sha256",
        "provenance_post_sha256",
        "provenance_unchanged",
    }
)
FORMAL_PROVENANCE_SNAPSHOT_SCHEMA = "motionllm-inprocess-provenance-v2"
_SNAPSHOT_STATUS = "captured_before_model_data_load_after_entrypoint_imports"
_SNAPSHOT_EVIDENCE_KEYS = frozenset(
    {"path", "algorithm", "kind", "digest", "file_count", "total_bytes"}
)
_SNAPSHOT_REQUIRED_ROLES = frozenset(
    {
        "base_artifact",
        "train_data",
        "validation_data",
        "benchmark",
        "leakage_audit",
        "config",
        "code",
        "environment",
        "runner_code",
    }
)
_SNAPSHOT_OPTIONAL_ROLES = frozenset({"motion_vqvae"})


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise TrainingReceiptError(f"{field} must be a safe non-empty identifier")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrainingReceiptError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingReceiptError(f"{field} must be a positive integer")
    return value


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingReceiptError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingReceiptError(f"{field} must be a finite positive number")
    return result


def _absolute_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise TrainingReceiptError(f"{field} must be an absolute path")
    normalized = Path(os.path.abspath(value))
    if os.path.normcase(str(normalized)) != os.path.normcase(value):
        raise TrainingReceiptError(f"{field} must be a normalized absolute path")
    return value


def _absolute_path_object(value: Any, field: str) -> Path:
    return Path(_absolute_path(value, field))


def _is_within(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath((str(path), str(root)))
        return os.path.normcase(common) == os.path.normcase(str(root))
    except ValueError:
        return False


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TrainingReceiptError(f"snapshot path cannot be inspected: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _require_canonical_directory(path: Path, field: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingReceiptError(f"{field} cannot be resolved") from exc
    if (
        not path.is_dir()
        or _is_link_or_reparse(path)
        or os.path.normcase(str(path)) != os.path.normcase(str(resolved))
    ):
        raise TrainingReceiptError(f"{field} must be a canonical unlinked directory")
    return resolved


def _decode_record_sha256(value: str, field: str) -> str:
    if not value.startswith("sha256="):
        raise TrainingReceiptError(f"{field} must use a RECORD SHA-256 digest")
    encoded = value.split("=", 1)[1]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as exc:
        raise TrainingReceiptError(f"{field} is not valid base64url") from exc
    if len(raw) != 32:
        raise TrainingReceiptError(f"{field} has the wrong SHA-256 length")
    return raw.hex()


def _require_within(
    value: Any,
    *,
    roots: Sequence[Path],
    field: str,
    resolve_existing: bool = False,
) -> Path:
    candidate = _absolute_path_object(value, field)
    if resolve_existing:
        try:
            candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(f"{field} cannot be resolved") from exc
    if not any(_is_within(candidate, root) for root in roots):
        raise TrainingReceiptError(f"{field} escapes its allowed roots")
    return candidate


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingReceiptError(f"{field} must be a non-negative integer")
    return value


def _validate_snapshot_evidence(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_EVIDENCE_KEYS:
        raise TrainingReceiptError(f"{field} evidence schema is invalid")
    result = dict(value)
    _absolute_path(result.get("path"), f"{field}.path")
    for name in ("algorithm", "kind"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise TrainingReceiptError(f"{field}.{name} must be non-empty")
    _digest(result.get("digest"), f"{field}.digest")
    file_count = _nonnegative_int(result.get("file_count"), f"{field}.file_count")
    total_bytes = _nonnegative_int(result.get("total_bytes"), f"{field}.total_bytes")
    if file_count == 0 or total_bytes == 0:
        raise TrainingReceiptError(f"{field} evidence must be non-empty")
    return result


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingReceiptError(f"{field} must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainingReceiptError(f"{field} must be a safe relative path")
    return value


def _validate_source_snapshot_manifest(
    value: Any,
    *,
    role: str,
    evidence: Mapping[str, Any],
    expected_root: Path,
    project_root: Path,
    runner_root: Path,
) -> None:
    fields = {"schema_version", "role", "root", "files", "manifest_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingReceiptError(f"snapshot {role} manifest schema is invalid")
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    if (
        value.get("schema_version") != "motionllm-source-allowlist-v2"
        or value.get("role") != role
        or value.get("manifest_sha256") != sha256_json(body)
        or value.get("manifest_sha256") != evidence.get("digest")
        or value.get("root") != evidence.get("path")
        or evidence.get("algorithm") != "motionllm-source-allowlist-v2"
        or evidence.get("kind") != "source-allowlist"
    ):
        raise TrainingReceiptError(f"snapshot {role} manifest binding is invalid")
    manifest_root = _absolute_path_object(
        value.get("root"), f"snapshot.{role}.root"
    )
    if manifest_root != expected_root:
        raise TrainingReceiptError(f"snapshot {role} manifest has the wrong role root")
    try:
        inventory_root, expected_rows = formal_source_role_files(
            project_root, runner_root=runner_root, role=role
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingReceiptError(
            f"snapshot {role} fixed source inventory cannot be rebuilt: {exc}"
        ) from exc
    if inventory_root != manifest_root:
        raise TrainingReceiptError(f"snapshot {role} fixed role root differs")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise TrainingReceiptError(f"snapshot {role} file manifest is empty")
    seen: set[str] = set()
    total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256", "size"}:
            raise TrainingReceiptError(f"snapshot {role}.files[{index}] schema is invalid")
        relative = _safe_relative(row.get("relative_path"), f"snapshot.{role}.files[{index}]")
        if relative in seen:
            raise TrainingReceiptError(f"snapshot {role} contains duplicate paths")
        seen.add(relative)
        expected_sha256 = _digest(
            row.get("sha256"), f"snapshot.{role}.files[{index}].sha256"
        )
        expected_size = _positive_int(
            row.get("size"), f"snapshot.{role}.files[{index}].size"
        )
        candidate = manifest_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot {role}.files[{index}] cannot be resolved"
            ) from exc
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or os.path.normcase(str(resolved)) != os.path.normcase(str(candidate))
            or not _is_within(resolved, manifest_root)
        ):
            raise TrainingReceiptError(
                f"snapshot {role}.files[{index}] escapes/is not a regular role file"
            )
        if sha256_file(candidate) != expected_sha256 or candidate.stat().st_size != expected_size:
            raise TrainingReceiptError(
                f"snapshot {role}.files[{index}] content differs from its manifest"
            )
        if candidate.suffix.lower() in {".pyc", ".pyo"}:
            if candidate.parent.name == "__pycache__":
                source = candidate.parent.parent / (
                    candidate.name.split(".", 1)[0] + ".py"
                )
            else:
                source = candidate.with_suffix(".py")
            if not source.is_file() or _is_link_or_reparse(source):
                raise TrainingReceiptError(
                    f"snapshot {role}.files[{index}] is not source-backed bytecode"
                )
        total += expected_size
    if evidence.get("file_count") != len(rows) or evidence.get("total_bytes") != total:
        raise TrainingReceiptError(f"snapshot {role} evidence counts differ from manifest")
    if rows != list(expected_rows):
        raise TrainingReceiptError(
            f"snapshot {role} source inventory is incomplete/extra/different"
        )


def _validate_environment_snapshot_manifest(
    value: Any,
    *,
    evidence: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    fields = {
        "schema_version", "environment_root", "base_environment_root",
        "python_version", "python_implementation", "interpreter_entry",
        "interpreter_entry_link_sha256", "interpreter_target",
        "interpreter_target_sha256", "interpreter_target_size", "pyvenv_sha256",
        "pyvenv", "pth_files", "loading_environment", "sys_path", "meta_path",
        "stdlib_root", "stdlib_files", "native_runtime_files", "native_allowed_roots",
        "internal_links",
        "distributions", "files", "manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingReceiptError("snapshot environment manifest schema is invalid")
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    if (
        value.get("schema_version") != "motionllm-installed-environment-v2"
        or value.get("manifest_sha256") != sha256_json(body)
        or value.get("manifest_sha256") != evidence.get("digest")
        or value.get("environment_root") != evidence.get("path")
        or evidence.get("algorithm") != "motionllm-installed-environment-v2"
        or evidence.get("kind") != "installed-environment-manifest"
    ):
        raise TrainingReceiptError("snapshot environment manifest binding is invalid")
    environment_root = _absolute_path_object(
        value.get("environment_root"), "snapshot.environment.environment_root"
    )
    base_root = _absolute_path_object(
        value.get("base_environment_root"),
        "snapshot.environment.base_environment_root",
    )
    code_root = _absolute_path_object(
        identity.get("code_root"), "snapshot.canonical_identity.code_root"
    )
    runner_root = _absolute_path_object(
        identity.get("runner_code_root"),
        "snapshot.canonical_identity.runner_code_root",
    )
    identity_environment = _absolute_path_object(
        identity.get("environment_root"),
        "snapshot.canonical_identity.environment_root",
    )
    identity_base = _absolute_path_object(
        identity.get("base_environment_root"),
        "snapshot.canonical_identity.base_environment_root",
    )
    identity_interpreter = _absolute_path_object(
        identity.get("interpreter"), "snapshot.canonical_identity.interpreter"
    )
    for candidate, field in (
        (environment_root, "snapshot.environment.environment_root"),
        (base_root, "snapshot.environment.base_environment_root"),
        (code_root, "snapshot.canonical_identity.code_root"),
        (runner_root, "snapshot.canonical_identity.runner_code_root"),
    ):
        _require_canonical_directory(candidate, field)
    if (
        environment_root != identity_environment
        or base_root != identity_base
        or environment_root == base_root
        or not _is_within(runner_root, code_root)
    ):
        raise TrainingReceiptError("snapshot environment canonical role roots are invalid")
    interpreter_entry = _require_within(
        value.get("interpreter_entry"),
        roots=(environment_root,),
        field="snapshot.environment.interpreter_entry",
    )
    interpreter_target = _require_within(
        value.get("interpreter_target"),
        roots=(environment_root, base_root),
        field="snapshot.environment.interpreter_target",
    )
    if interpreter_target != identity_interpreter:
        raise TrainingReceiptError("snapshot environment interpreter role binding is invalid")
    stdlib_root = _require_within(
        value.get("stdlib_root"),
        roots=(base_root,),
        field="snapshot.environment.stdlib_root",
    )
    _require_canonical_directory(stdlib_root, "snapshot.environment.stdlib_root")
    for name in ("interpreter_target_sha256", "pyvenv_sha256"):
        _digest(value.get(name), f"snapshot.environment.{name}")
    link_digest = value.get("interpreter_entry_link_sha256")
    if link_digest is not None:
        _digest(link_digest, "snapshot.environment.interpreter_entry_link_sha256")
    interpreter_size = _positive_int(
        value.get("interpreter_target_size"), "snapshot.environment.interpreter_target_size"
    )
    try:
        resolved_interpreter_entry = interpreter_entry.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingReceiptError("snapshot environment interpreter entry cannot resolve") from exc
    if resolved_interpreter_entry != interpreter_target:
        raise TrainingReceiptError("snapshot environment interpreter entry/target differ")
    entry_is_link = _is_link_or_reparse(interpreter_entry)
    if link_digest is None:
        if entry_is_link or interpreter_entry != interpreter_target:
            raise TrainingReceiptError("snapshot environment unrecorded interpreter link")
    else:
        if not entry_is_link:
            raise TrainingReceiptError("snapshot environment falsely records interpreter link")
        try:
            actual_link_digest = hash_path(
                interpreter_entry, symlink_policy="link"
            ).digest
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrainingReceiptError(
                "snapshot environment interpreter link cannot be hashed"
            ) from exc
        if actual_link_digest != link_digest:
            raise TrainingReceiptError("snapshot environment interpreter link differs")
    if (
        not interpreter_target.is_file()
        or _is_link_or_reparse(interpreter_target)
        or sha256_file(interpreter_target) != value.get("interpreter_target_sha256")
        or interpreter_target.stat().st_size != interpreter_size
    ):
        raise TrainingReceiptError("snapshot environment interpreter content differs")
    if not isinstance(value.get("python_version"), str) or not value["python_version"]:
        raise TrainingReceiptError("snapshot environment python_version is invalid")
    if not isinstance(value.get("python_implementation"), str) or not value["python_implementation"]:
        raise TrainingReceiptError("snapshot environment python_implementation is invalid")
    pyvenv = value.get("pyvenv")
    if not isinstance(pyvenv, Mapping) or str(
        pyvenv.get("include-system-site-packages", "")
    ).lower() != "false":
        raise TrainingReceiptError("snapshot environment pyvenv isolation is invalid")
    if any(not isinstance(key, str) or not isinstance(child, str) for key, child in pyvenv.items()):
        raise TrainingReceiptError("snapshot environment pyvenv values are invalid")
    pyvenv_path = environment_root / "pyvenv.cfg"
    if (
        not pyvenv_path.is_file()
        or _is_link_or_reparse(pyvenv_path)
        or sha256_file(pyvenv_path) != value.get("pyvenv_sha256")
    ):
        raise TrainingReceiptError("snapshot environment pyvenv content differs")
    actual_pyvenv: dict[str, str] = {}
    for number, raw in enumerate(
        pyvenv_path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "=" not in raw:
            raise TrainingReceiptError(
                f"snapshot environment pyvenv line {number} is invalid"
            )
        name, child = raw.split("=", 1)
        normalized = name.strip().lower()
        if not normalized or normalized in actual_pyvenv:
            raise TrainingReceiptError("snapshot environment pyvenv keys are invalid")
        actual_pyvenv[normalized] = child.strip()
    if dict(pyvenv) != actual_pyvenv:
        raise TrainingReceiptError("snapshot environment pyvenv mapping differs")

    pth_files = value.get("pth_files")
    if not isinstance(pth_files, list):
        raise TrainingReceiptError("snapshot environment pth_files is invalid")
    pth_paths: set[str] = set()
    for index, row in enumerate(pth_files):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "sha256", "accepted_paths"
        }:
            raise TrainingReceiptError(f"snapshot environment pth_files[{index}] is invalid")
        relative = _safe_relative(
            row.get("relative_path"), f"snapshot.environment.pth_files[{index}]"
        )
        if not relative.lower().endswith(".pth"):
            raise TrainingReceiptError(
                f"snapshot environment pth_files[{index}] is not a .pth file"
            )
        if relative in pth_paths:
            raise TrainingReceiptError("snapshot environment pth_files are duplicated")
        pth_paths.add(relative)
        expected_pth_sha256 = _digest(
            row.get("sha256"), f"snapshot.environment.pth_files[{index}].sha256"
        )
        if not isinstance(row.get("accepted_paths"), list) or any(
            not isinstance(item, str) or not Path(item).is_absolute()
            for item in row["accepted_paths"]
        ):
            raise TrainingReceiptError(f"snapshot environment pth_files[{index}] paths are invalid")
        accepted_paths: list[Path] = []
        for item in row["accepted_paths"]:
            lexical_accepted = _absolute_path_object(
                item,
                f"snapshot.environment.pth_files[{index}].accepted_paths",
            )
            resolved_accepted = _require_within(
                item,
                roots=(environment_root,),
                field=f"snapshot.environment.pth_files[{index}].accepted_paths",
                resolve_existing=True,
            )
            if (
                not _is_within(lexical_accepted, environment_root)
                or not lexical_accepted.is_dir()
                or _is_link_or_reparse(lexical_accepted)
                or not _is_within(resolved_accepted, environment_root)
            ):
                raise TrainingReceiptError(
                    f"snapshot environment pth_files[{index}] path is unsafe"
                )
            accepted_paths.append(lexical_accepted)
        pth_path = environment_root / relative
        try:
            resolved_pth = pth_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment pth_files[{index}] cannot resolve"
            ) from exc
        if (
            not pth_path.is_file()
            or _is_link_or_reparse(pth_path)
            or not _is_within(resolved_pth, environment_root)
            or sha256_file(pth_path) != expected_pth_sha256
        ):
            raise TrainingReceiptError(
                f"snapshot environment pth_files[{index}] content/path differs"
            )
        parsed_paths: list[Path] = []
        for number, raw in enumerate(
            pth_path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("import ") or line.startswith("import\t"):
                raise TrainingReceiptError(
                    f"snapshot environment pth_files[{index}] has executable line {number}"
                )
            lexical_line = Path(os.path.abspath(pth_path.parent / line))
            resolved_line = _require_within(
                str(lexical_line),
                roots=(environment_root,),
                field=f"snapshot.environment.pth_files[{index}].line[{number}]",
                resolve_existing=True,
            )
            if (
                not _is_within(lexical_line, environment_root)
                or not lexical_line.is_dir()
                or _is_link_or_reparse(lexical_line)
                or not _is_within(resolved_line, environment_root)
            ):
                raise TrainingReceiptError(
                    f"snapshot environment pth_files[{index}] line {number} is unsafe"
                )
            parsed_paths.append(lexical_line)
        if parsed_paths != accepted_paths:
            raise TrainingReceiptError(
                f"snapshot environment pth_files[{index}] accepted paths differ"
            )

    loading = value.get("loading_environment")
    if not isinstance(loading, Mapping) or set(loading) != {
        "variables", "normalized_ld_library_path"
    } or not isinstance(loading.get("variables"), Mapping) or not isinstance(
        loading.get("normalized_ld_library_path"), list
    ):
        raise TrainingReceiptError("snapshot environment loading environment is invalid")
    variables = loading["variables"]
    safe_python = {
        "PYTHONHASHSEED", "PYTHONIOENCODING", "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    }
    fixed_loading = {
        "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
    }
    forbidden_loading = {
        "LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
        "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
        "PYTHONINSPECT", "PYTHONEXEC",
    }
    if not fixed_loading.issubset(variables):
        raise TrainingReceiptError(
            "snapshot environment loading environment omits fixed variables"
        )
    for name, child in variables.items():
        if (
            not isinstance(name, str)
            or child is not None and not isinstance(child, str)
            or name in forbidden_loading
            or name.startswith("DYLD_")
            or name.startswith("PYTHON") and name not in safe_python
            or name not in fixed_loading
            and name not in safe_python
            and not name.startswith("NCCL_")
        ):
            raise TrainingReceiptError(
                f"snapshot environment rejects loader/Python variable: {name}"
            )
    normalized_ld: list[str] = []
    ld_value = variables.get("LD_LIBRARY_PATH")
    if ld_value:
        for index, raw in enumerate(ld_value.split(os.pathsep)):
            candidate = _absolute_path_object(
                raw, f"snapshot.environment.LD_LIBRARY_PATH[{index}]"
            )
            try:
                candidate = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise TrainingReceiptError(
                    "snapshot environment LD_LIBRARY_PATH entry cannot resolve"
                ) from exc
            if not candidate.is_dir() or _is_link_or_reparse(
                _absolute_path_object(
                    raw, f"snapshot.environment.LD_LIBRARY_PATH[{index}]"
                )
            ):
                raise TrainingReceiptError(
                    "snapshot environment LD_LIBRARY_PATH entry is linked/not a directory"
                )
            normalized_ld.append(str(candidate))
    if loading["normalized_ld_library_path"] != normalized_ld:
        raise TrainingReceiptError("snapshot environment LD_LIBRARY_PATH is invalid")

    sys_path_rows = value.get("sys_path")
    if not isinstance(sys_path_rows, list) or not sys_path_rows:
        raise TrainingReceiptError("snapshot environment sys.path is invalid")
    declared_distribution_rows = value.get("distributions")
    if not isinstance(declared_distribution_rows, list) or not declared_distribution_rows:
        raise TrainingReceiptError("snapshot environment distributions are invalid")
    declared_site_roots: set[Path] = set()
    for index, row in enumerate(declared_distribution_rows):
        if not isinstance(row, Mapping):
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] is invalid"
            )
        relative_dist_info = _safe_relative(
            row.get("dist_info"),
            f"snapshot.environment.distributions[{index}].dist_info",
        )
        site_root = (environment_root / relative_dist_info).parent
        if not _is_within(site_root, environment_root):
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] site root escapes"
            )
        declared_site_roots.add(
            _require_canonical_directory(
                site_root,
                f"snapshot.environment.distributions[{index}] site root",
            )
        )
    import_roots_list = [
        _require_canonical_directory(
            code_root / "src", "snapshot.environment code src import root"
        ),
        _require_canonical_directory(
            code_root / "models", "snapshot.environment models import root"
        ),
        runner_root,
        *sorted(declared_site_roots, key=lambda item: str(item)),
        stdlib_root,
    ]
    for optional_import_root in (
        runner_root / "train",
        stdlib_root / "lib-dynload",
    ):
        if optional_import_root.is_dir():
            import_roots_list.append(
                _require_canonical_directory(
                    optional_import_root,
                    "snapshot.environment optional import root",
                )
            )
    import_roots = tuple(import_roots_list)
    for index, item in enumerate(sys_path_rows):
        declared_sys_path = _absolute_path_object(
            item, f"snapshot.environment.sys_path[{index}]"
        )
        candidate = _require_within(
            item,
            roots=import_roots,
            field=f"snapshot.environment.sys_path[{index}]",
            resolve_existing=True,
        )
        if (
            os.path.normcase(str(declared_sys_path))
            != os.path.normcase(str(candidate))
            or _is_link_or_reparse(declared_sys_path)
            or not candidate.is_dir()
            or candidate.suffix.lower() in {".zip", ".egg", ".whl"}
            or candidate not in import_roots
        ):
            raise TrainingReceiptError(
                f"snapshot environment sys_path[{index}] is not an allowed directory"
            )
    meta_rows = value.get("meta_path")
    if not isinstance(meta_rows, list) or not meta_rows:
        raise TrainingReceiptError("snapshot environment meta_path is invalid")
    originless_finders = {
        ("_frozen_importlib", "BuiltinImporter"),
        ("_frozen_importlib", "FrozenImporter"),
    }
    inventoried_meta_origins: set[str] = set()
    for role in ("code", "runner_code"):
        try:
            source_root, source_rows = formal_source_role_files(
                code_root, runner_root=runner_root, role=role
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrainingReceiptError(
                "snapshot environment cannot rebuild source origins"
            ) from exc
        inventoried_meta_origins.update(
            os.path.normcase(str((source_root / row["relative_path"]).resolve(strict=True)))
            for row in source_rows
        )
    raw_environment_files = value.get("files")
    raw_stdlib_files = value.get("stdlib_files")
    if not isinstance(raw_environment_files, list) or not isinstance(
        raw_stdlib_files, list
    ):
        raise TrainingReceiptError(
            "snapshot environment cannot derive inventoried meta_path origins"
        )
    for index, row in enumerate(raw_environment_files):
        if not isinstance(row, Mapping):
            raise TrainingReceiptError(
                f"snapshot environment files[{index}] is invalid"
            )
        relative = _safe_relative(
            row.get("relative_path"),
            f"snapshot.environment.files[{index}].relative_path",
        )
        inventoried_meta_origins.add(
            os.path.normcase(
                str((environment_root / relative).resolve(strict=True))
            )
        )
    for index, row in enumerate(raw_stdlib_files):
        if not isinstance(row, Mapping):
            raise TrainingReceiptError(
                f"snapshot environment stdlib_files[{index}] is invalid"
            )
        relative = _safe_relative(
            row.get("relative_path"),
            f"snapshot.environment.stdlib_files[{index}].relative_path",
        )
        inventoried_meta_origins.add(
            os.path.normcase(str((base_root / relative).resolve(strict=True)))
        )
    for index, row in enumerate(meta_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "module", "qualname", "origin", "origin_sha256"
        }:
            raise TrainingReceiptError(f"snapshot environment meta_path[{index}] is invalid")
        if not isinstance(row.get("module"), str) or not isinstance(row.get("qualname"), str):
            raise TrainingReceiptError(f"snapshot environment meta_path[{index}] identity is invalid")
        if row.get("origin") is None:
            if (
                row.get("origin_sha256") is not None
                or (row.get("module"), row.get("qualname"))
                not in originless_finders
            ):
                raise TrainingReceiptError(f"snapshot environment meta_path[{index}] hash is invalid")
        else:
            declared_origin = _absolute_path_object(
                row.get("origin"),
                f"snapshot.environment.meta_path[{index}].origin",
            )
            origin = _require_within(
                row.get("origin"),
                roots=import_roots,
                field=f"snapshot.environment.meta_path[{index}].origin",
                resolve_existing=True,
            )
            if (
                os.path.normcase(str(declared_origin))
                != os.path.normcase(str(origin))
                or _is_link_or_reparse(declared_origin)
                or not origin.is_file()
                or origin.suffix.lower() in {".pyc", ".pyo"}
                or os.path.normcase(str(origin)) not in inventoried_meta_origins
            ):
                raise TrainingReceiptError(
                    f"snapshot environment meta_path[{index}] origin is invalid"
                )
            _digest(row.get("origin_sha256"), f"snapshot.environment.meta_path[{index}].sha256")
            if sha256_file(origin) != row.get("origin_sha256"):
                raise TrainingReceiptError(
                    f"snapshot environment meta_path[{index}] origin content differs"
                )

    actual_distribution_paths: set[str] = set()
    for directory, directory_names, _file_names in os.walk(
        environment_root, followlinks=False
    ):
        base = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = base / name
            if name == "__pycache__":
                if _is_link_or_reparse(child) or not child.is_dir():
                    raise TrainingReceiptError(
                        "snapshot environment has a linked/non-directory bytecode cache"
                    )
                continue
            if _is_link_or_reparse(child):
                raise TrainingReceiptError(
                    "snapshot environment rejects every directory link/reparse"
                )
            if name.endswith(".dist-info"):
                actual_distribution_paths.add(
                    child.relative_to(environment_root).as_posix()
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories

    links = value.get("internal_links")
    if not isinstance(links, list) or links:
        raise TrainingReceiptError(
            "snapshot environment rejects recorded directory links/reparse points"
        )

    distributions = value.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise TrainingReceiptError("snapshot environment distributions are invalid")
    distribution_paths: set[str] = set()
    declared_distribution_files: dict[
        str, tuple[str | None, int | None, str]
    ] = {}
    for index, row in enumerate(distributions):
        if not isinstance(row, Mapping) or set(row) != {
            "name", "version", "dist_info", "record_sha256", "record_file_count"
        }:
            raise TrainingReceiptError(f"snapshot environment distributions[{index}] is invalid")
        for name in ("name", "version"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise TrainingReceiptError(f"snapshot environment distributions[{index}].{name} is invalid")
        relative = _safe_relative(
            row.get("dist_info"), f"snapshot.environment.distributions[{index}]"
        )
        if relative in distribution_paths or not relative.endswith(".dist-info"):
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] path is invalid/duplicated"
            )
        distribution_paths.add(relative)
        expected_record_sha256 = _digest(
            row.get("record_sha256"),
            f"snapshot.environment.distributions[{index}].record",
        )
        expected_record_count = _positive_int(
            row.get("record_file_count"),
            f"snapshot.environment.distributions[{index}].count",
        )
        dist_info = environment_root / relative
        metadata = dist_info / "METADATA"
        record = dist_info / "RECORD"
        try:
            resolved_dist_info = dist_info.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] cannot resolve"
            ) from exc
        if (
            not dist_info.is_dir()
            or _is_link_or_reparse(dist_info)
            or not _is_within(resolved_dist_info, environment_root)
            or not metadata.is_file()
            or not record.is_file()
            or _is_link_or_reparse(metadata)
            or _is_link_or_reparse(record)
            or sha256_file(record) != expected_record_sha256
        ):
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] files differ"
            )
        try:
            metadata_text = metadata.read_text(encoding="utf-8", errors="strict")
        except UnicodeError as exc:
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] METADATA is invalid"
            ) from exc
        actual_name = next(
            (
                line[6:].strip()
                for line in metadata_text.splitlines()
                if line.startswith("Name: ")
            ),
            None,
        )
        actual_version = next(
            (
                line[9:].strip()
                for line in metadata_text.splitlines()
                if line.startswith("Version: ")
            ),
            None,
        )
        if actual_name != row["name"] or actual_version != row["version"]:
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] identity differs"
            )
        direct_url = dist_info / "direct_url.json"
        if direct_url.is_file():
            if _is_link_or_reparse(direct_url):
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] direct_url is linked"
                )
            try:
                direct_url_value = json.loads(
                    direct_url.read_text(encoding="utf-8", errors="strict")
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] direct_url is invalid"
                ) from exc
            if (
                isinstance(direct_url_value, Mapping)
                and isinstance(direct_url_value.get("dir_info"), Mapping)
                and direct_url_value["dir_info"].get("editable") is True
            ):
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] is editable"
                )
        try:
            record_bytes = record.read_bytes()
            record_rows = list(
                csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict")))
            )
        except UnicodeError as exc:
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] RECORD is invalid"
            ) from exc
        if len(record_rows) != expected_record_count:
            raise TrainingReceiptError(
                f"snapshot environment distributions[{index}] RECORD count differs"
            )
        owner = f"{row['name']}=={row['version']}"
        for number, record_row in enumerate(record_rows, 1):
            if len(record_row) != 3 or not record_row[0]:
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] RECORD row {number} is invalid"
                )
            logical = Path(os.path.abspath(dist_info.parent / Path(record_row[0])))
            if not _is_within(logical, environment_root) or not logical.is_file():
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] RECORD row {number} escapes/is missing"
                )
            expected_sha256 = (
                _decode_record_sha256(
                    record_row[1],
                    f"snapshot.environment.distributions[{index}].RECORD[{number}]",
                )
                if record_row[1]
                else None
            )
            try:
                expected_size = int(record_row[2]) if record_row[2] else None
            except ValueError as exc:
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] RECORD row {number} size is invalid"
                ) from exc
            if expected_size is not None and expected_size < 0:
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] RECORD row {number} size is invalid"
                )
            relative_file = logical.relative_to(environment_root).as_posix()
            current = (expected_sha256, expected_size, owner)
            prior = declared_distribution_files.get(relative_file)
            if prior is not None and prior[:2] != current[:2]:
                raise TrainingReceiptError(
                    "snapshot environment has conflicting distribution RECORD rows"
                )
            declared_distribution_files[relative_file] = current
            if (
                expected_sha256 is not None
                and sha256_file(logical) != expected_sha256
            ) or (
                expected_size is not None
                and logical.stat().st_size != expected_size
            ):
                raise TrainingReceiptError(
                    f"snapshot environment distributions[{index}] RECORD content differs"
                )
    if distribution_paths != actual_distribution_paths:
        raise TrainingReceiptError(
            "snapshot environment distribution inventory is incomplete"
        )

    environment_files = value.get("files")
    if not isinstance(environment_files, list) or not environment_files:
        raise TrainingReceiptError("snapshot environment installed files are invalid")
    installed_total = 0
    installed_paths: set[str] = set()
    for index, row in enumerate(environment_files):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "sha256", "size", "owner", "record_sha256"
        }:
            raise TrainingReceiptError(f"snapshot environment files[{index}] is invalid")
        relative = _safe_relative(
            row.get("relative_path"), f"snapshot.environment.files[{index}]"
        )
        if relative in installed_paths or relative.lower().endswith(".egg-link"):
            raise TrainingReceiptError(
                f"snapshot environment files[{index}] path is invalid/duplicated"
            )
        installed_paths.add(relative)
        expected_file_sha256 = _digest(
            row.get("sha256"), f"snapshot.environment.files[{index}].sha256"
        )
        expected_file_size = _nonnegative_int(
            row.get("size"), f"snapshot.environment.files[{index}].size"
        )
        installed_total += expected_file_size
        if not isinstance(row.get("owner"), str) or not row["owner"]:
            raise TrainingReceiptError(f"snapshot environment files[{index}].owner is invalid")
        if row.get("record_sha256") is not None:
            _digest(row.get("record_sha256"), f"snapshot.environment.files[{index}].record")
        declared_file = declared_distribution_files.get(relative)
        if declared_file is None:
            if (
                row.get("owner") != "unowned-direct-content"
                or row.get("record_sha256") is not None
            ):
                raise TrainingReceiptError(
                    f"snapshot environment files[{index}] has a false distribution owner"
                )
        elif (
            row.get("owner") != declared_file[2]
            or row.get("record_sha256") != declared_file[0]
        ):
            raise TrainingReceiptError(
                f"snapshot environment files[{index}] differs from its distribution RECORD"
            )
        installed = environment_root / relative
        try:
            resolved_installed = installed.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment files[{index}] cannot resolve"
            ) from exc
        if (
            not installed.is_file()
            or _is_link_or_reparse(installed)
            or not _is_within(resolved_installed, environment_root)
            or sha256_file(installed) != expected_file_sha256
            or installed.stat().st_size != expected_file_size
        ):
            raise TrainingReceiptError(
                f"snapshot environment files[{index}] content/path differs"
            )
        if installed.suffix.lower() in {".pyc", ".pyo"}:
            if installed.parent.name == "__pycache__":
                source = installed.parent.parent / (installed.name.split(".", 1)[0] + ".py")
            else:
                source = installed.with_suffix(".py")
            if not source.is_file():
                raise TrainingReceiptError(
                    f"snapshot environment files[{index}] is sourceless bytecode"
                )
    if not set(declared_distribution_files).issubset(installed_paths):
        raise TrainingReceiptError(
            "snapshot environment omits files declared by distribution RECORDs"
        )

    expected_installed_paths = set(declared_distribution_files)
    site_roots = {
        (environment_root / relative).parent for relative in distribution_paths
    }
    for site_root in sorted(site_roots, key=lambda item: str(item)):
        for egg_link in site_root.glob("*.egg-link"):
            raise TrainingReceiptError(
                f"snapshot environment rejects legacy editable egg-link: {egg_link}"
            )
        for directory, directory_names, file_names in os.walk(
            site_root, followlinks=False
        ):
            base = Path(directory)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                child = base / name
                if name == "__pycache__":
                    if _is_link_or_reparse(child) or not child.is_dir():
                        raise TrainingReceiptError(
                            "snapshot environment rejects linked bytecode cache"
                        )
                    for bytecode in sorted(child.iterdir()):
                        if (
                            _is_link_or_reparse(bytecode)
                            or not bytecode.is_file()
                            or bytecode.suffix.lower() not in {".pyc", ".pyo"}
                        ):
                            raise TrainingReceiptError(
                                "snapshot environment has an invalid bytecode cache entry"
                            )
                        if bytecode.parent.name == "__pycache__":
                            source = bytecode.parent.parent / (
                                bytecode.name.split(".", 1)[0] + ".py"
                            )
                        else:  # pragma: no cover - guarded by cache parent
                            source = bytecode.with_suffix(".py")
                        if not source.is_file() or _is_link_or_reparse(source):
                            raise TrainingReceiptError(
                                "snapshot environment has sourceless bytecode"
                            )
                        expected_installed_paths.add(
                            bytecode.relative_to(environment_root).as_posix()
                        )
                    continue
                if _is_link_or_reparse(child):
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                child = base / name
                if _is_link_or_reparse(child) or not child.is_file():
                    raise TrainingReceiptError(
                        "snapshot environment has a linked/non-regular package file"
                    )
                if child.suffix.lower() in {".pyc", ".pyo"}:
                    if child.parent.name == "__pycache__":
                        source = child.parent.parent / (
                            child.name.split(".", 1)[0] + ".py"
                        )
                    else:
                        source = child.with_suffix(".py")
                    if not source.is_file() or _is_link_or_reparse(source):
                        raise TrainingReceiptError(
                            "snapshot environment has sourceless bytecode"
                        )
                expected_installed_paths.add(
                    child.relative_to(environment_root).as_posix()
                )
    expected_installed_paths.add(pyvenv_path.relative_to(environment_root).as_posix())
    scripts_root = environment_root / ("Scripts" if os.name == "nt" else "bin")
    if scripts_root.is_dir():
        for child in scripts_root.iterdir():
            if child.is_file() and not _is_link_or_reparse(child):
                expected_installed_paths.add(
                    child.relative_to(environment_root).as_posix()
                )
    if installed_paths != expected_installed_paths:
        raise TrainingReceiptError(
            "snapshot environment installed file inventory is incomplete/extra"
        )
    if {
        relative for relative in installed_paths if relative.lower().endswith(".pth")
    } != pth_paths:
        raise TrainingReceiptError("snapshot environment .pth inventory is incomplete")

    stdlib_files = value.get("stdlib_files")
    native_files = value.get("native_runtime_files")
    if not isinstance(stdlib_files, list) or not stdlib_files:
        raise TrainingReceiptError("snapshot environment stdlib inventory is invalid")
    if not isinstance(native_files, list) or not native_files:
        raise TrainingReceiptError("snapshot environment native runtime inventory is invalid")
    actual_stdlib_files: dict[str, tuple[str, int]] = {}
    for directory, directory_names, file_names in os.walk(
        stdlib_root, followlinks=False
    ):
        base = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = base / name
            if name in {"site-packages", "dist-packages"}:
                continue
            if _is_link_or_reparse(child):
                raise TrainingReceiptError(
                    "snapshot environment base stdlib has a linked directory"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            child = base / name
            if _is_link_or_reparse(child) or not child.is_file():
                raise TrainingReceiptError(
                    "snapshot environment base stdlib has a non-regular file"
                )
            if child.suffix.lower() in {".pyc", ".pyo"}:
                if child.parent.name == "__pycache__":
                    source = child.parent.parent / (
                        child.name.split(".", 1)[0] + ".py"
                    )
                else:
                    source = child.with_suffix(".py")
                if not source.is_file() or _is_link_or_reparse(source):
                    raise TrainingReceiptError(
                        "snapshot environment base stdlib has sourceless bytecode"
                    )
            actual_stdlib_files[child.relative_to(base_root).as_posix()] = (
                sha256_file(child),
                child.stat().st_size,
            )
    stdlib_total = 0
    stdlib_paths: set[str] = set()
    for index, row in enumerate(stdlib_files):
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256", "size"}:
            raise TrainingReceiptError(f"snapshot environment stdlib_files[{index}] is invalid")
        relative = _safe_relative(
            row.get("relative_path"), f"snapshot.environment.stdlib_files[{index}]"
        )
        if relative in stdlib_paths:
            raise TrainingReceiptError("snapshot environment stdlib paths are duplicated")
        stdlib_paths.add(relative)
        expected_stdlib_sha256 = _digest(
            row.get("sha256"), f"snapshot.environment.stdlib_files[{index}].sha256"
        )
        expected_stdlib_size = _nonnegative_int(
            row.get("size"), f"snapshot.environment.stdlib_files[{index}].size"
        )
        stdlib_total += expected_stdlib_size
        if actual_stdlib_files.get(relative) != (
            expected_stdlib_sha256,
            expected_stdlib_size,
        ):
            raise TrainingReceiptError(
                f"snapshot environment stdlib_files[{index}] differs from inventory"
            )
        stdlib_file = base_root / relative
        try:
            resolved_stdlib = stdlib_file.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment stdlib_files[{index}] cannot resolve"
            ) from exc
        if (
            not stdlib_file.is_file()
            or stdlib_file.is_symlink()
            or not _is_within(resolved_stdlib, stdlib_root)
            or sha256_file(stdlib_file) != expected_stdlib_sha256
            or stdlib_file.stat().st_size != expected_stdlib_size
        ):
            raise TrainingReceiptError(
                f"snapshot environment stdlib_files[{index}] content/path differs"
            )
        if stdlib_file.suffix.lower() in {".pyc", ".pyo"}:
            if stdlib_file.parent.name == "__pycache__":
                source = stdlib_file.parent.parent / (
                    stdlib_file.name.split(".", 1)[0] + ".py"
                )
            else:
                source = stdlib_file.with_suffix(".py")
            if not source.is_file():
                raise TrainingReceiptError(
                    f"snapshot environment stdlib_files[{index}] is sourceless bytecode"
                )
    if stdlib_paths != set(actual_stdlib_files):
        raise TrainingReceiptError(
            "snapshot environment stdlib inventory is incomplete/extra"
        )
    derived_native_roots: set[Path] = {environment_root, base_root}
    derived_native_roots.update(Path(item) for item in normalized_ld)
    for name in ("CUDA_HOME", "CUDA_PATH"):
        raw = variables.get(name)
        if raw in (None, ""):
            continue
        lexical_candidate = _absolute_path_object(
            raw, f"snapshot.environment.loading_environment.{name}"
        )
        try:
            candidate = lexical_candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment {name} cannot resolve"
            ) from exc
        if not candidate.is_dir() or _is_link_or_reparse(lexical_candidate):
            raise TrainingReceiptError(
                f"snapshot environment {name} is linked/not a directory"
            )
        derived_native_roots.add(candidate)
    expected_native_roots = sorted(
        (str(item) for item in derived_native_roots), key=os.path.normcase
    )
    declared_native_roots = value.get("native_allowed_roots")
    if declared_native_roots != expected_native_roots:
        raise TrainingReceiptError("snapshot environment native allowed roots are invalid")
    native_roots = tuple(
        _require_canonical_directory(
            Path(item), f"snapshot.environment.native_allowed_roots[{index}]"
        )
        for index, item in enumerate(expected_native_roots)
    )

    native_total = 0
    native_paths: set[tuple[str, str]] = set()
    for index, row in enumerate(native_files):
        if not isinstance(row, Mapping) or set(row) != {
            "logical_path", "link_sha256", "resolved_target", "sha256", "size"
        }:
            raise TrainingReceiptError(f"snapshot environment native_runtime_files[{index}] is invalid")
        logical = _require_within(
            row.get("logical_path"),
            roots=native_roots,
            field=f"snapshot.environment.native_runtime_files[{index}].logical_path",
        )
        target = _require_within(
            row.get("resolved_target"),
            roots=native_roots,
            field=f"snapshot.environment.native_runtime_files[{index}].resolved_target",
            resolve_existing=True,
        )
        declared_target = _absolute_path_object(
            row.get("resolved_target"),
            f"snapshot.environment.native_runtime_files[{index}].resolved_target",
        )
        if os.path.normcase(str(declared_target)) != os.path.normcase(str(target)):
            raise TrainingReceiptError(
                f"snapshot environment native_runtime_files[{index}] target is not canonical"
            )
        identity_pair = (str(logical), str(target))
        if identity_pair in native_paths:
            raise TrainingReceiptError("snapshot environment native runtimes are duplicated")
        native_paths.add(identity_pair)
        if row.get("link_sha256") is not None:
            _digest(
                row.get("link_sha256"),
                f"snapshot.environment.native_runtime_files[{index}].link_sha256",
            )
        expected_native_sha256 = _digest(
            row.get("sha256"), f"snapshot.environment.native_runtime_files[{index}].sha256"
        )
        expected_native_size = _positive_int(
            row.get("size"), f"snapshot.environment.native_runtime_files[{index}].size"
        )
        native_total += expected_native_size
        try:
            actual_target = logical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrainingReceiptError(
                f"snapshot environment native_runtime_files[{index}] cannot resolve"
            ) from exc
        if (
            actual_target != target
            or not target.is_file()
            or _is_link_or_reparse(target)
            or sha256_file(target) != expected_native_sha256
            or target.stat().st_size != expected_native_size
        ):
            raise TrainingReceiptError(
                f"snapshot environment native_runtime_files[{index}] content/path differs"
            )
        logical_is_link = _is_link_or_reparse(logical)
        if row.get("link_sha256") is None:
            if logical_is_link or logical != target:
                raise TrainingReceiptError(
                    f"snapshot environment native_runtime_files[{index}] has an unrecorded link"
                )
        else:
            if not logical_is_link:
                raise TrainingReceiptError(
                    f"snapshot environment native_runtime_files[{index}] falsely records a link"
                )
            try:
                actual_link_sha256 = hash_path(logical, symlink_policy="link").digest
            except (OSError, RuntimeError, ValueError) as exc:
                raise TrainingReceiptError(
                    f"snapshot environment native_runtime_files[{index}] link is invalid"
                ) from exc
            if actual_link_sha256 != row.get("link_sha256"):
                raise TrainingReceiptError(
                    f"snapshot environment native_runtime_files[{index}] link differs"
                )
    expected_count = len(environment_files) + len(links) + len(stdlib_files) + len(native_files) + 1
    expected_bytes = installed_total + stdlib_total + native_total + interpreter_size
    if evidence.get("file_count") != expected_count or evidence.get("total_bytes") != expected_bytes:
        raise TrainingReceiptError("snapshot environment evidence counts differ from manifest")


def validate_formal_provenance_snapshot(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly parse and self-bind a Qwen in-process provenance snapshot."""

    fields = {
        "schema_version", "status", "batch_id", "model_id", "training_mode",
        "canonical_identity", "provenance", "manifests", "snapshot_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingReceiptError("formal provenance snapshot schema is invalid")
    snapshot = dict(value)
    body = {key: child for key, child in snapshot.items() if key != "snapshot_sha256"}
    if (
        snapshot.get("schema_version") != FORMAL_PROVENANCE_SNAPSHOT_SCHEMA
        or snapshot.get("status") != _SNAPSHOT_STATUS
        or snapshot.get("snapshot_sha256") != sha256_json(body)
    ):
        raise TrainingReceiptError("formal provenance snapshot self-hash/status is invalid")
    _digest(snapshot.get("snapshot_sha256"), "snapshot_sha256")
    for name in ("batch_id", "model_id"):
        _identifier(snapshot.get(name), f"snapshot.{name}")
    if snapshot.get("training_mode") not in {"full_sft", "lora_sft"}:
        raise TrainingReceiptError("formal provenance snapshot training_mode is invalid")
    identity = snapshot.get("canonical_identity")
    identity_fields = {
        "model_id", "model_family", "modality", "base_artifact", "code_root",
        "runner_code_root", "environment_root", "interpreter", "base_environment_root",
    }
    if not isinstance(identity, Mapping) or set(identity) != identity_fields:
        raise TrainingReceiptError("formal provenance snapshot canonical identity is invalid")
    if identity.get("model_id") != snapshot.get("model_id"):
        raise TrainingReceiptError("formal provenance snapshot model identity differs")
    _identifier(identity.get("model_family"), "snapshot.model_family")
    if identity.get("modality") not in {"V", "M", "VM"}:
        raise TrainingReceiptError("formal provenance snapshot modality is invalid")
    for name in (
        "base_artifact", "code_root", "runner_code_root", "environment_root",
        "interpreter", "base_environment_root",
    ):
        _absolute_path(identity.get(name), f"snapshot.canonical_identity.{name}")

    provenance = snapshot.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingReceiptError("formal provenance snapshot evidence is invalid")
    roles = set(provenance)
    if not _SNAPSHOT_REQUIRED_ROLES.issubset(roles) or not roles.issubset(
        _SNAPSHOT_REQUIRED_ROLES | _SNAPSHOT_OPTIONAL_ROLES
    ):
        raise TrainingReceiptError("formal provenance snapshot roles are incomplete/unknown")
    evidence = {
        role: _validate_snapshot_evidence(child, f"snapshot.provenance.{role}")
        for role, child in provenance.items()
    }
    if evidence["base_artifact"]["path"] != identity["base_artifact"]:
        raise TrainingReceiptError("formal provenance base artifact identity differs")
    if evidence["code"]["path"] != identity["code_root"]:
        raise TrainingReceiptError("formal provenance code identity differs")
    if evidence["runner_code"]["path"] != identity["runner_code_root"]:
        raise TrainingReceiptError("formal provenance runner identity differs")
    if evidence["environment"]["path"] != identity["environment_root"]:
        raise TrainingReceiptError("formal provenance environment identity differs")

    manifests = snapshot.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {
        "code", "runner_code", "environment"
    }:
        raise TrainingReceiptError("formal provenance detailed manifests are incomplete")
    _validate_source_snapshot_manifest(
        manifests["code"],
        role="code",
        evidence=evidence["code"],
        expected_root=_absolute_path_object(
            identity["code_root"], "snapshot.canonical_identity.code_root"
        ),
        project_root=_absolute_path_object(
            identity["code_root"], "snapshot.canonical_identity.code_root"
        ),
        runner_root=_absolute_path_object(
            identity["runner_code_root"],
            "snapshot.canonical_identity.runner_code_root",
        ),
    )
    _validate_source_snapshot_manifest(
        manifests["runner_code"],
        role="runner_code",
        evidence=evidence["runner_code"],
        expected_root=_absolute_path_object(
            identity["runner_code_root"],
            "snapshot.canonical_identity.runner_code_root",
        ),
        project_root=_absolute_path_object(
            identity["code_root"], "snapshot.canonical_identity.code_root"
        ),
        runner_root=_absolute_path_object(
            identity["runner_code_root"],
            "snapshot.canonical_identity.runner_code_root",
        ),
    )
    _validate_environment_snapshot_manifest(
        manifests["environment"],
        evidence=evidence["environment"],
        identity=identity,
    )
    if expected is not None:
        allowed = {"batch_id", "model_id", "training_mode", "snapshot_sha256"}
        unknown = set(expected) - allowed
        if unknown:
            raise TrainingReceiptError(f"unknown snapshot expectations: {sorted(unknown)}")
        mismatches = [name for name, item in expected.items() if snapshot.get(name) != item]
        if mismatches:
            raise TrainingReceiptError(f"formal provenance snapshot mismatch: {sorted(mismatches)}")
    return snapshot


def load_and_validate_formal_provenance_snapshot(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _digest(expected_file_sha256, "provenance_snapshot_file_sha256")
    candidate = Path(path)
    if not candidate.is_file() or sha256_file(candidate) != expected_file_sha256:
        raise TrainingReceiptError("formal provenance snapshot file hash changed")
    return validate_formal_provenance_snapshot(load_json_strict(candidate), expected=expected)


def validate_training_receipt(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact schema, self-hash, progress, gradients, and weight change."""

    if not isinstance(value, Mapping):
        raise TrainingReceiptError("training receipt schema is invalid")
    receipt = dict(value)
    schema_version = receipt.get("schema_version")
    expected_keys = (
        TRAINING_RECEIPT_V2_KEYS
        if schema_version == "2.0"
        else TRAINING_RECEIPT_KEYS
        if schema_version == "1.0"
        else None
    )
    if expected_keys is None or set(receipt) != expected_keys:
        raise TrainingReceiptError("training receipt schema is invalid")
    if receipt.get("status") != "trained":
        raise TrainingReceiptError("training receipt is not completed evidence")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != sha256_json(body):
        raise TrainingReceiptError("training receipt self-hash mismatch")

    for field in ("batch_id", "model_id", "backend_id", "model_family"):
        _identifier(receipt.get(field), field)
    if receipt.get("modality") not in {"V", "M", "VM"}:
        raise TrainingReceiptError("modality must be V, M, or VM")
    if receipt.get("training_mode") not in _TRAINING_MODES:
        raise TrainingReceiptError("training_mode is not a supported formal mode")
    created_at = receipt.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise TrainingReceiptError("created_at must be a non-empty timestamp")
    try:
        parsed_time = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise TrainingReceiptError("created_at must be an ISO-8601 timestamp") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise TrainingReceiptError("created_at must include a timezone")

    planned_global = _positive_int(receipt.get("planned_global_steps"), "planned_global_steps")
    actual_global = _positive_int(receipt.get("actual_global_steps"), "actual_global_steps")
    planned_optimizer = _positive_int(
        receipt.get("planned_optimizer_steps"), "planned_optimizer_steps"
    )
    actual_optimizer = _positive_int(
        receipt.get("actual_optimizer_steps"), "actual_optimizer_steps"
    )
    if actual_global != planned_global or actual_optimizer != planned_optimizer:
        raise TrainingReceiptError("training did not complete every planned step")
    if actual_global != actual_optimizer:
        raise TrainingReceiptError("global and optimizer step counts must agree")
    gradient_steps = _positive_int(
        receipt.get("nonzero_finite_gradient_steps"),
        "nonzero_finite_gradient_steps",
    )
    if gradient_steps != actual_optimizer:
        raise TrainingReceiptError(
            "every optimizer step must have a nonzero finite trainable gradient"
        )
    _finite_positive(receipt.get("max_gradient"), "max_gradient")

    losses = receipt.get("finite_losses")
    if not isinstance(losses, list) or not losses:
        raise TrainingReceiptError("finite_losses must be a non-empty list")
    for index, loss in enumerate(losses):
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise TrainingReceiptError(f"finite_losses[{index}] must be a number")
        if not math.isfinite(float(loss)):
            raise TrainingReceiptError(f"finite_losses[{index}] must be finite")

    tensor_count = _positive_int(
        receipt.get("trainable_tensor_count"), "trainable_tensor_count"
    )
    _positive_int(
        receipt.get("trainable_parameter_count"), "trainable_parameter_count"
    )
    changed_count = _positive_int(
        receipt.get("changed_trainable_tensor_count"),
        "changed_trainable_tensor_count",
    )
    if changed_count > tensor_count:
        raise TrainingReceiptError(
            "changed_trainable_tensor_count exceeds trainable_tensor_count"
        )
    initial_hash = _digest(
        receipt.get("initial_trainable_sha256"), "initial_trainable_sha256"
    )
    final_hash = _digest(
        receipt.get("final_trainable_sha256"), "final_trainable_sha256"
    )
    if initial_hash == final_hash:
        raise TrainingReceiptError("trainable state did not change")
    update = receipt.get("max_parameter_update")
    if update is not None:
        _finite_positive(update, "max_parameter_update")

    for field in (
        "batch_receipt_sha256",
        "attempt_sha256",
        "train_sha256",
        "validation_sha256",
        "leakage_audit_sha256",
        "base_artifact_sha256",
        "config_sha256",
        "code_sha256",
        "runner_code_sha256",
        "environment_sha256",
        "artifact_sha256",
        "receipt_sha256",
    ):
        _digest(receipt.get(field), field)

    if schema_version == "2.0":
        snapshot_path = receipt.get("provenance_snapshot_path")
        if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_absolute():
            raise TrainingReceiptError(
                "provenance_snapshot_path must be an absolute path"
            )
        for field in (
            "provenance_snapshot_file_sha256",
            "provenance_pre_sha256",
            "provenance_post_sha256",
        ):
            _digest(receipt.get(field), field)
        if (
            receipt.get("provenance_unchanged") is not True
            or receipt["provenance_pre_sha256"]
            != receipt["provenance_post_sha256"]
        ):
            raise TrainingReceiptError(
                "formal provenance pre/post snapshot must be unchanged"
            )

    if expected is not None:
        unknown = set(expected) - expected_keys
        if unknown:
            raise TrainingReceiptError(
                f"validator requested unknown receipt bindings: {sorted(unknown)}"
            )
        mismatches = [
            field for field, expected_value in expected.items()
            if receipt.get(field) != expected_value
        ]
        if mismatches:
            raise TrainingReceiptError(
                f"training receipt binding mismatch: {sorted(mismatches)}"
            )
    return receipt


def make_training_receipt(
    *,
    finite_losses: Sequence[float],
    **fields: Any,
) -> dict[str, Any]:
    """Build and immediately validate a canonical receipt payload."""

    toctou_fields = TRAINING_RECEIPT_V2_KEYS - TRAINING_RECEIPT_KEYS
    supplied_toctou = toctou_fields & set(fields)
    if supplied_toctou and supplied_toctou != toctou_fields:
        raise TrainingReceiptError(
            "formal provenance receipt fields must be supplied together"
        )
    body = {
        "schema_version": "2.0" if supplied_toctou else "1.0",
        "status": "trained",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **fields,
        "finite_losses": [float(value) for value in finite_losses],
    }
    receipt = {**body, "receipt_sha256": sha256_json(body)}
    return validate_training_receipt(receipt)


def load_and_validate_training_receipt(
    path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_training_receipt(load_json_strict(path), expected=expected)


def write_training_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
    *,
    root: str | Path,
    overwrite: bool = False,
) -> Path:
    validated = validate_training_receipt(receipt)
    return atomic_write_json(path, validated, root=root, overwrite=overwrite)


__all__ = [
    "FORMAL_PROVENANCE_SNAPSHOT_SCHEMA",
    "TRAINING_RECEIPT_KEYS",
    "TRAINING_RECEIPT_V2_KEYS",
    "TrainingReceiptError",
    "load_and_validate_formal_provenance_snapshot",
    "load_and_validate_training_receipt",
    "make_training_receipt",
    "validate_training_receipt",
    "validate_formal_provenance_snapshot",
    "write_training_receipt",
]
