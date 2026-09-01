"""Frozen input receipts for the clean evaluation controller.

This module defines a new, local receipt schema.  It intentionally does not
claim compatibility with receipts produced by an unknown historical tree.
Only receipts carrying :data:`BATCH_RECEIPT_SCHEMA_VERSION` are accepted.
"""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motion_eval.core import (
    atomic_write_json,
    hash_path,
    is_link_or_reparse as _is_link_or_reparse,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

from .benchmark import BenchmarkItem, load_benchmark
from .jsonio import StrictJsonError, load_json_strict, load_jsonl_strict


BATCH_RECEIPT_SCHEMA_VERSION = "motion-eval-batch-receipt/1"
LEAKAGE_AUDIT_SCHEMA_VERSION = "2.0"
LEAKAGE_ALGORITHM_VERSION = "motion-eval-clean-leakage/1"
LEAKAGE_ALGORITHM_SHA256 = sha256_json(
    {
        "version": LEAKAGE_ALGORITHM_VERSION,
        "features": [
            "sample_id",
            "group_id",
            "media_sha256",
            "normalized_question_options",
            "near_duplicate",
        ],
        "comparison": "all-cross-split-pairs",
        "normalization": "unicode-nfkc-casefold-whitespace-v1",
    }
)
PRETRAINED_INDEX_SCHEMA_VERSION = "2.0"
MEDIA_MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_INPUT_ROLES = frozenset(
    {
        "train",
        "validation",
        "benchmark",
        "media_manifest",
        "derivation_code",
        "leakage_audit",
    }
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HASH_FIELDS = ("algorithm", "kind", "digest", "file_count", "total_bytes")
_CHECK_NAMES = (
    "sample_id",
    "group_id",
    "media_sha256",
    "normalized_question_options",
    "near_duplicate",
)


class BatchReceiptError(ValueError):
    """Input evidence is incomplete, changed, or internally inconsistent."""


def validate_batch_id(batch_id: str) -> str:
    """Return a conservative filesystem-safe batch identifier."""

    if not isinstance(batch_id, str) or not _SAFE_ID.fullmatch(batch_id):
        raise BatchReceiptError(
            "batch_id must be 1-128 ASCII letters, digits, dot, underscore, or hyphen"
        )
    if batch_id in {".", ".."}:
        raise BatchReceiptError("batch_id cannot be a path component alias")
    return batch_id


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BatchReceiptError(f"{label} must be an object")
    return value


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): child for key, child in value.items()}


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise BatchReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BatchReceiptError(f"{label} must be a non-empty explicit string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BatchReceiptError(f"{label} contains a control character")
    return value


def _absolute_existing(path: str | Path, label: str) -> Path:
    try:
        candidate = Path(path)
    except (TypeError, ValueError) as exc:
        raise BatchReceiptError(f"{label} path is invalid") from exc
    if not candidate.is_absolute():
        raise BatchReceiptError(f"{label} path must be absolute")
    try:
        if _is_link_or_reparse(candidate):
            raise BatchReceiptError(f"{label} path must not be a link/reparse point")
        resolved = candidate.resolve(strict=True)
    except BatchReceiptError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise BatchReceiptError(f"{label} path cannot be resolved") from exc
    return resolved


def _hash_receipt(path: str | Path, label: str) -> dict[str, Any]:
    resolved = _absolute_existing(path, label)
    try:
        digest = hash_path(resolved, symlink_policy="reject")
    except Exception as exc:
        raise BatchReceiptError(f"{label} cannot be hashed safely") from exc
    return {"path": str(resolved), **digest.to_dict()}


def _verify_hash_receipt(value: object, label: str) -> Path:
    receipt = _mapping(value, label)
    path = _absolute_existing(receipt.get("path"), label)
    try:
        actual = hash_path(path, symlink_policy="reject").to_dict()
    except Exception as exc:
        raise BatchReceiptError(f"{label} cannot be rehashed safely") from exc
    expected = {field: receipt.get(field) for field in _HASH_FIELDS}
    if expected != actual:
        raise BatchReceiptError(f"{label} changed after batch freeze")
    return path


def _root_identity(path: Path) -> dict[str, int]:
    info = path.stat()
    return {"device": int(info.st_dev), "inode": int(info.st_ino), "mode": int(info.st_mode)}


def _entry_identity(path: Path) -> dict[str, int | str]:
    info = path.lstat()
    mode = info.st_mode
    if stat.S_ISREG(mode):
        kind = "file"
    elif stat.S_ISDIR(mode):
        kind = "directory"
    elif stat.S_ISLNK(mode):
        kind = "symlink"
    else:
        kind = "other"
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(mode),
        "size": int(info.st_size),
        "kind": kind,
    }


def _safe_relative_path(value: object, label: str) -> Path:
    text = _identity(value, label)
    relative = Path(text)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise BatchReceiptError(f"{label} must be a traversal-free relative path")
    return relative


def _freeze_pretrained_assets(
    pretrained_root: str | Path,
    specs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Hash every present pretrained component; retain explicit missing rows."""

    root = _absolute_existing(pretrained_root, "pretrained root")
    if not root.is_dir():
        raise BatchReceiptError("pretrained root must be a directory")
    if not isinstance(specs, Mapping) or not specs:
        raise BatchReceiptError("pretrained artifact specs must be a non-empty object")

    frozen: dict[str, list[dict[str, Any]]] = {}
    for model_id, raw_rows in specs.items():
        model = _identity(model_id, "pretrained model_id")
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise BatchReceiptError(f"pretrained specs for {model} must be a list")
        rows: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        for raw in raw_rows:
            spec = _mapping(raw, f"pretrained spec for {model}")
            role = _identity(spec.get("role"), f"pretrained role for {model}")
            if role in seen_roles:
                raise BatchReceiptError(f"duplicate pretrained role for {model}: {role}")
            seen_roles.add(role)
            relative = _safe_relative_path(
                spec.get("path"), f"pretrained path for {model}/{role}"
            )
            kind = _identity(spec.get("kind"), f"pretrained kind for {model}/{role}")
            expected = spec.get("expected_sha256")
            if expected is not None:
                expected = _sha(expected, f"expected SHA-256 for {model}/{role}")
            candidate = root.joinpath(relative)
            lexical = Path(os.path.abspath(candidate))
            try:
                lexical.relative_to(root)
            except ValueError as exc:
                raise BatchReceiptError("pretrained component escaped the pretrained root") from exc

            content: dict[str, Any] | None
            if candidate.exists() or candidate.is_symlink():
                try:
                    content = hash_path(
                        candidate, symlink_policy="follow", allowed_root=root
                    ).to_dict()
                    resolved_path = candidate.resolve(strict=True)
                except Exception as exc:
                    raise BatchReceiptError(
                        f"pretrained component cannot be frozen: {model}/{role}"
                    ) from exc
                if expected is not None and content["digest"] != expected:
                    raise BatchReceiptError(
                        f"pretrained component expected hash mismatch: {model}/{role}"
                    )
                state = "present"
                path_text = str(lexical)
            else:
                content = None
                state = "missing"
                path_text = str(lexical)
            rows.append(
                {
                    "role": role,
                    "relative_path": relative.as_posix(),
                    "path": path_text,
                    "kind": kind,
                    "expected_sha256": expected,
                    "state": state,
                    "content": content,
                }
            )
        frozen[model] = rows
    return frozen, sha256_json(frozen)


def _verify_pretrained_assets(assets: object) -> None:
    """Perform the explicit expensive content barrier for frozen assets."""

    asset_map = _mapping(assets, "pretrained assets")
    for model_id, raw_rows in asset_map.items():
        if not isinstance(raw_rows, list):
            raise BatchReceiptError("pretrained asset rows must be lists")
        for raw in raw_rows:
            row = _mapping(raw, f"pretrained asset for {model_id}")
            state = row.get("state")
            try:
                path = Path(row.get("path"))
            except (TypeError, ValueError) as exc:
                raise BatchReceiptError("pretrained asset path is invalid") from exc
            if state == "missing":
                if path.exists() or path.is_symlink():
                    raise BatchReceiptError(
                        "frozen missing pretrained asset appeared after batch freeze"
                    )
                continue
            if state != "present" or not isinstance(row.get("content"), Mapping):
                raise BatchReceiptError("pretrained asset state/content is invalid")
            try:
                actual = hash_path(path, symlink_policy="follow").to_dict()
            except Exception as exc:
                raise BatchReceiptError("pretrained asset changed after batch freeze") from exc
            if actual != dict(row["content"]):
                raise BatchReceiptError("pretrained asset changed after batch freeze")


def _build_trusted_pretrained_index(
    pretrained_root: str | Path,
    assets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    event_trust: Mapping[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    """Build a metadata-only index used on ordinary controller transitions."""

    root = _absolute_existing(pretrained_root, "pretrained root")
    if not root.is_dir():
        raise BatchReceiptError("pretrained root must be a directory")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise BatchReceiptError("trusted index timestamp must be timezone-aware")
    trust = _plain_mapping(_mapping(event_trust, "event trust"))
    bindings: list[dict[str, Any]] = []
    for model_id, raw_rows in assets.items():
        for raw in raw_rows:
            row = _mapping(raw, f"pretrained asset for {model_id}")
            logical = Path(_identity(row.get("path"), "pretrained logical path"))
            binding: dict[str, Any] = {
                "model_id": model_id,
                "role": row.get("role"),
                "logical_path": str(logical),
                "state": row.get("state"),
            }
            if row.get("state") == "present":
                try:
                    binding["entry_identity"] = _entry_identity(logical)
                except OSError as exc:
                    raise BatchReceiptError("pretrained entry disappeared during index build") from exc
                if _is_link_or_reparse(logical):
                    try:
                        binding["link_target_sha256"] = hash_path(
                            logical, symlink_policy="link", allowed_root=root
                        ).digest
                    except Exception as exc:
                        raise BatchReceiptError("pretrained link target cannot be frozen") from exc
            elif row.get("state") != "missing":
                raise BatchReceiptError("pretrained asset state is invalid")
            bindings.append(binding)
    body = {
        "schema_version": PRETRAINED_INDEX_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "pretrained_root": str(root),
        "root_identity": _root_identity(root),
        "assets_sha256": sha256_json(assets),
        "event_trust_sha256": sha256_json(trust),
        "path_bindings": bindings,
    }
    return {**body, "index_sha256": sha256_json(body)}


def _verify_trusted_pretrained_index(
    index: object,
    *,
    assets: Mapping[str, Sequence[Mapping[str, Any]]],
    pretrained_root: str | Path,
    event_trust: Mapping[str, Any],
) -> None:
    """Verify identity metadata without hashing ordinary asset content."""

    trusted = _mapping(index, "trusted pretrained index")
    body = {key: value for key, value in trusted.items() if key != "index_sha256"}
    if (
        trusted.get("schema_version") != PRETRAINED_INDEX_SCHEMA_VERSION
        or trusted.get("index_sha256") != sha256_json(body)
        or trusted.get("assets_sha256") != sha256_json(assets)
        or trusted.get("event_trust_sha256") != sha256_json(event_trust)
    ):
        raise BatchReceiptError("trusted pretrained index hash/trust binding is invalid")

    root = _absolute_existing(pretrained_root, "pretrained root")
    if (
        trusted.get("pretrained_root") != str(root)
        or trusted.get("root_identity") != _root_identity(root)
    ):
        raise BatchReceiptError("trusted pretrained root was exchanged")
    bindings = trusted.get("path_bindings")
    if not isinstance(bindings, list):
        raise BatchReceiptError("trusted pretrained index hash/trust binding is invalid")
    for raw in bindings:
        binding = _mapping(raw, "trusted pretrained path binding")
        path = Path(_identity(binding.get("logical_path"), "trusted logical path"))
        state = binding.get("state")
        if state == "missing":
            # The fast index authenticates receipt/path bindings but does not
            # turn a formerly absent path into trusted content.  Appearance is
            # revalidated by the explicit full-content barrier, where the
            # controller can report the frozen missing-component condition.
            continue
        if state != "present":
            raise BatchReceiptError("trusted pretrained path state is invalid")
        try:
            actual_identity = _entry_identity(path)
        except OSError as exc:
            raise BatchReceiptError("pretrained entry identity changed") from exc
        if actual_identity != binding.get("entry_identity"):
            raise BatchReceiptError("pretrained entry identity changed")
        if _is_link_or_reparse(path):
            try:
                target = hash_path(path, symlink_policy="link", allowed_root=root).digest
            except Exception as exc:
                raise BatchReceiptError("pretrained link target changed") from exc
            if target != binding.get("link_target_sha256"):
                raise BatchReceiptError("pretrained link target changed")
        elif "link_target_sha256" in binding:
            raise BatchReceiptError("pretrained link target changed")


def _read_slice(path: Path, offset: int, length: int) -> bytes:
    try:
        before = path.stat()
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        after = path.stat()
    except OSError as exc:
        raise BatchReceiptError("media resource cannot be read") from exc
    if len(payload) != length or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BatchReceiptError("media resource changed while reading its byte range")
    return payload


def _validate_media_manifest(
    path: str | Path,
    benchmark_items: Sequence[BenchmarkItem],
    *,
    required_kinds: frozenset[str],
) -> dict[str, Any]:
    """Verify manifest bytes and one canonical resource per required modality."""

    try:
        source = _absolute_existing(path, "media manifest")
        manifest = _mapping(load_json_strict(source), "media manifest")
    except (StrictJsonError, OSError) as exc:
        raise BatchReceiptError("media manifest is invalid") from exc
    if manifest.get("schema_version") != MEDIA_MANIFEST_SCHEMA_VERSION:
        raise BatchReceiptError("media manifest schema_version is unsupported")
    if not required_kinds or not required_kinds.issubset({"video", "motion"}):
        raise BatchReceiptError("media required kinds must be video and/or motion")
    resources_raw = manifest.get("resources")
    rows_raw = manifest.get("rows")
    if not isinstance(resources_raw, list) or not isinstance(rows_raw, list):
        raise BatchReceiptError("media manifest resources and rows must be lists")
    if manifest.get("row_count") != len(benchmark_items) or len(rows_raw) != len(
        benchmark_items
    ):
        raise BatchReceiptError("media manifest row_count must equal benchmark size")

    resources: dict[str, dict[str, Any]] = {}
    container_hashes: dict[Path, str] = {}
    for raw in resources_raw:
        resource = _mapping(raw, "media resource")
        resource_id = _identity(resource.get("resource_id"), "media resource_id")
        if resource_id in resources:
            raise BatchReceiptError("media resource_id must be unique")
        kind = resource.get("kind")
        if kind not in {"video", "motion"}:
            raise BatchReceiptError("media resource kind must be video or motion")
        media_path = _absolute_existing(resource.get("path"), "media resource")
        if not media_path.is_file():
            raise BatchReceiptError("media resource must be a regular file")
        expected_container = _sha(resource.get("sha256"), "media container SHA-256")
        if media_path not in container_hashes:
            try:
                container_hashes[media_path] = sha256_file(media_path)
            except Exception as exc:
                raise BatchReceiptError("media resource cannot be hashed safely") from exc
        if container_hashes[media_path] != expected_container:
            raise BatchReceiptError("media resource hash mismatch")

        has_range = any(
            key in resource for key in ("offset", "length", "content_sha256")
        )
        canonical: dict[str, Any] = {
            "path": str(media_path),
            "sha256": expected_container,
        }
        if has_range:
            offset, length = resource.get("offset"), resource.get("length")
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length <= 0
                or offset + length > media_path.stat().st_size
            ):
                raise BatchReceiptError("media resource byte range is invalid")
            content_sha = _sha(
                resource.get("content_sha256"), "media content SHA-256"
            )
            if sha256_bytes(_read_slice(media_path, offset, length)) != content_sha:
                raise BatchReceiptError("media resource content hash mismatch")
            canonical.update(
                {"offset": offset, "length": length, "content_sha256": content_sha}
            )
        resources[resource_id] = {
            "resource_id": resource_id,
            "kind": kind,
            "canonical": canonical,
            "media_identity": canonical.get("content_sha256", expected_container),
        }

    benchmark_by_id = {item.sample_id: item for item in benchmark_items}
    if len(benchmark_by_id) != len(benchmark_items):
        raise BatchReceiptError("benchmark sample_id values must be unique")
    seen_samples: set[str] = set()
    used_resources: set[str] = set()
    sample_media: dict[str, dict[str, str]] = {}
    for raw in rows_raw:
        row = _mapping(raw, "media manifest row")
        sample_id = _identity(row.get("sample_id"), "media row sample_id")
        if sample_id in seen_samples or sample_id not in benchmark_by_id:
            raise BatchReceiptError("media rows must map uniquely to benchmark samples")
        seen_samples.add(sample_id)
        resource_ids = row.get("resource_ids")
        if not isinstance(resource_ids, list) or any(
            not isinstance(value, str) for value in resource_ids
        ):
            raise BatchReceiptError("media row resource_ids must be a string list")
        if len(resource_ids) != len(set(resource_ids)):
            raise BatchReceiptError("media row resource_ids must be unique")
        by_kind: dict[str, dict[str, Any]] = {}
        for resource_id in resource_ids:
            if resource_id not in resources:
                raise BatchReceiptError("media row references an unknown resource")
            resource = resources[resource_id]
            kind = resource["kind"]
            if kind in by_kind:
                raise BatchReceiptError("each media row needs exactly one resource per kind")
            by_kind[kind] = resource
            used_resources.add(resource_id)
        if set(by_kind) != set(required_kinds):
            kinds = " and ".join(sorted(required_kinds))
            raise BatchReceiptError(
                f"each benchmark row must have exactly one {kinds} resource"
            )
        item = benchmark_by_id[sample_id]
        for kind in required_kinds:
            benchmark_reference = getattr(item, kind)
            if benchmark_reference is None or dict(benchmark_reference) != by_kind[kind][
                "canonical"
            ]:
                raise BatchReceiptError(
                    "benchmark does not carry the canonical linked media resource"
                )
        sample_media[sample_id] = {
            kind: by_kind[kind]["media_identity"] for kind in required_kinds
        }
    if seen_samples != set(benchmark_by_id) or used_resources != set(resources):
        raise BatchReceiptError("media manifest contains missing or unbound rows/resources")
    body = {
        "schema_version": "motion-eval-media-verification/1",
        "manifest_sha256": sha256_file(source),
        "row_count": len(rows_raw),
        "resource_count": len(resources),
        "required_kinds": sorted(required_kinds),
        "sample_media_sha256": sha256_json(sample_media),
    }
    return {**body, "verification_sha256": sha256_json(body)}


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise BatchReceiptError("question/option text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _semantic_hash(question: object, options: object) -> str:
    if options is None:
        normalized_options: Any = None
    elif isinstance(options, Mapping):
        normalized_options = {
            str(key): _normalize_text(value)
            for key, value in sorted(options.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        normalized_options = [_normalize_text(value) for value in options]
    else:
        raise BatchReceiptError("options must be an object or list")
    return sha256_json(
        {"question": _normalize_text(question), "options": normalized_options}
    )


def _raw_media_digests(row: Mapping[str, Any], *, split: str) -> frozenset[str]:
    digests: set[str] = set()
    actual_reference = False
    for kind in ("video", "motion"):
        value = row.get(kind)
        if value is None:
            continue
        actual_reference = True
        if isinstance(value, str):
            media_path = _absolute_existing(value, f"{split} {kind}")
            if not media_path.is_file():
                raise BatchReceiptError(f"{split} {kind} must be a regular file")
            try:
                digests.add(sha256_file(media_path))
            except Exception as exc:
                raise BatchReceiptError(f"{split} {kind} cannot be hashed") from exc
        elif isinstance(value, Mapping):
            media_path = _absolute_existing(value.get("path"), f"{split} {kind}")
            expected = _sha(value.get("sha256"), f"{split} {kind} SHA-256")
            if sha256_file(media_path) != expected:
                raise BatchReceiptError(f"{split} {kind} actual reference hash mismatch")
            digests.add(_sha(value.get("content_sha256", expected), f"{split} media identity"))
        else:
            raise BatchReceiptError(f"{split} {kind} actual reference is invalid")
    if "media_sha256" in row and not actual_reference:
        raise BatchReceiptError(
            f"{split} media_sha256 is unbound to an actual media reference"
        )
    return frozenset(digests)


def _raw_feature(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    sample_id = _identity(row.get("sample_id"), f"{split} sample_id")
    group_id = _identity(row.get("group_id"), f"{split} group_id")
    semantic = _semantic_hash(row.get("question"), row.get("options"))
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "media_sha256": _raw_media_digests(row, split=split),
        "normalized_question_options": semantic,
        "near_duplicate": semantic,
    }


def _benchmark_features(items: Sequence[BenchmarkItem]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for item in items:
        media = set()
        for reference in (item.video, item.motion):
            if reference is not None:
                digest = reference.get("content_sha256", reference.get("sha256"))
                media.add(_sha(digest, "benchmark media identity"))
        semantic = _semantic_hash(item.question, item.options)
        result.append(
            {
                "sample_id": item.sample_id,
                "group_id": item.group_id,
                "media_sha256": frozenset(media),
                "normalized_question_options": semantic,
                "near_duplicate": semantic,
            }
        )
    return tuple(result)


def _feature_collision(left: Mapping[str, Any], right: Mapping[str, Any], name: str) -> bool:
    if name == "media_sha256":
        left_values = left.get(name)
        right_values = right.get(name)
        return bool(
            isinstance(left_values, (set, frozenset))
            and isinstance(right_values, (set, frozenset))
            and left_values.intersection(right_values)
        )
    return left.get(name) == right.get(name)


def _recompute_leakage_counts(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    benchmark_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {name: 0 for name in _CHECK_NAMES}
    for left_split, right_split in (
        (train_records, validation_records),
        (train_records, benchmark_records),
        (validation_records, benchmark_records),
    ):
        for left in left_split:
            for right in right_split:
                for name in _CHECK_NAMES:
                    if _feature_collision(left, right, name):
                        counts[name] += 1
    return counts


def _validate_leakage_audit(
    path: str | Path,
    *,
    bindings: Mapping[str, str],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    benchmark_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute all cross-split collision counts and verify the caller audit."""

    try:
        source = _absolute_existing(path, "leakage audit")
        audit = _mapping(load_json_strict(source), "leakage audit")
    except (StrictJsonError, OSError) as exc:
        raise BatchReceiptError("leakage audit is invalid") from exc
    expected_bindings = dict(bindings)
    for key, value in expected_bindings.items():
        _sha(value, f"leakage binding {key}")
    algorithm = {
        "version": LEAKAGE_ALGORITHM_VERSION,
        "sha256": LEAKAGE_ALGORITHM_SHA256,
    }
    checks = audit.get("checks")
    if (
        audit.get("schema_version") != LEAKAGE_AUDIT_SCHEMA_VERSION
        or audit.get("status") != "passed"
        or audit.get("algorithm") != algorithm
        or audit.get("bindings") != expected_bindings
        or not isinstance(checks, Mapping)
        or set(checks) != set(_CHECK_NAMES)
        or any(type(checks.get(name)) is not int or checks[name] < 0 for name in _CHECK_NAMES)
    ):
        raise BatchReceiptError("leakage audit contract is invalid")
    computation = {
        "algorithm": algorithm,
        "bindings": expected_bindings,
        "checks": dict(checks),
    }
    if audit.get("computed_sha256") != sha256_json(computation):
        raise BatchReceiptError("leakage audit computation hash is invalid")
    recomputed = _recompute_leakage_counts(
        train_records, validation_records, benchmark_records
    )
    if dict(checks) != recomputed:
        raise BatchReceiptError(
            "leakage audit counts differ from controller recomputation"
        )
    body = {
        "schema_version": "motion-eval-leakage-verification/1",
        "audit_sha256": sha256_file(source),
        "algorithm": algorithm,
        "bindings": expected_bindings,
        "checks": recomputed,
        "computed_sha256": audit.get("computed_sha256"),
    }
    return {**body, "verification_sha256": sha256_json(body)}


def _required_media_kinds(model_modalities: Mapping[str, str]) -> frozenset[str]:
    kinds: set[str] = set()
    for model_id, modality in model_modalities.items():
        _identity(model_id, "model_id")
        if modality not in {"V", "M", "VM"}:
            raise BatchReceiptError(f"model modality is invalid for {model_id}")
        if "V" in modality:
            kinds.add("video")
        if "M" in modality:
            kinds.add("motion")
    if not kinds:
        raise BatchReceiptError("at least one model modality is required")
    return frozenset(kinds)


def _validate_runtime_contract(contract: object, model_ids: Sequence[str]) -> dict[str, Any]:
    runtime = _plain_mapping(_mapping(contract, "runtime contract"))
    body = {key: value for key, value in runtime.items() if key != "runtime_contract_sha256"}
    if runtime.get("runtime_contract_sha256") != sha256_json(body):
        raise BatchReceiptError("runtime contract hash is invalid")
    interpreter = _mapping(runtime.get("interpreter"), "runtime interpreter")
    launcher = interpreter.get("launcher_path")
    target = interpreter.get("path")
    if (
        not isinstance(launcher, str)
        or not Path(launcher).is_absolute()
        or not isinstance(target, str)
        or not Path(target).is_absolute()
    ):
        raise BatchReceiptError("runtime interpreter paths are invalid")
    try:
        if Path(launcher).resolve(strict=True) != Path(target):
            raise BatchReceiptError("runtime interpreter launcher target changed")
    except (OSError, RuntimeError) as exc:
        raise BatchReceiptError("runtime interpreter launcher cannot be resolved") from exc
    _verify_hash_receipt(interpreter, "runtime interpreter")
    models = _mapping(runtime.get("models"), "runtime models")
    if set(models) != set(model_ids):
        raise BatchReceiptError("runtime model coverage differs from registry")
    for model_id, raw_roles in models.items():
        roles = _mapping(raw_roles, f"runtime roles for {model_id}")
        if set(roles) != {"finetune", "evaluation", "verifier"}:
            raise BatchReceiptError("runtime role coverage is incomplete")
        for role, raw in roles.items():
            frozen = _mapping(raw, f"runtime role {model_id}/{role}")
            template = _mapping(frozen.get("command_template"), "runtime command template")
            if frozen.get("command_template_sha256") != sha256_json(template):
                raise BatchReceiptError("runtime command template hash is invalid")
            argv = template.get("argv")
            if not isinstance(argv, list) or not argv or argv[0] != launcher:
                raise BatchReceiptError(
                    "runtime role uses a different interpreter launcher"
                )
            for state_key, receipt_key in (
                ("state", "runner_receipt"),
                ("backend_state", "backend_receipt"),
            ):
                state = frozen.get(state_key)
                receipt = frozen.get(receipt_key)
                if state == "present":
                    _verify_hash_receipt(receipt, f"runtime {receipt_key}")
                elif state != "missing" or receipt is not None:
                    raise BatchReceiptError("runtime runner/backend state is invalid")
    return runtime


def _registry_receipt(
    path: str | Path, model_ids: Sequence[str], label: str
) -> dict[str, Any]:
    source = _absolute_existing(path, label)
    if not source.is_file():
        raise BatchReceiptError(f"{label} must be a regular file")
    return {"path": str(source), "sha256": sha256_file(source), "model_ids": list(model_ids)}


def _verify_registry_receipt(value: object, label: str) -> None:
    receipt = _mapping(value, label)
    source = _absolute_existing(receipt.get("path"), label)
    if not source.is_file() or sha256_file(source) != receipt.get("sha256"):
        raise BatchReceiptError(f"{label} changed after batch freeze")
    if not isinstance(receipt.get("model_ids"), list):
        raise BatchReceiptError(f"{label} model_ids are invalid")


def create_batch_receipt(
    destination: str | Path,
    *,
    batch_id: str,
    inputs: Mapping[str, str | Path],
    registry_path: str | Path,
    pretrained_registry_path: str | Path,
    code_root: str | Path,
    runner_code_root: str | Path,
    model_ids: Sequence[str],
    model_modalities: Mapping[str, str],
    pretrained_artifact_specs: Mapping[str, Sequence[Mapping[str, Any]]],
    runtime_roots: Mapping[str, str | Path],
    runtime_contract: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Freeze one batch and atomically publish its immutable receipt."""

    validate_batch_id(batch_id)
    if set(inputs) != set(REQUIRED_INPUT_ROLES):
        raise BatchReceiptError(
            "inputs must contain exactly train, validation, benchmark, media_manifest, "
            "derivation_code, and leakage_audit"
        )
    if not isinstance(description, str):
        raise BatchReceiptError("description must be a string")
    ids = [_identity(value, "model_id") for value in model_ids]
    if len(ids) != len(set(ids)) or not ids:
        raise BatchReceiptError("model_ids must be non-empty and unique")
    if set(model_modalities) != set(ids) or set(pretrained_artifact_specs) != set(ids):
        raise BatchReceiptError("model metadata coverage differs from model_ids")

    resolved_inputs: dict[str, Path] = {}
    for role in sorted(REQUIRED_INPUT_ROLES):
        source = _absolute_existing(inputs[role], f"input {role}")
        if not source.is_file():
            raise BatchReceiptError(f"input {role} must be a regular file")
        for other_role, other in resolved_inputs.items():
            if source == other:
                raise BatchReceiptError(
                    f"input roles {other_role} and {role} must reference different files"
                )
        resolved_inputs[role] = source
    input_receipts = {
        role: _hash_receipt(source, f"input {role}")
        for role, source in resolved_inputs.items()
    }

    try:
        benchmark = load_benchmark(resolved_inputs["benchmark"])
        train_rows = load_jsonl_strict(resolved_inputs["train"])
        validation_rows = load_jsonl_strict(resolved_inputs["validation"])
    except StrictJsonError as exc:
        raise BatchReceiptError("batch train/validation/benchmark data is invalid") from exc
    train_features = tuple(_raw_feature(row, split="train") for row in train_rows)
    validation_features = tuple(
        _raw_feature(row, split="validation") for row in validation_rows
    )
    benchmark_features = _benchmark_features(benchmark)
    required_kinds = _required_media_kinds(model_modalities)
    media_verification = _validate_media_manifest(
        resolved_inputs["media_manifest"], benchmark, required_kinds=required_kinds
    )
    leakage_bindings = {
        "train_sha256": input_receipts["train"]["digest"],
        "validation_sha256": input_receipts["validation"]["digest"],
        "benchmark_sha256": input_receipts["benchmark"]["digest"],
        "media_manifest_sha256": input_receipts["media_manifest"]["digest"],
    }
    leakage_verification = _validate_leakage_audit(
        resolved_inputs["leakage_audit"],
        bindings=leakage_bindings,
        train_records=train_features,
        validation_records=validation_features,
        benchmark_records=benchmark_features,
    )

    roots = _mapping(runtime_roots, "runtime roots")
    if set(roots) != {"controller_root", "pretrained_root"}:
        raise BatchReceiptError("runtime roots must contain controller_root and pretrained_root")
    controller_root = _absolute_existing(roots["controller_root"], "controller root")
    pretrained_root = _absolute_existing(roots["pretrained_root"], "pretrained root")
    if not controller_root.is_dir() or not pretrained_root.is_dir():
        raise BatchReceiptError("runtime roots must be directories")
    runtime = _validate_runtime_contract(runtime_contract, ids)
    frozen_roots = {
        "controller_root": str(controller_root),
        "pretrained_root": str(pretrained_root),
    }
    assets, assets_sha = _freeze_pretrained_assets(
        pretrained_root, pretrained_artifact_specs
    )
    index = _build_trusted_pretrained_index(
        pretrained_root,
        assets,
        event_trust=_mapping(runtime.get("event_trust"), "event trust"),
        generated_at=datetime.now(timezone.utc),
    )
    frozen_config = dict(config or {})
    config_sha = sha256_json(frozen_config)
    environment_sha = sha256_json(
        {
            "schema_version": "motion-eval-environment-binding/1",
            "runtime_roots": frozen_roots,
            "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        }
    )
    body = {
        "schema_version": BATCH_RECEIPT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "description": description,
        "inputs": input_receipts,
        "registry": _registry_receipt(registry_path, ids, "model registry"),
        "pretrained_registry": _registry_receipt(
            pretrained_registry_path, ids, "pretrained registry"
        ),
        "code": _hash_receipt(code_root, "code root"),
        "runner_code": _hash_receipt(runner_code_root, "runner code root"),
        "runtime_roots": frozen_roots,
        "runtime_contract": runtime,
        "model_modalities": dict(model_modalities),
        "config": frozen_config,
        "config_sha256": config_sha,
        "environment_sha256": environment_sha,
        "pretrained_assets": assets,
        "pretrained_assets_sha256": assets_sha,
        "pretrained_assets_index": index,
        "media_verification": media_verification,
        "leakage_verification": leakage_verification,
    }
    receipt = {**body, "receipt_sha256": sha256_json(body)}
    atomic_write_json(destination, receipt, overwrite=False)
    return receipt


def load_and_validate_batch_receipt(
    path: str | Path, *, verify_pretrained_content: bool = False
) -> dict[str, Any]:
    """Load the clean schema and revalidate all non-pretrained input evidence."""

    try:
        source = _absolute_existing(path, "batch receipt")
        value = load_json_strict(source)
    except (StrictJsonError, OSError) as exc:
        raise BatchReceiptError("batch receipt is invalid") from exc
    receipt = _plain_mapping(_mapping(value, "batch receipt"))
    if receipt.get("schema_version") != BATCH_RECEIPT_SCHEMA_VERSION:
        raise BatchReceiptError(
            "unsupported batch receipt schema; unknown historical receipts are not compatible"
        )
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != sha256_json(body):
        raise BatchReceiptError("batch receipt hash is invalid")
    validate_batch_id(receipt.get("batch_id"))

    inputs = _mapping(receipt.get("inputs"), "receipt inputs")
    if set(inputs) != set(REQUIRED_INPUT_ROLES):
        raise BatchReceiptError("receipt input role coverage is invalid")
    input_paths = {
        role: _verify_hash_receipt(inputs[role], f"input {role}")
        for role in REQUIRED_INPUT_ROLES
    }
    if len(set(input_paths.values())) != len(input_paths):
        raise BatchReceiptError("receipt inputs must reference different files")
    _verify_registry_receipt(receipt.get("registry"), "model registry")
    _verify_registry_receipt(receipt.get("pretrained_registry"), "pretrained registry")
    model_ids = list(_mapping(receipt["registry"], "model registry").get("model_ids", []))
    if _mapping(receipt["pretrained_registry"], "pretrained registry").get(
        "model_ids"
    ) != model_ids:
        raise BatchReceiptError("registry model coverage differs")
    _verify_hash_receipt(receipt.get("code"), "code root")
    _verify_hash_receipt(receipt.get("runner_code"), "runner code root")
    runtime = _validate_runtime_contract(receipt.get("runtime_contract"), model_ids)
    roots = _mapping(receipt.get("runtime_roots"), "runtime roots")
    if set(roots) != {"controller_root", "pretrained_root"}:
        raise BatchReceiptError("runtime roots are invalid")
    controller_root = _absolute_existing(roots["controller_root"], "controller root")
    pretrained_root = _absolute_existing(roots["pretrained_root"], "pretrained root")
    if not controller_root.is_dir() or not pretrained_root.is_dir():
        raise BatchReceiptError("runtime roots must be directories")
    expected_environment = sha256_json(
        {
            "schema_version": "motion-eval-environment-binding/1",
            "runtime_roots": {
                "controller_root": str(controller_root),
                "pretrained_root": str(pretrained_root),
            },
            "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        }
    )
    if receipt.get("environment_sha256") != expected_environment:
        raise BatchReceiptError("environment binding hash is invalid")
    config = _mapping(receipt.get("config"), "batch config")
    if receipt.get("config_sha256") != sha256_json(config):
        raise BatchReceiptError("batch config hash is invalid")
    modalities = _mapping(receipt.get("model_modalities"), "model modalities")
    if set(modalities) != set(model_ids):
        raise BatchReceiptError("model modality coverage differs")

    assets = _mapping(receipt.get("pretrained_assets"), "pretrained assets")
    if receipt.get("pretrained_assets_sha256") != sha256_json(assets):
        raise BatchReceiptError("pretrained assets hash is invalid")
    _verify_trusted_pretrained_index(
        receipt.get("pretrained_assets_index"),
        assets=assets,
        pretrained_root=pretrained_root,
        event_trust=_mapping(runtime.get("event_trust"), "event trust"),
    )
    if verify_pretrained_content:
        _verify_pretrained_assets(assets)

    try:
        benchmark = load_benchmark(input_paths["benchmark"])
        train_rows = load_jsonl_strict(input_paths["train"])
        validation_rows = load_jsonl_strict(input_paths["validation"])
    except StrictJsonError as exc:
        raise BatchReceiptError("receipt data no longer satisfies its contract") from exc
    media = _validate_media_manifest(
        input_paths["media_manifest"],
        benchmark,
        required_kinds=_required_media_kinds(modalities),
    )
    if receipt.get("media_verification") != media:
        raise BatchReceiptError("media verification binding is invalid")
    leakage = _validate_leakage_audit(
        input_paths["leakage_audit"],
        bindings={
            "train_sha256": inputs["train"]["digest"],
            "validation_sha256": inputs["validation"]["digest"],
            "benchmark_sha256": inputs["benchmark"]["digest"],
            "media_manifest_sha256": inputs["media_manifest"]["digest"],
        },
        train_records=tuple(_raw_feature(row, split="train") for row in train_rows),
        validation_records=tuple(
            _raw_feature(row, split="validation") for row in validation_rows
        ),
        benchmark_records=_benchmark_features(benchmark),
    )
    if receipt.get("leakage_verification") != leakage:
        raise BatchReceiptError("leakage verification binding is invalid")
    return receipt


__all__ = [
    "BATCH_RECEIPT_SCHEMA_VERSION",
    "BatchReceiptError",
    "LEAKAGE_ALGORITHM_SHA256",
    "LEAKAGE_ALGORITHM_VERSION",
    "LEAKAGE_AUDIT_SCHEMA_VERSION",
    "REQUIRED_INPUT_ROLES",
    "create_batch_receipt",
    "load_and_validate_batch_receipt",
    "validate_batch_id",
]
