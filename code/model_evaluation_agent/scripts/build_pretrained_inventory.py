#!/usr/bin/env python3
"""Build a content-hashed inventory for every canonical pretrain input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".tar",
}
PROCESSOR_NAMES = {
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def sha256(path: Path, cache: dict[str, str]) -> str:
    resolved = str(path.resolve(strict=True))
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    cache[resolved] = value
    return value


def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_inventory(path: Path) -> dict[str, Any]:
    try:
        revision = git_value(path, "rev-parse", "HEAD")
        remote = git_value(path, "remote", "get-url", "origin")
        dirty_lines = subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {
            "status": "valid",
            "revision": revision,
            "remote": remote,
            "tracked_dirty_file_count": len(dirty_lines),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"status": "invalid", "error": repr(exc)}


def selected_files(directory: Path) -> list[Path]:
    found: list[Path] = []
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {".git", ".cache", "__pycache__"}]
        base = Path(root)
        for name in files:
            path = base / name
            suffixes = path.suffixes
            if (
                path.suffix.lower() in WEIGHT_SUFFIXES
                or name in PROCESSOR_NAMES
                or suffixes[-2:] == [".pth", ".tar"]
            ):
                found.append(path)
    return sorted(found, key=lambda item: str(item.relative_to(directory)))


def directory_inventory(path: Path, cache: dict[str, str]) -> dict[str, Any]:
    real = path.resolve(strict=True)
    files = selected_files(real)
    records = []
    weight_count = 0
    processor_count = 0
    total_bytes = 0
    tree_digest = hashlib.sha256()
    for index, file_path in enumerate(files, start=1):
        rel = str(file_path.relative_to(real))
        size = file_path.stat().st_size
        digest = sha256(file_path, cache)
        record = {"relative_path": rel, "size": size, "sha256": digest}
        records.append(record)
        total_bytes += size
        if file_path.suffix.lower() in WEIGHT_SUFFIXES or file_path.suffixes[-2:] == [
            ".pth",
            ".tar",
        ]:
            weight_count += 1
        if file_path.name in PROCESSOR_NAMES:
            processor_count += 1
        tree_digest.update(
            f"{rel}\0{size}\0{digest}\n".encode("utf-8", errors="strict")
        )
        print(
            f"HASH {index}/{len(files)} {real.name}/{rel} {size}",
            file=sys.stderr,
            flush=True,
        )
    status = "valid" if files and weight_count else "invalid"
    return {
        "status": status,
        "resolved_path": str(real),
        "selected_file_count": len(records),
        "weight_file_count": weight_count,
        "processor_file_count": processor_count,
        "selected_file_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": records,
    }


def runtime_directory_inventory(path: Path, cache: dict[str, str]) -> dict[str, Any]:
    """Hash an isolated Python runtime tree, including package source files."""
    real = path.resolve(strict=True)
    files: list[Path] = []
    for root, dirs, names in os.walk(real, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", ".cache", "__pycache__"}
        ]
        base = Path(root)
        for name in names:
            file_path = base / name
            if file_path.suffix not in {".pyc", ".pyo"}:
                files.append(file_path)
    files.sort(key=lambda item: str(item.relative_to(real)))

    records = []
    total_bytes = 0
    tree_digest = hashlib.sha256()
    for index, file_path in enumerate(files, start=1):
        rel = str(file_path.relative_to(real))
        size = file_path.stat().st_size
        digest = sha256(file_path, cache)
        records.append({"relative_path": rel, "size": size, "sha256": digest})
        total_bytes += size
        tree_digest.update(
            f"{rel}\0{size}\0{digest}\n".encode("utf-8", errors="strict")
        )
        print(
            f"HASH_RUNTIME {index}/{len(files)} {real.name}/{rel} {size}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "status": "valid" if files else "invalid",
        "resolved_path": str(real),
        "selected_file_count": len(records),
        "selected_file_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": records,
    }


def file_inventory(path: Path, cache: dict[str, str]) -> dict[str, Any]:
    real = path.resolve(strict=True)
    return {
        "status": "valid",
        "resolved_path": str(real),
        "size": real.stat().st_size,
        "sha256": sha256(real, cache),
    }


def inspect_artifact(
    pretrain_root: Path, artifact: dict[str, Any], cache: dict[str, str]
) -> dict[str, Any]:
    path = pretrain_root / artifact["path"]
    result = dict(artifact)
    result["absolute_path"] = str(path)
    result["is_symlink"] = path.is_symlink()
    if not path.exists():
        result.update({"status": "missing"})
        return result
    kind = artifact["kind"]
    if kind == "git_repo":
        detail = git_inventory(path.resolve(strict=True))
        detail["resolved_path"] = str(path.resolve(strict=True))
    elif kind == "python_package_tree":
        detail = runtime_directory_inventory(path, cache)
    elif path.is_dir():
        detail = directory_inventory(path, cache)
    else:
        detail = file_inventory(path, cache)
    expected_sha = artifact.get("sha256")
    if expected_sha and detail.get("sha256") != expected_sha:
        detail["status"] = "hash_mismatch"
        detail["expected_sha256"] = expected_sha
    result.update(detail)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    pretrain_root = Path(registry["remote_root"])
    cache: dict[str, str] = {}
    models = []
    started = time.time()
    for spec in registry["models"]:
        print(f"MODEL {spec['id']}", file=sys.stderr, flush=True)
        artifacts = [
            inspect_artifact(pretrain_root, item, cache)
            for item in spec["artifacts"]
        ]
        pretrain_ready = all(item["status"] == "valid" for item in artifacts)
        models.append(
            {
                "id": spec["id"],
                "source": spec["source"],
                "source_revision": spec["source_revision"],
                "license": spec["license"],
                "pretrained_weight_requirement": spec.get(
                    "pretrained_weight_requirement", "required"
                ),
                "pretrain_asset_ready": pretrain_ready,
                "artifacts": artifacts,
            }
        )
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "registry_path": str(args.registry),
        "pretrain_root": str(pretrain_root),
        "model_count": len(models),
        "pretrain_asset_ready_count": sum(
            int(model["pretrain_asset_ready"]) for model in models
        ),
        "all_pretrain_assets_ready": all(
            model["pretrain_asset_ready"] for model in models
        ),
        "unique_hashed_file_count": len(cache),
        "elapsed_seconds": time.time() - started,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: output[k] for k in (
        "model_count",
        "pretrain_asset_ready_count",
        "all_pretrain_assets_ready",
        "unique_hashed_file_count",
        "elapsed_seconds",
    )}, ensure_ascii=False, indent=2))
    return 0 if output["all_pretrain_assets_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
