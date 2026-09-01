#!/usr/bin/env python3
"""Download a model snapshot from ModelScope first, then Hugging Face."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id).strip("_")


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def link_alias(alias: Path, target: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        return
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        marker = alias.with_suffix(".path.txt")
        marker.write_text(str(target), encoding="utf-8")


def download_modelscope(model_id: str, cache_dir: Path, revision: Optional[str]) -> Path:
    from modelscope import snapshot_download

    kwargs: Dict[str, Any] = {"model_id": model_id, "cache_dir": str(cache_dir)}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs)).resolve()


def download_huggingface(model_id: str, cache_dir: Path, revision: Optional[str]) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / safe_name(model_id)
    kwargs: Dict[str, Any] = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs)).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model id, e.g. Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--cache_dir", required=True, help="Root cache directory")
    p.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto")
    p.add_argument("--revision", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    alias = cache_dir / safe_name(args.model)
    manifest = cache_dir / f"{safe_name(args.model)}.download.json"

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    errors: Dict[str, str] = {}
    sources = ["modelscope", "hf"] if args.source == "auto" else [args.source]

    for source in sources:
        try:
            if source == "modelscope":
                path = download_modelscope(args.model, cache_dir, args.revision)
            else:
                path = download_huggingface(args.model, cache_dir, args.revision)
            link_alias(alias, path)
            write_manifest(
                manifest,
                {
                    "model": args.model,
                    "source": source,
                    "path": str(path),
                    "alias": str(alias),
                    "revision": args.revision,
                    "started_at": started,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "ok",
                },
            )
            print(json.dumps({"status": "ok", "source": source, "path": str(path), "alias": str(alias)}, ensure_ascii=False))
            return 0
        except Exception as exc:  # keep trying fallback source
            errors[source] = f"{type(exc).__name__}: {exc}"
            print(f"[download] {source} failed for {args.model}: {errors[source]}", file=sys.stderr, flush=True)

    write_manifest(
        manifest,
        {
            "model": args.model,
            "revision": args.revision,
            "started_at": started,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "failed",
            "errors": errors,
        },
    )
    print(json.dumps({"status": "failed", "model": args.model, "errors": errors}, ensure_ascii=False), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
