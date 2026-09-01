#!/usr/bin/env python3
"""Download public pretrained artifacts that were missing from the server.

This script deliberately downloads only public, upstream-published assets. It
does not read or persist any credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_once(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve(strict=True) == source:
            return
        raise RuntimeError(f"conflicting symlink: {target}")
    if target.exists():
        raise RuntimeError(f"target exists and is not a symlink: {target}")
    target.symlink_to(source)


def download_motionllm(root: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "EvanTHU/MotionLLM-7B"
    filenames = [
        "motionllm-ckpt/lora.pth",
        "motionllm-ckpt/linear.pth",
    ]
    target = root / "downloads" / "EvanTHU--MotionLLM-7B"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for filename in filenames:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision="main",
                local_dir=target,
            )
        )
        files.append(
            {
                "path": str(downloaded),
                "size": downloaded.stat().st_size,
                "sha256": sha256(downloaded),
            }
        )
    revision = HfApi().model_info(repo_id, revision="main").sha
    link_once(target / filenames[0], root / "by_model/motionllm_official/official_lora.pth")
    link_once(target / filenames[1], root / "by_model/motionllm_official/official_projector.pth")
    return {
        "source": f"https://huggingface.co/{repo_id}",
        "revision": revision,
        "license": "idea",
        "files": files,
    }


def ensure_gdown(root: Path) -> Path:
    module_root = root / "tools" / "gdown-5.2.0"
    marker = module_root / "gdown/__init__.py"
    if not marker.is_file():
        module_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target",
                str(module_root),
                "gdown==5.2.0",
            ],
            check=True,
        )
    return module_root


def download_motionclip(root: Path) -> dict:
    file_id = "1VTIN0kJd2-0NW1sKckKgXddwl4tFZVDp"
    url = f"https://drive.google.com/uc?id={file_id}"
    target_dir = root / "downloads" / "MotionCLIP"
    archive = target_dir / "paper-model.zip"
    extracted = target_dir / "paper-model"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        module_root = ensure_gdown(root)
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(module_root)
            if not env.get("PYTHONPATH")
            else str(module_root) + os.pathsep + env["PYTHONPATH"]
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                "--fuzzy",
                url,
                "--output",
                str(archive),
            ],
            env=env,
            check=True,
        )
    if not extracted.is_dir():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target_dir)
    candidates = sorted(target_dir.rglob("checkpoint_0100.pth.tar"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one MotionCLIP paper checkpoint, found {len(candidates)}"
        )
    checkpoint = candidates[0]
    link_once(checkpoint, root / "by_model/motionclip_official/pretrained.pth.tar")
    return {
        "source": (
            "https://drive.google.com/file/d/"
            "1VTIN0kJd2-0NW1sKckKgXddwl4tFZVDp/view"
        ),
        "revision": "upstream-paper-model-download",
        "license": "MIT code; model/data dependency terms also apply",
        "archive": {
            "path": str(archive),
            "size": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Explicit absolute destination root for downloaded pretrained assets.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "motionllm", "motionclip"],
        default="all",
    )
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be an absolute path")
    root = args.root.resolve()
    output: dict[str, object] = {"schema_version": "1.0"}
    if args.only in {"all", "motionllm"}:
        output["motionllm"] = download_motionllm(root)
    if args.only in {"all", "motionclip"}:
        output["motionclip"] = download_motionclip(root)
    manifest = root / "download_manifest.json"
    manifest.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
