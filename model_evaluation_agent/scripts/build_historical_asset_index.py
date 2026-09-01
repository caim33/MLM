#!/usr/bin/env python3
"""Index historical recovery weights without copying or promoting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
MLLM_ROOT = AGENT_ROOT.parents[1]
HISTORICAL_RUN = MLLM_ROOT / "codex_runs" / "finetune_goal_20260717"
SHARED_ROOT = AGENT_ROOT / "shared_assets"


ASSETS: dict[str, dict[str, Any]] = {
    "qwen36_27b_lora": {
        "artifact": HISTORICAL_RUN / "qwen_lora" / "qwen36_27b_video_lora_r16_z3tuned",
        "weights": ["adapter_model.safetensors"],
        "base": MLLM_ROOT / "codex_models" / "Qwen__Qwen3.6-27B",
    },
    "motionr1_vm_lora": {
        "artifact": HISTORICAL_RUN / "qwen_lora" / "motionr1_vm" / "checkpoint-482",
        "weights": ["adapter_model.safetensors"],
        "base": MLLM_ROOT / "codex_models" / "qwen3_vl_motion_checkpoint_0426",
    },
    "qwen3vl_8b_lora": {
        "artifact": HISTORICAL_RUN / "qwen_lora" / "qwen3vl8b_video_lora",
        "weights": ["adapter_model.safetensors"],
        "base": MLLM_ROOT / "codex_models" / "Qwen__Qwen3-VL-8B-Instruct",
    },
    "qwen3vl_4b_lora": {
        "artifact": HISTORICAL_RUN / "qwen_lora" / "qwen3vl4b_video",
        "weights": ["adapter_model.safetensors"],
        "base": MLLM_ROOT / "codex_models" / "Qwen__Qwen3-VL-4B-Instruct",
    },
    "qwen35_4b_lora": {
        "artifact": HISTORICAL_RUN / "qwen_lora" / "qwen35_4b_video_lora",
        "weights": ["adapter_model.safetensors"],
        "base": MLLM_ROOT / "codex_models" / "Qwen__Qwen3.5-4B",
    },
    "videollava_7b_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "videollava",
        "weights": ["adapter_model.bin", "non_lora_trainables.bin"],
        "base": None,
    },
    "videochatgpt_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "videochatgpt",
        "weights": ["adapter_model.bin"],
        "base": None,
    },
    "videochat2_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "videochat2",
        "weights": ["adapter_model.bin", "videochat2_lora_trainables.pth"],
        "base": None,
    },
    "videollama_trainables": {
        "artifact": HISTORICAL_RUN / "video_lora" / "videollama",
        "weights": ["videollama_trainables.pth"],
        "base": None,
    },
    "videollama_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "videollama_lora",
        "weights": ["adapter_model.bin"],
        "base": None,
        "expected_missing": True,
    },
    "mplug_owl_video_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "mplug_owl",
        "weights": ["adapter_model.bin"],
        "base": None,
    },
    "otter_video_lora": {
        "artifact": HISTORICAL_RUN / "video_lora" / "otter",
        "weights": ["adapter_model.bin"],
        "base": None,
    },
    "agcn_official": {
        "artifact": None,
        "weights": [],
        "base": None,
        "expected_missing": True,
    },
    "motionclip_official": {
        "artifact": None,
        "weights": [],
        "base": None,
        "expected_missing": True,
    },
    "motionllm_official": {
        "artifact": None,
        "weights": [],
        "base": None,
        "expected_missing": True,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve(strict=False) != target.resolve(strict=False):
            raise RuntimeError(f"refusing to replace different symlink: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    present_count = 0
    total_weight_bytes = 0
    for model_id, spec in ASSETS.items():
        artifact: Path | None = spec["artifact"]
        base: Path | None = spec["base"]
        row: dict[str, Any] = {
            "model_id": model_id,
            "classification": "historical_recovery_only",
            "usable_as_current_batch_finetune": False,
            "artifact_path": str(artifact) if artifact else "",
            "artifact_symlink": "",
            "base_path": str(base) if base else "",
            "base_symlink": "",
            "base_status": "not_yet_verified" if base is None else "missing",
            "weight_files": [],
            "status": "missing",
        }

        if artifact and artifact.exists():
            missing = [name for name in spec["weights"] if not (artifact / name).is_file()]
            if not missing:
                row["status"] = "present_historical_recovery_only"
                present_count += 1
                link = SHARED_ROOT / "historical_recovery" / model_id
                safe_symlink(link, artifact)
                row["artifact_symlink"] = str(link)
                for name in spec["weights"]:
                    weight = artifact / name
                    size = weight.stat().st_size
                    total_weight_bytes += size
                    row["weight_files"].append(
                        {
                            "path": str(weight),
                            "size_bytes": size,
                            "sha256": "" if args.skip_sha256 else sha256(weight),
                        }
                    )
            else:
                row["status"] = "missing_primary_weight"
                row["missing_weight_files"] = missing

        if base and base.exists():
            row["base_status"] = "present_path_only"
            base_link = SHARED_ROOT / "base_refs" / model_id
            safe_symlink(base_link, base)
            row["base_symlink"] = str(base_link)
        rows.append(row)

    inventory = {
        "schema_version": "1.0",
        "generated_at": now(),
        "historical_run": str(HISTORICAL_RUN),
        "policy": (
            "Recovery references only. These artifacts never satisfy a new batch's "
            "fresh-finetune gate and never enter a new main result directly."
        ),
        "registry_models": len(rows),
        "historical_artifacts_present": present_count,
        "historical_artifacts_missing": len(rows) - present_count,
        "selected_weight_bytes": total_weight_bytes,
        "models": rows,
    }
    write_json(SHARED_ROOT / "checkpoint_inventory.json", inventory)

    lines = [
        "# Historical checkpoint recovery index",
        "",
        "This directory contains symlinks and hashes only; no large weight was copied.",
        "Historical weights are recovery evidence and cannot satisfy a new batch finetune.",
        "",
        f"- Registry models: {len(rows)}",
        f"- Historical artifacts present: {present_count}",
        f"- Missing official/current artifacts: {len(rows) - present_count}",
        f"- Selected primary-weight bytes: {total_weight_bytes}",
        "",
        "| Model ID | Historical artifact | Base reference |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model_id']}` | `{row['status']}` | `{row['base_status']}` |"
        )
    (SHARED_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"INDEXED present={present_count} missing={len(rows) - present_count}")
    print(f"SELECTED_WEIGHT_BYTES {total_weight_bytes}")
    print(f"INVENTORY {SHARED_ROOT / 'checkpoint_inventory.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
