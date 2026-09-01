#!/usr/bin/env python3
"""DEPRECATED legacy gate; use ``python -m motion_eval gate open-eval``.

This file accepts manually edited manifests and therefore cannot authorize a
new main-table batch. It remains only for historical workflow fixture replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_batch import REGISTRY_PATH, load_json, validate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    print(
        "DEPRECATED: legacy gate only; use `python -m motion_eval gate open-eval`",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    eval_root = batch_root / "03_eval"
    gate_path = eval_root / "EVAL_STAGE_OPEN.json"
    if gate_path.is_file():
        print(f"ALREADY_OPEN {eval_root}")
        return 0
    if eval_root.exists():
        print(f"REFUSED: unmanaged eval directory exists: {eval_root}", file=sys.stderr)
        return 2

    errors = validate(batch_root, stage="finetune")
    if errors:
        print("REFUSED: global finetune gate is closed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    registry = load_json(REGISTRY_PATH)
    completed: list[str] = []
    blocked: list[str] = []
    manifest_hashes: dict[str, str] = {}
    for model in registry["models"]:
        model_id = model["id"]
        finetune_manifest = batch_root / "02_finetune" / model_id / "run_manifest.json"
        payload = load_json(finetune_manifest)
        if payload["status"] == "blocked":
            blocked.append(model_id)
            continue
        completed.append(model_id)
        manifest_hashes[model_id] = sha256(finetune_manifest)
        model_eval = eval_root / model_id
        write_json(
            model_eval / "eval_plan.json",
            {
                "schema_version": "1.0",
                "batch_id": batch_root.name,
                "model_id": model_id,
                "evaluation_mode": model["evaluation_mode"],
                "finetune_checkpoint_sha256": payload["finetune"]["checkpoint_sha256"],
                "finetune_manifest_sha256": manifest_hashes[model_id],
                "status": "smoke_pending",
            },
        )
        for smoke, expected in (("smoke_1", 1), ("smoke_8", 8), ("smoke_32", 32)):
            write_json(
                model_eval / smoke / "status.json",
                {
                    "schema_version": "1.0",
                    "batch_id": batch_root.name,
                    "model_id": model_id,
                    "stage": smoke,
                    "status": "pending",
                    "expected_sample_count": expected,
                    "observed_sample_count": None,
                    "evidence": [],
                },
            )

    write_json(
        gate_path,
        {
            "schema_version": "1.0",
            "batch_id": batch_root.name,
            "opened_at": now(),
            "completed_finetune_models": completed,
            "blocked_models": blocked,
            "finetune_manifest_sha256": manifest_hashes,
        },
    )
    state_path = batch_root / "batch_state.json"
    state = load_json(state_path)
    state.update({"phase": "smoke_eval", "eval_stage_open": True, "updated_at": now()})
    write_json(state_path, state)
    print(f"OPENED {eval_root}")
    print(f"EVALUABLE {len(completed)}")
    print(f"BLOCKED {len(blocked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
