#!/usr/bin/env python3
"""DEPRECATED legacy metadata skeleton; use ``python -m motion_eval batch create``.

This script is retained only to read/replay historical controller fixtures. It
does not create hash-bound receipts and must not be used for a new main-table
batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = AGENT_ROOT / "model_registry.json"
MANIFEST_TEMPLATE = AGENT_ROOT / "templates" / "run_manifest.template.json"
BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    print(
        "DEPRECATED: legacy skeleton only; use `python -m motion_eval batch create` for new batches",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id")
    parser.add_argument("--description", default="")
    parser.add_argument("--batches-root", type=Path, default=AGENT_ROOT / "batches")
    args = parser.parse_args()

    if not BATCH_ID_RE.fullmatch(args.batch_id):
        parser.error("batch_id must use only letters, digits, dot, underscore, and hyphen")

    batches_root = args.batches_root.resolve()
    batch_root = batches_root / args.batch_id
    if batch_root.exists():
        print(f"REFUSED: batch already exists: {batch_root}", file=sys.stderr)
        return 2

    registry = load_json(REGISTRY_PATH)
    manifest_template = load_json(MANIFEST_TEMPLATE)
    created_at = now()
    batch_root.mkdir(parents=True)

    write_json(
        batch_root / "batch_state.json",
        {
            "schema_version": "1.0",
            "batch_id": args.batch_id,
            "phase": "input_freeze",
            "created_at": created_at,
            "description": args.description,
            "eval_stage_open": False,
        },
    )
    write_json(
        batch_root / "00_inputs" / "batch_manifest.json",
        {
            "schema_version": "1.0",
            "batch_id": args.batch_id,
            "status": "pending_freeze",
            "description": args.description,
            "created_at": created_at,
            "canonical_data": {"path": "", "sha256": ""},
            "train": {"manifest": "", "rows": None, "sha256": ""},
            "validation": {"manifest": "", "rows": None, "sha256": ""},
            "media_manifest": {"path": "", "sha256": ""},
            "derivation_code_sha256": "",
            "option_permutation_seed": None,
        },
    )
    write_json(
        batch_root / "00_inputs" / "benchmark_manifest.json",
        {
            "schema_version": "1.0",
            "batch_id": args.batch_id,
            "status": "pending_freeze",
            "version": "",
            "canonical_path": "",
            "sha256": "",
            "fixed_denominator": None,
            "derived_views": {},
            "option_permutation_sha256": "",
            "media_manifest_sha256": "",
            "prompt_sha256": "",
            "evaluator_code_sha256": "",
        },
    )
    write_json(
        batch_root / "00_inputs" / "leakage_audit.json",
        {
            "schema_version": "1.0",
            "batch_id": args.batch_id,
            "status": "pending",
            "passed": False,
            "checks": {
                "sample_id": None,
                "group_id": None,
                "media_content_hash": None,
                "normalized_question_options": None,
                "near_duplicate": None,
            },
            "evidence": [],
        },
    )

    status_template = (
        "# Model status\n\n"
        "- Batch ID: {batch_id}\n"
        "- Model ID: {model_id}\n"
        "- Stage: finetune\n"
        "- Status: pending\n"
        "- Current-batch checkpoint SHA-256:\n"
        "- Evidence:\n"
        "- Blocker or failure reason:\n"
        "- Next safe action: complete asset audit and fresh finetune\n"
    )
    for model in registry["models"]:
        model_id = model["id"]
        write_json(
            batch_root / "01_asset_audit" / f"{model_id}.json",
            {
                "schema_version": "1.0",
                "batch_id": args.batch_id,
                "model_id": model_id,
                "status": "pending",
                "official_repo": "",
                "revision": "",
                "license": "",
                "base_weights": "",
                "processor": "",
                "training_entrypoint": "",
                "modality_verified": False,
                "evidence": [],
            },
        )
        manifest = copy.deepcopy(manifest_template)
        manifest.update(
            {
                "batch_id": args.batch_id,
                "stage": "finetune",
                "model_id": model_id,
                "status": "pending",
                "modality": model["main_modality"],
                "started_at": "",
                "finished_at": "",
            }
        )
        model_root = batch_root / "02_finetune" / model_id
        write_json(model_root / "run_manifest.json", manifest)
        (model_root / "status.md").write_text(
            status_template.format(batch_id=args.batch_id, model_id=model_id),
            encoding="utf-8",
        )

    print(f"CREATED {batch_root}")
    print(f"MODELS {len(registry['models'])}")
    print("PHASE input_freeze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
