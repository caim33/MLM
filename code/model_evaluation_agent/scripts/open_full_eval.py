#!/usr/bin/env python3
"""DEPRECATED legacy gate; use ``python -m motion_eval gate open-full``.

This file trusts manually edited status files and cannot authorize a new
main-table full evaluation. It remains for historical fixture replay only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = AGENT_ROOT / "model_registry.json"
MANIFEST_TEMPLATE = AGENT_ROOT / "templates" / "run_manifest.template.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    print(
        "DEPRECATED: legacy gate only; use `python -m motion_eval gate open-full`",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--model-id", action="append")
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    eval_root = batch_root / "03_eval"
    if not (eval_root / "EVAL_STAGE_OPEN.json").is_file():
        print("REFUSED: eval stage is not open", file=sys.stderr)
        return 2

    registry = load_json(REGISTRY_PATH)
    models = {item["id"]: item for item in registry["models"]}
    completed = []
    for model_id in models:
        manifest = load_json(batch_root / "02_finetune" / model_id / "run_manifest.json")
        if manifest.get("status") == "finetune_complete":
            completed.append(model_id)
    targets = completed if args.all else args.model_id
    unknown = sorted(set(targets) - set(models))
    if unknown:
        print(f"REFUSED: unknown model IDs: {unknown}", file=sys.stderr)
        return 2

    errors: list[str] = []
    for model_id in targets:
        if model_id not in completed:
            errors.append(f"{model_id}: finetune is not complete")
            continue
        for smoke, expected in (("smoke_1", 1), ("smoke_8", 8), ("smoke_32", 32)):
            path = eval_root / model_id / smoke / "status.json"
            try:
                payload = load_json(path)
                if payload.get("status") != "passed":
                    errors.append(f"{model_id}: {smoke} is not passed")
                if payload.get("observed_sample_count") != expected:
                    errors.append(f"{model_id}: {smoke} sample count is not {expected}")
            except Exception as exc:
                errors.append(f"{model_id}: invalid {smoke} status: {exc}")
    if errors:
        print("REFUSED: smoke gate is closed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    benchmark = load_json(batch_root / "00_inputs" / "benchmark_manifest.json")
    leakage = batch_root / "00_inputs" / "leakage_audit.json"
    template = load_json(MANIFEST_TEMPLATE)
    created = 0
    for model_id in targets:
        full_root = eval_root / model_id / "full"
        if full_root.exists():
            print(f"SKIP_EXISTS {full_root}")
            continue
        finetune = load_json(
            batch_root / "02_finetune" / model_id / "run_manifest.json"
        )
        model = models[model_id]
        manifest = copy.deepcopy(template)
        manifest.update(
            {
                "batch_id": batch_root.name,
                "stage": "eval",
                "model_id": model_id,
                "status": "pending",
                "modality": model["main_modality"],
            }
        )
        manifest["official_source"] = finetune["official_source"]
        manifest["base_model"] = finetune["base_model"]
        manifest["data"].update(
            {
                "benchmark_version": benchmark["version"],
                "benchmark_sha256": benchmark["sha256"],
                "leakage_audit": str(leakage),
            }
        )
        manifest["evaluation"].update(
            {
                "finetune_checkpoint_sha256": finetune["finetune"]["checkpoint_sha256"],
                "parser_or_adapter": (
                    "canonical_abcd_score_adapter"
                    if model["evaluation_mode"] == "discriminative_abcd_scores"
                    else "exact_full_answer_tag"
                ),
                "fixed_denominator": benchmark["fixed_denominator"],
            }
        )
        full_root.mkdir(parents=True)
        write_json(full_root / "run_manifest.json", manifest)
        write_json(
            full_root / "FULL_EVAL_OPEN.json",
            {
                "schema_version": "1.0",
                "batch_id": batch_root.name,
                "model_id": model_id,
                "opened_at": now(),
                "finetune_checkpoint_sha256": finetune["finetune"]["checkpoint_sha256"],
            },
        )
        (full_root / "status.md").write_text(
            f"# Model status\n\n- Batch ID: {batch_root.name}\n"
            f"- Model ID: {model_id}\n- Stage: eval\n- Status: pending\n",
            encoding="utf-8",
        )
        created += 1
    print(f"FULL_EVAL_OPENED {created}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
