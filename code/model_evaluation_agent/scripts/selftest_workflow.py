#!/usr/bin/env python3
"""Exercise every workflow gate with synthetic metadata and no model execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = AGENT_ROOT / "model_registry.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> int:
    registry = load_json(REGISTRY_PATH)
    with tempfile.TemporaryDirectory(prefix="unified_eval_selftest_") as temp:
        batches = Path(temp) / "batches"
        batch_id = "workflow_selftest"
        batch = batches / batch_id
        run(
            str(SCRIPT_DIR / "new_batch.py"),
            batch_id,
            "--description",
            "synthetic no-model workflow self-test",
            "--batches-root",
            str(batches),
        )

        batch_manifest = load_json(batch / "00_inputs" / "batch_manifest.json")
        batch_manifest.update(
            {
                "status": "frozen",
                "canonical_data": {"path": "/synthetic/data.jsonl", "sha256": "1" * 64},
                "train": {"manifest": "/synthetic/train.jsonl", "rows": 2, "sha256": "2" * 64},
                "validation": {"manifest": "/synthetic/val.jsonl", "rows": 1, "sha256": "3" * 64},
            }
        )
        write_json(batch / "00_inputs" / "batch_manifest.json", batch_manifest)
        benchmark = load_json(batch / "00_inputs" / "benchmark_manifest.json")
        benchmark.update(
            {
                "status": "frozen",
                "version": "synthetic-v1",
                "canonical_path": "/synthetic/benchmark.jsonl",
                "sha256": "4" * 64,
                "fixed_denominator": 2,
            }
        )
        write_json(batch / "00_inputs" / "benchmark_manifest.json", benchmark)
        leakage = load_json(batch / "00_inputs" / "leakage_audit.json")
        leakage.update({"status": "passed", "passed": True})
        write_json(batch / "00_inputs" / "leakage_audit.json", leakage)

        blocked_ids = {
            model["id"]
            for model in registry["models"]
            if str(model["current_asset_state"]).startswith("blocked")
        }
        complete_ids: list[str] = []
        for model in registry["models"]:
            model_id = model["id"]
            asset_path = batch / "01_asset_audit" / f"{model_id}.json"
            asset = load_json(asset_path)
            finetune_path = batch / "02_finetune" / model_id / "run_manifest.json"
            finetune = load_json(finetune_path)
            if model_id in blocked_ids:
                asset.update({"status": "blocked", "evidence": ["synthetic blocker"]})
                finetune.update(
                    {"status": "blocked", "blocker_reason": "synthetic self-test blocker"}
                )
            else:
                complete_ids.append(model_id)
                asset.update({"status": "passed", "modality_verified": True})
                finetune.update({"status": "finetune_complete"})
                finetune["official_source"].update(
                    {"revision": "synthetic-rev", "license": "synthetic-license", "code_sha256": "5" * 64}
                )
                finetune["base_model"].update(
                    {"identity": "synthetic-base", "sha256": "6" * 64}
                )
                finetune["data"].update(
                    {"train_sha256": "2" * 64, "validation_sha256": "3" * 64}
                )
                finetune["finetune"].update(
                    {"checkpoint_path": f"/synthetic/{model_id}", "checkpoint_sha256": model_id.encode().hex().ljust(64, "0")[:64]}
                )
            write_json(asset_path, asset)
            write_json(finetune_path, finetune)

        run(str(SCRIPT_DIR / "validate_batch.py"), "--stage", "finetune", str(batch))
        run(str(SCRIPT_DIR / "open_eval_stage.py"), str(batch))
        for model_id in complete_ids:
            for smoke, expected in (("smoke_1", 1), ("smoke_8", 8), ("smoke_32", 32)):
                path = batch / "03_eval" / model_id / smoke / "status.json"
                status = load_json(path)
                status.update({"status": "passed", "observed_sample_count": expected})
                write_json(path, status)
        run(str(SCRIPT_DIR / "open_full_eval.py"), str(batch), "--all")

        modes = {model["id"]: model["evaluation_mode"] for model in registry["models"]}
        for model_id in complete_ids:
            full = batch / "03_eval" / model_id / "full"
            rows = []
            for index, letter in enumerate(("A", "B")):
                row = {
                    "benchmark_id": f"synthetic-{index}",
                    "sample_id": f"synthetic-{index}",
                    "gold": letter,
                    "pred": letter,
                    "correct": True,
                    "parse_status": "valid",
                    "error_type": None,
                }
                if modes[model_id] == "discriminative_abcd_scores":
                    row["raw_scores"] = {"A": 1.0, "B": 0.0, "C": -1.0, "D": -2.0}
                else:
                    row["raw_output"] = f"<answer>{letter}</answer>"
                rows.append(row)
            (full / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            write_json(
                full / "summary.json",
                {"total": 2, "correct": 2, "accuracy": 1.0, "invalid": 0, "runtime_errors": 0},
            )
            eval_manifest = load_json(full / "run_manifest.json")
            eval_manifest["status"] = "complete"
            write_json(full / "run_manifest.json", eval_manifest)

        release = batch / "04_release"
        release.mkdir(parents=True)
        (release / "all_models_results.csv").write_text("model,status\n", encoding="utf-8")
        (release / "all_models_results.md").write_text("# Synthetic results\n", encoding="utf-8")
        (release / "blocked_models.md").write_text("# Synthetic blockers\n", encoding="utf-8")
        write_json(release / "evaluation_release_manifest.json", {"batch_id": batch_id})
        run(str(SCRIPT_DIR / "validate_batch.py"), "--stage", "release", str(batch))

    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
