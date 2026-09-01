#!/usr/bin/env python3
"""DEPRECATED weak legacy validator; use ``python -m motion_eval batch validate``.

It is retained only for historical fixture replay and is not an authorization
gate for new finetune, evaluation, or release work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR.parent / "model_registry.json"
FINETUNE_TERMINAL = {"finetune_complete", "blocked"}
RELEASE_FILES = (
    "all_models_results.csv",
    "all_models_results.md",
    "blocked_models.md",
    "evaluation_release_manifest.json",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_inputs(batch_root: Path, errors: list[str]) -> tuple[dict[str, Any], int | None]:
    inputs = batch_root / "00_inputs"
    payloads: dict[str, Any] = {}
    for name in ("batch_manifest.json", "benchmark_manifest.json", "leakage_audit.json"):
        path = inputs / name
        if not path.is_file():
            errors.append(f"missing input gate artifact: {path}")
            continue
        try:
            payloads[name] = load_json(path)
        except Exception as exc:
            errors.append(f"invalid input gate artifact {path}: {exc}")

    batch = payloads.get("batch_manifest.json", {})
    benchmark = payloads.get("benchmark_manifest.json", {})
    leakage = payloads.get("leakage_audit.json", {})

    if batch and batch.get("status") != "frozen":
        errors.append("batch_manifest status must be 'frozen'")
    if batch and batch.get("batch_id") != batch_root.name:
        errors.append("batch_manifest batch_id must match its directory name")
    if benchmark and benchmark.get("status") != "frozen":
        errors.append("benchmark_manifest status must be 'frozen'")
    if benchmark and not nonempty(benchmark.get("version")):
        errors.append("benchmark_manifest has no unique version")
    if benchmark and not nonempty(benchmark.get("sha256")):
        errors.append("benchmark_manifest has no SHA-256")
    denominator = benchmark.get("fixed_denominator")
    if benchmark and (not isinstance(denominator, int) or denominator <= 0):
        errors.append("benchmark_manifest fixed_denominator must be a positive integer")
        denominator = None
    if leakage and (
        leakage.get("status") != "passed" or leakage.get("passed") is not True
    ):
        errors.append("leakage_audit must have status='passed' and passed=true")
    return benchmark, denominator


def validate(batch_root: Path, stage: str = "release") -> list[str]:
    if stage not in {"finetune", "release"}:
        raise ValueError(f"unsupported stage: {stage}")

    errors: list[str] = []
    registry = load_json(REGISTRY_PATH)
    models = registry["models"]
    model_ids = [item["id"] for item in models]
    evaluation_modes = {item["id"]: item["evaluation_mode"] for item in models}
    benchmark, fixed_denominator = validate_inputs(batch_root, errors)
    finetune_states: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}

    for model_id in model_ids:
        asset_path = batch_root / "01_asset_audit" / f"{model_id}.json"
        asset_status = "missing"
        if not asset_path.is_file():
            errors.append(f"{model_id}: missing asset audit")
        else:
            try:
                asset_status = str(load_json(asset_path).get("status", ""))
            except Exception as exc:
                errors.append(f"{model_id}: invalid asset audit: {exc}")

        manifest_path = batch_root / "02_finetune" / model_id / "run_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{model_id}: missing finetune manifest")
            finetune_states[model_id] = "missing"
            continue
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            errors.append(f"{model_id}: invalid finetune manifest: {exc}")
            finetune_states[model_id] = "invalid"
            continue

        state = str(manifest.get("status", ""))
        finetune_states[model_id] = state
        if manifest.get("batch_id") != batch_root.name:
            errors.append(f"{model_id}: finetune batch_id mismatch")
        if manifest.get("model_id") != model_id:
            errors.append(f"{model_id}: finetune model_id mismatch")
        if manifest.get("stage") != "finetune":
            errors.append(f"{model_id}: manifest stage must be finetune")
        if state not in FINETUNE_TERMINAL:
            errors.append(
                f"{model_id}: finetune state must be finetune_complete or blocked, got {state!r}"
            )

        if state == "finetune_complete":
            if asset_status != "passed":
                errors.append(f"{model_id}: completed finetune requires passed asset audit")
            ckpt_hash = manifest.get("finetune", {}).get("checkpoint_sha256")
            if not nonempty(ckpt_hash):
                errors.append(f"{model_id}: completed finetune has no checkpoint SHA-256")
            else:
                checkpoint_hashes[model_id] = ckpt_hash
            required = (
                ("official source revision", manifest.get("official_source", {}).get("revision")),
                ("official source license", manifest.get("official_source", {}).get("license")),
                ("official source code hash", manifest.get("official_source", {}).get("code_sha256")),
                ("base identity", manifest.get("base_model", {}).get("identity")),
                ("base hash", manifest.get("base_model", {}).get("sha256")),
                ("train hash", manifest.get("data", {}).get("train_sha256")),
                ("validation hash", manifest.get("data", {}).get("validation_sha256")),
            )
            for label, value in required:
                if not nonempty(value):
                    errors.append(f"{model_id}: completed finetune missing {label}")

        if state == "blocked":
            if asset_status != "blocked":
                errors.append(f"{model_id}: blocked finetune requires blocked asset audit")
            if not nonempty(manifest.get("blocker_reason")):
                errors.append(f"{model_id}: blocked finetune has no blocker_reason")

    barrier_open = all(
        finetune_states.get(model_id) in FINETUNE_TERMINAL for model_id in model_ids
    )
    eval_root = batch_root / "03_eval"
    if eval_root.exists() and not barrier_open:
        errors.append("global finetune barrier is closed but 03_eval exists")

    if stage == "finetune":
        if eval_root.exists():
            errors.append("finetune gate expects 03_eval to remain unopened")
        return errors

    if not eval_root.is_dir():
        errors.append("release gate requires an opened 03_eval directory")

    for model_id in model_ids:
        model_eval = eval_root / model_id
        if finetune_states.get(model_id) == "blocked":
            if model_eval.exists():
                errors.append(f"{model_id}: blocked model must not have eval outputs")
            continue
        if not model_eval.exists():
            errors.append(f"{model_id}: completed finetune has no eval directory")
            continue
        if finetune_states.get(model_id) != "finetune_complete":
            errors.append(f"{model_id}: eval exists without completed finetune")
            continue

        full = model_eval / "full"
        if not full.exists():
            errors.append(f"{model_id}: missing full eval directory")
            continue
        for smoke in ("smoke_1", "smoke_8", "smoke_32"):
            smoke_status = model_eval / smoke / "status.json"
            if not smoke_status.is_file():
                errors.append(f"{model_id}: full eval exists without {smoke}/status.json")
            else:
                try:
                    if load_json(smoke_status).get("status") != "passed":
                        errors.append(f"{model_id}: {smoke} has not passed")
                except Exception as exc:
                    errors.append(f"{model_id}: invalid {smoke} status: {exc}")

        for name in ("predictions.jsonl", "summary.json", "run_manifest.json", "status.md"):
            if not (full / name).is_file():
                errors.append(f"{model_id}: missing full eval artifact {name}")

        eval_manifest_path = full / "run_manifest.json"
        if eval_manifest_path.is_file():
            try:
                manifest = load_json(eval_manifest_path)
                linked = manifest.get("evaluation", {}).get("finetune_checkpoint_sha256")
                if linked != checkpoint_hashes.get(model_id):
                    errors.append(
                        f"{model_id}: eval does not reference the current finetune checkpoint hash"
                    )
                adapter = manifest.get("evaluation", {}).get("parser_or_adapter")
                mode = evaluation_modes[model_id]
                if mode.startswith("generative") and adapter != "exact_full_answer_tag":
                    errors.append(f"{model_id}: non-strict main-table parser {adapter!r}")
                if (
                    mode == "discriminative_abcd_scores"
                    and adapter != "canonical_abcd_score_adapter"
                ):
                    errors.append(f"{model_id}: invalid discriminative adapter {adapter!r}")
                if manifest.get("status") != "complete":
                    errors.append(f"{model_id}: eval manifest status is not complete")
                if manifest.get("data", {}).get("benchmark_sha256") != benchmark.get("sha256"):
                    errors.append(f"{model_id}: eval benchmark hash mismatch")
                if manifest.get("evaluation", {}).get("fixed_denominator") != fixed_denominator:
                    errors.append(f"{model_id}: eval fixed denominator mismatch")
            except Exception as exc:
                errors.append(f"{model_id}: invalid eval manifest: {exc}")

        predictions_path = full / "predictions.jsonl"
        if predictions_path.is_file() and isinstance(fixed_denominator, int):
            try:
                rows = [
                    json.loads(line)
                    for line in predictions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) != fixed_denominator:
                    errors.append(
                        f"{model_id}: predictions rows {len(rows)} != denominator {fixed_denominator}"
                    )
                sample_ids = [row.get("benchmark_id") or row.get("sample_id") for row in rows]
                if any(not nonempty(sample_id) for sample_id in sample_ids):
                    errors.append(f"{model_id}: prediction row missing benchmark/sample ID")
                if len(set(sample_ids)) != len(sample_ids):
                    errors.append(f"{model_id}: duplicate prediction sample IDs")
                if evaluation_modes[model_id] == "discriminative_abcd_scores":
                    for index, row in enumerate(rows):
                        scores = row.get("raw_scores")
                        if not isinstance(scores, dict) or set(scores) != {"A", "B", "C", "D"}:
                            errors.append(
                                f"{model_id}: row {index} lacks four canonical raw_scores"
                            )
                            break
            except Exception as exc:
                errors.append(f"{model_id}: invalid predictions.jsonl: {exc}")

        summary_path = full / "summary.json"
        if summary_path.is_file() and isinstance(fixed_denominator, int):
            try:
                summary = load_json(summary_path)
                if summary.get("total") != fixed_denominator:
                    errors.append(f"{model_id}: summary total mismatch")
                correct = summary.get("correct")
                if not isinstance(correct, int) or not 0 <= correct <= fixed_denominator:
                    errors.append(f"{model_id}: summary correct is invalid")
                elif abs(float(summary.get("accuracy", -1)) - correct / fixed_denominator) > 1e-12:
                    errors.append(f"{model_id}: summary accuracy is inconsistent")
                for key in ("invalid", "runtime_errors"):
                    if not isinstance(summary.get(key), int):
                        errors.append(f"{model_id}: summary missing integer {key}")
            except Exception as exc:
                errors.append(f"{model_id}: invalid summary.json: {exc}")

    release_root = batch_root / "04_release"
    for name in RELEASE_FILES:
        if not (release_root / name).is_file():
            errors.append(f"missing release artifact: {release_root / name}")
    return errors


def main() -> int:
    print(
        "DEPRECATED: weak legacy validator only; use `python -m motion_eval batch validate`",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("finetune", "release"), default="release")
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args()
    errors = validate(args.batch_root.resolve(), stage=args.stage)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALID stage={args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
