#!/usr/bin/env python3
"""Fail closed unless all canonical pretrain assets are staged and smoked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("pretrained_registry.json"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("shared_assets/pretrained/pretrained_inventory.json"),
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        default=Path("shared_assets/pretrained/component_smoke.json"),
    )
    parser.add_argument(
        "--runner-smoke",
        type=Path,
        default=Path("server_audit/20260730_finetune_smoke.json"),
    )
    args = parser.parse_args()

    registry = read_json(args.registry)
    inventory = read_json(args.inventory)
    smoke = read_json(args.smoke)
    runner_smoke = read_json(args.runner_smoke)

    expected = [model["id"] for model in registry["models"]]
    actual = [model["id"] for model in inventory["models"]]
    errors: list[str] = []
    if len(expected) != 15:
        errors.append(f"registry model count is {len(expected)}, expected 15")
    if expected != actual:
        errors.append("inventory model IDs/order do not match pretrained registry")
    if inventory.get("model_count") != len(expected):
        errors.append("inventory model_count is inconsistent")
    if not inventory.get("all_pretrain_assets_ready"):
        errors.append("inventory does not mark all pretrain assets ready")
    not_ready = [
        model["id"]
        for model in inventory["models"]
        if not model.get("pretrain_asset_ready")
    ]
    if not_ready:
        errors.append("not ready: " + ", ".join(not_ready))
    if smoke.get("status") != "passed":
        errors.append("component smoke status is not passed")
    failed_smokes = [
        test.get("name", "<unnamed>")
        for test in smoke.get("tests", [])
        if test.get("status") != "passed"
    ]
    if failed_smokes:
        errors.append("failed component smokes: " + ", ".join(failed_smokes))
    expected_runner_smokes = {"videollama_lora", "motionllm_official"}
    runner_results = {
        result.get("model_id"): result
        for result in runner_smoke.get("results", [])
    }
    missing_runner_smokes = sorted(expected_runner_smokes - runner_results.keys())
    if missing_runner_smokes:
        errors.append(
            "missing finetune runner smokes: " + ", ".join(missing_runner_smokes)
        )
    failed_runner_smokes = sorted(
        model_id
        for model_id in expected_runner_smokes
        if model_id in runner_results
        and runner_results[model_id].get("status") != "passed"
    )
    if failed_runner_smokes:
        errors.append(
            "failed finetune runner smokes: " + ", ".join(failed_runner_smokes)
        )

    if errors:
        for error in errors:
            print(f"PRETRAIN_BLOCKED {error}")
        return 2
    print(
        "PRETRAIN_READY "
        f"models={len(expected)} "
        f"hashed_files={inventory.get('unique_hashed_file_count')} "
        f"component_smokes={len(smoke.get('tests', []))} "
        f"runner_smokes={len(expected_runner_smokes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
