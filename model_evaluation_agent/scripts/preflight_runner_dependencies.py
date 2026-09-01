#!/usr/bin/env python3
"""Read-only dependency/backend inventory for production catalog runners."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from runner_specs import MODEL_SPECS, backend_for, dependencies_for


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only production dependency and backend preflight."
    )
    parser.add_argument("--model-id", choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        help="Frozen pretrained root; required for MotionLLM's pinned runtime tree.",
    )
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    pretrained_root = (
        args.pretrained_root.resolve(strict=True)
        if args.pretrained_root is not None
        else None
    )
    selected = (args.model_id,) if args.model_id else tuple(MODEL_SPECS)
    results = []
    passed = True
    for model_id in selected:
        runtime_tree = None
        runtime_tree_ready = True
        if model_id == "motionllm_official":
            runtime_tree = (
                pretrained_root / "runtime_deps" / "motionllm"
                if pretrained_root is not None
                else None
            )
            runtime_tree_ready = runtime_tree is not None and runtime_tree.is_dir()
            if runtime_tree_ready:
                sys.path.insert(0, str(runtime_tree))
        missing = []
        for name in dependencies_for(model_id):
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                found = False
            if not found:
                missing.append(name)
        roles = {
            role: backend_for(model_id, role) is not None
            for role in ("finetune", "evaluation", "verifier")
        }
        cuda = None
        if args.require_cuda and roles["finetune"]:
            try:
                import torch

                cuda = bool(torch.cuda.is_available())
            except ImportError:
                cuda = False
        gpu_bound = None
        if args.require_cuda and roles["finetune"]:
            gpu_bound = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
        finetune_ready = (
            roles["finetune"]
            and runtime_tree_ready
            and not missing
            and cuda is not False
            and gpu_bound is not False
        )
        production_ready = finetune_ready and all(roles.values())
        passed = passed and production_ready
        results.append(
            {
                "model_id": model_id,
                "backend_roles": roles,
                "missing_dependencies": missing,
                "runtime_tree": str(runtime_tree) if runtime_tree is not None else None,
                "runtime_tree_ready": runtime_tree_ready,
                "cuda_available": cuda,
                "cuda_visible_devices_bound": gpu_bound,
                "finetune_preflight": "passed" if finetune_ready else "blocked",
                "production_preflight": "passed" if production_ready else "blocked",
            }
        )
    print(json.dumps({"schema_version": "1.0", "results": results}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
