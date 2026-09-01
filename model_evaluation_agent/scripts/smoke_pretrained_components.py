#!/usr/bin/env python3
"""Offline component-load smoke tests for canonical pretrained assets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


@contextmanager
def prepend_sys_path(path: Path):
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


def run_test(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    try:
        detail = fn()
        return {
            "name": name,
            "status": "passed",
            "elapsed_seconds": time.time() - started,
            "detail": detail,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def qwen_processors(root: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoProcessor

    ids = [
        "qwen36_27b_lora",
        "motionr1_vm_lora",
        "qwen3vl_8b_lora",
        "qwen3vl_4b_lora",
        "qwen35_4b_lora",
    ]
    result = {}
    for model_id in ids:
        path = root / "by_model" / model_id / "base"
        config = AutoConfig.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        result[model_id] = {
            "config_class": type(config).__name__,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        }
    return result


def motionclip_encoder(root: Path) -> dict[str, Any]:
    source = (root / "by_model/motionclip_official/source").resolve(strict=True)
    checkpoint = (
        root / "by_model/motionclip_official/pretrained.pth.tar"
    ).resolve(strict=True)
    with prepend_sys_path(source):
        from src.models.architectures.transformer import Encoder_TRANSFORMER

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        encoder = Encoder_TRANSFORMER(
            modeltype="cvae",
            njoints=25,
            nfeats=6,
            num_frames=60,
            num_classes=1,
            translation=True,
            pose_rep="rot6d",
            glob=True,
            glob_rot=[math.pi, 0.0, 0.0],
            latent_dim=512,
            num_layers=8,
            activation="gelu",
        )
        incompatible = encoder.load_state_dict(encoder_state, strict=True)
        encoder.eval()
        with torch.no_grad():
            output = encoder(
                {
                    "x": torch.zeros(1, 25, 6, 4),
                    "y": torch.zeros(1, dtype=torch.long),
                    "mask": torch.ones(1, 4, dtype=torch.bool),
                }
            )["mu"]
        return {
            "checkpoint_key_count": len(state),
            "encoder_key_count": len(encoder_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "output_shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all()),
        }


def agcn_official(root: Path) -> dict[str, Any]:
    source = (root / "by_model/agcn_official/source").resolve(strict=True)
    with prepend_sys_path(source):
        from model.agcn import Model

        model = Model(
            num_class=4,
            num_point=25,
            num_person=2,
            graph="graph.ntu_rgb_d.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 8, 25, 2, device=device))
        return {
            "initialization": "official_random_initialization",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all()),
        }


def motionllm_weights(root: Path) -> dict[str, Any]:
    source = (root / "by_model/motionllm_official/source").resolve(strict=True)
    asset_dir = root / "by_model/motionllm_official"
    runtime_deps = root / "runtime_deps/motionllm"
    with prepend_sys_path(runtime_deps), prepend_sys_path(source):
        from lit_gpt.lora import GPT, Config
        from lit_gpt.utils import lazy_load
        from lit_llama import Tokenizer
        from lit_llama.lora import lora

        with torch.device("meta"), lora(
            r=64, alpha=16, dropout=0.05, enabled=True
        ):
            config = Config.from_name(
                name="vicuna-7b-v1.5",
                r=64,
                alpha=16,
                dropout=0.05,
                to_query=True,
                to_key=False,
                to_value=True,
                to_projection=False,
                to_mlp=False,
                to_head=False,
            )
            model = GPT(config)

        base = lazy_load(asset_dir / "vicuna_lit/lit_model.pth")
        adapter = lazy_load(asset_dir / "official_lora.pth")
        combined = {**base, **adapter}
        incompatible = model.load_state_dict(combined, strict=True)

        projector_state = torch.load(
            asset_dir / "official_projector.pth",
            map_location="cpu",
            weights_only=False,
        )
        projector = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 4096),
        )
        projector_incompatible = projector.load_state_dict(
            projector_state, strict=True
        )
        tokenizer = Tokenizer(asset_dir / "vicuna_lit/tokenizer.model")

        video_config = json.loads(
            (asset_dir / "video_tower/config.json").read_text(encoding="utf-8")
        )
        return {
            "base_key_count": len(base),
            "official_lora_key_count": len(adapter),
            "combined_key_count": len(combined),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "projector_missing_keys": list(projector_incompatible.missing_keys),
            "projector_unexpected_keys": list(
                projector_incompatible.unexpected_keys
            ),
            "tokenizer_eos_id": tokenizer.eos_id,
            "video_model_type": video_config.get("model_type"),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    tests = [
        run_test("qwen_config_processor_offline", lambda: qwen_processors(root)),
        run_test("agcn_official_init_forward", lambda: agcn_official(root)),
        run_test(
            "motionclip_pretrained_encoder_strict_load",
            lambda: motionclip_encoder(root),
        ),
        run_test(
            "motionllm_base_lora_projector_strict_load",
            lambda: motionllm_weights(root),
        ),
    ]
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "host": os.uname().nodename,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pretrain_root": str(root),
        "status": "passed"
        if all(test["status"] == "passed" for test in tests)
        else "failed",
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
