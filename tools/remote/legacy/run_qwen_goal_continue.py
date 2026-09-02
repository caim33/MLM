#!/usr/bin/env python3
"""Continue the goal finetune pipeline after Motion-R1 checkpoint creation.

This runner intentionally disables trainer-time validation. The acceptance
benchmark is the fixed QA_500 set, so each model is trained once, saved, and
then evaluated with the separate QA_500 scoring scripts.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import run_qwen_goal_pipeline as base


ROOT = base.ROOT
RUN = base.RUN
QWEN_ROOT = base.QWEN_ROOT
TOOLS = base.TOOLS
PYTHON = base.PYTHON
TORCHRUN = base.TORCHRUN
MODEL_QWEN3VL4B = base.MODEL_QWEN3VL4B
MODEL_MOTION_BASE = base.MODEL_MOTION_BASE
VQVAE = base.VQVAE
SFT_DIR = base.SFT_DIR
EVAL_DIR = RUN / "data" / "eval"
LOG_DIR = RUN / "logs"
STATUS_PATH = RUN / "qwen_pipeline" / "continue_status.json"


def now() -> str:
    return base.now()


def load_status() -> Dict[str, Any]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"created_at": now(), "steps": {}, "run": str(RUN)}


def save_status(status: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = now()
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def set_step(status: Dict[str, Any], name: str, **kwargs: Any) -> None:
    step = status.setdefault("steps", {}).setdefault(name, {})
    step.update(kwargs)
    step["updated_at"] = now()
    save_status(status)


def run_cmd(status: Dict[str, Any], name: str, cmd: List[str], log_path: Path, *, cwd: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_step(status, name, state="running", command=cmd, log=str(log_path), cwd=str(cwd), started_at=now())
    with log_path.open("ab") as log:
        log.write((f"\n===== {now()} START {name} =====\n").encode())
        log.write((" ".join(cmd) + "\n").encode())
        log.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd), env=env)
            rc = proc.wait()
        except FileNotFoundError as exc:
            log.write((f"\nFileNotFoundError: {exc}\n").encode())
            rc = 127
        log.write((f"\n===== {now()} END {name} rc={rc} =====\n").encode())
    set_step(status, name, state="completed" if rc == 0 else "failed", returncode=rc, finished_at=now())
    return rc


def base_env() -> Dict[str, str]:
    env = base.base_env()
    env["CODEX_SFT_DIR"] = str(SFT_DIR)
    env["CODEX_ATTN_IMPLEMENTATION"] = "sdpa"
    return env


def latest_adapter_dir(output_dir: Path) -> Optional[Path]:
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / "adapter_model.safetensors").exists():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def eval_adapter_dir(source: Path, name: str) -> Path:
    """Create a PEFT-loadable adapter view with duplicate modules removed."""
    target = RUN / "qwen_lora" / "_eval_adapters" / name
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    modules = config.get("modules_to_save")
    if isinstance(modules, list) and "visual" in modules:
        config["modules_to_save"] = [module for module in modules if module != "visual.merger"]
    (target / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in source.iterdir():
        if item.name == "adapter_config.json" or not item.is_file():
            continue
        dest = target / item.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(item)
    return target


def train_args(output_dir: Path, dataset: str, eval_dataset: str, run_name: str, modules_to_save: str) -> List[str]:
    args = base.common_train_args(output_dir, dataset, eval_dataset, run_name, modules_to_save)
    for idx, value in enumerate(args):
        if value == "--eval_strategy" and idx + 1 < len(args):
            args[idx + 1] = "no"
    return args


def torchrun_cmd(model: Path, output_dir: Path, dataset: str, eval_dataset: str, run_name: str, modules_to_save: str, *, motion: bool) -> List[str]:
    launcher = [str(TORCHRUN)] if TORCHRUN.exists() else [str(PYTHON), "-m", "torch.distributed.run"]
    cmd = [
        *launcher,
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        f"--master_port={random.randint(21000, 29999)}",
        str(TOOLS / "run_lora_sft_codex.py"),
        "--model_name_or_path",
        str(model),
    ]
    if motion:
        cmd.extend(
            [
                "--motion_vqvae_path",
                str(VQVAE),
                "--motion_dataname",
                "t2m",
                "--motion_quantizer",
                "ema",
                "--vqvae_nb_code",
                "512",
                "--vqvae_code_dim",
                "512",
                "--vqvae_output_emb_width",
                "512",
                "--vqvae_down_t",
                "2",
                "--vqvae_stride_t",
                "2",
                "--vqvae_width",
                "512",
                "--vqvae_depth",
                "3",
                "--vqvae_dilation_growth_rate",
                "3",
                "--vqvae_activation",
                "relu",
                "--vqvae_norm",
                "none",
                "--motion_length_divisor",
                "4",
            ]
        )
    cmd.extend(train_args(output_dir, dataset, eval_dataset, run_name, modules_to_save))
    return cmd


def completed_output(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and path.with_suffix(".summary.json").exists()


def main() -> int:
    base.ensure_motion_alias()
    status = load_status()
    env = base_env()

    try:
        base.stop_keepalive_all()

        motion_out = RUN / "qwen_lora" / "motionr1_vm"
        motion_adapter = latest_adapter_dir(motion_out)
        if motion_adapter is None:
            set_step(status, "reuse_motionr1_vm_lora", state="failed", reason=f"no adapter under {motion_out}")
            return 2
        motion_eval_adapter = eval_adapter_dir(motion_adapter, "motionr1_vm")
        set_step(
            status,
            "reuse_motionr1_vm_lora",
            state="completed",
            adapter=str(motion_adapter),
            eval_adapter=str(motion_eval_adapter),
        )

        motion_eval = RUN / "qwen_eval" / "motionr1_vm_lora_QA500.jsonl"
        if completed_output(motion_eval):
            set_step(status, "eval_motionr1_vm_lora_QA500", state="completed", skipped=True, output=str(motion_eval))
        else:
            rc = run_cmd(
                status,
                "eval_motionr1_vm_lora_QA500",
                [
                    str(PYTHON),
                    str(TOOLS / "eval_motionr1_lora_mcq_score.py"),
                    "--model",
                    str(MODEL_MOTION_BASE),
                    "--adapter",
                    str(motion_eval_adapter),
                    "--processor",
                    str(motion_eval_adapter),
                    "--dataset",
                    str(EVAL_DIR / "QA_500_motion.abs.jsonl"),
                    "--output",
                    str(motion_eval),
                    "--branch",
                    "vm",
                    "--prompt_mode",
                    "short",
                    "--device",
                    "cuda:0",
                    "--attn_implementation",
                    "sdpa",
                ],
                LOG_DIR / "eval_motionr1_vm_lora_QA500.continue.log",
                cwd=QWEN_ROOT,
                env={**env, "CUDA_VISIBLE_DEVICES": "0"},
            )
            if rc != 0:
                return rc

        video_out = RUN / "qwen_lora" / "qwen3vl4b_video"
        video_adapter = latest_adapter_dir(video_out)
        if video_adapter is None:
            rc = run_cmd(
                status,
                "train_qwen3vl4b_video_lora",
                torchrun_cmd(
                    MODEL_QWEN3VL4B,
                    video_out,
                    "codex_motionx_qa_train_v",
                    "codex_motionx_qa_val_v",
                    "qwen3vl4b-video-lora-goal-20260717",
                    "visual",
                    motion=False,
                ),
                LOG_DIR / "train_qwen3vl4b_video_lora.continue.log",
                cwd=QWEN_ROOT,
                env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            )
            if rc != 0:
                return rc
            video_adapter = latest_adapter_dir(video_out)
        if video_adapter is None:
            set_step(status, "train_qwen3vl4b_video_lora", state="failed", reason=f"no adapter under {video_out}")
            return 3
        video_eval_adapter = eval_adapter_dir(video_adapter, "qwen3vl4b_video")
        set_step(
            status,
            "qwen3vl4b_video_adapter",
            state="completed",
            adapter=str(video_adapter),
            eval_adapter=str(video_eval_adapter),
        )

        qwen_eval = RUN / "qwen_eval" / "qwen3vl4b_video_lora_QA500.jsonl"
        if completed_output(qwen_eval):
            set_step(status, "eval_qwen3vl4b_video_lora_QA500", state="completed", skipped=True, output=str(qwen_eval))
        else:
            rc = run_cmd(
                status,
                "eval_qwen3vl4b_video_lora_QA500",
                [
                    str(PYTHON),
                    str(TOOLS / "eval_open_vlm_mcq_score.py"),
                    "--model",
                    str(MODEL_QWEN3VL4B),
                    "--adapter",
                    str(video_eval_adapter),
                    "--processor",
                    str(video_eval_adapter),
                    "--dataset",
                    str(EVAL_DIR / "QA_500_video.abs.jsonl"),
                    "--output",
                    str(qwen_eval),
                    "--branch",
                    "all",
                    "--input_mode",
                    "video",
                    "--model_class",
                    "qwen3vl",
                    "--device",
                    "cuda:0",
                    "--device_map",
                    "single",
                    "--attn_implementation",
                    "sdpa",
                    "--prompt_mode",
                    "short",
                    "--fps",
                    "1.0",
                    "--max_pixels",
                    "200704",
                    "--total_pixels",
                    "1605632",
                ],
                LOG_DIR / "eval_qwen3vl4b_video_lora_QA500.continue.log",
                cwd=QWEN_ROOT,
                env={**env, "CUDA_VISIBLE_DEVICES": "0"},
            )
            if rc != 0:
                return rc
        return 0
    finally:
        base.start_keepalive_all()


if __name__ == "__main__":
    raise SystemExit(main())
