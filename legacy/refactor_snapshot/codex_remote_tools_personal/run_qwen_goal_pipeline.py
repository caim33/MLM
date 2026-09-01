#!/usr/bin/env python3
"""Run Qwen/Motion-R1 finetune and QA_500 eval for the user's goal."""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
QWEN_ROOT = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune")
RUN = ROOT / "codex_runs" / "finetune_goal_20260717"
TOOLS = ROOT / "codex_tools"
PYTHON = ROOT / "codex_envs" / "mllm" / "bin" / "python"
TORCHRUN = ROOT / "codex_envs" / "mllm" / "bin" / "torchrun"
STATUS_PATH = RUN / "qwen_pipeline" / "status.json"
LOG_DIR = RUN / "logs"
SFT_DIR = RUN / "data" / "sft"
EVAL_DIR = RUN / "data" / "eval"
MODEL_QWEN3VL4B = ROOT / "codex_models" / "qwen3vl4b"
MODEL_MOTION_BASE = ROOT / "codex_models" / "qwen3_vl_motion_checkpoint_0426"
MOTION_BASE_TARGET = Path("/wangbenyou-sulongjie/Motion-r1/model/checkpoint_0426")
VQVAE = Path("/wangbenyou-sulongjie/Motion-r1/model/pretrained/VQVAE/net_best_fid.pth")
KEEPALIVE = TOOLS / "gpu_keepalive.py"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_status() -> Dict[str, Any]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"created_at": now(), "steps": {}, "run": str(RUN)}


def write_status(status: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = now()
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def set_step(status: Dict[str, Any], name: str, **payload: Any) -> None:
    step = status.setdefault("steps", {}).setdefault(name, {})
    step.update(payload)
    step["updated_at"] = now()
    write_status(status)


def ensure_motion_alias() -> None:
    MODEL_MOTION_BASE.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_MOTION_BASE.exists() or MODEL_MOTION_BASE.is_symlink():
        return
    MODEL_MOTION_BASE.symlink_to(MOTION_BASE_TARGET, target_is_directory=True)


def base_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{QWEN_ROOT}:{QWEN_ROOT / 'qwenvl' / 'train'}:{TOOLS}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DISABLED"] = "true"
    env["CODEX_SFT_DIR"] = str(SFT_DIR)
    env["CODEX_ATTN_IMPLEMENTATION"] = "sdpa"
    return env


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
    state = "completed" if rc == 0 else "failed"
    set_step(status, name, state=state, returncode=rc, finished_at=now())
    return rc


def common_train_args(output_dir: Path, dataset: str, eval_dataset: str, run_name: str, modules_to_save: str) -> List[str]:
    return [
        "--dataset_use",
        dataset,
        "--eval_dataset_use",
        eval_dataset,
        "--data_flatten",
        "True",
        "--tune_mm_vision",
        "False",
        "--tune_mm_mlp",
        "False",
        "--tune_mm_llm",
        "True",
        "--tune_mm_motion",
        "False",
        "--bf16",
        "--output_dir",
        str(output_dir),
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--max_pixels",
        "50176",
        "--min_pixels",
        "784",
        "--eval_strategy",
        "epoch",
        "--save_strategy",
        "steps",
        "--save_steps",
        "400",
        "--save_total_limit",
        "2",
        "--learning_rate",
        "2e-4",
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        "0.03",
        "--max_grad_norm",
        "1",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--model_max_length",
        "4096",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "4",
        "--run_name",
        run_name,
        "--report_to",
        "none",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--lora_dropout",
        "0.05",
        "--lora_bias",
        "none",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--lora_modules_to_save",
        modules_to_save,
        "--lora_use_dora",
        "False",
    ]


def torchrun_cmd(model: Path, output_dir: Path, dataset: str, eval_dataset: str, run_name: str, modules_to_save: str, *, motion: bool) -> List[str]:
    port = str(random.randint(21000, 29999))
    launcher = [str(TORCHRUN)] if TORCHRUN.exists() else [str(PYTHON), "-m", "torch.distributed.run"]
    cmd = [
        *launcher,
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
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
    cmd.extend(common_train_args(output_dir, dataset, eval_dataset, run_name, modules_to_save))
    return cmd


def stop_keepalive_all() -> None:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return
    for line in out.splitlines():
        if "gpu_keepalive.py" not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts or not parts[0].isdigit():
            continue
        try:
            os.kill(int(parts[0]), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(3)


def start_keepalive_all() -> None:
    if not KEEPALIVE.exists():
        return
    for gpu in range(4):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if any(line.strip().isdigit() for line in out.splitlines()):
                continue
        except Exception:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = LOG_DIR / f"keepalive_gpu{gpu}.log"
        with log.open("ab") as f:
            subprocess.Popen([str(PYTHON), str(KEEPALIVE)], stdout=f, stderr=f, env=env, start_new_session=True)


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = load_status()
    ensure_motion_alias()
    env = base_env()

    try:
        stop_keepalive_all()
        rc = run_cmd(
            status,
            "prepare_data",
            [
                str(PYTHON),
                str(TOOLS / "prepare_goal_finetune_data.py"),
                "--raw-dir",
                str(QWEN_ROOT / "data" / "grpo_training" / "raw"),
                "--benchmark",
                str(QWEN_ROOT / "data" / "benchmark" / "text" / "QA" / "QA_500.json"),
                "--qwen-root",
                str(QWEN_ROOT),
                "--output-dir",
                str(RUN / "data"),
            ],
            LOG_DIR / "qwen_prepare_data.log",
            cwd=QWEN_ROOT,
            env=env,
        )
        if rc != 0:
            return rc

        motion_out = RUN / "qwen_lora" / "motionr1_vm"
        rc = run_cmd(
            status,
            "train_motionr1_vm_lora",
            torchrun_cmd(
                MODEL_MOTION_BASE,
                motion_out,
                "codex_motionx_qa_train_vm",
                "codex_motionx_qa_val_vm",
                "motionr1-vm-lora-goal-20260717",
                "visual,visual.merger,motion_encoder,motion_embed,motion_proj",
                motion=True,
            ),
            LOG_DIR / "train_motionr1_vm_lora.log",
            cwd=QWEN_ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        )
        if rc != 0:
            return rc

        rc = run_cmd(
            status,
            "eval_motionr1_vm_lora_QA500",
            [
                str(PYTHON),
                str(TOOLS / "eval_motionr1_lora_mcq_score.py"),
                "--model",
                str(MODEL_MOTION_BASE),
                "--adapter",
                str(motion_out),
                "--processor",
                str(motion_out),
                "--dataset",
                str(EVAL_DIR / "QA_500_motion.abs.jsonl"),
                "--output",
                str(RUN / "qwen_eval" / "motionr1_vm_lora_QA500.jsonl"),
                "--branch",
                "vm",
                "--prompt_mode",
                "short",
                "--device",
                "cuda:0",
                "--attn_implementation",
                "sdpa",
            ],
            LOG_DIR / "eval_motionr1_vm_lora_QA500.log",
            cwd=QWEN_ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        if rc != 0:
            return rc

        video_out = RUN / "qwen_lora" / "qwen3vl4b_video"
        rc = run_cmd(
            status,
            "train_qwen3vl4b_video_lora",
            torchrun_cmd(
                MODEL_QWEN3VL4B,
                video_out,
                "codex_motionx_qa_train_v",
                "codex_motionx_qa_val_v",
                "qwen3vl4b-video-lora-goal-20260717",
                "visual,visual.merger",
                motion=False,
            ),
            LOG_DIR / "train_qwen3vl4b_video_lora.log",
            cwd=QWEN_ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        )
        if rc != 0:
            return rc

        rc = run_cmd(
            status,
            "eval_qwen3vl4b_video_lora_QA500",
            [
                str(PYTHON),
                str(TOOLS / "eval_open_vlm_mcq_score.py"),
                "--model",
                str(MODEL_QWEN3VL4B),
                "--adapter",
                str(video_out),
                "--processor",
                str(video_out),
                "--dataset",
                str(EVAL_DIR / "QA_500_video.abs.jsonl"),
                "--output",
                str(RUN / "qwen_eval" / "qwen3vl4b_video_lora_QA500.jsonl"),
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
            LOG_DIR / "eval_qwen3vl4b_video_lora_QA500.log",
            cwd=QWEN_ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        return rc
    finally:
        start_keepalive_all()


if __name__ == "__main__":
    raise SystemExit(main())
