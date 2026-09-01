#!/usr/bin/env python3
"""Run Video-LLaVA LoRA after the Qwen continuation pipeline releases GPUs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
RUN = ROOT / "codex_runs" / "finetune_goal_20260717"
TOOLS = ROOT / "codex_tools"
PYTHON = Path("/usr/bin/python3")
BASE_MODEL = ROOT / "MVBench_Eval" / "models" / "Video-LLaVA-7B"
CACHE_DIR = ROOT / "MVBench_Eval" / "cache_dir"
TRAIN_JSON = RUN / "data" / "sft" / "motionx_qa_train_v.json"
SMOKE_JSON = RUN / "data" / "sft" / "motionx_qa_train_v_smoke32.json"
EVAL_JSON = RUN / "data" / "eval" / "QA_500_video.abs.jsonl"
OUT_DIR = RUN / "video_lora" / "videollava"
SMOKE_DIR = RUN / "video_lora" / "_smoke_videollava"
VIDEO_EVAL = RUN / "video_eval" / "videollava_lora_QA500.jsonl"
LOG_DIR = RUN / "logs"
STATUS_PATH = RUN / "video_pipeline" / "videollava_status.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def set_step(status: Dict[str, Any], name: str, **payload: Any) -> None:
    step = status.setdefault("steps", {}).setdefault(name, {})
    step.update(payload)
    step["updated_at"] = now()
    save_status(status)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_qwen(status: Dict[str, Any]) -> None:
    pid_file = RUN / "qwen_pipeline" / "continue.pid"
    set_step(status, "wait_for_qwen_continue", state="waiting", pid_file=str(pid_file))
    while True:
        if not pid_file.exists():
            break
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            break
        if not pid_alive(pid):
            break
        time.sleep(120)
    set_step(status, "wait_for_qwen_continue", state="completed")


def gpu_compute_pids(gpu: int) -> List[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def stop_keepalive(gpu: int) -> None:
    for pid in gpu_compute_pids(gpu):
        if "gpu_keepalive.py" not in cmdline(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)


def wait_for_gpu(status: Dict[str, Any], gpu: int) -> None:
    set_step(status, f"wait_for_gpu{gpu}", state="waiting")
    while True:
        stop_keepalive(gpu)
        busy = [pid for pid in gpu_compute_pids(gpu) if "gpu_keepalive.py" not in cmdline(pid)]
        if not busy:
            break
        set_step(status, f"wait_for_gpu{gpu}", state="waiting", busy_pids=busy)
        time.sleep(120)
    set_step(status, f"wait_for_gpu{gpu}", state="completed")


def env_for(gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    src = ROOT / "codex_runs" / "video_model_sources" / "Video-LLaVA"
    legacy = ROOT / "codex_envs" / "legacy_torch211_cu128"
    legacy_tf = ROOT / "codex_envs" / "legacy_tf431"
    video_extra = ROOT / "codex_envs" / "video_extra"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DISABLED"] = "true"
    env["PYTHONPATH"] = ":".join(str(p) for p in [legacy, legacy_tf, video_extra, src]) + ":" + env.get("PYTHONPATH", "")
    return env


def train_cmd(data_path: Path, output_dir: Path, *, smoke: bool) -> List[str]:
    cmd = [
        str(PYTHON),
        str(TOOLS / "run_videollava_lora_sft_goal.py"),
        "--lora_enable",
        "True",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--lora_dropout",
        "0.05",
        "--mm_projector_lr",
        "2e-5",
        "--model_name_or_path",
        str(BASE_MODEL),
        "--version",
        "v1",
        "--data_path",
        str(data_path),
        "--video_folder",
        "/",
        "--image_tower",
        "LanguageBind/LanguageBind_Image",
        "--video_tower",
        "LanguageBind/LanguageBind_Video_merge",
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "pad",
        "--group_by_modality_length",
        "True",
        "--bf16",
        "True",
        "--output_dir",
        str(output_dir),
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "no" if smoke else "steps",
        "--save_steps",
        "500",
        "--save_total_limit",
        "1",
        "--learning_rate",
        "2e-4",
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "5",
        "--tf32",
        "True",
        "--model_max_length",
        "2048",
        "--tokenizer_model_max_length",
        "3072",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "4",
        "--lazy_preprocess",
        "True",
        "--report_to",
        "none",
        "--cache_dir",
        str(CACHE_DIR),
    ]
    if smoke:
        cmd.extend(["--max_steps", "1"])
    return cmd


def eval_cmd() -> List[str]:
    return [
        str(PYTHON),
        str(TOOLS / "eval_videollava_lora_mcq_generate.py"),
        "--model",
        str(BASE_MODEL),
        "--adapter",
        str(OUT_DIR),
        "--dataset",
        str(EVAL_JSON),
        "--output",
        str(VIDEO_EVAL),
        "--branch",
        "all",
        "--device",
        "cuda",
        "--device_map",
        "single",
        "--prompt_mode",
        "short",
        "--max_new_tokens",
        "32",
    ]


def run_cmd(status: Dict[str, Any], name: str, cmd: List[str], log_path: Path, *, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_step(status, name, state="running", command=cmd, log=str(log_path), gpu=gpu, started_at=now())
    with log_path.open("ab") as log:
        log.write((f"\n===== {now()} START {name} =====\n").encode())
        log.write((" ".join(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env_for(gpu), stdout=log, stderr=subprocess.STDOUT)
        rc = proc.wait()
        log.write((f"\n===== {now()} END {name} rc={rc} =====\n").encode())
    set_step(status, name, state="completed" if rc == 0 else "failed", returncode=rc, finished_at=now())
    return rc


def adapter_ready(path: Path) -> bool:
    return (path / "adapter_model.bin").exists() or (path / "adapter_model.safetensors").exists()


def main() -> int:
    status = load_status()
    wait_for_qwen(status)
    wait_for_gpu(status, 0)

    if not adapter_ready(SMOKE_DIR):
        rc = run_cmd(status, "smoke_train_videollava_lora", train_cmd(SMOKE_JSON, SMOKE_DIR, smoke=True), LOG_DIR / "smoke_videollava_lora.log", gpu=0)
        if rc != 0:
            return rc
    else:
        set_step(status, "smoke_train_videollava_lora", state="completed", skipped=True, output=str(SMOKE_DIR))

    if not adapter_ready(OUT_DIR):
        rc = run_cmd(status, "train_videollava_lora", train_cmd(TRAIN_JSON, OUT_DIR, smoke=False), LOG_DIR / "train_videollava_lora.log", gpu=0)
        if rc != 0:
            return rc
    else:
        set_step(status, "train_videollava_lora", state="completed", skipped=True, output=str(OUT_DIR))

    if not VIDEO_EVAL.with_suffix(".summary.json").exists():
        rc = run_cmd(status, "eval_videollava_lora_QA500", eval_cmd(), LOG_DIR / "eval_videollava_lora_QA500.log", gpu=0)
        if rc != 0:
            return rc
    else:
        set_step(status, "eval_videollava_lora_QA500", state="completed", skipped=True, output=str(VIDEO_EVAL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
