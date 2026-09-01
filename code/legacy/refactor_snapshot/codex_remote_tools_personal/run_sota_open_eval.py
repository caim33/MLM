#!/usr/bin/env python3
"""Wait for downloaded open models and evaluate them across available GPUs."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
MODEL_ROOT = ROOT / "codex_models"
RUN_ROOT = ROOT / "codex_runs" / "sota_open_eval_20260716"
TOOLS = ROOT / "codex_tools"
PYTHON = ROOT / "codex_envs" / "mllm" / "bin" / "python"
KEEPALIVE = Path("/tmp/codex_motionllm/gpu_keepalive.py")

DATASETS = [
    {
        "name": "QA_v_only_500",
        "path": ROOT / "codex_runs" / "qwen_eval_data" / "QA_v_only_500.abs.jsonl",
        "branch": "v",
    },
    {
        "name": "QA_500_video_only",
        "path": ROOT / "codex_runs" / "qwen_eval_data" / "QA_500.abs.jsonl",
        "branch": "vm",
    },
]


@dataclass
class ModelJob:
    model_id: str
    input_mode: str
    gpu: int
    model_class: str = "auto"
    image_patch_size: int = 16
    fps: float = 1.0
    nframes: Optional[int] = None
    num_frames: int = 8
    max_pixels: int = 200704
    total_pixels: int = 1605632


JOBS = [
    ModelJob("Qwen/Qwen3-VL-4B-Instruct", "video", 0, model_class="qwen3vl", fps=1.0, max_pixels=200704, total_pixels=1605632),
    ModelJob("Qwen/Qwen3-VL-8B-Instruct", "video", 1, model_class="qwen3vl", fps=1.0, max_pixels=200704, total_pixels=1605632),
    ModelJob("Qwen/Qwen3.5-4B", "frames", 2, model_class="image_text", num_frames=8, max_pixels=150528),
    ModelJob("Qwen/Qwen3.6-27B", "frames", 3, model_class="image_text", num_frames=6, max_pixels=100352),
]


def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id).strip("_")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_status(status: Dict[str, Dict[str, object]]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = RUN_ROOT / "status.json.tmp"
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(RUN_ROOT / "status.json")


def resolve_model_path(model_id: str) -> Optional[Path]:
    alias = MODEL_ROOT / safe_name(model_id)
    if alias.exists():
        return alias.resolve()
    alias_marker = alias.with_suffix(".path.txt")
    if alias_marker.exists():
        p = Path(alias_marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p.resolve()
    manifest = MODEL_ROOT / f"{safe_name(model_id)}.download.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("status") == "ok":
                p = Path(payload["path"])
                if p.exists():
                    return p.resolve()
        except Exception:
            return None
    return None


def gpu_keepalive_pids(gpu: int) -> List[int]:
    try:
        output = subprocess.check_output(
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
    pids: List[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or not line.isdigit():
            continue
        pid = int(line)
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if "gpu_keepalive.py" in cmdline:
            pids.append(pid)
    return pids


def stop_keepalive(gpu: int) -> None:
    for pid in gpu_keepalive_pids(gpu):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)


def start_keepalive(gpu: int) -> None:
    if not KEEPALIVE.exists() or gpu_keepalive_pids(gpu):
        return
    log = Path("/tmp/codex_motionllm") / f"gpu{gpu}_keepalive.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("ab") as f:
        subprocess.Popen(["python3", str(KEEPALIVE)], stdout=f, stderr=f, env=env, start_new_session=True)


def run_command(cmd: List[str], log_path: Path, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        return proc.wait()


def eval_args(job: ModelJob, model_path: Path, dataset: Dict[str, object], output: Path, limit: Optional[int]) -> List[str]:
    cmd = [
        str(PYTHON),
        str(TOOLS / "eval_open_vlm_mcq_score.py"),
        "--model",
        str(model_path),
        "--dataset",
        str(dataset["path"]),
        "--output",
        str(output),
        "--branch",
        str(dataset["branch"]),
        "--input_mode",
        job.input_mode,
        "--model_class",
        job.model_class,
        "--device",
        "cuda:0",
        "--device_map",
        "single",
        "--attn_implementation",
        "sdpa",
        "--image_patch_size",
        str(job.image_patch_size),
        "--max_pixels",
        str(job.max_pixels),
        "--total_pixels",
        str(job.total_pixels),
    ]
    if job.input_mode == "video":
        cmd.extend(["--fps", str(job.fps)])
        if job.nframes is not None:
            cmd.extend(["--nframes", str(job.nframes)])
    if job.input_mode == "frames":
        cmd.extend(["--num_frames", str(job.num_frames)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def run_job(job: ModelJob, status: Dict[str, Dict[str, object]], lock: threading.Lock) -> None:
    name = safe_name(job.model_id)
    out_dir = RUN_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with lock:
        status[name] = {"model_id": job.model_id, "gpu": job.gpu, "state": "waiting_for_download", "updated_at": now()}
        write_status(status)

    model_path: Optional[Path] = None
    while model_path is None:
        model_path = resolve_model_path(job.model_id)
        if model_path is None:
            time.sleep(120)

    with lock:
        status[name].update({"state": "downloaded", "model_path": str(model_path), "updated_at": now()})
        write_status(status)

    stop_keepalive(job.gpu)
    try:
        smoke_out = out_dir / "smoke_QA_v_only_10.jsonl"
        smoke_log = out_dir / "smoke_QA_v_only_10.log"
        if not smoke_out.with_suffix(".summary.json").exists():
            with lock:
                status[name].update({"state": "smoke_running", "updated_at": now(), "log": str(smoke_log)})
                write_status(status)
            rc = run_command(eval_args(job, model_path, DATASETS[0], smoke_out, limit=10), smoke_log, job.gpu)
            if rc != 0:
                with lock:
                    status[name].update({"state": "smoke_failed", "returncode": rc, "updated_at": now(), "log": str(smoke_log)})
                    write_status(status)
                return

        for dataset in DATASETS:
            out = out_dir / f"{dataset['name']}.jsonl"
            log = out_dir / f"{dataset['name']}.log"
            if out.with_suffix(".summary.json").exists():
                continue
            with lock:
                status[name].update({"state": f"running_{dataset['name']}", "updated_at": now(), "log": str(log)})
                write_status(status)
            rc = run_command(eval_args(job, model_path, dataset, out, limit=None), log, job.gpu)
            if rc != 0:
                with lock:
                    status[name].update({"state": f"failed_{dataset['name']}", "returncode": rc, "updated_at": now(), "log": str(log)})
                    write_status(status)
                return

        with lock:
            status[name].update({"state": "complete", "updated_at": now()})
            write_status(status)
    finally:
        start_keepalive(job.gpu)


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Dict[str, object]] = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=run_job, args=(job, status, lock), daemon=False) for job in JOBS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
