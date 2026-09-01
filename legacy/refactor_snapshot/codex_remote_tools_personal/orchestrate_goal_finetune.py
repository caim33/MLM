#!/usr/bin/env python3
"""Orchestrate the user's all-model finetune/eval goal on the remote server.

This script does not claim the whole goal is complete. It starts the pieces
that are currently executable from the repository state, records blockers for
models that lack a finetune entry, and keeps a machine-readable status file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
QWEN_ROOT = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune")
RUN_ROOT = ROOT / "codex_runs/finetune_goal_20260717"
TOOLS = ROOT / "codex_tools"
PYTHON = ROOT / "codex_envs/mllm/bin/python"
SYSTEM_PYTHON = Path(sys.executable)

RAW_DIR = QWEN_ROOT / "data/grpo_training/raw"
BENCHMARK = QWEN_ROOT / "data/benchmark/text/QA/QA_500.json"
DATA_DIR = RUN_ROOT / "data"
LOG_DIR = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.json"

KEEPALIVE = TOOLS / "gpu_keepalive.py"


@dataclass
class Job:
    name: str
    gpu: int
    cmd: List[str]
    log_path: Path
    kind: str


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_status() -> Dict[str, Any]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"created_at": now(), "jobs": {}, "blockers": {}, "notes": []}


def update_job(status: Dict[str, Any], name: str, **updates: Any) -> None:
    jobs = status.setdefault("jobs", {})
    jobs.setdefault(name, {}).update(updates)
    jobs[name]["updated_at"] = now()
    write_json(STATUS_PATH, status)


def run_checked(cmd: List[str], log_path: Path, cwd: Path = ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(("CMD " + " ".join(cmd) + "\n").encode("utf-8"))
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def query_gpu_compute_pids(gpu: int) -> List[int]:
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
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def keepalive_pids(gpu: int) -> List[int]:
    return [pid for pid in query_gpu_compute_pids(gpu) if "gpu_keepalive.py" in pid_cmdline(pid)]


def start_keepalive(gpu: int) -> None:
    if not KEEPALIVE.exists() or keepalive_pids(gpu):
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = LOG_DIR / f"gpu{gpu}_keepalive.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        subprocess.Popen(
            [str(PYTHON if PYTHON.exists() else SYSTEM_PYTHON), str(KEEPALIVE)],
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )


def stop_keepalive(gpu: int) -> None:
    for pid in keepalive_pids(gpu):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)


def ensure_keepalive_on_idle_gpus(active_gpus: Iterable[int]) -> None:
    active = set(active_gpus)
    for gpu in range(4):
        if gpu in active:
            continue
        if not query_gpu_compute_pids(gpu):
            start_keepalive(gpu)


def prepare_data(status: Dict[str, Any], force: bool) -> Dict[str, Any]:
    manifest = DATA_DIR / "manifest.json"
    if manifest.exists() and not force:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        status["data"] = payload
        write_json(STATUS_PATH, status)
        return payload
    cmd = [
        str(PYTHON if PYTHON.exists() else SYSTEM_PYTHON),
        str(TOOLS / "prepare_goal_finetune_data.py"),
        "--raw-dir",
        str(RAW_DIR),
        "--benchmark",
        str(BENCHMARK),
        "--qwen-root",
        str(QWEN_ROOT),
        "--output-dir",
        str(DATA_DIR),
    ]
    rc = run_checked(cmd, LOG_DIR / "prepare_goal_finetune_data.log")
    if rc != 0:
        raise RuntimeError(f"prepare data failed rc={rc}; see {LOG_DIR / 'prepare_goal_finetune_data.log'}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    status["data"] = payload
    write_json(STATUS_PATH, status)
    return payload


def proxy_jobs(manifest: Dict[str, Any]) -> List[Job]:
    train = manifest["paths"]["train_raw"]
    val = manifest["paths"]["val_raw"]
    test = manifest["paths"].get("test_motion") or manifest["paths"]["test_all"]
    out_dir = RUN_ROOT / "proxy"
    models = ["agcn_mlp", "agcn_rnn", "motionclip_mlp", "motionclip_rnn"]
    jobs: List[Job] = []
    for gpu, model in enumerate(models):
        jobs.append(
            Job(
                name=model,
                gpu=gpu,
                kind="proxy",
                log_path=LOG_DIR / f"proxy_{model}.log",
                cmd=[
                    str(PYTHON if PYTHON.exists() else SYSTEM_PYTHON),
                    str(TOOLS / "motion_proxy_train_eval.py"),
                    "--model",
                    model,
                    "--train",
                    train,
                    "--val",
                    val,
                    "--test",
                    test,
                    "--qwen-root",
                    str(QWEN_ROOT),
                    "--output-dir",
                    str(out_dir),
                    "--device",
                    "cuda:0",
                    "--epochs",
                    "80",
                    "--batch-size",
                    "64",
                ],
            )
        )
    return jobs


def launch_job(job: Job, status: Dict[str, Any]) -> Optional[int]:
    summary = RUN_ROOT / "proxy" / job.name / "summary.json"
    if job.kind == "proxy" and summary.exists():
        update_job(
            status,
            job.name,
            state="already_complete",
            summary=str(summary),
            gpu=job.gpu,
            kind=job.kind,
        )
        return None
    stop_keepalive(job.gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("ab") as log:
        log.write(("CMD " + " ".join(job.cmd) + "\n").encode("utf-8"))
        proc = subprocess.Popen(job.cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    update_job(
        status,
        job.name,
        state="running",
        pid=proc.pid,
        gpu=job.gpu,
        log=str(job.log_path),
        kind=job.kind,
        cmd=job.cmd,
    )
    return proc.pid


def record_model_blockers(status: Dict[str, Any]) -> None:
    blockers = status.setdefault("blockers", {})
    for model in [
        "VideoChatGPT",
        "Video-LLaVA-7B",
        "VideoChat2",
        "VideoLLaMA",
        "mPLUG-Owl-Video",
        "Otter-Video",
    ]:
        blockers[model] = {
            "state": "needs_finetune_entry",
            "note": (
                "Eval runner/model weights may exist under MLLM, but no confirmed local finetune "
                "entry was found in the current workspace. Need official training code or a new "
                "LoRA/SFT adapter before this model can satisfy the goal."
            ),
        }
    blockers["MotionLLM"] = {
        "state": "needs_repo_and_checkpoints",
        "note": (
            "Previous eval runner expected /dataset_rc_mm/caim4@xiaopeng.com/code/MotionLLM "
            "and checkpoints iter-015000-ckpt.pth, linear-iter-015000-ckpt.pth, "
            "vicuna lit_model.pth/tokenizer.model. Recheck remote and restore before finetune."
        ),
    }
    write_json(STATUS_PATH, status)


def collect_finished_proxy(status: Dict[str, Any]) -> None:
    for model in ["agcn_mlp", "agcn_rnn", "motionclip_mlp", "motionclip_rnn"]:
        summary = RUN_ROOT / "proxy" / model / "summary.json"
        if summary.exists():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            update_job(
                status,
                model,
                state="complete",
                summary=str(summary),
                test_accuracy=payload.get("test", {}).get("accuracy"),
                checkpoint=payload.get("checkpoint"),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--no-keepalive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = load_status()
    status.update(
        {
            "root": str(ROOT),
            "qwen_root": str(QWEN_ROOT),
            "raw_dir": str(RAW_DIR),
            "benchmark": str(BENCHMARK),
            "run_root": str(RUN_ROOT),
            "updated_at": now(),
        }
    )
    write_json(STATUS_PATH, status)
    record_model_blockers(status)
    manifest = prepare_data(status, args.force_prepare)
    if args.prepare_only:
        return 0

    jobs = proxy_jobs(manifest)
    active_gpus = [job.gpu for job in jobs]
    for job in jobs:
        launch_job(job, status)
    if not args.no_keepalive:
        ensure_keepalive_on_idle_gpus(active_gpus)
    collect_finished_proxy(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
