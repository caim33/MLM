#!/usr/bin/env python3
"""Upload V2 Rubric-RL assets and run Qwen3.6-27B on the remote GPU host."""

from __future__ import annotations

import argparse
import getpass
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
DATA_FILES = (
    "sample_summary_full_gt_v2.jsonl",
    "sample_summary_candidates_v2.jsonl",
)
OUTPUT_FILES = (
    "sample_summary_qwen36_27b_v2_full_criteria.jsonl",
    "sample_summary_qwen36_27b_v2_full_rewarded.jsonl",
    "sample_summary_qwen36_27b_v2_full_run.log",
    "sample_summary_qwen36_27b_v2_full_judge.nohup.log",
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("CODEX_REMOTE_PASSWORD") or getpass.getpass("Remote SSH password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run_capture(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def run_stream(client: paramiko.SSHClient, cmd: str) -> int:
    channel = client.get_transport().open_session()
    channel.exec_command(cmd)
    while True:
        wrote = False
        while channel.recv_ready():
            sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
            sys.stdout.flush()
            wrote = True
        while channel.recv_stderr_ready():
            sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
            sys.stderr.flush()
            wrote = True
        if channel.exit_status_ready():
            break
        if not wrote:
            time.sleep(0.5)
    while channel.recv_ready():
        sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
    while channel.recv_stderr_ready():
        sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
    sys.stdout.flush()
    sys.stderr.flush()
    return channel.recv_exit_status()


def sftp_mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    cur = ""
    for part in [part for part in path.split("/") if part]:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_inputs(client: paramiko.SSHClient, remote_root: str) -> None:
    sftp = client.open_sftp()
    try:
        remote_rubric = posixpath.join(remote_root, "rubric_rl")
        remote_templates = posixpath.join(remote_rubric, "prompt_templates")
        remote_data = posixpath.join(remote_root, "data", "rubric_rl")
        for path in (remote_rubric, remote_templates, remote_data):
            sftp_mkdirs(sftp, path)
        for local in sorted((LOCAL_ROOT / "rubric_rl").glob("*.py")):
            remote = posixpath.join(remote_rubric, local.name)
            sftp.put(str(local), remote)
            print(f"uploaded {local} -> {remote}")
        for local in sorted((LOCAL_ROOT / "rubric_rl" / "prompt_templates").glob("*.txt")):
            remote = posixpath.join(remote_templates, local.name)
            sftp.put(str(local), remote)
            print(f"uploaded {local} -> {remote}")
        for name in DATA_FILES:
            local = LOCAL_ROOT / "data" / "rubric_rl" / name
            remote = posixpath.join(remote_data, name)
            sftp.put(str(local), remote)
            print(f"uploaded {local} -> {remote}")
    finally:
        sftp.close()


def download_outputs(client: paramiko.SSHClient, remote_root: str) -> None:
    sftp = client.open_sftp()
    try:
        remote_data = posixpath.join(remote_root, "data", "rubric_rl")
        local_data = LOCAL_ROOT / "data" / "rubric_rl"
        for name in OUTPUT_FILES:
            remote = posixpath.join(remote_data, name)
            local = local_data / name
            try:
                sftp.stat(remote)
            except FileNotFoundError:
                print(f"missing remote output: {remote}")
                continue
            sftp.get(remote, str(local))
            print(f"downloaded {remote} -> {local}")
    finally:
        sftp.close()


def remote_command(args: argparse.Namespace) -> str:
    remote_root = args.remote_root
    data_dir = posixpath.join(remote_root, "data", "rubric_rl")
    py = args.python or posixpath.join(remote_root, "codex_envs", "mllm", "bin", "python")
    model = args.model or posixpath.join(remote_root, "codex_models", "Qwen__Qwen3.6-27B")
    criteria = posixpath.join(data_dir, "sample_summary_qwen36_27b_v2_full_criteria.jsonl")
    rewarded = posixpath.join(data_dir, "sample_summary_qwen36_27b_v2_full_rewarded.jsonl")
    log = posixpath.join(data_dir, "sample_summary_qwen36_27b_v2_full_run.log")
    return f"""
set -euo pipefail
cd {q(remote_root)}
export CUDA_VISIBLE_DEVICES={q(args.gpus)}
{{
  echo "started_at=$(date -Is)"
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits || true
  test -x {q(py)}
  test -d {q(model)}
  {q(py)} -m py_compile rubric_rl/prompts_v2.py rubric_rl/reward_v2.py rubric_rl/extract_motion_criteria_v2.py rubric_rl/judge_motion_caption_v2.py
  {q(py)} -m rubric_rl.extract_motion_criteria_v2 \\
    --model {q(model)} \\
    --input {q(posixpath.join(data_dir, "sample_summary_full_gt_v2.jsonl"))} \\
    --output {q(criteria)} \\
    --gt-key gt_description \\
    --limit 1 \\
    --max-memory {q(args.max_memory)} \\
    --max-new-tokens 3600 \\
    --keep-raw
  {q(py)} -m rubric_rl.judge_motion_caption_v2 \\
    --model {q(model)} \\
    --criteria {q(criteria)} \\
    --candidates {q(posixpath.join(data_dir, "sample_summary_candidates_v2.jsonl"))} \\
    --output {q(rewarded)} \\
    --candidate-key candidate \\
    --limit 1 \\
    --max-memory {q(args.max_memory)} \\
    --max-new-tokens 2600 \\
    --keep-raw
  echo "finished_at=$(date -Is)"
}} 2>&1 | tee {q(log)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.26.6.88")
    parser.add_argument("--port", type=int, default=31066)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--python", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-memory", default="0:38GiB,1:38GiB,cpu:120GiB")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = connect(args)
    try:
        rc, out, err = run_capture(
            client,
            f"date; hostname; test -d {q(args.remote_root)} && echo remote_root_ok",
            timeout=60,
        )
        print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")
        if rc != 0:
            raise RuntimeError(f"remote probe failed rc={rc}")
        if not args.skip_upload:
            upload_inputs(client, args.remote_root)
        if not args.skip_run:
            rc = run_stream(client, remote_command(args))
            if rc != 0:
                print(f"remote run failed rc={rc}", file=sys.stderr)
        if not args.skip_download:
            download_outputs(client, args.remote_root)
        return rc if "rc" in locals() else 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
