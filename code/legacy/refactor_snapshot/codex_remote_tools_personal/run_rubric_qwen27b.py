#!/usr/bin/env python3
"""Upload Rubric-RL sample data to the remote GPU host and run Qwen 27B.

No secrets are stored here. Provide the SSH password with:
  set CODEX_REMOTE_PASSWORD=...
or enter it when prompted.
"""

from __future__ import annotations

import argparse
import getpass
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path
from typing import Iterable

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
DATA_FILES = (
    "sample_summary_gt.jsonl",
    "sample_summary_criteria.jsonl",
    "sample_summary_candidates.jsonl",
    "sample_summary_expected_judgment.jsonl",
)
OUTPUT_FILES = (
    "sample_summary_qwen36_27b_criteria.jsonl",
    "sample_summary_qwen36_27b_rewarded_manualcriteria.jsonl",
    "sample_summary_qwen36_27b_rewarded_qwencriteria.jsonl",
    "sample_summary_qwen36_27b_run.log",
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
    parts = [part for part in path.split("/") if part]
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_tree(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str, patterns: Iterable[str]) -> None:
    sftp_mkdirs(sftp, remote_dir)
    for pattern in patterns:
        for local in sorted(local_dir.glob(pattern)):
            if local.is_dir():
                continue
            remote = posixpath.join(remote_dir, local.name)
            sftp.put(str(local), remote)
            print(f"uploaded {local} -> {remote}")


def upload_inputs(client: paramiko.SSHClient, remote_root: str) -> None:
    sftp = client.open_sftp()
    try:
        upload_tree(
            sftp,
            LOCAL_ROOT / "rubric_rl",
            posixpath.join(remote_root, "rubric_rl"),
            ("*.py", "*.md"),
        )
        upload_tree(
            sftp,
            LOCAL_ROOT / "rubric_rl" / "prompt_templates",
            posixpath.join(remote_root, "rubric_rl", "prompt_templates"),
            ("*.txt",),
        )
        remote_data = posixpath.join(remote_root, "data", "rubric_rl")
        sftp_mkdirs(sftp, remote_data)
        for name in DATA_FILES:
            local = LOCAL_ROOT / "data" / "rubric_rl" / name
            remote = posixpath.join(remote_data, name)
            sftp.put(str(local), remote)
            print(f"uploaded {local} -> {remote}")
    finally:
        sftp.close()


def download_outputs(client: paramiko.SSHClient, remote_root: str) -> None:
    local_dir = LOCAL_ROOT / "data" / "rubric_rl"
    remote_data = posixpath.join(remote_root, "data", "rubric_rl")
    sftp = client.open_sftp()
    try:
        for name in OUTPUT_FILES:
            remote = posixpath.join(remote_data, name)
            local = local_dir / name
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
    py = args.python or posixpath.join(remote_root, "codex_envs", "mllm", "bin", "python")
    model = args.model or posixpath.join(remote_root, "codex_models", "Qwen__Qwen3.6-27B")
    data_dir = posixpath.join(remote_root, "data", "rubric_rl")
    max_memory = args.max_memory
    return f"""
set -euo pipefail
cd {q(remote_root)}
export CUDA_VISIBLE_DEVICES={q(args.gpus)}
LOG={q(posixpath.join(data_dir, "sample_summary_qwen36_27b_run.log"))}
{{
  echo "started_at=$(date -Is)"
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits || true
  test -x {q(py)}
  test -d {q(model)}
  {q(py)} -m rubric_rl.extract_motion_criteria \\
    --model {q(model)} \\
    --input {q(posixpath.join(data_dir, "sample_summary_gt.jsonl"))} \\
    --output {q(posixpath.join(data_dir, "sample_summary_qwen36_27b_criteria.jsonl"))} \\
    --gt-key gt_description \\
    --limit 1 \\
    --max-memory {q(max_memory)} \\
    --keep-raw
  {q(py)} -m rubric_rl.judge_motion_caption \\
    --model {q(model)} \\
    --criteria {q(posixpath.join(data_dir, "sample_summary_criteria.jsonl"))} \\
    --candidates {q(posixpath.join(data_dir, "sample_summary_candidates.jsonl"))} \\
    --output {q(posixpath.join(data_dir, "sample_summary_qwen36_27b_rewarded_manualcriteria.jsonl"))} \\
    --candidate-key candidate \\
    --limit 1 \\
    --max-memory {q(max_memory)} \\
    --keep-raw
  {q(py)} -m rubric_rl.judge_motion_caption \\
    --model {q(model)} \\
    --criteria {q(posixpath.join(data_dir, "sample_summary_qwen36_27b_criteria.jsonl"))} \\
    --candidates {q(posixpath.join(data_dir, "sample_summary_candidates.jsonl"))} \\
    --output {q(posixpath.join(data_dir, "sample_summary_qwen36_27b_rewarded_qwencriteria.jsonl"))} \\
    --candidate-key candidate \\
    --limit 1 \\
    --max-memory {q(max_memory)} \\
    --keep-raw
  echo "finished_at=$(date -Is)"
}} 2>&1 | tee "$LOG"
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
