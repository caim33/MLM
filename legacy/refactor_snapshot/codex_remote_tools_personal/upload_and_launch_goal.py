#!/usr/bin/env python3
"""Upload goal finetune tools to the remote host and launch orchestration.

No secrets are stored in this file. Pass SSH password through environment:
  CODEX_REMOTE_PASSWORD=... python upload_and_launch_goal.py --port 30671
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
REMOTE_TOOLS = posixpath.join(REMOTE_ROOT, "codex_tools")

UPLOADS: Tuple[Tuple[Path, str], ...] = (
    (LOCAL_ROOT / "codex_remote_tools" / "prepare_goal_finetune_data.py", "prepare_goal_finetune_data.py"),
    (LOCAL_ROOT / "codex_remote_tools" / "orchestrate_goal_finetune.py", "orchestrate_goal_finetune.py"),
    (LOCAL_ROOT / "codex_remote_tools" / "gpu_keepalive.py", "gpu_keepalive.py"),
    (LOCAL_ROOT / "remote_scripts" / "motion_proxy_train_eval.py", "motion_proxy_train_eval.py"),
)


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=15, banner_timeout=15, auth_timeout=15)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def upload(client: paramiko.SSHClient, files: Iterable[Tuple[Path, str]]) -> None:
    rc, out, err = run(client, f"mkdir -p {REMOTE_TOOLS!r}")
    if rc != 0:
        raise RuntimeError(f"mkdir failed: {err or out}")
    sftp = client.open_sftp()
    try:
        for local, remote_name in files:
            if not local.exists():
                raise FileNotFoundError(local)
            remote = posixpath.join(REMOTE_TOOLS, remote_name)
            sftp.put(str(local), remote)
            sftp.chmod(remote, 0o755)
            print(f"uploaded {local} -> {remote}")
    finally:
        sftp.close()


def launch(client: paramiko.SSHClient) -> None:
    cmd = rf"""
set -u
ROOT={REMOTE_ROOT}
RUN=$ROOT/codex_runs/finetune_goal_20260717
mkdir -p "$RUN/logs"
PY="$ROOT/codex_envs/mllm/bin/python"
if [ ! -x "$PY" ]; then PY=python3; fi
nohup "$PY" "$ROOT/codex_tools/orchestrate_goal_finetune.py" --force-prepare > "$RUN/logs/orchestrator.nohup.log" 2>&1 &
echo "orchestrator_pid=$!"
sleep 2
ps -p "$!" -o pid,stat,etime,cmd || true
"""
    rc, out, err = run(client, cmd, timeout=60)
    print(out)
    if err:
        print(err, file=sys.stderr)
    if rc != 0:
        raise RuntimeError(f"launch failed rc={rc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.26.6.88")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("CODEX_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("Missing CODEX_REMOTE_PASSWORD")
    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"connect attempt {attempt}/{args.retries} {args.host}:{args.port}")
            client = connect(args.host, args.port, args.user, password)
            try:
                rc, out, err = run(client, "date; hostname; test -d /wangbenyou-sulongjie/Motion-r1/caimeng/MLLM && echo root_ok")
                print(out)
                if rc != 0:
                    raise RuntimeError(err or out)
                upload(client, UPLOADS)
                launch(client)
                return 0
            finally:
                client.close()
        except Exception as exc:
            last_exc = exc
            print(f"attempt failed: {type(exc).__name__}: {exc}")
            if attempt < args.retries:
                time.sleep(args.sleep)
    raise SystemExit(f"all attempts failed: {last_exc}")


if __name__ == "__main__":
    raise SystemExit(main())
