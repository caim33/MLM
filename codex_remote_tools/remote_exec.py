#!/usr/bin/env python3
"""Small SSH helper for the current Codex remote environment.

Reads connection details from dev_env_connection.txt by default so secrets do
not need to be placed on the command line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONN = ROOT / "dev_env_connection.txt"


def read_conn(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if not values.get("host"):
        m = re.search(r"ssh\s+(\S+?)@([\d.]+)\s+-p\s+(\d+)", values.get("ssh_command", ""))
        if m:
            values.setdefault("username", m.group(1))
            values["host"] = m.group(2)
            values.setdefault("port", m.group(3))
    return values


def connect(conn: dict[str, str]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        conn.get("host", "10.26.6.88"),
        port=int(conn["port"]),
        username=conn.get("username", "root"),
        password=conn["password"],
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int | None) -> int:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    for stream in (stdout, stderr):
        data = stream.read().decode("utf-8", errors="replace")
        if data:
            print(data, end="")
    return stdout.channel.recv_exit_status()


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def transfer(client: paramiko.SSHClient, pairs: list[tuple[str, str]], direction: str) -> None:
    sftp = client.open_sftp()
    try:
        for left, right in pairs:
            if direction == "put":
                remote_dir = os.path.dirname(right)
                if remote_dir:
                    mkdir_p(sftp, remote_dir)
                sftp.put(left, right)
                print(f"put {left} -> {right}")
            else:
                local = Path(right)
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(left, right)
                print(f"get {left} -> {right}")
    finally:
        sftp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", default=str(DEFAULT_CONN))
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--cmd")
    parser.add_argument("--cmd-file")
    parser.add_argument("--put", nargs=2, action="append", metavar=("LOCAL", "REMOTE"))
    parser.add_argument("--get", nargs=2, action="append", metavar=("REMOTE", "LOCAL"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = read_conn(Path(args.conn))
    missing = [k for k in ("host", "port", "username", "password") if not conn.get(k)]
    if missing:
        raise SystemExit(f"missing connection fields: {', '.join(missing)}")
    client = connect(conn)
    try:
        if args.put:
            transfer(client, [(a, b) for a, b in args.put], "put")
        if args.get:
            transfer(client, [(a, b) for a, b in args.get], "get")
        cmd = args.cmd
        if args.cmd_file:
            cmd = Path(args.cmd_file).read_text(encoding="utf-8")
        if cmd:
            return run(client, cmd, args.timeout)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
