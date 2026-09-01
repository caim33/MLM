#!/usr/bin/env python3
"""Upload and launch Qwen3.6-27B CoT-to-QA generation on the remote host."""

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
DEFAULT_HOST = "10.26.6.88"
DEFAULT_REMOTE_ROOT = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
DEFAULT_INPUT = (
    "/wangbenyou-sulongjie/qwen-vl-finetune/data/description_eval/"
    "motionx_desc_eval_random1000_after850_sft.jsonl"
)
DEFAULT_MODEL = posixpath.join(DEFAULT_REMOTE_ROOT, "codex_models", "Qwen__Qwen3.6-27B")
DEFAULT_PYTHON = posixpath.join(DEFAULT_REMOTE_ROOT, "codex_envs", "mllm", "bin", "python")
DEFAULT_RUN_DIR = posixpath.join(
    DEFAULT_REMOTE_ROOT,
    "codex_runs",
    "qa_gen_cot_20260724",
    "qwen36_27b_cot_prompt_v2",
)
DEFAULT_PROMPT = (
    r"D:\dd\WeChat Files\wxid_plv1nogj4wl222\FileStorage\File\2026-07\qa_generation_prompt_v2.txt"
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


def sftp_mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    cur = ""
    for part in [part for part in path.split("/") if part]:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp_mkdirs(sftp, posixpath.dirname(remote))
    sftp.put(str(local), remote)
    print(f"uploaded {local} -> {remote}")


def upload_assets(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    sftp = client.open_sftp()
    try:
        upload_file(
            sftp,
            LOCAL_ROOT / "tools" / "generate_qa_from_cot.py",
            posixpath.join(args.remote_root, "tools", "generate_qa_from_cot.py"),
        )
        upload_file(
            sftp,
            LOCAL_ROOT / "rubric_rl" / "qwen_text.py",
            posixpath.join(args.remote_root, "rubric_rl", "qwen_text.py"),
        )
        upload_file(
            sftp,
            Path(args.prompt),
            posixpath.join(args.run_dir, "qa_generation_prompt_v2.txt"),
        )
        run_script = build_run_script(args)
        remote_run_script = posixpath.join(args.run_dir, "run_qa_generation.sh")
        with sftp.open(remote_run_script, "w") as f:
            f.write(run_script)
        sftp.chmod(remote_run_script, 0o755)
        print(f"uploaded run script -> {remote_run_script}")
    finally:
        sftp.close()


def build_run_script(args: argparse.Namespace) -> str:
    prompt = posixpath.join(args.run_dir, "qa_generation_prompt_v2.txt")
    output = posixpath.join(args.run_dir, "qwen36_27b_cot_qa_by_sample.jsonl")
    flat_output = posixpath.join(args.run_dir, "qwen36_27b_cot_qa_flat.jsonl")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.data_parallel:
        worker_blocks = []
        for shard, gpu in enumerate(gpus):
            shard_output = posixpath.join(args.run_dir, f"qwen36_27b_cot_qa_by_sample_shard{shard}.jsonl")
            shard_flat = posixpath.join(args.run_dir, f"qwen36_27b_cot_qa_flat_shard{shard}.jsonl")
            shard_log = posixpath.join(args.run_dir, f"worker_shard{shard}.log")
            worker_blocks.append(
                f"""echo "launch_worker shard={shard} gpu={gpu} started_at=$(date -Is)"
CUDA_VISIBLE_DEVICES={q(gpu)} {q(args.python)} tools/generate_qa_from_cot.py \\
  --input {q(args.input)} \\
  --prompt {q(prompt)} \\
  --output {q(shard_output)} \\
  --flat-output {q(shard_flat)} \\
  --model {q(args.model)} \\
  --cuda-visible-devices {q(gpu)} \\
  --dtype bfloat16 \\
  --device-map auto \\
  --attn-implementation {q(args.attn_implementation)} \\
  --max-memory {q(args.worker_max_memory)} \\
  --max-new-tokens {args.max_new_tokens} \\
  --min-new-tokens {args.min_new_tokens} \\
  --batch-size {args.batch_size} \\
  --retries {args.retries} \\
  {args.retry_sampling_flag} \\
  --retry-temperature {args.retry_temperature} \\
  --retry-top-p {args.retry_top_p} \\
  --model-class {q(args.model_class)} \\
  --shard-index {shard} \\
  --num-shards {len(gpus)} \\
  --resume > {q(shard_log)} 2>&1 &
pids+=($!)
"""
            )
        worker_script = "\n".join(worker_blocks)
        merge_script = f"""{q(args.python)} - <<'PY'
import glob
import json
from pathlib import Path

run_dir = Path({args.run_dir!r})
sample_out = run_dir / "qwen36_27b_cot_qa_by_sample.jsonl"
flat_out = run_dir / "qwen36_27b_cot_qa_flat.jsonl"

def merge(pattern, output, key):
    rows = []
    for path in sorted(run_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=key)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
    print(f"merged {{len(rows)}} rows -> {{output}}")

merge(
    "qwen36_27b_cot_qa_by_sample_shard*.jsonl",
    sample_out,
    lambda r: (int(r.get("source_index", 10**12)), str(r.get("sample_id", ""))),
)
merge(
    "qwen36_27b_cot_qa_flat_shard*.jsonl",
    flat_out,
    lambda r: (int(r.get("source_index", 10**12)), str(r.get("qa_id", ""))),
)
PY
"""
        generation_command = f"""pids=()
{worker_script}
rc=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
{merge_script}
exit "$rc"
"""
    else:
        generation_command = f"""{q(args.python)} tools/generate_qa_from_cot.py \\
  --input {q(args.input)} \\
  --prompt {q(prompt)} \\
  --output {q(output)} \\
  --flat-output {q(flat_output)} \\
  --model {q(args.model)} \\
  --cuda-visible-devices {q(args.gpus)} \\
  --dtype bfloat16 \\
  --device-map auto \\
  --attn-implementation {q(args.attn_implementation)} \\
  --max-memory {q(args.max_memory)} \\
  --max-new-tokens {args.max_new_tokens} \\
  --min-new-tokens {args.min_new_tokens} \\
  --batch-size {args.batch_size} \\
  --retries {args.retries} \\
  {args.retry_sampling_flag} \\
  --retry-temperature {args.retry_temperature} \\
  --retry-top-p {args.retry_top_p} \\
  --model-class {q(args.model_class)} \\
  --resume
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(args.remote_root)}
export CUDA_VISIBLE_DEVICES={q(args.gpus)}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "remote_root={args.remote_root}"
echo "input={args.input}"
echo "prompt={prompt}"
echo "output={output}"
echo "flat_output={flat_output}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
test -x {q(args.python)}
test -d {q(args.model)}
test -f {q(args.input)}
test -f {q(prompt)}
{q(args.python)} -m py_compile tools/generate_qa_from_cot.py rubric_rl/qwen_text.py
{generation_command}
echo "finished_at=$(date -Is)"
"""


def launch(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    run_script = posixpath.join(args.run_dir, "run_qa_generation.sh")
    nohup_log = posixpath.join(args.run_dir, "qa_generation.nohup.log")
    pid_file = posixpath.join(args.run_dir, "qa_generation.pid")
    cmd = f"""
set -euo pipefail
mkdir -p {q(args.run_dir)}
if pgrep -af '[t]ools/generate_qa_from_cot.py' >/tmp/cot_qa_existing.txt; then
  echo 'existing_generation_process:'
  cat /tmp/cot_qa_existing.txt
  exit 0
fi
nohup bash {q(run_script)} > {q(nohup_log)} 2>&1 &
pid=$!
echo "$pid" > {q(pid_file)}
echo "launched_pid=$pid"
echo "nohup_log={nohup_log}"
echo "pid_file={pid_file}"
"""
    rc, out, err = run_capture(client, cmd, timeout=60)
    print(out, end="")
    if err:
        print(err, file=sys.stderr, end="")
    if rc != 0:
        raise RuntimeError(f"launch failed rc={rc}")


def status(client: paramiko.SSHClient, args: argparse.Namespace, *, tail_lines: int = 80) -> None:
    log = posixpath.join(args.run_dir, "qa_generation.nohup.log")
    out = posixpath.join(args.run_dir, "qwen36_27b_cot_qa_by_sample.jsonl")
    flat = posixpath.join(args.run_dir, "qwen36_27b_cot_qa_flat.jsonl")
    pid = posixpath.join(args.run_dir, "qa_generation.pid")
    cmd = f"""
set +e
echo '--- process ---'
if test -f {q(pid)}; then
  p=$(cat {q(pid)})
  ps -fp "$p" || true
else
  pgrep -af '[t]ools/generate_qa_from_cot.py' || true
fi
echo '--- gpu ---'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
echo '--- counts ---'
for f in {q(out)} {q(flat)} {q(log)} {q(args.run_dir)}/qwen36_27b_cot_qa_by_sample_shard*.jsonl {q(args.run_dir)}/qwen36_27b_cot_qa_flat_shard*.jsonl {q(args.run_dir)}/worker_shard*.log; do
  if test -f "$f"; then
    printf '%s\\t' "$f"
    wc -l < "$f"
  else
    echo "$f MISSING"
  fi
done
echo '--- log tail ---'
if test -f {q(log)}; then tail -n {tail_lines} {q(log)}; fi
"""
    rc, out_text, err = run_capture(client, cmd, timeout=60)
    print(out_text, end="")
    if err:
        print(err, file=sys.stderr, end="")
    if rc != 0:
        raise RuntimeError(f"status failed rc={rc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-memory", default="0:75GiB,1:75GiB,2:75GiB,3:75GiB,cpu:160GiB")
    parser.add_argument("--worker-max-memory", default="0:75GiB,cpu:120GiB")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-do-sample", action="store_true")
    parser.add_argument("--retry-temperature", type=float, default=0.7)
    parser.add_argument("--retry-top-p", type=float, default=0.95)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.retry_sampling_flag = "--retry-do-sample" if args.retry_do_sample else ""
    client = connect(args)
    try:
        rc, out, err = run_capture(
            client,
            (
                f"date -Is; hostname; "
                f"test -d {q(args.remote_root)} && echo remote_root_ok; "
                f"test -f {q(args.input)} && wc -l {q(args.input)}; "
                f"test -d {q(args.model)} && echo model_ok"
            ),
            timeout=60,
        )
        print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")
        if rc != 0:
            raise RuntimeError(f"remote probe failed rc={rc}")

        if args.status_only:
            status(client, args)
            return 0
        if not args.skip_upload:
            upload_assets(client, args)
        if not args.skip_launch:
            launch(client, args)
            time.sleep(10)
        status(client, args)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
