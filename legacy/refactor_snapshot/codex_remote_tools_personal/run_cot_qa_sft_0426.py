#!/usr/bin/env python3
"""Launch 0426 Motion-R1 LoRA SFT on generated CoT-QA data.

The script reads the current Aistation SSH details from dev_env_connection.txt,
prepares a leakage-safe sample-level train/val split, launches LoRA SFT from
the 0426 checkpoint, then evaluates train/val MCQ accuracy with log-prob
scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import random
import re
import shlex
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parents[1]
CONNECTION_FILE = LOCAL_ROOT / "dev_env_connection.txt"

REMOTE_ROOT = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
QWEN_ROOT = "/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune"
RUN_DIR = posixpath.join(REMOTE_ROOT, "codex_runs", "cot_qa_sft_0426_20260726")
TOOLS_DIR = posixpath.join(REMOTE_ROOT, "codex_tools")
MODEL_0426 = posixpath.join(REMOTE_ROOT, "codex_models", "qwen3_vl_motion_checkpoint_0426")
MODEL_0426_FALLBACK = "/wangbenyou-sulongjie/Motion-r1/model/checkpoint_0426"
VQVAE = "/wangbenyou-sulongjie/Motion-r1/model/pretrained/VQVAE/net_best_fid.pth"
PYTHON = "/wangbenyou-sulongjie/anaconda3/envs/qwen3_vl/bin/python3.10"
TORCHRUN = "/wangbenyou-sulongjie/anaconda3/envs/qwen3_vl/bin/torchrun"

QA_FLAT = posixpath.join(
    REMOTE_ROOT,
    "codex_runs",
    "qa_gen_cot_20260724",
    "qwen36_27b_cot_prompt_v2",
    "qwen36_27b_cot_qa_flat.jsonl",
)
SOURCE_COT_CANDIDATES = [
    "/wangbenyou-sulongjie/qwen-vl-finetune/data/description_eval/motionx_desc_eval_random1000_after850_sft.jsonl",
    "/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/data/description_eval/motionx_desc_eval_random1000_after850_sft.jsonl",
]


def q(value: str) -> str:
    return shlex.quote(value)


def parse_connection_file(path: Path) -> Tuple[str, int, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    kv: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            kv[key.strip()] = value.strip()
    host = kv.get("host") or ""
    if not host:
        match = re.search(r'"sshIp"\s*:\s*"([^"]+)"', text) or re.search(r"ssh\s+\w+@([\d.]+)", text)
        if match:
            host = match.group(1)
    port_text = kv.get("port") or ""
    if not port_text:
        match = re.search(r'"sshPort"\s*:\s*"(\d+)"', text) or re.search(r"-p\s+(\d+)", text)
        if match:
            port_text = match.group(1)
    password = kv.get("password") or os.environ.get("CODEX_REMOTE_PASSWORD") or ""
    if not host or not port_text or not password:
        raise SystemExit(f"Missing host/port/password in {path}")
    return host, int(port_text), password


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    host, port, password = parse_connection_file(Path(args.connection_file))
    if args.host:
        host = args.host
    if args.port:
        port = args.port
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=port,
        username=args.user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str, *, timeout: int = 120, check: bool = True) -> str:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"remote command failed rc={rc}\nCMD:\n{command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    if err:
        out += "\nSTDERR:\n" + err
    return out


def sftp_put_text(client: paramiko.SSHClient, remote_path: str, text: str, mode: Optional[int] = None) -> None:
    sftp = client.open_sftp()
    parent = posixpath.dirname(remote_path)
    run(client, f"mkdir -p {q(parent)}")
    with sftp.file(remote_path, "w") as f:
        f.write(text)
    if mode is not None:
        sftp.chmod(remote_path, mode)
    sftp.close()


def sftp_put_file(client: paramiko.SSHClient, local_path: Path, remote_path: str, mode: Optional[int] = None) -> None:
    sftp = client.open_sftp()
    run(client, f"mkdir -p {q(posixpath.dirname(remote_path))}")
    sftp.put(str(local_path), remote_path)
    if mode is not None:
        sftp.chmod(remote_path, mode)
    sftp.close()


PREPARE_REMOTE = r'''
import json, random, os, re
from pathlib import Path
from collections import Counter

run_dir = Path(__RUN_DIR__)
flat_path = Path(__QA_FLAT__)
source_candidates = [Path(p) for p in __SOURCE_COT_CANDIDATES__]
qwen_root = Path(__QWEN_ROOT__)
seed = int(__SEED__)
train_sample_count = int(__TRAIN_SAMPLE_COUNT__)
balance_options = bool(__BALANCE_OPTIONS__)
LETTERS = ["A", "B", "C", "D"]

out_data = run_dir / "data"
sft_dir = out_data / "sft"
raw_dir = out_data / "raw_jsonl"
for p in (sft_dir, raw_dir):
    p.mkdir(parents=True, exist_ok=True)

source_path = next((p for p in source_candidates if p.exists()), None)
if source_path is None:
    raise FileNotFoundError(source_candidates)
if not flat_path.exists():
    raise FileNotFoundError(flat_path)

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def first_existing(paths):
    for value in paths:
        if not value:
            continue
        p = Path(str(value))
        candidates = (
            [p]
            if p.is_absolute()
            else [
                qwen_root / p,
                qwen_root / "data" / "description_eval" / "assets_random1000_after850" / p,
                source_path.parent / p,
            ]
        )
        for c in candidates:
            if c.exists():
                return str(c)
    return str(paths[0]) if paths and paths[0] else None

def first_glob(patterns):
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else qwen_root.glob(pattern))
        for match in matches:
            if match.exists():
                return str(match)
    return None

def resolve_video(row, sid):
    video = first_existing([row.get("source_video"), row.get("video")])
    if video and Path(video).exists():
        return video
    video_file = row.get("video_file") or (str(row.get("video_name")) + ".mp4" if row.get("video_name") else None)
    if video_file:
        video = first_glob([
            f"data/description_eval/assets_random1000_after850/Videos/Motion-X/*/{video_file}",
            f"data/benchmark/video/description/{sid}.mp4",
        ])
        if video:
            return video
    return video

def resolve_motion(row, sid):
    motion = first_existing([row.get("source_motion"), row.get("motion")])
    if motion and Path(motion).exists():
        return motion
    return first_glob([
        f"data/description_eval/assets_random1000_after850/npy_data/Motion-X/{sid}.npy",
        f"data/benchmark/motion/description/{sid}.npy",
        f"data/grpo/motionx_step2_selected_qa_assets/motions/{sid}.npy",
    ])

source_by_id = {}
source_order = []
for row in read_jsonl(source_path):
    sid = str(row.get("sample_id"))
    video = resolve_video(row, sid)
    motion = resolve_motion(row, sid)
    if not video or not motion:
        raise RuntimeError(f"missing media fields for sample_id={sid}")
    source_by_id[sid] = {"video": video, "motion": motion, "source_row": row}
    source_order.append(sid)

def clean_text(text):
    return str(text).replace("\ufffd", " ").strip()

def format_options(options):
    return "\n".join(f"{letter}. {clean_text(options.get(letter, ''))}" for letter in LETTERS)

def prompt_for(row):
    return (
        "<motion_start><motion><motion_end>\n"
        "You are given video evidence and motion-based human pose evidence for a human action multiple-choice question.\n"
        "Answer with exactly one final option. Do not explain.\n"
        "Return it only in the form <answer>A</answer>, <answer>B</answer>, <answer>C</answer>, or <answer>D</answer>.\n\n"
        f"Question: {clean_text(row['question'])}\n\n"
        "Choose exactly one option:\n"
        f"{format_options(row['options'])}"
    )

def refresh_prompt_fields(row):
    row["prompt"] = prompt_for(row)
    row["answer"] = f"<answer>{row['answer_key']}</answer>"
    row["solution"] = row["answer"]
    row["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": row["video"]},
                {"type": "text", "text": row["prompt"]},
            ],
        }
    ]
    return row

def move_answer_to_letter(row, target_letter, rng):
    old_answer = str(row["answer_key"]).upper()
    if old_answer not in LETTERS or target_letter not in LETTERS:
        raise RuntimeError(f"bad answer remap old={old_answer!r} target={target_letter!r}")
    old_options = {letter: clean_text(row["options"][letter]) for letter in LETTERS}
    wrong_letters = [letter for letter in LETTERS if letter != old_answer]
    rng.shuffle(wrong_letters)
    new_options = {}
    wrong_iter = iter(wrong_letters)
    for letter in LETTERS:
        source_letter = old_answer if letter == target_letter else next(wrong_iter)
        new_options[letter] = old_options[source_letter]
    out = dict(row)
    out["original_answer_key"] = old_answer
    out["original_options"] = old_options
    out["answer_key"] = target_letter
    out["options"] = new_options
    out["option_shuffle_mode"] = "balanced_answer_position"
    return refresh_prompt_fields(out)

def balance_answer_positions(items, split_name):
    if not balance_options:
        return items
    targets = [LETTERS[i % len(LETTERS)] for i in range(len(items))]
    rng = random.Random(seed + (10000 if split_name == "train" else 20000))
    rng.shuffle(targets)
    balanced = []
    for idx, (row, target) in enumerate(zip(items, targets)):
        row_rng = random.Random(seed * 1000003 + idx + (30000 if split_name == "train" else 40000))
        balanced.append(move_answer_to_letter(row, target, row_rng))
    counts = Counter(row["answer_key"] for row in balanced)
    spread = max(counts.values()) - min(counts.values())
    if spread > 1:
        raise RuntimeError(f"unbalanced {split_name} answer counts after shuffle: {dict(counts)}")
    return balanced

def make_records():
    rows = []
    missing = []
    for row in read_jsonl(flat_path):
        sid = str(row.get("sample_id"))
        media = source_by_id.get(sid)
        if not media:
            missing.append(sid)
            continue
        answer_key = str(row.get("answer_key", "")).strip().upper()
        if answer_key not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"bad answer key sample_id={sid}: {answer_key!r}")
        prompt = prompt_for(row)
        qa_index = int(row.get("qa_index", len(rows) % 5))
        qid = f"{sid}_qa{qa_index:02d}_{row.get('question_type')}"
        rec = dict(row)
        rec.update(
            {
                "id": qid,
                "sample_id": qid,
                "source_sample_id": sid,
                "group_id": sid,
                "branch": "vm",
                "qa_index": qa_index,
                "video": media["video"],
                "motion": media["motion"],
                "prompt": prompt,
                "answer": f"<answer>{answer_key}</answer>",
                "solution": f"<answer>{answer_key}</answer>",
                "messages": [{"role": "user", "content": [{"type": "video", "video": media["video"]}, {"type": "text", "text": prompt}]}],
            }
        )
        rows.append(rec)
    if missing:
        raise RuntimeError(f"missing source media for {len(missing)} rows, examples={missing[:10]}")
    return rows

rows = make_records()
by_source = {}
for row in rows:
    by_source.setdefault(row["source_sample_id"], []).append(row)
if len(rows) != 5000:
    raise RuntimeError(f"expected 5000 flat QA rows, got {len(rows)}")
bad_group_sizes = {k: len(v) for k, v in by_source.items() if len(v) != 5}
if bad_group_sizes:
    raise RuntimeError(f"expected 5 QA per source sample; bad examples={list(bad_group_sizes.items())[:10]}")

sample_ids = sorted(by_source)
rng = random.Random(seed)
rng.shuffle(sample_ids)
train_source_ids = set(sample_ids[:train_sample_count])
val_source_ids = set(sample_ids[train_sample_count:])
train = [row for sid in sample_ids if sid in train_source_ids for row in sorted(by_source[sid], key=lambda x: x["qa_index"])]
val = [row for sid in sample_ids if sid in val_source_ids for row in sorted(by_source[sid], key=lambda x: x["qa_index"])]
train = balance_answer_positions(train, "train")
val = balance_answer_positions(val, "val")

def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)

def sft_item(row):
    return {
        "id": row["sample_id"],
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "group_id": row["group_id"],
        "branch": "vm",
        "video": row["video"],
        "motion": row["motion"],
        "question_type": row.get("question_type"),
        "difficulty": row.get("difficulty"),
        "answer_key": row.get("answer_key"),
        "conversations": [
            {"from": "human", "value": ("<video>\n" + row["prompt"]).strip()},
            {"from": "gpt", "value": row["answer"]},
        ],
    }

def write_sft(path, items):
    payload = [sft_item(row) for row in items]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(payload)

counts = {
    "train_raw": write_jsonl(raw_dir / "train_raw.jsonl", train),
    "val_raw": write_jsonl(raw_dir / "val_raw.jsonl", val),
    "sft_train_motion": write_sft(sft_dir / "motionx_qa_train_vm.json", train),
    "sft_val_motion": write_sft(sft_dir / "motionx_qa_val_vm.json", val),
}
# Empty video-only files keep the generic Codex SFT launcher registry happy.
(sft_dir / "motionx_qa_train_v.json").write_text("[]", encoding="utf-8")
(sft_dir / "motionx_qa_val_v.json").write_text("[]", encoding="utf-8")
(sft_dir / "motionx_qa_train_v_smoke32.json").write_text("[]", encoding="utf-8")
(sft_dir / "motionx_qa_train_vm_smoke32.json").write_text(
    json.dumps([sft_item(row) for row in train[:32]], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

missing_media = []
for row in rows:
    for key in ("video", "motion"):
        if not Path(row[key]).exists():
            missing_media.append({"sample_id": row["sample_id"], "key": key, "path": row[key]})

manifest = {
    "run_dir": str(run_dir),
    "source_cot": str(source_path),
    "qa_flat": str(flat_path),
    "seed": seed,
    "balance_options": balance_options,
    "split": "sample_id_grouped",
    "train_source_samples": len(train_source_ids),
    "val_source_samples": len(val_source_ids),
    "counts": counts,
    "train_question_type_counts": dict(Counter(row.get("question_type") for row in train)),
    "val_question_type_counts": dict(Counter(row.get("question_type") for row in val)),
    "train_answer_counts": dict(Counter(row.get("answer_key") for row in train)),
    "val_answer_counts": dict(Counter(row.get("answer_key") for row in val)),
    "train_original_answer_counts": dict(Counter(row.get("original_answer_key", row.get("answer_key")) for row in train)),
    "val_original_answer_counts": dict(Counter(row.get("original_answer_key", row.get("answer_key")) for row in val)),
    "train_difficulty_counts": dict(Counter(row.get("difficulty") for row in train)),
    "val_difficulty_counts": dict(Counter(row.get("difficulty") for row in val)),
    "train_val_source_overlap": sorted(train_source_ids & val_source_ids)[:10],
    "missing_media_count": len(missing_media),
    "missing_media_preview": missing_media[:10],
    "paths": {
        "sft_dir": str(sft_dir),
        "train_raw": str(raw_dir / "train_raw.jsonl"),
        "val_raw": str(raw_dir / "val_raw.jsonl"),
        "train_sft": str(sft_dir / "motionx_qa_train_vm.json"),
        "val_sft": str(sft_dir / "motionx_qa_val_vm.json"),
    },
}
(out_data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
if missing_media:
    raise SystemExit("missing media")
'''


def prepare_data(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    code = (
        PREPARE_REMOTE.replace("__RUN_DIR__", json.dumps(args.run_dir))
        .replace("__QA_FLAT__", json.dumps(args.qa_flat))
        .replace("__SOURCE_COT_CANDIDATES__", json.dumps(args.source_cot_candidates))
        .replace("__QWEN_ROOT__", json.dumps(args.qwen_root))
        .replace("__SEED__", json.dumps(str(args.seed)))
        .replace("__TRAIN_SAMPLE_COUNT__", json.dumps(str(args.train_source_samples)))
        .replace("__BALANCE_OPTIONS__", repr(bool(args.balance_options)))
    )
    remote_prepare = posixpath.join(args.run_dir, "prepare_cot_qa_sft_data.py")
    sftp_put_text(client, remote_prepare, code, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print(run(client, f"{q(args.python)} {q(remote_prepare)}", timeout=600))


def upload_tools(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    tool_files = [
        LOCAL_ROOT / "codex_remote_tools" / "run_lora_sft_codex.py",
        LOCAL_ROOT / "codex_remote_tools" / "eval_motionr1_lora_mcq_score.py",
    ]
    for path in tool_files:
        remote_path = posixpath.join(args.tools_dir, path.name)
        sftp_put_file(client, path, remote_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        print(f"uploaded {path.name} -> {remote_path}")


def train_and_eval_script(args: argparse.Namespace) -> str:
    port = random.randint(21000, 29999)
    model_path = args.model
    cuda_visible_devices = args.cuda_visible_devices
    nproc_per_node = args.nproc_per_node
    if nproc_per_node is None:
        nproc_per_node = len([part for part in cuda_visible_devices.split(",") if part.strip()])
    return f'''#!/usr/bin/env bash
set -euo pipefail

RUN_DIR={q(args.run_dir)}
QWEN_ROOT={q(args.qwen_root)}
TOOLS_DIR={q(args.tools_dir)}
PY={q(args.python)}
TORCHRUN={q(args.torchrun)}
MODEL={q(model_path)}
VQVAE={q(args.vqvae)}
SFT_DIR="$RUN_DIR/data/sft"
OUT_DIR="$RUN_DIR/output/0426_cotqa_lora_vm"
EVAL_DIR="$RUN_DIR/eval"
LOG_DIR="$RUN_DIR/logs"
STATUS="$RUN_DIR/status.json"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$EVAL_DIR" "$LOG_DIR"
cd "$QWEN_ROOT"

export PYTHONPATH="$QWEN_ROOT:$QWEN_ROOT/qwenvl/train:$TOOLS_DIR:${{PYTHONPATH:-}}"
export CODEX_SFT_DIR="$SFT_DIR"
export CODEX_ATTN_IMPLEMENTATION=sdpa
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES={q(cuda_visible_devices)}

python_status() {{
  "$PY" - "$@" <<'PY'
import json, sys, time, pathlib
path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
data = {{}}
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {{}}
data[key] = value
data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}}

python_status "$STATUS" state training
trap 'python_status "$STATUS" state failed' ERR

"$TORCHRUN" --nproc_per_node={nproc_per_node} --master_addr=127.0.0.1 --master_port={port} \\
  "$TOOLS_DIR/run_lora_sft_codex.py" \\
  --model_name_or_path "$MODEL" \\
  --motion_vqvae_path "$VQVAE" \\
  --motion_dataname t2m \\
  --motion_quantizer ema \\
  --vqvae_nb_code 512 \\
  --vqvae_code_dim 512 \\
  --vqvae_output_emb_width 512 \\
  --vqvae_down_t 2 \\
  --vqvae_stride_t 2 \\
  --vqvae_width 512 \\
  --vqvae_depth 3 \\
  --vqvae_dilation_growth_rate 3 \\
  --vqvae_activation relu \\
  --vqvae_norm none \\
  --motion_length_divisor 4 \\
  --dataset_use codex_motionx_qa_train_vm \\
  --eval_dataset_use codex_motionx_qa_val_vm \\
  --data_flatten True \\
  --tune_mm_vision False \\
  --tune_mm_mlp False \\
  --tune_mm_llm True \\
  --tune_mm_motion False \\
  --bf16 \\
  --output_dir "$OUT_DIR" \\
  --num_train_epochs 1 \\
  --per_device_train_batch_size 1 \\
  --per_device_eval_batch_size 1 \\
  --gradient_accumulation_steps 1 \\
  --max_pixels 50176 \\
  --min_pixels 784 \\
  --eval_strategy epoch \\
  --save_strategy steps \\
  --save_steps 250 \\
  --save_total_limit 2 \\
  --learning_rate 2e-4 \\
  --weight_decay 0.01 \\
  --warmup_ratio 0.03 \\
  --max_grad_norm 1 \\
  --lr_scheduler_type cosine \\
  --logging_steps 10 \\
  --model_max_length 4096 \\
  --gradient_checkpointing False \\
  --dataloader_num_workers 4 \\
  --run_name cotqa-0426-lora-vm-20260726 \\
  --report_to none \\
  --lora_r 64 \\
  --lora_alpha 128 \\
  --lora_dropout 0.05 \\
  --lora_bias none \\
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
  --lora_modules_to_save visual,visual.merger,motion_encoder,motion_embed,motion_proj \\
  --lora_use_dora False \\
  2>&1 | tee "$LOG_DIR/train.log"

python_status "$STATUS" state evaluating_val
export CUDA_VISIBLE_DEVICES=0
"$PY" "$TOOLS_DIR/eval_motionr1_lora_mcq_score.py" \\
  --model "$MODEL" \\
  --adapter "$OUT_DIR" \\
  --processor "$OUT_DIR" \\
  --dataset "$RUN_DIR/data/raw_jsonl/val_raw.jsonl" \\
  --output "$EVAL_DIR/val_acc.jsonl" \\
  --branch vm \\
  --prompt_mode short \\
  --device cuda:0 \\
  --attn_implementation sdpa \\
  2>&1 | tee "$LOG_DIR/eval_val.log"

python_status "$STATUS" state evaluating_train
"$PY" "$TOOLS_DIR/eval_motionr1_lora_mcq_score.py" \\
  --model "$MODEL" \\
  --adapter "$OUT_DIR" \\
  --processor "$OUT_DIR" \\
  --dataset "$RUN_DIR/data/raw_jsonl/train_raw.jsonl" \\
  --output "$EVAL_DIR/train_acc.jsonl" \\
  --branch vm \\
  --prompt_mode short \\
  --device cuda:0 \\
  --attn_implementation sdpa \\
  2>&1 | tee "$LOG_DIR/eval_train.log"

python_status "$STATUS" state completed
'''


def launch(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    script_path = posixpath.join(args.run_dir, "run_train_eval.sh")
    nohup_path = posixpath.join(args.run_dir, "run_train_eval.nohup.log")
    pid_path = posixpath.join(args.run_dir, "run_train_eval.pid")
    sftp_put_text(client, script_path, train_and_eval_script(args), stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    inner = (
        f"cd {q(args.run_dir)} || exit 1; "
        f"nohup {q(script_path)} > {q(nohup_path)} 2>&1 < /dev/null & "
        f"pid=$!; echo $pid > {q(pid_path)}; echo $pid; disown || true"
    )
    cmd = f"bash -lc {q(inner)}"
    print("launched_pid=" + run(client, cmd).strip())
    print(f"nohup_log={nohup_path}")


def status(client: paramiko.SSHClient, args: argparse.Namespace) -> None:
    cmd = f'''
set +e
echo "run_dir={args.run_dir}"
echo "--- status ---"
test -f {q(posixpath.join(args.run_dir, "status.json"))} && cat {q(posixpath.join(args.run_dir, "status.json"))} || true
echo
echo "--- process ---"
ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_train_eval|run_lora_sft_codex|eval_motionr1_lora_mcq_score|torchrun' | grep -v grep || true
echo "--- gpu ---"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true
echo "--- files ---"
for f in \\
  {q(posixpath.join(args.run_dir, "data", "manifest.json"))} \\
  {q(posixpath.join(args.run_dir, "output", "0426_cotqa_lora_vm", "adapter_model.safetensors"))} \\
  {q(posixpath.join(args.run_dir, "eval", "val_acc.summary.json"))} \\
  {q(posixpath.join(args.run_dir, "eval", "train_acc.summary.json"))}; do
  if [ -f "$f" ]; then echo "$f"; ls -lh "$f"; fi
done
echo "--- train tail ---"
tail -n 40 {q(posixpath.join(args.run_dir, "logs", "train.log"))} 2>/dev/null || true
echo "--- val summary ---"
cat {q(posixpath.join(args.run_dir, "eval", "val_acc.summary.json"))} 2>/dev/null || true
echo "--- train summary ---"
cat {q(posixpath.join(args.run_dir, "eval", "train_acc.summary.json"))} 2>/dev/null || true
'''
    print(run(client, cmd, check=False, timeout=120))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-file", default=str(CONNECTION_FILE))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default="root")
    parser.add_argument("--run-dir", default=RUN_DIR)
    parser.add_argument("--tools-dir", default=TOOLS_DIR)
    parser.add_argument("--qwen-root", default=QWEN_ROOT)
    parser.add_argument("--qa-flat", default=QA_FLAT)
    parser.add_argument("--source-cot-candidates", nargs="*", default=SOURCE_COT_CANDIDATES)
    parser.add_argument("--model", default=MODEL_0426)
    parser.add_argument("--vqvae", default=VQVAE)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--torchrun", default=TORCHRUN)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train-source-samples", type=int, default=900)
    parser.add_argument("--balance-options", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.prepare = args.launch = True
    if not (args.prepare or args.launch or args.status):
        args.status = True

    client = connect(args)
    try:
        if args.model == MODEL_0426:
            out = run(
                client,
                f"if [ -e {q(MODEL_0426)} ]; then echo {q(MODEL_0426)}; "
                f"elif [ -e {q(MODEL_0426_FALLBACK)} ]; then echo {q(MODEL_0426_FALLBACK)}; "
                "else exit 3; fi",
            ).strip()
            args.model = out
        upload_tools(client, args)
        if args.prepare:
            prepare_data(client, args)
        if args.launch:
            launch(client, args)
        if args.status:
            status(client, args)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
