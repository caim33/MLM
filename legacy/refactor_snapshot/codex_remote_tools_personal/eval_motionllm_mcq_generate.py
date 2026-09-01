#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SCRIPT_DIR = ROOT / "MVBench_Eval" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_motionllm_motionx import load_motionllm, make_infer_fn


def load_jsonl(path: Path, limit: Optional[int], branch: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if branch != "all" and str(obj.get("branch", "")).lower() != branch:
                continue
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_option_letter(text: Any) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text, flags=re.I)
    return match.group(1).upper() if match else None


def extract_original_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(p for p in parts if p)


def make_short_prompt(text: str) -> str:
    text = text.replace("<motion_start><motion><motion_end>", "").replace("<video>", "").strip()
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text
    return (
        "Analyze the video carefully and answer the multiple-choice question.\n"
        "Do not explain. Return exactly one final option in the form <answer>A</answer>, "
        "<answer>B</answer>, <answer>C</answer>, or <answer>D</answer>.\n\n"
        f"{qa}"
    )


def build_prompt(rec: Dict[str, Any], prompt_mode: str) -> str:
    text = extract_original_text(rec.get("messages") or [])
    if prompt_mode == "short":
        return make_short_prompt(text)
    return text.replace("<motion_start><motion><motion_end>", "").replace("<video>", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    infer_fn = make_infer_fn(load_motionllm())

    correct = 0
    parsed = 0
    errors = 0
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(rows):
            gold = extract_option_letter(rec.get("answer") or rec.get("solution"))
            raw = ""
            pred = None
            err = None
            try:
                raw = infer_fn(str(rec.get("video")), build_prompt(rec, args.prompt_mode), "")
                pred = extract_option_letter(raw)
            except Exception as exc:
                errors += 1
                err = repr(exc)
            if pred:
                parsed += 1
            ok = pred == gold if gold else False
            correct += int(ok)
            f.write(json.dumps({
                "index": idx,
                "sample_id": rec.get("sample_id"),
                "group_id": rec.get("group_id"),
                "branch": rec.get("branch"),
                "gold": gold,
                "prediction": pred,
                "correct": ok,
                "raw_output": raw,
                "error": err,
            }, ensure_ascii=False) + "\n")
            f.flush()

    total = len(rows)
    summary = {
        "model": "MotionLLM",
        "dataset": str(args.dataset),
        "output": str(args.output),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
        "errors": errors,
        "elapsed_seconds": time.time() - start,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
