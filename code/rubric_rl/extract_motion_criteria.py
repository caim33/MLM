#!/usr/bin/env python3
"""Offline criteria extraction for temporal-caption Rubric RL.

Input: JSONL rows with a ground-truth dense motion description.
Output: JSONL rows with normalized rubric criteria and stable IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rubric_rl.prompts import build_offline_messages
from rubric_rl.qwen_text import QwenTextGenerator, parse_json_object
from rubric_rl.reward import ensure_criteria_ids


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def row_id(row: Dict[str, Any], id_key: str, index: int) -> str:
    return str(row.get(id_key) or row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)


def get_gt(row: Dict[str, Any], gt_key: str) -> str:
    for key in [gt_key, "gt_description", "reference", "description", "answer", "solution"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_max_memory(value: Optional[str]) -> Optional[Dict[Any, str]]:
    if not value:
        return None
    parsed: Dict[Any, str] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        key, mem = chunk.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = mem.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gt-key", default="gt_description")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None, help='Example: "0:38GiB,1:38GiB,cpu:120GiB"')
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--include-timing", action="store_true", help="Write model load and per-row generation timings.")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output.parent.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    generator = QwenTextGenerator(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_memory=parse_max_memory(args.max_memory),
        model_class=args.model_class,
    )
    model_load_seconds = time.perf_counter() - load_started
    if args.include_timing:
        print(
            json.dumps({"event": "model_loaded", "model_load_seconds": model_load_seconds}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    with args.output.open("w", encoding="utf-8") as out_f:
        for index, row in enumerate(load_jsonl(args.input)):
            if args.limit is not None and index >= args.limit:
                break
            sid = row_id(row, args.id_key, index)
            gt = get_gt(row, args.gt_key)
            if not gt:
                out_f.write(json.dumps({"sample_id": sid, "error": "missing_gt_description"}, ensure_ascii=False) + "\n")
                continue
            generation_started = time.perf_counter()
            raw = generator.generate(build_offline_messages(gt), max_new_tokens=args.max_new_tokens)
            generation_seconds = time.perf_counter() - generation_started
            postprocess_started = time.perf_counter()
            parsed = parse_json_object(raw)
            if parsed is None:
                payload: Dict[str, Any] = {"sample_id": sid, "error": "parse_failed", "raw_response": raw}
            else:
                payload = {
                    "sample_id": sid,
                    "criteria": ensure_criteria_ids(parsed),
                }
                if args.keep_raw:
                    payload["raw_response"] = raw
            postprocess_seconds = time.perf_counter() - postprocess_started
            if args.include_timing:
                payload["timing"] = {
                    "model_load_seconds": model_load_seconds,
                    "generation_seconds": generation_seconds,
                    "postprocess_seconds": postprocess_seconds,
                    "end_to_end_row_seconds": generation_seconds + postprocess_seconds,
                }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
