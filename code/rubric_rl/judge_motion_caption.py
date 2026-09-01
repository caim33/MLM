#!/usr/bin/env python3
"""Online-style Qwen judge for temporal-caption Rubric RL.

This script is useful for batch testing the same reward used during GRPO.
For training, reuse rubric_rl.reward.compute_reward after obtaining a judge
JSON object from your local Qwen service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rubric_rl.prompts import build_online_messages
from rubric_rl.qwen_text import QwenTextGenerator, parse_json_object
from rubric_rl.reward import compute_reward, ensure_criteria_ids


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def load_criteria(path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(load_jsonl(path)):
        sid = str(row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)
        criteria = row.get("criteria") if isinstance(row.get("criteria"), dict) else row
        by_id[sid] = ensure_criteria_ids(criteria)
    return by_id


def row_id(row: Dict[str, Any], id_key: str, index: int) -> str:
    return str(row.get(id_key) or row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)


def get_candidate(row: Dict[str, Any], candidate_key: str) -> str:
    for key in [candidate_key, "candidate", "prediction", "final_answer", "answer_text"]:
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
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-key", default="candidate")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None, help='Example: "0:38GiB,1:38GiB,cpu:120GiB"')
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--include-timing", action="store_true", help="Write model load and per-row judge timings.")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    criteria_by_id = load_criteria(args.criteria)

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
        for index, row in enumerate(load_jsonl(args.candidates)):
            if args.limit is not None and index >= args.limit:
                break
            sid = row_id(row, args.id_key, index)
            criteria = criteria_by_id.get(sid)
            candidate = get_candidate(row, args.candidate_key)
            if not criteria:
                payload: Dict[str, Any] = {"sample_id": sid, "error": "missing_criteria"}
            elif not candidate:
                payload = {"sample_id": sid, "error": "missing_candidate"}
            else:
                generation_started = time.perf_counter()
                raw = generator.generate(
                    build_online_messages(criteria, candidate),
                    max_new_tokens=args.max_new_tokens,
                )
                generation_seconds = time.perf_counter() - generation_started
                postprocess_started = time.perf_counter()
                judgment = parse_json_object(raw)
                if judgment is None:
                    payload = {"sample_id": sid, "error": "parse_failed", "raw_response": raw}
                else:
                    payload = {
                        "sample_id": sid,
                        "judgment": judgment,
                        "reward": compute_reward(criteria, judgment),
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
