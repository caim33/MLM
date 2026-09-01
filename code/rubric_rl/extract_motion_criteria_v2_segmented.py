#!/usr/bin/env python3
"""Segment-wise offline Qwen criteria extraction for V2 motion rubrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rubric_rl.prompts_v2 import build_offline_global_messages, build_offline_segment_messages
from rubric_rl.qwen_text import QwenTextGenerator, parse_json_object
from rubric_rl.reward_v2 import ensure_criteria_ids


SEGMENT_RE = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?)\]\s*"
    r"(?:cot_type:\s*(?P<cot_type>\w+)\s*)?"
    r"think:\s*(?P<think>.*?)\s*"
    r"answer:\s*(?P<answer>.*?)(?=\s*\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\]\s*(?:cot_type:\s*\w+\s*)?think:|\s*$)",
    re.DOTALL,
)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def normalize_time_range(value: str) -> str:
    nums = re.findall(r"\d+(?:\.\d+)?", value)
    if len(nums) >= 2:
        return f"{nums[0]}-{nums[1]}"
    return value.strip()


def parse_report(text: str) -> Dict[str, Any]:
    summary = ""
    final_answer = ""
    per_segment = text
    if "sample_summary:" in text and "per_segment:" in text:
        summary = text.split("sample_summary:", 1)[1].split("per_segment:", 1)[0].strip()
        per_segment = text.split("per_segment:", 1)[1]
    if "final_answer:" in per_segment:
        per_segment, final_answer = per_segment.rsplit("final_answer:", 1)
        final_answer = final_answer.strip()

    segments: List[Dict[str, str]] = []
    for match in SEGMENT_RE.finditer(per_segment.strip()):
        segments.append(
            {
                "time": normalize_time_range(match.group("time")),
                "think": " ".join(match.group("think").split()),
                "answer": " ".join(match.group("answer").split()),
            }
        )
    return {"summary": summary, "final_answer": final_answer, "segments": segments}


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


def row_id(row: Dict[str, Any], id_key: str, index: int) -> str:
    return str(row.get(id_key) or row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)


def compact_timeline(segments: List[Dict[str, str]]) -> str:
    lines = []
    for seg in segments:
        answer = seg["answer"]
        if len(answer) > 280:
            answer = answer[:277].rstrip() + "..."
        lines.append(f"[{seg['time']}] {answer}")
    return "\n".join(lines)


def generate_json(
    generator: QwenTextGenerator,
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    retry_tokens: int,
) -> tuple[Optional[Dict[str, Any]], str]:
    raw = generator.generate(messages, max_new_tokens=max_new_tokens)
    parsed = parse_json_object(raw)
    if parsed is not None:
        return parsed, raw

    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "Your previous output was not valid JSON or was too long. "
                "Return a shorter valid JSON object only, with concise criteria and evidence."
            ),
        }
    ]
    retry_raw = generator.generate(retry_messages, max_new_tokens=retry_tokens)
    parsed = parse_json_object(retry_raw)
    return parsed, raw + "\n\n---RETRY---\n\n" + retry_raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gt-key", default="gt_description")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--global-max-new-tokens", type=int, default=1200)
    parser.add_argument("--segment-max-new-tokens", type=int, default=1400)
    parser.add_argument("--retry-max-new-tokens", type=int, default=900)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    parser.error(
        "this legacy segmented generator is disabled because its fragment schema "
        "is not the frozen Motion Rubric V2 schema; use "
        "python -m rubric_rl.extract_motion_criteria_v2"
    )

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generator = QwenTextGenerator(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_memory=parse_max_memory(args.max_memory),
        model_class=args.model_class,
    )

    with args.output.open("w", encoding="utf-8") as out_f:
        for index, row in enumerate(load_jsonl(args.input)):
            if args.limit is not None and index >= args.limit:
                break
            sid = row_id(row, args.id_key, index)
            gt = row.get(args.gt_key)
            if not isinstance(gt, str) or not gt.strip():
                payload: Dict[str, Any] = {"sample_id": sid, "error": "missing_gt"}
                out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                continue

            report = parse_report(gt)
            if not report["segments"]:
                payload = {"sample_id": sid, "error": "no_segments_parsed"}
                out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                continue

            raw_outputs: Dict[str, Any] = {}
            global_obj, global_raw = generate_json(
                generator,
                build_offline_global_messages(
                    summary=report["summary"],
                    final_answer=report["final_answer"],
                    segment_timeline=compact_timeline(report["segments"]),
                ),
                max_new_tokens=args.global_max_new_tokens,
                retry_tokens=args.retry_max_new_tokens,
            )
            raw_outputs["global"] = global_raw
            if global_obj is None:
                payload = {"sample_id": sid, "error": "global_parse_failed"}
                if args.keep_raw:
                    payload["raw_outputs"] = raw_outputs
                out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            segment_objs = []
            segment_errors = []
            raw_outputs["segments"] = []
            for seg in report["segments"]:
                seg_obj, seg_raw = generate_json(
                    generator,
                    build_offline_segment_messages(
                        summary=report["summary"],
                        time_range=seg["time"],
                        think=seg["think"],
                        answer=seg["answer"],
                    ),
                    max_new_tokens=args.segment_max_new_tokens,
                    retry_tokens=args.retry_max_new_tokens,
                )
                raw_outputs["segments"].append({"time": seg["time"], "raw": seg_raw})
                if seg_obj is None:
                    segment_errors.append(seg["time"])
                else:
                    seg_obj["time"] = str(seg_obj.get("time") or seg["time"])
                    segment_objs.append(seg_obj)

            if segment_errors:
                payload = {"sample_id": sid, "error": "segment_parse_failed", "segments": segment_errors}
                if args.keep_raw:
                    payload["raw_outputs"] = raw_outputs
            else:
                criteria = {
                    "mode": "source_aware_reasoning_motion_rubric_v2",
                    "global_activity": global_obj.get("global_activity", []),
                    "segments": segment_objs,
                    "temporal_phases": global_obj.get("temporal_phases", []),
                    "negative_criteria": global_obj.get("negative_criteria", []),
                }
                payload = {"sample_id": sid, "criteria": ensure_criteria_ids(criteria)}
                if args.keep_raw:
                    payload["raw_outputs"] = raw_outputs
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
