#!/usr/bin/env python3
"""Rerun selected V2 segment criteria with an ultra-compact prompt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from rubric_rl.extract_motion_criteria_v2_fast_segmented import TRUST, load_jsonl, parse_report, row_id
from rubric_rl.qwen_text import QwenTextGenerator, parse_json_object


SYSTEM = "Return only one compact valid JSON object. No markdown. No explanation."


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


def segment_messages(summary: str, segment: Dict[str, str]) -> List[Dict[str, Any]]:
    user = f"""{TRUST}

Summary: {summary}
time: {segment['time']}
think: {segment['think']}
answer: {segment['answer']}

Extract this segment only. Keep it short. Do not output rejected_claims except as an empty list.
Required counts:
- basic_action_facts: exactly 2
- body_configuration: exactly 1
- numeric_kinematics: exactly 3
- laterality: 1 if left/right appears, else []
- camera_relative_orientation: 1 if camera/screen orientation appears, else []
- reasoning_criteria: exactly 2
- rejected_claims: []

Numeric tolerances: degrees strict=10 loose=20; m strict=0.05 loose=0.10; s strict=0.20 loose=0.50.
Use numeric ranges from the answer. For single values use [value, value].

JSON schema:
{{
  "time": "{segment['time']}",
  "basic_action_facts": [{{"criterion": "...", "source": "motion | video+motion"}}],
  "body_configuration": [{{"criterion": "...", "source": "motion"}}],
  "numeric_kinematics": [
    {{"criterion": "...", "quantity": "...", "body_part": "...", "target_range": [0, 0], "unit": "degrees | m | s", "strict_tolerance": 10, "loose_tolerance": 20, "source": "motion"}}
  ],
  "laterality": [{{"criterion": "...", "source": "motion"}}],
  "camera_relative_orientation": [{{"criterion": "...", "source": "motion"}}],
  "reasoning_criteria": [
    {{"criterion": "...", "type": "source_separation | conflict_detection | trust_hierarchy_application | numeric_evidence_use | reasoning_answer_consistency", "source": "think"}}
  ],
  "rejected_claims": []
}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate_json(generator: QwenTextGenerator, messages: List[Dict[str, Any]], max_new_tokens: int) -> tuple[Optional[Dict[str, Any]], str]:
    raw = generator.generate(messages, max_new_tokens=max_new_tokens)
    parsed = parse_json_object(raw)
    if parsed is not None:
        return parsed, raw
    retry = [dict(m) for m in messages]
    retry[-1]["content"] += "\n\nReturn an even shorter valid JSON object. Use no evidence and keep rejected_claims as []."
    retry_raw = generator.generate(retry, max_new_tokens=max_new_tokens)
    return parse_json_object(retry_raw), raw + "\n\n---RETRY---\n\n" + retry_raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gt-key", default="gt_description")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--times", required=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=850)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    parser.error(
        "this historical fragment-repair CLI is disabled: segment fragments "
        "are not publishable Motion Rubric V2 artifacts; rerun the strict full extractor"
    )

    wanted: Set[str] = {part.strip() for part in args.times.split(",") if part.strip()}
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("loading_model", file=sys.stderr, flush=True)
    generator = QwenTextGenerator(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_memory=parse_max_memory(args.max_memory),
        model_class=args.model_class,
    )
    print("model_loaded", file=sys.stderr, flush=True)

    with args.output.open("w", encoding="utf-8") as out_f:
        for index, row in enumerate(load_jsonl(args.input)):
            sid = row_id(row, args.id_key, index)
            report = parse_report(str(row.get(args.gt_key) or ""))
            parsed_segments = []
            errors = []
            raw_outputs = []
            for segment in report["segments"]:
                if segment["time"] not in wanted:
                    continue
                print(f"segment_begin {sid} {segment['time']}", file=sys.stderr, flush=True)
                parsed, raw = generate_json(generator, segment_messages(report["summary"], segment), args.max_new_tokens)
                raw_outputs.append({"time": segment["time"], "raw": raw})
                if parsed is None:
                    errors.append(segment["time"])
                else:
                    parsed["time"] = str(parsed.get("time") or segment["time"])
                    parsed_segments.append(parsed)
            payload: Dict[str, Any] = {"sample_id": sid, "segments": parsed_segments}
            if errors:
                payload["error"] = "segment_parse_failed"
                payload["failed_segments"] = errors
            if args.keep_raw:
                payload["raw_outputs"] = raw_outputs
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
