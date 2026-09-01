#!/usr/bin/env python3
"""Fast segment-wise Qwen criteria extraction with compact JSON prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rubric_rl.qwen_text import QwenTextGenerator, parse_json_object
from rubric_rl.reward_v2 import ensure_criteria_ids


SEGMENT_RE = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?)\]\s*"
    r"(?:cot_type:\s*(?P<cot_type>\w+)\s*)?"
    r"think:\s*(?P<think>.*?)\s*"
    r"answer:\s*(?P<answer>.*?)(?=\s*\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\]\s*(?:cot_type:\s*\w+\s*)?think:|\s*$)",
    re.DOTALL,
)

SYSTEM = (
    "You extract compact source-aware motion rubric criteria. "
    "Return only valid JSON. No markdown. No explanation."
)

TRUST = (
    "Trust hierarchy: motion is authoritative for numeric kinematics, laterality, "
    "camera/screen orientation, and numerically supported body configuration. "
    "Video/final summary is preferred for high-level activity semantics. "
    "Do not trust motion narrative verbs unless supported by numbers or video semantics."
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


def messages_global(summary: str, final_answer: str, segments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    timeline = "\n".join(f"[{s['time']}] {s['answer'][:160]}" for s in segments)
    user = f"""{TRUST}

Summary: {summary}
Final answer: {final_answer}
Timeline:
{timeline}

Extract compact global criteria only. Do not extract per-segment numeric/body details.
Use:
- 1 global_activity item
- 5 to 7 temporal_phases
- exactly 6 negative_criteria

Return JSON:
{{
  "mode": "source_aware_reasoning_motion_rubric_v2",
  "global_activity": [
    {{"criterion": "high-level activity criterion", "source": "video+motion"}}
  ],
  "temporal_phases": ["ordered phase"],
  "negative_criteria": [
    {{"criterion": "penalize a concrete contradiction", "type": "unsupported_detail | contradiction | wrong_laterality | wrong_orientation | numeric_contradiction | unrelated_motion", "source_of_truth": "motion | video | video+motion"}}
  ]
}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def messages_segment(summary: str, segment: Dict[str, str]) -> List[Dict[str, Any]]:
    user = f"""{TRUST}

Summary: {summary}
Segment time: {segment['time']}
Think: {segment['think']}
Answer: {segment['answer']}

Extract compact criteria for this segment only.
Use:
- 2 basic_action_facts
- 1 body_configuration
- 3 numeric_kinematics, choosing the most important explicit numbers
- 1 laterality item if left/right appears, otherwise []
- 1 camera_relative_orientation item if camera/screen orientation appears, otherwise []
- 2 reasoning_criteria from think
- rejected_claims: [] unless think explicitly rejects a claim

Numeric object rules:
- target_range is [low, high]; for a single value use [value, value].
- degrees use strict_tolerance 10 and loose_tolerance 20.
- m use strict_tolerance 0.05 and loose_tolerance 0.10.
- s use strict_tolerance 0.20 and loose_tolerance 0.50.

Return JSON:
{{
  "time": "{segment['time']}",
  "basic_action_facts": [{{"criterion": "concise action fact", "source": "video | motion | video+motion"}}],
  "body_configuration": [{{"criterion": "motion-grounded posture fact", "source": "motion"}}],
  "numeric_kinematics": [
    {{"criterion": "quantity/body part is within numeric range", "quantity": "name", "body_part": "body part", "target_range": [0, 0], "unit": "degrees | m | s", "strict_tolerance": 10, "loose_tolerance": 20, "source": "motion"}}
  ],
  "laterality": [{{"criterion": "left/right fact", "source": "motion"}}],
  "camera_relative_orientation": [{{"criterion": "camera/screen orientation fact", "source": "motion"}}],
  "reasoning_criteria": [
    {{"criterion": "reasoning must separate or resolve sources", "type": "source_separation | conflict_detection | trust_hierarchy_application | numeric_evidence_use | reasoning_answer_consistency", "source": "think"}}
  ],
  "rejected_claims": [{{"claim": "rejected claim", "rejected_because": "trusted source reason", "trusted_source": "motion | video"}}]
}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate_json(
    generator: QwenTextGenerator,
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    retry_max_new_tokens: int,
) -> tuple[Optional[Dict[str, Any]], str]:
    raw = generator.generate(messages, max_new_tokens=max_new_tokens)
    parsed = parse_json_object(raw)
    if parsed is not None:
        return parsed, raw
    retry = [dict(m) for m in messages]
    retry[-1]["content"] += (
        "\n\nPrevious output was invalid or too long. Return a much shorter valid JSON object only. "
        "Use fewer criteria and no evidence strings."
    )
    retry_raw = generator.generate(retry, max_new_tokens=retry_max_new_tokens)
    return parse_json_object(retry_raw), raw + "\n\n---RETRY---\n\n" + retry_raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--gt-key", default="gt_description")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--global-max-new-tokens", type=int, default=650)
    parser.add_argument("--segment-max-new-tokens", type=int, default=700)
    parser.add_argument("--retry-max-new-tokens", type=int, default=500)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    parser.error(
        "this legacy fast-segmented generator is disabled because it emits "
        "forbidden tolerance/evidence fields; use "
        "python -m rubric_rl.extract_motion_criteria_v2"
    )

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_output:
        args.progress_output.parent.mkdir(parents=True, exist_ok=True)

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

    progress_f = args.progress_output.open("w", encoding="utf-8") if args.progress_output else None
    try:
        with args.output.open("w", encoding="utf-8") as out_f:
            for index, row in enumerate(load_jsonl(args.input)):
                if args.limit is not None and index >= args.limit:
                    break
                sid = row_id(row, args.id_key, index)
                report = parse_report(str(row.get(args.gt_key) or ""))
                if not report["segments"]:
                    out_f.write(json.dumps({"sample_id": sid, "error": "no_segments_parsed"}, ensure_ascii=False) + "\n")
                    continue

                raw_outputs: Dict[str, Any] = {}
                print(f"global_begin {sid}", file=sys.stderr, flush=True)
                global_obj, global_raw = generate_json(
                    generator,
                    messages_global(report["summary"], report["final_answer"], report["segments"]),
                    max_new_tokens=args.global_max_new_tokens,
                    retry_max_new_tokens=args.retry_max_new_tokens,
                )
                raw_outputs["global"] = global_raw
                if progress_f:
                    progress_f.write(json.dumps({"sample_id": sid, "part": "global", "parsed": global_obj is not None}, ensure_ascii=False) + "\n")
                    progress_f.flush()
                if global_obj is None:
                    payload = {"sample_id": sid, "error": "global_parse_failed"}
                    if args.keep_raw:
                        payload["raw_outputs"] = raw_outputs
                    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out_f.flush()
                    continue

                segment_objs = []
                errors = []
                raw_outputs["segments"] = []
                for seg in report["segments"]:
                    print(f"segment_begin {sid} {seg['time']}", file=sys.stderr, flush=True)
                    seg_obj, seg_raw = generate_json(
                        generator,
                        messages_segment(report["summary"], seg),
                        max_new_tokens=args.segment_max_new_tokens,
                        retry_max_new_tokens=args.retry_max_new_tokens,
                    )
                    raw_outputs["segments"].append({"time": seg["time"], "raw": seg_raw})
                    if progress_f:
                        progress_f.write(json.dumps({"sample_id": sid, "part": seg["time"], "parsed": seg_obj is not None}, ensure_ascii=False) + "\n")
                        progress_f.flush()
                    if seg_obj is None:
                        errors.append(seg["time"])
                    else:
                        seg_obj["time"] = str(seg_obj.get("time") or seg["time"])
                        segment_objs.append(seg_obj)

                if errors:
                    payload = {"sample_id": sid, "error": "segment_parse_failed", "segments": errors}
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
    finally:
        if progress_f:
            progress_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
