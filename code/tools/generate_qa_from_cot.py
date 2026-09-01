#!/usr/bin/env python3
"""Generate MCQ QA data from CoT motion-description records."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_rl.qwen_text import QwenTextGenerator


REQUIRED_KEYS = {"question", "options", "answer_key", "difficulty", "question_type"}
OPTION_KEYS = ["A", "B", "C", "D"]
DIFFICULTIES = {"easy", "hard"}
QUESTION_TYPES = {
    "laterality",
    "orientation",
    "joint_angle",
    "temporal_localization",
    "temporal_comparison",
}
MAX_THINK_CHARS = 220
STRICT_SUFFIX = """

Final self-check before output, do not print this checklist:
- The JSON list must contain exactly one item for each question_type:
  laterality, orientation, joint_angle, temporal_localization, temporal_comparison.
- The JSON list must contain exactly 2 easy items and exactly 3 hard items.
- Every item must have exactly these fields:
  question, options, answer_key, difficulty, question_type.
- If any condition is not satisfied, rewrite the list before returning it.
"""


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def extract_json_list(text: str) -> Optional[List[Any]]:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : pos + 1])
                except Exception:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def truncate_middle(text: str, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return text[:head].rstrip() + " ... [truncated] ... " + text[-tail:].lstrip()


def compact_segment(segment: Dict[str, Any]) -> Dict[str, str]:
    return {
        "time_range": clean_text(segment.get("time_range") or segment.get("time")),
        "cot_type": clean_text(segment.get("cot_type")),
        "think": truncate_middle(segment.get("think"), MAX_THINK_CHARS),
        "answer": clean_text(segment.get("answer")),
    }


def parse_flat_gt_description(text: str) -> Tuple[str, str, str]:
    marker_a = "sample_summary:"
    marker_b = " per_segment:"
    marker_c = " final_answer:"
    if marker_a not in text or marker_b not in text:
        return "", text, ""
    sample_part, rest = text.split(marker_b, 1)
    sample_summary = sample_part.split(marker_a, 1)[1].strip()
    if marker_c in rest:
        per_segment, final_answer = rest.rsplit(marker_c, 1)
    else:
        per_segment, final_answer = rest, ""
    return sample_summary.strip(), per_segment.strip(), final_answer.strip()


def cot_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    desc = row.get("description_json")
    if isinstance(desc, dict):
        sample_summary = clean_text(desc.get("sample_summary"))
        final_answer = clean_text(desc.get("final_answer") or row.get("reference") or row.get("tgt"))
        segments = desc.get("per_segment")
        if isinstance(segments, list):
            per_segment = json.dumps(
                [compact_segment(seg) for seg in segments if isinstance(seg, dict)],
                ensure_ascii=False,
                indent=2,
            )
        else:
            per_segment = clean_text(segments)
        return sample_summary, per_segment, final_answer

    gt = row.get("gt_description")
    if isinstance(gt, str):
        return parse_flat_gt_description(gt)

    return (
        clean_text(row.get("sample_summary")),
        clean_text(row.get("per_segment")),
        clean_text(row.get("final_answer") or row.get("reference") or row.get("tgt")),
    )


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    sample_summary, per_segment, final_answer = cot_fields(row)
    return (
        template.replace("{SAMPLE_SUMMARY}", sample_summary)
        .replace("{PER_SEGMENT}", per_segment)
        .replace("{FINAL_ANSWER}", final_answer)
        + STRICT_SUFFIX
    )


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    options = item.get("options")
    if not isinstance(options, dict):
        options = {}
    return {
        "question": clean_text(item.get("question")),
        "options": {key: clean_text(options.get(key)) for key in OPTION_KEYS},
        "answer_key": clean_text(item.get("answer_key")).upper(),
        "difficulty": clean_text(item.get("difficulty")).lower(),
        "question_type": clean_text(item.get("question_type")).lower(),
    }


def rebalance_difficulty(items: List[Dict[str, Any]]) -> None:
    easy_preferred = {"laterality", "joint_angle"}
    hard_preferred = {"temporal_comparison", "temporal_localization", "orientation"}
    easy_count = sum(1 for item in items if item["difficulty"] == "easy")
    hard_count = sum(1 for item in items if item["difficulty"] == "hard")
    while easy_count > 2 and hard_count < 3:
        candidates = [
            item
            for item in items
            if item["difficulty"] == "easy" and item["question_type"] in hard_preferred
        ] or [item for item in items if item["difficulty"] == "easy"]
        candidates[-1]["difficulty"] = "hard"
        easy_count -= 1
        hard_count += 1
    while hard_count > 3 and easy_count < 2:
        candidates = [
            item
            for item in items
            if item["difficulty"] == "hard" and item["question_type"] in easy_preferred
        ] or [item for item in items if item["difficulty"] == "hard"]
        candidates[-1]["difficulty"] = "easy"
        hard_count -= 1
        easy_count += 1


def validate_items(items: Any) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    if not isinstance(items, list):
        return None, ["output_is_not_list"]
    if len(items) != 5:
        errors.append(f"expected_5_items_got_{len(items)}")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"item_{idx}_not_object")
            continue
        missing = REQUIRED_KEYS.difference(item.keys())
        if missing:
            errors.append(f"item_{idx}_missing_{','.join(sorted(missing))}")
        norm = normalize_item(item)
        if not norm["question"]:
            errors.append(f"item_{idx}_empty_question")
        if norm["answer_key"] not in OPTION_KEYS:
            errors.append(f"item_{idx}_bad_answer_key")
        if norm["difficulty"] not in DIFFICULTIES:
            errors.append(f"item_{idx}_bad_difficulty")
        if norm["question_type"] not in QUESTION_TYPES:
            errors.append(f"item_{idx}_bad_question_type")
        for key in OPTION_KEYS:
            if not norm["options"][key]:
                errors.append(f"item_{idx}_empty_option_{key}")
        normalized.append(norm)

    type_counts = {qtype: 0 for qtype in QUESTION_TYPES}
    diff_counts = {diff: 0 for diff in DIFFICULTIES}
    for item in normalized:
        if item["question_type"] in type_counts:
            type_counts[item["question_type"]] += 1
        if item["difficulty"] in diff_counts:
            diff_counts[item["difficulty"]] += 1
    missing_types = sorted(qtype for qtype, count in type_counts.items() if count < 1)
    if missing_types:
        errors.append("missing_question_types_" + ",".join(missing_types))
    if not errors:
        rebalance_difficulty(normalized)
        diff_counts = {diff: 0 for diff in DIFFICULTIES}
        for item in normalized:
            if item["difficulty"] in diff_counts:
                diff_counts[item["difficulty"]] += 1
    if diff_counts["easy"] != 2 or diff_counts["hard"] != 3:
        errors.append(f"bad_difficulty_counts_easy_{diff_counts['easy']}_hard_{diff_counts['hard']}")

    return normalized, errors


def make_messages(
    prompt: str,
    retry_note: Optional[str] = None,
    previous_response: Optional[str] = None,
) -> List[Dict[str, str]]:
    system = (
        "You are a precise multiple-choice QA data generator. "
        "Follow the user instructions exactly and return valid JSON only."
    )
    user = prompt
    if retry_note:
        if previous_response and len(clean_text(previous_response)) <= 5:
            previous_text = "[previous response was an incomplete one-character opening bracket]"
        else:
            previous_text = clean_text(previous_response)[:12000]
        user = (
            prompt
            + "\n\nYour previous response failed validation for these reasons: "
            + retry_note
            + "\n\nPrevious response to repair:\n"
            + previous_text
            + "\n\nRegenerate the corrected JSON list only. "
            + "Do not stop after the opening '['; complete all 5 JSON objects and the closing ']'. "
            + "Do not explain the changes."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def row_id(row: Dict[str, Any], index: int, id_key: str) -> str:
    for key in [id_key, "sample_id", "benchmark_id", "id", "index"]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = row.get("sample_id")
            if sid is not None and not row.get("error"):
                done.add(str(sid))
    return done


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


def target_answer_key(source_index: Any, item_index: int) -> str:
    try:
        base = int(source_index)
    except Exception:
        base = 0
    return OPTION_KEYS[(base * 5 + item_index) % len(OPTION_KEYS)]


def rotate_item_answer_key(item: Dict[str, Any], target_key: str) -> Dict[str, Any]:
    options = item.get("options")
    answer_key = item.get("answer_key")
    if not isinstance(options, dict) or answer_key not in OPTION_KEYS or target_key not in OPTION_KEYS:
        return item
    if answer_key == target_key:
        return item

    correct_value = options.get(answer_key, "")
    wrong_values = [options.get(key, "") for key in OPTION_KEYS if key != answer_key]
    wrong_iter = iter(wrong_values)
    new_options = {}
    for key in OPTION_KEYS:
        if key == target_key:
            new_options[key] = correct_value
        else:
            new_options[key] = next(wrong_iter)

    rotated = dict(item)
    rotated["options"] = new_options
    rotated["answer_key"] = target_key
    return rotated


def balance_qa_items_answer_keys(items: List[Dict[str, Any]], source_index: Any) -> List[Dict[str, Any]]:
    return [
        rotate_item_answer_key(item, target_answer_key(source_index, idx))
        for idx, item in enumerate(items)
    ]


def flat_rows(payload: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    items = payload.get("qa_items")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        row = {
            "qa_id": f"{payload['sample_id']}_qa{idx}",
            "sample_id": payload["sample_id"],
            "source_index": payload.get("source_index"),
            "video_name": payload.get("video_name"),
            "video_file": payload.get("video_file"),
            **item,
        }
        out.append(row)
    return out


def write_payload(
    out_f: Any,
    flat_f: Any,
    payload: Dict[str, Any],
    *,
    keep_raw: bool,
) -> None:
    if not keep_raw:
        payload.pop("raw_response", None)
    if flat_f is not None and not payload.get("error"):
        for flat in flat_rows(payload):
            flat_f.write(json.dumps(flat, ensure_ascii=False) + "\n")
        flat_f.flush()
    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    out_f.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flat-output", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--stop-after-json-list", action="store_true")
    parser.add_argument("--retry-do-sample", action="store_true")
    parser.add_argument("--retry-temperature", type=float, default=0.7)
    parser.add_argument("--retry-top-p", type=float, default=0.95)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--balance-answer-keys", action="store_true")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    template = args.prompt.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.flat_output:
        args.flat_output.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_ids(args.output) if args.resume else set()
    load_started = time.perf_counter()
    generator = QwenTextGenerator(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_memory=parse_max_memory(args.max_memory),
        model_class=args.model_class,
    )
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "input": str(args.input),
                "output": str(args.output),
                "flat_output": str(args.flat_output) if args.flat_output else None,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "load_seconds": round(time.perf_counter() - load_started, 3),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    sample_mode = "a" if args.resume else "w"
    flat_mode = "a" if args.resume else "w"
    written = 0
    skipped = 0
    failed = 0

    def process_batch(records: List[Dict[str, Any]], out_f: Any, flat_f: Any) -> None:
        nonlocal written, failed
        if not records:
            return
        prompts = [record["prompt"] for record in records]
        batch_started = time.perf_counter()
        try:
            raw_outputs = generator.generate_batch(
                [make_messages(prompt) for prompt in prompts],
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                prefill=args.prefill,
                stop_after_json_list=args.stop_after_json_list,
            )
            batch_seconds = round(time.perf_counter() - batch_started, 3)
            batch_error = None
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "batch_generate_failed_falling_back_to_single",
                        "shard_index": args.shard_index,
                        "batch_size": len(records),
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            raw_outputs = []
            batch_error = repr(exc)
            batch_seconds = 0.0
            for record in records:
                single_started = time.perf_counter()
                raw_outputs.append(
                    generator.generate(
                        make_messages(record["prompt"]),
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        prefill=args.prefill,
                        stop_after_json_list=args.stop_after_json_list,
                    )
                )
                record["single_generation_seconds"] = round(time.perf_counter() - single_started, 3)

        for record, raw in zip(records, raw_outputs):
            parsed = extract_json_list(raw)
            qa_items, errors = validate_items(parsed)
            retry_seconds = 0.0
            for attempt in range(1, args.retries + 1):
                if qa_items is not None and not errors:
                    break
                retry_started = time.perf_counter()
                raw = generator.generate(
                    make_messages(record["prompt"], "; ".join(errors), raw),
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    do_sample=args.retry_do_sample,
                    temperature=args.retry_temperature,
                    top_p=args.retry_top_p,
                    prefill=args.prefill,
                    stop_after_json_list=args.stop_after_json_list,
                )
                retry_seconds += time.perf_counter() - retry_started
                parsed = extract_json_list(raw)
                qa_items, errors = validate_items(parsed)

            payload: Dict[str, Any] = {
                "sample_id": record["sample_id"],
                "source_index": record["source_index"],
                "video_name": record["row"].get("video_name"),
                "video_file": record["row"].get("video_file"),
                "generation_model": args.model,
                "prompt_file": str(args.prompt),
                "generation_seconds": round(
                    record.get("single_generation_seconds", batch_seconds) + retry_seconds,
                    3,
                ),
                "batch_size": len(records) if batch_error is None else 1,
            }
            if batch_error is not None:
                payload["batch_fallback_error"] = batch_error
            if parsed is None or errors:
                payload["error"] = "validation_failed"
                payload["validation_errors"] = errors
                payload["raw_response"] = raw
                failed += 1
            else:
                if args.balance_answer_keys:
                    qa_items = balance_qa_items_answer_keys(qa_items, record["source_index"])
                payload["qa_items"] = qa_items
                if args.keep_raw:
                    payload["raw_response"] = raw

            write_payload(out_f, flat_f, payload, keep_raw=args.keep_raw or bool(payload.get("error")))
            written += 1
            if written % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "shard_index": args.shard_index,
                            "written": written,
                            "skipped": skipped,
                            "failed": failed,
                            "last_sample_id": record["sample_id"],
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    with args.output.open(sample_mode, encoding="utf-8") as out_f:
        flat_f = args.flat_output.open(flat_mode, encoding="utf-8") if args.flat_output else None
        try:
            pending: List[Dict[str, Any]] = []
            for index, row in enumerate(load_jsonl(args.input)):
                if args.limit is not None and index >= args.limit:
                    break
                if index % args.num_shards != args.shard_index:
                    continue
                sid = row_id(row, index, args.id_key)
                if sid in done:
                    skipped += 1
                    continue

                prompt = build_prompt(template, row)
                pending.append(
                    {
                        "sample_id": sid,
                        "source_index": index,
                        "row": row,
                        "prompt": prompt,
                    }
                )
                if len(pending) >= args.batch_size:
                    process_batch(pending, out_f, flat_f)
                    pending = []
            process_batch(pending, out_f, flat_f)
        finally:
            if flat_f is not None:
                flat_f.close()

    print(
        json.dumps(
            {
                "event": "done",
                "written": written,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
