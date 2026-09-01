#!/usr/bin/env python3
"""Offline criteria extraction for multiple-choice QA Rubric RL.

Input rows are QA JSONL records with a question, A/B/C/D options, and a
gold answer such as ``<answer>A</answer>``. The script builds a compact QA
record, applies the QA MC offline prompt, and writes normalized criteria with
stable IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from motionllm.grpo.qa_rubric import (
    QA_RUBRIC_VERSION,
    assert_qa_dataset_binding,
    validate_qa_criteria,
)
from motionllm.grpo.rubric_common import RubricValidationError, strict_json_object
import rubric_rl.artifacts as artifact_support
from rubric_rl.artifacts import (
    AtomicJsonlArtifact,
    freeze_source_records,
    iter_jsonl_objects,
)
from rubric_rl.qwen_text import QwenTextGenerator


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>")
QUESTION_RE = re.compile(r"Question:\s*(.*?)\n\s*\nChoose exactly one option:", re.S)
OPTION_RE = re.compile(r"\n([A-D])\.\s*(.*?)(?=\n[A-D]\.\s*|\Z)", re.S)


STANDARD_FORMAT_CRITERIA = [
    {
        "criterion": "Exactly one <think>...</think> block is present.",
        "type": "format",
    },
    {
        "criterion": "Exactly one <answer>...</answer> block is present.",
        "type": "format",
    },
    {
        "criterion": "The <answer> block contains only one uppercase option letter A, B, C, or D.",
        "type": "format",
    },
    {
        "criterion": "The reasoning appears before the final answer, with no extra final conclusion outside <answer>.",
        "type": "format",
    },
]

ALLOWED_QUESTION_TYPES = {
    "time_range",
    "orientation_transition",
    "posture_state",
    "limb_motion",
    "laterality",
    "numeric_comparison",
    "action_identification",
    "temporal_comparison",
    "other_motion_qa",
}

QUESTION_TYPE_ALIASES = {
    "orientation": "orientation_transition",
    "numeric": "numeric_comparison",
    "comparison": "temporal_comparison",
    "time": "time_range",
}


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    yield from iter_jsonl_objects(path)


def row_id(row: Dict[str, Any], index: int, id_key: str) -> str:
    for key in [id_key, "sample_id", "benchmark_id", "group_id", "id"]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def get_message_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_question_and_options(text: str) -> Tuple[str, Dict[str, str]]:
    question = ""
    match = QUESTION_RE.search(text)
    if match:
        question = normalize_space(match.group(1))
    options: Dict[str, str] = {}
    for key, value in OPTION_RE.findall(text):
        options[key] = normalize_space(value)
    return question, options


def extract_correct_option(row: Dict[str, Any]) -> str:
    for key in ["answer", "solution"]:
        value = row.get(key)
        if isinstance(value, str):
            match = ANSWER_RE.search(value)
            if match:
                return match.group(1)
            stripped = value.strip().upper()
            if stripped in {"A", "B", "C", "D"}:
                return stripped
    return ""


def compact_qa_record(row: Dict[str, Any], index: int, id_key: str) -> Dict[str, Any]:
    text = get_message_text(row)
    question, options = extract_question_and_options(text)
    correct = extract_correct_option(row)
    return {
        "sample_id": row_id(row, index, id_key),
        "task": row.get("task") or "QA",
        "branch": row.get("branch"),
        "question": question,
        "options": {key: options.get(key, "") for key in ["A", "B", "C", "D"]},
        "correct_option": correct,
        "correct_option_text": options.get(correct, "") if correct else "",
        "answer": row.get("answer") or row.get("solution") or "",
    }


def split_prompt_template(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    if "USER:" not in text:
        raise ValueError(f"Prompt template must contain USER: section: {path}")
    if text.startswith("SYSTEM:"):
        system_part, user_part = text.split("USER:", 1)
        system = system_part.removeprefix("SYSTEM:").strip()
        user = user_part.strip()
    else:
        system = "You are a rubric extractor for multiple-choice human-motion QA. Return only valid JSON."
        user = text.split("USER:", 1)[1].strip()
    return system, user


def build_messages(system_prompt: str, user_template: str, qa_record: Dict[str, Any]) -> List[Dict[str, str]]:
    qa_json = json.dumps(qa_record, ensure_ascii=False, indent=2)
    user = user_template.replace("{QA_JSON}", qa_json)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def criterion_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("criterion") or item.get("fact") or item.get("description") or "")
    return str(item)


def normalize_criterion_list(items: Any, prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for idx, item in enumerate(items, start=1):
        text = criterion_text(item).strip()
        if not text:
            continue
        if isinstance(item, dict):
            normalized = dict(item)
            normalized["criterion"] = text
        else:
            normalized = {"criterion": text}
        normalized["id"] = f"{prefix}{idx}"
        out.append(normalized)
    return out


def normalize_criteria(parsed: Dict[str, Any], qa_record: Dict[str, Any]) -> Dict[str, Any]:
    """Assign only stable IDs; reject every other generated schema mismatch."""

    checked = validate_qa_criteria(parsed, assign_missing_ids=True)
    canonical = {
        "benchmark_id": qa_record["sample_id"],
        "question": qa_record["question"],
        "options": qa_record["options"],
        "correct_option": qa_record["correct_option"],
        "correct_option_text": qa_record["correct_option_text"],
    }
    assert_qa_dataset_binding(checked, canonical)
    return checked


def deterministic_fallback(qa_record: Dict[str, Any]) -> Dict[str, Any]:
    correct = qa_record["correct_option"]
    correct_text = qa_record["correct_option_text"]
    question = qa_record["question"]
    criteria = {
        "mode": "qa_mc",
        "benchmark_id": qa_record["sample_id"],
        "task": qa_record.get("task") or "QA",
        "question_type": "other_motion_qa",
        "question_focus": question,
        "question": question,
        "options": qa_record["options"],
        "correct_option": correct,
        "correct_option_text": correct_text,
        "reasoning_criteria": [
            {
                "criterion": "The reasoning identifies the motion attribute or temporal relation asked by the question.",
                "type": "question_focus",
            },
            {
                "criterion": "The reasoning discusses the relevant option facts instead of giving only a bare answer.",
                "type": "correct_option_fact",
            },
            {
                "criterion": f"The reasoning supports the correct option {correct}.",
                "type": "option_mapping",
            },
            {
                "criterion": f"The reasoning is consistent with the correct option text: {correct_text}",
                "type": "answer_consistency",
            },
        ],
        "format_criteria": STANDARD_FORMAT_CRITERIA,
        "negative_criteria": [
            {
                "criterion": "The reasoning supports an option other than the gold option.",
                "type": "wrong_option_support",
            },
            {
                "criterion": "The reasoning is generic and does not address the question.",
                "type": "irrelevant_reasoning",
            },
            {
                "criterion": "The reasoning states a relation incompatible with the correct option text.",
                "type": "contradiction",
            },
        ],
    }
    return normalize_criteria(criteria, qa_record)


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
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--fallback-deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-timing", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    written = 0
    skipped = 0
    seen_input_ids: set[str] = set()
    code_paths = tuple(
        path.resolve()
        for path in (
            Path(__file__),
            Path(artifact_support.__file__),
            Path(__file__).with_name("qwen_text.py"),
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "qa_rubric.py",
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "rubric_common.py",
        )
    )
    source_paths = (args.input, args.prompt, *code_paths)
    frozen_sources = freeze_source_records(source_paths)
    frozen_sha256 = {
        record["path"]: record["sha256"] for record in frozen_sources
    }
    prompt_path = args.prompt.resolve(strict=True)
    run_contract = {
        "operation": "extract_qa_mc_criteria",
        "model": args.model,
        "model_revision": args.model_revision,
        "generation": {
            "dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "effective_attn_implementation": args.attn_implementation,
            "attention_backend_fallback": "forbidden",
            "max_memory": args.max_memory,
            "max_new_tokens": args.max_new_tokens,
            "model_class": args.model_class,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "prompt": {
            "path": str(prompt_path),
            "sha256": frozen_sha256[str(prompt_path)],
        },
        "fields": {"id_key": args.id_key},
        "selection": {
            "limit": args.limit,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        "fallback_deterministic": args.fallback_deterministic,
        "output": {
            "keep_raw": args.keep_raw,
            "include_timing": args.include_timing,
        },
        "code_sha256": {
            str(path): frozen_sha256[str(path)] for path in code_paths
        },
    }
    with AtomicJsonlArtifact(
        args.output,
        resume=args.resume,
        rubric_version=QA_RUBRIC_VERSION,
        source_paths=source_paths,
        expected_source_records=frozen_sources,
        run_contract=run_contract,
    ) as artifact:
        system_prompt, user_template = split_prompt_template(prompt_path)
        load_started = time.perf_counter()
        generator = QwenTextGenerator(
            args.model,
            revision=args.model_revision,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_memory=parse_max_memory(args.max_memory),
            model_class=args.model_class,
            allow_attention_fallback=False,
        )
        load_seconds = time.perf_counter() - load_started
        if generator.effective_attn_implementation != args.attn_implementation:
            raise RuntimeError("formal rubric producer changed attention backend")
        print(
            json.dumps(
                {
                    "event": "model_loaded",
                    "input": str(args.input),
                    "output": str(args.output),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "load_seconds": load_seconds,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        done = artifact.done_ids
        for index, row in enumerate(load_jsonl(args.input)):
            if args.limit is not None and index >= args.limit:
                break
            if index % args.num_shards != args.shard_index:
                continue
            qa = compact_qa_record(row, index, args.id_key)
            sid = str(qa["sample_id"])
            if sid in seen_input_ids:
                raise ValueError(f"duplicate input sample_id: {sid!r}")
            seen_input_ids.add(sid)
            if sid in done:
                skipped += 1
                continue
            if not qa["question"] or not qa["correct_option"]:
                payload: Dict[str, Any] = {
                    "sample_id": sid,
                    "error": "missing_question_or_answer",
                    "qa_record": qa,
                }
                artifact.append(payload)
                written += 1
                continue

            started = time.perf_counter()
            raw = generator.generate(
                build_messages(system_prompt, user_template, qa),
                max_new_tokens=args.max_new_tokens,
            )
            generation_seconds = time.perf_counter() - started
            try:
                parsed = strict_json_object(raw)
                criteria = normalize_criteria(parsed, qa)
            except RubricValidationError as exc:
                if args.fallback_deterministic:
                    criteria = deterministic_fallback(qa)
                    payload = {
                        "sample_id": sid,
                        "criteria": criteria,
                        "criteria_source": "fallback_deterministic_after_invalid_generation",
                        "generation_error": str(exc),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
                else:
                    payload = {
                        "sample_id": sid,
                        "error": "invalid_generated_criteria",
                        "validation_error": str(exc),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
            else:
                payload = {
                    "sample_id": sid,
                    "criteria": criteria,
                    "criteria_source": "llm_offline_prompt",
                }
                if args.keep_raw:
                    payload["raw_response"] = raw
            if args.include_timing:
                payload["timing"] = {
                    "load_seconds": load_seconds,
                    "generation_seconds": generation_seconds,
                }
            artifact.append(payload)
            written += 1
            if written % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "shard_index": args.shard_index,
                            "written": written,
                            "skipped": skipped,
                            "last_sample_id": sid,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        inventory = artifact.commit()
    print(
        json.dumps(
            {
                "event": "done",
                "shard_index": args.shard_index,
                "written": written,
                "skipped": skipped,
                "artifact_sha256": inventory["artifact_sha256"],
                "inventory": str(args.output.with_name(args.output.name + ".inventory.json")),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
