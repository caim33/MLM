#!/usr/bin/env python3
"""Export ms-swift GRPO completions grouped by question/branch tags in prompt."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QID_PATTERN = re.compile(r"\[QID=([^\]\s]+)\]")
BRANCH_PATTERN = re.compile(r"\[BRANCH=([^\]\s]+)\]")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _extract_text_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    texts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
    return "\n".join(t for t in texts if t)


def _extract_tag(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _collect_dataset_meta(dataset_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if dataset_path is None or not dataset_path.exists():
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(dataset_path):
        prompt_text = _extract_text_from_messages(row.get("messages"))
        qid = row.get("question_id") or _extract_tag(QID_PATTERN, prompt_text)
        if not qid:
            continue
        qid = str(qid)
        if qid not in meta:
            meta[qid] = {
                "question_id": qid,
                "difficulty": row.get("difficulty", ""),
                "group_id": row.get("group_id", ""),
                "source_sample_id": row.get("source_sample_id", ""),
                "answer": row.get("answer", ""),
            }
    return meta


def _extract_reward_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    reward_fields: Dict[str, Any] = {}
    for key, value in row.items():
        if key.endswith("_reward") or key.endswith("_rewards"):
            reward_fields[key] = value
    for key in ("reward", "total_reward", "advantages"):
        if key in row:
            reward_fields[key] = row.get(key)
    return reward_fields


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group GRPO completions by question tags.")
    parser.add_argument("--completions", type=Path, required=True, help="Path to completions.jsonl")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset JSONL for question metadata",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Grouped JSON output path",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=None,
        help="Optional flattened enriched JSONL output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_meta = _collect_dataset_meta(args.dataset)
    grouped: Dict[str, Dict[str, Any]] = {}
    flattened: List[Dict[str, Any]] = []

    for row in _read_jsonl(args.completions):
        prompt = str(row.get("prompt", ""))
        completion = row.get("completion", "")
        question_id = _extract_tag(QID_PATTERN, prompt)
        branch = _extract_tag(BRANCH_PATTERN, prompt)
        if not question_id:
            # Skip records without explicit question tag.
            continue

        if question_id not in grouped:
            grouped[question_id] = {
                **dataset_meta.get(question_id, {"question_id": question_id}),
                "rollouts": [],
            }

        rollout = {
            "step": row.get("step"),
            "branch": branch,
            "prompt": prompt,
            "completion": completion,
            "rewards": _extract_reward_fields(row),
        }
        grouped[question_id]["rollouts"].append(rollout)

        enriched = {
            "question_id": question_id,
            "branch": branch,
            "step": row.get("step"),
            "prompt": prompt,
            "completion": completion,
            "rewards": _extract_reward_fields(row),
            "difficulty": grouped[question_id].get("difficulty", ""),
            "group_id": grouped[question_id].get("group_id", ""),
            "source_sample_id": grouped[question_id].get("source_sample_id", ""),
        }
        flattened.append(enriched)

    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    grouped_items = [grouped[k] for k in sorted(grouped.keys())]
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(grouped_items, f, ensure_ascii=False, indent=2)

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in flattened:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "grouped_output": str(output_json),
                "flattened_output": str(args.output_jsonl) if args.output_jsonl else "",
                "questions_with_rollouts": len(grouped_items),
                "total_rollout_rows": len(flattened),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
