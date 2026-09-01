#!/usr/bin/env python3
"""Summarize MotionLLM JSONL predictions without changing model outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ANSWER = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


def label(text: object) -> str | None:
    match = ANSWER.search(str(text))
    return match.group(1).upper() if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    targets = [label(row.get("tgt")) for row in rows]
    predictions = [label(row.get("prediction")) for row in rows]
    valid = sum(value is not None for value in predictions)
    correct = sum(target == prediction for target, prediction in zip(targets, predictions))
    result = {
        "input": str(args.input),
        "rows": len(rows),
        "valid_prediction_labels": valid,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "target_counts": dict(sorted(Counter(targets).items())),
        "prediction_counts": dict(sorted(Counter(predictions).items())),
        "per_row": [
            {
                "index": row.get("index"),
                "target": target,
                "prediction": prediction,
                "correct": target == prediction,
            }
            for row, target, prediction in zip(rows, targets, predictions)
        ],
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
