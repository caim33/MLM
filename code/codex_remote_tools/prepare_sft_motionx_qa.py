#!/usr/bin/env python3
"""Convert MotionX GRPO raw JSONL records into qwen-vl-finetune SFT annotations."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_media_and_text(record: Dict[str, Any]) -> Tuple[Optional[str], str]:
    videos: List[str] = []
    texts: List[str] = []
    for msg in record.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "video" and part.get("video"):
                videos.append(str(part["video"]))
            elif part.get("type") == "text":
                texts.append(str(part.get("text", "")))
    video = record.get("video") or (videos[0] if videos else None)
    return str(video) if video else None, "\n".join(t.strip() for t in texts if t.strip())


def normalize_answer(text: Any) -> str:
    s = str(text or "").strip()
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", s, flags=re.I)
    if match:
        return f"<answer>{match.group(1).upper()}</answer>"
    match = re.search(r"\b([ABCD])\b", s, flags=re.I)
    if match:
        return f"<answer>{match.group(1).upper()}</answer>"
    return s


def convert_record(record: Dict[str, Any], branch: str) -> Optional[Dict[str, Any]]:
    if str(record.get("branch", "")).lower() != branch:
        return None
    video, text = extract_media_and_text(record)
    if not video or not text:
        return None
    item: Dict[str, Any] = {
        "id": record.get("sample_id") or record.get("benchmark_id") or record.get("group_id"),
        "sample_id": record.get("sample_id"),
        "group_id": record.get("group_id"),
        "branch": branch,
        "video": video,
        "conversations": [
            {"from": "human", "value": "<video>\n" + text},
            {"from": "gpt", "value": normalize_answer(record.get("answer") or record.get("solution"))},
        ],
    }
    if branch == "vm":
        motion = record.get("motion")
        if not motion:
            return None
        item["motion"] = str(motion)
    return item


def convert_file(input_path: Path, output_path: Path, branch: str, limit: Optional[int]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for record in load_jsonl(input_path):
        item = convert_record(record, branch)
        if item is None:
            skipped += 1
            continue
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "branch": branch,
        "count": len(rows),
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_raw", type=Path, required=True)
    parser.add_argument("--val_raw", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: List[Dict[str, Any]] = []
    summaries.append(
        convert_file(args.train_raw, args.output_dir / "motionx_qa_train_v.json", "v", args.train_limit)
    )
    summaries.append(
        convert_file(args.train_raw, args.output_dir / "motionx_qa_train_vm.json", "vm", args.train_limit)
    )
    summaries.append(
        convert_file(args.train_raw, args.output_dir / "motionx_qa_train_v_smoke32.json", "v", 32)
    )
    summaries.append(
        convert_file(args.train_raw, args.output_dir / "motionx_qa_train_vm_smoke32.json", "vm", 32)
    )
    if args.val_raw:
        summaries.append(
            convert_file(args.val_raw, args.output_dir / "motionx_qa_val_v.json", "v", args.val_limit)
        )
        summaries.append(
            convert_file(args.val_raw, args.output_dir / "motionx_qa_val_vm.json", "vm", args.val_limit)
        )
    manifest = {"datasets": summaries}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
