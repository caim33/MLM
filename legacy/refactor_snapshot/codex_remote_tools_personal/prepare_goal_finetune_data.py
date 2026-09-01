#!/usr/bin/env python3
"""Prepare the user's MotionX finetune/eval splits for all target models.

This script is deliberately conservative about the user's split contract:

- train/val are discovered under
  /wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/data/grpo_training/raw
- final eval is the explicit QA_500.json file
- QA_500 is never written into any training annotation

It normalizes JSON/JSONL inputs, resolves media paths relative to the
qwen-vl-finetune root, emits raw JSONL files for generic evaluators/proxy
models, and emits Qwen-style SFT JSON annotations for video-only and
motion-aware finetuning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_QWEN_ROOT = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune")
DEFAULT_RAW_DIR = DEFAULT_QWEN_ROOT / "data/grpo_training/raw"
DEFAULT_BENCHMARK = DEFAULT_QWEN_ROOT / "data/benchmark/text/QA/QA_500.json"
DEFAULT_OUT_DIR = Path(
    "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/finetune_goal_20260717/data"
)

ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.I)
LETTER_RE = re.compile(r"\b([A-D])\b", re.I)


def read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "samples", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


def resolve_existing_dataset_path(path: Path) -> Path:
    if path.exists():
        return path
    alternatives: List[Path] = []
    if path.suffix.lower() == ".json":
        alternatives.append(path.with_suffix(".jsonl"))
    elif path.suffix.lower() == ".jsonl":
        alternatives.append(path.with_suffix(".json"))
    for alt in alternatives:
        if alt.exists():
            return alt
    raise FileNotFoundError(path)


def iter_content_parts(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for msg in row.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    yield part


def extract_text(row: Dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    parts: List[str] = []
    for msg in row.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
    return "\n".join(x for x in parts if x.strip())


def first_media(row: Dict[str, Any], media_type: str) -> Optional[str]:
    keys = {
        "video": ("video", "video_path", "source_video"),
        "motion": ("motion", "motion_path", "source_motion"),
    }[media_type]
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    for part in iter_content_parts(row):
        if part.get("type") == media_type and part.get(media_type):
            return str(part[media_type])
    return None


def abs_under(root: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    m = ANSWER_RE.search(text)
    if m:
        return f"<answer>{m.group(1).upper()}</answer>"
    m = LETTER_RE.search(text)
    if m:
        return f"<answer>{m.group(1).upper()}</answer>"
    return text


def infer_branch(row: Dict[str, Any], motion: Optional[str], text: str) -> str:
    branch = str(row.get("branch") or "").lower()
    if branch in {"v", "vm"}:
        return branch
    if motion or "<motion_start>" in text or "<motion>" in text:
        return "vm"
    return "v"


def strip_motion_tokens(text: str) -> str:
    return (
        text.replace("<motion_start><motion><motion_end>\n", "")
        .replace("<motion_start><motion><motion_end>", "")
        .strip()
    )


def video_only_prompt(text: str) -> str:
    """Build a video-only SFT prompt without motion-evidence instructions."""
    text = strip_motion_tokens(text)
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text.strip()
    return (
        "You are given video evidence for a human action multiple-choice question.\n"
        "Analyze the video carefully and answer with exactly one final option.\n"
        "Do not explain. The final answer must be one of A, B, C, or D.\n"
        "Return it only in the form <answer>A</answer>, <answer>B</answer>, "
        "<answer>C</answer>, or <answer>D</answer>.\n\n"
        f"{qa}"
    )


def normalize_row(row: Dict[str, Any], qwen_root: Path, idx: int, source_file: Path) -> Dict[str, Any]:
    text = extract_text(row)
    video = abs_under(qwen_root, first_media(row, "video"))
    motion = abs_under(qwen_root, first_media(row, "motion"))
    answer = normalize_answer(row.get("answer") or row.get("solution"))
    branch = infer_branch(row, motion, text)
    sample_id = row.get("sample_id") or row.get("benchmark_id") or f"{source_file.stem}_{idx:06d}"
    out = dict(row)
    out.update(
        {
            "sample_id": str(sample_id),
            "group_id": str(row.get("group_id") or sample_id),
            "branch": branch,
            "prompt": text,
            "answer": answer,
            "solution": answer,
            "video": video,
            "motion": motion,
            "source_file": str(source_file),
        }
    )
    # Keep messages in a predictable shape for eval scripts.
    content: List[Dict[str, Any]] = []
    if video:
        content.append({"type": "video", "video": video})
    if motion:
        content.append({"type": "motion", "motion": motion})
    content.append({"type": "text", "text": text})
    out["messages"] = [{"role": "user", "content": content}]
    return out


def classify_file(path: Path) -> Optional[str]:
    name = path.name.lower()
    if any(tok in name for tok in ("val", "valid", "validation", "dev")):
        return "val"
    if "train" in name:
        return "train"
    return None


def split_rows(raw_dir: Path, qwen_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    files = sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}])
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []
    for path in files:
        split = classify_file(path)
        for idx, row in enumerate(read_records(path)):
            nrow = normalize_row(row, qwen_root, idx, path)
            record_split = str(row.get("split") or row.get("phase") or "").lower()
            target = split
            if not target:
                if record_split.startswith("train"):
                    target = "train"
                elif record_split.startswith(("val", "valid", "dev")):
                    target = "val"
            if target == "train":
                train.append(nrow)
            elif target == "val":
                val.append(nrow)
            else:
                unknown.append(nrow)
    return train, val, unknown


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sft_item(row: Dict[str, Any], *, mode: str) -> Optional[Dict[str, Any]]:
    video = row.get("video")
    motion = row.get("motion")
    if mode == "video" and not video:
        return None
    if mode == "motion" and not motion:
        return None
    text = str(row.get("prompt") or "")
    if mode == "video":
        user_text = "<video>\n" + video_only_prompt(text)
    else:
        body = text if "<motion_start>" in text else "<motion_start><motion><motion_end>\n" + strip_motion_tokens(text)
        user_text = "<video>\n" + body if video else body
    item: Dict[str, Any] = {
        "id": row["sample_id"],
        "sample_id": row["sample_id"],
        "group_id": row.get("group_id"),
        "branch": row.get("branch"),
        "video": video,
        "conversations": [
            {"from": "human", "value": user_text.strip()},
            {"from": "gpt", "value": row.get("answer")},
        ],
    }
    if mode == "motion":
        item["motion"] = motion
    return item


def write_sft(path: Path, rows: Iterable[Dict[str, Any]], *, mode: str, limit: Optional[int] = None) -> int:
    items: List[Dict[str, Any]] = []
    for row in rows:
        item = sft_item(row, mode=mode)
        if item is None:
            continue
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items)


def filter_rows(rows: Iterable[Dict[str, Any]], *, need_video: bool = False, need_motion: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if need_video and not row.get("video"):
            continue
        if need_motion and not row.get("motion"):
            continue
        out.append(row)
    return out


def missing_media(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    missing: List[Dict[str, str]] = []
    for row in rows:
        for key in ("video", "motion"):
            value = row.get(key)
            if value and not Path(value).exists():
                missing.append({"sample_id": row.get("sample_id", ""), "key": key, "path": value})
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.benchmark = resolve_existing_dataset_path(args.benchmark)
    test = [
        normalize_row(row, args.qwen_root, idx, args.benchmark)
        for idx, row in enumerate(read_records(args.benchmark))
    ]
    train, val, unknown = split_rows(args.raw_dir, args.qwen_root)
    if unknown and not train and not val:
        raise RuntimeError(
            f"Could not infer train/val split from {args.raw_dir}. "
            f"Unknown records={len(unknown)}; please inspect source_file names or split fields."
        )
    if not train or not val:
        raise RuntimeError(f"Empty split after discovery: train={len(train)} val={len(val)} unknown={len(unknown)}")
    if not test:
        raise RuntimeError(f"No benchmark records loaded from {args.benchmark}")

    original_split_counts = {"train": len(train), "val": len(val), "unknown": len(unknown), "test": len(test)}
    train_ids = {row.get("sample_id") for row in train}
    val_ids = {row.get("sample_id") for row in val}
    test_ids = {row.get("sample_id") for row in test}
    test_group_ids = {row.get("group_id") for row in test if row.get("group_id")}
    leakage = {
        "train_test_overlap": sorted(x for x in train_ids & test_ids if x)[:50],
        "val_test_overlap": sorted(x for x in val_ids & test_ids if x)[:50],
        "train_val_overlap": sorted(x for x in train_ids & val_ids if x)[:50],
    }

    def not_test_overlap(row: Dict[str, Any]) -> bool:
        sample_id = row.get("sample_id")
        group_id = row.get("group_id")
        return sample_id not in test_ids and group_id not in test_group_ids

    train_filtered = [row for row in train if not_test_overlap(row)]
    val_filtered = [row for row in val if not_test_overlap(row)]
    filtered_counts = {
        "train_removed_for_test_overlap": len(train) - len(train_filtered),
        "val_removed_for_test_overlap": len(val) - len(val_filtered),
    }
    train = train_filtered
    val = val_filtered
    if not train or not val:
        raise RuntimeError(
            f"Empty split after removing QA_500 overlaps: train={len(train)} val={len(val)} "
            f"filtered={filtered_counts}"
        )

    out = args.output_dir
    raw_dir = out / "raw_jsonl"
    sft_dir = out / "sft"
    eval_dir = out / "eval"

    counts: Dict[str, int] = {}
    counts["train_raw"] = write_jsonl(raw_dir / "train_raw.jsonl", train)
    counts["val_raw"] = write_jsonl(raw_dir / "val_raw.jsonl", val)
    counts["unknown_raw"] = write_jsonl(raw_dir / "unknown_raw.jsonl", unknown)
    counts["test_all"] = write_jsonl(eval_dir / "QA_500.abs.jsonl", test)
    counts["test_video"] = write_jsonl(eval_dir / "QA_500_video.abs.jsonl", filter_rows(test, need_video=True))
    counts["test_motion"] = write_jsonl(eval_dir / "QA_500_motion.abs.jsonl", filter_rows(test, need_motion=True))

    counts["sft_train_video"] = write_sft(sft_dir / "motionx_qa_train_v.json", train, mode="video")
    counts["sft_val_video"] = write_sft(sft_dir / "motionx_qa_val_v.json", val, mode="video")
    counts["sft_train_motion"] = write_sft(sft_dir / "motionx_qa_train_vm.json", train, mode="motion")
    counts["sft_val_motion"] = write_sft(sft_dir / "motionx_qa_val_vm.json", val, mode="motion")
    counts["sft_train_video_smoke32"] = write_sft(sft_dir / "motionx_qa_train_v_smoke32.json", train, mode="video", limit=32)
    counts["sft_train_motion_smoke32"] = write_sft(sft_dir / "motionx_qa_train_vm_smoke32.json", train, mode="motion", limit=32)

    post_train_ids = {row.get("sample_id") for row in train}
    post_val_ids = {row.get("sample_id") for row in val}
    post_leakage = {
        "train_test_overlap": sorted(x for x in post_train_ids & test_ids if x)[:50],
        "val_test_overlap": sorted(x for x in post_val_ids & test_ids if x)[:50],
        "train_val_overlap": sorted(x for x in post_train_ids & post_val_ids if x)[:50],
    }
    all_rows = train + val + test
    media_missing = missing_media(all_rows)

    manifest = {
        "raw_dir": str(args.raw_dir),
        "benchmark": str(args.benchmark),
        "qwen_root": str(args.qwen_root),
        "output_dir": str(out),
        "counts": counts,
        "original_split_counts": original_split_counts,
        "filtered_counts": filtered_counts,
        "paths": {
            "train_raw": str(raw_dir / "train_raw.jsonl"),
            "val_raw": str(raw_dir / "val_raw.jsonl"),
            "test_all": str(eval_dir / "QA_500.abs.jsonl"),
            "test_video": str(eval_dir / "QA_500_video.abs.jsonl"),
            "test_motion": str(eval_dir / "QA_500_motion.abs.jsonl"),
            "sft_dir": str(sft_dir),
        },
        "pre_filter_leakage_preview": leakage,
        "post_filter_leakage": post_leakage,
        "missing_media_count": len(media_missing),
        "missing_media_preview": media_missing[:50],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if post_leakage["train_test_overlap"] or post_leakage["val_test_overlap"]:
        raise SystemExit("Refusing to continue: QA_500 overlap remains after filtering.")
    if media_missing:
        raise SystemExit(f"Missing media files detected: {len(media_missing)}")


if __name__ == "__main__":
    main()
