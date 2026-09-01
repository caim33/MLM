#!/usr/bin/env python3
"""Prepare CoT description JSONL as Rubric-RL V2 GT input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import rubric_rl.artifacts as artifact_support
from rubric_rl.artifacts import (
    AtomicJsonlArtifact,
    freeze_source_records,
    iter_jsonl_objects,
    sha256_file,
)
from rubric_rl.reward_v2 import MOTION_RUBRIC_V2_VERSION


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    yield from iter_jsonl_objects(path)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def format_segment(segment: Dict[str, Any]) -> str:
    time_range = clean_text(segment.get("time_range") or segment.get("time"))
    cot_type = clean_text(segment.get("cot_type"))
    think = clean_text(segment.get("think"))
    answer = clean_text(segment.get("answer"))
    parts = [f"[{time_range}]"]
    if cot_type:
        parts.append(f"cot_type: {cot_type}")
    parts.append(f"think: {think}")
    parts.append(f"answer: {answer}")
    return " ".join(parts)


def build_gt_description(row: Dict[str, Any]) -> str:
    desc = row.get("description_json")
    if not isinstance(desc, dict):
        raise ValueError("missing description_json")
    sample_summary = clean_text(desc.get("sample_summary"))
    final_answer = clean_text(desc.get("final_answer") or row.get("reference") or row.get("tgt"))
    segments = desc.get("per_segment")
    if not isinstance(segments, list) or not segments:
        raise ValueError("missing description_json.per_segment")
    segment_text = " ".join(format_segment(seg) for seg in segments if isinstance(seg, dict))
    return f"sample_summary: {sample_summary} per_segment: {segment_text} final_answer: {final_answer}"


def _implementation_paths() -> tuple[Path, Path]:
    return Path(__file__).resolve(), Path(artifact_support.__file__).resolve()


def build_run_contract(
    *,
    limit: int | None,
    implementation_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    script_path, artifact_path = _implementation_paths()
    hashes = implementation_sha256 or {
        "rubric_rl/prepare_cot_gt_v2.py": sha256_file(script_path),
        "rubric_rl/artifacts.py": sha256_file(artifact_path),
    }
    return {
        "contract_version": "1.0",
        "operation": "prepare_cot_gt_v2",
        "rubric_version": MOTION_RUBRIC_V2_VERSION + "_gt_input",
        "limit": limit,
        "input_schema": {
            "format": "jsonl_objects",
            "sample_id_fallback_keys": ["sample_id", "id", "index"],
            "description_object_key": "description_json",
            "sample_summary_key": "sample_summary",
            "final_answer_fallback_keys": ["final_answer", "reference", "tgt"],
            "segments_key": "per_segment",
            "segment_time_fallback_keys": ["time_range", "time"],
            "segment_fields": ["cot_type", "think", "answer"],
        },
        "output_schema": {
            "format": "jsonl_objects",
            "id_key": "sample_id",
            "fields": ["sample_id", "video_name", "video_file", "gt_description"],
        },
        "policies": {
            "selection": "first_n_source_rows",
            "invalid_row": "fail_closed",
            "duplicate_sample_id": "reject",
            "resume": "exact_contract_and_source_prefix",
        },
        "implementation_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    written = 0
    skipped = 0
    seen: set[str] = set()
    implementation_paths = _implementation_paths()
    source_paths = (args.input, *implementation_paths)
    frozen_sources = freeze_source_records(source_paths)
    frozen_sha256 = {
        record["path"]: record["sha256"] for record in frozen_sources
    }
    implementation_sha256 = {
        "rubric_rl/prepare_cot_gt_v2.py": frozen_sha256[
            str(implementation_paths[0])
        ],
        "rubric_rl/artifacts.py": frozen_sha256[str(implementation_paths[1])],
    }
    with AtomicJsonlArtifact(
        args.output,
        resume=args.resume,
        rubric_version=MOTION_RUBRIC_V2_VERSION + "_gt_input",
        run_contract=build_run_contract(
            limit=args.limit,
            implementation_sha256=implementation_sha256,
        ),
        source_paths=source_paths,
        expected_source_records=frozen_sources,
    ) as artifact:
        done = artifact.done_ids
        for source_index, row in enumerate(load_jsonl(args.input)):
            if args.limit is not None and source_index >= args.limit:
                break
            sid = clean_text(row.get("sample_id") or row.get("id") or row.get("index"))
            if not sid:
                raise ValueError("input row is missing sample_id")
            if sid in seen:
                raise ValueError(f"duplicate input sample_id: {sid!r}")
            seen.add(sid)
            if sid in done:
                skipped += 1
                continue
            try:
                gt_description = build_gt_description(row)
            except Exception as exc:
                raise ValueError(f"invalid GT row {sid!r}: {exc}") from exc
            payload = {
                "sample_id": sid,
                "video_name": row.get("video_name"),
                "video_file": row.get("video_file"),
                "gt_description": gt_description,
            }
            artifact.append(payload)
            written += 1
        unexpected_done = done.difference(seen)
        if unexpected_done:
            examples = ", ".join(repr(value) for value in sorted(unexpected_done)[:3])
            raise ValueError(
                "resume artifact contains sample IDs outside the selected source prefix: "
                + examples
            )
        inventory = artifact.commit()
    print(
        json.dumps(
            {
                "written": written,
                "skipped": skipped,
                "artifact_sha256": inventory["artifact_sha256"],
                "inventory": str(args.output.with_name(args.output.name + ".inventory.json")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
