#!/usr/bin/env python3
"""Read-only, deterministic sanity experiments for the caimeng dataset catalog."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
BASE_ID_RE = re.compile(r"_q\d+$")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/wangbenyou-sulongjie/caimeng/dataset"),
    )
    parser.add_argument(
        "--qwen-repo-root",
        type=Path,
        default=Path("/wangbenyou-sulongjie/qwen-vl-finetune"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--npy-samples", type=int, default=32)
    parser.add_argument("--video-samples", type=int, default=12)
    return parser.parse_args()


class Audit:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "caimeng_dataset_sanity",
            "seed": args.seed,
            "dataset_root": str(args.dataset_root),
            "qwen_repo_root": str(args.qwen_repo_root),
            "read_only": True,
            "sections": {},
            "issues": [],
        }

    def issue(
        self,
        severity: str,
        code: str,
        message: str,
        evidence: Any | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if evidence is not None:
            item["evidence"] = evidence
        self.report["issues"].append(item)

    def section(self, name: str, value: Any) -> None:
        self.report["sections"][name] = value


def stable_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def iter_files(root: Path, suffixes: Sequence[str]) -> Iterator[Path]:
    wanted = tuple(value.lower() for value in suffixes)
    for current, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        base = Path(current)
        for filename in filenames:
            if filename.lower().endswith(wanted):
                yield base / filename


def reservoir_sample(
    root: Path,
    suffixes: Sequence[str],
    count: int,
    rng: random.Random,
) -> tuple[list[Path], int]:
    selected: list[Path] = []
    seen = 0
    for path in iter_files(root, suffixes):
        seen += 1
        if len(selected) < count:
            selected.append(path)
            continue
        replacement = rng.randrange(seen)
        if replacement < count:
            selected[replacement] = path
    selected.sort()
    return selected, seen


def first_files(root: Path, suffixes: Sequence[str], count: int) -> list[Path]:
    result: list[Path] = []
    for path in iter_files(root, suffixes):
        result.append(path)
        if len(result) >= count:
            break
    return result


def summarize_counter(counter: collections.Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def audit_npy(
    audit: Audit,
    name: str,
    root: Path,
    expected_dim: int,
    samples: int,
) -> None:
    import numpy as np

    selected, total = reservoir_sample(
        root,
        (".npy",),
        samples,
        stable_rng(audit.args.seed, f"npy:{name}"),
    )
    shapes: collections.Counter[str] = collections.Counter()
    dtypes: collections.Counter[str] = collections.Counter()
    lengths: list[int] = []
    failures: list[dict[str, str]] = []
    nonfinite: list[str] = []
    all_zero: list[str] = []
    wrong_dim: list[dict[str, Any]] = []

    for path in selected:
        rel = str(path.relative_to(audit.args.dataset_root))
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - this is an audit boundary
            failures.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        shapes[str(tuple(int(value) for value in array.shape))] += 1
        dtypes[str(array.dtype)] += 1
        if array.ndim >= 1:
            lengths.append(int(array.shape[0]))
        if array.ndim != 2 or array.shape[-1] != expected_dim:
            wrong_dim.append({"path": rel, "shape": list(array.shape)})
        if not bool(np.isfinite(array).all()):
            nonfinite.append(rel)
        if array.size and bool(np.all(array == 0)):
            all_zero.append(rel)

    result = {
        "root": str(root),
        "total_files_seen": total,
        "sample_size": len(selected),
        "sample_sha256": hashlib.sha256(
            "\n".join(str(path) for path in selected).encode("utf-8")
        ).hexdigest(),
        "shape_counts": summarize_counter(shapes),
        "dtype_counts": summarize_counter(dtypes),
        "length_min": min(lengths) if lengths else None,
        "length_median": statistics.median(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "load_failures": failures,
        "nonfinite": nonfinite,
        "all_zero": all_zero,
        "wrong_motion_dim": wrong_dim,
    }
    audit.section(f"npy_{name}", result)
    if failures:
        audit.issue("error", f"{name}_npy_load", "NPY files failed to load", failures)
    if nonfinite:
        audit.issue("error", f"{name}_npy_nonfinite", "NPY samples contain NaN/Inf", nonfinite)
    if wrong_dim:
        audit.issue(
            "error",
            f"{name}_npy_shape",
            f"NPY samples do not use the expected {expected_dim}-D feature dimension",
            wrong_dim,
        )
    if all_zero:
        audit.issue("warning", f"{name}_npy_zero", "NPY samples are entirely zero", all_zero)


def audit_videos(audit: Audit, name: str, root: Path, samples: int) -> None:
    try:
        import av
    except Exception as exc:  # noqa: BLE001
        audit.issue("critical", "pyav_missing", "PyAV is unavailable", str(exc))
        return

    selected, total = reservoir_sample(
        root,
        (".mp4", ".mov", ".mkv", ".avi"),
        samples,
        stable_rng(audit.args.seed, f"video:{name}"),
    )
    failures: list[dict[str, str]] = []
    decoded: list[dict[str, Any]] = []
    resolutions: collections.Counter[str] = collections.Counter()
    codecs: collections.Counter[str] = collections.Counter()
    durations: list[float] = []

    for path in selected:
        rel = str(path.relative_to(audit.args.dataset_root))
        container = None
        try:
            container = av.open(str(path))
            if not container.streams.video:
                raise ValueError("no video stream")
            stream = container.streams.video[0]
            frame = next(container.decode(stream))
            codec = str(stream.codec_context.name or "unknown")
            resolution = f"{stream.width}x{stream.height}"
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / 1_000_000)
            if duration is not None and math.isfinite(duration):
                durations.append(duration)
            codecs[codec] += 1
            resolutions[resolution] += 1
            decoded.append(
                {
                    "path": rel,
                    "codec": codec,
                    "resolution": resolution,
                    "duration_seconds": duration,
                    "first_frame_pts": frame.pts,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if container is not None:
                container.close()

    audit.section(
        f"video_{name}",
        {
            "root": str(root),
            "total_files_seen": total,
            "sample_size": len(selected),
            "decoded": decoded,
            "decode_failures": failures,
            "codec_counts": summarize_counter(codecs),
            "resolution_counts": summarize_counter(resolutions),
            "duration_min": min(durations) if durations else None,
            "duration_median": statistics.median(durations) if durations else None,
            "duration_max": max(durations) if durations else None,
        },
    )
    if failures:
        audit.issue("error", f"{name}_video_decode", "Video samples failed to decode", failures)


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def audit_humanml3d_captions(audit: Audit) -> None:
    root = audit.args.dataset_root / "humanml3d/captions"
    motion_root = audit.args.dataset_root / "humanml3d/motion"
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".txt")
    sample_rng = stable_rng(audit.args.seed, "caption:humanml3d")
    selected = sorted(sample_rng.sample(files, min(64, len(files))))
    empty: list[str] = []
    decode_failures: list[dict[str, str]] = []
    missing_motion: list[str] = []
    line_counts: list[int] = []
    for path in selected:
        try:
            text = read_utf8(path)
        except Exception as exc:  # noqa: BLE001
            decode_failures.append({"path": str(path), "error": str(exc)})
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        line_counts.append(len(lines))
        if not lines:
            empty.append(path.name)
        stem = path.stem[1:] if path.stem.startswith("M") else path.stem
        if not (motion_root / f"{stem}.npy").is_file():
            missing_motion.append(path.name)

    audit.section(
        "caption_humanml3d",
        {
            "total_caption_files": len(files),
            "sample_size": len(selected),
            "empty_samples": empty,
            "utf8_failures": decode_failures,
            "missing_motion_for_samples": missing_motion,
            "nonempty_lines_min": min(line_counts) if line_counts else None,
            "nonempty_lines_median": statistics.median(line_counts) if line_counts else None,
            "nonempty_lines_max": max(line_counts) if line_counts else None,
        },
    )
    if decode_failures or empty:
        audit.issue("error", "humanml3d_caption_read", "HumanML3D captions are unreadable/empty", decode_failures + empty)
    if missing_motion:
        audit.issue(
            "warning",
            "humanml3d_caption_alignment",
            "Sampled HumanML3D caption names do not map to motion files using M-prefix removal",
            missing_motion,
        )


def audit_sonic_captions(audit: Audit) -> None:
    path = audit.args.dataset_root / "sonic/captions/seed_metadata_v002_temporal_labels.jsonl"
    motion_root = audit.args.dataset_root / "sonic/motion"
    rows = 0
    parse_failures: list[dict[str, Any]] = []
    invalid_events: list[dict[str, Any]] = []
    missing_motion: list[str] = []
    for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        rows += 1
        try:
            record = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            parse_failures.append({"line": line_number, "error": str(exc)})
            continue
        filename = record.get("filename")
        events = record.get("events")
        if not isinstance(filename, str) or not filename:
            invalid_events.append({"line": line_number, "reason": "missing filename"})
            continue
        if not (motion_root / f"{filename}.npy").is_file():
            missing_motion.append(filename)
        if not isinstance(events, list) or record.get("num_events") != len(events):
            invalid_events.append({"line": line_number, "reason": "event count mismatch"})
            continue
        previous_end = -math.inf
        for event in events:
            start = event.get("start_time") if isinstance(event, dict) else None
            end = event.get("end_time") if isinstance(event, dict) else None
            description = event.get("description") if isinstance(event, dict) else None
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or start < 0
                or end <= start
                or start < previous_end
                or not isinstance(description, str)
                or not description.strip()
            ):
                invalid_events.append({"line": line_number, "reason": "invalid event interval/description"})
                break
            previous_end = end

    audit.section(
        "caption_sonic",
        {
            "rows": rows,
            "parse_failures": parse_failures[:20],
            "invalid_event_rows": invalid_events[:20],
            "missing_motion_count": len(missing_motion),
            "missing_motion_examples": missing_motion[:20],
        },
    )
    if parse_failures or invalid_events:
        audit.issue("error", "sonic_caption_schema", "SONIC caption JSONL contains invalid rows", {"parse": parse_failures[:20], "events": invalid_events[:20]})
    if missing_motion:
        audit.issue(
            "warning",
            "sonic_caption_alignment",
            "SONIC caption rows have no exact filename.npy match",
            {"count": len(missing_motion), "examples": missing_motion[:20]},
        )


def audit_motionx_captions(audit: Audit) -> None:
    frame_root = audit.args.dataset_root / "motionx/captions/frame/motion_script_only"
    motion_root = audit.args.dataset_root / "motionx/motion"
    frame_files = sorted(path for path in frame_root.iterdir() if path.is_file() and path.suffix == ".txt")
    rng = stable_rng(audit.args.seed, "caption:motionx_frame")
    selected = sorted(rng.sample(frame_files, min(64, len(frame_files))))
    empty: list[str] = []
    utf8_failures: list[dict[str, str]] = []
    missing_motion: list[str] = []
    for path in selected:
        try:
            if not read_utf8(path).strip():
                empty.append(path.name)
        except Exception as exc:  # noqa: BLE001
            utf8_failures.append({"path": path.name, "error": str(exc)})
        if not (motion_root / f"{path.stem}.npy").is_file():
            missing_motion.append(path.name)

    original_jsonl = audit.args.dataset_root / "motionx/captions/original/descriptions.jsonl"
    original_rows = 0
    original_parse_failures: list[dict[str, Any]] = []
    original_limit = 10_000
    with original_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > original_limit:
                break
            if not line.strip():
                continue
            original_rows += 1
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001
                original_parse_failures.append({"line": line_number, "error": str(exc)})

    complex_root = audit.args.dataset_root / "motionx/captions/complex"
    complex_selected = first_files(complex_root, (".txt", ".json", ".jsonl"), 32)
    complex_failures: list[dict[str, str]] = []
    complex_empty: list[str] = []
    for path in complex_selected:
        try:
            data = path.read_bytes()
            if not data:
                complex_empty.append(str(path.relative_to(complex_root)))
            data.decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            complex_failures.append({"path": str(path.relative_to(complex_root)), "error": str(exc)})

    audit.section(
        "caption_motionx",
        {
            "frame_total_files": len(frame_files),
            "frame_sample_size": len(selected),
            "frame_empty_samples": empty,
            "frame_utf8_failures": utf8_failures,
            "frame_missing_motion_count": len(missing_motion),
            "frame_missing_motion_examples": missing_motion[:20],
            "original_jsonl_rows_checked": original_rows,
            "original_jsonl_scope_limited": True,
            "original_jsonl_parse_failures": original_parse_failures,
            "complex_files_checked": len(complex_selected),
            "complex_scope_limited": True,
            "complex_empty": complex_empty,
            "complex_utf8_failures": complex_failures,
        },
    )
    if empty or utf8_failures:
        audit.issue("error", "motionx_frame_caption_read", "Motion-X frame captions are empty/unreadable", empty + utf8_failures)
    if missing_motion:
        audit.issue(
            "warning",
            "motionx_frame_caption_alignment",
            "Sampled frame caption names have no exact Motion-X motion filename match",
            {"count": len(missing_motion), "examples": missing_motion[:20]},
        )
    if original_parse_failures:
        audit.issue("error", "motionx_original_jsonl", "Motion-X original JSONL parse failures", original_parse_failures)
    if complex_empty or complex_failures:
        audit.issue("warning", "motionx_complex_caption_read", "Limited complex-caption sample contains empty/unreadable files", {"empty": complex_empty, "failures": complex_failures})


def load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"expected JSON array: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            result.append(value)
    return result


def answer_label(record: dict[str, Any]) -> str | None:
    candidates: list[Any] = [record.get("answer"), record.get("solution")]
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if isinstance(item, dict) and item.get("from") == "gpt":
                candidates.append(item.get("value"))
    for candidate in candidates:
        if isinstance(candidate, str):
            match = ANSWER_RE.search(candidate)
            if match:
                return match.group(1).upper()
    return None


def user_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if isinstance(item, dict) and item.get("from") == "human" and isinstance(item.get("value"), str):
                return item["value"]
    messages = record.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    value.get("text")
                    for value in content
                    if isinstance(value, dict) and isinstance(value.get("text"), str)
                ]
                return "\n".join(parts)
    return ""


def problem_signature(record: dict[str, Any]) -> str:
    text = user_text(record)
    if "Question:" in text:
        text = text.split("Question:", 1)[1]
    text = text.replace("<video>", " ")
    text = text.replace("<motion_start>", " ").replace("<motion>", " ").replace("<motion_end>", " ")
    return WS_RE.sub(" ", text).strip().lower()


def base_id(record: dict[str, Any]) -> str:
    group = record.get("group_id")
    if not isinstance(group, str):
        return ""
    return BASE_ID_RE.sub("", group)


def resolve_media_path(raw: str, qwen_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else qwen_root / path


def record_media_paths(record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("video", "motion", "source_video", "source_motion"):
        value = record.get(key)
        if isinstance(value, str) and value:
            result.append(value)
    videos = record.get("videos")
    if isinstance(videos, list):
        result.extend(value for value in videos if isinstance(value, str) and value)
    return sorted(set(result))


def summarize_records(
    audit: Audit,
    name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_ids: list[str] = [str(record.get("sample_id", "")) for record in records]
    groups: list[str] = [str(record.get("group_id", "")) for record in records]
    labels = collections.Counter(filter(None, (answer_label(record) for record in records)))
    branches = collections.Counter(str(record.get("branch", "")) for record in records)
    missing_assets: list[dict[str, str]] = []
    for record in records:
        for raw in record_media_paths(record):
            if not resolve_media_path(raw, audit.args.qwen_repo_root).is_file():
                missing_assets.append({"sample_id": str(record.get("sample_id", "")), "path": raw})
                if len(missing_assets) >= 50:
                    break
        if len(missing_assets) >= 50:
            break
    invalid_answers = [str(record.get("sample_id", "")) for record in records if answer_label(record) is None]
    empty_user_text = [str(record.get("sample_id", "")) for record in records if not user_text(record).strip()]
    legacy_motion_anchors = sum("<motion_start><motion><motion_end>" in user_text(record) for record in records)
    result = {
        "rows": len(records),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_group_ids": len(set(groups)),
        "duplicate_sample_ids": sorted(key for key, value in collections.Counter(sample_ids).items() if key and value > 1)[:50],
        "label_counts": summarize_counter(labels),
        "branch_counts": summarize_counter(branches),
        "invalid_answer_count": len(invalid_answers),
        "invalid_answer_examples": invalid_answers[:20],
        "empty_user_text_count": len(empty_user_text),
        "missing_asset_count_capped": len(missing_assets),
        "missing_asset_examples": missing_assets[:20],
        "legacy_motion_anchor_rows": legacy_motion_anchors,
    }
    if result["duplicate_sample_ids"]:
        audit.issue("error", f"{name}_duplicate_ids", f"{name} has duplicate sample_id values", result["duplicate_sample_ids"])
    if invalid_answers:
        audit.issue("error", f"{name}_answers", f"{name} has invalid/missing A-D answers", invalid_answers[:20])
    if empty_user_text:
        audit.issue("error", f"{name}_empty_prompt", f"{name} has empty user prompts", empty_user_text[:20])
    if missing_assets:
        audit.issue("error", f"{name}_missing_assets", f"{name} references missing media", missing_assets[:20])
    return result


def audit_qwen(audit: Audit) -> None:
    root = audit.args.dataset_root / "qwen_qa/source_tree"
    strict = root / "clean_keepbench_balanced_20260722"
    qtext = root / "clean_qtext_keepbench_balanced_20260722"
    benchmark = root / "benchmark_clean_keepbench_20260722/text"

    datasets: dict[str, list[dict[str, Any]]] = {
        "strict_train_vm": load_json_array(strict / "sft/motionx_qa_train_vm_keepbench_clean_balanced.json"),
        "strict_train_v": load_json_array(strict / "sft/motionx_qa_train_v_keepbench_clean_balanced.json"),
        "strict_train_direct": load_json_array(strict / "sft/motionx_qa_train_direct_v_vm_keepbench_clean_balanced.json"),
        "strict_val_vm": load_json_array(strict / "sft/motionx_qa_val_vm_keepbench_clean_balanced.json"),
        "strict_val_v": load_json_array(strict / "sft/motionx_qa_val_v_keepbench_clean_balanced.json"),
        "strict_val_direct": load_json_array(strict / "sft/motionx_qa_val_direct_v_vm_keepbench_clean_balanced.json"),
        "qtext_train_vm": load_json_array(qtext / "sft/motionx_qa_train_vm_qtext_keepbench_balanced.json"),
        "qtext_val_vm": load_json_array(qtext / "sft/motionx_qa_val_vm_qtext_keepbench_balanced.json"),
        "benchmark_vm": load_jsonl(benchmark / "QA/QA_500.jsonl"),
        "benchmark_m": load_jsonl(benchmark / "QA_motion_only/QA_motion_only_500.jsonl"),
        "benchmark_v": load_jsonl(benchmark / "QA_v_only/QA_v_only_500.jsonl"),
    }
    summaries = {name: summarize_records(audit, name, records) for name, records in datasets.items()}

    expected_rows = {
        "strict_train_vm": 813,
        "strict_train_v": 813,
        "strict_train_direct": 1626,
        "strict_val_vm": 86,
        "strict_val_v": 86,
        "strict_val_direct": 172,
        "qtext_train_vm": 1768,
        "qtext_val_vm": 86,
        "benchmark_vm": 500,
        "benchmark_m": 500,
        "benchmark_v": 500,
    }
    count_mismatches = {
        name: {"expected": expected, "actual": len(datasets[name])}
        for name, expected in expected_rows.items()
        if len(datasets[name]) != expected
    }
    if count_mismatches:
        audit.issue("error", "qwen_documented_counts", "Qwen dataset row counts changed", count_mismatches)

    train = datasets["strict_train_vm"]
    val = datasets["strict_val_vm"]
    bench = datasets["benchmark_vm"]
    qtext_train = datasets["qtext_train_vm"]

    def value_set(records: Iterable[dict[str, Any]], fn: Any) -> set[str]:
        return {value for record in records if (value := fn(record))}

    leakage = {
        "strict_train_val_group_overlap": sorted(value_set(train, lambda r: str(r.get("group_id", ""))) & value_set(val, lambda r: str(r.get("group_id", ""))))[:50],
        "strict_train_benchmark_base_overlap": sorted(value_set(train, base_id) & value_set(bench, base_id))[:50],
        "strict_val_benchmark_base_overlap": sorted(value_set(val, base_id) & value_set(bench, base_id))[:50],
        "strict_train_benchmark_signature_overlap": sorted(value_set(train, problem_signature) & value_set(bench, problem_signature))[:20],
        "strict_val_benchmark_signature_overlap": sorted(value_set(val, problem_signature) & value_set(bench, problem_signature))[:20],
        "qtext_train_benchmark_base_overlap_count": len(value_set(qtext_train, base_id) & value_set(bench, base_id)),
        "qtext_train_benchmark_signature_overlap_count": len(value_set(qtext_train, problem_signature) & value_set(bench, problem_signature)),
    }
    strict_leaks = {
        key: value
        for key, value in leakage.items()
        if key.startswith("strict_") and isinstance(value, list) and value
    }
    if strict_leaks:
        audit.issue("error", "qwen_strict_leakage", "Strict Qwen split has train/val/benchmark overlap", strict_leaks)

    def pair_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_group: dict[str, set[str]] = collections.defaultdict(set)
        for record in records:
            by_group[str(record.get("group_id", ""))].add(str(record.get("branch", "")))
        return [
            {"group_id": group, "branches": sorted(branches)}
            for group, branches in sorted(by_group.items())
            if branches != {"v", "vm"}
        ][:50]

    pairing = {
        "strict_train_direct_failures": pair_failures(datasets["strict_train_direct"]),
        "strict_val_direct_failures": pair_failures(datasets["strict_val_direct"]),
    }
    if pairing["strict_train_direct_failures"] or pairing["strict_val_direct_failures"]:
        audit.issue("error", "qwen_branch_pairing", "Direct V+VM datasets are not exactly paired", pairing)

    label_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in val:
        label = answer_label(record)
        if label:
            label_buckets[label].append(record)
    rng = stable_rng(audit.args.seed, "qwen:stratified_vm8")
    subset: list[dict[str, Any]] = []
    for label in "ABCD":
        bucket = label_buckets[label]
        if len(bucket) < 2:
            audit.issue("error", "qwen_vm8_stratification", f"Not enough {label} examples for stratified subset", len(bucket))
            subset.extend(bucket)
        else:
            subset.extend(rng.sample(bucket, 2))
    subset.sort(key=lambda record: str(record.get("sample_id", "")))
    subset_path = audit.args.output_dir / "qwen_vm_val_stratified_8.json"
    subset_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit.section(
        "qwen_qa",
        {
            "datasets": summaries,
            "expected_count_mismatches": count_mismatches,
            "leakage": leakage,
            "pairing": pairing,
            "stratified_subset": {
                "path": str(subset_path),
                "rows": len(subset),
                "sample_ids": [record.get("sample_id") for record in subset],
                "label_counts": summarize_counter(collections.Counter(answer_label(record) or "invalid" for record in subset)),
                "sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            },
        },
    )


def write_report(audit: Audit) -> None:
    output = audit.args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "data_sanity_report.json"
    report_path.write_text(
        json.dumps(audit.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    severity_counts = collections.Counter(item["severity"] for item in audit.report["issues"])
    qwen = audit.report["sections"].get("qwen_qa", {})
    leakage = qwen.get("leakage", {}) if isinstance(qwen, dict) else {}
    lines = [
        "# Dataset sanity experiment",
        "",
        f"- Seed: `{audit.args.seed}`",
        f"- Read-only: `true`",
        f"- Issues: `{dict(sorted(severity_counts.items()))}`",
        "",
        "## Important split checks",
        "",
        f"- strict train/val group overlap: `{len(leakage.get('strict_train_val_group_overlap', []))}`",
        f"- strict train/benchmark base overlap: `{len(leakage.get('strict_train_benchmark_base_overlap', []))}`",
        f"- strict train/benchmark signature overlap: `{len(leakage.get('strict_train_benchmark_signature_overlap', []))}`",
        f"- qtext train/benchmark base overlap: `{leakage.get('qtext_train_benchmark_base_overlap_count')}`",
        f"- qtext train/benchmark signature overlap: `{leakage.get('qtext_train_benchmark_signature_overlap_count')}`",
        "",
        "## Issues",
        "",
    ]
    if not audit.report["issues"]:
        lines.append("No issue was detected in the audited scope.")
    else:
        lines.extend(
            f"- **{item['severity']} / {item['code']}**: {item['message']}"
            for item in audit.report["issues"]
        )
    lines.extend(
        [
            "",
            "The JSON report contains exact samples, counters, missing-path evidence and the frozen VM8 subset hash.",
            "",
        ]
    )
    (output / "data_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "severity_counts": dict(severity_counts)}, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = Audit(args)
    for path in (args.dataset_root, args.qwen_repo_root):
        if not path.is_dir():
            audit.issue("critical", "root_missing", f"Required root does not exist: {path}")
    if any(item["severity"] == "critical" for item in audit.report["issues"]):
        write_report(audit)
        return 2

    experiments = [
        ("npy_motionx", lambda: audit_npy(audit, "motionx", args.dataset_root / "motionx/motion", 263, args.npy_samples)),
        ("npy_humanml3d", lambda: audit_npy(audit, "humanml3d", args.dataset_root / "humanml3d/motion", 263, args.npy_samples)),
        ("npy_sonic", lambda: audit_npy(audit, "sonic", args.dataset_root / "sonic/motion", 263, args.npy_samples)),
        ("video_motionx", lambda: audit_videos(audit, "motionx", args.dataset_root / "motionx/videos", args.video_samples)),
        ("video_qwen_motionx374", lambda: audit_videos(audit, "qwen_motionx374", args.dataset_root / "qwen_qa/media/motionx_374/videos", args.video_samples)),
        ("video_qwen_generated", lambda: audit_videos(audit, "qwen_generated", args.dataset_root / "qwen_qa/media/generated_success_assets/videos", args.video_samples)),
        ("caption_humanml3d", lambda: audit_humanml3d_captions(audit)),
        ("caption_sonic", lambda: audit_sonic_captions(audit)),
        ("caption_motionx", lambda: audit_motionx_captions(audit)),
        ("qwen_qa", lambda: audit_qwen(audit)),
    ]
    for name, experiment in experiments:
        print(f"RUN {name}", flush=True)
        try:
            experiment()
        except Exception as exc:  # noqa: BLE001 - preserve later experiments
            audit.issue(
                "critical",
                f"experiment_{name}",
                f"Experiment {name} crashed",
                f"{type(exc).__name__}: {exc}",
            )
    write_report(audit)
    return 2 if any(item["severity"] in {"critical", "error"} for item in audit.report["issues"]) else 0


if __name__ == "__main__":
    sys.exit(main())
