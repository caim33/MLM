#!/usr/bin/env python3
"""Convert the MotionX prod200 CoT tar.zst archive into QA-generator JSONL."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RUN_PREFIX = "PRODUCTION/runs/motionx_deepseek_prod200_20260414/"


def run_tar(args: List[str], *, text: bool = True) -> str:
    proc = subprocess.run(
        ["tar", "--zstd", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "tar failed: "
            + " ".join(["tar", "--zstd", *args])
            + "\n"
            + proc.stderr.decode("utf-8", errors="replace")
        )
    return proc.stdout.decode("utf-8", errors="replace") if text else ""


def iter_json_objects_from_process(proc: subprocess.Popen[bytes]) -> Iterable[Dict[str, Any]]:
    if proc.stdout is None:
        raise RuntimeError("tar stdout pipe was not created")
    stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    buf: List[str] = []
    started = False
    depth = 0
    in_string = False
    escape = False

    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        for ch in chunk:
            if not started:
                if ch.isspace():
                    continue
                if ch != "{":
                    raise RuntimeError(f"unexpected tar JSON stream character: {ch!r}")
                started = True
                depth = 1
                in_string = False
                escape = False
                buf = [ch]
                continue

            buf.append(ch)
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    data = json.loads("".join(buf))
                    if not isinstance(data, dict):
                        raise ValueError("tar JSON stream object is not a JSON object")
                    yield data
                    started = False
                    buf = []

    if started:
        raise RuntimeError("tar JSON stream ended inside an object")


def iter_json_objects_from_tar(archive: Path, members: List[str]) -> Iterable[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        member_file = Path(f.name)
        for member in members:
            f.write(member + "\n")

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            ["tar", "--zstd", "-xOf", str(archive), "-T", str(member_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield from iter_json_objects_from_process(proc)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"tar stream failed rc={rc}\n{stderr}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        member_file.unlink(missing_ok=True)


def iter_json_objects_from_tar_wildcard(archive: Path, pattern: str) -> Iterable[Dict[str, Any]]:
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            ["tar", "--zstd", "-xOf", str(archive), "--wildcards", pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield from iter_json_objects_from_process(proc)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"tar wildcard stream failed rc={rc}\n{stderr}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()


def extract_archive_members(archive: Path, extract_dir: Path, members: List[str]) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        member_file = Path(f.name)
        for member in members:
            f.write(member + "\n")
    try:
        run_tar(["-xf", str(archive), "-C", str(extract_dir), "-T", str(member_file)], text=False)
    finally:
        member_file.unlink(missing_ok=True)


def read_json_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def clean_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time_range": segment.get("time_range") or segment.get("time") or "",
        "cot_type": segment.get("cot_type") or "",
        "think": segment.get("think") or "",
        "answer": segment.get("answer") or "",
    }


def build_row_from_cot(
    cot: Dict[str, Any],
    *,
    source_index: int,
    source_member: str = "",
) -> Dict[str, Any]:
    sample_id = str(cot.get("sample_id") or "")
    segments = cot.get("per_segment")
    if not isinstance(segments, list):
        segments = []
    return {
        "source_index": source_index,
        "sample_id": sample_id,
        "source_archive_member": source_member,
        "description_json": {
            "sample_summary": cot.get("sample_summary") or "",
            "per_segment": [clean_segment(seg) for seg in segments if isinstance(seg, dict)],
            "final_answer": cot.get("final_answer") or "",
        },
    }


def build_row(step2_path: Path, *, source_index: int, root: Path) -> Dict[str, Any]:
    inputs_path = step2_path.parent / "inputs.json"
    cot = read_json_file(step2_path)
    inputs = read_json_file(inputs_path)

    sample = inputs.get("sample") if isinstance(inputs.get("sample"), dict) else {}
    sample_id = str(cot.get("sample_id") or sample.get("shorthand") or step2_path.parent.name)
    segments = cot.get("per_segment")
    if not isinstance(segments, list):
        segments = []

    return {
        "source_index": source_index,
        "sample_id": sample_id,
        "video_name": sample.get("video_name") or inputs.get("video_name") or "",
        "motion_name": sample.get("motion_name") or inputs.get("motion_name") or "",
        "full_id": sample.get("full_id") or "",
        "source_archive_member": step2_path.relative_to(root).as_posix(),
        "description_json": {
            "sample_summary": cot.get("sample_summary") or "",
            "per_segment": [clean_segment(seg) for seg in segments if isinstance(seg, dict)],
            "final_answer": cot.get("final_answer") or "",
        },
        "metadata": {
            "overview": inputs.get("overview") or "",
            "key_summary": inputs.get("key_summary") or "",
        },
    }


def validate_row(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    desc = row.get("description_json")
    if not row.get("sample_id"):
        errors.append("missing_sample_id")
    if not isinstance(desc, dict):
        return ["missing_description_json"]
    if not str(desc.get("sample_summary") or "").strip():
        errors.append("missing_sample_summary")
    segments = desc.get("per_segment")
    if not isinstance(segments, list) or not segments:
        errors.append("missing_per_segment")
    if not str(desc.get("final_answer") or "").strip():
        errors.append("missing_final_answer")
    return errors


def write_jsonl(rows: Iterable[Dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--extract-inputs",
        action="store_true",
        help="Also extract inputs.json metadata. Slower and not needed for QA generation.",
    )
    parser.add_argument(
        "--tar-wildcard-stream",
        action="store_true",
        help="Use GNU tar --wildcards to stream */step2_generation.json. Use on Linux/GNU tar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)

    listing = run_tar(["-tf", str(args.archive)])
    step2_members = sorted(path for path in listing.splitlines() if path.endswith("/step2_generation.json"))
    if args.limit:
        step2_members = step2_members[: args.limit]
    if not step2_members:
        raise RuntimeError("no step2_generation.json members found")

    if not args.extract_inputs:
        member_by_sample_id = {Path(member).parent.name: member for member in step2_members}
        rows = []
        bad = []
        iterator = (
            iter_json_objects_from_tar_wildcard(args.archive, "*/step2_generation.json")
            if args.tar_wildcard_stream and not args.limit
            else iter_json_objects_from_tar(args.archive, step2_members)
        )
        for idx, cot in enumerate(iterator):
            row = build_row_from_cot(
                cot,
                source_index=idx,
                source_member=member_by_sample_id.get(str(cot.get("sample_id") or ""), ""),
            )
            errors = validate_row(row)
            if errors:
                bad.append({"sample_id": row.get("sample_id"), "errors": errors})
            rows.append(row)
        count = write_jsonl(rows, args.output)
        summary = {
            "archive": str(args.archive),
            "output": str(args.output),
            "rows": count,
            "expected_rows": len(step2_members),
            "invalid_rows": len(bad),
            "invalid_examples": bad[:10],
            "mode": "stream_step2_only",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if bad or count != len(step2_members) else 0

    temp_context: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.extract_dir:
        extract_dir = args.extract_dir
        if extract_dir.exists() and any(extract_dir.iterdir()):
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="motionx_prod200_cot_")
        extract_dir = Path(temp_context.name)

    try:
        members = []
        for step2_member in step2_members:
            members.append(step2_member)
            members.append(step2_member.rsplit("/", 1)[0] + "/inputs.json")
        extract_archive_members(args.archive, extract_dir, members)
        step2_paths = sorted(extract_dir.rglob("step2_generation.json"))
        if not step2_paths:
            raise RuntimeError("no step2_generation.json members found")

        rows: List[Dict[str, Any]] = []
        bad: List[Dict[str, Any]] = []
        for idx, path in enumerate(step2_paths):
            row = build_row(path, source_index=idx, root=extract_dir)
            errors = validate_row(row)
            if errors:
                bad.append({"member": path.relative_to(extract_dir).as_posix(), "errors": errors})
            rows.append(row)
    finally:
        if temp_context is not None:
            temp_context.cleanup()
        elif args.extract_dir and not args.keep_extracted:
            shutil.rmtree(args.extract_dir, ignore_errors=True)

    count = write_jsonl(rows, args.output)
    summary = {
        "archive": str(args.archive),
        "output": str(args.output),
        "rows": count,
        "invalid_rows": len(bad),
        "invalid_examples": bad[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
