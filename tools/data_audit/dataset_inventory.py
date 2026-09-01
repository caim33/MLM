#!/usr/bin/env python3
"""Full file-level inventory of the canonical caimeng dataset tree.

The scan never follows symbolic links, so compatibility links and Qwen media
views are counted without double-counting their target bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LEAVES = (
    "motionx/motion",
    "motionx/videos",
    "motionx/captions/complex",
    "motionx/captions/original",
    "motionx/captions/frame",
    "humanml3d/motion",
    "humanml3d/captions",
    "sonic/motion",
    "sonic/captions",
    "qwen_qa/media/motionx_374",
    "qwen_qa/media/generated_success_assets",
    "qwen_qa/source_tree",
    "qwen_qa/views",
    "experiments",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inventory_tree(path: Path) -> dict[str, object]:
    extensions: Counter[str] = Counter()
    regular_files = symlinks = broken_symlinks = directories = other = 0
    regular_bytes = 0
    symlink_target_bytes = 0
    stack = [path]

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            symlinks += 1
                            suffix = Path(entry.name).suffix.lower() or "[no extension]"
                            extensions[f"symlink:{suffix}"] += 1
                            try:
                                stat = entry.stat(follow_symlinks=True)
                                if entry.is_file(follow_symlinks=True):
                                    symlink_target_bytes += stat.st_size
                            except FileNotFoundError:
                                broken_symlinks += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            directories += 1
                            stack.append(Path(entry.path))
                            continue
                        if entry.is_file(follow_symlinks=False):
                            regular_files += 1
                            stat = entry.stat(follow_symlinks=False)
                            regular_bytes += stat.st_size
                            suffix = Path(entry.name).suffix.lower() or "[no extension]"
                            extensions[suffix] += 1
                        else:
                            other += 1
                    except OSError:
                        other += 1
        except OSError as exc:
            return {
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "directories": directories,
                "regular_files": regular_files,
                "symlinks": symlinks,
                "broken_symlinks": broken_symlinks,
                "other_entries": other,
                "regular_bytes": regular_bytes,
                "symlink_target_bytes": symlink_target_bytes,
                "extensions": dict(sorted(extensions.items())),
            }

    return {
        "path": str(path),
        "directories": directories,
        "regular_files": regular_files,
        "symlinks": symlinks,
        "broken_symlinks": broken_symlinks,
        "other_entries": other,
        "regular_bytes": regular_bytes,
        "symlink_target_bytes": symlink_target_bytes,
        "extensions": dict(sorted(extensions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = utc_now()
    rows: list[dict[str, object]] = []
    for relative in LEAVES:
        path = root / relative
        print(f"SCAN_START\t{relative}", flush=True)
        row = inventory_tree(path)
        row["relative_path"] = relative
        rows.append(row)
        print(
            "SCAN_DONE\t{}\tfiles={}\tsymlinks={}\tbytes={}".format(
                relative,
                row.get("regular_files", 0),
                row.get("symlinks", 0),
                row.get("regular_bytes", 0),
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "root": str(root),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "link_policy": "do not follow symlinks; target bytes are informational only",
        "trees": rows,
    }
    json_path = args.output_dir / "dataset_full_inventory.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tsv_path = args.output_dir / "dataset_full_inventory.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "regular_files",
            "symlinks",
            "broken_symlinks",
            "directories",
            "regular_bytes",
            "symlink_target_bytes",
            "extensions_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, 0) for key in fieldnames[:-1]},
                    "extensions_json": json.dumps(
                        row.get("extensions", {}), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    print(f"WROTE\t{json_path}", flush=True)
    print(f"WROTE\t{tsv_path}", flush=True)


if __name__ == "__main__":
    main()
