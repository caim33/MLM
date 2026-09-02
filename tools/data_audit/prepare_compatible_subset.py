#!/usr/bin/env python3
"""Create a traceable current-anchor view of legacy MotionLLM QA JSON rows.

The default remains strict for motion-only/VM subsets: every row must contain
exactly one legacy anchor.  ``--allow-video-only-rows`` additionally accepts
V rows without a motion payload, which makes the tool safe for the mixed
``combined_v_vm`` SFT files without weakening validation of motion rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEGACY_ANCHOR = "<motion_start><motion><motion_end>"
CURRENT_ANCHOR = "<motion>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(value: object) -> int:
    replacements = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                count = item.count(LEGACY_ANCHOR)
                if count:
                    value[key] = item.replace(LEGACY_ANCHOR, CURRENT_ANCHOR)
                    replacements += count
            else:
                replacements += migrate(item)
    elif isinstance(value, list):
        for item in value:
            replacements += migrate(item)
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "--allow-video-only-rows",
        action="store_true",
        help=(
            "allow rows without a motion payload to remain unchanged; motion "
            "rows must still contain exactly one legacy anchor"
        ),
    )
    args = parser.parse_args()

    source_rows = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise TypeError("expected a top-level JSON list")

    changed_rows = 0
    replacements = 0
    unchanged_video_rows = 0
    for row_index, row in enumerate(source_rows):
        if not isinstance(row, dict):
            raise TypeError(f"row {row_index} must be an object")
        changed = migrate(row)
        has_motion = isinstance(row.get("motion"), str) and bool(row["motion"].strip())
        if has_motion and changed != 1:
            raise ValueError(
                f"motion row {row_index} must contain exactly one legacy anchor; "
                f"found {changed}"
            )
        if not has_motion:
            if changed:
                raise ValueError(
                    f"video-only row {row_index} contains a legacy motion anchor"
                )
            if not args.allow_video_only_rows:
                raise ValueError(
                    f"row {row_index} has no motion payload; pass "
                    "--allow-video-only-rows only for a mixed V/VM file"
                )
            unchanged_video_rows += 1
        replacements += changed
        changed_rows += int(changed > 0)

    if not args.allow_video_only_rows and (
        changed_rows != len(source_rows) or replacements != len(source_rows)
    ):
        raise ValueError(
            f"expected exactly one legacy anchor per row; rows={len(source_rows)}, "
            f"changed_rows={changed_rows}, replacements={replacements}"
        )

    args.output.write_text(
        json.dumps(source_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "operation": "replace exact legacy motion anchor with current standalone anchor",
        "legacy_anchor": LEGACY_ANCHOR,
        "current_anchor": CURRENT_ANCHOR,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "row_count": len(source_rows),
        "changed_rows": changed_rows,
        "unchanged_video_rows": unchanged_video_rows,
        "replacement_count": replacements,
    }
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
