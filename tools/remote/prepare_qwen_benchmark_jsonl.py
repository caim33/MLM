#!/usr/bin/env python3
"""Normalize Motion-R1 benchmark jsonl paths for standalone evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def abs_under(root: str, path: str | None) -> str | None:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def normalize_item(item: Dict[str, Any], root: str) -> Dict[str, Any]:
    out = dict(item)
    if "video" in out:
        out["video"] = abs_under(root, out["video"])
    if "motion" in out:
        out["motion"] = abs_under(root, out["motion"])
    for key in ["source_video", "source_motion"]:
        if key in out:
            out[key] = abs_under(root, out[key])
    messages = []
    for msg in out.get("messages", []):
        msg2 = dict(msg)
        content = []
        for part in msg.get("content", []):
            if isinstance(part, dict):
                part2 = dict(part)
                if part2.get("type") == "video" and "video" in part2:
                    part2["video"] = abs_under(root, part2["video"])
                content.append(part2)
            else:
                content.append(part)
        msg2["content"] = content
        messages.append(msg2)
    out["messages"] = messages
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing = []
    with open(args.input, "r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = normalize_item(json.loads(line), args.root)
            for key in ["video", "motion"]:
                path = item.get(key)
                if path and not os.path.exists(path):
                    missing.append({"sample_id": item.get("sample_id"), "key": key, "path": path})
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    report = {"input": args.input, "output": args.output, "root": args.root, "count": count, "missing": missing[:20], "missing_count": len(missing)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
