#!/usr/bin/env python3
"""Collect SOTA open/API eval summaries into markdown and JSON reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
OPEN_ROOT = ROOT / "codex_runs" / "sota_open_eval_20260716"
API_ROOT = ROOT / "codex_runs" / "sota_api_eval_20260716"
OUT_JSON = ROOT / "codex_runs" / "sota_report_20260716.json"
OUT_MD = ROOT / "codex_runs" / "sota_report_20260716.md"


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*.summary.json")) + sorted(root.glob("*.summary.json")):
        try:
            s = load_summary(path)
        except Exception:
            continue
        overall = s.get("metrics", {}).get("overall", {})
        rows.append(
            {
                "summary_path": str(path),
                "model": s.get("model"),
                "provider": s.get("provider", "open"),
                "dataset": Path(str(s.get("dataset", ""))).name,
                "input_mode": s.get("input_mode") or "api_frames",
                "total": overall.get("total"),
                "correct": overall.get("correct"),
                "accuracy": overall.get("accuracy"),
                "parse_rate": overall.get("parse_rate"),
            }
        )
    return rows


def main() -> int:
    open_rows = collect(OPEN_ROOT)
    api_rows = collect(API_ROOT)
    status_path = OPEN_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    env_blockers = {
        "OPENAI_API_KEY": bool(__import__("os").environ.get("OPENAI_API_KEY")),
        "DASHSCOPE_API_KEY": bool(__import__("os").environ.get("DASHSCOPE_API_KEY")),
    }
    payload = {
        "open_results": open_rows,
        "api_results": api_rows,
        "open_status": status,
        "api_key_present": env_blockers,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# SOTA Model Evaluation Report (2026-07-16)", ""]
    lines.append("## Open-source models")
    if open_rows:
        lines.append("| model | dataset | input | correct/total | accuracy | parse | summary |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in open_rows:
            acc = row["accuracy"]
            parse = row["parse_rate"]
            lines.append(
                f"| {row['model']} | {row['dataset']} | {row['input_mode']} | "
                f"{row['correct']}/{row['total']} | {acc:.2%} | {parse:.2%} | {row['summary_path']} |"
            )
    else:
        lines.append("No completed open-source summaries yet. See `sota_open_eval_20260716/status.json`.")
    lines.append("")
    lines.append("## Closed-source API models")
    if api_rows:
        lines.append("| provider | model | dataset | correct/total | accuracy | summary |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in api_rows:
            acc = row["accuracy"]
            lines.append(
                f"| {row['provider']} | {row['model']} | {row['dataset']} | "
                f"{row['correct']}/{row['total']} | {acc:.2%} | {row['summary_path']} |"
            )
    else:
        missing = [k for k, present in env_blockers.items() if not present]
        lines.append(f"Not run yet. Missing API key env vars: {', '.join(missing) if missing else 'none'}")
    lines.append("")
    lines.append("## Current open run status")
    lines.append("```json")
    lines.append(json.dumps(status, ensure_ascii=False, indent=2))
    lines.append("```")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUT_MD))
    print(str(OUT_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
