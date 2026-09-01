"""Deterministic release tables; source evidence is validated by the controller."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from motion_eval.core import atomic_write_json, resolve_within_root, sha256_file, sha256_json


def _atomic_text(path: Path, text: str, *, root: Path) -> None:
    destination = resolve_within_root(path, root, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Revalidate after mkdir and refuse writes through links/reparse points via
    # resolve_within_root.  The final replace is same-directory and atomic.
    destination = resolve_within_root(destination, root, must_exist=False)
    descriptor, raw = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    temp = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        destination = resolve_within_root(destination, root, must_exist=False)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def build_release_files(
    release_root: str | Path,
    *,
    batch_root: str | Path,
    batch_id: str,
    batch_receipt_sha256: str,
    model_results: Sequence[Mapping[str, Any]],
    blocked_models: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(batch_root).resolve(strict=True)
    release = resolve_within_root(release_root, root, must_exist=False)
    release.mkdir(parents=True, exist_ok=True)
    release = resolve_within_root(release, root, must_exist=True)

    fields = [
        "model_id", "display_name", "modality", "evaluation_mode", "correct",
        "denominator", "accuracy", "invalid_output", "media_error", "timeout", "oom",
        "runtime_error", "predictions_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for result in model_results:
        writer.writerow({field: result.get(field, "") for field in fields})
    csv_path = release / "all_models_results.csv"
    _atomic_text(csv_path, buffer.getvalue(), root=root)

    md_lines = [
        "# Current-batch evaluation results",
        "",
        "| Model | Modality | Status | Correct / 500 | Accuracy |",
        "|---|---:|---|---:|---:|",
    ]
    for result in model_results:
        md_lines.append(
            f"| {result['display_name']} | {result['modality']} | complete | "
            f"{result['correct']} / {result['denominator']} | {result['accuracy']:.6f} |"
        )
    for blocked in blocked_models:
        md_lines.append(
            f"| {blocked['display_name']} | {blocked['modality']} | blocked | — | — |"
        )
    results_md = release / "all_models_results.md"
    _atomic_text(results_md, "\n".join(md_lines) + "\n", root=root)

    blocked_lines = ["# Evidence-backed blocked models", ""]
    if not blocked_models:
        blocked_lines.append("None.")
    else:
        for blocked in blocked_models:
            blocked_lines.extend(
                [
                    f"## {blocked['display_name']} (`{blocked['model_id']}`)",
                    "",
                    f"- Reason code: `{blocked['reason_code']}`",
                    f"- Evidence SHA-256: `{blocked['evidence_sha256']}`",
                    "",
                ]
            )
    blocked_md = release / "blocked_models.md"
    _atomic_text(blocked_md, "\n".join(blocked_lines) + "\n", root=root)

    source_models = [dict(result) for result in model_results]
    source_blocked = [dict(item) for item in blocked_models]
    manifest_body: dict[str, Any] = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "batch_receipt_sha256": batch_receipt_sha256,
        "policy": {
            "fresh_finetune_only": True,
            "historical_results_allowed": False,
            "proxy_results_allowed": False,
            "fixed_denominator": 500,
            "strict_generative_answer": "<answer>[A-D]</answer>",
        },
        "models": source_models,
        "blocked_models": source_blocked,
        "files": {
            "all_models_results.csv": sha256_file(csv_path),
            "all_models_results.md": sha256_file(results_md),
            "blocked_models.md": sha256_file(blocked_md),
        },
    }
    manifest = {**manifest_body, "manifest_sha256": sha256_json(manifest_body)}
    atomic_write_json(
        release / "evaluation_release_manifest.json", manifest, root=root, overwrite=True
    )
    return manifest
