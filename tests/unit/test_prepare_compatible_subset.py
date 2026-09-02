from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "data_audit" / "prepare_compatible_subset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_compatible_subset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, rows: list[dict], *extra: str) -> tuple[list[dict], dict]:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    receipt = tmp_path / "receipt.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    old_argv = sys.argv
    sys.argv = [str(SCRIPT), str(source), str(output), str(receipt), *extra]
    try:
        _load_module().main()
    finally:
        sys.argv = old_argv
    return json.loads(output.read_text(encoding="utf-8")), json.loads(
        receipt.read_text(encoding="utf-8")
    )


def test_migrates_one_anchor_per_motion_row(tmp_path: Path) -> None:
    rows, receipt = _run(
        tmp_path,
        [
            {
                "motion": "motion/a.npy",
                "conversations": [
                    {"from": "human", "value": "<motion_start><motion><motion_end>\nQ"}
                ],
            }
        ],
    )

    assert rows[0]["conversations"][0]["value"] == "<motion>\nQ"
    assert receipt["changed_rows"] == 1
    assert receipt["unchanged_video_rows"] == 0
    assert receipt["replacement_count"] == 1


def test_mixed_view_requires_explicit_video_only_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-video-only-rows"):
        _run(tmp_path, [{"video": "video/a.mp4", "conversations": []}])


def test_mixed_view_preserves_v_and_migrates_vm(tmp_path: Path) -> None:
    rows, receipt = _run(
        tmp_path,
        [
            {"branch": "v", "video": "video/a.mp4", "conversations": []},
            {
                "branch": "vm",
                "video": "video/a.mp4",
                "motion": "motion/a.npy",
                "conversations": [
                    {"from": "human", "value": "<motion_start><motion><motion_end>"}
                ],
            },
        ],
        "--allow-video-only-rows",
    )

    assert rows[0]["branch"] == "v"
    assert rows[1]["conversations"][0]["value"] == "<motion>"
    assert receipt["row_count"] == 2
    assert receipt["changed_rows"] == 1
    assert receipt["unchanged_video_rows"] == 1


def test_rejects_motion_row_without_legacy_anchor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one legacy anchor"):
        _run(
            tmp_path,
            [{"motion": "motion/a.npy", "conversations": [{"value": "<motion>"}]}],
            "--allow-video-only-rows",
        )
