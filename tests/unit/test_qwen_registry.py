from __future__ import annotations

import json
from pathlib import Path

import pytest

from motionllm.qwen.registry import (
    DATASET_CONFIG_DIR_ENV,
    DatasetRegistryError,
    default_config_dir,
    validate_motion_normalization_binding,
)


REPO = Path(__file__).resolve().parents[2]


def test_default_config_dir_resolves_repository_configs(monkeypatch) -> None:
    monkeypatch.delenv(DATASET_CONFIG_DIR_ENV, raising=False)
    assert default_config_dir() == (REPO / "configs" / "datasets").resolve()


def _dataset_config(
    root: Path,
    *,
    name: str,
    annotation: Path,
    media: Path,
    mean: Path,
    std: Path,
    dim: int = 263,
) -> None:
    payload = {
        "schema_version": 1,
        "name": name,
        "annotation_path": str(annotation.resolve()),
        "media_root": str(media.resolve()),
        "split": "train",
        "motion_mean_path": str(mean.resolve()),
        "motion_std_path": str(std.resolve()),
        "expected_motion_dim": dim,
    }
    (root / f"{name}.dataset.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_normalization_binding_matches_dataset_model_and_cli(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    media = tmp_path / "media"
    config_dir.mkdir()
    media.mkdir()
    annotation = tmp_path / "train.jsonl"
    annotation.write_text("{}\n", encoding="utf-8")
    mean = tmp_path / "Mean.npy"
    std = tmp_path / "Std.npy"
    mean.write_bytes(b"mean")
    std.write_bytes(b"std")
    _dataset_config(
        config_dir,
        name="motion_train",
        annotation=annotation,
        media=media,
        mean=mean,
        std=std,
    )

    assert validate_motion_normalization_binding(
        ["motion_train"],
        motion_mean_path=mean,
        motion_std_path=std,
        expected_motion_dim=263,
        config_dir=config_dir,
    ) == (mean.resolve(), std.resolve())


def test_normalization_binding_rejects_partial_and_mismatched_assets(tmp_path) -> None:
    with pytest.raises(DatasetRegistryError, match="provided together"):
        validate_motion_normalization_binding(
            ["motion_train"],
            motion_mean_path=tmp_path / "Mean.npy",
            motion_std_path=None,
            config_dir=tmp_path,
        )

    config_dir = tmp_path / "configs"
    media = tmp_path / "media"
    config_dir.mkdir()
    media.mkdir()
    annotation = tmp_path / "train.jsonl"
    annotation.write_text("{}\n", encoding="utf-8")
    mean = tmp_path / "Mean.npy"
    other_mean = tmp_path / "OtherMean.npy"
    std = tmp_path / "Std.npy"
    mean.write_bytes(b"mean")
    other_mean.write_bytes(b"other")
    std.write_bytes(b"std")
    _dataset_config(
        config_dir,
        name="motion_train",
        annotation=annotation,
        media=media,
        mean=mean,
        std=std,
    )

    with pytest.raises(DatasetRegistryError, match="does not match"):
        validate_motion_normalization_binding(
            ["motion_train"],
            motion_mean_path=other_mean,
            motion_std_path=std,
            expected_motion_dim=263,
            config_dir=config_dir,
        )
