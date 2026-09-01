from __future__ import annotations

import json
from pathlib import Path

import pytest

from motionllm.contracts import OptionLabel, SampleContractError
from motionllm.data import JsonlLineError, SampleDataset


def _text_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_id": "group",
        "modality": "T",
        "question": "Choose one.",
        "options": {label.value: f"option {label.value}" for label in OptionLabel},
        "gold": "<answer>A</answer>",
        "video": None,
        "motion": None,
    }


def test_dataset_keeps_exact_index_identity(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    rows = [_text_row("first"), _text_row("second")]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    dataset = SampleDataset.from_jsonl(source)
    assert dataset[0].sample_id == "first"
    assert dataset[1].sample_id == "second"
    with pytest.raises(IndexError):
        _ = dataset[2]


def test_dataset_construction_aborts_instead_of_substituting_bad_row(
    tmp_path: Path,
) -> None:
    source = tmp_path / "samples.jsonl"
    bad = _text_row("bad")
    bad["gold"] = "A"
    rows = [bad, _text_row("valid-but-must-not-substitute")]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(JsonlLineError) as captured:
        SampleDataset.from_jsonl(source)
    assert captured.value.line_number == 1
    assert captured.value.sample_id == "bad"


def test_dataset_rejects_duplicate_ids_even_for_programmatic_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.jsonl"
    source.write_text(json.dumps(_text_row("same")) + "\n", encoding="utf-8")
    sample = SampleDataset.from_jsonl(source)[0]
    with pytest.raises(SampleContractError, match="duplicate sample_id"):
        SampleDataset([sample, sample])
