from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from motionllm.contracts import Modality, OptionLabel
from motionllm.data import (
    DuplicateSampleIdError,
    JsonlLineError,
    JsonlOpenError,
    read_jsonl,
    read_samples_jsonl,
)


def _payload(
    sample_id: str,
    modality: str,
    *,
    video: str | None = None,
    motion: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "group_id": f"group-{sample_id}",
        "modality": modality,
        "question": "What happens?",
        "options": {label.value: f"option {label.value}" for label in OptionLabel},
        "gold": "<answer>B</answer>",
        "video": video,
        "motion": motion,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_reads_all_four_modalities_with_resolved_media(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"video")
    (tmp_path / "motion.npy").write_bytes(b"motion")
    source = tmp_path / "samples.jsonl"
    _write_rows(
        source,
        [
            _payload("v", "V", video="clip.mp4"),
            _payload("m", "M", motion="motion.npy"),
            _payload("vm", "VM", video="clip.mp4", motion="motion.npy"),
            _payload("t", "T"),
        ],
    )
    samples = read_samples_jsonl(source)
    assert [sample.modality for sample in samples] == list(Modality)
    assert samples[0].video == (tmp_path / "clip.mp4").resolve()
    assert samples[1].motion == (tmp_path / "motion.npy").resolve()


def test_explicit_legacy_branch_is_supported_but_must_agree(tmp_path: Path) -> None:
    row = _payload("legacy", "T")
    row.pop("modality")
    row["branch"] = "t"
    source = tmp_path / "samples.jsonl"
    _write_rows(source, [row])
    assert read_samples_jsonl(source)[0].modality is Modality.TEXT

    row["modality"] = "V"
    _write_rows(source, [row])
    with pytest.raises(JsonlLineError, match="modality and branch disagree"):
        read_samples_jsonl(source)


def test_jsonl_rejects_blank_rows_and_malformed_json_with_line_number(
    tmp_path: Path,
) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_text('{}\n\n{"ok": true}\n', encoding="utf-8")
    with pytest.raises(JsonlLineError) as captured:
        read_jsonl(blank)
    assert captured.value.line_number == 2

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{}\n{"broken":}\n', encoding="utf-8")
    with pytest.raises(JsonlLineError) as captured:
        read_jsonl(malformed)
    assert captured.value.line_number == 2
    assert "column" in captured.value.reason


@pytest.mark.parametrize(
    "line",
    [
        '{"sample_id":"one","sample_id":"two"}\n',
        '{"value": NaN}\n',
        '[1, 2, 3]\n',
    ],
)
def test_jsonl_rejects_duplicate_keys_nonfinite_numbers_and_nonobjects(
    tmp_path: Path, line: str
) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(line, encoding="utf-8")
    with pytest.raises(JsonlLineError) as captured:
        read_jsonl(source)
    assert captured.value.line_number == 1


def test_jsonl_rejects_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "bad-encoding.jsonl"
    source.write_bytes(b'{"ok": true}\n\xff\n')
    with pytest.raises(JsonlLineError, match="valid UTF-8"):
        read_jsonl(source)


def test_sample_contract_rejects_unpaired_unicode_surrogate(tmp_path: Path) -> None:
    source = tmp_path / "surrogate.jsonl"
    row = _payload("bad-surrogate", "T")
    row["question"] = "bad \ud800 question"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(JsonlLineError, match="control character"):
        read_samples_jsonl(source)


def test_jsonl_open_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(JsonlOpenError):
        read_jsonl(tmp_path / "missing.jsonl")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(JsonlOpenError):
        read_jsonl(directory)


def test_invalid_current_sample_is_not_replaced_by_next_row(tmp_path: Path) -> None:
    (tmp_path / "valid.mp4").write_bytes(b"video")
    source = tmp_path / "samples.jsonl"
    _write_rows(
        source,
        [
            _payload("broken", "V", video="missing.mp4"),
            _payload("would-have-been-valid", "V", video="valid.mp4"),
        ],
    )
    with pytest.raises(JsonlLineError) as captured:
        read_samples_jsonl(source)
    assert captured.value.line_number == 1
    assert captured.value.sample_id == "broken"
    assert "does not exist" in captured.value.reason


def test_duplicate_sample_id_fails_at_exact_later_row(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    _write_rows(source, [_payload("same", "T"), _payload("same", "T")])
    with pytest.raises(DuplicateSampleIdError) as captured:
        read_samples_jsonl(source)
    assert captured.value.line_number == 2
    assert captured.value.first_line_number == 1


def test_sample_reader_rejects_loose_gold_unknown_fields_and_path_escape(
    tmp_path: Path,
) -> None:
    source = tmp_path / "samples.jsonl"

    loose_gold = _payload("gold", "T")
    loose_gold["gold"] = "B"
    _write_rows(source, [loose_gold])
    with pytest.raises(JsonlLineError, match="gold must be exactly"):
        read_samples_jsonl(source)

    unknown = _payload("unknown", "T")
    unknown["vidoe"] = "typo.mp4"
    _write_rows(source, [unknown])
    with pytest.raises(JsonlLineError, match="unknown fields"):
        read_samples_jsonl(source)

    escaping = _payload("escape", "V", video="../outside.mp4")
    _write_rows(source, [escaping])
    with pytest.raises(JsonlLineError, match="outside the declared root"):
        read_samples_jsonl(source)


def test_option_array_requires_explicit_labels_and_is_canonicalized(
    tmp_path: Path,
) -> None:
    row = _payload("array", "T")
    row["options"] = [
        {"label": label.value, "text": f"option {label.value}"}
        for label in reversed(list(OptionLabel))
    ]
    source = tmp_path / "samples.jsonl"
    _write_rows(source, [row])
    sample = read_samples_jsonl(source)[0]
    assert [option.label for option in sample.options] == list(OptionLabel)


def test_missing_media_can_be_deferred_only_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    _write_rows(source, [_payload("deferred", "V", video="later.mp4")])
    sample = read_samples_jsonl(source, check_media_exists=False)[0]
    assert sample.video == (tmp_path / "later.mp4").resolve()
