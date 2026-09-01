from __future__ import annotations

from pathlib import Path

import pytest

from motion_eval.data import StrictJsonError, load_json_strict, load_jsonl_strict
from motionllm.data import JsonlLineError, read_jsonl


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_controller_json_rejects_nonfinite_values_at_any_depth(
    tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(
        '{"outer":{"items":[0,{"score":' + constant + "}]}}",
        encoding="utf-8",
    )

    with pytest.raises(StrictJsonError, match="non-finite"):
        load_json_strict(path)


def test_nested_duplicate_key_is_never_hidden_by_an_outer_object(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer":{"bound":1,"bound":2}}', encoding="utf-8")

    with pytest.raises(StrictJsonError, match="duplicate JSON key: 'bound'"):
        load_json_strict(path)


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"", "empty"),
        (b"\n", "blank JSONL row"),
        (b'{"row":1}\n\n{"row":2}\n', "blank JSONL row"),
        (b"[1,2,3]\n", "must be an object"),
        (b'{"row":"\xff"}\n', "UTF-8"),
    ],
)
def test_controller_jsonl_rejects_empty_blank_scalar_and_invalid_utf8(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(payload)

    with pytest.raises(StrictJsonError, match=message):
        load_jsonl_strict(path)


def test_very_long_but_valid_json_string_is_not_truncated(tmp_path: Path) -> None:
    path = tmp_path / "long.json"
    value = "motion-evidence-" * 65_536  # Just over one MiB of UTF-8 text.
    path.write_text('{"value":"' + value + '"}', encoding="utf-8")

    assert load_json_strict(path) == {"value": value}


def test_deep_controller_json_fails_with_typed_source_error(tmp_path: Path) -> None:
    path = tmp_path / "deep-controller.json"
    payload_marker = "adversarial-payload-must-not-be-echoed"
    nested = "[" * 10_000 + f'"{payload_marker}"' + "]" * 10_000
    path.write_text(nested, encoding="utf-8")

    with pytest.raises(StrictJsonError) as captured:
        load_json_strict(path)

    assert str(path) in str(captured.value)
    assert payload_marker not in str(captured.value)
    assert isinstance(captured.value.__cause__, RecursionError)
    assert payload_marker not in str(captured.value.__cause__)


def test_deep_dataset_jsonl_fails_with_typed_line_error(tmp_path: Path) -> None:
    path = tmp_path / "deep-dataset.jsonl"
    payload_marker = "dataset-payload-must-not-be-echoed"
    nested = "[" * 10_000 + f'"{payload_marker}"' + "]" * 10_000
    path.write_text('{"nested":' + nested + "}\n", encoding="utf-8")

    with pytest.raises(JsonlLineError) as captured:
        read_jsonl(path)

    rendered = str(captured.value)
    assert str(path.resolve()) in rendered
    assert "line 1" in rendered.lower() or ":1" in rendered
    assert payload_marker not in rendered
    assert isinstance(captured.value.__cause__, RecursionError)
    assert payload_marker not in str(captured.value.__cause__)
