from __future__ import annotations

import pytest

from motionllm.contracts import (
    GoldAnswer,
    GoldSyntaxError,
    Modality,
    ModalityContractError,
    Option,
    OptionContractError,
    OptionLabel,
    format_gold_answer,
    parse_gold_answer,
)


@pytest.mark.parametrize("value", ["A", "B", "C", "D"])
def test_option_labels_are_exact(value: str) -> None:
    assert OptionLabel.parse(value).value == value


@pytest.mark.parametrize("value", ["a", "E", " A", "A ", "AB", 0, None])
def test_option_labels_reject_noncanonical_values(value: object) -> None:
    with pytest.raises(OptionContractError):
        OptionLabel.parse(value)


def test_option_requires_typed_label_and_canonical_text() -> None:
    with pytest.raises(OptionContractError):
        Option("A", "first")  # type: ignore[arg-type]
    with pytest.raises(OptionContractError):
        Option(OptionLabel.A, " first")
    assert Option(OptionLabel.A, "first").text == "first"


@pytest.mark.parametrize("label", list(OptionLabel))
def test_strict_gold_round_trip(label: OptionLabel) -> None:
    tag = f"<answer>{label.value}</answer>"
    assert GoldAnswer.parse(tag).label is label
    assert parse_gold_answer(tag) is label
    assert format_gold_answer(label) == tag


@pytest.mark.parametrize(
    "value",
    [
        "A",
        "<answer> A </answer>",
        " <answer>A</answer>",
        "<answer>A</answer>\n",
        "<Answer>A</Answer>",
        "<answer>a</answer>",
        "<answer>AA</answer>",
        "<answer>A</answer><answer>B</answer>",
        "reason <answer>A</answer>",
        None,
    ],
)
def test_strict_gold_rejects_loose_or_multiple_syntax(value: object) -> None:
    with pytest.raises(GoldSyntaxError):
        GoldAnswer.parse(value)


@pytest.mark.parametrize("value", ["V", "M", "VM", "T"])
def test_modality_parse_is_canonical(value: str) -> None:
    modality = Modality.parse(value)
    assert modality.value == value
    assert Modality.from_branch(value.lower()) is modality
    assert modality.branch == value.lower()


@pytest.mark.parametrize("value", ["v", "Vm", "video", "", None])
def test_modality_parse_does_not_silently_normalize(value: object) -> None:
    with pytest.raises(ModalityContractError):
        Modality.parse(value)
