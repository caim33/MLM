import pytest

from motion_eval.contracts import ParseStatus
from motion_eval.evaluation import parse_strict_answer


@pytest.mark.parametrize("choice", ["A", "B", "C", "D"])
def test_accepts_each_choice_in_one_complete_literal_tag(choice):
    result = parse_strict_answer(f"reasoning before\n<answer>{choice}</answer>\nafter")

    assert result.is_valid
    assert result.status is ParseStatus.VALID
    assert result.answer == choice
    assert result.complete_tag_count == 1


@pytest.mark.parametrize(
    "output",
    [
        "A",
        "The answer is B.",
        "(C)",
        "<answer>a</answer>",
        "<answer> A </answer>",
        "<answer value='A'>A</answer>",
        "<ANSWER>A</ANSWER>",
        "<answer>E</answer>",
    ],
)
def test_rejects_loose_or_non_literal_answers(output):
    result = parse_strict_answer(output)

    assert not result.is_valid
    assert result.answer is None


@pytest.mark.parametrize(
    "output",
    [
        "<answer>A</answer><answer>B</answer>",
        "<answer>C</answer> text <answer>C</answer>",
    ],
)
def test_rejects_multiple_complete_tags_even_if_answers_match(output):
    result = parse_strict_answer(output)

    assert result.status is ParseStatus.MULTIPLE_ANSWER_TAGS
    assert result.answer is None
    assert result.complete_tag_count == 2


@pytest.mark.parametrize(
    "output",
    [
        "<answer>A",
        "</answer>",
        "<answer>A</answer> then <answer>B",
        "<answer>A</answer> then <ANSWER>B</ANSWER>",
        "<answer><answer>A</answer></answer>",
    ],
)
def test_rejects_malformed_or_additional_answer_like_tags(output):
    result = parse_strict_answer(output)

    assert result.status is ParseStatus.MALFORMED_ANSWER_TAG
    assert result.answer is None


def test_no_answer_like_marker_is_missing_not_malformed():
    assert parse_strict_answer("no tagged answer").status is ParseStatus.MISSING_ANSWER_TAG


def test_non_string_is_programming_error_not_a_prediction():
    with pytest.raises(TypeError, match="must be a string"):
        parse_strict_answer(None)  # type: ignore[arg-type]
