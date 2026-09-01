from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from motionllm.grpo import (
    RubricValidationError,
    build_qa_judge_messages,
    compute_qa_rubric_reward,
    parse_qa_completion,
    parse_qa_judgment_text,
    validate_qa_criteria,
    validate_qa_judgment,
)
from motionllm.grpo.rubric_common import build_judgment_binding


QA_PERFECT = (
    "<think>The person starts facing the camera and then faces slightly left, "
    "so option A.</think><answer>A</answer>"
)


def qa_criteria() -> dict:
    path = Path(__file__).resolve().parents[2] / "rubric_rl" / "prompt_templates" / "qa01_eval_criteria.json"
    return json.loads(path.read_text(encoding="utf-8"))


def qa_judgment(
    criteria: dict,
    candidate: str = QA_PERFECT,
    *,
    satisfied: list[str] | None = None,
    contradicted: list[str] | None = None,
) -> dict:
    reasoning = [item["id"] for item in criteria["reasoning_criteria"]]
    contradicted = [] if contradicted is None else contradicted
    satisfied = (
        sorted(set(reasoning) - set(contradicted)) if satisfied is None else satisfied
    )
    return {
        "binding": build_judgment_binding(
            criteria,
            candidate,
            sample_id=criteria["benchmark_id"],
            nonce="unit-test-nonce",
        ),
        "reasoning_quality": {
            "satisfied_ids": satisfied,
            "missed_ids": sorted(
                set(reasoning) - set(satisfied) - set(contradicted)
            ),
            "contradicted_ids": contradicted,
        },
        "negative_criteria": {"triggered_ids": []},
        "language_conciseness_score": 5,
        "contradiction_penalty": 0,
        "notes": [],
    }


def test_qa_schema_accepts_frozen_pilot_and_returns_detached_value():
    source = qa_criteria()
    checked = validate_qa_criteria(source)
    checked["options"]["A"] = "changed"
    assert source["options"]["A"] != "changed"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update(mode="temporal_caption"), "mode"),
        (lambda value: value.update(task="caption"), "task"),
        (lambda value: value["options"].update(B=value["options"]["A"]), "distinct"),
        (lambda value: value.update(correct_option_text="wrong"), "exactly match"),
        (lambda value: value["reasoning_criteria"][1].update(id="r1"), "globally unique"),
        (lambda value: value["reasoning_criteria"][0].update(type="vague"), "unsupported"),
        (lambda value: value.update(reasoning_criteria=value["reasoning_criteria"][:3]), "between 4 and 7"),
    ],
)
def test_qa_schema_rejects_malformed_or_ambiguous_criteria(mutate, message):
    value = qa_criteria()
    mutate(value)
    with pytest.raises(RubricValidationError, match=message):
        validate_qa_criteria(value)


def test_qa_reward_perfect_wrong_invalid_and_missing_reasoning_caps():
    criteria = validate_qa_criteria(qa_criteria())
    perfect = compute_qa_rubric_reward(
        criteria,
        QA_PERFECT,
        qa_judgment(criteria, QA_PERFECT),
    )
    assert perfect["total_score"] == 100
    assert perfect["reward"] == 1

    wrong_candidate = QA_PERFECT.replace("<answer>A</answer>", "<answer>B</answer>")
    wrong = compute_qa_rubric_reward(
        criteria, wrong_candidate, qa_judgment(criteria, wrong_candidate)
    )
    assert wrong["total_score"] == 50
    assert "wrong_answer_max_50" in wrong["debug"]["applied_caps"]

    invalid_candidate = "<think>reasoning</think><answer>A.</answer>"
    invalid = compute_qa_rubric_reward(
        criteria, invalid_candidate, qa_judgment(criteria, invalid_candidate)
    )
    assert invalid["total_score"] <= 20
    assert "no_valid_answer_max_20" in invalid["debug"]["applied_caps"]

    no_think_candidate = "<answer>A</answer>"
    no_think = compute_qa_rubric_reward(
        criteria, no_think_candidate, qa_judgment(criteria, no_think_candidate)
    )
    assert no_think["reasoning_score"] == 0
    assert "no_visible_reasoning_max_45" in no_think["debug"]["applied_caps"]


def test_qa_reasoning_contradiction_and_negative_ids_are_adverse():
    criteria = validate_qa_criteria(qa_criteria())
    candidate = "<think>specific reasoning</think><answer>A</answer>"
    judgment = qa_judgment(criteria, candidate, contradicted=["r1"])
    judgment["negative_criteria"]["triggered_ids"] = ["n1"]
    result = compute_qa_rubric_reward(criteria, candidate, judgment)
    assert "r1" not in result["debug"]["satisfied_ids"]
    assert result["contradiction_penalty"] <= -10
    assert result["reasoning_score"] == pytest.approx(55 * 5 / 6)


def test_qa_reasoning_partition_conflicts_fail_closed():
    criteria = validate_qa_criteria(qa_criteria())
    judgment = qa_judgment(criteria, QA_PERFECT)
    judgment["reasoning_quality"]["contradicted_ids"] = ["r1"]
    with pytest.raises(RubricValidationError, match="disjoint exhaustive partition"):
        validate_qa_judgment(
            judgment,
            criteria,
            candidate_response=QA_PERFECT,
            reject_unknown_ids=True,
        )


def test_qa_invalid_judge_ids_are_rejected_online_and_removed_in_reward():
    criteria = validate_qa_criteria(qa_criteria())
    candidate = "<think>specific reasoning</think><answer>A</answer>"
    judgment = qa_judgment(criteria, candidate)
    judgment["reasoning_quality"]["satisfied_ids"].append("r999")
    with pytest.raises(RubricValidationError, match="unknown IDs"):
        validate_qa_judgment(
            judgment,
            criteria,
            candidate_response=candidate,
            reject_unknown_ids=True,
        )
    result = compute_qa_rubric_reward(criteria, candidate, judgment)
    assert result["reward"] <= 1
    assert any("r999" in item for item in result["debug"]["invalid_ids_removed"])


def test_qa_completion_parser_rejects_nested_repeated_and_extra_final_text():
    nested = parse_qa_completion(
        "<think>outer <think>inner</think></think><answer>A</answer>"
    )
    assert not nested.has_one_think
    repeated = parse_qa_completion(
        "<think>x</think><think>y</think><answer>A</answer>"
    )
    assert not repeated.has_one_think
    extra = parse_qa_completion(
        "<think>x</think><answer>A</answer> Final answer A"
    )
    assert not extra.ordered_without_extra_text
    assert extra.format_score == 7


@pytest.mark.parametrize("invisible", ["\x00", "\u200b", "\u2060", "\ufeff"])
def test_qa_invisible_or_control_only_think_never_gets_reasoning_credit(invisible):
    criteria = validate_qa_criteria(qa_criteria())
    candidate = f"<think>{invisible}</think><answer>A</answer>"
    parsed = parse_qa_completion(candidate)
    result = compute_qa_rubric_reward(
        criteria,
        candidate,
        qa_judgment(criteria, candidate),
    )
    assert not parsed.has_visible_reasoning
    assert result["reasoning_score"] == 0
    assert result["reward"] < 1
    assert "no_visible_reasoning_max_45" in result["debug"]["applied_caps"]


def test_qa_judge_parser_requires_one_exact_json_object_and_no_duplicate_keys():
    criteria = validate_qa_criteria(qa_criteria())
    judgment = qa_judgment(criteria, QA_PERFECT)
    parsed = parse_qa_judgment_text(
        json.dumps(judgment), criteria, candidate_response=QA_PERFECT
    )
    assert parsed["language_conciseness_score"] == 5
    with pytest.raises(RubricValidationError):
        parse_qa_judgment_text(
            "```json\n" + json.dumps(judgment) + "\n```",
            criteria,
            candidate_response=QA_PERFECT,
        )
    duplicate = json.dumps(judgment)[:-1] + ',"notes":[]}'
    with pytest.raises(RubricValidationError, match="duplicate JSON key"):
        parse_qa_judgment_text(
            duplicate, criteria, candidate_response=QA_PERFECT
        )


def test_qa_judge_message_builder_validates_template_and_does_not_mutate():
    criteria = qa_criteria()
    original = copy.deepcopy(criteria)
    messages = build_qa_judge_messages(criteria, "candidate")
    assert criteria == original
    assert messages[0]["role"] == "system"
    assert "candidate" in messages[1]["content"]
    with pytest.raises(RubricValidationError, match="placeholders"):
        build_qa_judge_messages(criteria, "candidate", user_template="missing")
