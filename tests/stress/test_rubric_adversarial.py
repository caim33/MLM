from __future__ import annotations

import copy
import json

import pytest

from motionllm.grpo import (
    OnlineJudgeConfig,
    RubricValidationError,
    compute_motion_reward_v2,
    compute_qa_rubric_reward,
    parse_qa_judgment_text,
    validate_motion_criteria_v2,
)
from motionllm.grpo.rubric_common import strict_json_object
from tests.unit.test_motion_rubric_v2 import (
    MOTION_SAMPLE_ID,
    candidate_and_observations,
    motion_criteria,
    motion_judgment,
)
from tests.unit.test_qa_rubric import QA_PERFECT, qa_criteria, qa_judgment


@pytest.mark.stress
@pytest.mark.parametrize(
    "payload",
    [
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1,"a":2}',
        'prefix {"a":1}',
        '{"a":1} suffix',
        "\n{\"a\":1}",
        "[]",
    ],
)
def test_strict_judge_json_rejects_nonfinite_duplicate_or_wrapped_payloads(payload):
    with pytest.raises(RubricValidationError):
        strict_json_object(payload)


@pytest.mark.stress
@pytest.mark.parametrize(
    "candidate",
    [
        "<THINK>x</THINK><answer>A</answer>",
        "<think>x</think><answer>a</answer>",
        "<think>x</think><answer>A</answer><answer>A</answer>",
        "<think><think>x</think></think><answer>A</answer>",
        "<think>x</think><answer>A</answer>\x00A",
        "<think></think><answer>A</answer>",
    ],
)
def test_qa_reward_malformed_candidates_never_receive_full_reward(candidate):
    criteria = qa_criteria()
    result = compute_qa_rubric_reward(
        criteria, candidate, qa_judgment(criteria, candidate)
    )
    assert 0 <= result["reward"] < 1


@pytest.mark.stress
def test_qa_online_parser_rejects_unknown_flood_and_conflicting_duplicate_ids():
    criteria = qa_criteria()
    judgment = qa_judgment(criteria, QA_PERFECT)
    judgment["reasoning_quality"]["satisfied_ids"] = [f"r{index}" for index in range(100, 200)]
    with pytest.raises(RubricValidationError, match="unknown IDs"):
        parse_qa_judgment_text(
            json.dumps(judgment), criteria, candidate_response=QA_PERFECT
        )
    judgment = qa_judgment(criteria, QA_PERFECT)
    judgment["reasoning_quality"]["satisfied_ids"] = ["r1", "r1"]
    with pytest.raises(RubricValidationError, match="duplicate"):
        parse_qa_judgment_text(
            json.dumps(judgment), criteria, candidate_response=QA_PERFECT
        )


@pytest.mark.stress
@pytest.mark.parametrize(
    "observation,candidate",
    [
        (
            {"id": "s1_n1", "candidate_value": 50, "unit": "degrees", "candidate_text": "right knee angle is 999 degrees"},
            "right knee angle is 999 degrees",
        ),
        (
            {"id": "s1_n1", "candidate_value": 50, "unit": "m", "candidate_text": "right knee angle is 50 m"},
            "right knee angle is 50 m",
        ),
        (
            {"id": "s1_n1", "candidate_value": 50, "unit": "degrees", "candidate_text": "unrelated value is 50 degrees"},
            "unrelated value is 50 degrees",
        ),
        (
            {"id": "s1_n1", "candidate_value": 50, "unit": "degrees", "candidate_text": "right knee angle is 50 degrees"},
            "the actual candidate contains no number",
        ),
    ],
)
def test_motion_numeric_observation_spoofing_never_creates_credit(observation, candidate):
    criteria = validate_motion_criteria_v2(motion_criteria())
    judgment = motion_judgment(criteria, candidate)
    numeric = judgment["final_motion_answer"]["numeric_kinematics"]
    numeric["observed_values"] = [observation]
    numeric["strict_value_match_ids"] = ["s1_n1"]
    numeric["semantic_present_ids"] = []
    numeric["missed_ids"] = ["s1_n2", "s1_n3"]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert "s1_n1" not in result["debug"]["numeric_strict_ids"]
    assert "s1_n1" not in result["debug"]["numeric_semantic_ids"]


@pytest.mark.stress
def test_motion_cross_category_conflicts_fail_closed_even_with_negative_flood():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    judgment["final_motion_answer"]["laterality"]["wrong_ids"] = ["s1_l1"]
    judgment["final_motion_answer"]["camera_relative_orientation"]["wrong_ids"] = ["s1_o1"]
    judgment["reasoning_process"]["contradicted_ids"] = ["s1_r1", "s1_r2"]
    judgment["negative_criteria"]["triggered_ids"] = [
        "neg1",
        "neg2",
        "neg3",
        "neg4",
        "neg5",
        "neg6",
    ]
    with pytest.raises(RubricValidationError, match="disjoint exhaustive partition"):
        compute_motion_reward_v2(
            criteria,
            judgment,
            candidate_response=candidate,
            sample_id=MOTION_SAMPLE_ID,
        )


@pytest.mark.stress
def test_motion_schema_rejects_nonfinite_ranges_and_duplicate_segment_times():
    value = motion_criteria()
    value["segments"][0]["numeric_kinematics"][0]["target_range"] = [float("nan"), 50]
    with pytest.raises(RubricValidationError, match="finite"):
        validate_motion_criteria_v2(value)
    value = motion_criteria()
    value["segments"].append(copy.deepcopy(value["segments"][0]))
    with pytest.raises(RubricValidationError, match="duplicate segment time"):
        validate_motion_criteria_v2(value)


@pytest.mark.stress
def test_online_judge_config_never_reveals_token_and_rejects_unsafe_urls():
    token = "secret-sentinel-token"
    config = OnlineJudgeConfig(
        endpoint="https://judge.invalid/v1/rubric",
        bearer_token=token,
    )
    assert token not in repr(config)
    with pytest.raises(RubricValidationError, match="HTTPS"):
        OnlineJudgeConfig(endpoint="http://10.0.0.1/judge")
    with pytest.raises(RubricValidationError, match="credentials"):
        OnlineJudgeConfig(endpoint="https://user:password@judge.invalid/judge")
