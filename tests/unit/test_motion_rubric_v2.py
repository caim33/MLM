from __future__ import annotations

import copy

import pytest

from motionllm.grpo import (
    RubricValidationError,
    compute_motion_reward_v2,
    validate_motion_criteria_v2,
    validate_motion_judgment_v2,
)
from motionllm.grpo.rubric_common import build_judgment_binding
from rubric_rl.reward import ensure_criteria_ids as ensure_stage1_ids


MOTION_SAMPLE_ID = "motion-unit-1"


def motion_criteria(*, optional_direction_fields: bool = True) -> dict:
    return {
        "mode": "source_aware_reasoning_motion_rubric_v2",
        "global_activity": [
            {"criterion": "The person performs a controlled exercise.", "source": "video+motion"}
        ],
        "segments": [
            {
                "time": "0.00-1.00",
                "basic_action_facts": [
                    {"criterion": "The person bends the knee.", "source": "video+motion"},
                    {"criterion": "The person holds the pose.", "source": "video"},
                ],
                "body_configuration": [
                    {"criterion": "The torso remains above the hips.", "source": "motion"}
                ],
                "numeric_kinematics": [
                    {
                        "criterion": "Right knee angle is 50 degrees.",
                        "quantity": "flexion angle",
                        "body_part": "right knee",
                        "target_range": [50, 50],
                        "unit": "degrees",
                        "source": "motion",
                    },
                    {
                        "criterion": "Head height is 1.00 m.",
                        "quantity": "vertical height",
                        "body_part": "head",
                        "target_range": [1.0, 1.0],
                        "unit": "m",
                        "source": "motion",
                    },
                    {
                        "criterion": "The hold duration is 0.50 s.",
                        "quantity": "duration",
                        "body_part": "hold",
                        "target_range": [0.5, 0.5],
                        "unit": "s",
                        "source": "motion",
                    },
                ],
                "laterality": (
                    [{"criterion": "The right leg bends.", "source": "motion"}]
                    if optional_direction_fields
                    else []
                ),
                "camera_relative_orientation": (
                    [{"criterion": "The person faces the camera.", "source": "motion"}]
                    if optional_direction_fields
                    else []
                ),
                "reasoning_criteria": [
                    {
                        "criterion": "The reasoning separates video and motion evidence.",
                        "type": "source_separation",
                        "source": "think",
                    },
                    {
                        "criterion": "The reasoning uses numeric evidence.",
                        "type": "numeric_evidence_use",
                        "source": "think",
                    },
                ],
                "rejected_claims": [],
            }
        ],
        "temporal_phases": ["start", "bend", "hold", "rise", "finish"],
        "negative_criteria": [
            {"criterion": "An unsupported object is added.", "type": "unsupported_detail", "source_of_truth": "video"},
            {"criterion": "The action is reversed.", "type": "contradiction", "source_of_truth": "video+motion"},
            {"criterion": "The left leg is named instead.", "type": "wrong_laterality", "source_of_truth": "motion"},
            {"criterion": "The person faces away.", "type": "wrong_orientation", "source_of_truth": "motion"},
            {"criterion": "A wrong joint angle is stated.", "type": "numeric_contradiction", "source_of_truth": "motion"},
            {"criterion": "An unrelated activity is described.", "type": "unrelated_motion", "source_of_truth": "video"},
        ],
    }


def candidate_and_observations() -> tuple[str, list[dict]]:
    spans = [
        ("the right knee angle is 50 degrees", 50, "degrees"),
        ("head height is 1.00 m", 1.0, "m"),
        ("hold duration is 0.50 s", 0.5, "s"),
    ]
    candidate = (
        "<think>I separate video observations from motion measurements and use the "
        "numeric evidence to check the answer.</think>"
        "In the first phase, the right knee angle is 50 degrees, head height is 1.00 m, "
        "and hold duration is 0.50 s. The right leg bends while the person faces the camera."
    )
    observations = [
        {"id": f"s1_n{index}", "candidate_value": value, "unit": unit, "candidate_text": text}
        for index, (text, value, unit) in enumerate(spans, start=1)
    ]
    return candidate, observations


def motion_judgment(
    criteria: dict,
    candidate: str | None = None,
    *,
    sample_id: str = MOTION_SAMPLE_ID,
) -> dict:
    checked = validate_motion_criteria_v2(criteria)
    segment = checked["segments"][0]
    default_candidate, observations = candidate_and_observations()
    candidate = default_candidate if candidate is None else candidate
    return {
        "binding": build_judgment_binding(
            checked,
            candidate,
            sample_id=sample_id,
            nonce="motion-unit-nonce",
        ),
        "final_motion_answer": {
            "global_activity": {"satisfied_ids": ["g1"], "missed_ids": []},
            "basic_action_facts": {
                "present_ids": [item["id"] for item in segment["basic_action_facts"]],
                "aligned_ids": [item["id"] for item in segment["basic_action_facts"]],
                "missed_ids": [],
                "misplaced_ids": [],
            },
            "body_configuration": {
                "present_ids": [item["id"] for item in segment["body_configuration"]],
                "aligned_ids": [item["id"] for item in segment["body_configuration"]],
                "missed_ids": [],
                "misplaced_ids": [],
            },
            "numeric_kinematics": {
                "semantic_present_ids": [],
                "strict_value_match_ids": [item["id"] for item in segment["numeric_kinematics"]],
                "loose_value_match_ids": [],
                "wrong_value_ids": [],
                "missed_ids": [],
                "observed_values": observations,
            },
            "laterality": {
                "correct_ids": [item["id"] for item in segment["laterality"]],
                "wrong_ids": [],
                "missed_ids": [],
            },
            "camera_relative_orientation": {
                "correct_ids": [item["id"] for item in segment["camera_relative_orientation"]],
                "wrong_ids": [],
                "missed_ids": [],
            },
            "temporal_structure_score": 10,
        },
        "reasoning_process": {
            "satisfied_ids": [item["id"] for item in segment["reasoning_criteria"]],
            "missed_ids": [],
            "contradicted_ids": [],
        },
        "negative_criteria": {"triggered_ids": []},
        "language_format_score": 10,
        "hallucination_or_source_contradiction_penalty": 0,
        "hallucinations": [],
    }


def test_motion_schema_assigns_stable_ids_and_rejects_empty_or_stage1():
    checked = validate_motion_criteria_v2(motion_criteria())
    assert checked["segments"][0]["numeric_kinematics"][0]["id"] == "s1_n1"
    with pytest.raises(RubricValidationError):
        validate_motion_criteria_v2({})
    with pytest.raises(RubricValidationError, match="mode"):
        validate_motion_criteria_v2({"mode": "temporal_caption"})
    with pytest.raises(ValueError, match="Stage 1"):
        ensure_stage1_ids(checked)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value["segments"][0]["basic_action_facts"][1].update(id="s1_a1"), "globally unique"),
        (lambda value: value["segments"][0]["body_configuration"][0].update(id="s1_a9"), "namespace"),
        (lambda value: value["segments"][0]["numeric_kinematics"][0].update(strict_tolerance=10), "forbidden"),
        (lambda value: value["segments"][0].update(rejected_claims=[{"claim": "x"}]), "empty list"),
        (lambda value: value.update(temporal_phases=[]), "5 to 7"),
        (lambda value: value.update(negative_criteria=[]), "exactly 6"),
    ],
)
def test_motion_schema_rejects_duplicate_cross_namespace_and_incomplete_categories(mutate, message):
    value = motion_criteria()
    mutate(value)
    with pytest.raises(RubricValidationError, match=message):
        validate_motion_criteria_v2(value)


def test_motion_perfect_reward_requires_candidate_verified_numbers():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["total_score"] == 100
    assert result["numeric_kinematics_score"] == 10

    without_candidate = compute_motion_reward_v2(
        criteria,
        motion_judgment(criteria, ""),
        candidate_response="",
        sample_id=MOTION_SAMPLE_ID,
    )
    assert without_candidate["numeric_kinematics_score"] == 0
    assert without_candidate["total_score"] < 100


def test_wrong_numeric_never_receives_semantic_credit_and_forces_penalty():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    candidate = candidate.replace("right knee angle is 50 degrees", "right knee angle is 999 degrees")
    judgment = motion_judgment(criteria, candidate)
    numeric = judgment["final_motion_answer"]["numeric_kinematics"]
    numeric["strict_value_match_ids"].remove("s1_n1")
    numeric["wrong_value_ids"] = ["s1_n1"]
    numeric["observed_values"][0] = {
        "id": "s1_n1",
        "candidate_value": 999,
        "unit": "degrees",
        "candidate_text": "right knee angle is 999 degrees",
    }
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert "s1_n1" in result["debug"]["numeric_wrong_ids"]
    assert "s1_n1" not in result["debug"]["numeric_semantic_ids"]
    assert result["numeric_kinematics_score"] == pytest.approx(20 / 3)
    assert result["hallucination_or_source_contradiction_penalty"] <= -10


def test_fabricated_observed_value_is_not_trusted_without_verbatim_candidate_span():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate = "The movement is controlled and smooth without measurements."
    judgment = motion_judgment(criteria, candidate)
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["numeric_kinematics_score"] == 0
    assert result["debug"]["invalid_or_conflicting_numeric_observations"] == [
        "s1_n1",
        "s1_n2",
        "s1_n3",
    ]


def test_numeric_value_claim_requires_an_observation_row():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    judgment["final_motion_answer"]["numeric_kinematics"]["observed_values"] = []
    with pytest.raises(RubricValidationError, match="requires an observed_values row"):
        validate_motion_judgment_v2(
            judgment,
            criteria,
            candidate_response=candidate,
            sample_id=MOTION_SAMPLE_ID,
        )


@pytest.mark.parametrize(
    "candidate_text,candidate_value",
    [
        ("left knee angle is 50 degrees", 50),
        ("right knee angle ranges from 50 to 999 degrees", 50),
    ],
)
def test_numeric_verification_rejects_wrong_body_side_or_hidden_extra_value(
    candidate_text, candidate_value
):
    criteria = validate_motion_criteria_v2(motion_criteria())
    judgment = motion_judgment(criteria, candidate_text)
    numeric = judgment["final_motion_answer"]["numeric_kinematics"]
    numeric["semantic_present_ids"] = []
    numeric["strict_value_match_ids"] = ["s1_n1"]
    numeric["missed_ids"] = ["s1_n2", "s1_n3"]
    numeric["observed_values"] = [
        {
            "id": "s1_n1",
            "candidate_value": candidate_value,
            "unit": "degrees",
            "candidate_text": candidate_text,
        }
    ]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate_text,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert "s1_n1" not in result["debug"]["numeric_strict_ids"]
    assert "s1_n1" not in result["debug"]["numeric_semantic_ids"]


def test_motion_partition_conflicts_fail_closed():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    judgment["final_motion_answer"]["laterality"]["wrong_ids"] = ["s1_l1"]
    judgment["final_motion_answer"]["camera_relative_orientation"]["wrong_ids"] = ["s1_o1"]
    judgment["reasoning_process"]["contradicted_ids"] = ["s1_r1"]
    with pytest.raises(RubricValidationError, match="disjoint exhaustive partition"):
        validate_motion_judgment_v2(
            judgment,
            criteria,
            candidate_response=candidate,
            sample_id=MOTION_SAMPLE_ID,
        )


def test_wrong_and_contradicted_partitions_lose_credit_and_are_penalized():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    laterality = judgment["final_motion_answer"]["laterality"]
    laterality["correct_ids"] = []
    laterality["wrong_ids"] = ["s1_l1"]
    orientation = judgment["final_motion_answer"]["camera_relative_orientation"]
    orientation["correct_ids"] = []
    orientation["wrong_ids"] = ["s1_o1"]
    reasoning = judgment["reasoning_process"]
    reasoning["satisfied_ids"].remove("s1_r1")
    reasoning["contradicted_ids"] = ["s1_r1"]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["laterality_score"] == 0
    assert result["camera_orientation_score"] == 0
    assert "s1_r1" not in result["debug"]["reasoning_satisfied_ids"]
    assert result["hallucination_or_source_contradiction_penalty"] <= -20


def test_negative_criteria_drive_deterministic_penalty_even_when_judge_scalar_is_zero():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    judgment["negative_criteria"]["triggered_ids"] = ["neg5"]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["hallucination_or_source_contradiction_penalty"] <= -10


def test_optional_empty_direction_categories_score_zero_not_free_points():
    criteria = validate_motion_criteria_v2(motion_criteria(optional_direction_fields=False))
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["laterality_score"] == 0
    assert result["camera_orientation_score"] == 0


def test_motion_judgment_unknown_ids_are_rejected_at_online_boundary():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    judgment["reasoning_process"]["satisfied_ids"].append("s9_r9")
    with pytest.raises(RubricValidationError, match="unknown IDs"):
        validate_motion_judgment_v2(
            judgment,
            criteria,
            candidate_response=candidate,
            sample_id=MOTION_SAMPLE_ID,
            reject_unknown_ids=True,
        )


def test_motion_without_explicit_reasoning_marks_every_reasoning_criterion_missed():
    criteria = validate_motion_criteria_v2(motion_criteria())
    tagged, _ = candidate_and_observations()
    candidate = tagged.split("</think>", 1)[1]
    judgment = motion_judgment(criteria, candidate)
    validated = validate_motion_judgment_v2(
        judgment,
        criteria,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    reasoning_ids = sorted(
        item["id"]
        for segment in criteria["segments"]
        for item in segment["reasoning_criteria"]
    )
    assert result["reasoning_score"] == 0
    assert validated["reasoning_process"] == {
        "satisfied_ids": [],
        "missed_ids": reasoning_ids,
        "contradicted_ids": [],
    }
    assert result["debug"]["reasoning_satisfied_ids"] == []
    assert result["debug"]["reasoning_missed_ids"] == reasoning_ids
    assert not result["debug"]["has_explicit_reasoning"]


@pytest.mark.parametrize("invisible", ["\x00", "\u200b", "\u2060", "\ufeff"])
def test_motion_invisible_or_control_only_think_never_gets_reasoning_credit(
    invisible,
):
    criteria = validate_motion_criteria_v2(motion_criteria())
    tagged, _ = candidate_and_observations()
    final_answer = tagged.split("</think>", 1)[1]
    candidate = f"<think>{invisible}</think>{final_answer}"
    result = compute_motion_reward_v2(
        criteria,
        motion_judgment(criteria, candidate),
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert not result["debug"]["has_explicit_reasoning"]
    assert result["reasoning_score"] == 0
    assert result["reward"] < 1


@pytest.mark.parametrize(
    "candidate,evidence",
    [
        (
            "The report says 'right knee angle is 50 degrees', but the actual angle is 999 degrees.",
            "right knee angle is 50 degrees",
        ),
        (
            "The right knee angle is not 50 degrees; it is 999 degrees.",
            "right knee angle is not 50 degrees",
        ),
        (
            "It might be that the right knee angle is 50 degrees.",
            "right knee angle is 50 degrees",
        ),
        (
            "The right knee angle is 50 degrees is false.",
            "right knee angle is 50 degrees",
        ),
        (
            "Is the right knee angle 50 degrees?",
            "right knee angle 50 degrees",
        ),
        (
            "The right knee angle is 50 degrees. Correction: it is 999 degrees.",
            "right knee angle is 50 degrees",
        ),
        (
            "> right knee angle is 50 degrees\nThe actual value is 999 degrees.",
            "right knee angle is 50 degrees",
        ),
    ],
)
def test_numeric_quote_negation_and_uncertainty_spoofs_receive_no_credit(
    candidate, evidence
):
    criteria = validate_motion_criteria_v2(motion_criteria())
    judgment = motion_judgment(criteria, candidate)
    numeric = judgment["final_motion_answer"]["numeric_kinematics"]
    numeric["strict_value_match_ids"] = ["s1_n1"]
    numeric["missed_ids"] = ["s1_n2", "s1_n3"]
    numeric["observed_values"] = [
        {
            "id": "s1_n1",
            "candidate_value": 50,
            "unit": "degrees",
            "candidate_text": evidence,
        }
    ]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert result["numeric_kinematics_score"] == 0
    assert result["debug"]["numeric_strict_ids"] == []
    assert result["debug"]["numeric_loose_ids"] == []
    assert result["debug"]["numeric_semantic_ids"] == []


@pytest.mark.parametrize(
    "candidate",
    [
        "The right knee angle is 50 degrees. That statement is wrong.",
        "The right knee angle is 50 degrees. "
        + ("filler " * 30)
        + "Actually, the right knee angle is 999 degrees.",
        "The right knee angle is 999 degrees. "
        + ("filler " * 30)
        + "The right knee angle is 50 degrees.",
    ],
)
def test_numeric_retraction_or_unbounded_conflicting_value_fails_closed(candidate):
    criteria = validate_motion_criteria_v2(motion_criteria())
    judgment = motion_judgment(criteria, candidate)
    numeric = judgment["final_motion_answer"]["numeric_kinematics"]
    numeric["semantic_present_ids"] = []
    numeric["strict_value_match_ids"] = ["s1_n1"]
    numeric["loose_value_match_ids"] = []
    numeric["wrong_value_ids"] = []
    numeric["missed_ids"] = ["s1_n2", "s1_n3"]
    numeric["observed_values"] = [
        {
            "id": "s1_n1",
            "candidate_value": 50,
            "unit": "degrees",
            "candidate_text": "right knee angle is 50 degrees",
        }
    ]
    result = compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate,
        sample_id=MOTION_SAMPLE_ID,
    )
    assert "s1_n1" not in result["debug"]["numeric_strict_ids"]
    assert "s1_n1" not in result["debug"]["numeric_loose_ids"]
    assert "s1_n1" not in result["debug"]["numeric_semantic_ids"]
    assert "s1_n1" in result["debug"]["numeric_wrong_ids"]
    assert result["hallucination_or_source_contradiction_penalty"] <= -10


def test_motion_judgment_binding_rejects_candidate_sample_and_nonce_swaps():
    criteria = validate_motion_criteria_v2(motion_criteria())
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate)
    for changed_candidate, changed_sample, expected_nonce in (
        (candidate + " swapped", MOTION_SAMPLE_ID, None),
        (candidate, "motion-unit-2", None),
        (candidate, MOTION_SAMPLE_ID, "different-nonce"),
    ):
        with pytest.raises(RubricValidationError, match="binding"):
            validate_motion_judgment_v2(
                judgment,
                criteria,
                candidate_response=changed_candidate,
                sample_id=changed_sample,
                expected_nonce=expected_nonce,
            )
