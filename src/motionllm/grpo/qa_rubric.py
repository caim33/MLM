"""Strict QA multiple-choice Rubric-RL contracts and deterministic reward."""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from .rubric_common import (
    RubricValidationError,
    require_exact_keys,
    require_mapping,
    require_partition,
    strict_id_list,
    strict_identifier,
    strict_json_object,
    strict_text,
    validate_judgment_binding,
)

QA_RUBRIC_VERSION = "qa_mc_rubric_v1"
QA_MODE = "qa_mc"
QUESTION_TYPES = frozenset(
    {
        "time_range",
        "orientation_transition",
        "posture_state",
        "limb_motion",
        "laterality",
        "numeric_comparison",
        "action_identification",
        "temporal_comparison",
        "other_motion_qa",
    }
)
REASONING_TYPES = frozenset(
    {
        "question_focus",
        "segment_reference",
        "correct_option_fact",
        "comparison",
        "option_mapping",
        "answer_consistency",
    }
)
NEGATIVE_TYPES = frozenset(
    {
        "contradiction",
        "wrong_option_support",
        "irrelevant_reasoning",
        "unrelated_motion",
        "malformed_answer",
    }
)
PENALTIES = frozenset({0, -5, -10, -15, -20})
CHOICES = frozenset("ABCD")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ID_RE = {
    "reasoning_criteria": re.compile(r"^r[1-9][0-9]*$"),
    "format_criteria": re.compile(r"^f[1-9][0-9]*$"),
    "negative_criteria": re.compile(r"^n[1-9][0-9]*$"),
}


def _has_substantive_visible_reasoning(value: str | None) -> bool:
    """Require real visible alphanumeric reasoning, never control-only text.

    ``str.strip`` does not remove Unicode format characters such as ZERO WIDTH
    SPACE, WORD JOINER, or BOM.  Reject every Unicode ``C*`` category and
    require at least one actual Unicode letter or number.
    """

    if not isinstance(value, str) or not value:
        return False
    categories = [unicodedata.category(character) for character in value]
    if any(category.startswith("C") for category in categories):
        return False
    return any(category.startswith(("L", "N")) for category in categories)


def _criterion_list(
    value: Any,
    *,
    name: str,
    allowed_types: frozenset[str],
    minimum: int,
    maximum: int,
    assign_missing_ids: bool,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise RubricValidationError(
            f"{name} must contain between {minimum} and {maximum} criteria"
        )
    output: list[dict[str, str]] = []
    prefix = {"reasoning_criteria": "r", "format_criteria": "f", "negative_criteria": "n"}[name]
    for index, raw in enumerate(value, start=1):
        item = require_mapping(raw, name=f"{name}[{index - 1}]")
        required = {"criterion", "type"}
        optional = {"id"} if assign_missing_ids else set()
        if "id" in item:
            required.add("id")
            optional.discard("id")
        require_exact_keys(
            item,
            name=f"{name}[{index - 1}]",
            required=required,
            optional=optional,
        )
        identifier = item.get("id")
        if identifier is None:
            if not assign_missing_ids:
                raise RubricValidationError(f"{name}[{index - 1}].id is required")
            identifier = f"{prefix}{index}"
        identifier = strict_identifier(identifier, name=f"{name}[{index - 1}].id")
        if _ID_RE[name].fullmatch(identifier) is None:
            raise RubricValidationError(f"{name} id has the wrong namespace: {identifier!r}")
        criterion = strict_text(
            item.get("criterion"), name=f"{name}[{index - 1}].criterion", max_length=2_000
        )
        criterion_type = strict_text(
            item.get("type"), name=f"{name}[{index - 1}].type", max_length=64
        )
        if criterion_type not in allowed_types:
            raise RubricValidationError(
                f"unsupported {name} type: {criterion_type!r}"
            )
        output.append({"id": identifier, "criterion": criterion, "type": criterion_type})
    return output


def validate_qa_criteria(
    value: Mapping[str, Any], *, assign_missing_ids: bool = False
) -> dict[str, Any]:
    """Validate and return a detached canonical QA rubric.

    LLM extraction may omit IDs, but every other mismatch is rejected.  The
    canonical question/options/gold values must be supplied by the dataset and
    compared by the caller before publication.
    """

    criteria = require_mapping(value, name="criteria")
    required = {
        "mode",
        "benchmark_id",
        "task",
        "question_type",
        "question_focus",
        "question",
        "options",
        "correct_option",
        "correct_option_text",
        "reasoning_criteria",
        "format_criteria",
        "negative_criteria",
    }
    require_exact_keys(
        criteria,
        name="criteria",
        required=required,
        optional={"case_name"},
    )
    if criteria.get("mode") != QA_MODE:
        raise RubricValidationError(f"criteria.mode must be {QA_MODE!r}")
    benchmark_id = strict_identifier(criteria.get("benchmark_id"), name="criteria.benchmark_id")
    task = strict_text(criteria.get("task"), name="criteria.task", max_length=64)
    if task != "QA":
        raise RubricValidationError("criteria.task must be exactly 'QA'")
    question_type = strict_text(
        criteria.get("question_type"), name="criteria.question_type", max_length=64
    )
    if question_type not in QUESTION_TYPES:
        raise RubricValidationError(f"unsupported question_type: {question_type!r}")
    question_focus = strict_text(
        criteria.get("question_focus"), name="criteria.question_focus", max_length=4_000
    )
    question = strict_text(criteria.get("question"), name="criteria.question", max_length=8_000)

    raw_options = require_mapping(criteria.get("options"), name="criteria.options")
    require_exact_keys(raw_options, name="criteria.options", required=set(CHOICES))
    options = {
        choice: strict_text(raw_options.get(choice), name=f"criteria.options.{choice}", max_length=4_000)
        for choice in "ABCD"
    }
    if len({re.sub(r"\s+", " ", item).casefold() for item in options.values()}) != 4:
        raise RubricValidationError("criteria options must be semantically distinct strings")

    correct_option = criteria.get("correct_option")
    if not isinstance(correct_option, str) or correct_option not in CHOICES:
        raise RubricValidationError("criteria.correct_option must be exactly A, B, C, or D")
    correct_option_text = strict_text(
        criteria.get("correct_option_text"),
        name="criteria.correct_option_text",
        max_length=4_000,
    )
    if correct_option_text != options[correct_option]:
        raise RubricValidationError("correct_option_text must exactly match the selected option")

    reasoning = _criterion_list(
        criteria.get("reasoning_criteria"),
        name="reasoning_criteria",
        allowed_types=REASONING_TYPES,
        minimum=4,
        maximum=7,
        assign_missing_ids=assign_missing_ids,
    )
    formatting = _criterion_list(
        criteria.get("format_criteria"),
        name="format_criteria",
        allowed_types=frozenset({"format"}),
        minimum=4,
        maximum=4,
        assign_missing_ids=assign_missing_ids,
    )
    negatives = _criterion_list(
        criteria.get("negative_criteria"),
        name="negative_criteria",
        allowed_types=NEGATIVE_TYPES,
        minimum=1,
        maximum=10,
        assign_missing_ids=assign_missing_ids,
    )
    all_ids = [item["id"] for item in reasoning + formatting + negatives]
    if len(all_ids) != len(set(all_ids)):
        raise RubricValidationError("criterion IDs must be globally unique across all categories")

    result = {
        "mode": QA_MODE,
        "benchmark_id": benchmark_id,
        "task": task,
        "question_type": question_type,
        "question_focus": question_focus,
        "question": question,
        "options": options,
        "correct_option": correct_option,
        "correct_option_text": correct_option_text,
        "reasoning_criteria": reasoning,
        "format_criteria": formatting,
        "negative_criteria": negatives,
    }
    if "case_name" in criteria:
        result["case_name"] = strict_identifier(criteria["case_name"], name="criteria.case_name")
    return result


def assert_qa_dataset_binding(criteria: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
    """Reject criteria that alter canonical QA question, options, or gold."""

    checked = validate_qa_criteria(criteria)
    for field in ("benchmark_id", "question", "options", "correct_option", "correct_option_text"):
        if checked[field] != canonical.get(field):
            raise RubricValidationError(f"criteria {field} does not match the canonical QA row")


def _judgment_id_lists(
    raw: Mapping[str, Any],
    *,
    criteria: Mapping[str, Any],
    reject_unknown_ids: bool,
) -> tuple[dict[str, Any], list[str]]:
    reasoning = require_mapping(raw.get("reasoning_quality"), name="judgment.reasoning_quality")
    require_exact_keys(
        reasoning,
        name="judgment.reasoning_quality",
        required={"satisfied_ids", "missed_ids", "contradicted_ids"},
    )
    negatives = require_mapping(raw.get("negative_criteria"), name="judgment.negative_criteria")
    require_exact_keys(
        negatives,
        name="judgment.negative_criteria",
        required={"triggered_ids"},
    )
    valid_reasoning = {item["id"] for item in criteria["reasoning_criteria"]}
    valid_negative = {item["id"] for item in criteria["negative_criteria"]}
    invalid: list[str] = []

    def sanitize(value: Any, *, name: str, valid: set[str]) -> list[str]:
        reported = strict_id_list(value, name=name)
        unknown = sorted(set(reported) - valid)
        invalid.extend(f"{name}:{item}" for item in unknown)
        if unknown and reject_unknown_ids:
            raise RubricValidationError(f"{name} contains unknown IDs: {unknown}")
        return [item for item in reported if item in valid]

    return (
        {
            "reasoning_quality": {
                "satisfied_ids": sanitize(
                    reasoning.get("satisfied_ids"),
                    name="reasoning_quality.satisfied_ids",
                    valid=valid_reasoning,
                ),
                "missed_ids": sanitize(
                    reasoning.get("missed_ids"),
                    name="reasoning_quality.missed_ids",
                    valid=valid_reasoning,
                ),
                "contradicted_ids": sanitize(
                    reasoning.get("contradicted_ids"),
                    name="reasoning_quality.contradicted_ids",
                    valid=valid_reasoning,
                ),
            },
            "negative_criteria": {
                "triggered_ids": sanitize(
                    negatives.get("triggered_ids"),
                    name="negative_criteria.triggered_ids",
                    valid=valid_negative,
                )
            },
        },
        invalid,
    )


def validate_qa_judgment(
    value: Mapping[str, Any],
    criteria: Mapping[str, Any],
    *,
    candidate_response: str,
    expected_nonce: str | None = None,
    reject_unknown_ids: bool = True,
) -> dict[str, Any]:
    checked_criteria = validate_qa_criteria(criteria)
    raw = require_mapping(value, name="judgment")
    require_exact_keys(
        raw,
        name="judgment",
        required={
            "binding",
            "reasoning_quality",
            "negative_criteria",
            "language_conciseness_score",
            "contradiction_penalty",
            "notes",
        },
    )
    normalized, invalid_ids = _judgment_id_lists(
        raw, criteria=checked_criteria, reject_unknown_ids=reject_unknown_ids
    )
    reasoning_lists = normalized["reasoning_quality"]
    require_partition(
        {item["id"] for item in checked_criteria["reasoning_criteria"]},
        {
            "satisfied_ids": reasoning_lists["satisfied_ids"],
            "missed_ids": reasoning_lists["missed_ids"],
            "contradicted_ids": reasoning_lists["contradicted_ids"],
        },
        name="judgment.reasoning_quality",
    )
    binding = validate_judgment_binding(
        raw.get("binding"),
        checked_criteria,
        candidate_response,
        sample_id=checked_criteria["benchmark_id"],
        expected_nonce=expected_nonce,
    )
    language = raw.get("language_conciseness_score")
    if isinstance(language, bool) or not isinstance(language, int) or not 0 <= language <= 5:
        raise RubricValidationError("language_conciseness_score must be an integer from 0 to 5")
    penalty = raw.get("contradiction_penalty")
    if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty not in PENALTIES:
        raise RubricValidationError(f"contradiction_penalty must be one of {sorted(PENALTIES)}")
    notes = raw.get("notes")
    if not isinstance(notes, list) or len(notes) > 20:
        raise RubricValidationError("judgment.notes must be a list with at most 20 strings")
    normalized_notes = [
        strict_text(item, name=f"judgment.notes[{index}]", max_length=1_000)
        for index, item in enumerate(notes)
    ]
    return {
        "binding": binding,
        **normalized,
        "language_conciseness_score": language,
        "contradiction_penalty": penalty,
        "notes": normalized_notes,
        "invalid_ids_removed": sorted(invalid_ids),
    }


def parse_qa_judgment_text(
    text: str,
    criteria: Mapping[str, Any],
    *,
    candidate_response: str,
    expected_nonce: str | None = None,
    reject_unknown_ids: bool = True,
) -> dict[str, Any]:
    return validate_qa_judgment(
        strict_json_object(text),
        criteria,
        candidate_response=candidate_response,
        expected_nonce=expected_nonce,
        reject_unknown_ids=reject_unknown_ids,
    )


@dataclass(frozen=True)
class QACompletion:
    has_one_think: bool
    has_visible_reasoning: bool
    think_text: str | None
    has_one_answer: bool
    parsed_answer: str | None
    ordered_without_extra_text: bool
    format_score: int


def parse_qa_completion(candidate: Any) -> QACompletion:
    if not isinstance(candidate, str):
        candidate = ""
    think_matches = list(_THINK_RE.finditer(candidate))
    answer_matches = list(_ANSWER_RE.finditer(candidate))
    has_one_think = (
        len(think_matches) == 1
        and candidate.count("<think>") == 1
        and candidate.count("</think>") == 1
    )
    has_one_answer = (
        len(answer_matches) == 1
        and candidate.count("<answer>") == 1
        and candidate.count("</answer>") == 1
    )
    think_text = think_matches[0].group(1) if has_one_think else None
    answer_text = answer_matches[0].group(1).strip() if has_one_answer else ""
    parsed_answer = answer_text if answer_text in CHOICES else None
    visible = bool(
        has_one_think and _has_substantive_visible_reasoning(think_text)
    )
    ordered_without_extra = False
    if has_one_think and has_one_answer:
        think_match = think_matches[0]
        answer_match = answer_matches[0]
        outside = (
            candidate[: think_match.start()]
            + candidate[think_match.end() : answer_match.start()]
            + candidate[answer_match.end() :]
        ) if think_match.end() <= answer_match.start() else candidate
        ordered_without_extra = think_match.end() <= answer_match.start() and not outside.strip()
    score = (
        (2 if has_one_think else 0)
        + (2 if has_one_answer else 0)
        + (3 if parsed_answer is not None else 0)
        + (3 if ordered_without_extra else 0)
    )
    return QACompletion(
        has_one_think=has_one_think,
        has_visible_reasoning=visible,
        think_text=think_text,
        has_one_answer=has_one_answer,
        parsed_answer=parsed_answer,
        ordered_without_extra_text=ordered_without_extra,
        format_score=max(0, min(10, score)),
    )


def compute_qa_rubric_reward(
    criteria: Mapping[str, Any], candidate: Any, judgment: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_qa_criteria(criteria)
    judgment_input = dict(judgment)
    judgment_input.pop("invalid_ids_removed", None)
    judged = validate_qa_judgment(
        judgment_input,
        checked,
        candidate_response=candidate if isinstance(candidate, str) else "",
        reject_unknown_ids=False,
    )
    parsed = parse_qa_completion(candidate)
    valid_reasoning = {item["id"] for item in checked["reasoning_criteria"]}
    satisfied_reported = set(judged["reasoning_quality"]["satisfied_ids"])
    contradicted = set(judged["reasoning_quality"]["contradicted_ids"])
    conflicts = satisfied_reported & contradicted
    satisfied = satisfied_reported - contradicted
    if not parsed.has_visible_reasoning:
        satisfied.clear()
        contradicted.clear()
    missed = valid_reasoning - satisfied
    reasoning_score = 55.0 * len(satisfied) / len(valid_reasoning)

    negative_by_id = {item["id"]: item for item in checked["negative_criteria"]}
    triggered = set(judged["negative_criteria"]["triggered_ids"])
    deterministic_penalty = 0
    if triggered:
        deterministic_penalty = -5
    serious = {
        identifier
        for identifier in triggered
        if negative_by_id[identifier]["type"] in {"contradiction", "wrong_option_support"}
    }
    if serious or contradicted or conflicts:
        deterministic_penalty = min(deterministic_penalty, -10)
    penalty = min(judged["contradiction_penalty"], deterministic_penalty)

    answer_score = 30 if parsed.parsed_answer == checked["correct_option"] else 0
    language_score = judged["language_conciseness_score"]
    raw_score = answer_score + reasoning_score + parsed.format_score + language_score + penalty
    total_score = max(0.0, min(100.0, raw_score))
    applied_caps: list[str] = []
    if parsed.parsed_answer != checked["correct_option"]:
        total_score = min(total_score, 50.0)
        applied_caps.append("wrong_answer_max_50")
    if parsed.parsed_answer is None:
        total_score = min(total_score, 20.0)
        applied_caps.append("no_valid_answer_max_20")
    if not parsed.has_visible_reasoning:
        total_score = min(total_score, 45.0)
        applied_caps.append("no_visible_reasoning_max_45")

    return {
        "rubric_version": QA_RUBRIC_VERSION,
        "answer_score": answer_score,
        "format_score": parsed.format_score,
        "reasoning_score": reasoning_score,
        "language_score": language_score,
        "contradiction_penalty": penalty,
        "raw_score": raw_score,
        "total_score": total_score,
        "reward": total_score / 100.0,
        "debug": {
            "parsed_answer": parsed.parsed_answer,
            "correct_option": checked["correct_option"],
            "has_one_think": parsed.has_one_think,
            "has_visible_reasoning": parsed.has_visible_reasoning,
            "has_one_answer": parsed.has_one_answer,
            "ordered_without_extra_text": parsed.ordered_without_extra_text,
            "satisfied_ids": sorted(satisfied),
            "missed_ids": sorted(missed),
            "contradicted_ids": sorted(contradicted),
            "conflicting_reasoning_ids": sorted(conflicts),
            "triggered_negative_ids": sorted(triggered),
            "invalid_ids_removed": judged["invalid_ids_removed"],
            "applied_caps": applied_caps,
        },
    }


DEFAULT_QA_JUDGE_SYSTEM = (
    "You are a strict QA reasoning rubric judge. Use only the supplied criteria "
    "and candidate. Return exactly one JSON object and never compute reward."
)
DEFAULT_QA_JUDGE_USER = """Criteria:\n{CRITERIA_JSON}\n\nCandidate response:\n{CANDIDATE_RESPONSE}\n\nJudge only visible text inside exactly one non-empty <think>...</think> block. Put every reasoning criterion ID in exactly one of satisfied_ids, missed_ids, or contradicted_ids; the three lists must be pairwise disjoint and exhaustive. Use only supplied IDs. Do not output a binding; the trusted host adds it. Return exactly this JSON shape and no other text: {"reasoning_quality":{"satisfied_ids":[],"missed_ids":[],"contradicted_ids":[]},"negative_criteria":{"triggered_ids":[]},"language_conciseness_score":0,"contradiction_penalty":0,"notes":[]}."""


def build_qa_judge_messages(
    criteria: Mapping[str, Any],
    candidate: Any,
    *,
    system_prompt: str = DEFAULT_QA_JUDGE_SYSTEM,
    user_template: str = DEFAULT_QA_JUDGE_USER,
) -> list[dict[str, str]]:
    checked = validate_qa_criteria(criteria)
    if not isinstance(candidate, str):
        raise RubricValidationError("candidate response must be a string")
    if "{CRITERIA_JSON}" not in user_template or "{CANDIDATE_RESPONSE}" not in user_template:
        raise RubricValidationError("QA judge template is missing required placeholders")
    user = user_template.replace(
        "{CRITERIA_JSON}", json.dumps(checked, ensure_ascii=False, sort_keys=True)
    ).replace("{CANDIDATE_RESPONSE}", candidate)
    return [
        {"role": "system", "content": strict_text(system_prompt, name="system_prompt")},
        {"role": "user", "content": user},
    ]


__all__ = [
    "QACompletion",
    "QA_MODE",
    "QA_RUBRIC_VERSION",
    "assert_qa_dataset_binding",
    "build_qa_judge_messages",
    "compute_qa_rubric_reward",
    "parse_qa_completion",
    "parse_qa_judgment_text",
    "validate_qa_criteria",
    "validate_qa_judgment",
]
