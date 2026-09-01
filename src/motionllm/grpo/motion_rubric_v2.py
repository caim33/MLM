"""Strict Motion Rubric V2 schema and deterministic reward post-processing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from .rubric_common import (
    RubricValidationError,
    finite_number,
    require_exact_keys,
    require_mapping,
    require_partition,
    strict_id_list,
    strict_identifier,
    strict_json_object,
    strict_text,
    validate_judgment_binding,
)

MOTION_RUBRIC_V2_VERSION = "source_aware_reasoning_motion_rubric_v2"
MOTION_MODE_V2 = "source_aware_reasoning_motion_rubric_v2"
TEMPORAL_SCORES = frozenset({0, 3, 6, 10})
PENALTIES = frozenset({0, -5, -10, -15, -20, -25})
REASONING_WEIGHTS = {
    "source_separation": 4.0,
    "conflict_detection": 4.0,
    "trust_hierarchy_application": 6.0,
    "numeric_evidence_use": 4.0,
    "reasoning_answer_consistency": 2.0,
}
NEGATIVE_TYPES = frozenset(
    {
        "unsupported_detail",
        "contradiction",
        "wrong_laterality",
        "wrong_orientation",
        "numeric_contradiction",
        "unrelated_motion",
    }
)
NUMERIC_UNITS = frozenset({"degrees", "m", "s"})
NUMERIC_POLICY_FIELDS = frozenset(
    {
        "strict_tolerance",
        "loose_tolerance",
        "tolerance",
        "margin",
        "score",
        "points",
        "reward",
        "penalty",
    }
)
TOLERANCES = {
    "degrees": (10.0, 20.0),
    "m": (0.05, 0.10),
    "s": (0.20, 0.50),
}
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_QUOTED_RE = re.compile(
    r'''"[^"\r\n]*"|'[^'\r\n]*'|\u201c[^\u201d\r\n]*\u201d|'''
    r'''\u2018[^\u2019\r\n]*\u2019|\u00ab[^\u00bb\r\n]*\u00bb|'''
    r'''\u300c[^\u300d\r\n]*\u300d|\u300e[^\u300f\r\n]*\u300f|'''
    r'''`[^`\r\n]*`|(?i:&quot;)[^\r\n]*?(?i:&quot;)'''
)
_NON_ASSERTIVE_RE = re.compile(
    r"\b(?:not|never|no|false|incorrect|wrong|reject(?:s|ed|ing)?|"
    r"deny|denies|denied|quote|quoted|claim|criterion|example|hypothetical|"
    r"may|might|could|possibly|perhaps|uncertain|unknown|purported|alleged|"
    r"reported|label|reads|if|assuming|suppose|supposing|would|unclear|doubt|"
    r"unlikely|isn'?t|aren'?t|doesn'?t|cannot|can'?t|approximately|approx|"
    r"about|around|roughly)\b|\b(?:less|more)\s+than\b|\bat\s+(?:most|least)\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(?:but|however|actually|instead|rather|correction)\b",
    re.IGNORECASE,
)
_EXPLICIT_RETRACTION_RE = re.compile(
    r"\b(?:(?:that|this|the|previous|preceding|above|earlier)\s+)?"
    r"(?:(?:statement|claim|value|measurement|number|angle)\s+)?"
    r"(?:is|was|were|seems?|becomes?)?\s*"
    r"(?:not\s+(?:correct|true|valid)|false|incorrect|wrong|invalid|"
    r"retracted|withdrawn|disregard(?:ed)?|ignore(?:d)?)\b",
    re.IGNORECASE,
)
_ADVERSE_POLARITY_RE = re.compile(
    r"\b(?:not|never|no|false|incorrect|wrong|reject(?:s|ed|ing)?|"
    r"deny|denies|denied|isn'?t|aren'?t|doesn'?t|cannot|can'?t|"
    r"retracted|withdrawn|invalid)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[!?;\r\n]+|(?<!\d)\.(?!\d)|,(?=\s)")


def _has_substantive_visible_reasoning(value: str | None) -> bool:
    """Reject invisible/control-only think blocks and require letters/numbers."""

    if not isinstance(value, str) or not value:
        return False
    categories = [unicodedata.category(character) for character in value]
    if any(category.startswith("C") for category in categories):
        return False
    return any(category.startswith(("L", "N")) for category in categories)


@dataclass(frozen=True)
class MotionCompletion:
    has_explicit_reasoning: bool
    reasoning_text: str | None
    final_answer_text: str


def parse_motion_completion(candidate: Any) -> MotionCompletion:
    """Separate one exact visible ``<think>`` block from final-answer text."""

    if not isinstance(candidate, str):
        candidate = ""
    matches = list(_THINK_RE.finditer(candidate))
    exact_one = (
        len(matches) == 1
        and candidate.count("<think>") == 1
        and candidate.count("</think>") == 1
    )
    if not exact_one:
        malformed_tags = "<think>" in candidate or "</think>" in candidate
        return MotionCompletion(False, None, "" if malformed_tags else candidate.strip())
    match = matches[0]
    reasoning = match.group(1)
    visible = _has_substantive_visible_reasoning(reasoning)
    final_text = (candidate[: match.start()] + candidate[match.end() :]).strip()
    return MotionCompletion(visible, reasoning if visible else None, final_text)


def _canonical_id(
    item: Mapping[str, Any], *, default: str, pattern: str, name: str
) -> str:
    identifier = strict_identifier(item.get("id", default), name=f"{name}.id")
    if re.fullmatch(pattern, identifier) is None:
        raise RubricValidationError(f"{name}.id has the wrong namespace: {identifier!r}")
    return identifier


def _fact_list(
    value: Any,
    *,
    name: str,
    count_min: int,
    count_max: int,
    segment_index: int | None,
    id_code: str,
    sources: frozenset[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not count_min <= len(value) <= count_max:
        raise RubricValidationError(
            f"{name} must contain between {count_min} and {count_max} criteria"
        )
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        item = require_mapping(raw, name=f"{name}[{index - 1}]")
        require_exact_keys(
            item,
            name=f"{name}[{index - 1}]",
            required={"criterion", "source"},
            optional={"id"},
        )
        default = f"g{index}" if segment_index is None else f"s{segment_index}_{id_code}{index}"
        pattern = r"g[1-9][0-9]*" if segment_index is None else rf"s{segment_index}_{id_code}[1-9][0-9]*"
        identifier = _canonical_id(
            item,
            default=default,
            pattern=pattern,
            name=f"{name}[{index - 1}]",
        )
        source = strict_text(item.get("source"), name=f"{name}[{index - 1}].source", max_length=32)
        if source not in sources:
            raise RubricValidationError(f"unsupported source {source!r} in {name}")
        output.append(
            {
                "id": identifier,
                "criterion": strict_text(
                    item.get("criterion"),
                    name=f"{name}[{index - 1}].criterion",
                    max_length=2_000,
                ),
                "source": source,
            }
        )
    return output


def _numeric_list(value: Any, *, segment_index: int) -> list[dict[str, Any]]:
    name = f"segments[{segment_index - 1}].numeric_kinematics"
    if not isinstance(value, list) or len(value) != 3:
        raise RubricValidationError(f"{name} must contain exactly 3 criteria")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        item = require_mapping(raw, name=f"{name}[{index - 1}]")
        if set(item) & NUMERIC_POLICY_FIELDS:
            raise RubricValidationError(
                f"{name}[{index - 1}] contains forbidden scoring/tolerance fields"
            )
        require_exact_keys(
            item,
            name=f"{name}[{index - 1}]",
            required={"criterion", "quantity", "body_part", "target_range", "unit", "source"},
            optional={"id"},
        )
        identifier = _canonical_id(
            item,
            default=f"s{segment_index}_n{index}",
            pattern=rf"s{segment_index}_n[1-9][0-9]*",
            name=f"{name}[{index - 1}]",
        )
        target = item.get("target_range")
        if not isinstance(target, list) or len(target) != 2:
            raise RubricValidationError(f"{name}[{index - 1}].target_range must have two values")
        low = finite_number(target[0], name=f"{name}[{index - 1}].target_range[0]")
        high = finite_number(target[1], name=f"{name}[{index - 1}].target_range[1]")
        if low > high:
            raise RubricValidationError(f"{name}[{index - 1}].target_range must be ordered")
        unit = strict_text(item.get("unit"), name=f"{name}[{index - 1}].unit", max_length=16)
        if unit not in NUMERIC_UNITS:
            raise RubricValidationError(f"unsupported numeric unit: {unit!r}")
        if item.get("source") != "motion":
            raise RubricValidationError(f"{name}[{index - 1}].source must be 'motion'")
        output.append(
            {
                "id": identifier,
                "criterion": strict_text(
                    item.get("criterion"), name=f"{name}[{index - 1}].criterion", max_length=2_000
                ),
                "quantity": strict_text(
                    item.get("quantity"), name=f"{name}[{index - 1}].quantity", max_length=128
                ),
                "body_part": strict_text(
                    item.get("body_part"), name=f"{name}[{index - 1}].body_part", max_length=128
                ),
                "target_range": [low, high],
                "unit": unit,
                "source": "motion",
            }
        )
    return output


def _reasoning_list(value: Any, *, segment_index: int) -> list[dict[str, Any]]:
    name = f"segments[{segment_index - 1}].reasoning_criteria"
    if not isinstance(value, list) or len(value) != 2:
        raise RubricValidationError(f"{name} must contain exactly 2 criteria")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        item = require_mapping(raw, name=f"{name}[{index - 1}]")
        require_exact_keys(
            item,
            name=f"{name}[{index - 1}]",
            required={"criterion", "type", "source"},
            optional={"id"},
        )
        identifier = _canonical_id(
            item,
            default=f"s{segment_index}_r{index}",
            pattern=rf"s{segment_index}_r[1-9][0-9]*",
            name=f"{name}[{index - 1}]",
        )
        criterion_type = strict_text(
            item.get("type"), name=f"{name}[{index - 1}].type", max_length=64
        )
        if criterion_type not in REASONING_WEIGHTS:
            raise RubricValidationError(f"unsupported reasoning type: {criterion_type!r}")
        if item.get("source") != "think":
            raise RubricValidationError(f"{name}[{index - 1}].source must be 'think'")
        output.append(
            {
                "id": identifier,
                "criterion": strict_text(
                    item.get("criterion"), name=f"{name}[{index - 1}].criterion", max_length=2_000
                ),
                "type": criterion_type,
                "source": "think",
            }
        )
    return output


def _negative_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 6:
        raise RubricValidationError("negative_criteria must contain exactly 6 criteria")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        item = require_mapping(raw, name=f"negative_criteria[{index - 1}]")
        require_exact_keys(
            item,
            name=f"negative_criteria[{index - 1}]",
            required={"criterion", "type", "source_of_truth"},
            optional={"id"},
        )
        identifier = _canonical_id(
            item,
            default=f"neg{index}",
            pattern=r"neg[1-9][0-9]*",
            name=f"negative_criteria[{index - 1}]",
        )
        criterion_type = strict_text(
            item.get("type"), name=f"negative_criteria[{index - 1}].type", max_length=64
        )
        if criterion_type not in NEGATIVE_TYPES:
            raise RubricValidationError(f"unsupported negative criterion type: {criterion_type!r}")
        source = strict_text(
            item.get("source_of_truth"),
            name=f"negative_criteria[{index - 1}].source_of_truth",
            max_length=32,
        )
        if source not in {"motion", "video", "video+motion"}:
            raise RubricValidationError(f"unsupported source_of_truth: {source!r}")
        output.append(
            {
                "id": identifier,
                "criterion": strict_text(
                    item.get("criterion"),
                    name=f"negative_criteria[{index - 1}].criterion",
                    max_length=2_000,
                ),
                "type": criterion_type,
                "source_of_truth": source,
            }
        )
    return output


def validate_motion_criteria_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = require_mapping(value, name="criteria")
    if "mode" in raw and raw.get("mode") != MOTION_MODE_V2:
        raise RubricValidationError(f"criteria.mode must be {MOTION_MODE_V2!r}")
    require_exact_keys(
        raw,
        name="criteria",
        required={"mode", "global_activity", "segments", "temporal_phases", "negative_criteria"},
    )
    if raw.get("mode") != MOTION_MODE_V2:
        raise RubricValidationError(f"criteria.mode must be {MOTION_MODE_V2!r}")
    global_activity = _fact_list(
        raw.get("global_activity"),
        name="global_activity",
        count_min=1,
        count_max=1,
        segment_index=None,
        id_code="g",
        sources=frozenset({"video", "motion", "video+motion"}),
    )
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RubricValidationError("segments must be a non-empty list")
    segments: list[dict[str, Any]] = []
    times: set[str] = set()
    for segment_index, raw_segment in enumerate(raw_segments, start=1):
        name = f"segments[{segment_index - 1}]"
        segment = require_mapping(raw_segment, name=name)
        require_exact_keys(
            segment,
            name=name,
            required={
                "time",
                "basic_action_facts",
                "body_configuration",
                "numeric_kinematics",
                "laterality",
                "camera_relative_orientation",
                "reasoning_criteria",
                "rejected_claims",
            },
        )
        time_range = strict_text(segment.get("time"), name=f"{name}.time", max_length=128)
        if time_range in times:
            raise RubricValidationError(f"duplicate segment time: {time_range!r}")
        times.add(time_range)
        rejected = segment.get("rejected_claims")
        if rejected != []:
            raise RubricValidationError(f"{name}.rejected_claims must be an empty list in V2")
        segments.append(
            {
                "time": time_range,
                "basic_action_facts": _fact_list(
                    segment.get("basic_action_facts"),
                    name=f"{name}.basic_action_facts",
                    count_min=2,
                    count_max=2,
                    segment_index=segment_index,
                    id_code="a",
                    sources=frozenset({"video", "motion", "video+motion"}),
                ),
                "body_configuration": _fact_list(
                    segment.get("body_configuration"),
                    name=f"{name}.body_configuration",
                    count_min=1,
                    count_max=1,
                    segment_index=segment_index,
                    id_code="b",
                    sources=frozenset({"motion"}),
                ),
                "numeric_kinematics": _numeric_list(
                    segment.get("numeric_kinematics"), segment_index=segment_index
                ),
                "laterality": _fact_list(
                    segment.get("laterality"),
                    name=f"{name}.laterality",
                    count_min=0,
                    count_max=1,
                    segment_index=segment_index,
                    id_code="l",
                    sources=frozenset({"motion"}),
                ),
                "camera_relative_orientation": _fact_list(
                    segment.get("camera_relative_orientation"),
                    name=f"{name}.camera_relative_orientation",
                    count_min=0,
                    count_max=1,
                    segment_index=segment_index,
                    id_code="o",
                    sources=frozenset({"motion", "video+motion"}),
                ),
                "reasoning_criteria": _reasoning_list(
                    segment.get("reasoning_criteria"), segment_index=segment_index
                ),
                "rejected_claims": [],
            }
        )

    phases_raw = raw.get("temporal_phases")
    if not isinstance(phases_raw, list) or not 5 <= len(phases_raw) <= 7:
        raise RubricValidationError("temporal_phases must contain 5 to 7 phases")
    phases = [
        strict_text(item, name=f"temporal_phases[{index}]", max_length=256)
        for index, item in enumerate(phases_raw)
    ]
    if len(phases) != len(set(phases)):
        raise RubricValidationError("temporal_phases must be unique")
    negatives = _negative_list(raw.get("negative_criteria"))

    all_ids = [item["id"] for item in global_activity + negatives]
    for segment in segments:
        for field in (
            "basic_action_facts",
            "body_configuration",
            "numeric_kinematics",
            "laterality",
            "camera_relative_orientation",
            "reasoning_criteria",
        ):
            all_ids.extend(item["id"] for item in segment[field])
    if len(all_ids) != len(set(all_ids)):
        raise RubricValidationError("criterion IDs must be globally unique across categories")
    return {
        "mode": MOTION_MODE_V2,
        "global_activity": global_activity,
        "segments": segments,
        "temporal_phases": phases,
        "negative_criteria": negatives,
    }


def _ids_by_field(criteria: Mapping[str, Any], field: str) -> set[str]:
    return {
        item["id"]
        for segment in criteria["segments"]
        for item in segment[field]
    }


def _reasoning_by_type(criteria: Mapping[str, Any]) -> dict[str, set[str]]:
    output = {name: set() for name in REASONING_WEIGHTS}
    for segment in criteria["segments"]:
        for item in segment["reasoning_criteria"]:
            output[item["type"]].add(item["id"])
    return output


def _sanitize_ids(
    value: Any,
    *,
    name: str,
    valid: set[str],
    reject_unknown_ids: bool,
    invalid: list[str],
) -> list[str]:
    reported = strict_id_list(value, name=name)
    unknown = sorted(set(reported) - valid)
    invalid.extend(f"{name}:{item}" for item in unknown)
    if unknown and reject_unknown_ids:
        raise RubricValidationError(f"{name} contains unknown IDs: {unknown}")
    return [item for item in reported if item in valid]


def validate_motion_judgment_v2(
    value: Mapping[str, Any],
    criteria: Mapping[str, Any],
    *,
    candidate_response: str,
    sample_id: str,
    expected_nonce: str | None = None,
    reject_unknown_ids: bool = True,
) -> dict[str, Any]:
    checked = validate_motion_criteria_v2(criteria)
    raw = require_mapping(value, name="judgment")
    require_exact_keys(
        raw,
        name="judgment",
        required={
            "binding",
            "final_motion_answer",
            "reasoning_process",
            "negative_criteria",
            "language_format_score",
            "hallucination_or_source_contradiction_penalty",
            "hallucinations",
        },
    )
    final = require_mapping(raw.get("final_motion_answer"), name="final_motion_answer")
    require_exact_keys(
        final,
        name="final_motion_answer",
        required={
            "global_activity",
            "basic_action_facts",
            "body_configuration",
            "numeric_kinematics",
            "laterality",
            "camera_relative_orientation",
            "temporal_structure_score",
        },
    )
    valid = {
        "global": {item["id"] for item in checked["global_activity"]},
        "basic": _ids_by_field(checked, "basic_action_facts"),
        "body": _ids_by_field(checked, "body_configuration"),
        "numeric": _ids_by_field(checked, "numeric_kinematics"),
        "laterality": _ids_by_field(checked, "laterality"),
        "orientation": _ids_by_field(checked, "camera_relative_orientation"),
        "reasoning": set().union(*_reasoning_by_type(checked).values()),
        "negative": {item["id"] for item in checked["negative_criteria"]},
    }
    invalid: list[str] = []

    def section(name: str, keys: set[str]) -> Mapping[str, Any]:
        result = require_mapping(final.get(name), name=f"final_motion_answer.{name}")
        require_exact_keys(result, name=f"final_motion_answer.{name}", required=keys)
        return result

    global_j = section("global_activity", {"satisfied_ids", "missed_ids"})
    basic_j = section("basic_action_facts", {"present_ids", "aligned_ids", "missed_ids", "misplaced_ids"})
    body_j = section("body_configuration", {"present_ids", "aligned_ids", "missed_ids", "misplaced_ids"})
    numeric_j = section(
        "numeric_kinematics",
        {
            "semantic_present_ids",
            "strict_value_match_ids",
            "loose_value_match_ids",
            "wrong_value_ids",
            "missed_ids",
            "observed_values",
        },
    )
    lat_j = section("laterality", {"correct_ids", "wrong_ids", "missed_ids"})
    ori_j = section("camera_relative_orientation", {"correct_ids", "wrong_ids", "missed_ids"})
    reasoning_j = require_mapping(raw.get("reasoning_process"), name="reasoning_process")
    require_exact_keys(
        reasoning_j,
        name="reasoning_process",
        required={"satisfied_ids", "missed_ids", "contradicted_ids"},
    )
    negative_j = require_mapping(raw.get("negative_criteria"), name="negative_criteria")
    require_exact_keys(negative_j, name="negative_criteria", required={"triggered_ids"})

    def ids(value_: Any, *, name: str, kind: str) -> list[str]:
        return _sanitize_ids(
            value_,
            name=name,
            valid=valid[kind],
            reject_unknown_ids=reject_unknown_ids,
            invalid=invalid,
        )

    normalized_numeric_ids = {
        key: ids(numeric_j.get(key), name=f"numeric.{key}", kind="numeric")
        for key in (
            "semantic_present_ids",
            "strict_value_match_ids",
            "loose_value_match_ids",
            "wrong_value_ids",
            "missed_ids",
        )
    }
    observed = numeric_j.get("observed_values")
    if not isinstance(observed, list) or len(observed) > len(valid["numeric"]):
        raise RubricValidationError("numeric_kinematics.observed_values has invalid size")
    normalized_observed: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for index, raw_observation in enumerate(observed):
        observation = require_mapping(raw_observation, name=f"observed_values[{index}]")
        require_exact_keys(
            observation,
            name=f"observed_values[{index}]",
            required={"id", "candidate_value", "unit", "candidate_text"},
        )
        identifier_values = ids(
            [observation.get("id")], name=f"observed_values[{index}].id", kind="numeric"
        )
        if not identifier_values:
            continue
        identifier = identifier_values[0]
        if identifier in observed_ids:
            raise RubricValidationError(f"duplicate observed numeric id: {identifier!r}")
        observed_ids.add(identifier)
        candidate_value = observation.get("candidate_value")
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, (str, int, float)):
            raise RubricValidationError("candidate_value must be a string or finite number")
        if isinstance(candidate_value, (int, float)):
            finite_number(candidate_value, name=f"observed_values[{index}].candidate_value")
        elif not _numbers(candidate_value):
            raise RubricValidationError(
                f"observed_values[{index}].candidate_value must contain a finite number"
            )
        observation_unit = strict_text(
            observation.get("unit"), name=f"observed_values[{index}].unit", max_length=16
        )
        if observation_unit not in NUMERIC_UNITS:
            raise RubricValidationError(
                f"observed_values[{index}].unit is unsupported: {observation_unit!r}"
            )
        normalized_observed.append(
            {
                "id": identifier,
                "candidate_value": candidate_value,
                "unit": observation_unit,
                "candidate_text": strict_text(
                    observation.get("candidate_text"),
                    name=f"observed_values[{index}].candidate_text",
                    max_length=1_000,
                ),
            }
        )

    numeric_claim_ids: set[str] = set()
    for key in ("strict_value_match_ids", "loose_value_match_ids", "wrong_value_ids"):
        numeric_claim_ids.update(normalized_numeric_ids[key])
    missing_observations = sorted(numeric_claim_ids - observed_ids)
    if missing_observations:
        raise RubricValidationError(
            "every numeric value-match/wrong claim requires an observed_values row; "
            f"missing={missing_observations}"
        )

    temporal = final.get("temporal_structure_score")
    if isinstance(temporal, bool) or not isinstance(temporal, int) or temporal not in TEMPORAL_SCORES:
        raise RubricValidationError(f"temporal_structure_score must be one of {sorted(TEMPORAL_SCORES)}")
    language = raw.get("language_format_score")
    if isinstance(language, bool) or not isinstance(language, int) or not 0 <= language <= 10:
        raise RubricValidationError("language_format_score must be an integer from 0 to 10")
    penalty = raw.get("hallucination_or_source_contradiction_penalty")
    if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty not in PENALTIES:
        raise RubricValidationError(f"penalty must be one of {sorted(PENALTIES)}")
    hallucinations = raw.get("hallucinations")
    if not isinstance(hallucinations, list) or len(hallucinations) > 100:
        raise RubricValidationError("hallucinations must be a list with at most 100 strings")
    hallucination_text = [
        strict_text(item, name=f"hallucinations[{index}]", max_length=1_000)
        for index, item in enumerate(hallucinations)
    ]
    binding = validate_judgment_binding(
        raw.get("binding"),
        checked,
        candidate_response,
        sample_id=sample_id,
        expected_nonce=expected_nonce,
    )
    result = {
        "binding": binding,
        "final_motion_answer": {
            "global_activity": {
                "satisfied_ids": ids(global_j.get("satisfied_ids"), name="global.satisfied_ids", kind="global"),
                "missed_ids": ids(global_j.get("missed_ids"), name="global.missed_ids", kind="global"),
            },
            "basic_action_facts": {
                key: ids(basic_j.get(key), name=f"basic.{key}", kind="basic")
                for key in ("present_ids", "aligned_ids", "missed_ids", "misplaced_ids")
            },
            "body_configuration": {
                key: ids(body_j.get(key), name=f"body.{key}", kind="body")
                for key in ("present_ids", "aligned_ids", "missed_ids", "misplaced_ids")
            },
            "numeric_kinematics": {
                **normalized_numeric_ids,
                "observed_values": normalized_observed,
            },
            "laterality": {
                key: ids(lat_j.get(key), name=f"laterality.{key}", kind="laterality")
                for key in ("correct_ids", "wrong_ids", "missed_ids")
            },
            "camera_relative_orientation": {
                key: ids(ori_j.get(key), name=f"orientation.{key}", kind="orientation")
                for key in ("correct_ids", "wrong_ids", "missed_ids")
            },
            "temporal_structure_score": temporal,
        },
        "reasoning_process": {
            key: ids(reasoning_j.get(key), name=f"reasoning.{key}", kind="reasoning")
            for key in ("satisfied_ids", "missed_ids", "contradicted_ids")
        },
        "negative_criteria": {
            "triggered_ids": ids(
                negative_j.get("triggered_ids"), name="negative.triggered_ids", kind="negative"
            )
        },
        "language_format_score": language,
        "hallucination_or_source_contradiction_penalty": penalty,
        "hallucinations": hallucination_text,
        "invalid_ids_removed": sorted(invalid),
    }

    normalized_final = result["final_motion_answer"]
    require_partition(
        valid["global"],
        normalized_final["global_activity"],
        name="judgment.final_motion_answer.global_activity",
    )
    for section_name, kind in (
        ("basic_action_facts", "basic"),
        ("body_configuration", "body"),
    ):
        section_value = normalized_final[section_name]
        present_partition = require_partition(
            valid[kind],
            {
                "present_ids": section_value["present_ids"],
                "missed_ids": section_value["missed_ids"],
            },
            name=f"judgment.final_motion_answer.{section_name}.presence",
        )
        require_partition(
            present_partition["present_ids"],
            {
                "aligned_ids": section_value["aligned_ids"],
                "misplaced_ids": section_value["misplaced_ids"],
            },
            name=f"judgment.final_motion_answer.{section_name}.placement",
        )
    require_partition(
        valid["numeric"],
        {
            key: normalized_final["numeric_kinematics"][key]
            for key in (
                "semantic_present_ids",
                "strict_value_match_ids",
                "loose_value_match_ids",
                "wrong_value_ids",
                "missed_ids",
            )
        },
        name="judgment.final_motion_answer.numeric_kinematics",
    )
    claimed_numeric = set().union(
        *(
            set(normalized_final["numeric_kinematics"][key])
            for key in ("strict_value_match_ids", "loose_value_match_ids", "wrong_value_ids")
        )
    )
    if observed_ids != claimed_numeric:
        raise RubricValidationError(
            "numeric observed_values IDs must exactly match strict/loose/wrong IDs"
        )
    for section_name, kind in (
        ("laterality", "laterality"),
        ("camera_relative_orientation", "orientation"),
    ):
        require_partition(
            valid[kind],
            normalized_final[section_name],
            name=f"judgment.final_motion_answer.{section_name}",
        )
    require_partition(
        valid["reasoning"],
        result["reasoning_process"],
        name="judgment.reasoning_process",
    )
    if not parse_motion_completion(candidate_response).has_explicit_reasoning:
        result["reasoning_process"] = {
            "satisfied_ids": [],
            "missed_ids": sorted(valid["reasoning"]),
            "contradicted_ids": [],
        }
    return result


def parse_motion_judgment_v2_text(
    text: str,
    criteria: Mapping[str, Any],
    *,
    candidate_response: str,
    sample_id: str,
    expected_nonce: str | None = None,
    reject_unknown_ids: bool = True,
) -> dict[str, Any]:
    return validate_motion_judgment_v2(
        strict_json_object(text),
        criteria,
        candidate_response=candidate_response,
        sample_id=sample_id,
        expected_nonce=expected_nonce,
        reject_unknown_ids=reject_unknown_ids,
    )


def _fraction(max_score: float, matched: float, total: int) -> float:
    return 0.0 if total <= 0 else max_score * matched / total


def _present_aligned(max_score: float, present: set[str], aligned: set[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return max_score * (0.3 * len(present) / total + 0.7 * len(aligned) / total)


def _numeric_items(criteria: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: item
        for segment in criteria["segments"]
        for item in segment["numeric_kinematics"]
    }


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, str):
        return []
    output: list[float] = []
    for match in _NUMBER_RE.finditer(
        value.replace("\u2013", " ").replace("\u2014", " ")
    ):
        try:
            number = float(match.group(0))
        except ValueError:
            continue
        if number == number and abs(number) != float("inf"):
            output.append(number)
    return output


def _has_unit(text: str, unit: str) -> bool:
    lowered = text.casefold()
    if unit == "degrees":
        return "\u00b0" in text or re.search(r"\b(?:deg|degree|degrees)\b", lowered) is not None
    if unit == "m":
        return re.search(r"(?<![A-Za-z])m(?![A-Za-z])|\bmeters?\b", lowered) is not None
    return re.search(r"(?<![A-Za-z])s(?![A-Za-z])|\b(?:sec|secs|second|seconds)\b", lowered) is not None


def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    """Build a conservative phrase matcher for body-part association.

    Numeric evidence is security-sensitive reward input.  Directional tokens
    such as ``left``/``right`` must therefore remain part of the match; accepting
    a shared token such as ``knee`` would let a left-knee value satisfy a
    right-knee criterion.
    """

    tokens = [token.casefold() for token in _WORD_RE.findall(phrase)]
    if not tokens:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])"
        + r"(?:[\s_-]+)".join(re.escape(token) for token in tokens)
        + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _measurement_windows(text: str, item: Mapping[str, Any]) -> list[str]:
    """Return complete clauses containing body-part and quantity evidence.

    Clause boundaries, rather than fixed character windows, keep association
    deterministic without allowing a long filler string to move a conflicting
    value outside the verifier's view.
    """

    body_pattern = _phrase_pattern(str(item["body_part"]))
    if body_pattern is None:
        return []
    quantity = str(item["quantity"])
    quantity_pattern = _phrase_pattern(quantity)
    quantity_tokens = {
        token.casefold()
        for token in _WORD_RE.findall(quantity)
        if token.casefold() not in {"the", "and", "with", "from", "value", "motion"}
    }
    clauses: list[str] = []
    start = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(text):
        clauses.append(text[start : boundary.start()])
        start = boundary.end()
    clauses.append(text[start:])

    output: list[str] = []
    for clause in clauses:
        if body_pattern.search(clause) is None:
            continue
        lowered = clause.casefold()
        quantity_present = bool(quantity_pattern and quantity_pattern.search(clause))
        if not quantity_present and quantity_tokens:
            quantity_present = any(
                re.search(rf"\b{re.escape(token)}\b", lowered)
                for token in quantity_tokens
            )
        if quantity_present:
            output.append(clause)
    return output


def _mentions_measurement(text: str, item: Mapping[str, Any]) -> bool:
    return bool(_measurement_windows(text, item))


def _measurement_window_has_number(text: str, item: Mapping[str, Any]) -> bool:
    return any(_numbers(window) for window in _measurement_windows(text, item))


def _value_class(values: list[float], item: Mapping[str, Any]) -> str:
    low, high = item["target_range"]
    strict, loose = TOLERANCES[item["unit"]]
    if values and all(low - strict <= value <= high + strict for value in values):
        return "strict"
    if values and all(low - loose <= value <= high + loose for value in values):
        return "loose"
    return "wrong"


def _evidence_is_assertive(candidate: str, evidence: str) -> bool:
    """Reject quoted, negated, hypothetical, or subsequently retracted evidence."""

    quoted = tuple((match.start(), match.end()) for match in _QUOTED_RE.finditer(candidate))
    start = 0
    while True:
        index = candidate.find(evidence, start)
        if index < 0:
            return False
        end = index + len(evidence)
        start = index + 1
        if any(index < quote_end and end > quote_start for quote_start, quote_end in quoted):
            continue
        left_boundary = max(
            candidate.rfind(".", 0, index),
            candidate.rfind("!", 0, index),
            candidate.rfind("?", 0, index),
            candidate.rfind("\n", 0, index),
        )
        right_candidates = [
            position
            for delimiter in (".", "!", "?", "\n")
            if (position := candidate.find(delimiter, end)) >= 0
        ]
        right_boundary = min(right_candidates) if right_candidates else len(candidate)
        prefix = candidate[left_boundary + 1 : index]
        suffix = candidate[end:right_boundary]
        line_prefix = candidate[candidate.rfind("\n", 0, index) + 1 : index]
        if re.match(r"^\s*>", line_prefix):
            continue
        if right_boundary < len(candidate) and candidate[right_boundary] == "?":
            continue
        local = prefix + evidence + suffix
        if _NON_ASSERTIVE_RE.search(local):
            continue
        correction_tail = candidate[end:]
        if _EXPLICIT_RETRACTION_RE.search(correction_tail):
            continue
        if _CORRECTION_RE.search(correction_tail) and _numbers(correction_tail):
            continue
        return True


def _evidence_has_adverse_polarity(candidate: str, evidence: str) -> bool:
    """Return true when visible evidence is negated, withdrawn, or corrected."""

    start = 0
    while True:
        index = candidate.find(evidence, start)
        if index < 0:
            return False
        end = index + len(evidence)
        start = index + 1
        left_boundary = max(
            candidate.rfind(".", 0, index),
            candidate.rfind("!", 0, index),
            candidate.rfind("?", 0, index),
            candidate.rfind("\n", 0, index),
        )
        right_candidates = [
            position
            for delimiter in (".", "!", "?", "\n")
            if (position := candidate.find(delimiter, end)) >= 0
        ]
        right_boundary = min(right_candidates) if right_candidates else len(candidate)
        local = candidate[left_boundary + 1 : right_boundary]
        tail = candidate[end:]
        if _ADVERSE_POLARITY_RE.search(local):
            return True
        if _EXPLICIT_RETRACTION_RE.search(tail):
            return True
        if _CORRECTION_RE.search(tail) and _numbers(tail):
            return True


def _assertive_measurement_mention(text: str, item: Mapping[str, Any]) -> bool:
    return _mentions_measurement(text, item) and _evidence_is_assertive(
        text, str(item["body_part"])
    )


def _verified_numeric(
    criteria: Mapping[str, Any],
    numeric_j: Mapping[str, Any],
    candidate_response: Any,
) -> tuple[set[str], set[str], set[str], set[str], list[str]]:
    items = _numeric_items(criteria)
    reported_strict = set(numeric_j["strict_value_match_ids"])
    reported_loose = set(numeric_j["loose_value_match_ids"])
    strict: set[str] = set()
    loose: set[str] = set()
    wrong: set[str] = set(numeric_j["wrong_value_ids"])
    invalid_observations: list[str] = []
    candidate = candidate_response if isinstance(candidate_response, str) else ""
    for identifier, item in items.items():
        claim_classes = [
            _value_class(_numbers(clause), item)
            for clause in _measurement_windows(candidate, item)
            if _has_unit(clause, item["unit"]) and _numbers(clause)
        ]
        if "wrong" in claim_classes:
            wrong.add(identifier)
    for observation in numeric_j["observed_values"]:
        identifier = observation["id"]
        item = items[identifier]
        text = observation["candidate_text"]
        observed_numbers = _numbers(observation["candidate_value"])
        text_numbers = _numbers(text)
        verified = (
            bool(candidate)
            and _evidence_is_assertive(candidate, text)
            and observation["unit"] == item["unit"]
            and _has_unit(text, item["unit"])
            and _mentions_measurement(text, item)
            and bool(observed_numbers)
            and len(observed_numbers) == len(text_numbers)
            and all(
                abs(value - text_value) <= 1e-9
                for value, text_value in zip(sorted(observed_numbers), sorted(text_numbers))
            )
        )
        if not verified:
            invalid_observations.append(identifier)
            if text in candidate and _evidence_has_adverse_polarity(candidate, text):
                wrong.add(identifier)
            continue
        classification = _value_class(observed_numbers, item)
        if identifier in wrong or classification == "wrong":
            wrong.add(identifier)
        elif classification == "strict" and identifier in reported_strict:
            strict.add(identifier)
        elif identifier in reported_strict | reported_loose:
            # A disagreement between strict and loose is deterministically
            # resolved to the less favorable classification.
            loose.add(identifier)
        else:
            invalid_observations.append(identifier)
    conflicts = (strict | loose) & wrong
    strict -= wrong
    loose -= wrong | strict

    semantic: set[str] = set()
    # Semantic-only credit is deliberately conservative: a number close to the
    # criterion's body part/quantity requires a verified observation. Unrelated
    # timestamps elsewhere in a long report do not suppress semantic credit.
    if isinstance(candidate_response, str):
        for identifier in numeric_j["semantic_present_ids"]:
            if (
                identifier not in wrong
                and _assertive_measurement_mention(
                    candidate_response, items[identifier]
                )
                and not _measurement_window_has_number(candidate_response, items[identifier])
            ):
                semantic.add(identifier)
    semantic -= wrong
    return strict, loose, semantic, wrong, sorted(set(invalid_observations) | conflicts)


def compute_motion_reward_v2(
    criteria: Mapping[str, Any],
    judgment: Mapping[str, Any],
    *,
    sample_id: str,
    candidate_response: Any = None,
) -> dict[str, Any]:
    checked = validate_motion_criteria_v2(criteria)
    candidate_text = candidate_response if isinstance(candidate_response, str) else ""
    completion = parse_motion_completion(candidate_text)
    judgment_input = dict(judgment)
    judgment_input.pop("invalid_ids_removed", None)
    judged = validate_motion_judgment_v2(
        judgment_input,
        checked,
        candidate_response=candidate_text,
        sample_id=sample_id,
        reject_unknown_ids=False,
    )
    final = judged["final_motion_answer"]
    valid_global = {item["id"] for item in checked["global_activity"]}
    valid_basic = _ids_by_field(checked, "basic_action_facts")
    valid_body = _ids_by_field(checked, "body_configuration")
    valid_numeric = _ids_by_field(checked, "numeric_kinematics")
    valid_lat = _ids_by_field(checked, "laterality")
    valid_ori = _ids_by_field(checked, "camera_relative_orientation")
    reasoning_by_type = _reasoning_by_type(checked)

    global_sat = set(final["global_activity"]["satisfied_ids"])
    basic_present = set(final["basic_action_facts"]["present_ids"])
    basic_aligned = set(final["basic_action_facts"]["aligned_ids"]) & basic_present
    body_present = set(final["body_configuration"]["present_ids"])
    body_aligned = set(final["body_configuration"]["aligned_ids"]) & body_present
    numeric_strict, numeric_loose, numeric_semantic, numeric_wrong, numeric_invalid = _verified_numeric(
        checked, final["numeric_kinematics"], completion.final_answer_text
    )

    lat_correct_reported = set(final["laterality"]["correct_ids"])
    lat_wrong = set(final["laterality"]["wrong_ids"])
    lat_conflicts = lat_correct_reported & lat_wrong
    lat_correct = lat_correct_reported - lat_wrong
    ori_correct_reported = set(final["camera_relative_orientation"]["correct_ids"])
    ori_wrong = set(final["camera_relative_orientation"]["wrong_ids"])
    ori_conflicts = ori_correct_reported & ori_wrong
    ori_correct = ori_correct_reported - ori_wrong
    valid_reasoning = set().union(*reasoning_by_type.values())
    reasoning_sat_reported = set(judged["reasoning_process"]["satisfied_ids"])
    reasoning_contra = set(judged["reasoning_process"]["contradicted_ids"])
    reasoning_missed = set(judged["reasoning_process"]["missed_ids"])
    reasoning_conflicts = reasoning_sat_reported & reasoning_contra
    reasoning_sat = reasoning_sat_reported - reasoning_contra
    if not completion.has_explicit_reasoning:
        reasoning_sat.clear()
        reasoning_contra.clear()
        reasoning_conflicts.clear()
        reasoning_missed = valid_reasoning

    numeric_sum = (
        len(numeric_strict)
        + 0.7 * len(numeric_loose)
        + 0.25 * len(numeric_semantic)
    )
    reasoning_raw = 0.0
    reasoning_possible = 0.0
    for criterion_type, weight in REASONING_WEIGHTS.items():
        identifiers = reasoning_by_type[criterion_type]
        if identifiers:
            reasoning_possible += weight
            reasoning_raw += _fraction(weight, len(reasoning_sat & identifiers), len(identifiers))
    reasoning_score = 0.0 if reasoning_possible <= 0 else 20.0 * reasoning_raw / reasoning_possible

    global_score = _fraction(5.0, len(global_sat), len(valid_global))
    basic_score = _present_aligned(25.0, basic_present, basic_aligned, len(valid_basic))
    temporal_score = final["temporal_structure_score"]
    body_score = _present_aligned(10.0, body_present, body_aligned, len(valid_body))
    numeric_score = _fraction(10.0, numeric_sum, len(valid_numeric))
    laterality_score = _fraction(5.0, len(lat_correct), len(valid_lat))
    orientation_score = _fraction(5.0, len(ori_correct), len(valid_ori))
    language_score = judged["language_format_score"]

    negative_by_id = {item["id"]: item for item in checked["negative_criteria"]}
    triggered = set(judged["negative_criteria"]["triggered_ids"])
    severe_negative = {
        identifier
        for identifier in triggered
        if negative_by_id[identifier]["type"]
        in {"contradiction", "wrong_laterality", "wrong_orientation", "numeric_contradiction"}
    }
    contradiction_ids = (
        numeric_wrong
        | lat_wrong
        | ori_wrong
        | reasoning_contra
        | lat_conflicts
        | ori_conflicts
        | reasoning_conflicts
    )
    contradiction_count = len(contradiction_ids) + len(severe_negative)
    deterministic_penalty = 0
    if triggered or judged["hallucinations"]:
        deterministic_penalty = -5
    if contradiction_count:
        deterministic_penalty = min(deterministic_penalty, -10)
    if contradiction_count >= 2:
        deterministic_penalty = min(deterministic_penalty, -15)
    if contradiction_count >= 3:
        deterministic_penalty = min(deterministic_penalty, -20)
    if contradiction_count >= 5:
        deterministic_penalty = -25
    penalty = min(
        judged["hallucination_or_source_contradiction_penalty"],
        deterministic_penalty,
    )

    raw_score = (
        global_score
        + basic_score
        + temporal_score
        + body_score
        + numeric_score
        + laterality_score
        + orientation_score
        + reasoning_score
        + language_score
        + penalty
    )
    total_score = max(0.0, min(100.0, raw_score))
    return {
        "rubric_version": MOTION_RUBRIC_V2_VERSION,
        "global_activity_score": global_score,
        "basic_action_score": basic_score,
        "temporal_structure_score": temporal_score,
        "body_configuration_score": body_score,
        "numeric_kinematics_score": numeric_score,
        "laterality_score": laterality_score,
        "camera_orientation_score": orientation_score,
        "reasoning_score": reasoning_score,
        "language_score": language_score,
        "hallucination_or_source_contradiction_penalty": penalty,
        "raw_score": raw_score,
        "total_score": total_score,
        "reward": total_score / 100.0,
        "debug": {
            "global_total": len(valid_global),
            "basic_total": len(valid_basic),
            "body_total": len(valid_body),
            "numeric_total": len(valid_numeric),
            "laterality_total": len(valid_lat),
            "orientation_total": len(valid_ori),
            "reasoning_total": sum(len(value) for value in reasoning_by_type.values()),
            "has_explicit_reasoning": completion.has_explicit_reasoning,
            "numeric_strict_ids": sorted(numeric_strict),
            "numeric_loose_ids": sorted(numeric_loose),
            "numeric_semantic_ids": sorted(numeric_semantic),
            "numeric_wrong_ids": sorted(numeric_wrong),
            "invalid_or_conflicting_numeric_observations": numeric_invalid,
            "laterality_correct_ids": sorted(lat_correct),
            "laterality_wrong_ids": sorted(lat_wrong),
            "laterality_conflict_ids": sorted(lat_conflicts),
            "orientation_correct_ids": sorted(ori_correct),
            "orientation_wrong_ids": sorted(ori_wrong),
            "orientation_conflict_ids": sorted(ori_conflicts),
            "reasoning_satisfied_ids": sorted(reasoning_sat),
            "reasoning_missed_ids": sorted(reasoning_missed),
            "reasoning_contradicted_ids": sorted(reasoning_contra),
            "reasoning_conflict_ids": sorted(reasoning_conflicts),
            "triggered_negative_ids": sorted(triggered),
            "invalid_ids_removed": judged["invalid_ids_removed"],
        },
    }


__all__ = [
    "MotionCompletion",
    "MOTION_MODE_V2",
    "MOTION_RUBRIC_V2_VERSION",
    "compute_motion_reward_v2",
    "parse_motion_completion",
    "parse_motion_judgment_v2_text",
    "validate_motion_criteria_v2",
    "validate_motion_judgment_v2",
]
