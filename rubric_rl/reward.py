"""Criteria normalization and reward post-processing for rubric RL."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Set


VALID_TEMPORAL_SCORES = (0, 5, 10, 15)
VALID_HALLUCINATION_PENALTIES = (0, -5, -10, -15, -20)
STAGE1_MODE = "temporal_caption"


def _nearest(value: Any, allowed: Iterable[int], default: int = 0) -> int:
    try:
        numeric = float(value)
    except Exception:
        return default
    return min(allowed, key=lambda x: abs(x - numeric))


def _clamp_language(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = 0
    return max(0, min(10, numeric))


def _list_ids(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    return [str(x) for x in items if isinstance(x, (str, int))]


def ensure_criteria_ids(criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of criteria where every fact has a stable ID.

    Offline extraction prompts return fact strings by default. This normalizer
    converts them to HealthBench-like criterion objects:
    global facts -> g1, g2, ...
    segment facts -> s{segment_index}_f{fact_index}
    negative criteria -> n1, n2, ...
    """
    out = deepcopy(criteria)
    mode = out.get("mode")
    if mode is not None and mode != STAGE1_MODE:
        raise ValueError(
            f"Stage 1 reward accepts only mode={STAGE1_MODE!r}; got {mode!r}"
        )

    global_facts = []
    for idx, item in enumerate(out.get("global_facts") or [], start=1):
        if isinstance(item, dict):
            criterion = str(item.get("criterion") or item.get("fact") or "")
            fact = dict(item)
            fact["id"] = str(fact.get("id") or f"g{idx}")
            fact["criterion"] = criterion
        else:
            fact = {"id": f"g{idx}", "criterion": str(item)}
        if fact["criterion"].strip():
            global_facts.append(fact)
    out["global_facts"] = global_facts

    segments = []
    for seg_idx, segment in enumerate(out.get("segments") or [], start=1):
        if not isinstance(segment, dict):
            continue
        facts = []
        for fact_idx, item in enumerate(segment.get("facts") or [], start=1):
            if isinstance(item, dict):
                criterion = str(item.get("criterion") or item.get("fact") or "")
                fact = dict(item)
                fact["id"] = str(fact.get("id") or f"s{seg_idx}_f{fact_idx}")
                fact["criterion"] = criterion
            else:
                fact = {"id": f"s{seg_idx}_f{fact_idx}", "criterion": str(item)}
            if fact["criterion"].strip():
                facts.append(fact)
        segment_out = dict(segment)
        segment_out["facts"] = facts
        segments.append(segment_out)
    out["segments"] = segments

    negatives = []
    for idx, item in enumerate(out.get("negative_criteria") or [], start=1):
        if isinstance(item, dict):
            criterion = str(item.get("criterion") or item.get("fact") or "")
            neg = dict(item)
            neg["id"] = str(neg.get("id") or f"n{idx}")
            neg["criterion"] = criterion
            neg.setdefault("type", "unsupported_detail")
        else:
            neg = {"id": f"n{idx}", "criterion": str(item), "type": "unsupported_detail"}
        if neg["criterion"].strip():
            negatives.append(neg)
    out["negative_criteria"] = negatives
    out["mode"] = STAGE1_MODE
    return out


def global_ids(criteria: Dict[str, Any]) -> Set[str]:
    return {str(item["id"]) for item in criteria.get("global_facts", []) if isinstance(item, dict) and "id" in item}


def segment_ids(criteria: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for segment in criteria.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for fact in segment.get("facts", []):
            if isinstance(fact, dict) and "id" in fact:
                ids.add(str(fact["id"]))
    return ids


def compute_reward(criteria: Dict[str, Any], judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute final reward from Qwen judge ID output.

    The model is intentionally not trusted for counts or arithmetic:
    invalid IDs are removed, aligned IDs are intersected with present IDs,
    and all category scores are computed here.
    """
    criteria = ensure_criteria_ids(criteria)
    valid_global = global_ids(criteria)
    valid_segment = segment_ids(criteria)

    global_judgment = judgment.get("global_action") if isinstance(judgment.get("global_action"), dict) else {}
    segment_judgment = (
        judgment.get("segment_level_caption_facts")
        if isinstance(judgment.get("segment_level_caption_facts"), dict)
        else {}
    )

    raw_global_satisfied = set(_list_ids(global_judgment.get("satisfied_ids")))
    raw_global_missed = set(_list_ids(global_judgment.get("missed_ids")))
    raw_present = set(_list_ids(segment_judgment.get("present_ids")))
    raw_aligned = set(_list_ids(segment_judgment.get("aligned_ids")))
    raw_missed = set(_list_ids(segment_judgment.get("missed_ids")))
    raw_misplaced = set(_list_ids(segment_judgment.get("misplaced_ids")))

    satisfied_global = raw_global_satisfied & valid_global
    present = raw_present & valid_segment
    aligned = (raw_aligned & valid_segment) & present
    missed = raw_missed & valid_segment
    misplaced = raw_misplaced & valid_segment

    g_total = max(len(valid_global), 1)
    s_total = max(len(valid_segment), 1)
    global_score = 20.0 * len(satisfied_global) / g_total
    segment_score = 55.0 * (
        0.3 * len(present) / s_total
        + 0.7 * len(aligned) / s_total
    )
    temporal_score = _nearest(
        judgment.get("temporal_order_score"),
        VALID_TEMPORAL_SCORES,
        default=0,
    )
    language_score = _clamp_language(judgment.get("language_conciseness_score"))
    hallucination_penalty = _nearest(
        judgment.get("hallucination_penalty"),
        VALID_HALLUCINATION_PENALTIES,
        default=0,
    )

    raw_score = global_score + segment_score + temporal_score + language_score + hallucination_penalty
    total_score = max(0.0, min(100.0, raw_score))

    invalid_ids = sorted(
        (raw_global_satisfied | raw_global_missed) - valid_global
    ) + sorted((raw_present | raw_aligned | raw_missed | raw_misplaced) - valid_segment)

    return {
        "global_score": global_score,
        "segment_score": segment_score,
        "temporal_score": temporal_score,
        "language_score": language_score,
        "hallucination_penalty": hallucination_penalty,
        "raw_score": raw_score,
        "total_score": total_score,
        "reward": total_score / 100.0,
        "debug": {
            "global_total": len(valid_global),
            "segment_total": len(valid_segment),
            "global_satisfied_ids": sorted(satisfied_global),
            "segment_present_ids": sorted(present),
            "segment_aligned_ids": sorted(aligned),
            "segment_missed_ids_reported": sorted(missed),
            "segment_misplaced_ids_reported": sorted(misplaced),
            "invalid_ids_removed": invalid_ids,
            "aligned_not_present_removed": sorted((raw_aligned & valid_segment) - present),
            "present_missed_overlap": sorted(present & missed),
            "global_satisfied_missed_overlap": sorted(satisfied_global & raw_global_missed),
        },
    }
