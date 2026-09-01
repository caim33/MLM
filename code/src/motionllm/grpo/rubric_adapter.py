"""Fail-closed batch adapters from ms-swift columns to Rubric-RL cores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .motion_rubric_v2 import (
    compute_motion_reward_v2,
    validate_motion_criteria_v2,
    validate_motion_judgment_v2,
)
from .qa_rubric import (
    compute_qa_rubric_reward,
    validate_qa_criteria,
    validate_qa_judgment,
)
from .rewards import completion_text
from .rubric_common import (
    RubricValidationError,
    strict_identifier,
    strict_json_object,
)
from .rubric_online import OnlineRubricJudge
from .swift_adapter import build_reward_metadata_batch


def _num_generations(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        unique = set(value)
        if len(unique) != 1:
            raise RubricValidationError("num_generations must be constant")
        value = next(iter(unique))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RubricValidationError("num_generations must be a positive integer")
    return value


def _column(
    value: Any,
    size: int,
    *,
    name: str,
    num_generations: int | None,
    allow_none: bool = False,
    allow_scalar_generation_broadcast: bool = True,
    allow_sequence_generation_expansion: bool = True,
) -> list[Any]:
    if value is None:
        if allow_none:
            return [None] * size
        raise RubricValidationError(f"{name} is required")
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        if size == 1:
            return [value]
        if allow_scalar_generation_broadcast and num_generations == size:
            return [value] * size
        raise RubricValidationError(
            f"{name} must provide one value per completion; scalar broadcast is ambiguous"
        )
    items = list(value)
    if len(items) == size:
        return items
    if (
        allow_sequence_generation_expansion
        and num_generations is not None
        and len(items) * num_generations == size
    ):
        return [item for item in items for _ in range(num_generations)]
    raise RubricValidationError(
        f"{name} length {len(items)} does not match completions {size}"
    )


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        return strict_json_object(value)
    if not isinstance(value, Mapping):
        raise RubricValidationError(f"{name} must be a JSON object or JSON-object string")
    return value


def _generation_broadcast_ranges(
    value: Any,
    size: int,
    *,
    num_generations: int | None,
) -> list[tuple[int, int]]:
    """Identify completion ranges produced by criteria generation-broadcast."""

    if size <= 1 or num_generations is None:
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        return [(0, size)] if num_generations == size else []
    item_count = len(value)
    if item_count >= size or item_count * num_generations != size:
        return []
    return [
        (index * num_generations, (index + 1) * num_generations)
        for index in range(item_count)
    ]


def qa_rubric_rewards(
    completions: Sequence[Any],
    *,
    qa_rubric_criteria: Any,
    qa_rubric_judgment: Any = None,
    judge_client: OnlineRubricJudge | None = None,
    sample_id: Any,
    group_id: Any,
    branch: Any,
    rollout_id: Any,
    answer: Any = None,
    gold_answer: Any = None,
    solution: Any = None,
    request_id: Any = None,
    num_generations: Any = None,
) -> list[float]:
    values = list(completions)
    metadata = build_reward_metadata_batch(
        len(values),
        sample_id=sample_id,
        group_id=group_id,
        branch=branch,
        rollout_id=rollout_id,
        answer=answer,
        gold_answer=gold_answer,
        solution=solution,
        request_id=request_id,
        num_generations=num_generations,
    )
    generation_count = _num_generations(num_generations)
    criteria_values = _column(
        qa_rubric_criteria,
        len(values),
        name="qa_rubric_criteria",
        num_generations=generation_count,
    )
    judgment_values = _column(
        qa_rubric_judgment,
        len(values),
        name="qa_rubric_judgment",
        num_generations=generation_count,
        allow_none=True,
        allow_scalar_generation_broadcast=False,
        allow_sequence_generation_expansion=False,
    )
    output: list[float] = []
    for index, (completion, meta, criteria_value, judgment_value) in enumerate(
        zip(values, metadata, criteria_values, judgment_values)
    ):
        text = completion_text(completion)
        if text is None:
            output.append(0.0)
            continue
        criteria = validate_qa_criteria(
            _object(criteria_value, name=f"qa_rubric_criteria[{index}]")
        )
        if criteria["benchmark_id"] != meta.sample_id:
            raise RubricValidationError("QA rubric benchmark_id does not match sample_id")
        if criteria["correct_option"] != meta.gold_answer:
            raise RubricValidationError("QA rubric gold does not match reward metadata answer")
        if judgment_value is None:
            if judge_client is None:
                raise RubricValidationError(
                    "qa_rubric_judgment or a configured online judge is required"
                )
            judgment = judge_client.judge_qa(criteria, text)
        else:
            judgment = validate_qa_judgment(
                _object(judgment_value, name=f"qa_rubric_judgment[{index}]"),
                criteria,
                candidate_response=text,
                reject_unknown_ids=True,
            )
        output.append(compute_qa_rubric_reward(criteria, text, judgment)["reward"])
    return output


def motion_rubric_v2_rewards(
    completions: Sequence[Any],
    *,
    motion_rubric_v2_criteria: Any,
    motion_rubric_v2_id: Any,
    sample_id: Any,
    motion_rubric_v2_judgment: Any = None,
    judge_client: OnlineRubricJudge | None = None,
    num_generations: Any = None,
) -> list[float]:
    values = list(completions)
    generation_count = _num_generations(num_generations)
    criteria_values = _column(
        motion_rubric_v2_criteria,
        len(values),
        name="motion_rubric_v2_criteria",
        num_generations=generation_count,
    )
    rubric_ids = _column(
        motion_rubric_v2_id,
        len(values),
        name="motion_rubric_v2_id",
        num_generations=generation_count,
    )
    sample_ids = _column(
        sample_id,
        len(values),
        name="sample_id",
        num_generations=generation_count,
    )
    normalized_rubric_ids = [
        strict_identifier(value, name=f"motion_rubric_v2_id[{index}]")
        for index, value in enumerate(rubric_ids)
    ]
    normalized_sample_ids = [
        strict_identifier(value, name=f"sample_id[{index}]")
        for index, value in enumerate(sample_ids)
    ]
    for start, end in _generation_broadcast_ranges(
        motion_rubric_v2_criteria,
        len(values),
        num_generations=generation_count,
    ):
        if len(set(normalized_sample_ids[start:end])) != 1:
            raise RubricValidationError(
                "Motion V2 criteria generation broadcast requires one sample_id"
            )
    judgment_values = _column(
        motion_rubric_v2_judgment,
        len(values),
        name="motion_rubric_v2_judgment",
        num_generations=generation_count,
        allow_none=True,
        allow_scalar_generation_broadcast=False,
        allow_sequence_generation_expansion=False,
    )
    output: list[float] = []
    for index, (completion, criteria_value, rubric_id, sid, judgment_value) in enumerate(
        zip(
            values,
            criteria_values,
            normalized_rubric_ids,
            normalized_sample_ids,
            judgment_values,
        )
    ):
        if rubric_id != sid:
            raise RubricValidationError("Motion V2 rubric ID does not match sample_id")
        text = completion_text(completion)
        if text is None:
            output.append(0.0)
            continue
        criteria = validate_motion_criteria_v2(
            _object(criteria_value, name=f"motion_rubric_v2_criteria[{index}]")
        )
        if judgment_value is None:
            if judge_client is None:
                raise RubricValidationError(
                    "motion_rubric_v2_judgment or a configured online judge is required"
                )
            judgment = judge_client.judge_motion_v2(criteria, text, sample_id=sid)
        else:
            judgment = validate_motion_judgment_v2(
                _object(
                    judgment_value, name=f"motion_rubric_v2_judgment[{index}]"
                ),
                criteria,
                candidate_response=text,
                sample_id=sid,
                reject_unknown_ids=True,
            )
        output.append(
            compute_motion_reward_v2(
                criteria, judgment, candidate_response=text, sample_id=sid
            )["reward"]
        )
    return output


__all__ = ["motion_rubric_v2_rewards", "qa_rubric_rewards"]
