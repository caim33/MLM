"""Narrow, fail-closed adapters from ms-swift columns to pure rewards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .group import GroupBonusConfig, GroupScore, compute_group_bonus
from .colocation import record_runtime_colocation
from .rewards import answer_reward, format_reward, semantic_reward
from .schema import RewardMetadata, RewardMetadataError


def _num_generations(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        unique = set(value)
        if len(unique) != 1:
            raise RewardMetadataError("num_generations must be constant across the batch")
        value = next(iter(unique))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RewardMetadataError("num_generations must be a positive integer")
    return value


def _column(
    value: Any,
    size: int,
    name: str,
    *,
    num_generations: int | None,
) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        if value is not None and (size == 1 or num_generations == size):
            return [value] * size
        raise RewardMetadataError(f"{name} must contain exactly {size} values")
    result = list(value)
    if len(result) == size:
        return result
    if num_generations is not None and len(result) * num_generations == size:
        return [item for item in result for _ in range(num_generations)]
    raise RewardMetadataError(f"{name} length {len(result)} does not match completions {size}")


def build_reward_metadata_batch(
    size: int,
    *,
    sample_id: Any = None,
    group_id: Any = None,
    branch: Any = None,
    rollout_id: Any = None,
    answer: Any = None,
    gold_answer: Any = None,
    solution: Any = None,
    request_id: Any = None,
    num_generations: Any = None,
) -> tuple[RewardMetadata, ...]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RewardMetadataError("size must be a non-negative integer")
    if size == 0:
        return ()
    gold_source = answer if answer is not None else gold_answer
    if gold_source is None:
        raise RewardMetadataError("answer/gold_answer is required")
    generation_count = _num_generations(num_generations)
    columns = {
        "sample_id": _column(sample_id, size, "sample_id", num_generations=generation_count),
        "group_id": _column(group_id, size, "group_id", num_generations=generation_count),
        "branch": _column(branch, size, "branch", num_generations=generation_count),
        "rollout_id": _column(rollout_id, size, "rollout_id", num_generations=generation_count),
        "gold_answer": _column(gold_source, size, "answer", num_generations=generation_count),
        "solution": _column(solution, size, "solution", num_generations=generation_count) if solution is not None else [None] * size,
        "request_id": _column(request_id, size, "request_id", num_generations=generation_count) if request_id is not None else [None] * size,
    }
    return tuple(
        RewardMetadata.from_mapping(
            {name: values[index] for name, values in columns.items()},
            generation_id=index,
        )
        for index in range(size)
    )


def _metadata(completions: Sequence[Any], columns: Mapping[str, Any]) -> tuple[RewardMetadata, ...]:
    try:
        return build_reward_metadata_batch(len(completions), **columns)
    except TypeError as exc:
        raise RewardMetadataError(f"unsupported reward metadata columns: {sorted(columns)}") from exc


def option_accuracy_rewards(completions: Sequence[Any], **columns: Any) -> list[float]:
    values = list(completions)
    metadata = _metadata(values, columns)
    return [answer_reward(value, item.gold_answer) for value, item in zip(values, metadata)]


def format_rewards(completions: Sequence[Any], **columns: Any) -> list[float]:
    values = list(completions)
    _metadata(values, columns)
    return [format_reward(value) for value in values]


def semantic_rewards(completions: Sequence[Any], **columns: Any) -> list[float]:
    values = list(completions)
    metadata = _metadata(values, columns)
    result: list[float] = []
    for value, item in zip(values, metadata):
        reference = item.solution or f"<answer>{item.gold_answer}</answer>"
        result.append(semantic_reward(value, reference))
    return result


def vm_v_group_bonus_rewards(
    completions: Sequence[Any],
    *,
    config: GroupBonusConfig | None = None,
    **columns: Any,
) -> list[float]:
    values = list(completions)
    metadata = _metadata(values, columns)
    record_runtime_colocation(metadata)
    baseline = [answer_reward(value, item.gold_answer) for value, item in zip(values, metadata)]
    result = compute_group_bonus(
        [GroupScore(item, score) for item, score in zip(metadata, baseline)],
        config=config,
    )
    return list(result.bonuses)
