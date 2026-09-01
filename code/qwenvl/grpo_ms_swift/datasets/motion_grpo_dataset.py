"""Legacy ms-swift dataset facade over strict reward metadata contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from motionllm.grpo import RewardMetadata

REQUIRED_REWARD_FIELDS = (
    "sample_id",
    "group_id",
    "branch",
    "rollout_id",
    "answer",
)


def build_reward_metadata(
    sample: Mapping[str, Any],
    sample_index: int,
    branch: Optional[str] = None,
    rollout_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and preserve explicit metadata; ``sample_index`` is never an ID fallback."""

    del sample_index
    candidate = dict(sample)
    if branch is not None:
        candidate["branch"] = branch
    if rollout_id is not None:
        candidate["rollout_id"] = rollout_id
    metadata = RewardMetadata.from_mapping(candidate)
    return {
        "sample_id": metadata.sample_id,
        "group_id": metadata.group_id,
        "branch": metadata.branch.value,
        "rollout_id": metadata.rollout_id,
        "answer": metadata.gold_answer,
        "solution": metadata.solution,
        "request_id": metadata.request_id,
    }


def collate_reward_metadata(
    batch: MutableMapping[str, Any],
    metadata_list: Sequence[Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Validate all rows before attaching aligned metadata columns."""

    rows = [RewardMetadata.from_mapping(item) for item in metadata_list]
    batch["sample_id"] = [item.sample_id for item in rows]
    batch["group_id"] = [item.group_id for item in rows]
    batch["branch"] = [item.branch.value for item in rows]
    batch["rollout_id"] = [item.rollout_id for item in rows]
    batch["answer"] = [item.gold_answer for item in rows]
    batch["solution"] = [item.solution for item in rows]
    batch["request_id"] = [item.request_id for item in rows]
    return batch
