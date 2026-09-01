"""Legacy facade for the strict, order-independent VM/V group reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from motionllm.grpo.group import (
    GroupBonusConfig as CoreGroupBonusConfig,
    GroupScore,
    compute_group_bonus,
)
from motionllm.grpo.schema import RewardMetadata, finite_reward


@dataclass(frozen=True)
class RolloutScoreRecord:
    index: int
    group_id: str
    branch: str
    baseline_score: float
    sample_id: str | None = None
    rollout_id: int | None = None
    gold_answer: str | None = None


@dataclass(frozen=True)
class GroupBonusConfig:
    threshold: float = 1.0
    bonus_value: float = 0.1
    qualify_threshold: float = 0.1
    epsilon: float = 1e-8

    def to_core(self) -> CoreGroupBonusConfig:
        # epsilon belonged to a legacy max(v_mean, epsilon) rule that changed
        # reward semantics for all-zero groups.  It is retained in the facade
        # signature only; the canonical core uses the explicit mathematical gate.
        finite_reward(self.epsilon, name="epsilon")
        return CoreGroupBonusConfig(
            threshold=self.threshold,
            bonus_value=self.bonus_value,
            qualify_threshold=self.qualify_threshold,
        )


@dataclass
class GroupBonusResult:
    bonuses: List[float]
    group_gate: Dict[str, bool]
    vm_mean: Dict[str, float]
    v_mean: Dict[str, float]


def _core_records(records: Sequence[RolloutScoreRecord]) -> list[GroupScore]:
    indices = [record.index for record in records]
    if sorted(indices) != list(range(len(records))):
        raise ValueError("legacy record indices must be a unique 0..N-1 alignment")
    result: list[GroupScore] = []
    for record in records:
        if record.sample_id is None or record.rollout_id is None or record.gold_answer is None:
            raise ValueError(
                "sample_id, rollout_id and gold_answer are required; legacy metadata defaults were removed"
            )
        metadata = RewardMetadata(
            sample_id=record.sample_id,
            group_id=record.group_id,
            branch=record.branch,
            rollout_id=record.rollout_id,
            gold_answer=record.gold_answer,
            generation_id=record.index,
        )
        result.append(GroupScore(metadata, record.baseline_score))
    return result


def compute_vm_v_group_bonus(
    records: Sequence[RolloutScoreRecord],
    config: GroupBonusConfig | None = None,
) -> GroupBonusResult:
    core_records = _core_records(records)
    core = compute_group_bonus(core_records, (config or GroupBonusConfig()).to_core())
    aligned = [0.0] * len(records)
    for record, bonus in zip(records, core.bonuses):
        aligned[record.index] = bonus
    return GroupBonusResult(
        bonuses=aligned,
        group_gate=dict(core.group_gate),
        vm_mean=dict(core.vm_mean),
        v_mean=dict(core.v_mean),
    )


def apply_vm_v_bonus_to_rewards(
    rewards: Sequence[float],
    records: Sequence[RolloutScoreRecord],
    config: GroupBonusConfig | None = None,
) -> GroupBonusResult:
    if len(rewards) != len(records):
        raise ValueError(f"rewards/records length mismatch: {len(rewards)} vs {len(records)}")
    result = compute_vm_v_group_bonus(records, config)
    result.bonuses = [
        finite_reward(reward, name="reward") + bonus
        for reward, bonus in zip(rewards, result.bonuses)
    ]
    return result
