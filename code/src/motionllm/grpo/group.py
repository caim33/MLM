"""Order-independent VM-versus-V group reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import RewardBranch, RewardMetadata, RewardMetadataError, finite_reward


@dataclass(frozen=True)
class GroupScore:
    metadata: RewardMetadata
    baseline_score: float

    def __post_init__(self) -> None:
        score = finite_reward(self.baseline_score, name="baseline_score")
        if not 0.0 <= score <= 1.0:
            raise RewardMetadataError("baseline_score must be in [0, 1]")
        object.__setattr__(self, "baseline_score", score)


@dataclass(frozen=True)
class GroupBonusConfig:
    threshold: float = 1.0
    bonus_value: float = 0.1
    qualify_threshold: float = 0.1

    def __post_init__(self) -> None:
        for name in ("threshold", "bonus_value", "qualify_threshold"):
            value = finite_reward(getattr(self, name), name=name)
            if value < 0:
                raise RewardMetadataError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GroupBonusResult:
    bonuses: tuple[float, ...]
    group_gate: Mapping[str, bool]
    vm_mean: Mapping[str, float]
    v_mean: Mapping[str, float]


def compute_group_bonus(
    records: Sequence[GroupScore],
    config: GroupBonusConfig | None = None,
) -> GroupBonusResult:
    """Compute stable group means while preserving caller output alignment."""

    config = config or GroupBonusConfig()
    seen: set[tuple[str, str, str, int, int]] = set()
    grouped: dict[str, dict[RewardBranch, list[float]]] = {}
    for record in records:
        if not isinstance(record, GroupScore):
            raise RewardMetadataError("group bonus records must be GroupScore instances")
        key = record.metadata.rollout_key
        if key in seen:
            raise RewardMetadataError(f"duplicate rollout identity: {key}")
        seen.add(key)
        if record.metadata.branch not in {RewardBranch.V, RewardBranch.VM}:
            raise RewardMetadataError("VM/V group bonus accepts only v and vm branches")
        branches = grouped.setdefault(record.metadata.group_id, {})
        branches.setdefault(record.metadata.branch, []).append(record.baseline_score)

    vm_mean: dict[str, float] = {}
    v_mean: dict[str, float] = {}
    gate: dict[str, bool] = {}
    for group_id in sorted(grouped):
        branches = grouped[group_id]
        vm_values = branches.get(RewardBranch.VM, [])
        v_values = branches.get(RewardBranch.V, [])
        vm = math.fsum(sorted(vm_values)) / len(vm_values) if vm_values else 0.0
        v = math.fsum(sorted(v_values)) / len(v_values) if v_values else 0.0
        vm_mean[group_id] = vm
        v_mean[group_id] = v
        gate[group_id] = bool(vm_values and v_values and vm >= config.threshold * v)

    bonuses: list[float] = []
    for record in records:
        qualifies = (
            record.metadata.branch is RewardBranch.VM
            and gate[record.metadata.group_id]
            and record.baseline_score >= config.qualify_threshold
        )
        bonuses.append(config.bonus_value if qualifies else 0.0)
    return GroupBonusResult(tuple(bonuses), gate, vm_mean, v_mean)
