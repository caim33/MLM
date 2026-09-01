"""Strict GRPO metadata contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from motion_eval.evaluation import parse_strict_answer

_IDENTIFIER = re.compile(r"^[^\x00\r\n]{1,256}$")
_CHOICES = frozenset("ABCD")


class RewardMetadataError(ValueError):
    pass


class RewardBranch(str, Enum):
    V = "v"
    M = "m"
    VM = "vm"
    T = "t"


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None or not value.strip():
        raise RewardMetadataError(f"{name} must be a non-empty string without control newlines")
    return value


def normalize_gold_answer(value: Any) -> str:
    """Accept only canonical dataset gold forms, never loose model-output text."""

    if not isinstance(value, str):
        raise RewardMetadataError("gold answer must be a string")
    if value in _CHOICES:
        return value
    parsed = parse_strict_answer(value)
    if (
        not parsed.is_valid
        or parsed.answer is None
        or value != f"<answer>{parsed.answer}</answer>"
    ):
        raise RewardMetadataError("gold answer must be A-D or one strict <answer>[A-D]</answer> tag")
    return parsed.answer


@dataclass(frozen=True)
class RewardMetadata:
    sample_id: str
    group_id: str
    branch: RewardBranch
    rollout_id: int
    gold_answer: str
    generation_id: int | None = None
    solution: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        _identifier("sample_id", self.sample_id)
        _identifier("group_id", self.group_id)
        if isinstance(self.branch, str) and not isinstance(self.branch, RewardBranch):
            try:
                object.__setattr__(self, "branch", RewardBranch(self.branch))
            except ValueError as exc:
                raise RewardMetadataError(f"unsupported branch: {self.branch!r}") from exc
        if not isinstance(self.branch, RewardBranch):
            raise RewardMetadataError("branch must be an explicit RewardBranch")
        if isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) or self.rollout_id < 0:
            raise RewardMetadataError("rollout_id must be a non-negative integer")
        object.__setattr__(self, "gold_answer", normalize_gold_answer(self.gold_answer))
        if self.generation_id is not None and (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 0
        ):
            raise RewardMetadataError("generation_id must be a non-negative integer")
        if self.solution is not None:
            if not isinstance(self.solution, str):
                raise RewardMetadataError("solution must be a string when present")
            parsed_solution = parse_strict_answer(self.solution)
            if not parsed_solution.is_valid or parsed_solution.answer != self.gold_answer:
                raise RewardMetadataError(
                    "solution must contain exactly one strict answer matching gold_answer"
                )
        if self.request_id is not None:
            _identifier("request_id", self.request_id)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        generation_id: int | None = None,
    ) -> "RewardMetadata":
        if not isinstance(value, Mapping):
            raise RewardMetadataError("reward metadata must be a mapping")
        required = ("sample_id", "group_id", "branch", "rollout_id")
        missing = [name for name in required if name not in value]
        if missing:
            raise RewardMetadataError(f"reward metadata is missing fields: {missing}")
        gold = value.get("answer")
        if gold is None:
            gold = value.get("gold_answer")
        if gold is None:
            raise RewardMetadataError("reward metadata is missing answer/gold_answer")
        branch_raw = value["branch"]
        if not isinstance(branch_raw, str):
            raise RewardMetadataError("branch must be a string")
        try:
            branch = RewardBranch(branch_raw.strip().lower())
        except ValueError as exc:
            raise RewardMetadataError(f"unsupported branch: {branch_raw!r}") from exc
        return cls(
            sample_id=_identifier("sample_id", value["sample_id"]),
            group_id=_identifier("group_id", value["group_id"]),
            branch=branch,
            rollout_id=value["rollout_id"],
            gold_answer=gold,
            generation_id=(
                generation_id
                if generation_id is not None
                else value.get("generation_id")
            ),
            solution=value.get("solution"),
            request_id=value.get("request_id"),
        )

    @property
    def rollout_key(self) -> tuple[str, str, str, int, int]:
        if self.generation_id is None:
            raise RewardMetadataError(
                "generation_id is required before grouping generated completions"
            )
        return (
            self.group_id,
            self.sample_id,
            self.branch.value,
            self.rollout_id,
            self.generation_id,
        )


def finite_reward(value: Any, *, name: str = "reward") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardMetadataError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RewardMetadataError(f"{name} must be finite")
    return result
