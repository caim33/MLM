"""Compatibility facade for strict Motion Rubric V2.

Stage 1 remains in :mod:`rubric_rl.reward`; importing this module can only
produce or score V2 artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping

from motionllm.grpo.motion_rubric_v2 import (
    MOTION_MODE_V2,
    MOTION_RUBRIC_V2_VERSION,
    compute_motion_reward_v2,
    parse_motion_judgment_v2_text,
    validate_motion_criteria_v2,
    validate_motion_judgment_v2,
)
from motionllm.grpo.rubric_common import RubricValidationError


def ensure_criteria_ids(criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Assign missing stable IDs while enforcing the complete V2 schema."""

    return validate_motion_criteria_v2(criteria)


def compute_reward(
    criteria: Mapping[str, Any],
    judgment: Mapping[str, Any],
    candidate_response: Any,
    *,
    sample_id: str,
) -> dict[str, Any]:
    return compute_motion_reward_v2(
        criteria,
        judgment,
        candidate_response=candidate_response,
        sample_id=sample_id,
    )


__all__ = [
    "MOTION_MODE_V2",
    "MOTION_RUBRIC_V2_VERSION",
    "RubricValidationError",
    "compute_motion_reward_v2",
    "compute_reward",
    "ensure_criteria_ids",
    "parse_motion_judgment_v2_text",
    "validate_motion_criteria_v2",
    "validate_motion_judgment_v2",
]
