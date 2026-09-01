"""Deterministic GRPO domain logic with optional framework adapters."""

from .group import GroupBonusConfig, GroupBonusResult, GroupScore, compute_group_bonus
from .colocation import (
    COLOCATION_CONFIG_ENV,
    COLOCATION_DATASET_ENV,
    COLOCATION_ENV_KEYS,
    COLOCATION_NONCE_ENV,
    COLOCATION_PATH_ENV,
    COLOCATION_PLAN_ENV,
    initialize_runtime_colocation_plan,
    record_runtime_colocation,
    validate_reward_call_colocation,
    validate_runtime_colocation_receipt,
)
from .redaction import (
    assert_manifest_secret_free,
    describe_environment_overrides,
    redact_command_for_log,
    redact_mapping_for_log,
)
from .rewards import answer_reward, format_reward, semantic_reward, validate_semantic_reference
from .schema import RewardBranch, RewardMetadata, RewardMetadataError, normalize_gold_answer
from .swift_adapter import (
    build_reward_metadata_batch,
    format_rewards,
    option_accuracy_rewards,
    semantic_rewards,
    vm_v_group_bonus_rewards,
)
from .qa_rubric import (
    QA_RUBRIC_VERSION,
    build_qa_judge_messages,
    compute_qa_rubric_reward,
    parse_qa_completion,
    parse_qa_judgment_text,
    validate_qa_criteria,
    validate_qa_judgment,
)
from .motion_rubric_v2 import (
    MOTION_RUBRIC_V2_VERSION,
    compute_motion_reward_v2,
    parse_motion_judgment_v2_text,
    validate_motion_criteria_v2,
    validate_motion_judgment_v2,
)
from .rubric_adapter import motion_rubric_v2_rewards, qa_rubric_rewards
from .rubric_common import RubricValidationError
from .rubric_online import OnlineJudgeConfig, OnlineRubricJudge

__all__ = [
    "GroupBonusConfig",
    "GroupBonusResult",
    "GroupScore",
    "COLOCATION_CONFIG_ENV",
    "COLOCATION_DATASET_ENV",
    "COLOCATION_ENV_KEYS",
    "COLOCATION_NONCE_ENV",
    "COLOCATION_PATH_ENV",
    "COLOCATION_PLAN_ENV",
    "RewardBranch",
    "RewardMetadata",
    "RewardMetadataError",
    "RubricValidationError",
    "QA_RUBRIC_VERSION",
    "MOTION_RUBRIC_V2_VERSION",
    "answer_reward",
    "assert_manifest_secret_free",
    "build_reward_metadata_batch",
    "build_qa_judge_messages",
    "compute_group_bonus",
    "compute_qa_rubric_reward",
    "compute_motion_reward_v2",
    "describe_environment_overrides",
    "format_reward",
    "format_rewards",
    "initialize_runtime_colocation_plan",
    "normalize_gold_answer",
    "motion_rubric_v2_rewards",
    "OnlineJudgeConfig",
    "OnlineRubricJudge",
    "option_accuracy_rewards",
    "parse_qa_completion",
    "parse_qa_judgment_text",
    "parse_motion_judgment_v2_text",
    "qa_rubric_rewards",
    "redact_command_for_log",
    "redact_mapping_for_log",
    "record_runtime_colocation",
    "semantic_reward",
    "semantic_rewards",
    "validate_semantic_reference",
    "validate_qa_criteria",
    "validate_qa_judgment",
    "validate_motion_criteria_v2",
    "validate_motion_judgment_v2",
    "validate_reward_call_colocation",
    "validate_runtime_colocation_receipt",
    "vm_v_group_bonus_rewards",
]
