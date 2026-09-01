"""Compatibility facade for the strict QA MC rubric contract."""

from motionllm.grpo.qa_rubric import (
    QA_MODE,
    QA_RUBRIC_VERSION,
    assert_qa_dataset_binding,
    build_qa_judge_messages,
    parse_qa_judgment_text,
    validate_qa_criteria,
    validate_qa_judgment,
)
from motionllm.grpo.rubric_common import RubricValidationError

__all__ = [
    "QA_MODE",
    "QA_RUBRIC_VERSION",
    "RubricValidationError",
    "assert_qa_dataset_binding",
    "build_qa_judge_messages",
    "parse_qa_judgment_text",
    "validate_qa_criteria",
    "validate_qa_judgment",
]
