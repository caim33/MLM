"""Compatibility facade for deterministic QA MC Rubric-RL reward."""

from motionllm.grpo.qa_rubric import (
    QACompletion,
    compute_qa_rubric_reward,
    parse_qa_completion,
)

compute_reward = compute_qa_rubric_reward

__all__ = [
    "QACompletion",
    "compute_qa_rubric_reward",
    "compute_reward",
    "parse_qa_completion",
]
