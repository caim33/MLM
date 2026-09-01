"""Legacy reward names backed by deterministic, strict GRPO core functions."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Protocol, Sequence

from motionllm.grpo import format_rewards, option_accuracy_rewards, semantic_rewards

_THINK = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
_WORDS = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]")


class SemanticBackend(Protocol):
    def score_pairs(self, generated: Sequence[str], targets: Sequence[str]) -> List[float]: ...


class ClipTextSemanticBackend:
    """Compatibility name for a deterministic text backend.

    The historical lazy CLIP load made rewards environment- and failure-order
    dependent.  Canonical GRPO now uses the pure reward in ``motionllm.grpo``.
    """

    def score_pairs(self, generated: Sequence[str], targets: Sequence[str]) -> List[float]:
        if len(generated) != len(targets):
            raise ValueError("generated/targets length mismatch")
        return [1.0 if left == right and left != "" else 0.0 for left, right in zip(generated, targets)]


_DEFAULT_BACKEND: SemanticBackend = ClipTextSemanticBackend()


def set_default_semantic_backend(backend: SemanticBackend) -> None:
    # Retained for import compatibility. The canonical plugin deliberately does
    # not call a mutable global backend.
    global _DEFAULT_BACKEND
    if not callable(getattr(backend, "score_pairs", None)):
        raise TypeError("backend must expose score_pairs")
    _DEFAULT_BACKEND = backend


def _columns(answer: Any, solution: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": kwargs.get("sample_id"),
        "group_id": kwargs.get("group_id"),
        "branch": kwargs.get("branch"),
        "rollout_id": kwargs.get("rollout_id"),
        "request_id": kwargs.get("request_id"),
        "answer": answer if answer is not None else kwargs.get("gold_answer"),
        "solution": solution,
        "num_generations": kwargs.get("num_generations"),
    }


def semantic_reward_plugin(
    completions: List[Any],
    answer: Optional[Any] = None,
    solution: Optional[Any] = None,
    backend: Optional[SemanticBackend] = None,
    **kwargs: Any,
) -> List[float]:
    del backend
    return semantic_rewards(list(completions), **_columns(answer, solution, kwargs))


def option_accuracy_reward_plugin(
    completions: List[Any],
    answer: Optional[Any] = None,
    solution: Optional[Any] = None,
    **kwargs: Any,
) -> List[float]:
    return option_accuracy_rewards(list(completions), **_columns(answer, solution, kwargs))


def format_reward_plugin(completions: List[Any], **kwargs: Any) -> List[float]:
    return format_rewards(
        list(completions),
        **_columns(kwargs.get("answer"), kwargs.get("solution"), kwargs),
    )


def length_control_reward_plugin(
    completions: List[Any],
    answer: Optional[Any] = None,
    solution: Optional[Any] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    **kwargs: Any,
) -> List[float]:
    minimum = 70 if min_words is None else max(0, int(min_words))
    maximum = 200 if max_words is None else int(max_words)
    if maximum < minimum:
        maximum = minimum
    correct = option_accuracy_rewards(list(completions), **_columns(answer, solution, kwargs))
    formatted = format_rewards(
        list(completions),
        **_columns(answer, solution, kwargs),
    )
    rewards: list[float] = []
    for completion, is_correct, is_formatted in zip(completions, correct, formatted):
        text = completion if isinstance(completion, str) else ""
        match = _THINK.fullmatch(text.split("<answer>", 1)[0])
        count = len(_WORDS.findall(match.group(1))) if match else -1
        rewards.append(
            1.0
            if is_correct == 1.0 and is_formatted == 1.0 and minimum <= count <= maximum
            else 0.0
        )
    return rewards
