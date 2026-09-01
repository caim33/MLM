"""Pure deterministic sample-level rewards."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from motion_eval.evaluation import parse_strict_answer

from .schema import RewardMetadataError, normalize_gold_answer

_STRICT_REASONED_FORMAT = re.compile(
    r"<think>(?P<think>.*?)</think>\s*(?P<answer><answer>[A-D]</answer>)",
    flags=re.DOTALL,
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]")


def completion_text(value: Any) -> str | None:
    """Convert only known Swift completion shapes; unknown shapes fail closed."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], dict):
        content = value[0].get("content")
        return content if isinstance(content, str) else None
    return None


def answer_reward(completion: Any, gold_answer: Any) -> float:
    text = completion_text(completion)
    if text is None:
        return 0.0
    gold = normalize_gold_answer(gold_answer)
    parsed = parse_strict_answer(text)
    return 1.0 if parsed.is_valid and parsed.answer == gold else 0.0


def format_reward(completion: Any) -> float:
    text = completion_text(completion)
    if text is None:
        return 0.0
    stripped = text.strip()
    match = _STRICT_REASONED_FORMAT.fullmatch(stripped)
    if match is None:
        return 0.0
    reasoning = match.group("think")
    # Reject nested/repeated/malformed reasoning tags rather than letting a
    # permissive DOTALL match absorb them as ordinary text.
    if (
        not reasoning.strip()
        or "<think" in reasoning
        or "</think" in reasoning
        or not _reasoning_tokens(reasoning)
    ):
        return 0.0
    return 1.0 if parse_strict_answer(stripped).is_valid else 0.0


def _reasoning_tokens(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(_WORD.findall(normalized))


def _reasoning_text(text: str) -> str | None:
    match = _STRICT_REASONED_FORMAT.fullmatch(text.strip())
    if match is None:
        return None
    reasoning = match.group("think")
    return (
        None
        if (
            not reasoning.strip()
            or "<think" in reasoning
            or "</think" in reasoning
            or not _reasoning_tokens(reasoning)
        )
        else reasoning
    )


def validate_semantic_reference(reference: Any, gold_answer: Any) -> str:
    """Require one non-empty strict think block and the canonical gold answer."""

    text = completion_text(reference)
    if text is None or _reasoning_text(text) is None:
        raise RewardMetadataError(
            "semantic solution must be <think>non-empty reasoning</think><answer>[A-D]</answer>"
        )
    parsed = parse_strict_answer(text)
    gold = normalize_gold_answer(gold_answer)
    if not parsed.is_valid or parsed.answer != gold:
        raise RewardMetadataError("semantic solution answer must match gold_answer")
    return text


def semantic_reward(completion: Any, reference: Any) -> float:
    """Deterministic lexical semantic score, gated by strict MCQ correctness.

    Both generated and reference text must contain strict, lexically non-empty
    reasoning.  A set Dice score over normalized reasoning tokens is returned
    only after the strict answer gate passes.
    """

    generated = completion_text(completion)
    target = completion_text(reference)
    if generated is None or target is None:
        return 0.0
    target_parse = parse_strict_answer(target)
    generated_parse = parse_strict_answer(generated)
    if (
        not target_parse.is_valid
        or not generated_parse.is_valid
        or target_parse.answer != generated_parse.answer
    ):
        return 0.0
    target_reasoning = _reasoning_text(target)
    if target_reasoning is None:
        return 0.0
    generated_reasoning = _reasoning_text(generated)
    if generated_reasoning is None:
        return 0.0
    target_tokens = _reasoning_tokens(target_reasoning)
    generated_tokens = _reasoning_tokens(generated_reasoning)
    if not target_tokens or not generated_tokens:
        return 0.0
    return (2.0 * len(target_tokens & generated_tokens)) / (
        len(target_tokens) + len(generated_tokens)
    )
