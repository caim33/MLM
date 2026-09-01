"""Idempotent motion-token setup shared by full SFT and LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MOTION_START_TOKEN = "<motion_start>"
MOTION_PLACEHOLDER_TOKEN = "<motion>"
MOTION_END_TOKEN = "<motion_end>"
MOTION_BOUNDARY_TOKENS = (MOTION_START_TOKEN, MOTION_END_TOKEN)


class MotionTokenError(ValueError):
    pass


@dataclass(frozen=True)
class MotionTokenReceipt:
    vocabulary_size_before: int
    vocabulary_size_after: int
    added_count: int
    resized: bool
    motion_start_token_id: int
    motion_end_token_id: int

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "vocabulary_size_before": self.vocabulary_size_before,
            "vocabulary_size_after": self.vocabulary_size_after,
            "added_count": self.added_count,
            "resized": self.resized,
            "motion_start_token_id": self.motion_start_token_id,
            "motion_end_token_id": self.motion_end_token_id,
        }


def _tokenizer_size(tokenizer: Any) -> int:
    try:
        size = len(tokenizer)
    except Exception as exc:
        raise MotionTokenError("tokenizer must implement len()") from exc
    if not isinstance(size, int) or size <= 0:
        raise MotionTokenError("tokenizer vocabulary size must be positive")
    return size


def _exact_token_id(tokenizer: Any, token: str) -> int:
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise MotionTokenError("tokenizer must expose get_vocab()") from exc
    if not isinstance(vocab, dict) or token not in vocab:
        raise MotionTokenError(f"token was not registered exactly: {token}")
    token_id = vocab[token]
    if not isinstance(token_id, int) or token_id < 0:
        raise MotionTokenError(f"invalid token id for {token}: {token_id!r}")
    try:
        recovered = tokenizer.convert_ids_to_tokens(token_id)
    except Exception as exc:
        raise MotionTokenError("tokenizer cannot verify registered token ids") from exc
    if recovered != token:
        raise MotionTokenError(
            f"token id round-trip mismatch for {token}: recovered {recovered!r}"
        )
    return token_id


def _embedding_size(model: Any) -> int | None:
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        return None
    embedding = getter()
    if embedding is None:
        return None
    size = getattr(embedding, "num_embeddings", None)
    return int(size) if isinstance(size, int) else None


def verify_motion_tokens(tokenizer: Any, model: Any) -> tuple[int, int]:
    """Verify exact boundary tokens, embedding size and model config bindings."""

    start_id, end_id = verify_motion_tokenizer_tokens(tokenizer)
    vocab_size = _tokenizer_size(tokenizer)
    embedding_size = _embedding_size(model)
    if embedding_size is not None and embedding_size != vocab_size:
        raise MotionTokenError(
            f"model/tokenizer vocabulary mismatch: model={embedding_size}, tokenizer={vocab_size}"
        )
    config = getattr(model, "config", None)
    if config is None:
        raise MotionTokenError("model must expose config for motion token ids")
    for name, expected in (
        ("motion_start_token_id", start_id),
        ("motion_end_token_id", end_id),
    ):
        actual = getattr(config, name, None)
        if actual != expected:
            raise MotionTokenError(f"model.config.{name}={actual!r}, expected {expected}")
    return start_id, end_id


def verify_motion_tokenizer_tokens(tokenizer: Any) -> tuple[int, int]:
    """Read-only proof that both saved boundary tokens exist with exact IDs."""

    vocab_size = _tokenizer_size(tokenizer)
    start_id = _exact_token_id(tokenizer, MOTION_START_TOKEN)
    end_id = _exact_token_id(tokenizer, MOTION_END_TOKEN)
    if start_id == end_id:
        raise MotionTokenError("motion boundary token ids must be distinct")
    if start_id >= vocab_size or end_id >= vocab_size:
        raise MotionTokenError(
            "motion boundary token id lies outside the tokenizer vocabulary size"
        )
    return start_id, end_id


def bind_model_to_motion_tokens(
    tokenizer: Any,
    model: Any,
    *,
    expected_token_ids: tuple[int, int] | None = None,
) -> MotionTokenReceipt:
    """Bind/resize only the model after a tokenizer has passed read-only verification."""

    before = _tokenizer_size(tokenizer)
    token_ids = verify_motion_tokenizer_tokens(tokenizer)
    if expected_token_ids is not None and token_ids != expected_token_ids:
        raise MotionTokenError(
            f"reloaded motion token ids changed: {expected_token_ids} -> {token_ids}"
        )
    config = getattr(model, "config", None)
    if config is None:
        raise MotionTokenError("model must expose config for motion token ids")
    setattr(config, "motion_start_token_id", token_ids[0])
    setattr(config, "motion_end_token_id", token_ids[1])

    resized = False
    embedding_size = _embedding_size(model)
    if embedding_size != before:
        resize = getattr(model, "resize_token_embeddings", None)
        if not callable(resize):
            raise MotionTokenError(
                "model embedding size does not match tokenizer and resize_token_embeddings is unavailable"
            )
        resize(before)
        resized = True
    verify_motion_tokens(tokenizer, model)
    return MotionTokenReceipt(
        before,
        before,
        0,
        resized,
        token_ids[0],
        token_ids[1],
    )


def setup_motion_tokens(tokenizer: Any, model: Any) -> MotionTokenReceipt:
    """Register boundary tokens once and resize embeddings exactly when needed."""

    before = _tokenizer_size(tokenizer)
    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    if not callable(add_special_tokens):
        raise MotionTokenError("tokenizer must expose add_special_tokens()")
    payload = {"additional_special_tokens": list(MOTION_BOUNDARY_TOKENS)}
    try:
        added = add_special_tokens(payload, replace_additional_special_tokens=False)
    except TypeError:
        existing = list(getattr(tokenizer, "additional_special_tokens", ()) or ())
        combined = existing + [token for token in MOTION_BOUNDARY_TOKENS if token not in existing]
        added = add_special_tokens({"additional_special_tokens": combined})
    if not isinstance(added, int) or added < 0:
        raise MotionTokenError(f"invalid add_special_tokens result: {added!r}")
    after = _tokenizer_size(tokenizer)
    if after - before != added:
        raise MotionTokenError(
            f"tokenizer size delta ({after - before}) differs from reported additions ({added})"
        )

    binding = bind_model_to_motion_tokens(tokenizer, model)
    return MotionTokenReceipt(
        before,
        after,
        added,
        binding.resized,
        binding.motion_start_token_id,
        binding.motion_end_token_id,
    )
