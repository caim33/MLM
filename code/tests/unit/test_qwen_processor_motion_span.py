from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from motionllm.qwen.processor import (
    QwenDataAdapterError,
    _derive_assistant_mask_from_im_spans,
    _replace_motion_anchor_tokens,
)


class ContextMergingTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<motion_start>": 10, "<motion_end>": 11}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "<motion>":
            return [90, 91, 92]
        if text == "<motion><motion>":
            # Deliberately not two copies of the isolated marker encoding.
            return [20, 21, 22]
        raise AssertionError(text)


class TurnTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<|im_start|>": 100, "<|im_end|>": 101}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "assistant\n":
            return [200, 201]
        if text == "\n":
            return [201]
        raise AssertionError(text)


def _encoded_result(interior: list[int]) -> dict[str, torch.Tensor]:
    ids = torch.tensor([[1, 10, *interior, 11, 2, 3]], dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "assistant_masks": torch.tensor(
            [[False, False, *([False] * len(interior)), False, True, True]]
        ),
    }


def test_replaces_context_merged_boundary_interior_atomically() -> None:
    result = _encoded_result([20, 21, 22])
    processor = SimpleNamespace(tokenizer=ContextMergingTokenizer())
    data_args = SimpleNamespace(motion_placeholder_token_id=160001)

    _replace_motion_anchor_tokens(
        result,
        processor=processor,
        data_args=data_args,
        expected_count=2,
    )

    assert result["input_ids"].tolist() == [[1, 10, 160001, 160001, 11, 2, 3]]
    assert result["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1]]
    assert result["assistant_masks"].tolist() == [
        [False, False, False, False, False, True, True]
    ]


def test_rejects_boundary_interior_that_was_not_generated_by_clean_expansion() -> None:
    result = _encoded_result([20, 99, 22])
    processor = SimpleNamespace(tokenizer=ContextMergingTokenizer())
    data_args = SimpleNamespace(motion_placeholder_token_id=160001)

    with pytest.raises(QwenDataAdapterError, match="exact expanded anchor text"):
        _replace_motion_anchor_tokens(
            result,
            processor=processor,
            data_args=data_args,
            expected_count=2,
        )


def test_derives_visual_assistant_mask_from_exact_turn_span() -> None:
    input_ids = torch.tensor(
        [[100, 300, 201, 400, 101, 201, 100, 200, 201, 500, 501, 101, 201]]
    )
    mask = _derive_assistant_mask_from_im_spans(
        input_ids,
        tokenizer=TurnTokenizer(),
        expected_assistant_turns=1,
    )
    assert mask.tolist() == [
        [False, False, False, False, False, False, False, False, False,
         True, True, True, True]
    ]


def test_assistant_mask_fallback_rejects_injected_extra_turn() -> None:
    input_ids = torch.tensor(
        [[100, 200, 201, 500, 101, 201, 100, 200, 201, 501, 101, 201]]
    )
    with pytest.raises(QwenDataAdapterError, match="turn count changed"):
        _derive_assistant_mask_from_im_spans(
            input_ids,
            tokenizer=TurnTokenizer(),
            expected_assistant_turns=1,
        )
