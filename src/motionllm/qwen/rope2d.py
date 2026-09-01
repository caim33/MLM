"""Verified Qwen3-VL position IDs used by the compatibility data adapter.

Qwen2-VL and Qwen2.5-VL used materially different RoPE rules in the old
checkout.  They are rejected here until they receive their own tested adapter;
silently selecting a position algorithm from a model path is not supported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class QwenRopeError(ValueError):
    """Inputs cannot be mapped to a safe Qwen multimodal position sequence."""


def _exact_token_id(tokenizer: Any, token: str) -> int:
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise QwenRopeError("tokenizer must expose get_vocab()") from exc
    if not isinstance(vocab, Mapping) or token not in vocab:
        raise QwenRopeError(f"tokenizer is missing required Qwen token {token!r}")
    value = vocab[token]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QwenRopeError(f"tokenizer returned an invalid ID for {token!r}")
    return value


def qwen3_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    """Return exact image-pad, video-pad and vision-start IDs."""

    return (
        _exact_token_id(tokenizer, "<|image_pad|>"),
        _exact_token_id(tokenizer, "<|video_pad|>"),
        _exact_token_id(tokenizer, "<|vision_start|>"),
    )


def _text_positions(
    input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        delta = torch.zeros(
            (input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device
        )
        return position_ids, delta
    mask = attention_mask.to(device=input_ids.device)
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    maximum = position_ids.max(0).values.max(-1, keepdim=True).values
    return position_ids, maximum + 1 - mask.shape[-1]


def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    *,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen3-VL 3-axis positions from explicit token IDs and grids."""

    del second_per_grid_ts  # Qwen3 encodes video time in timestamp text tokens.
    if input_ids is None or not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise QwenRopeError("input_ids must be a 2-D tensor")
    if isinstance(spatial_merge_size, bool) or not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
        raise QwenRopeError("spatial_merge_size must be a positive integer")
    if image_grid_thw is None and video_grid_thw is None:
        return _text_positions(input_ids, attention_mask)

    if video_grid_thw is not None:
        if not torch.is_tensor(video_grid_thw) or video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            raise QwenRopeError("video_grid_thw must have shape (N, 3)")
        if bool((video_grid_thw <= 0).any()):
            raise QwenRopeError("video_grid_thw values must be positive")
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        ).clone()
        video_grid_thw[:, 0] = 1
    if image_grid_thw is not None:
        if not torch.is_tensor(image_grid_thw) or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise QwenRopeError("image_grid_thw must have shape (N, 3)")
        if bool((image_grid_thw <= 0).any()):
            raise QwenRopeError("image_grid_thw values must be positive")

    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    if not torch.is_tensor(mask) or mask.shape != input_ids.shape:
        raise QwenRopeError("attention_mask must match input_ids")
    mask = mask.to(device=input_ids.device)
    position_ids = torch.ones(
        (3, input_ids.shape[0], input_ids.shape[1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index = 0
    video_index = 0
    deltas: list[torch.Tensor] = []
    for batch_index, row in enumerate(input_ids):
        row = row[mask[batch_index] == 1]
        starts = torch.argwhere(row == vision_start_token_id).flatten()
        if bool((starts + 1 >= row.numel()).any()):
            raise QwenRopeError("vision_start token is truncated at sequence end")
        vision_tokens = row[starts + 1]
        image_count = int((vision_tokens == image_token_id).sum().item())
        video_count = int((vision_tokens == video_token_id).sum().item())
        tokens = row.tolist()
        segments: list[torch.Tensor] = []
        cursor = 0
        remaining_images = image_count
        remaining_videos = video_count
        for _ in range(image_count + video_count):
            next_image = (
                tokens.index(image_token_id, cursor)
                if remaining_images and image_token_id in tokens[cursor:]
                else len(tokens) + 1
            )
            next_video = (
                tokens.index(video_token_id, cursor)
                if remaining_videos and video_token_id in tokens[cursor:]
                else len(tokens) + 1
            )
            if next_image < next_video:
                if image_grid_thw is None or image_index >= len(image_grid_thw):
                    raise QwenRopeError("image token/grid count mismatch")
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remaining_images -= 1
                end = next_image
            else:
                if video_grid_thw is None or video_index >= len(video_grid_thw):
                    raise QwenRopeError("video token/grid count mismatch")
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remaining_videos -= 1
                end = next_video
            grid_t = int(t.item())
            grid_h = int(h.item()) // spatial_merge_size
            grid_w = int(w.item()) // spatial_merge_size
            if grid_h <= 0 or grid_w <= 0:
                raise QwenRopeError("vision grid is smaller than spatial_merge_size")
            text_length = end - cursor
            offset = int(segments[-1].max().item()) + 1 if segments else 0
            segments.append(
                torch.arange(text_length).view(1, -1).expand(3, -1) + offset
            )
            t_index = (
                torch.arange(grid_t)
                .view(-1, 1)
                .expand(-1, grid_h * grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(grid_h)
                .view(1, -1, 1)
                .expand(grid_t, -1, grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(grid_w)
                .view(1, 1, -1)
                .expand(grid_t, grid_h, -1)
                .flatten()
            )
            segments.append(
                torch.stack((t_index, h_index, w_index)) + text_length + offset
            )
            cursor = end + grid_t * grid_h * grid_w
        if cursor < len(tokens):
            offset = int(segments[-1].max().item()) + 1 if segments else 0
            text_length = len(tokens) - cursor
            segments.append(
                torch.arange(text_length).view(1, -1).expand(3, -1) + offset
            )
        if not segments:
            raise QwenRopeError("visual grids were supplied but no vision tokens were found")
        row_positions = torch.cat(segments, dim=1).reshape(3, -1)
        if row_positions.shape[1] != int(mask[batch_index].sum().item()):
            raise QwenRopeError("vision token expansion does not match sequence length")
        position_ids[:, batch_index, mask[batch_index] == 1] = row_positions.to(
            device=position_ids.device, dtype=position_ids.dtype
        )
        deltas.append(row_positions.max() + 1 - input_ids.shape[1])
    if image_grid_thw is not None and image_index != len(image_grid_thw):
        raise QwenRopeError("unused image grids remain after position construction")
    if video_grid_thw is not None and video_index != len(video_grid_thw):
        raise QwenRopeError("unused video grids remain after position construction")
    return position_ids, torch.stack(deltas).to(input_ids.device).unsqueeze(1)


def build_position_ids(
    processor: Any,
    data: Mapping[str, Any],
    *,
    model_type: str,
) -> torch.Tensor:
    """Build positions for the explicitly selected Qwen family."""

    input_ids = data.get("input_ids")
    attention_mask = data.get("attention_mask")
    image_grid = data.get("image_grid_thw")
    video_grid = data.get("video_grid_thw")
    if model_type != "qwen3vl":
        if image_grid is not None or video_grid is not None:
            raise QwenRopeError(
                f"visual RoPE adapter for model_type={model_type!r} is not verified"
            )
        return _text_positions(input_ids, attention_mask)[0]
    image_id, video_id, vision_start_id = qwen3_token_ids(processor.tokenizer)
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    return get_rope_index_3(
        spatial_merge_size=int(merge_size),
        input_ids=input_ids,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        attention_mask=attention_mask,
        image_token_id=image_id,
        video_token_id=video_id,
        vision_start_token_id=vision_start_id,
    )[0]


def get_rope_index_25(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    raise QwenRopeError("Qwen2.5-VL RoPE compatibility is not yet verified")


def get_rope_index_2(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    raise QwenRopeError("Qwen2-VL RoPE compatibility is not yet verified")


__all__ = [
    "QwenRopeError",
    "build_position_ids",
    "get_rope_index_2",
    "get_rope_index_3",
    "get_rope_index_25",
    "qwen3_token_ids",
]
