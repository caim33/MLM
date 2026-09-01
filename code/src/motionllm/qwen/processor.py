"""Qwen message and tensor preprocessing with explicit motion contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from motionllm.data import build_legacy_messages
from motionllm.fusion import find_motion_anchors, replace_motion_anchors


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
MOTION_ANCHOR_TOKEN = "<motion>"


class QwenDataAdapterError(ValueError):
    """The Qwen compatibility input cannot be represented safely."""


def update_processor_pixels(processor: Any, data_args: Any) -> Any:
    """Apply only explicitly present image/video limits to a processor."""

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise QwenDataAdapterError("processor must expose image_processor")
    for target_name, argument_name in (
        ("min_pixels", "min_pixels"),
        ("max_pixels", "max_pixels"),
    ):
        value = getattr(data_args, argument_name, None)
        if value is not None and hasattr(image_processor, target_name):
            setattr(image_processor, target_name, value)
    image_size = getattr(image_processor, "size", None)
    if isinstance(image_size, dict):
        if getattr(data_args, "min_pixels", None) is not None:
            image_size["shortest_edge"] = data_args.min_pixels
        if getattr(data_args, "max_pixels", None) is not None:
            image_size["longest_edge"] = data_args.max_pixels

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        for target_name, argument_name in (
            ("min_pixels", "video_min_pixels"),
            ("max_pixels", "video_max_pixels"),
            ("min_frames", "video_min_frames"),
            ("max_frames", "video_max_frames"),
            ("fps", "video_fps"),
        ):
            value = getattr(data_args, argument_name, None)
            if value is not None and hasattr(video_processor, target_name):
                setattr(video_processor, target_name, value)
        video_size = getattr(video_processor, "size", None)
        if isinstance(video_size, dict):
            if getattr(data_args, "video_min_pixels", None) is not None:
                video_size["shortest_edge"] = data_args.video_min_pixels
            if getattr(data_args, "video_max_pixels", None) is not None:
                video_size["longest_edge"] = data_args.video_max_pixels
    return processor


def _canonical_conversations(source: Mapping[str, Any]) -> list[dict[str, str]]:
    conversations = source.get("conversations")
    if conversations is not None:
        if not isinstance(conversations, Sequence) or isinstance(
            conversations, (str, bytes, bytearray)
        ):
            raise QwenDataAdapterError("conversations must be a sequence")
        result: list[dict[str, str]] = []
        for index, turn in enumerate(conversations):
            if not isinstance(turn, Mapping):
                raise QwenDataAdapterError(f"conversations[{index}] must be an object")
            speaker = turn.get("from", turn.get("role"))
            value = turn.get("value", turn.get("content"))
            if speaker not in {"human", "user", "assistant", "gpt"}:
                raise QwenDataAdapterError(
                    f"conversations[{index}].from has an unsupported role"
                )
            if not isinstance(value, str) or not value:
                raise QwenDataAdapterError(
                    f"conversations[{index}].value must be a non-empty string"
                )
            result.append({"from": str(speaker), "value": value})
        if not result:
            raise QwenDataAdapterError("conversations must not be empty")
        return result

    question = source.get("question")
    options = source.get("options")
    gold = source.get("gold")
    if not isinstance(question, str) or not question:
        raise QwenDataAdapterError("sample requires conversations or a canonical question")
    if not isinstance(options, Mapping) or set(options) != set("ABCD"):
        raise QwenDataAdapterError("canonical options must be an A/B/C/D mapping")
    if not isinstance(gold, str) or not re.fullmatch(r"<answer>[A-D]</answer>", gold):
        raise QwenDataAdapterError("canonical gold must be exactly <answer>[A-D]</answer>")
    prompt = question + "\n" + "\n".join(f"{label}. {options[label]}" for label in "ABCD")
    return [
        {"from": "human", "value": prompt},
        {"from": "assistant", "value": gold},
    ]


def build_messages(
    source: Mapping[str, Any],
    *,
    motion_placeholder_count: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve media under the declared root and build Qwen chat messages."""

    root_raw = source.get("data_path", source.get("media_root"))
    if root_raw in (None, ""):
        raise QwenDataAdapterError("sample has no explicit media_root/data_path")
    if source.get("image") not in (None, "", []):
        raise QwenDataAdapterError(
            "image-only legacy rows require a separately verified image modality adapter"
        )
    working = dict(source)
    working["conversations"] = _canonical_conversations(source)
    try:
        messages = build_legacy_messages(working, media_root=Path(root_raw))
    except Exception as exc:
        raise QwenDataAdapterError("legacy message contract validation failed") from exc
    motion_locations: list[tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        for content_index, content in enumerate(message["content"]):
            if content.get("type") == "text" and find_motion_anchors(content["text"]):
                motion_locations.append((message_index, content_index))
    if motion_placeholder_count is None and motion_locations:
        raise QwenDataAdapterError("motion anchor is present but no motion payload was prepared")
    if motion_placeholder_count is not None:
        if len(motion_locations) != 1:
            raise QwenDataAdapterError(
                f"one motion payload requires exactly one <motion> anchor; found {len(motion_locations)}"
            )
        message_index, content_index = motion_locations[0]
        content = dict(messages[message_index]["content"][content_index])
        content["text"] = replace_motion_anchors(
            content["text"], motion_placeholder_count
        )
        messages[message_index]["content"][content_index] = content
    return messages


def _motion_placeholder_id(processor: Any, data_args: Any | None) -> int:
    candidates = (
        getattr(data_args, "motion_placeholder_token_id", None) if data_args is not None else None,
        getattr(processor, "motion_placeholder_token_id", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, Mapping):
            value = vocab.get(MOTION_ANCHOR_TOKEN)
            if isinstance(value, int) and value >= 0:
                return value
    raise QwenDataAdapterError(
        "motion_placeholder_token_id must be provided explicitly by the bound model/tokenizer"
    )


def _replacement_starts(values: list[int], pattern: tuple[int, ...]) -> tuple[int, ...]:
    starts: list[int] = []
    index = 0
    while index <= len(values) - len(pattern):
        if tuple(values[index : index + len(pattern)]) == pattern:
            starts.append(index)
            index += len(pattern)
        else:
            index += 1
    return tuple(starts)


def _replace_aligned_row(
    row: torch.Tensor,
    *,
    starts: tuple[int, ...],
    pattern_length: int,
    replacement: int | bool,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    cursor = 0
    for start in starts:
        pieces.append(row[cursor:start])
        pieces.append(torch.tensor([replacement], dtype=row.dtype, device=row.device))
        cursor = start + pattern_length
    pieces.append(row[cursor:])
    return torch.cat(pieces, dim=0)


def _replace_motion_anchor_tokens(
    result: dict[str, Any],
    *,
    processor: Any,
    data_args: Any | None,
    expected_count: int,
) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise QwenDataAdapterError("processor must expose tokenizer")
    if expected_count <= 0:
        raise QwenDataAdapterError("motion placeholder count must be positive")
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise QwenDataAdapterError("tokenizer must expose its exact vocabulary") from exc
    if not isinstance(vocab, Mapping):
        raise QwenDataAdapterError("tokenizer vocabulary must be a mapping")
    start_id = vocab.get("<motion_start>")
    end_id = vocab.get("<motion_end>")
    if not isinstance(start_id, int) or not isinstance(end_id, int):
        raise QwenDataAdapterError("tokenizer must register exact motion boundary tokens")

    input_ids = result.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise QwenDataAdapterError("Qwen processor must return one 2-D input_ids tensor")
    values = input_ids[0].tolist()
    start_positions = [index for index, value in enumerate(values) if value == start_id]
    end_positions = [index for index, value in enumerate(values) if value == end_id]
    if len(start_positions) != 1 or len(end_positions) != 1:
        raise QwenDataAdapterError(
            "encoded sample must contain exactly one motion boundary pair; "
            f"found starts={len(start_positions)}, ends={len(end_positions)}"
        )
    start = start_positions[0]
    end = end_positions[0]
    if start >= end:
        raise QwenDataAdapterError("motion boundary tokens are out of order")

    # Adjacent ``<motion>`` strings are not necessarily a repetition of the
    # isolated tokenization because BPE may merge across marker boundaries.
    # Verify the entire generated interior, then replace the whole span.
    expected_interior_raw = tokenizer.encode(
        MOTION_ANCHOR_TOKEN * expected_count,
        add_special_tokens=False,
    )
    expected_interior = [int(value) for value in expected_interior_raw]
    if not expected_interior:
        raise QwenDataAdapterError("tokenizer produced no IDs for motion anchors")
    if values[start + 1 : end] != expected_interior:
        raise QwenDataAdapterError(
            "encoded motion boundary interior differs from the exact expanded anchor text"
        )

    placeholder_id = _motion_placeholder_id(processor, data_args)
    replacement_ids = torch.full(
        (expected_count,),
        placeholder_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    result["input_ids"] = torch.cat(
        [input_ids[0, : start + 1], replacement_ids, input_ids[0, end:]],
        dim=0,
    ).unsqueeze(0)
    for key in ("attention_mask", "assistant_masks"):
        value = result.get(key)
        if torch.is_tensor(value) and value.ndim == 2 and value.shape == input_ids.shape:
            replacement_value = False if key == "assistant_masks" else 1
            replacement = torch.full(
                (expected_count,),
                replacement_value,
                dtype=value.dtype,
                device=value.device,
            )
            result[key] = torch.cat(
                [value[0, : start + 1], replacement, value[0, end:]],
                dim=0,
            ).unsqueeze(0)


def _derive_assistant_mask_from_im_spans(
    input_ids: torch.Tensor,
    *,
    tokenizer: Any,
    expected_assistant_turns: int,
) -> torch.Tensor:
    """Rebuild a zeroed visual assistant mask from exact Qwen turn boundaries."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise QwenDataAdapterError("assistant-mask fallback requires one 2-D input row")
    if expected_assistant_turns <= 0:
        raise QwenDataAdapterError("supervised sample must contain an assistant turn")
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise QwenDataAdapterError("tokenizer must expose its exact vocabulary") from exc
    if not isinstance(vocab, Mapping):
        raise QwenDataAdapterError("tokenizer vocabulary must be a mapping")
    im_start_id = vocab.get("<|im_start|>")
    im_end_id = vocab.get("<|im_end|>")
    if not isinstance(im_start_id, int) or not isinstance(im_end_id, int):
        raise QwenDataAdapterError("tokenizer must register exact Qwen turn tokens")
    role_ids = [
        int(value)
        for value in tokenizer.encode("assistant\n", add_special_tokens=False)
    ]
    newline_ids = [
        int(value) for value in tokenizer.encode("\n", add_special_tokens=False)
    ]
    if not role_ids:
        raise QwenDataAdapterError("tokenizer produced no assistant role IDs")

    values = input_ids[0].tolist()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    spans: list[tuple[int, int]] = []
    for start, value in enumerate(values):
        if value != im_start_id:
            continue
        role_start = start + 1
        role_end = role_start + len(role_ids)
        if values[role_start:role_end] != role_ids:
            continue
        try:
            end = values.index(im_end_id, role_end)
        except ValueError as exc:
            raise QwenDataAdapterError("assistant turn has no closing <|im_end|>") from exc
        if im_start_id in values[role_end:end]:
            raise QwenDataAdapterError("assistant turn contains an unexpected nested turn")
        mask_end = end + 1
        if newline_ids and values[mask_end : mask_end + len(newline_ids)] == newline_ids:
            mask_end += len(newline_ids)
        spans.append((role_end, mask_end))

    if len(spans) != expected_assistant_turns:
        raise QwenDataAdapterError(
            "assistant-mask fallback turn count changed: "
            f"expected {expected_assistant_turns}, found {len(spans)}"
        )
    for start, end in spans:
        if start >= end:
            raise QwenDataAdapterError("assistant turn contains no supervised tokens")
        mask[0, start:end] = True
    return mask


def preprocess_qwen_visual(
    sources: Sequence[Mapping[str, Any]],
    processor: Any,
    data_args: Any | None = None,
    *,
    motion_placeholder_count: int | None = None,
) -> dict[str, Any]:
    """Tokenize one supervised sample using assistant-mask-aware templates.

    Hard-coded assistant/end token IDs are intentionally not supported.  The
    selected chat template must expose a Transformers assistant token mask.
    """

    if len(sources) != 1 or not isinstance(sources[0], Mapping):
        raise QwenDataAdapterError("preprocess_qwen_visual expects exactly one sample")
    messages = build_messages(
        sources[0], motion_placeholder_count=motion_placeholder_count
    )
    try:
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
        )
    except TypeError as exc:
        raise QwenDataAdapterError(
            "processor chat template must support return_assistant_tokens_mask"
        ) from exc
    result = dict(encoded)
    assistant_mask = result.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = result.get("assistant_mask")
        if assistant_mask is not None:
            result["assistant_masks"] = assistant_mask
    if isinstance(assistant_mask, list):
        assistant_mask = torch.tensor(assistant_mask, dtype=torch.bool)
        if assistant_mask.ndim == 1:
            assistant_mask = assistant_mask.unsqueeze(0)
        result["assistant_masks"] = assistant_mask
    if motion_placeholder_count is not None:
        _replace_motion_anchor_tokens(
            result,
            processor=processor,
            data_args=data_args,
            expected_count=motion_placeholder_count,
        )
        assistant_mask = result["assistant_masks"]
    input_ids = result.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise QwenDataAdapterError("processor must return a 2-D input_ids tensor")
    if not torch.is_tensor(assistant_mask) or assistant_mask.shape != input_ids.shape:
        raise QwenDataAdapterError(
            "chat template did not return an assistant mask aligned with input_ids"
        )
    assistant_mask = assistant_mask.to(dtype=torch.bool, device=input_ids.device)
    if not bool(assistant_mask.any()):
        assistant_mask = _derive_assistant_mask_from_im_spans(
            input_ids,
            tokenizer=processor.tokenizer,
            expected_assistant_turns=sum(
                message.get("role") == "assistant" for message in messages
            ),
        ).to(device=input_ids.device)
    labels = input_ids.clone()
    labels.masked_fill_(~assistant_mask, IGNORE_INDEX)
    result["labels"] = labels
    result.pop("assistant_masks", None)
    result.pop("assistant_mask", None)
    return result


__all__ = [
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_VIDEO_TOKEN",
    "IGNORE_INDEX",
    "MOTION_ANCHOR_TOKEN",
    "QwenDataAdapterError",
    "build_messages",
    "preprocess_qwen_visual",
    "update_processor_pixels",
]
