"""Strict adaptation of legacy Qwen conversation rows into typed messages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from motionllm.contracts import Modality

from .errors import MessageContractError
from .paths import resolve_media_path


@dataclass(frozen=True, slots=True)
class LegacySampleDescriptor:
    sample_id: str
    group_id: str
    modality: Modality


def _identifier(source: Mapping[str, Any], field_name: str) -> str:
    value = source.get(field_name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise MessageContractError(f"{field_name} must be a non-empty canonical string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise MessageContractError(f"{field_name} contains a control character")
    return value


def _media_value(source: Mapping[str, Any], field_name: str) -> str | os.PathLike[str] | None:
    value = source.get(field_name)
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
        raise MessageContractError(f"{field_name} must be a path string or null")
    raw = os.fspath(value)
    if not raw or not raw.strip() or "\x00" in raw:
        raise MessageContractError(f"{field_name} must be a non-empty path string")
    return value


def infer_legacy_modality(source: Mapping[str, Any]) -> Modality:
    """Infer the exact V/M/VM/T matrix and verify any declared identity."""

    if not isinstance(source, Mapping):
        raise MessageContractError("legacy sample must be an object")
    video = _media_value(source, "video")
    motion = _media_value(source, "motion")
    actual = {
        (True, False): Modality.VIDEO,
        (False, True): Modality.MOTION,
        (True, True): Modality.VIDEO_MOTION,
        (False, False): Modality.TEXT,
    }[(video is not None, motion is not None)]

    declared: list[Modality] = []
    if "modality" in source:
        try:
            declared.append(Modality.parse(source["modality"]))
        except ValueError as exc:
            raise MessageContractError(str(exc)) from exc
    if "branch" in source:
        try:
            declared.append(Modality.from_branch(source["branch"]))
        except ValueError as exc:
            raise MessageContractError(str(exc)) from exc
    if declared and any(value is not actual for value in declared):
        raise MessageContractError(
            f"declared modality disagrees with video/motion paths for {actual.value}"
        )
    if len(declared) == 2 and declared[0] is not declared[1]:
        raise MessageContractError("modality and branch disagree")
    return actual


def describe_legacy_sample(source: Mapping[str, Any]) -> LegacySampleDescriptor:
    """Return stable identity and modality without reading any media."""

    if not isinstance(source, Mapping):
        raise MessageContractError("legacy sample must be an object")
    return LegacySampleDescriptor(
        sample_id=_identifier(source, "sample_id"),
        group_id=_identifier(source, "group_id"),
        modality=infer_legacy_modality(source),
    )


def _conversation_rows(source: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = source.get("conversations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MessageContractError("conversations must be a non-empty array")
    rows: list[tuple[str, str]] = []
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise MessageContractError(f"conversations[{index}] must be an object")
        role_value = entry.get("from", entry.get("role"))
        if role_value not in role_map:
            raise MessageContractError(f"conversations[{index}] has an unsupported role")
        text = entry.get("value", entry.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise MessageContractError(f"conversations[{index}] text must not be empty")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in text):
            raise MessageContractError(f"conversations[{index}] contains invalid Unicode")
        rows.append((role_map[role_value], text))
    return rows


def build_legacy_messages(
    source: Mapping[str, Any],
    *,
    media_root: str | os.PathLike[str],
    check_media_exists: bool = True,
) -> list[dict[str, Any]]:
    """Build Qwen-style messages while preserving motion anchors and identity."""

    descriptor = describe_legacy_sample(source)
    root = Path(media_root)
    video_value = _media_value(source, "video")
    motion_value = _media_value(source, "motion")
    resolved_video = (
        resolve_media_path(root, video_value, must_exist=check_media_exists)
        if video_value is not None
        else None
    )
    if motion_value is not None:
        resolve_media_path(root, motion_value, must_exist=check_media_exists)

    conversations = _conversation_rows(source)
    motion_anchor_count = sum(text.count("<motion>") for _, text in conversations)
    expected_motion_anchors = 1 if descriptor.modality.requires_motion else 0
    if motion_anchor_count != expected_motion_anchors:
        raise MessageContractError(
            f"motion anchor count must be exactly {expected_motion_anchors}"
        )
    video_anchor_count = sum(text.count("<video>") for _, text in conversations)
    expected_video_anchors = 1 if descriptor.modality.requires_video else 0
    if video_anchor_count != expected_video_anchors:
        raise MessageContractError(
            f"video anchor count must be exactly {expected_video_anchors}"
        )

    messages: list[dict[str, Any]] = []
    for role, original_text in conversations:
        if role == "assistant" and ("<video>" in original_text or "<motion>" in original_text):
            raise MessageContractError("media anchors must appear in a user message")
        content: list[dict[str, str]] = []
        text = original_text
        if "<video>" in text:
            if role != "user" or resolved_video is None:
                raise MessageContractError("video anchor disagrees with media identity")
            content.append({"type": "video", "video": str(resolved_video)})
            text = text.replace("<video>", "", 1)
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        if not content:
            raise MessageContractError("conversation becomes empty after media adaptation")
        messages.append({"role": role, "content": content})
    return messages
