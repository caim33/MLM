from __future__ import annotations

from pathlib import Path

import pytest

from motionllm.contracts import Modality
from motionllm.data import (
    MessageContractError,
    build_legacy_messages,
    describe_legacy_sample,
    infer_legacy_modality,
)


def source(**overrides):
    value = {
        "sample_id": "s1",
        "group_id": "g1",
        "conversations": [
            {"from": "human", "value": "<video>\n<motion>\nQuestion?"},
            {"from": "gpt", "value": "<answer>A</answer>"},
        ],
        "video": "clip.mp4",
        "motion": "motion.npy",
    }
    value.update(overrides)
    return value


def media_root(tmp_path: Path) -> Path:
    (tmp_path / "clip.mp4").write_bytes(b"video")
    (tmp_path / "motion.npy").write_bytes(b"motion")
    return tmp_path


def test_descriptor_and_messages_preserve_vm_identity(tmp_path):
    root = media_root(tmp_path)
    item = source(branch="vm")
    descriptor = describe_legacy_sample(item)
    assert descriptor.sample_id == "s1"
    assert descriptor.group_id == "g1"
    assert descriptor.modality is Modality.VIDEO_MOTION
    messages = build_legacy_messages(item, media_root=root)
    user = messages[0]
    assert [part["type"] for part in user["content"]] == ["video", "text"]
    assert user["content"][0]["video"] == str((root / "clip.mp4").resolve())
    assert "<motion>" in user["content"][1]["text"]


@pytest.mark.parametrize(
    "video,motion,expected",
    [
        ("clip.mp4", None, Modality.VIDEO),
        (None, "motion.npy", Modality.MOTION),
        ("clip.mp4", "motion.npy", Modality.VIDEO_MOTION),
        (None, None, Modality.TEXT),
    ],
)
def test_exact_media_matrix(video, motion, expected):
    item = source(video=video, motion=motion)
    assert infer_legacy_modality(item) is expected


def test_declared_modality_cannot_disagree_with_paths():
    with pytest.raises(MessageContractError, match="disagrees"):
        infer_legacy_modality(source(branch="v"))
    with pytest.raises(MessageContractError):
        infer_legacy_modality(source(video="", motion=None))
    with pytest.raises(MessageContractError):
        infer_legacy_modality(source(video=[], motion=None))


@pytest.mark.parametrize(
    "text",
    [
        "<video> Question without motion anchor",
        "<video> <motion> duplicate <motion>",
    ],
)
def test_motion_anchor_count_is_exact(tmp_path, text):
    root = media_root(tmp_path)
    item = source(
        conversations=[
            {"from": "human", "value": text},
            {"from": "gpt", "value": "answer"},
        ]
    )
    with pytest.raises(MessageContractError, match="motion anchor"):
        build_legacy_messages(item, media_root=root)


def test_video_and_motion_are_both_forbidden_for_text(tmp_path):
    item = source(
        video=None,
        motion=None,
        conversations=[
            {"from": "human", "value": "Text only"},
            {"from": "gpt", "value": "answer"},
        ],
    )
    assert build_legacy_messages(item, media_root=tmp_path)[0]["content"] == [
        {"type": "text", "text": "Text only"}
    ]


def test_missing_identity_is_never_replaced_by_another_row():
    item = source()
    del item["sample_id"]
    with pytest.raises(MessageContractError, match="sample_id"):
        describe_legacy_sample(item)


def test_media_escape_is_rejected(tmp_path):
    media_root(tmp_path)
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    item = source(video="../outside.mp4")
    with pytest.raises(Exception, match="outside"):
        build_legacy_messages(item, media_root=tmp_path)
