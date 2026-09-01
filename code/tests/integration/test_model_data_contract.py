from __future__ import annotations

import numpy as np

from motionllm.contracts import Modality
from motionllm.data import (
    build_legacy_messages,
    describe_legacy_sample,
    logical_sample_payload,
    plan_collation,
)
from motionllm.motion import (
    load_motion_array,
    load_normalization_stats,
    normalize_motion,
    prepare_motion_temporal,
)


def test_vm_source_to_messages_motion_and_collation(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    raw_motion = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    np.save(tmp_path / "motion.npy", raw_motion)
    np.save(tmp_path / "Mean.npy", np.zeros(3, dtype=np.float32))
    np.save(tmp_path / "Std.npy", np.ones(3, dtype=np.float32))
    source = {
        "sample_id": "sample-1",
        "group_id": "group-1",
        "modality": "VM",
        "video": "video.mp4",
        "motion": "motion.npy",
        "conversations": [
            {"from": "human", "value": "<video> <motion> What happens?"},
            {"from": "gpt", "value": "<answer>A</answer>"},
        ],
    }

    descriptor = describe_legacy_sample(source)
    messages = build_legacy_messages(source, media_root=tmp_path)
    loaded = load_motion_array(tmp_path / "motion.npy", expected_feature_dim=3)
    mean, std = load_normalization_stats(
        tmp_path / "Mean.npy", tmp_path / "Std.npy", expected_feature_dim=3
    )
    normalized = normalize_motion(loaded, mean, std)
    padded, temporal = prepare_motion_temporal(normalized, downsample_factor=4)
    instance = {
        "motion": padded,
        "motion_lengths": (padded.shape[0],),
        "logical_samples": (
            logical_sample_payload(
                sample_id=descriptor.sample_id,
                group_id=descriptor.group_id,
                modality=descriptor.modality,
                motion_length=padded.shape[0],
            ),
        ),
    }
    plan = plan_collation([instance])

    assert descriptor.modality is Modality.VIDEO_MOTION
    assert messages[0]["content"][0]["type"] == "video"
    assert temporal.raw_length == 7
    assert temporal.padded_length == 8
    assert temporal.placeholder_count == 2
    assert plan.motion_lengths == ((8,),)
    assert plan.packed_sample_ids == ("sample-1",)
