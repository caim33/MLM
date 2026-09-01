from __future__ import annotations

import json
from pathlib import Path

from motionllm.contracts import Modality, OptionLabel
from motionllm.data import read_samples_jsonl


def test_canonical_jsonl_contract_round_trips_without_media_substitution(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    motion = tmp_path / "motion.npy"
    video.write_bytes(b"video")
    motion.write_bytes(b"motion")
    rows = [
        {
            "sample_id": "sample-vm",
            "group_id": "group-1",
            "modality": "VM",
            "branch": "vm",
            "question": "Which motion is shown?",
            "options": {
                label.value: f"choice {label.value}" for label in OptionLabel
            },
            "gold": "<answer>C</answer>",
            "video": "clip.mp4",
            "motion": "motion.npy",
            "rollout_id": 7,
            "request_id": "request-7",
            "motion_lengths": [20],
            "metadata": {"split": "train"},
        }
    ]
    source = tmp_path / "canonical.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    sample = read_samples_jsonl(source)[0]
    serialized = sample.to_dict()

    assert sample.modality is Modality.VIDEO_MOTION
    assert sample.video == video.resolve()
    assert sample.motion == motion.resolve()
    assert serialized["sample_id"] == "sample-vm"
    assert serialized["branch"] == "vm"
    assert serialized["gold"] == "<answer>C</answer>"
    assert serialized["motion_lengths"] == [20]
