from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from motionllm.contracts import Modality
from motionllm.data import logical_sample_payload
from qwenvl.data.data_processor import (
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)


def prepared(sample_id, modality, length, motion=None, video=False):
    item = {
        "input_ids": torch.arange(length).unsqueeze(0),
        "labels": torch.arange(length).unsqueeze(0),
        "position_ids": torch.arange(length).reshape(1, 1, length).repeat(3, 1, 1),
        "attention_mask": [length],
        "sample_id": sample_id,
        "group_id": f"g-{sample_id}",
        "branch": modality.branch,
        "logical_samples": (
            logical_sample_payload(
                sample_id=sample_id,
                group_id=f"g-{sample_id}",
                modality=modality,
                motion_length=(int(motion.shape[0]) if motion is not None else None),
            ),
        ),
    }
    if motion is not None:
        item["motion"] = motion
        item["motion_lengths"] = [int(motion.shape[0])]
    if video:
        item["pixel_values_videos"] = torch.zeros((1, 2))
        item["video_grid_thw"] = torch.ones((1, 3), dtype=torch.long)
    return item


def tokenizer():
    return SimpleNamespace(pad_token_id=0, model_max_length=128)


def test_legacy_standard_collator_preserves_mixed_batch_rows():
    instances = [
        prepared("v", Modality.VIDEO, 4, video=True),
        prepared("m", Modality.MOTION, 5, torch.zeros((8, 251))),
        prepared(
            "vm",
            Modality.VIDEO_MOTION,
            6,
            torch.zeros((12, 251)),
            video=True,
        ),
        prepared("t", Modality.TEXT, 3),
    ]
    batch = DataCollatorForSupervisedDataset(tokenizer())(instances)
    assert len(batch["motion"]) == 4
    assert batch["motion"][0] is None
    assert batch["motion"][1].shape == (8, 251)
    assert batch["motion_lengths"] == [None, (8,), (12,), None]
    assert batch["branch"] == ["v", "m", "vm", "t"]
    assert batch["sample_id"] == ["v", "m", "vm", "t"]
    assert batch["motion_owner_indices"] == (1, 2)


def test_legacy_flattened_collator_keeps_batch_greater_than_one_ownership():
    first = prepared(
        "one", Modality.MOTION, 4, torch.zeros((8, 251))
    )
    second = prepared(
        "two", Modality.VIDEO_MOTION, 5, torch.zeros((12, 251)), video=True
    )
    batch = FlattenedDataCollatorForSupervisedDataset(tokenizer())([first, second])
    assert batch["input_ids"].shape == (1, 9)
    assert batch["motion"].shape == (20, 251)
    assert batch["motion_lengths"] == ((8, 12),)
    assert batch["branch"] == "vm"
    assert batch["sample_id"] == (("one", "two"),)
    assert batch["packed_branch"] == ("m", "vm")
    assert batch["motion_owner_indices"] == (0, 1)


def test_legacy_collator_rejects_missing_required_video_payload():
    item = prepared("v", Modality.VIDEO, 4, video=False)
    with pytest.raises(ValueError, match="video tensor presence"):
        DataCollatorForSupervisedDataset(tokenizer())([item])


def test_legacy_dataset_never_substitutes_the_next_index():
    dataset = object.__new__(LazySupervisedDataset)
    dataset.grouped_sampling = False
    dataset.groups = None
    dataset.data_args = SimpleNamespace(
        sample_read_retries=2,
        sample_read_retry_delay_seconds=0,
    )
    dataset.list_data_dict = [
        {"sample_id": "bad", "group_id": "g"},
        {"sample_id": "good", "group_id": "g"},
    ]
    calls = []

    def item_fn(sources):
        calls.append(sources[0]["sample_id"])
        if sources[0]["sample_id"] == "bad":
            raise OSError("broken media")
        return {"sample_id": "good"}

    dataset.item_fn = item_fn
    with pytest.raises(RuntimeError, match="no substitute row"):
        dataset[0]
    assert calls == ["bad", "bad"]
