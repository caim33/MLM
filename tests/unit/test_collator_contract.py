from __future__ import annotations

import numpy as np
import pytest

from motionllm.contracts import Modality
from motionllm.data import (
    CollationContractError,
    logical_sample_payload,
    plan_collation,
)


def instance(sample_id, group_id, branch, motion=None, **extra):
    value = {
        "sample_id": sample_id,
        "group_id": group_id,
        "branch": branch,
    }
    if motion is not None:
        value["motion"] = motion
        value["motion_lengths"] = (motion.shape[0],)
    value.update(extra)
    return value


def test_standard_mixed_batch_preserves_per_row_motion_and_metadata():
    m1 = np.zeros((8, 251), dtype=np.float32)
    m2 = np.zeros((12, 251), dtype=np.float32)
    plan = plan_collation(
        [
            instance("v", "g-v", "v"),
            instance("m", "g-m", "m", m1),
            instance("vm", "g-vm", "vm", m2),
            instance("t", "g-t", "t"),
        ]
    )
    assert plan.motions == (None, m1, m2, None)
    assert plan.motion_lengths == (None, (8,), (12,), None)
    assert plan.physical_branches == ("v", "m", "vm", "t")
    assert plan.packed_sample_ids == ("v", "m", "vm", "t")
    assert plan.packed_group_ids == ("g-v", "g-m", "g-vm", "g-t")
    assert plan.motion_owner_indices == (1, 2)


def test_flattened_batch_greater_than_one_keeps_nested_ownership():
    packed_a_motion = np.zeros((20, 251), dtype=np.float32)
    packed_b_motion = np.zeros((4, 251), dtype=np.float32)
    packed_a = {
        "motion": packed_a_motion,
        "motion_lengths": (8, 12),
        "logical_samples": (
            logical_sample_payload(
                sample_id="a-m",
                group_id="ga",
                modality=Modality.MOTION,
                motion_length=8,
            ),
            logical_sample_payload(
                sample_id="a-vm",
                group_id="ga",
                modality=Modality.VIDEO_MOTION,
                motion_length=12,
            ),
        ),
    }
    packed_b = {
        "motion": packed_b_motion,
        "motion_lengths": (4,),
        "logical_samples": (
            logical_sample_payload(
                sample_id="b-v",
                group_id="gb",
                modality=Modality.VIDEO,
                motion_length=None,
            ),
            logical_sample_payload(
                sample_id="b-m",
                group_id="gb",
                modality=Modality.MOTION,
                motion_length=4,
            ),
        ),
    }
    plan = plan_collation([packed_a, packed_b])
    assert plan.motion_lengths == ((8, 12), (4,))
    assert plan.packed_branches == ("m", "vm", "v", "m")
    assert plan.packed_sample_ids == ("a-m", "a-vm", "b-v", "b-m")
    assert plan.motion_owner_indices == (0, 1, 3)
    assert plan.physical_branches == ("vm", "vm")


def test_modality_motion_presence_mismatch_fails_closed():
    with pytest.raises(CollationContractError, match="disagree"):
        plan_collation([instance("bad", "g", "m")])
    with pytest.raises(CollationContractError, match="disagree"):
        plan_collation(
            [instance("bad", "g", "v", np.zeros((4, 251), dtype=np.float32))]
        )


def test_motion_lengths_cannot_claim_other_rows():
    with pytest.raises(CollationContractError, match="owned rows"):
        plan_collation(
            [
                {
                    "motion": np.zeros((9, 251), dtype=np.float32),
                    "motion_lengths": (8,),
                    "logical_samples": (
                        logical_sample_payload(
                            sample_id="m",
                            group_id="g",
                            modality=Modality.MOTION,
                            motion_length=8,
                        ),
                    ),
                }
            ]
        )

