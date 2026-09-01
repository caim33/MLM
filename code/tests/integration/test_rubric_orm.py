from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from motionllm.grpo import RubricValidationError
from motionllm.grpo.rubric_adapter import (
    motion_rubric_v2_rewards,
    qa_rubric_rewards,
)
from tests.unit.test_motion_rubric_v2 import (
    candidate_and_observations,
    motion_criteria,
    motion_judgment,
)
from tests.unit.test_qa_rubric import qa_criteria, qa_judgment


def test_qa_rubric_batch_adapter_binds_sample_gold_and_supports_generations():
    criteria = qa_criteria()
    completion = "<think>All required specific facts support option A.</think><answer>A</answer>"
    judgment = qa_judgment(criteria, completion)
    rewards = qa_rubric_rewards(
        [completion, completion],
        qa_rubric_criteria=[criteria],
        qa_rubric_judgment=[judgment, judgment],
        sample_id=["QA_000001"],
        group_id=["g1"],
        branch=["vm"],
        rollout_id=[0],
        answer=["A"],
        num_generations=2,
    )
    assert rewards == [1.0, 1.0]

    with pytest.raises(RubricValidationError, match="sample_id"):
        qa_rubric_rewards(
            [completion],
            qa_rubric_criteria=criteria,
            qa_rubric_judgment=judgment,
            sample_id="wrong",
            group_id="g1",
            branch="vm",
            rollout_id=0,
            answer="A",
        )


def test_rubric_adapters_require_judgment_or_explicit_online_client():
    criteria = qa_criteria()
    completion = "<think>specific</think><answer>A</answer>"
    with pytest.raises(RubricValidationError, match="configured online judge"):
        qa_rubric_rewards(
            [completion],
            qa_rubric_criteria=criteria,
            sample_id="QA_000001",
            group_id="g1",
            branch="vm",
            rollout_id=0,
            answer="A",
        )

    class FakeJudge:
        def judge_qa(self, criterion, candidate):
            return qa_judgment(criterion, candidate)

    assert qa_rubric_rewards(
        [completion],
        qa_rubric_criteria=criteria,
        judge_client=FakeJudge(),
        sample_id="QA_000001",
        group_id="g1",
        branch="vm",
        rollout_id=0,
        answer="A",
    ) == [1.0]


def test_motion_v2_adapter_passes_raw_candidate_for_numeric_verification_and_rejects_stage1():
    criteria = motion_criteria()
    candidate, _ = candidate_and_observations()
    judgment = motion_judgment(criteria, candidate, sample_id="motion_1")
    assert motion_rubric_v2_rewards(
        [candidate],
        motion_rubric_v2_criteria=criteria,
        motion_rubric_v2_id="motion_1",
        motion_rubric_v2_judgment=judgment,
        sample_id="motion_1",
    ) == [1.0]
    with pytest.raises(RubricValidationError, match="mode"):
        motion_rubric_v2_rewards(
            [candidate],
            motion_rubric_v2_criteria={"mode": "temporal_caption"},
            motion_rubric_v2_id="motion_1",
            motion_rubric_v2_judgment=judgment,
            sample_id="motion_1",
        )


def test_swift_plugin_registers_both_versioned_rubric_orms(monkeypatch):
    swift = types.ModuleType("swift")
    rewards_module = types.ModuleType("swift.rewards")
    callbacks_module = types.ModuleType("swift.callbacks")

    class ORM:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    rewards_module.ORM = ORM
    rewards_module.orms = {}
    callbacks_module.TrainerCallback = ORM
    callbacks_module.callbacks_map = {}
    swift.rewards = rewards_module
    swift.callbacks = callbacks_module
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.rewards", rewards_module)
    monkeypatch.setitem(sys.modules, "swift.callbacks", callbacks_module)
    plugin = Path(__file__).resolve().parents[2] / "qwenvl" / "grpo_ms_swift" / "plugins" / "swift_external_rewards.py"
    spec = importlib.util.spec_from_file_location("rubric_plugin_test", plugin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "qa_mc_rubric" in rewards_module.orms
    assert "motion_rubric_v2" in rewards_module.orms
    assert "motion_training_receipt" in callbacks_module.callbacks_map

    orm = rewards_module.orms["qa_mc_rubric"]()
    criteria = qa_criteria()
    completion = "<think>specific answer-relevant reasoning</think><answer>A</answer>"
    judgment = qa_judgment(criteria, completion)
    assert orm(
        [completion],
        sample_id=["QA_000001"],
        group_id=["g1"],
        branch=["vm"],
        rollout_id=[0],
        answer=["A"],
        qa_rubric_criteria=[json.dumps(criteria)],
        qa_rubric_judgment=[json.dumps(judgment)],
    ) == [1.0]


def test_qa_adapter_rejects_scalar_judgment_broadcast_across_samples():
    first = qa_criteria()
    second = json.loads(json.dumps(first))
    second["benchmark_id"] = "QA_000002"
    completion = "<think>specific reasoning</think><answer>A</answer>"
    scalar_judgment = qa_judgment(first, completion)
    with pytest.raises(RubricValidationError, match="scalar broadcast"):
        qa_rubric_rewards(
            [completion, completion],
            qa_rubric_criteria=[first, second],
            qa_rubric_judgment=scalar_judgment,
            sample_id=["QA_000001", "QA_000002"],
            group_id=["g1", "g2"],
            branch=["vm", "vm"],
            rollout_id=[0, 0],
            answer=["A", "A"],
        )


def test_motion_adapter_rejects_scalar_criteria_broadcast_across_samples():
    criteria = motion_criteria()
    candidate, _ = candidate_and_observations()
    sample_ids = ["motion_1", "motion_2"]
    judgments = [
        motion_judgment(criteria, candidate, sample_id=sample_id)
        for sample_id in sample_ids
    ]
    with pytest.raises(RubricValidationError, match="one sample_id"):
        motion_rubric_v2_rewards(
            [candidate, candidate],
            motion_rubric_v2_criteria=criteria,
            motion_rubric_v2_id=sample_ids,
            motion_rubric_v2_judgment=judgments,
            sample_id=sample_ids,
            num_generations=2,
        )

    same_sample = ["motion_1", "motion_1"]
    same_judgment = motion_judgment(
        criteria, candidate, sample_id="motion_1"
    )
    assert motion_rubric_v2_rewards(
        [candidate, candidate],
        motion_rubric_v2_criteria=criteria,
        motion_rubric_v2_id=same_sample,
        motion_rubric_v2_judgment=[same_judgment, same_judgment],
        sample_id=same_sample,
        num_generations=2,
    ) == [1.0, 1.0]
