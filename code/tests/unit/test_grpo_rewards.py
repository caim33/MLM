from __future__ import annotations

import math
import random

import pytest

from motionllm.grpo import (
    COLOCATION_CONFIG_ENV,
    COLOCATION_DATASET_ENV,
    COLOCATION_NONCE_ENV,
    COLOCATION_PATH_ENV,
    COLOCATION_PLAN_ENV,
    GroupBonusConfig,
    GroupScore,
    RewardMetadata,
    RewardMetadataError,
    answer_reward,
    build_reward_metadata_batch,
    compute_group_bonus,
    format_reward,
    initialize_runtime_colocation_plan,
    option_accuracy_rewards,
    semantic_reward,
    validate_semantic_reference,
    validate_runtime_colocation_receipt,
    vm_v_group_bonus_rewards,
)


@pytest.mark.parametrize(
    "completion",
    [
        "A",
        "Answer: A",
        "<answer> A </answer>",
        "<answer>a</answer>",
        "<answer>A.</answer>",
        "<answer>A</answer><answer>A</answer>",
        "<answer>B</answer> then A",
    ],
)
def test_answer_reward_rejects_wide_or_wrong_outputs(completion):
    assert answer_reward(completion, "A") == 0.0


def test_answer_and_format_rewards_use_strict_parser():
    completion = "<think>visible reasoning</think><answer>A</answer>"
    assert answer_reward(completion, "<answer>A</answer>") == 1.0
    assert format_reward(completion) == 1.0
    assert format_reward("reasoning <answer>A</answer>") == 0.0
    with pytest.raises(RewardMetadataError):
        answer_reward("<answer>A</answer>", "reasoning <answer>A</answer>")


@pytest.mark.parametrize(
    "reasoning, expected",
    [
        ("visible evidence", 1.0),
        ("人物向左转身", 1.0),
        ("", 0.0),
        ("   ", 0.0),
        ("!!! 💩 ???", 0.0),
    ],
)
def test_reasoning_validity_requires_a_normalized_english_or_chinese_token(reasoning, expected):
    completion = f"<think>{reasoning}</think><answer>A</answer>"
    assert format_reward(completion) == expected


def test_semantic_reward_is_deterministic_and_answer_gated():
    reference = "<think>person turns left slowly</think><answer>B</answer>"
    same = "<think>person turns left</think><answer>B</answer>"
    wrong = "<think>person turns left</think><answer>C</answer>"
    assert semantic_reward(same, reference) == semantic_reward(same, reference)
    assert 0.0 < semantic_reward(same, reference) <= 1.0
    assert semantic_reward(wrong, reference) == 0.0


def test_semantic_reward_rejects_missing_or_tokenless_reasoning_reference():
    assert semantic_reward(
        "<think>valid</think><answer>A</answer>", "<answer>A</answer>"
    ) == 0.0
    assert semantic_reward(
        "<think>!!!</think><answer>A</answer>",
        "<think>??? 💩</think><answer>A</answer>",
    ) == 0.0
    with pytest.raises(RewardMetadataError, match="non-empty reasoning"):
        validate_semantic_reference("garbage <answer>A</answer>", "A")
    with pytest.raises(RewardMetadataError, match="non-empty reasoning"):
        validate_semantic_reference("<think>!!!</think><answer>A</answer>", "A")


def metadata(sample, group, branch, rollout, answer="A", generation=0):
    return RewardMetadata(
        sample_id=sample,
        group_id=group,
        branch=branch,
        rollout_id=rollout,
        gold_answer=answer,
        generation_id=generation,
    )


def test_group_bonus_is_shuffle_invariant_by_rollout_identity():
    records = [
        GroupScore(metadata("s1", "g1", "vm", 0), 1.0),
        GroupScore(metadata("s1", "g1", "v", 1), 0.5),
        GroupScore(metadata("s2", "g2", "vm", 0), 0.0),
        GroupScore(metadata("s2", "g2", "v", 1), 1.0),
    ]
    original = compute_group_bonus(records)
    shuffled = list(records)
    random.Random(123).shuffle(shuffled)
    changed = compute_group_bonus(shuffled)
    original_by_key = {
        record.metadata.rollout_key: bonus
        for record, bonus in zip(records, original.bonuses)
    }
    changed_by_key = {
        record.metadata.rollout_key: bonus
        for record, bonus in zip(shuffled, changed.bonuses)
    }
    assert original_by_key == changed_by_key
    assert original.group_gate == changed.group_gate


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_group_bonus_rejects_nonfinite_or_out_of_range_scores(score):
    with pytest.raises(RewardMetadataError):
        GroupScore(metadata("s", "g", "vm", 0), score)


def test_group_bonus_rejects_duplicate_rollouts_and_bad_branch():
    record = GroupScore(metadata("s", "g", "vm", 0), 1.0)
    with pytest.raises(RewardMetadataError, match="duplicate"):
        compute_group_bonus([record, record])
    with pytest.raises(RewardMetadataError, match="only v and vm"):
        compute_group_bonus([GroupScore(metadata("s", "g", "m", 0), 1.0)])


def test_swift_adapter_raises_on_missing_or_misaligned_metadata():
    completions = ["<answer>A</answer>", "<answer>B</answer>"]
    with pytest.raises(RewardMetadataError):
        option_accuracy_rewards(completions, answer=["A", "B"])
    from motionllm.grpo import format_rewards

    with pytest.raises(RewardMetadataError):
        format_rewards(["<think>x</think><answer>A</answer>"])
    with pytest.raises(RewardMetadataError):
        option_accuracy_rewards(
            completions,
            sample_id=["s1"],
            group_id=["g1", "g2"],
            branch=["vm", "v"],
            rollout_id=[0, 1],
            answer=["A", "B"],
        )


def test_swift_metadata_has_no_heuristic_defaults():
    with pytest.raises(RewardMetadataError):
        build_reward_metadata_batch(
            1,
            sample_id=None,
            group_id="g",
            branch="vm",
            rollout_id=0,
            answer="A",
        )


def test_swift_group_reward_happy_path_and_nan_config_fail_closed():
    completions = ["<answer>A</answer>", "<answer>A</answer>"]
    columns = {
        "sample_id": ["s", "s"],
        "group_id": ["g", "g"],
        "branch": ["vm", "v"],
        "rollout_id": [0, 1],
        "answer": ["A", "A"],
    }
    assert vm_v_group_bonus_rewards(completions, **columns) == [0.1, 0.0]
    with pytest.raises(RewardMetadataError):
        GroupBonusConfig(threshold=math.nan)
    with pytest.raises(RewardMetadataError, match="does not co-locate"):
        vm_v_group_bonus_rewards(
            ["<answer>A</answer>"],
            sample_id=["s_vm"],
            group_id=["g"],
            branch=["vm"],
            rollout_id=[0],
            answer=["A"],
        )


def test_solution_must_be_strict_and_match_gold():
    with pytest.raises(RewardMetadataError, match="matching"):
        RewardMetadata(
            sample_id="s",
            group_id="g",
            branch="vm",
            rollout_id=0,
            gold_answer="A",
            solution="<answer>B</answer>",
        )
    with pytest.raises(RewardMetadataError, match="matching"):
        RewardMetadata(
            sample_id="s",
            group_id="g",
            branch="vm",
            rollout_id=0,
            gold_answer="A",
            solution="Answer A",
        )


@pytest.mark.parametrize(
    "completion",
    [
        "<think>outer <think>inner</think></think><answer>A</answer>",
        "<think>x</think><think>y</think><answer>A</answer>",
        "<think attr='x'>x</think><answer>A</answer>",
        "<think>x</think></think><answer>A</answer>",
    ],
)
def test_format_reward_rejects_nested_repeated_or_malformed_think(completion):
    assert format_reward(completion) == 0.0


def test_num_generations_expands_metadata_and_assigns_unique_generation_ids():
    rows = build_reward_metadata_batch(
        4,
        sample_id=["s_vm", "s_v"],
        group_id=["g", "g"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
        solution=["<answer>A</answer>", "<answer>A</answer>"],
        num_generations=2,
    )
    assert [row.sample_id for row in rows] == ["s_vm", "s_vm", "s_v", "s_v"]
    assert [row.generation_id for row in rows] == [0, 1, 2, 3]
    assert len({row.rollout_key for row in rows}) == 4

    completions = [
        "<answer>A</answer>",
        "<answer>B</answer>",
        "<answer>A</answer>",
        "<answer>B</answer>",
    ]
    bonuses = vm_v_group_bonus_rewards(
        completions,
        sample_id=["s_vm", "s_v"],
        group_id=["g", "g"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
        num_generations=2,
    )
    assert bonuses == [0.1, 0.0, 0.0, 0.0]


def test_actual_group_reward_call_writes_nonce_and_hash_bound_runtime_receipt(
    tmp_path, monkeypatch
):
    path = tmp_path / "runtime_colocation.json"
    nonce = "a" * 32
    dataset_digest = "b" * 64
    config_digest = "c" * 64
    planned = build_reward_metadata_batch(
        2,
        sample_id=["s_vm", "s_v"],
        group_id=["g", "g"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
    )
    initialized = initialize_runtime_colocation_plan(
        path,
        nonce=nonce,
        dataset_digest=dataset_digest,
        config_digest=config_digest,
        planned_calls=[planned],
    )
    plan_digest = initialized["plan"]["plan_sha256"]
    monkeypatch.setenv(COLOCATION_PATH_ENV, str(path))
    monkeypatch.setenv(COLOCATION_NONCE_ENV, nonce)
    monkeypatch.setenv(COLOCATION_DATASET_ENV, dataset_digest)
    monkeypatch.setenv(COLOCATION_CONFIG_ENV, config_digest)
    monkeypatch.setenv(COLOCATION_PLAN_ENV, plan_digest)
    rewards = vm_v_group_bonus_rewards(
        ["<answer>A</answer>", "<answer>A</answer>"],
        sample_id=["s_vm", "s_v"],
        group_id=["g", "g"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
    )
    assert rewards == [0.1, 0.0]
    receipt = validate_runtime_colocation_receipt(
        path,
        nonce=nonce,
        dataset_digest=dataset_digest,
        config_digest=config_digest,
        plan_digest=plan_digest,
    )
    assert len(receipt["observed_calls"]) == 1
    with pytest.raises(RewardMetadataError, match="binding mismatch"):
        validate_runtime_colocation_receipt(
            path,
            nonce="d" * 32,
            dataset_digest=dataset_digest,
            config_digest=config_digest,
            plan_digest=plan_digest,
        )


def test_runtime_colocation_rejects_missing_duplicate_and_unplanned_calls(
    tmp_path, monkeypatch
):
    path = tmp_path / "runtime_colocation.json"
    nonce = "e" * 32
    dataset_digest = "f" * 64
    config_digest = "0" * 64
    planned = build_reward_metadata_batch(
        2,
        sample_id=["s_vm", "s_v"],
        group_id=["g", "g"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
    )
    initialized = initialize_runtime_colocation_plan(
        path,
        nonce=nonce,
        dataset_digest=dataset_digest,
        config_digest=config_digest,
        planned_calls=[planned, planned],
    )
    plan_digest = initialized["plan"]["plan_sha256"]
    monkeypatch.setenv(COLOCATION_PATH_ENV, str(path))
    monkeypatch.setenv(COLOCATION_NONCE_ENV, nonce)
    monkeypatch.setenv(COLOCATION_DATASET_ENV, dataset_digest)
    monkeypatch.setenv(COLOCATION_CONFIG_ENV, config_digest)
    monkeypatch.setenv(COLOCATION_PLAN_ENV, plan_digest)

    call = {
        "sample_id": ["s_vm", "s_v"],
        "group_id": ["g", "g"],
        "branch": ["vm", "v"],
        "rollout_id": [0, 1],
        "answer": ["A", "A"],
    }
    vm_v_group_bonus_rewards(["<answer>A</answer>", "<answer>A</answer>"], **call)
    with pytest.raises(RewardMetadataError, match="incomplete"):
        validate_runtime_colocation_receipt(
            path,
            nonce=nonce,
            dataset_digest=dataset_digest,
            config_digest=config_digest,
            plan_digest=plan_digest,
        )
    vm_v_group_bonus_rewards(["<answer>A</answer>", "<answer>A</answer>"], **call)
    validate_runtime_colocation_receipt(
        path,
        nonce=nonce,
        dataset_digest=dataset_digest,
        config_digest=config_digest,
        plan_digest=plan_digest,
    )
    with pytest.raises(RewardMetadataError, match="unexpected, duplicated"):
        vm_v_group_bonus_rewards(
            ["<answer>A</answer>", "<answer>A</answer>"], **call
        )


def test_runtime_colocation_requires_initialized_full_binding(tmp_path, monkeypatch):
    path = tmp_path / "missing_plan.json"
    monkeypatch.setenv(COLOCATION_PATH_ENV, str(path))
    monkeypatch.setenv(COLOCATION_NONCE_ENV, "a" * 32)
    monkeypatch.setenv(COLOCATION_DATASET_ENV, "b" * 64)
    monkeypatch.setenv(COLOCATION_CONFIG_ENV, "c" * 64)
    monkeypatch.setenv(COLOCATION_PLAN_ENV, "d" * 64)
    with pytest.raises(RewardMetadataError, match="not initialized"):
        vm_v_group_bonus_rewards(
            ["<answer>A</answer>", "<answer>A</answer>"],
            sample_id=["s_vm", "s_v"],
            group_id=["g", "g"],
            branch=["vm", "v"],
            rollout_id=[0, 1],
            answer=["A", "A"],
        )
