from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from motionllm.grpo import (
    RewardMetadataError,
    build_reward_metadata_batch,
    format_reward,
    semantic_reward,
    vm_v_group_bonus_rewards,
)
from qwenvl.grpo_ms_swift.runner import train_grpo_ms_swift as grpo_runner


@pytest.mark.parametrize(
    "reasoning",
    [
        "!!! ??? ...",
        "💩 🤡 🚀",
        "---___===",
        "\u200b\u200c\u200d",
    ],
)
def test_format_reward_rejects_nonempty_but_lexically_empty_think(
    reasoning: str,
) -> None:
    completion = f"<think>{reasoning}</think><answer>A</answer>"
    assert format_reward(completion) == 0.0


def test_semantic_reward_never_awards_two_unrelated_garbage_thinks() -> None:
    generated = "<think>!!! 💩 ???</think><answer>A</answer>"
    reference = "<think>### 🤡 ...</think><answer>A</answer>"

    assert semantic_reward(generated, reference) == 0.0


@pytest.mark.parametrize("failure_kind", ["called_process", "os_error", "unexpected"])
def test_runner_failure_boundary_never_renders_raw_argv_secrets(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bearer = "Bearer adversarial-secret-token"
    token_only_uri = "https://token-only-secret@example.invalid/v1"
    command = [
        "swift",
        "--api_key",
        bearer,
        "--vllm_server_base_url",
        token_only_uri,
        "--model",
        "safe-public-model-name",
    ]
    raw_argv_repr = repr(command)
    raw_shell_command = shlex.join(command)

    def fail(actual_command: list[str], **_: object) -> None:
        assert actual_command == command
        detail = f"raw subprocess detail: {bearer}; endpoint={token_only_uri}"
        if failure_kind == "called_process":
            raise subprocess.CalledProcessError(
                9,
                actual_command,
                output=detail,
                stderr=raw_argv_repr,
            )
        if failure_kind == "os_error":
            raise OSError(detail)
        raise RuntimeError(f"{detail}; argv={raw_shell_command}")

    monkeypatch.setattr(grpo_runner.subprocess, "run", fail)

    with pytest.raises(RuntimeError) as captured:
        grpo_runner._run_swift_safely(command, env={})

    streams = capsys.readouterr()
    rendered = "\n".join(
        (streams.out, streams.err, str(captured.value), repr(captured.value))
    )
    for forbidden in (bearer, token_only_uri, raw_argv_repr, raw_shell_command):
        assert forbidden not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def _grpo_row(index: int, branch: str) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": f"sample_{branch}_{index}",
        "group_id": "ambiguous_group",
        "branch": branch,
        "rollout_id": index,
        "answer": "A",
        "solution": "<think>visible body motion evidence</think><answer>A</answer>",
    }
    if branch == "vm":
        row["motion"] = f"motion_{index}.npy"
    return row


def test_dataset_preflight_rejects_ambiguous_two_vm_two_v_base_rows(
    tmp_path: Path,
) -> None:
    # Generation fan-out may create 2 VM + 2 V completions at runtime.  The
    # frozen *dataset*, however, must not smuggle two unrelated base pairs into
    # one group without an explicit pair identity.
    rows = [
        _grpo_row(0, "vm"),
        _grpo_row(1, "vm"),
        _grpo_row(2, "v"),
        _grpo_row(3, "v"),
    ]
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match=r"(?i)(exactly one|ambiguous|pair)"):
        grpo_runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_large_generation_fanout_has_unique_ids_and_stable_alignment() -> None:
    generation_count = 64
    size = 2 * generation_count
    metadata = build_reward_metadata_batch(
        size,
        sample_id=["sample_vm", "sample_v"],
        group_id=["group", "group"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
        solution=[
            "<think>motion evidence</think><answer>A</answer>",
            "<think>video evidence</think><answer>A</answer>",
        ],
        num_generations=generation_count,
    )

    assert len(metadata) == size
    assert len({item.rollout_key for item in metadata}) == size
    assert [item.generation_id for item in metadata] == list(range(size))

    completions = ["<answer>A</answer>"] * size
    bonuses = vm_v_group_bonus_rewards(
        completions,
        sample_id=["sample_vm", "sample_v"],
        group_id=["group", "group"],
        branch=["vm", "v"],
        rollout_id=[0, 1],
        answer=["A", "A"],
        num_generations=generation_count,
    )
    assert bonuses == [0.1] * generation_count + [0.0] * generation_count


@pytest.mark.parametrize(
    "num_generations",
    [[2, 3], [2, True], 0, -1, True, "2"],
)
def test_generation_count_ambiguity_fails_closed(num_generations: object) -> None:
    with pytest.raises(RewardMetadataError):
        build_reward_metadata_batch(
            4,
            sample_id=["sample_vm", "sample_v"],
            group_id=["group", "group"],
            branch=["vm", "v"],
            rollout_id=[0, 1],
            answer=["A", "A"],
            num_generations=num_generations,
        )


def _copy_formal_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    source = grpo_runner.REPO_ROOT / "tests" / "fixtures" / "grpo"
    destination = tmp_path / "grpo"
    shutil.copytree(source, destination)
    return (
        destination / "formal_vm_v_train.jsonl",
        destination / "formal_vm_v_validation.jsonl",
    )


def _formal_data_config(train: Path, validation: Path) -> dict[str, object]:
    return {
        "run": {"dataset_precheck": True},
        "data": {
            "dataset": [str(train)],
            "val_dataset": [str(validation)],
            "load_from_cache_file": False,
        },
        "rewards": {"reward_funcs": ["motion_semantic"]},
    }


def test_formal_media_binding_rejects_conflicting_video_injection(tmp_path: Path) -> None:
    train, validation = _copy_formal_fixture_tree(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows[0]["video"] = "media/injected_second_video.fixture"
    train.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record schema differs.*video"):
        grpo_runner._precheck_dataset_records(
            _formal_data_config(train, validation), formal_artifact=True
        )


def test_formal_media_binding_rejects_hash_valid_symlink(tmp_path: Path) -> None:
    train, validation = _copy_formal_fixture_tree(tmp_path)
    video = train.parent / "media" / "train_video.fixture"
    target = train.parent / "media" / "validation_video.fixture"
    video.unlink()
    try:
        os.symlink(target, video)
    except OSError as exc:
        pytest.skip(f"test host cannot create symlinks: {exc}")
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    target_digest = "43cf271602610f277d6ccb0271a058e503dacb1ab35d02edd7094726a6c433b8"
    for row in rows:
        row["video_sha256"] = target_digest
    train.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not traverse symlinks"):
        grpo_runner._precheck_dataset_records(
            _formal_data_config(train, validation), formal_artifact=True
        )


@pytest.mark.parametrize(
    "media_type, field", [("image", "image"), ("audio", "audio"), ("file", "file")]
)
def test_formal_media_binding_rejects_unhashed_extra_media(
    tmp_path: Path, media_type: str, field: str
) -> None:
    train, validation = _copy_formal_fixture_tree(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows[0]["messages"][0]["content"].append(
        {"type": media_type, field: "media/unhashed-extra.bin"}
    )
    train.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forbids content type"):
        grpo_runner._precheck_dataset_records(
            _formal_data_config(train, validation), formal_artifact=True
        )


def test_formal_normalized_prompt_leakage_rejects_new_ids_and_media(tmp_path: Path) -> None:
    train, validation = _copy_formal_fixture_tree(tmp_path)
    train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    validation_rows = [
        json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()
    ]
    for index, row in enumerate(validation_rows):
        copied_text = train_rows[index]["messages"][0]["content"][-1]["text"]
        copied_text = copied_text.replace("train_group_001", "validation_group_001")
        row["messages"][0]["content"][-1]["text"] = copied_text
        row["solution"] = train_rows[index]["solution"]
        row["answer"] = "A"
    validation.write_text(
        "\n".join(json.dumps(row) for row in validation_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalized_prompt_sha256"):
        grpo_runner._precheck_dataset_records(
            _formal_data_config(train, validation), formal_artifact=True
        )
