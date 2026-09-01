from __future__ import annotations

import json
import importlib
import inspect
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from motionllm.grpo import COLOCATION_PLAN_ENV
from motionllm.training import validate_artifact_policy
from qwenvl.grpo_ms_swift.runner import train_grpo_ms_swift as runner


FIXTURE_ROOT = runner.REPO_ROOT / "tests" / "fixtures" / "grpo"


def write_test_adapter_config(
    destination,
    *,
    base_model_name_or_path=None,
    task_type=None,
    target_modules=("layer",),
    modules_to_save=("extra",),
    r=1,
    lora_alpha=2,
    lora_dropout=0.05,
):
    config = LoraConfig(
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save),
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=task_type,
        bias="none",
        use_rslora=False,
        use_dora=False,
    )
    config.base_model_name_or_path = (
        str(base_model_name_or_path) if base_model_name_or_path is not None else None
    )
    payload = config.to_dict()
    payload["inference_mode"] = True
    for key, value in tuple(payload.items()):
        if isinstance(value, set):
            payload[key] = sorted(value)
    destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_test_swift_extension(
    destination,
    *,
    lora_dtype=None,
    lorap_lr_ratio=None,
    lorap_emb_lr=1.0e-6,
):
    destination.write_text(
        json.dumps(
            {
                "lora_dtype": lora_dtype,
                "lorap_lr_ratio": lorap_lr_ratio,
                "lorap_emb_lr": lorap_emb_lr,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def pair_rows():
    return [
        {
            "sample_id": "sample_vm",
            "group_id": "group_1",
            "branch": "vm",
            "rollout_id": 0,
            "answer": "A",
            "solution": "<think>motion evidence</think><answer>A</answer>",
            "motion": "motion.npy",
        },
        {
            "sample_id": "sample_v",
            "group_id": "group_1",
            "branch": "v",
            "rollout_id": 1,
            "answer": "A",
            "solution": "<answer>A</answer>",
        },
    ]


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_runner_preflight_validates_complete_local_json_formats(tmp_path, suffix):
    path = tmp_path / f"train{suffix}"
    rows = pair_rows()
    if suffix == ".json":
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_runner_preflight_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"sample_id":"s","sample_id":"other","group_id":"g",'
        '"branch":"v","rollout_id":0,"answer":"A"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "solution": "<answer>B</answer>"}), "Invalid reward metadata"),
        (lambda rows: rows.pop(), "co-locate exactly vm and v"),
        (lambda rows: rows[0].pop("motion"), "VM sample missing motion"),
        (lambda rows: rows[1].__setitem__("motion", "forbidden.npy"), "V sample must not include motion"),
        (
            lambda rows: rows.__setitem__(
                1, {**rows[1], "answer": "B", "solution": "<answer>B</answer>"}
            ),
            "conflicting gold answers",
        ),
    ],
)
def test_runner_preflight_rejects_invalid_pair_contracts(tmp_path, mutator, message):
    rows = pair_rows()
    mutator(rows)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_runner_preflight_rejects_duplicate_or_conflicting_rollout_identity(tmp_path):
    rows = pair_rows()
    rows.append({**rows[1]})
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate rollout identity"):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})

    rows = pair_rows()
    rows.append({**rows[1], "group_id": "other"})
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting rollout identity"):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_runner_preserves_motion_and_pixel_model_options():
    command = runner._build_command(
        {
            "model": {
                "model": "base",
                "min_pixels": 784,
                "max_pixels": 50176,
                "video_min_frames": 4,
                "video_max_frames": 32,
                "video_fps": 2,
                "motion_length_divisor": 4,
                "motion_timestamps_sync_with_video": True,
            },
            "data": {"dataset": ["hub/name"]},
            "rewards": {"reward_funcs": ["motion_option_accuracy"]},
        }
    )
    rendered = " ".join(command)
    for flag in (
        "--min_pixels 784",
        "--max_pixels 50176",
        "--video_min_frames 4",
        "--video_max_frames 32",
        "--video_fps 2",
        "--motion_length_divisor 4",
        "--motion_timestamps_sync_with_video true",
    ):
        assert flag in rendered


def test_grpo_formal_policy_cannot_omit_manifest(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")
    arguments = runner._artifact_arguments(
        {"run": {}, "model": {"model": "base"}},
        config_path=config_path,
    )
    with pytest.raises(ValueError, match="formal training requires"):
        validate_artifact_policy(arguments, training_mode="grpo")

    arguments = runner._artifact_arguments(
        {"run": {"unsafe_legacy_no_manifest": True}, "model": {"model": "base"}},
        config_path=config_path,
    )
    assert validate_artifact_policy(arguments, training_mode="grpo") is False


@pytest.mark.parametrize("suffix, contents", [(".json", "[]"), (".jsonl", "\n \n")])
def test_runner_preflight_rejects_empty_local_json_or_jsonl(tmp_path, suffix, contents):
    path = tmp_path / f"empty{suffix}"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="at least one record"):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_grpo_config_requires_one_explicit_deterministic_seed():
    config = {
        "run": {"colocation_expected_calls": 1},
        "model": {"model": "base"},
        "data": {"dataset": ["train.json"]},
        "rewards": {"reward_funcs": ["motion_option_accuracy"]},
    }
    with pytest.raises(ValueError, match="run.seed"):
        runner._validate_required(config)
    config["run"]["seed"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        runner._validate_required(config)
    config["run"]["seed"] = 123
    runner._validate_required(config)


def test_formal_preflight_cannot_be_disabled_or_truncated():
    with pytest.raises(ValueError, match="disabling"):
        runner._enforce_formal_preflight_policy(
            {"run": {"dataset_precheck": False}}, formal_artifact=True
        )
    with pytest.raises(ValueError, match="full dataset scan"):
        runner._enforce_formal_preflight_policy(
            {"run": {"dataset_precheck_max_samples": 1}}, formal_artifact=True
        )
    with pytest.raises(ValueError, match="full dataset scan"):
        runner._enforce_formal_preflight_policy(
            {"data": {"max_train_samples": 1}}, formal_artifact=True
        )


def test_semantic_preflight_requires_nonempty_strict_reasoning_solution(tmp_path):
    rows = pair_rows()
    rows[1]["solution"] = "<think>video evidence</think><answer>A</answer>"
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    config = {
        "data": {"dataset": [str(path)]},
        "rewards": {"reward_funcs": ["motion_semantic"]},
    }
    runner._precheck_dataset_records(config)
    for bad_solution in (None, "garbage <answer>A</answer>", "<think>!!!</think><answer>A</answer>"):
        bad_rows = pair_rows()
        bad_rows[0]["solution"] = bad_solution
        path.write_text(json.dumps(bad_rows), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid semantic solution"):
            runner._precheck_dataset_records(config)


def test_preflight_rejects_ambiguous_two_vm_two_v_prompts_per_group(tmp_path):
    rows = pair_rows()
    rows.extend(
        [
            {**rows[0], "sample_id": "sample_vm_2", "rollout_id": 2},
            {**rows[1], "sample_id": "sample_v_2", "rollout_id": 3},
        ]
    )
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one vm and one v"):
        runner._precheck_dataset_records({"data": {"dataset": [str(path)]}})


def test_formal_grpo_binds_real_dataset_base_and_motion_asset_paths(tmp_path):
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text(json.dumps(pair_rows()), encoding="utf-8")
    validation.write_text(json.dumps(pair_rows()), encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    vqvae = tmp_path / "motion.pth"
    vqvae.write_bytes(b"motion")
    config_path = tmp_path / "formal.yaml"
    config_path.write_text("formal: true\n", encoding="utf-8")
    config = {
        "data": {"dataset": [str(train)], "val_dataset": [str(validation)]},
        "model": {
            "model": str(base),
            "model_kwargs": {"vqvae_path": str(vqvae)},
        },
    }
    arguments = SimpleNamespace(
        train_data_path=str(train),
        validation_data_path=str(validation),
        base_artifact_path=str(base),
        motion_vqvae_asset_path=str(vqvae),
        config_path=str(config_path),
    )
    assert runner._bind_formal_loader_inputs(
        config, arguments, config_path=config_path
    ) == train.resolve()

    config["data"]["dataset"] = [str(validation)]
    with pytest.raises(ValueError, match="data.dataset"):
        runner._bind_formal_loader_inputs(config, arguments)
    config["data"]["dataset"] = [str(train)]
    other = tmp_path / "other.pth"
    other.write_bytes(b"other")
    config["model"]["model_kwargs"]["vqvae_path"] = str(other)
    with pytest.raises(ValueError, match="vqvae_path"):
        runner._bind_formal_loader_inputs(config, arguments)

    config["model"]["model_kwargs"]["vqvae_path"] = str(vqvae)
    validation.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one record"):
        runner._bind_formal_loader_inputs(
            config, arguments, config_path=config_path
        )

    validation.write_text(json.dumps(pair_rows()), encoding="utf-8")
    other_config = tmp_path / "other.yaml"
    other_config.write_text("formal: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executed GRPO config"):
        runner._bind_formal_loader_inputs(
            config, arguments, config_path=other_config
        )


def test_group_aware_batch_plan_overrides_per_device_batch_one_and_rejects_split_pair(tmp_path):
    rows = pair_rows()
    checked = [(tmp_path / "train.json", index, row) for index, row in enumerate(rows, 1)]
    config = {
        "data": {"dataset_shuffle": False},
        "training": {"per_device_train_batch_size": 1},
        "grpo": {"num_generations": 2, "generation_batch_size": 4},
        "rewards": {"reward_funcs": ["motion_vm_v_bonus"]},
    }
    assert runner._validate_group_aware_batching(config, checked) == (("group_1",),)
    config["grpo"]["generation_batch_size"] = 2
    with pytest.raises(ValueError, match="at least two prompts"):
        runner._validate_group_aware_batching(config, checked)
    config["grpo"]["generation_batch_size"] = 6
    with pytest.raises(ValueError, match="even prompt count"):
        runner._validate_group_aware_batching(config, checked)


def test_formal_launch_requires_fresh_runtime_colocation_receipt_destination(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps(pair_rows()), encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")
    checked = [(train, index, row) for index, row in enumerate(pair_rows(), 1)]
    config = {
        "run": {"colocation_expected_calls": 1},
        "data": {"dataset_shuffle": False},
        "training": {"per_device_train_batch_size": 1},
        "grpo": {"num_generations": 2, "generation_batch_size": 4},
        "rewards": {"reward_funcs": ["motion_vm_v_bonus"]},
    }
    arguments = SimpleNamespace(artifact_root=str(tmp_path))
    with pytest.raises(ValueError, match="colocation_receipt_path"):
        runner._prepare_colocation_launch(
            config,
            artifact_arguments=arguments,
            config_path=config_path,
            train_path=train,
            checked_rows=checked,
            launch_env={},
        )
    receipt = tmp_path / "colocation.json"
    receipt.write_text("stale", encoding="utf-8")
    config["run"]["colocation_receipt_path"] = str(receipt)
    with pytest.raises(ValueError, match="must be fresh"):
        runner._prepare_colocation_launch(
            config,
            artifact_arguments=arguments,
            config_path=config_path,
            train_path=train,
            checked_rows=checked,
            launch_env={},
        )

    receipt.unlink()
    launch_env = {}
    binding = runner._prepare_colocation_launch(
        config,
        artifact_arguments=arguments,
        config_path=config_path,
        train_path=train,
        checked_rows=checked,
        launch_env=launch_env,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "runtime_colocation_pending"
    assert payload["plan"]["planned_call_count"] == 1
    assert payload["plan"]["planned_rollout_count"] == 4
    assert launch_env[COLOCATION_PLAN_ENV] == binding.plan_digest


def test_formal_grpo_output_is_fresh_exact_and_keeps_receipts_outside_artifact(tmp_path):
    output = tmp_path / "adapter"
    arguments = SimpleNamespace(
        artifact_root=str(tmp_path),
        reload_receipt_path=str(tmp_path / "reload.json"),
        artifact_manifest_path=str(tmp_path / "manifest.json"),
    )
    config = {
        "run": {
            "output_dir": str(output),
            "artifact_path": str(output),
            "add_version": False,
            "colocation_receipt_path": str(tmp_path / "colocation.json"),
        }
    }
    runner._validate_formal_output_destinations(config, arguments)

    config["run"]["artifact_path"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="explicit descendant"):
        runner._validate_formal_output_destinations(config, arguments)
    config["run"]["artifact_path"] = str(output)

    output.mkdir()
    (output / "stale.bin").write_bytes(b"stale")
    with pytest.raises(ValueError, match="fresh and empty"):
        runner._validate_formal_output_destinations(config, arguments)

    (output / "stale.bin").unlink()
    arguments.reload_receipt_path = str(output / "reload.json")
    with pytest.raises(ValueError, match="outside the artifact"):
        runner._validate_formal_output_destinations(config, arguments)


def test_subprocess_failure_never_exposes_raw_argv_secrets(monkeypatch):
    secret = "adversarial-secret-token"
    command = [
        "swift",
        "--api_key",
        secret,
        "--vllm_server_base_url",
        f"https://{secret}@example.invalid/v1",
    ]

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as captured:
        runner._run_swift_safely(command, env={})
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None


def test_grpo_reload_receipt_is_generated_from_fresh_state_before_publication(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_model.bin").write_bytes(b"adapter")
    arguments = SimpleNamespace(
        reload_receipt_path=str(tmp_path / "reload.json"),
        artifact_root=str(tmp_path),
        batch_id="batch_1",
        model_registry_id="model_1",
    )
    config = {
        "training": {"tuner_type": "lora", "modules_to_save": ["motion_proj"]}
    }
    monkeypatch.setattr(
        runner,
        "_load_grpo_lora_states",
        lambda config, artifact_path: (
            {"adapter.lora_A.weight": [1.0], "adapter.lora_B.weight": [2.0]},
            {"adapter.lora_A.weight": [1.0], "adapter.lora_B.weight": [2.0]},
            (10, 11),
            ("c" * 64, "c" * 64, "d" * 64),
        ),
    )
    receipt_path = runner._generate_grpo_reload_receipt(config, arguments, artifact)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "reload_verified"
    assert payload["artifact_hash"]

    receipt_path.unlink()
    monkeypatch.setattr(
        runner,
        "_load_grpo_lora_states",
        lambda config, artifact_path: (
            {"adapter.lora_A.weight": [1.0], "adapter.lora_B.weight": [2.0]},
            {"adapter.lora_A.weight": [99.0], "adapter.lora_B.weight": [2.0]},
            (10, 11),
            ("c" * 64, "c" * 64, "d" * 64),
        ),
    )
    with pytest.raises(Exception, match="content differs"):
        runner._generate_grpo_reload_receipt(config, arguments, artifact)


def test_formal_reward_policy_requires_semantic_format_and_group_rewards():
    good = {
        "rewards": {
            "reward_funcs": ["motion_semantic", "motion_format", "motion_vm_v_bonus"],
            "reward_weights": [1.0, 0.5, 0.1],
        }
    }
    runner._validate_formal_reward_policy(good)
    for names in (
        ["motion_format", "motion_vm_v_bonus"],
        ["motion_semantic", "motion_vm_v_bonus"],
        ["motion_semantic", "motion_format"],
        ["motion_semantic", "motion_format", "motion_vm_v_bonus", "motion_format"],
    ):
        with pytest.raises(ValueError, match="exactly|unique"):
            runner._validate_formal_reward_policy(
                {"rewards": {"reward_funcs": names}}
            )


@pytest.mark.parametrize(
    "name",
    [
        "VQVAE_PATH",
        "GROUP_NUM_SAMPLES",
        "MOTION_GRPO_VM_V_THRESHOLD",
        "MOTION_GRPO_BATCH_ID",
        "MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS",
        "MOTION_GRPO_ARTIFACT_PATH",
    ],
)
def test_formal_grpo_rejects_ambient_vq_and_reward_overrides(monkeypatch, name):
    monkeypatch.setenv(name, "unexpected")
    with pytest.raises(ValueError, match="ambient"):
        runner._enforce_formal_ambient_environment({"run": {}})
    runner._enforce_formal_ambient_environment({"run": {"env": {name: None}}})


def test_grpo_modules_to_save_must_be_explicit_and_present_in_adapter_state():
    with pytest.raises(ValueError, match="modules_to_save"):
        runner._expected_modules_to_save({"training": {"tuner_type": "lora"}})
    with pytest.raises(Exception, match="missing expected modules_to_save"):
        runner._require_modules_in_adapter_state(
            {
                "base.q_proj.lora_A.default.weight": [1.0],
                "base.q_proj.lora_B.default.weight": [2.0],
            },
            ("motion_proj",),
        )


def test_formal_full_grpo_publication_fails_without_independent_disk_proof(tmp_path):
    with pytest.raises(Exception, match="independent raw-disk producer"):
        runner._load_grpo_full_states({}, tmp_path)


def _copied_formal_fixtures(tmp_path):
    destination = tmp_path / "grpo"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination / "formal_vm_v_train.jsonl", destination / "formal_vm_v_validation.jsonl"


def _formal_fixture_data_config(train, validation):
    return {
        "run": {"dataset_precheck": True},
        "data": {
            "dataset": [str(train)],
            "val_dataset": [str(validation)],
            "load_from_cache_file": False,
        },
        "rewards": {
            "reward_funcs": ["motion_semantic", "motion_format", "motion_vm_v_bonus"]
        },
    }


def test_formal_fixtures_scan_train_and_validation_media_hashes(tmp_path):
    train, validation = _copied_formal_fixtures(tmp_path)
    checked = runner._precheck_dataset_records(
        _formal_fixture_data_config(train, validation), formal_artifact=True
    )
    assert len(checked) == 2
    assert all(path == train.resolve() for path, _, _ in checked)


@pytest.mark.parametrize("attack", ["tamper_bytes", "uppercase_digest", "missing_digest"])
def test_formal_media_hash_attacks_fail_closed(tmp_path, attack):
    train, validation = _copied_formal_fixtures(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    if attack == "tamper_bytes":
        (train.parent / "media" / "train_video.fixture").write_bytes(b"tampered")
    elif attack == "uppercase_digest":
        rows[0]["video_sha256"] = rows[0]["video_sha256"].upper()
        train.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    else:
        rows[0].pop("motion_sha256")
        train.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256|sha256"):
        runner._precheck_dataset_records(
            _formal_fixture_data_config(train, validation), formal_artifact=True
        )


def test_formal_validation_schema_is_fully_scanned(tmp_path):
    train, validation = _copied_formal_fixtures(tmp_path)
    rows = [json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()]
    rows[1].pop("messages")
    validation.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record schema differs.*messages"):
        runner._precheck_dataset_records(
            _formal_fixture_data_config(train, validation), formal_artifact=True
        )


def test_formal_train_validation_media_identity_leakage_fails_closed(tmp_path):
    train, validation = _copied_formal_fixtures(tmp_path)
    train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    validation_rows = [
        json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()
    ]
    for row in validation_rows:
        row["messages"][0]["content"][0]["video"] = "media/train_video.fixture"
        row["video_sha256"] = train_rows[0]["video_sha256"]
    validation.write_text(
        "\n".join(json.dumps(row) for row in validation_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity leakage for video_sha256"):
        runner._precheck_dataset_records(
            _formal_fixture_data_config(train, validation), formal_artifact=True
        )


def test_legacy_full_configs_are_enumerated_and_rejected():
    config_root = runner.REPO_ROOT / "configs" / "grpo"
    quarantine_root = (
        runner.REPO_ROOT
        / "legacy"
        / "refactor_snapshot"
        / "configs"
        / "grpo_personal"
    )
    inventory = json.loads(
        (config_root / "legacy_full_inventory.json").read_text(encoding="utf-8")
    )
    root_yamls = {path.name for path in quarantine_root.glob("*.yaml")}
    assert set(inventory["configs"]) == root_yamls
    assert inventory["config_count"] == len(root_yamls) == 22
    for name in root_yamls:
        config = runner._read_yaml(quarantine_root / name)
        with pytest.raises(ValueError, match="quarantined"):
            runner._validate_grpo_training_mode(config)

    formal = runner._read_yaml(
        config_root / "formal" / "motionr1_vm_lora.template.yaml"
    )
    runner._validate_grpo_training_mode(formal)
    assert formal["run"]["add_version"] is False
    assert formal["training"]["tuner_type"] == "lora"
    assert formal["training"]["modules_to_save"] == ["motion_prenorm", "motion_proj"]


def test_explicit_descendant_adapter_leaf_is_verified(tmp_path):
    output = tmp_path / "output"
    artifact = output / "checkpoint-1"
    artifact.mkdir(parents=True)
    write_test_adapter_config(artifact / "adapter_config.json")
    write_test_swift_extension(artifact / "additional_config.json")
    save_safetensors(
        {
            "base.layer.lora_A.weight": torch.ones(1, 2),
            "base.layer.lora_B.weight": torch.ones(2, 1),
        },
        str(artifact / "adapter_model.safetensors"),
    )
    config = {
        "run": {
            "output_dir": str(output),
            "artifact_path": str(artifact),
            "add_version": False,
        }
    }
    arguments = SimpleNamespace(artifact_root=str(tmp_path))
    assert runner._resolve_grpo_artifact_path(config, arguments) == artifact.resolve()
    (artifact / "adapter_model.bin").write_bytes(b"second-format")
    with pytest.raises(ValueError, match="forbids adapter_model.bin"):
        runner._resolve_grpo_artifact_path(config, arguments)
    (artifact / "adapter_model.bin").unlink()
    (artifact / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="adapter_config"):
        runner._resolve_grpo_artifact_path(config, arguments)


class _ReceiptTinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.extra = torch.nn.Linear(2, 1)


def test_runner_receipt_binds_current_exact_adapter_payload(tmp_path):
    model = get_peft_model(
        _ReceiptTinyBase(),
        LoraConfig(
            target_modules=["proj"],
            modules_to_save=["extra"],
            r=1,
            lora_alpha=2,
        ),
    )
    with torch.no_grad():
        for tensor in get_peft_model_state_dict(model).values():
            tensor.add_(2.0)
    artifact = tmp_path / "checkpoint-1"
    model.save_pretrained(artifact, safe_serialization=True)
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    write_test_adapter_config(
        artifact / "adapter_config.json",
        base_model_name_or_path=base_model,
        task_type="CAUSAL_LM",
        target_modules=("proj",),
        modules_to_save=("extra",),
        r=1,
        lora_alpha=2,
        lora_dropout=0.0,
    )
    write_test_swift_extension(artifact / "additional_config.json")
    binding = runner.capture_adapter_checkpoint(
        artifact, require_swift_extension=True
    )
    assert binding.extension_config is not None
    nonce = "c" * 64
    payload = {
        "schema": runner.FORMAL_TRAINING_RECEIPT_SCHEMA,
        "status": "optimizer_and_checkpoint_verified",
        "nonce": nonce,
        "batch_id": "batch_current",
        "model_registry_id": runner.FORMAL_MODEL_REGISTRY_ID,
        "expected_optimizer_steps": 1,
        "observed_global_step": 1,
        "optimizer_step_count": 1,
        "finite_loss_count": 1,
        "gradient_observation_count": 1,
        "trainable_tensor_count": 4,
        "changed_tensor_count": 4,
        "initial_trainable_state_sha256": "a" * 64,
        "final_trainable_state_sha256": "b" * 64,
        "adapter_filename": binding.filename,
        "adapter_payload_sha256": binding.payload_sha256,
        "adapter_payload_size_bytes": binding.payload_size_bytes,
        "adapter_tensor_count": binding.tensor_count,
        "frozen_embedding_tensor_count": 0,
        "final_saveable_adapter_state_sha256": binding.state_sha256,
        "adapter_config_filename": binding.config.filename,
        "adapter_config_payload_sha256": binding.config.payload_sha256,
        "adapter_config_payload_size_bytes": binding.config.payload_size_bytes,
        "adapter_config_semantic_sha256": binding.config.semantic_sha256,
        "adapter_config_critical": runner.adapter_config_critical_fields(
            binding.config.semantics
        ),
        "adapter_extension_filename": binding.extension_config.filename,
        "adapter_extension_payload_sha256": binding.extension_config.payload_sha256,
        "adapter_extension_payload_size_bytes": binding.extension_config.payload_size_bytes,
        "adapter_extension_semantic_sha256": binding.extension_config.semantic_sha256,
        "adapter_extension_semantics": binding.extension_config.semantics,
        "max_abs_gradient": 1.0,
        "max_abs_delta": 0.5,
    }
    receipt_path = artifact / "grpo_training_receipt.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "run": {"expected_optimizer_steps": 1},
        "model": {"model": str(base_model)},
        "training": {
            "lora_rank": 1,
            "lora_alpha": 2,
            "lora_dropout": 0.0,
            "modules_to_save": ["extra"],
            "lora_bias": "none",
            "use_rslora": False,
            "use_dora": False,
            "lora_dtype": None,
            "lorap_lr_ratio": None,
            "lorap_emb_lr": 1.0e-6,
        },
    }
    arguments = SimpleNamespace(batch_id="batch_current")

    observed = runner._validate_grpo_training_update_receipt(
        config, arguments, artifact, nonce=nonce
    )
    assert observed.payload_sha256 == binding.payload_sha256

    for key in (
        "expected_optimizer_steps",
        "observed_global_step",
        "optimizer_step_count",
        "finite_loss_count",
        "gradient_observation_count",
    ):
        original = payload[key]
        for attacked_value in (0, 2, True):
            payload[key] = attacked_value
            receipt_path.write_text(json.dumps(payload), encoding="utf-8")
            with pytest.raises(ValueError, match=r"training.*receipt"):
                runner._validate_grpo_training_update_receipt(
                    config, arguments, artifact, nonce=nonce
                )
        payload[key] = original
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    payload["adapter_payload_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the current exact"):
        runner._validate_grpo_training_update_receipt(
            config, arguments, artifact, nonce=nonce
        )

    payload["adapter_payload_sha256"] = binding.payload_sha256
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = artifact / "adapter_config.json"
    original_config = config_path.read_text(encoding="utf-8")
    attacked_config = json.loads(original_config)
    attacked_config["lora_alpha"] = 4
    config_path.write_text(json.dumps(attacked_config, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the current exact"):
        runner._validate_grpo_training_update_receipt(
            config, arguments, artifact, nonce=nonce
        )
    config_path.write_text(original_config, encoding="utf-8")

    extension_path = artifact / "additional_config.json"
    original_extension = extension_path.read_text(encoding="utf-8")
    extension_path.write_text(
        json.dumps(
            {
                "lora_dtype": None,
                "lorap_lr_ratio": None,
                "lorap_emb_lr": 2.0e-6,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not bind the current exact"):
        runner._validate_grpo_training_update_receipt(
            config, arguments, artifact, nonce=nonce
        )
    extension_path.write_text(original_extension, encoding="utf-8")

    stale = load_safetensors(
        str(artifact / "adapter_model.safetensors"), device="cpu"
    )
    first_key = sorted(stale)[0]
    stale[first_key] = stale[first_key] + 9.0
    save_safetensors(stale, str(artifact / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="does not bind the current exact"):
        runner._validate_grpo_training_update_receipt(
            config, arguments, artifact, nonce=nonce
        )


def test_formal_output_rejects_implicit_swift_versioning(tmp_path):
    arguments = SimpleNamespace(
        artifact_root=str(tmp_path),
        reload_receipt_path=str(tmp_path / "reload.json"),
        artifact_manifest_path=str(tmp_path / "manifest.json"),
    )
    config = {
        "run": {
            "output_dir": str(tmp_path / "output"),
            "artifact_path": str(tmp_path / "output" / "checkpoint-1"),
            "colocation_receipt_path": str(tmp_path / "colocation.json"),
        }
    }
    with pytest.raises(ValueError, match="add_version=false"):
        runner._validate_formal_output_destinations(config, arguments)


def test_runtime_missing_swift_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "_require_bound_interpreter", lambda env: runner.Path(sys.executable)
    )
    monkeypatch.setattr(
        runner,
        "_require_same_environment_swift",
        lambda env, python_path: tmp_path / "swift",
    )

    def missing(name):
        if name == "swift":
            raise RuntimeError(
                "GRPO runtime preflight failed before launch: cannot import swift (ImportError)"
            )
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_import_runtime_module", missing)
    with pytest.raises(RuntimeError, match="before launch.*swift"):
        runner._validate_runtime_environment(
            {}, SimpleNamespace(), formal_artifact=False, env={}
        )


def test_formal_runtime_contract_rejects_provenance_environment_drift(tmp_path):
    contract_path = (
        runner.REPO_ROOT
        / "qwenvl"
        / "grpo_ms_swift"
        / "runtime"
        / "grpo_api_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    environment = tmp_path / "environment.freeze.txt"
    pins = dict(contract["packages"])
    pins["peft"] = "0.0.0"
    environment.write_text(
        "\n".join(f"{name}=={version}" for name, version in pins.items()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="environment snapshot pin differs for peft"):
        runner._load_formal_runtime_contract(
            {"run": {"runtime_contract_path": str(contract_path)}},
            SimpleNamespace(environment_path=str(environment)),
        )


def test_checkpoint_binding_runtime_api_signatures_match_exact_pins():
    contract = json.loads(
        runner.FORMAL_RUNTIME_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    assert contract["packages"]["accelerate"] == "1.13.0"
    assert contract["packages"]["peft"] == "0.18.0"
    assert contract["packages"]["safetensors"] == "0.8.0"
    selected = {"accelerate", "peft", "peft.utils.save_and_load", "safetensors"}
    for check in contract["api_checks"]:
        if check["module"] not in selected:
            continue
        module = importlib.import_module(check["module"])
        for name, expected in check.get("signatures", {}).items():
            target = runner._runtime_api_attribute(module, name)
            assert list(inspect.signature(target).parameters) == expected


def test_formal_template_resolves_through_complete_dry_run(tmp_path, monkeypatch, capsys):
    template_path = (
        runner.REPO_ROOT
        / "configs"
        / "grpo"
        / "formal"
        / "motionr1_vm_lora.template.yaml"
    )
    config = runner._read_yaml(template_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    vqvae = tmp_path / "vqvae.pth"
    vqvae.write_bytes(b"vqvae")
    pretrained_registry = json.loads(
        runner.FORMAL_PRETRAINED_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    pretrained_inventory = tmp_path / "pretrained_inventory.json"
    base_receipt = runner.hash_path(base, symlink_policy="follow")
    vq_receipt = runner.hash_path(vqvae, symlink_policy="follow")
    pretrained_inventory.write_text(
        json.dumps(
            {
                "all_pretrain_assets_ready": True,
                "pretrain_root": pretrained_registry["remote_root"],
                "models": [
                    {
                        "id": runner.FORMAL_MODEL_REGISTRY_ID,
                        "pretrain_asset_ready": True,
                        "artifacts": [
                            {
                                "role": "base_and_processor",
                                "status": "valid",
                                "tree_sha256": base_receipt.digest,
                                "selected_file_count": base_receipt.file_count,
                            },
                            {
                                "role": "motion_vqvae",
                                "status": "valid",
                                "sha256": vq_receipt.digest,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner, "FORMAL_PRETRAINED_INVENTORY_PATH", pretrained_inventory
    )
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps({"sample_id": "qa500_001", "question": "A frozen question?"}) + "\n",
        encoding="utf-8",
    )
    leakage = tmp_path / "leakage.json"
    leakage.write_text("{}", encoding="utf-8")
    contract = json.loads(
        (
            runner.REPO_ROOT
            / "qwenvl"
            / "grpo_ms_swift"
            / "runtime"
            / "grpo_api_contract.json"
        ).read_text(encoding="utf-8")
    )
    environment = tmp_path / "environment.freeze.txt"
    environment.write_text(
        "\n".join(f"{name}=={version}" for name, version in contract["packages"].items())
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "formal.yaml"
    output = tmp_path / "artifacts" / "output"
    artifact = output / "checkpoint-1"
    train = FIXTURE_ROOT / "formal_vm_v_train.jsonl"
    validation = FIXTURE_ROOT / "formal_vm_v_validation.jsonl"
    config["run"].update(
        {
            "batch_id": "batch_fixture_001",
            "expected_optimizer_steps": 1,
            "colocation_expected_calls": 1,
            "run_name": "batch_fixture_001-motionr1-vm-grpo-lora",
            "artifact_root": str(tmp_path),
            "output_dir": str(output),
            "artifact_path": str(artifact),
            "artifact_manifest_path": str(tmp_path / "receipts" / "manifest.json"),
            "reload_receipt_path": str(tmp_path / "receipts" / "reload.json"),
            "colocation_receipt_path": str(tmp_path / "receipts" / "colocation.json"),
        }
    )
    config["model"]["model"] = str(base)
    config["model"]["model_kwargs"]["vqvae_path"] = str(vqvae)
    config["data"]["dataset"] = [str(train)]
    config["data"]["val_dataset"] = [str(validation)]
    config["training"]["max_steps"] = 1
    config["training"]["save_steps"] = 1
    config["provenance"].update(
        {
            "base_artifact_path": str(base),
            "train_data_path": str(train),
            "validation_data_path": str(validation),
            "benchmark_path": str(benchmark),
            "leakage_audit_path": str(leakage),
            "config_path": str(config_path),
            "code_path": str(runner.REPO_ROOT),
            "environment_path": str(environment),
            "motion_vqvae_asset_path": str(vqvae),
        }
    )
    leakage.write_text(
        json.dumps(
            {
                "schema": runner.FORMAL_LEAKAGE_SCHEMA,
                "status": "passed",
                "batch_id": "batch_fixture_001",
                "normalization_algorithm": runner.FORMAL_NORMALIZATION_ALGORITHM,
                "train_sha256": runner.hash_path(train, symlink_policy="reject").digest,
                "validation_sha256": runner.hash_path(
                    validation, symlink_policy="reject"
                ).digest,
                "benchmark_sha256": runner.hash_path(
                    benchmark, symlink_policy="reject"
                ).digest,
                "train_rows": 2,
                "validation_rows": 2,
                "benchmark_rows": 1,
                "overlap_counts": {
                    "sample_id": 0,
                    "group_id": 0,
                    "media_sha256": 0,
                    "normalized_prompt": 0,
                    "normalized_solution": 0,
                    "near_duplicate": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    assert runner.yaml is not None
    config_path.write_text(runner.yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.delenv(runner.FORMAL_BOUND_INPUTS_ENV, raising=False)
    for name in runner._FORMAL_DANGEROUS_ENV_KEYS - {"PATH"}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["train_grpo_ms_swift.py", "--config", str(config_path), "--dry_run"]
    )
    runner.main()
    rendered = capsys.readouterr().out
    assert "Dataset precheck passed: 4 records validated" in rendered
    assert "--tuner_type lora" in rendered
    assert "--add_version false" in rendered
    assert "--callbacks motion_training_receipt" in rendered
    assert "--save_safetensors true" in rendered
    (base / "config.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical Motion-R1 base"):
        runner._validate_canonical_pretrained_assets(
            runner._artifact_arguments(config, config_path=config_path)
        )


def test_grpo_launchers_bind_one_interpreter_and_quarantine_legacy_shortcuts():
    scripts = runner.REPO_ROOT / "scripts"
    formal = (scripts / "train_grpo_ms_swift.sh").read_text(encoding="utf-8")
    assert "MOTION_GRPO_PYTHON" in formal
    assert "conda activate" not in formal
    assert "python qwenvl" not in formal
    assert '"${MOTION_GRPO_PYTHON}" -I -B' in formal
    assert "PYTHONNOUSERSITE=1" in formal
    assert "PYTHONSAFEPATH=1" in formal
    assert "PYTHONDONTWRITEBYTECODE=1" in formal
    assert "--config must name an existing absolute" in formal
    for name in ("debug_grpo_ms_swift.sh", "train_grpo_motionx_real.sh"):
        legacy = (scripts / name).read_text(encoding="utf-8")
        assert "quarantined historical full-GRPO" in legacy
        assert "exit 64" in legacy


def test_yaml_and_cli_duplicates_fail_closed(tmp_path, monkeypatch):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("run:\n  seed: 1\n  seed: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML"):
        runner._read_yaml(duplicate)

    merge = tmp_path / "merge.yaml"
    merge.write_text("defaults: &d {seed: 1}\nrun:\n  <<: *d\n", encoding="utf-8")
    with pytest.raises(ValueError, match="merge keys"):
        runner._read_yaml(merge)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_grpo_ms_swift.py", "--config", "one.yaml", "--config", "two.yaml"],
    )
    with pytest.raises(SystemExit):
        runner._parse_args()


def test_cli_help_is_config_free_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_grpo_ms_swift.py", "--help"])

    with pytest.raises(SystemExit) as raised:
        runner._parse_args()

    assert raised.value.code == 0
    rendered = capsys.readouterr().out
    assert "--config" in rendered
    assert "--dry_run" in rendered
    assert "--preflight_only" in rendered


def test_isolated_help_ignores_malicious_pythonpath_sitecustomize(tmp_path):
    sentinel = tmp_path / "sitecustomize-ran"
    (tmp_path / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    env["PYTHONUSERBASE"] = str(tmp_path / "userbase")

    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(runner.Path(runner.__file__)), "--help"],
        cwd=str(runner.REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert not sentinel.exists()


def test_formal_swift_child_isolated_and_python_env_sanitized(tmp_path, monkeypatch):
    swift = tmp_path / "swift"
    swift.write_text("entrypoint", encoding="utf-8")
    command = runner._isolated_formal_swift_command(
        ["swift", "rlhf", "--rlhf_type", "grpo"],
        python_path=runner.Path(sys.executable),
        swift_path=swift,
    )
    assert command == [
        str(runner.Path(sys.executable).resolve()),
        "-I",
        "-B",
        str(swift.resolve()),
        "rlhf",
        "--rlhf_type",
        "grpo",
    ]

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64")
    launch_env, _ = runner._build_launch_env({}, formal_artifact=True)
    assert launch_env["PYTHONNOUSERSITE"] == "1"
    assert launch_env["PYTHONSAFEPATH"] == "1"
    assert launch_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert set(key for key in launch_env if key.upper().startswith("PYTHON")) == {
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONDONTWRITEBYTECODE",
    }
    assert launch_env["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"

    monkeypatch.setattr(runner.os, "environ", {"LD_LIBRARY_PATH": "/opt/cuda/lib64"})
    runner._enforce_formal_ambient_environment({})


def test_formal_model_and_code_identity_are_hard_bound(tmp_path):
    good = {
        "run": {"model_registry_id": runner.FORMAL_MODEL_REGISTRY_ID},
        "model": {
            "model_family": runner.FORMAL_MODEL_FAMILY,
            "model_type": runner.FORMAL_MODEL_TYPE,
        },
        "plugins": {
            "external_plugins": [str(runner.FORMAL_EXTERNAL_PLUGIN_PATH)],
            "custom_register_path": [str(runner.FORMAL_CUSTOM_REGISTER_PATH)],
        },
    }
    good["run"]["runtime_contract_path"] = str(runner.FORMAL_RUNTIME_CONTRACT_PATH)
    assert len(runner._validate_formal_model_identity(good)) == 64
    code_arguments = SimpleNamespace(
        code_path=str(runner.REPO_ROOT),
        runner_code_path=str(runner.Path(runner.__file__).resolve()),
    )
    runner._bind_formal_code_identity(
        good, code_arguments
    )

    for model_id in runner._FORMAL_REGISTRY_IDS - {runner.FORMAL_MODEL_REGISTRY_ID}:
        attacked = json.loads(json.dumps(good))
        attacked["run"]["model_registry_id"] = model_id
        with pytest.raises(ValueError, match="model_registry_id"):
            runner._validate_formal_model_identity(attacked)

    arbitrary = tmp_path / "plugin.py"
    arbitrary.write_text("# arbitrary\n", encoding="utf-8")
    attacked = json.loads(json.dumps(good))
    attacked["plugins"]["external_plugins"] = [str(arbitrary)]
    with pytest.raises(ValueError, match="approved file"):
        runner._bind_formal_code_identity(
            attacked, code_arguments
        )
    with pytest.raises(ValueError, match="actual frozen code root"):
        runner._bind_formal_code_identity(
            good,
            SimpleNamespace(
                code_path=str(tmp_path),
                runner_code_path=str(runner.Path(runner.__file__).resolve()),
            ),
        )


def _formal_training_contract_config(tmp_path):
    return {
        "run": {
            "purpose": runner.FORMAL_PURPOSE,
            "expected_optimizer_steps": 2,
            "run_name": "batch_001-motionr1-grpo",
            "artifact_path": str(tmp_path / "output" / "checkpoint-2"),
        },
        "training": {
            "tuner_type": "lora",
            "tuner_backend": "peft",
            "use_swift_lora": False,
            "target_modules": "all-linear",
            "target_regex": None,
            "target_parameters": None,
            "modules_to_save": ["motion_prenorm", "motion_proj"],
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "use_rslora": False,
            "use_dora": False,
            "lora_dtype": None,
            "lorap_lr_ratio": None,
            "lorap_emb_lr": 1.0e-6,
            "freeze_llm": False,
            "freeze_vit": True,
            "freeze_aligner": True,
            "max_steps": 2,
            "learning_rate": 1.0e-6,
            "save_only_model": False,
            "save_safetensors": True,
            "save_strategy": "steps",
            "save_steps": 1,
            "logging_steps": 1,
            "callbacks": [runner.FORMAL_TRAINING_CALLBACK],
        },
    }


def test_formal_training_contract_rejects_smoke_zero_step_and_wrong_leaf(tmp_path):
    good = _formal_training_contract_config(tmp_path)
    assert runner._validate_formal_training_contract(good) == 2
    attacks = [
        ("expected_optimizer_steps", 0, "positive integer"),
        ("run_name", "batch-smoke", "smoke/debug"),
        ("artifact_path", str(tmp_path / "output" / "checkpoint-1"), "exact expected"),
    ]
    for key, value, message in attacks:
        attacked = json.loads(json.dumps(good))
        attacked["run"][key] = value
        with pytest.raises(ValueError, match=message):
            runner._validate_formal_training_contract(attacked)

    for callbacks in (None, [], [runner.FORMAL_TRAINING_CALLBACK, "other"]):
        attacked = json.loads(json.dumps(good))
        if callbacks is None:
            attacked["training"].pop("callbacks")
        else:
            attacked["training"]["callbacks"] = callbacks
        with pytest.raises(ValueError, match="training-receipt callback"):
            runner._validate_formal_training_contract(attacked)

    attacked = json.loads(json.dumps(good))
    attacked["training"]["deepspeed"] = "zero2"
    with pytest.raises(ValueError, match="deepspeed is unsupported"):
        runner._validate_formal_training_contract(attacked)

    attacked = json.loads(json.dumps(good))
    attacked["training"]["save_safetensors"] = False
    with pytest.raises(ValueError, match="save_safetensors=true"):
        runner._validate_formal_training_contract(attacked)

    attacked = json.loads(json.dumps(good))
    attacked["training"]["logging_steps"] = 2
    with pytest.raises(ValueError, match="logging_steps=1"):
        runner._validate_formal_training_contract(attacked)


def test_formal_training_evidence_requires_exact_step_finite_loss_and_optimizer(tmp_path):
    artifact = tmp_path / "checkpoint-2"
    artifact.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"):
        (artifact / name).write_bytes(b"evidence")
    trainer_state = {
        "global_step": 2,
        "log_history": [{"step": 1, "loss": 1.25}, {"step": 2, "loss": 0.75}],
    }
    (artifact / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    config = {"run": {"expected_optimizer_steps": 2}}
    runner._validate_grpo_training_evidence(config, artifact)

    trainer_state["global_step"] = 0
    (artifact / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="global_step"):
        runner._validate_grpo_training_evidence(config, artifact)

    trainer_state["global_step"] = 2
    trainer_state["log_history"] = [{"step": 2, "loss": float("nan")}]
    (artifact / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        runner._validate_grpo_training_evidence(config, artifact)


def test_adapter_update_evidence_rejects_zero_lora_b():
    torch = pytest.importorskip("torch")
    runner._validate_grpo_adapter_update_state(
        {
            "layer.lora_A.weight": torch.ones(2, 2),
            "layer.lora_B.weight": torch.ones(2, 2),
        }
    )
    with pytest.raises(Exception, match="non-zero LoRA-B"):
        runner._validate_grpo_adapter_update_state(
            {
                "layer.lora_A.weight": torch.ones(2, 2),
                "layer.lora_B.weight": torch.zeros(2, 2),
            }
        )


def test_formal_environment_and_config_credentials_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="configured values are forbidden"):
        runner._enforce_formal_ambient_environment(
            {"run": {"env": {"VIDEO_MAX_TOKEN_NUM": "999999"}}}
        )
    with pytest.raises(ValueError, match="credential field"):
        runner._reject_formal_config_secrets(
            {"run": {"env": {"WANDB_API_KEY": "secret"}}}
        )
    config = {"run": {"context_expansion_guard": True}}
    with pytest.raises(ValueError, match="active environment limits"):
        runner._enforce_context_expansion_guard(
            config, env={"VIDEO_MAX_TOKEN_NUM": "999999"}, formal_artifact=True
        )


@pytest.mark.parametrize("mutation", ["data", "transitive_code"])
def test_formal_input_snapshot_detects_post_preflight_mutation(
    tmp_path, monkeypatch, mutation
):
    names = (
        "base_artifact_path",
        "train_data_path",
        "validation_data_path",
        "benchmark_path",
        "leakage_audit_path",
        "config_path",
        "runner_code_path",
        "environment_path",
        "motion_vqvae_asset_path",
    )
    values = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        values[name] = str(path)
    code = tmp_path / "src" / "motion_eval"
    critical = code / "data" / "training_receipt.py"
    critical.parent.mkdir(parents=True)
    critical.write_text("frozen = True\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_FORMAL_CODE_PATHS", (code,))
    arguments = SimpleNamespace(**values)
    snapshot = runner._capture_formal_input_snapshot(arguments, media_bindings={})
    runner._verify_formal_input_snapshot(snapshot)
    if mutation == "data":
        (tmp_path / "train_data_path").write_bytes(b"changed")
    else:
        critical.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during Swift"):
        runner._verify_formal_input_snapshot(snapshot)


def test_formal_code_snapshot_covers_motion_model_import_tree():
    expected = {
        runner.REPO_ROOT / "src" / "motionllm",
        runner.REPO_ROOT / "src" / "motion_eval",
    }
    assert expected.issubset(set(runner._FORMAL_CODE_PATHS))
    resolved = [path.resolve() for path in runner._FORMAL_CODE_PATHS]
    assert len(resolved) == len(set(resolved))
    for package_root in expected:
        assert not any(
            path != package_root and package_root in path.parents
            for path in runner._FORMAL_CODE_PATHS
        )
