from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenvl.grpo_ms_swift.datasets.motion_grpo_dataset import (
    build_reward_metadata,
    collate_reward_metadata,
)
from qwenvl.grpo_ms_swift.plugins.rewards_semantic_format import (
    format_reward_plugin,
    option_accuracy_reward_plugin,
    semantic_reward_plugin,
)
from qwenvl.grpo_ms_swift.runner import train_grpo_ms_swift as runner


def columns():
    return {
        "sample_id": ["s1", "s1"],
        "group_id": ["g1", "g1"],
        "branch": ["vm", "v"],
        "rollout_id": [0, 1],
        "answer": ["A", "A"],
        "solution": [
            "<think>reason</think><answer>A</answer>",
            "<think>reason</think><answer>A</answer>",
        ],
    }


def test_legacy_reward_names_are_strict_core_facades():
    completions = [
        "<think>reason</think><answer>A</answer>",
        "A",
    ]
    assert option_accuracy_reward_plugin(completions, **columns()) == [1.0, 0.0]
    assert format_reward_plugin(completions, **columns()) == [1.0, 0.0]
    assert semantic_reward_plugin(completions, **columns()) == [1.0, 0.0]


def test_legacy_dataset_adapter_has_no_id_or_branch_defaults():
    row = {
        "sample_id": "s",
        "group_id": "g",
        "branch": "vm",
        "rollout_id": 0,
        "answer": "B",
        "solution": "<answer>B</answer>",
    }
    built = build_reward_metadata(row, sample_index=999)
    assert built["sample_id"] == "s"
    assert built["rollout_id"] == 0
    batch = collate_reward_metadata({}, [built])
    assert batch["answer"] == ["B"]
    with pytest.raises(ValueError):
        build_reward_metadata({"answer": "A"}, sample_index=7)


def test_runner_env_report_never_contains_values():
    secret = "sentinel-run-env-secret"
    launch_env, report = runner._build_launch_env(
        {"run": {"env": {"WANDB_API_KEY": secret, "SAFE_SETTING": "visible"}}}
    )
    assert launch_env["WANDB_API_KEY"] == secret
    assert secret not in repr(report)
    assert "visible" not in repr(report)
    assert report == {"WANDB_API_KEY": "<redacted>", "SAFE_SETTING": "<set>"}


def test_runner_dataset_precheck_requires_explicit_metadata(tmp_path):
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        json.dumps(
            {"sample_id": "s_vm", "group_id": "g", "branch": "vm", "rollout_id": 0,
             "answer": "A", "solution": "<answer>A</answer>", "motion": "motion.npy"}
        )
        + "\n"
        + json.dumps(
            {"sample_id": "s_v", "group_id": "g", "branch": "v", "rollout_id": 1,
             "answer": "A", "solution": "<answer>A</answer>"}
        )
        + "\n",
        encoding="utf-8",
    )
    runner._precheck_dataset_records({"data": {"dataset": [str(valid)]}})

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps({"branch": "v", "answer": "A"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid reward metadata"):
        runner._precheck_dataset_records({"data": {"dataset": [str(invalid)]}})


def test_swift_external_plugin_standalone_import_is_a_thin_adapter(monkeypatch):
    swift = types.ModuleType("swift")
    rewards = types.ModuleType("swift.rewards")
    callbacks = types.ModuleType("swift.callbacks")

    class ORM:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    rewards.ORM = ORM
    rewards.orms = {}
    callbacks.TrainerCallback = ORM
    callbacks.callbacks_map = {}
    swift.rewards = rewards
    swift.callbacks = callbacks
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.rewards", rewards)
    monkeypatch.setitem(sys.modules, "swift.callbacks", callbacks)

    plugin_path = (
        Path(__file__).resolve().parents[2]
        / "qwenvl"
        / "grpo_ms_swift"
        / "plugins"
        / "swift_external_rewards.py"
    )
    spec = importlib.util.spec_from_file_location("standalone_motion_rewards", plugin_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(rewards.orms) == {
        "motion_semantic",
        "motion_option_accuracy",
        "motion_format",
        "motion_vm_v_bonus",
        "qa_mc_rubric",
        "motion_rubric_v2",
    }
    assert callbacks.callbacks_map == {
        "motion_training_receipt": module.MotionTrainingReceiptCallback
    }
    orm = rewards.orms["motion_option_accuracy"]()
    assert orm(
        ["<answer>A</answer>"],
        sample_id=["s"],
        group_id=["g"],
        branch=["vm"],
        rollout_id=[0],
        answer=["A"],
    ) == [1.0]


@pytest.mark.parametrize(
    "script_name, requires_reload",
    [("full_sft.sh", True), ("lora_sft.sh", True)],
)
def test_formal_sft_launchers_pass_frozen_split_vq_and_seed_arguments(
    script_name, requires_reload
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / script_name
    ).read_text(encoding="utf-8")
    for required in (
        "TRAIN_DATASET_USE",
        "VALIDATION_DATASET_USE",
        "MOTION_VQVAE_ASSET_PATH",
        "SEED",
        "PYTHON_EXECUTABLE",
        "RUNNER_CODE_PATH",
        "TRAINING_RECEIPT_PATH",
        "BATCH_RECEIPT_SHA256",
        "ATTEMPT_SHA256",
    ):
        assert required in script
    for flag in (
        "--dataset_use",
        "--eval_dataset_use",
        "--motion_vqvae_asset_path",
        "--motion_vqvae_path",
        "--seed",
        "--runner_code_path",
        "--training_receipt_path",
        "--batch_receipt_sha256",
        "--attempt_sha256",
    ):
        assert flag in script
    assert '--model_name_or_path "$BASE_ARTIFACT_PATH"' in script
    assert '--motion_vqvae_path "$MOTION_VQVAE_ASSET_PATH"' in script
    assert "args=(" in script
    assert "missing controller external-HMAC-bound pre-spawn snapshot" in script
    assert "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}" in script
    assert "NPROC_PER_NODE=${NPROC_PER_NODE:-1}" in script
    assert "args=\"" not in script
    if requires_reload:
        assert "RELOAD_RECEIPT_PATH" in script
        assert "--reload_receipt_path" in script


@pytest.mark.parametrize("script_name", ["full_sft.sh", "lora_sft.sh"])
def test_formal_sft_launchers_are_identity_bound_and_do_not_use_ambient_torchrun(
    script_name,
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / script_name
    ).read_text(encoding="utf-8")

    assert script.index("usage()") < script.index("required_env=(")
    assert script.index("while [[ $# -gt 0 ]]") < script.index("required_env=(")
    assert "Train dataset override may be supplied only once" in script
    assert "Validation dataset override may be supplied only once" in script
    assert "canonical_model_registry_id=motionr1_vm_lora" in script
    assert "canonical_model_family=qwen3_vl_motion" in script
    assert 'environment_real" != "$environment_identity' in script
    assert 'base_environment_identity' in script
    assert 'runner_code_real" != "${project_root}/qwenvl' in script
    assert 'GROUP_NUM_MV:-1' in script
    assert 'GROUP_NUM_MOTION:-0' in script
    assert 'GROUP_NUM_VIDEO:-0' in script
    assert 'GROUP_NUM_TEXT:-0' in script
    assert "LEARNING_RATE must be finite and greater than zero" in script
    assert "BATCH_RECEIPT_SHA256 ATTEMPT_SHA256" in script
    assert "^[0-9a-f]{64}$" in script
    assert '"$PYTHON_EXECUTABLE" -I -m torch.distributed.run' not in script
    assert "exit 78" in script
    assert not any(line.lstrip().startswith("torchrun ") for line in script.splitlines())
    if script_name == "lora_sft.sh":
        assert "LORA_DROPOUT must be finite and in the range [0, 1)" in script
        assert "LORA_USE_DORA must be exactly true or false" in script


@pytest.mark.parametrize("script_name", ["full_sft.sh", "lora_sft.sh"])
def test_formal_sft_workers_reject_sitecustomize_injection_and_pin_torchrun_argv(
    script_name,
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / script_name
    ).read_text(encoding="utf-8")

    for variable in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHON_EXEC",
    ):
        assert variable in script
    assert "Formal SFT rejects ambient Python environment variable" in script
    assert "unset PYTHONPATH PYTHONHOME PYTHONUSERBASE" in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert "export PYTHONSAFEPATH=1" in script
    assert "export PYTHONDONTWRITEBYTECODE=1" in script
    assert 'export PYTHON_EXEC="$PYTHON_EXECUTABLE"' in script
    assert script.index("exit 78") < script.index("unsafe_python_env=(")
    assert script.index("exit 78") < script.index('interpreter_identity=$("$PYTHON_EXECUTABLE"')


@pytest.mark.parametrize("script_name", ["full_sft.sh", "lora_sft.sh"])
def test_formal_sft_launcher_block_never_executes_the_supplied_interpreter(
    tmp_path, script_name
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this Windows test host")
    marker = tmp_path / "interpreter_started"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env sh\nprintf started > \"$FORMAL_TEST_MARKER\"\nexit 99\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    environment = dict(os.environ)
    environment.pop("BASH_ENV", None)
    environment.update(
        {"PYTHON_EXECUTABLE": str(fake), "FORMAL_TEST_MARKER": str(marker)}
    )
    completed = subprocess.run(
        [bash, str(script)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 78
    assert not marker.exists()
    assert "pre-spawn snapshot" in completed.stderr


def test_lora_fresh_disk_token_checks_precede_model_mutation_and_peft_load():
    source = (
        Path(__file__).resolve().parents[2] / "qwenvl" / "train" / "lora_sft.py"
    ).read_text(encoding="utf-8")
    formal_reload = source[source.index("if formal_artifact and is_primary_process(torch):") :]
    ordered_markers = (
        "transformers.AutoProcessor.from_pretrained",
        "disk_motion_ids = verify_motion_tokenizer_tokens(reloaded_tokenizer)",
        "verify_processor_save_reload(",
        "bind_model_to_motion_tokens(",
        "reloaded_model = PeftModel.from_pretrained(",
    )
    positions = [formal_reload.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert "setup_motion_tokens(reloaded_tokenizer" not in formal_reload
    assert "supports_motion=spec.supports_motion" in formal_reload


def test_video_only_default_apply_lora_can_publish_without_modules_to_save(
    tmp_path, monkeypatch
):
    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTaskType:
        CAUSAL_LM = "causal_lm"

    def fake_get_peft_model(model, config):
        model.peft_config = config
        return model

    fake_peft = types.ModuleType("peft")
    fake_peft.__spec__ = importlib.util.spec_from_loader("peft", loader=None)
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.PeftModel = type("PeftModel", (), {})
    fake_peft.TaskType = FakeTaskType
    fake_peft.get_peft_model = fake_get_peft_model
    fake_trainer = types.ModuleType("qwenvl.train.trainer")
    fake_trainer.replace_qwen2_vl_attention_class = lambda: None
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "qwenvl.train.trainer", fake_trainer)

    module_name = "_motionllm_test_lora_sft_default_publication"
    module_path = (
        Path(__file__).resolve().parents[2] / "qwenvl" / "train" / "lora_sft.py"
    )
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert module_spec is not None and module_spec.loader is not None
    facade = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, facade)
    module_spec.loader.exec_module(facade)

    class Parameter:
        def __init__(self, values, *, requires_grad):
            self.values = values
            self.requires_grad = requires_grad

        def numpy(self):
            import numpy as np

            return np.asarray(self.values, dtype=np.float32)

    class VideoOnlyPeftModel:
        def __init__(self, *, trainable):
            self.config = SimpleNamespace()
            self._named = {
                "base.q_proj.lora_A.default.weight": Parameter(
                    (1.0, 2.0), requires_grad=trainable
                ),
                "base.q_proj.lora_B.default.weight": Parameter(
                    (3.0, 4.0), requires_grad=trainable
                ),
            }

        def named_parameters(self):
            return iter(self._named.items())

        def print_trainable_parameters(self):
            return None

    class Tokenizer:
        additional_special_tokens = ()

        def __len__(self):
            return 1

        def get_vocab(self):
            return {"base": 0}

    class Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    original = VideoOnlyPeftModel(trainable=True)
    adapted = facade.apply_lora(
        original,
        facade.LoraArguments(),
        facade.FreezePolicy(),
    )
    modules_to_save = facade._publication_modules_to_save(
        adapted, supports_motion=False
    )
    assert modules_to_save == ()
    assert adapted.peft_config.kwargs["modules_to_save"] is None

    reloaded = VideoOnlyPeftModel(trainable=False)
    tokenizer = Tokenizer()
    reloaded_tokenizer = Tokenizer()
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_model.safetensors").write_bytes(b"adapter")
    (artifact / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (artifact / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    from motion_eval.core import hash_path

    artifact_hash = hash_path(artifact, symlink_policy="reject").digest
    receipt = facade.verify_lora_save_reload(
        adapted,
        reloaded,
        tokenizer=tokenizer,
        reloaded_tokenizer=reloaded_tokenizer,
        processor=Processor(tokenizer),
        reloaded_processor=Processor(reloaded_tokenizer),
        processor_artifact_path=artifact,
        module_names=modules_to_save,
        batch_id="batch_1",
        model_id="video_model",
        artifact_hash=artifact_hash,
        supports_motion=False,
    )
    assert receipt.expected_modules == ()
    reload_path = facade.write_reload_verification_receipt(
        tmp_path / "reload.json", receipt, allowed_root=tmp_path
    )

    provenance = {}
    for role in (
        "base_artifact",
        "train_data",
        "validation_data",
        "benchmark",
        "leakage_audit",
        "config",
        "code",
        "runner_code",
        "environment",
    ):
        path = tmp_path / f"{role}.txt"
        path.write_text(role, encoding="utf-8")
        provenance[f"{role}_path"] = str(path)
    from motion_eval.training_receipt import (
        make_training_receipt,
        write_training_receipt,
    )

    batch_receipt_sha256 = "a" * 64
    attempt_sha256 = "b" * 64
    provenance_hashes = {
        role: hash_path(
            provenance[f"{role}_path"],
            symlink_policy="reject",
            allowed_root=tmp_path,
        ).digest
        for role in (
            "base_artifact",
            "train_data",
            "validation_data",
            "leakage_audit",
            "config",
            "code",
            "runner_code",
            "environment",
        )
    }
    training_receipt = make_training_receipt(
        batch_id="batch_1",
        model_id="video_model",
        backend_id="test.video.lora",
        model_family="video_test",
        modality="V",
        training_mode="lora_sft",
        planned_global_steps=1,
        actual_global_steps=1,
        planned_optimizer_steps=1,
        actual_optimizer_steps=1,
        finite_losses=[0.25],
        nonzero_finite_gradient_steps=1,
        max_gradient=0.5,
        trainable_tensor_count=2,
        trainable_parameter_count=4,
        changed_trainable_tensor_count=1,
        initial_trainable_sha256="c" * 64,
        final_trainable_sha256="d" * 64,
        max_parameter_update=0.1,
        batch_receipt_sha256=batch_receipt_sha256,
        attempt_sha256=attempt_sha256,
        train_sha256=provenance_hashes["train_data"],
        validation_sha256=provenance_hashes["validation_data"],
        leakage_audit_sha256=provenance_hashes["leakage_audit"],
        base_artifact_sha256=provenance_hashes["base_artifact"],
        config_sha256=provenance_hashes["config"],
        code_sha256=provenance_hashes["code"],
        runner_code_sha256=provenance_hashes["runner_code"],
        environment_sha256=provenance_hashes["environment"],
        artifact_sha256=artifact_hash,
    )
    training_receipt_path = write_training_receipt(
        tmp_path / "training.json",
        training_receipt,
        root=tmp_path,
    )
    arguments = SimpleNamespace(
        artifact_manifest_path=str(tmp_path / "manifest.json"),
        unsafe_legacy_no_manifest=False,
        artifact_root=str(tmp_path),
        batch_id="batch_1",
        model_registry_id="video_model",
        reload_receipt_path=str(reload_path),
        training_receipt_path=str(training_receipt_path),
        batch_receipt_sha256=batch_receipt_sha256,
        attempt_sha256=attempt_sha256,
        resume_manifest=None,
        motion_vqvae_asset_path=None,
        **provenance,
    )
    facade.publish_artifact_distributed(
        arguments,
        training_mode="lora_sft",
        artifact_path=artifact,
        torch_module=SimpleNamespace(distributed=None),
    )
    assert (tmp_path / "manifest.json").is_file()
