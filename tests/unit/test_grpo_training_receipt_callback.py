from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


peft = pytest.importorskip("peft")


@dataclass
class _SwiftLikeLoraConfig(peft.LoraConfig):
    """Minimal pinned-ms-swift save contract used by callback tests."""

    lora_dtype: str | None = None
    lorap_lr_ratio: float | None = None
    lorap_emb_lr: float = 1.0e-6

    def to_peft_config(self):
        values = self.to_dict()
        for key in ("lora_dtype", "lorap_lr_ratio", "lorap_emb_lr"):
            values.pop(key)
        return peft.LoraConfig(**values)

    def save_pretrained(self, save_directory, **kwargs):
        self.to_peft_config().save_pretrained(save_directory, **kwargs)
        extension = {
            "lora_dtype": self.lora_dtype,
            "lorap_lr_ratio": self.lorap_lr_ratio,
            "lorap_emb_lr": self.lorap_emb_lr,
        }
        path = Path(save_directory) / "additional_config.json"
        path.write_text(
            json.dumps(extension, ensure_ascii=True, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )


class _TinyAdapterBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.extra = torch.nn.Linear(2, 1)

    def forward(self, values):
        return self.extra(self.proj(values))


def _peft_model():
    return peft.get_peft_model(
        _TinyAdapterBase(),
        _SwiftLikeLoraConfig(
            target_modules=["proj"],
            modules_to_save=["extra"],
            r=1,
            lora_alpha=2,
            lora_dropout=0.0,
        ),
    )


def _load_plugin(monkeypatch):
    swift = types.ModuleType("swift")
    rewards = types.ModuleType("swift.rewards")
    callbacks = types.ModuleType("swift.callbacks")

    class SwiftBase:
        def __init__(self, args=None, trainer=None, **kwargs):
            del kwargs
            self.args = args
            self.trainer = trainer

    rewards.ORM = SwiftBase
    rewards.orms = {}
    callbacks.TrainerCallback = SwiftBase
    callbacks.callbacks_map = {}
    swift.rewards = rewards
    swift.callbacks = callbacks
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.rewards", rewards)
    monkeypatch.setitem(sys.modules, "swift.callbacks", callbacks)

    path = (
        Path(__file__).resolve().parents[2]
        / "qwenvl"
        / "grpo_ms_swift"
        / "plugins"
        / "swift_external_rewards.py"
    )
    spec = importlib.util.spec_from_file_location("training_receipt_plugin_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, callbacks.callbacks_map


def _configure_env(monkeypatch, artifact: Path, *, steps: int = 2):
    monkeypatch.setenv("MOTION_GRPO_TRAINING_NONCE", "a" * 64)
    monkeypatch.setenv("MOTION_GRPO_BATCH_ID", "batch_20260821")
    monkeypatch.setenv("MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS", str(steps))
    monkeypatch.setenv("MOTION_GRPO_ARTIFACT_PATH", str(artifact))


def _run_step(callback, model, state, *, loss: float):
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    callback.on_pre_optimizer_step(None, state, None, model=model)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(-0.1)
    callback.on_optimizer_step(None, state, None, model=model)
    state.global_step += 1
    callback.on_step_end(None, state, None, model=model)
    callback.on_log(None, state, None, logs={"loss": loss}, model=model)


def test_training_callback_writes_exact_nonce_bound_update_receipt(tmp_path, monkeypatch):
    module, callbacks = _load_plugin(monkeypatch)
    assert callbacks == {"motion_training_receipt": module.MotionTrainingReceiptCallback}

    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-2"
    output.mkdir()
    _configure_env(monkeypatch, artifact)
    model = _peft_model()
    args = SimpleNamespace(output_dir=str(output), max_steps=2, tuner_type="lora")
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
    )
    callback = module.MotionTrainingReceiptCallback(args, trainer)
    state = SimpleNamespace(
        global_step=0,
        max_steps=2,
        is_world_process_zero=True,
    )

    callback.on_train_begin(args, state, None, model=model)
    _run_step(callback, model, state, loss=1.0)
    _run_step(callback, model, state, loss=0.5)
    model.save_pretrained(artifact, safe_serialization=True)
    callback.on_save(args, state, None, model=model)

    receipt_path = artifact / "grpo_training_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "motionllm.grpo.training_update.v2"
    assert payload["status"] == "optimizer_and_checkpoint_verified"
    assert payload["nonce"] == "a" * 64
    assert payload["batch_id"] == "batch_20260821"
    assert payload["model_registry_id"] == "motionr1_vm_lora"
    assert payload["expected_optimizer_steps"] == 2
    assert payload["observed_global_step"] == 2
    assert payload["optimizer_step_count"] == 2
    assert payload["finite_loss_count"] == 2
    assert payload["gradient_observation_count"] == 2
    assert payload["changed_tensor_count"] == payload["trainable_tensor_count"]
    assert payload["trainable_tensor_count"] == 4
    assert payload["initial_trainable_state_sha256"] != payload["final_trainable_state_sha256"]
    assert payload["max_abs_gradient"] > 0
    assert payload["max_abs_delta"] > 0
    assert payload["adapter_filename"] == "adapter_model.safetensors"
    assert payload["adapter_payload_size_bytes"] > 0
    assert payload["adapter_tensor_count"] == 4
    assert payload["frozen_embedding_tensor_count"] == 0
    assert len(payload["adapter_payload_sha256"]) == 64
    assert len(payload["final_saveable_adapter_state_sha256"]) == 64
    assert payload["adapter_config_filename"] == "adapter_config.json"
    assert len(payload["adapter_config_payload_sha256"]) == 64
    assert len(payload["adapter_config_semantic_sha256"]) == 64
    assert payload["adapter_extension_filename"] == "additional_config.json"
    assert len(payload["adapter_extension_payload_sha256"]) == 64
    assert len(payload["adapter_extension_semantic_sha256"]) == 64
    assert payload["adapter_extension_semantics"] == {
        "lora_dtype": None,
        "lorap_emb_lr": 1.0e-6,
        "lorap_lr_ratio": None,
    }
    with pytest.raises(FileExistsError):
        callback.on_save(args, state, None, model=model)


def test_training_callback_requires_one_finite_loss_proof_per_step(
    tmp_path, monkeypatch
):
    module, _ = _load_plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-2"
    output.mkdir()
    _configure_env(monkeypatch, artifact)
    model = _peft_model()
    args = SimpleNamespace(output_dir=str(output), max_steps=2, tuner_type="lora")
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
    )
    callback = module.MotionTrainingReceiptCallback(args, trainer)
    state = SimpleNamespace(global_step=0, max_steps=2, is_world_process_zero=True)
    callback.on_train_begin(args, state, None, model=model)
    _run_step(callback, model, state, loss=1.0)
    _run_step(callback, model, state, loss=0.5)
    callback._finite_loss_count = 1
    model.save_pretrained(artifact, safe_serialization=True)

    with pytest.raises(RuntimeError, match="one finite loss proof per optimizer step"):
        callback.on_save(args, state, None, model=model)
    assert not (artifact / "grpo_training_receipt.json").exists()


def test_training_callback_fails_when_optimizer_events_do_not_change_parameters(tmp_path, monkeypatch):
    module, _ = _load_plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-1"
    output.mkdir()
    _configure_env(monkeypatch, artifact, steps=1)
    model = _peft_model()
    args = SimpleNamespace(output_dir=str(output), max_steps=1, tuner_type="lora")
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
    )
    callback = module.MotionTrainingReceiptCallback(args, trainer)
    state = SimpleNamespace(global_step=0, max_steps=1, is_world_process_zero=True)

    callback.on_train_begin(args, state, None, model=model)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    callback.on_pre_optimizer_step(args, state, None, model=model)
    callback.on_optimizer_step(args, state, None, model=model)
    state.global_step = 1
    callback.on_step_end(args, state, None, model=model)
    callback.on_log(args, state, None, logs={"loss": 0.0}, model=model)
    model.save_pretrained(artifact, safe_serialization=True)
    with pytest.raises(RuntimeError, match="no trainable tensor change"):
        callback.on_save(args, state, None, model=model)
    assert not (artifact / "grpo_training_receipt.json").exists()


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("MOTION_GRPO_TRAINING_NONCE", "A" * 64, "nonce"),
        ("MOTION_GRPO_BATCH_ID", "../escape", "batch id"),
        ("MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS", "01", "canonical positive decimal"),
    ],
)
def test_training_callback_rejects_spoofed_bindings(tmp_path, monkeypatch, name, value, match):
    module, _ = _load_plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-2"
    output.mkdir()
    _configure_env(monkeypatch, artifact)
    monkeypatch.setenv(name, value)
    model = torch.nn.Linear(2, 1, bias=False)
    args = SimpleNamespace(output_dir=str(output), max_steps=2, tuner_type="lora")
    callback = module.MotionTrainingReceiptCallback(args, SimpleNamespace(model=model))
    state = SimpleNamespace(global_step=0, max_steps=2, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match=match):
        callback.on_train_begin(args, state, None, model=model)


@pytest.mark.parametrize("flag", ["is_deepspeed_enabled", "is_fsdp_enabled"])
def test_training_callback_rejects_sharded_trainable_state(tmp_path, monkeypatch, flag):
    module, _ = _load_plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-1"
    output.mkdir()
    _configure_env(monkeypatch, artifact, steps=1)
    model = torch.nn.Linear(2, 1, bias=False)
    args = SimpleNamespace(output_dir=str(output), max_steps=1, tuner_type="lora")
    trainer = SimpleNamespace(model=model, **{flag: True})
    callback = module.MotionTrainingReceiptCallback(args, trainer)
    state = SimpleNamespace(global_step=0, max_steps=1, is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="DeepSpeed/FSDP"):
        callback.on_train_begin(args, state, None, model=model)


def _started_linear_callback(tmp_path, monkeypatch):
    module, _ = _load_plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-1"
    output.mkdir()
    _configure_env(monkeypatch, artifact, steps=1)
    model = torch.nn.Linear(2, 1, bias=False)
    args = SimpleNamespace(output_dir=str(output), max_steps=1, tuner_type="lora")
    accelerator = SimpleNamespace(optimizer_step_was_skipped=False)
    trainer = SimpleNamespace(model=model, accelerator=accelerator)
    callback = module.MotionTrainingReceiptCallback(args, trainer)
    state = SimpleNamespace(global_step=0, max_steps=1, is_world_process_zero=True)
    callback.on_train_begin(args, state, None, model=model)
    return callback, model, state, accelerator


@pytest.mark.parametrize("gradient_kind", ["missing", "zero"])
def test_training_callback_rejects_missing_or_all_zero_step_gradient(
    tmp_path, monkeypatch, gradient_kind
):
    callback, model, state, _ = _started_linear_callback(tmp_path, monkeypatch)
    if gradient_kind == "zero":
        model.weight.grad = torch.zeros_like(model.weight)
    with pytest.raises(RuntimeError, match="no gradient|only zero gradients"):
        callback.on_pre_optimizer_step(None, state, None, model=model)
    assert callback._gradient_observation_count == 0
    assert callback._optimizer_step_count == 0


def test_training_callback_rejects_skipped_amp_step_before_counting(
    tmp_path, monkeypatch
):
    callback, model, state, accelerator = _started_linear_callback(
        tmp_path, monkeypatch
    )
    model.weight.grad = torch.ones_like(model.weight)
    callback.on_pre_optimizer_step(None, state, None, model=model)
    accelerator.optimizer_step_was_skipped = True
    with pytest.raises(RuntimeError, match="skipped AMP optimizer step"):
        callback.on_optimizer_step(None, state, None, model=model)
    assert callback._gradient_observation_count == 0
    assert callback._optimizer_step_count == 0


def test_training_callback_rejects_missing_skip_api_and_optimizer_without_pre(
    tmp_path, monkeypatch
):
    callback, model, state, accelerator = _started_linear_callback(
        tmp_path, monkeypatch
    )
    with pytest.raises(RuntimeError, match="no pending gradient"):
        callback.on_optimizer_step(None, state, None, model=model)
    model.weight.grad = torch.ones_like(model.weight)
    callback.on_pre_optimizer_step(None, state, None, model=model)
    del accelerator.optimizer_step_was_skipped
    with pytest.raises(RuntimeError, match="optimizer_step_was_skipped"):
        callback.on_optimizer_step(None, state, None, model=model)


def test_training_callback_rejects_duplicate_optimizer_event(tmp_path, monkeypatch):
    callback, model, state, _ = _started_linear_callback(tmp_path, monkeypatch)
    model.weight.grad = torch.ones_like(model.weight)
    callback.on_pre_optimizer_step(None, state, None, model=model)
    callback.on_optimizer_step(None, state, None, model=model)
    with pytest.raises(RuntimeError, match="duplicate optimizer event"):
        callback.on_optimizer_step(None, state, None, model=model)
