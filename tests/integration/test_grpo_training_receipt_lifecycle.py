from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments


@dataclass
class _SwiftLikeLoraConfig(LoraConfig):
    """Minimal pinned-ms-swift save contract used by lifecycle tests."""

    lora_dtype: str | None = None
    lorap_lr_ratio: float | None = None
    lorap_emb_lr: float = 1.0e-6

    def to_peft_config(self):
        values = self.to_dict()
        for key in ("lora_dtype", "lorap_lr_ratio", "lorap_emb_lr"):
            values.pop(key)
        return LoraConfig(**values)

    def save_pretrained(self, save_directory, **kwargs):
        self.to_peft_config().save_pretrained(save_directory, **kwargs)
        extension = {
            "lora_dtype": self.lora_dtype,
            "lorap_lr_ratio": self.lorap_lr_ratio,
            "lorap_emb_lr": self.lorap_emb_lr,
        }
        (Path(save_directory) / "additional_config.json").write_text(
            json.dumps(extension, ensure_ascii=True, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )


class _SwiftCallbackBase(TrainerCallback):
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer


class _ORM:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _Rows(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = float(index + 1)
        return {
            "input_values": torch.tensor([value, value + 1.0]),
            "labels": torch.tensor([value * 0.25]),
        }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.extra = torch.nn.Linear(2, 1)

    def forward(self, input_values, labels=None):
        logits = self.extra(self.proj(input_values.float()))
        loss = torch.nn.functional.mse_loss(logits, labels.float())
        return {"loss": loss, "logits": logits}


def _plugin(monkeypatch):
    swift = types.ModuleType("swift")
    rewards = types.ModuleType("swift.rewards")
    callbacks = types.ModuleType("swift.callbacks")
    rewards.ORM = _ORM
    rewards.orms = {}
    callbacks.TrainerCallback = _SwiftCallbackBase
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
    spec = importlib.util.spec_from_file_location("training_receipt_lifecycle_plugin", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_transformers_callback_lifecycle_writes_final_receipt(tmp_path, monkeypatch):
    pytest.importorskip("accelerate", reason="Transformers Trainer lifecycle requires accelerate")
    module = _plugin(monkeypatch)
    output = (tmp_path / "output").resolve()
    artifact = output / "checkpoint-2"
    monkeypatch.setenv("MOTION_GRPO_TRAINING_NONCE", "b" * 64)
    monkeypatch.setenv("MOTION_GRPO_BATCH_ID", "callback_lifecycle")
    monkeypatch.setenv("MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS", "2")
    monkeypatch.setenv("MOTION_GRPO_ARTIFACT_PATH", str(artifact))
    arguments = TrainingArguments(
        output_dir=str(output),
        max_steps=2,
        per_device_train_batch_size=1,
        learning_rate=0.01,
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        save_only_model=False,
        save_safetensors=True,
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    arguments.tuner_type = "lora"
    model = get_peft_model(
        _TinyModel(),
        _SwiftLikeLoraConfig(
            target_modules=["proj"],
            modules_to_save=["extra"],
            r=1,
            lora_alpha=2,
            lora_dropout=0.0,
        ),
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=_Rows())
    trainer.add_callback(module.MotionTrainingReceiptCallback(arguments, trainer))
    trainer.train()

    receipt = json.loads(
        (artifact / "grpo_training_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["observed_global_step"] == 2
    assert receipt["optimizer_step_count"] == 2
    assert receipt["gradient_observation_count"] == 2
    assert receipt["finite_loss_count"] == 2
    assert receipt["changed_tensor_count"] > 0
    assert receipt["max_abs_gradient"] > 0
    assert receipt["max_abs_delta"] > 0
    assert receipt["adapter_filename"] == "adapter_model.safetensors"
    assert receipt["adapter_tensor_count"] == 4
    assert receipt["frozen_embedding_tensor_count"] == 0
    assert len(receipt["adapter_payload_sha256"]) == 64
    assert len(receipt["final_saveable_adapter_state_sha256"]) == 64
    assert receipt["adapter_extension_filename"] == "additional_config.json"
    assert receipt["adapter_extension_semantics"] == {
        "lora_dtype": None,
        "lorap_emb_lr": 1.0e-6,
        "lorap_lr_ratio": None,
    }
