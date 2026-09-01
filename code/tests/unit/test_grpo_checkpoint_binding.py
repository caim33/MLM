from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from motionllm.grpo import checkpoint_binding
from motionllm.grpo.checkpoint_binding import (
    CheckpointBindingError,
    bind_live_peft_adapter_to_checkpoint,
    capture_adapter_config,
    capture_adapter_checkpoint,
    extract_peft_adapter_state,
    live_peft_adapter_config_semantics,
    require_identical_adapter_configs,
    require_identical_adapter_states,
    require_trainable_saveable_coverage,
)
from transformers import PretrainedConfig, PreTrainedModel


peft = pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")


class _TinyAdapterBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)
        self.extra = torch.nn.Linear(3, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.extra(self.proj(values))


def _model(*, marker: float = 1.0):
    config = peft.LoraConfig(
        target_modules=["proj"],
        modules_to_save=["extra"],
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    model = peft.get_peft_model(_TinyAdapterBase(), config)
    with torch.no_grad():
        state = peft.get_peft_model_state_dict(model)
        for index, tensor in enumerate(state.values(), start=1):
            tensor.fill_(marker + index)
    return model


def _save(model, checkpoint: Path) -> Path:
    model.save_pretrained(checkpoint, safe_serialization=True)
    return checkpoint


@dataclass
class _SwiftLikeLoraConfig(peft.LoraConfig):
    """Minimal ms-swift 4.2.2 PEFT config/serialization contract."""

    lora_dtype: str | None = None
    lorap_lr_ratio: float | None = None
    lorap_emb_lr: float = 1e-6

    def to_peft_config(self):
        payload = self.to_dict()
        for key in ("lora_dtype", "lorap_lr_ratio", "lorap_emb_lr"):
            payload.pop(key)
        return peft.LoraConfig(**payload)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.to_peft_config().save_pretrained(save_directory, **kwargs)
        extension = {
            "lora_dtype": self.lora_dtype,
            "lorap_lr_ratio": self.lorap_lr_ratio,
            "lorap_emb_lr": self.lorap_emb_lr,
        }
        (Path(save_directory) / "additional_config.json").write_text(
            json.dumps(extension), encoding="utf-8"
        )


def _swift_model(*, marker: float = 1.0):
    config = _SwiftLikeLoraConfig(
        target_modules=["proj"],
        modules_to_save=["extra"],
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    model = peft.get_peft_model(_TinyAdapterBase(), config)
    with torch.no_grad():
        state = peft.get_peft_model_state_dict(model)
        for index, tensor in enumerate(state.values(), start=1):
            tensor.fill_(marker + index)
    return model


def test_real_peft_lora_and_modules_to_save_bind_exactly(tmp_path):
    model = _model()
    checkpoint = _save(model, tmp_path / "checkpoint-1")

    binding = bind_live_peft_adapter_to_checkpoint(model, checkpoint)

    assert binding.filename == "adapter_model.safetensors"
    assert binding.payload_size_bytes > 0
    assert len(binding.payload_sha256) == 64
    assert len(binding.state_sha256) == 64
    assert binding.tensor_count == 4
    assert any("lora_A" in name for name in binding.state)
    assert any("lora_B" in name for name in binding.state)
    assert any("extra.weight" in name for name in binding.state)
    assert any("extra.bias" in name for name in binding.state)
    assert binding.config.filename == "adapter_config.json"
    assert binding.config.payload_size_bytes > 0
    assert len(binding.config.payload_sha256) == 64
    assert len(binding.config.semantic_sha256) == 64
    assert binding.config.semantics["r"] == 2
    assert binding.config.semantics["lora_alpha"] == 4
    assert binding.extension_config is None


def test_old_or_unrelated_checkpoint_weights_cannot_bind_live_final_state(tmp_path):
    model = _model(marker=1.0)
    checkpoint = _save(model, tmp_path / "checkpoint-2")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(7.0)

    with pytest.raises(CheckpointBindingError, match="tensor bytes differ"):
        bind_live_peft_adapter_to_checkpoint(model, checkpoint)


def test_extra_trainable_parameter_not_covered_by_peft_save_state_is_rejected(
    tmp_path,
):
    model = _model()
    model.base_model.model.register_parameter(
        "rogue_trainable", torch.nn.Parameter(torch.ones(3))
    )
    checkpoint = _save(model, tmp_path / "checkpoint-1")

    with pytest.raises(
        CheckpointBindingError, match="not default LoRA/modules_to_save"
    ):
        bind_live_peft_adapter_to_checkpoint(model, checkpoint)


def test_arbitrary_frozen_saveable_extra_is_not_whitelisted():
    model = _model()
    state = dict(peft.get_peft_model_state_dict(model))
    state["base_model.model.arbitrary.weight"] = torch.ones(2, 2)

    with pytest.raises(
        CheckpointBindingError, match="unrecognized non-trainable|non-whitelisted extra"
    ):
        require_trainable_saveable_coverage(model, state)


class _ResizedConfig(PretrainedConfig):
    model_type = "grpo-binding-resized"

    def __init__(self, vocab_size=5, hidden_size=4, **kwargs):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class _ResizedEmbeddingModel(PreTrainedModel):
    config_class = _ResizedConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.extra = torch.nn.Linear(config.hidden_size, 2)
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids):
        return self.lm_head(self.proj(self.embed(input_ids)))


def test_peft_auto_saved_resized_frozen_embeddings_are_strictly_allowed(tmp_path):
    base_path = tmp_path / "base"
    _ResizedEmbeddingModel(_ResizedConfig()).save_pretrained(base_path)
    base = _ResizedEmbeddingModel.from_pretrained(base_path)
    base.resize_token_embeddings(7, mean_resizing=False)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            target_modules=["proj"],
            modules_to_save=["extra"],
            r=1,
            lora_alpha=2,
        ),
    )
    checkpoint = _save(model, tmp_path / "checkpoint-1")

    binding = bind_live_peft_adapter_to_checkpoint(model, checkpoint)

    assert binding.tensor_count == 6
    assert "base_model.model.embed.weight" in binding.state
    assert "base_model.model.lm_head.weight" in binding.state
    assert not model.base_model.model.embed.weight.requires_grad
    assert not model.base_model.model.lm_head.weight.requires_grad


def test_missing_adapter_payload_fails_closed(tmp_path):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    (checkpoint / "adapter_model.safetensors").unlink()

    with pytest.raises(CheckpointBindingError, match="lacks adapter_model.safetensors"):
        capture_adapter_checkpoint(checkpoint)


def test_adapter_key_drift_fails_closed(tmp_path):
    model = _model()
    checkpoint = _save(model, tmp_path / "checkpoint-1")
    weights = checkpoint / "adapter_model.safetensors"
    state = safetensors_torch.load_file(str(weights), device="cpu")
    original = sorted(state)[0]
    state[original + ".drift"] = state.pop(original)
    safetensors_torch.save_file(state, str(weights))

    with pytest.raises(CheckpointBindingError, match="adapter key drift"):
        bind_live_peft_adapter_to_checkpoint(model, checkpoint)


@pytest.mark.parametrize("empty_kind", ["mapping", "tensor"])
def test_empty_adapter_weights_fail_closed(tmp_path, empty_kind):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    state = {} if empty_kind == "mapping" else {"empty.weight": torch.empty(0)}
    safetensors_torch.save_file(state, str(checkpoint / "adapter_model.safetensors"))

    with pytest.raises(CheckpointBindingError, match="no tensors|empty tensor"):
        capture_adapter_checkpoint(checkpoint)


def test_unsafe_bin_payload_is_rejected_even_with_safetensors(tmp_path):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    (checkpoint / "adapter_model.bin").write_bytes(b"old-pickle-payload")

    with pytest.raises(CheckpointBindingError, match="adapter_model.bin"):
        capture_adapter_checkpoint(checkpoint)


def test_symlinked_checkpoint_is_rejected(tmp_path):
    real = _save(_model(), tmp_path / "real-checkpoint-1")
    linked = tmp_path / "checkpoint-1"
    try:
        os.symlink(real, linked, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable for this account")

    with pytest.raises(CheckpointBindingError, match="symlink"):
        capture_adapter_checkpoint(linked)


def test_payload_mutation_during_safetensors_inspection_is_detected(
    tmp_path, monkeypatch
):
    checkpoint = _save(_model(marker=1.0), tmp_path / "checkpoint-1")
    weights = checkpoint / "adapter_model.safetensors"
    replacement = safetensors_torch.load_file(str(weights), device="cpu")
    first_key = sorted(replacement)[0]
    replacement[first_key] = replacement[first_key] + 1.0
    original_reader = checkpoint_binding._read_frozen_safetensors

    def mutate_after_frozen_read(frozen_path):
        state = original_reader(frozen_path)
        safetensors_torch.save_file(replacement, str(weights))
        return state

    monkeypatch.setattr(
        checkpoint_binding, "_read_frozen_safetensors", mutate_after_frozen_read
    )
    with pytest.raises(CheckpointBindingError, match="changed while safetensors"):
        capture_adapter_checkpoint(checkpoint)


def test_real_frozen_peft_reload_extracts_saveable_state_without_trainables(tmp_path):
    trained = _model()
    checkpoint = _save(trained, tmp_path / "checkpoint-1")
    frozen = peft.PeftModel.from_pretrained(
        _TinyAdapterBase(), checkpoint, is_trainable=False
    )

    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    disk = capture_adapter_checkpoint(checkpoint)
    reloaded_state = extract_peft_adapter_state(frozen)
    assert require_identical_adapter_states(reloaded_state, disk.state) == disk.state_sha256
    reloaded_config = live_peft_adapter_config_semantics(
        frozen, expect_trainable=False
    )
    assert (
        require_identical_adapter_configs(reloaded_config, disk.config.semantics)
        == disk.config.semantic_sha256
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("r", 8),
        ("lora_alpha", 99),
        ("target_modules", ["other"]),
        ("modules_to_save", ["other"]),
        ("bias", "all"),
        ("use_rslora", True),
    ],
)
def test_adapter_config_semantic_mutation_cannot_bind_live_model(
    tmp_path, field, replacement
):
    model = _model()
    checkpoint = _save(model, tmp_path / "checkpoint-1")
    path = checkpoint / "adapter_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = replacement
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(CheckpointBindingError, match="adapter config"):
        bind_live_peft_adapter_to_checkpoint(model, checkpoint)


@pytest.mark.parametrize("schema_attack", ["extra", "missing", "duplicate"])
def test_adapter_config_strict_json_schema_rejects_attack(tmp_path, schema_attack):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    path = checkpoint / "adapter_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if schema_attack == "extra":
        payload["rogue"] = 1
        raw = json.dumps(payload)
    elif schema_attack == "missing":
        payload.pop("r")
        raw = json.dumps(payload)
    else:
        raw = path.read_text(encoding="utf-8").lstrip()
        raw = '{"r": 999,' + raw[1:]
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(
        CheckpointBindingError, match="schema differs|strict UTF-8 JSON"
    ):
        capture_adapter_config(checkpoint)


def test_adapter_config_mutation_during_strict_parse_is_detected(tmp_path, monkeypatch):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    path = checkpoint / "adapter_config.json"
    original_parser = checkpoint_binding._parse_strict_lora_config

    def mutate_after_frozen_parse(raw):
        parsed = original_parser(raw)
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["lora_alpha"] += 1
        path.write_text(json.dumps(changed), encoding="utf-8")
        return parsed

    monkeypatch.setattr(
        checkpoint_binding, "_parse_strict_lora_config", mutate_after_frozen_parse
    )
    with pytest.raises(CheckpointBindingError, match="changed while it was inspected"):
        capture_adapter_config(checkpoint)


def test_symlinked_adapter_config_is_rejected(tmp_path):
    checkpoint = _save(_model(), tmp_path / "checkpoint-1")
    config = checkpoint / "adapter_config.json"
    real = checkpoint / "real-adapter-config.json"
    config.replace(real)
    try:
        os.symlink(real, config)
    except OSError:
        pytest.skip("symlink creation is unavailable for this account")

    with pytest.raises(CheckpointBindingError, match="symlink"):
        capture_adapter_config(checkpoint)


def test_swift_extension_config_binds_exact_live_and_disk_semantics(tmp_path):
    model = _swift_model()
    checkpoint = _save(model, tmp_path / "checkpoint-1")

    binding = bind_live_peft_adapter_to_checkpoint(
        model, checkpoint, require_swift_extension=True
    )

    extension = binding.extension_config
    assert extension is not None
    assert extension.filename == "additional_config.json"
    assert extension.payload_size_bytes > 0
    assert len(extension.payload_sha256) == 64
    assert len(extension.semantic_sha256) == 64
    assert extension.semantics == {
        "lora_dtype": None,
        "lorap_lr_ratio": None,
        "lorap_emb_lr": 1e-6,
    }


def test_required_swift_extension_rejects_missing_sidecar(tmp_path):
    checkpoint = _save(_swift_model(), tmp_path / "checkpoint-1")
    (checkpoint / "additional_config.json").unlink()

    with pytest.raises(CheckpointBindingError, match="lacks additional_config.json"):
        capture_adapter_checkpoint(checkpoint, require_swift_extension=True)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("lora_dtype", "bfloat16"),
        ("lorap_lr_ratio", 16.0),
        ("lorap_emb_lr", 2e-6),
    ],
)
def test_swift_extension_semantic_mutation_cannot_bind_live_model(
    tmp_path, field, replacement
):
    model = _swift_model()
    checkpoint = _save(model, tmp_path / "checkpoint-1")
    path = checkpoint / "additional_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = replacement
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        CheckpointBindingError,
        match="live/disk Swift LoRA extension semantics differ",
    ):
        bind_live_peft_adapter_to_checkpoint(
            model, checkpoint, require_swift_extension=True
        )


@pytest.mark.parametrize(
    ("schema_attack", "raw", "message"),
    [
        (
            "extra",
            '{"lora_dtype":null,"lorap_lr_ratio":null,'
            '"lorap_emb_lr":1e-6,"rogue":1}',
            "extension config schema differs",
        ),
        (
            "missing",
            '{"lora_dtype":null,"lorap_lr_ratio":null}',
            "extension config schema differs",
        ),
        (
            "duplicate",
            '{"lora_dtype":null,"lora_dtype":"float16",'
            '"lorap_lr_ratio":null,"lorap_emb_lr":1e-6}',
            "not strict UTF-8 JSON",
        ),
        (
            "nan",
            '{"lora_dtype":null,"lorap_lr_ratio":null,"lorap_emb_lr":NaN}',
            "not strict UTF-8 JSON",
        ),
        (
            "lora_dtype_type",
            '{"lora_dtype":7,"lorap_lr_ratio":null,"lorap_emb_lr":1e-6}',
            "lora_dtype is invalid",
        ),
        (
            "lorap_lr_ratio_type",
            '{"lora_dtype":null,"lorap_lr_ratio":true,"lorap_emb_lr":1e-6}',
            "lorap_lr_ratio is invalid",
        ),
        (
            "lorap_emb_lr_type",
            '{"lora_dtype":null,"lorap_lr_ratio":null,'
            '"lorap_emb_lr":"1e-6"}',
            "lorap_emb_lr is invalid",
        ),
    ],
)
def test_swift_extension_strict_json_schema_rejects_attack(
    tmp_path, schema_attack, raw, message
):
    del schema_attack
    checkpoint = _save(_swift_model(), tmp_path / "checkpoint-1")
    (checkpoint / "additional_config.json").write_text(raw, encoding="utf-8")

    with pytest.raises(CheckpointBindingError, match=message):
        capture_adapter_checkpoint(checkpoint, require_swift_extension=True)


def test_swift_extension_mutation_during_strict_parse_is_detected(
    tmp_path, monkeypatch
):
    checkpoint = _save(_swift_model(), tmp_path / "checkpoint-1")
    path = checkpoint / "additional_config.json"
    original_parser = checkpoint_binding._parse_strict_swift_lora_extension

    def mutate_after_frozen_parse(raw):
        parsed = original_parser(raw)
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["lorap_emb_lr"] = 2e-6
        path.write_text(json.dumps(changed), encoding="utf-8")
        return parsed

    monkeypatch.setattr(
        checkpoint_binding,
        "_parse_strict_swift_lora_extension",
        mutate_after_frozen_parse,
    )
    with pytest.raises(CheckpointBindingError, match="changed while it was inspected"):
        capture_adapter_checkpoint(checkpoint, require_swift_extension=True)


def test_symlinked_swift_extension_config_is_rejected(tmp_path):
    checkpoint = _save(_swift_model(), tmp_path / "checkpoint-1")
    config = checkpoint / "additional_config.json"
    real = checkpoint / "real-additional-config.json"
    config.replace(real)
    try:
        os.symlink(real, config)
    except OSError:
        pytest.skip("symlink creation is unavailable for this account")

    with pytest.raises(CheckpointBindingError, match="symlink"):
        capture_adapter_checkpoint(checkpoint, require_swift_extension=True)
