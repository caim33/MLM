from __future__ import annotations

from pathlib import Path

import pytest
import torch

from motion_eval.core import sha256_json
from model_evaluation_agent.scripts.backends import finetune_motionllm_lora
from model_evaluation_agent.scripts.backends import finetune_videollama_lora


def test_motionllm_saved_state_must_equal_final_memory(tmp_path: Path) -> None:
    expected = {
        "transformer.h.0.attn.lora_A": torch.tensor([[1.0, 2.0]]),
        "transformer.h.0.attn.lora_B": torch.tensor([[3.0], [4.0]]),
    }
    path = tmp_path / "lora.pth"
    torch.save({**expected, "transformer.h.0.attn.lora_B": torch.zeros(2, 1)}, path)

    with pytest.raises(RuntimeError, match="does not match final in-memory state"):
        finetune_motionllm_lora.verify_saved_tensor_state(
            path,
            expected,
            label="MotionLLM LoRA",
        )


def test_motionllm_projector_rejects_wrong_saved_keys(tmp_path: Path) -> None:
    expected = {
        "0.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "0.bias": torch.ones(2, dtype=torch.float32),
    }
    path = tmp_path / "linear.pth"
    torch.save({"0.weight": expected["0.weight"]}, path)

    with pytest.raises(RuntimeError, match=r"missing=\['0.bias'\]"):
        finetune_motionllm_lora.verify_saved_tensor_state(
            path,
            expected,
            label="MotionLLM projector",
        )


def test_motionllm_exact_saved_state_is_accepted(tmp_path: Path) -> None:
    expected = {
        "lora_A": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "lora_B": torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16),
    }
    path = tmp_path / "lora.pth"
    torch.save(expected, path)

    saved_sha256 = finetune_motionllm_lora.verify_saved_tensor_state(
        path,
        expected,
        label="MotionLLM LoRA",
    )

    assert len(saved_sha256) == 64


def test_motionllm_rejects_trainable_language_parameter_omitted_by_lora_filter(
    monkeypatch,
) -> None:
    class Language(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.ones(2))
            self.accidentally_trainable = torch.nn.Parameter(torch.ones(2))

    monkeypatch.setattr(
        finetune_motionllm_lora,
        "lora_filter",
        lambda name, _value: name == "lora_weight",
        raising=False,
    )
    with pytest.raises(RuntimeError, match=r"missing=\['accidentally_trainable'\]"):
        finetune_motionllm_lora.language_trainable_serialized_state(Language())


def test_motionllm_rejects_extra_serialized_language_parameter(
    monkeypatch,
) -> None:
    class Language(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.ones(2))
            self.frozen_weight = torch.nn.Parameter(
                torch.zeros(2), requires_grad=False
            )

    monkeypatch.setattr(
        finetune_motionllm_lora,
        "lora_filter",
        lambda _name, _value: True,
        raising=False,
    )
    with pytest.raises(RuntimeError, match=r"unexpected=\['frozen_weight'\]"):
        finetune_motionllm_lora.language_trainable_serialized_state(Language())


def test_motionllm_rejects_state_dict_hook_that_changes_live_trainable_bytes(
    monkeypatch,
) -> None:
    class RewritingLanguage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.ones(2))

        def state_dict(self, *args, **kwargs):
            del args, kwargs
            return {"lora_weight": self.lora_weight.detach() + 1.0}

    monkeypatch.setattr(
        finetune_motionllm_lora,
        "lora_filter",
        lambda name, _value: name == "lora_weight",
        raising=False,
    )
    with pytest.raises(RuntimeError, match="does not match final in-memory state"):
        finetune_motionllm_lora.language_trainable_serialized_state(
            RewritingLanguage()
        )


def test_motionllm_state_binding_hashes_and_counts_are_strict() -> None:
    body = {
        "schema_version": "1.0",
        "model_id": "motionllm_official",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "language_trainable_tensor_count": 2,
        "projector_trainable_tensor_count": 2,
        "projector_state_tensor_count": 2,
        "combined_trainable_tensor_count": 4,
        "combined_trainable_parameter_count": 16,
        "language_live_sha256": "1" * 64,
        "language_disk_sha256": "1" * 64,
        "language_reload_sha256": "1" * 64,
        "projector_live_sha256": "2" * 64,
        "projector_disk_sha256": "2" * 64,
        "projector_reload_sha256": "2" * 64,
        "combined_live_sha256": "3" * 64,
        "combined_reload_sha256": "3" * 64,
    }
    binding = {**body, "state_binding_sha256": sha256_json(body)}
    assert (
        finetune_motionllm_lora.validate_state_binding(binding)
        == binding
    )

    tampered = {**binding, "language_reload_sha256": "4" * 64}
    tampered_body = {
        key: value for key, value in tampered.items() if key != "state_binding_sha256"
    }
    tampered["state_binding_sha256"] = sha256_json(tampered_body)
    with pytest.raises(RuntimeError, match="language live/disk/reload"):
        finetune_motionllm_lora.validate_state_binding(tampered)


def test_videollama_saved_adapter_checks_modules_to_save(tmp_path: Path) -> None:
    expected = {
        "base_model.model.layers.0.self_attn.lora_A.weight": torch.ones(2, 2),
        # PEFT removes the ``modules_to_save.default`` wrapper from this key
        # when producing its serialized adapter state.
        "base_model.model.lm_head.weight": torch.arange(6.0).reshape(2, 3),
    }
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    torch.save(
        {
            **expected,
            "base_model.model.lm_head.weight": torch.zeros(2, 3),
        },
        adapter / "adapter_model.bin",
    )

    with pytest.raises(RuntimeError, match="does not match final in-memory state"):
        finetune_videollama_lora.verify_saved_peft_state(adapter, expected)


def test_videollama_fingerprint_includes_modules_to_save() -> None:
    class PeftLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.Linear(2, 2)
            self.modules_to_save = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2)}
            )
            self.frozen = torch.nn.Linear(2, 2)
            self.frozen.requires_grad_(False)

    _flat, _digest, entries = finetune_videollama_lora.trainable_fingerprint(
        PeftLike()
    )

    assert any("lora_A" in name for name in entries)
    assert any("modules_to_save.default" in name for name in entries)


def test_videollama_rejects_unserializable_trainable_parameter() -> None:
    with pytest.raises(RuntimeError, match="may not serialize"):
        finetune_videollama_lora.trainable_fingerprint(torch.nn.Linear(2, 2))
