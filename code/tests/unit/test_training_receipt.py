from __future__ import annotations

import json

import pytest

from motion_eval.training_receipt import (
    TrainingReceiptError,
    load_and_validate_training_receipt,
    make_training_receipt,
    validate_formal_provenance_snapshot,
    validate_training_receipt,
)


def valid_fields():
    return {
        "batch_id": "batch_1",
        "model_id": "model_1",
        "backend_id": "backend:v1",
        "model_family": "family_1",
        "modality": "VM",
        "training_mode": "lora_sft",
        "planned_global_steps": 2,
        "actual_global_steps": 2,
        "planned_optimizer_steps": 2,
        "actual_optimizer_steps": 2,
        "nonzero_finite_gradient_steps": 2,
        "max_gradient": 0.5,
        "trainable_tensor_count": 2,
        "trainable_parameter_count": 10,
        "changed_trainable_tensor_count": 1,
        "initial_trainable_sha256": "1" * 64,
        "final_trainable_sha256": "2" * 64,
        "max_parameter_update": None,
        "batch_receipt_sha256": "3" * 64,
        "attempt_sha256": "4" * 64,
        "train_sha256": "5" * 64,
        "validation_sha256": "6" * 64,
        "leakage_audit_sha256": "7" * 64,
        "base_artifact_sha256": "8" * 64,
        "config_sha256": "9" * 64,
        "code_sha256": "a" * 64,
        "runner_code_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "artifact_sha256": "d" * 64,
    }


def test_training_receipt_proves_steps_gradients_losses_and_change():
    receipt = make_training_receipt(finite_losses=[1.25, 0.75], **valid_fields())
    assert validate_training_receipt(receipt)["receipt_sha256"] == receipt["receipt_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actual_global_steps", 0, "positive integer"),
        ("actual_optimizer_steps", 1, "every planned step"),
        ("nonzero_finite_gradient_steps", 1, "every optimizer step"),
        ("max_gradient", 0.0, "finite positive"),
        ("changed_trainable_tensor_count", 0, "positive integer"),
        ("final_trainable_sha256", "1" * 64, "did not change"),
    ],
)
def test_training_receipt_rejects_fake_or_zero_step_proof(field, value, message):
    fields = valid_fields()
    fields[field] = value
    with pytest.raises(TrainingReceiptError, match=message):
        make_training_receipt(finite_losses=[1.0], **fields)


def test_training_receipt_reader_rejects_duplicate_nonfinite_and_extra_keys(tmp_path):
    receipt = make_training_receipt(finite_losses=[1.0], **valid_fields())
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_and_validate_training_receipt(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"loss":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_and_validate_training_receipt(nonfinite)

    receipt["extra"] = True
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(TrainingReceiptError, match="schema"):
        load_and_validate_training_receipt(extra)


def test_formal_snapshot_parser_rejects_self_hashed_but_incomplete_payload():
    payload = {
        "schema_version": "motionllm-inprocess-provenance-v2",
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": "batch_1",
        "model_id": "model_1",
        "training_mode": "lora_sft",
        "canonical_identity": {},
        "provenance": {},
        "manifests": {},
    }
    from motion_eval.core import sha256_json

    payload["snapshot_sha256"] = sha256_json(payload)
    with pytest.raises(TrainingReceiptError, match="canonical identity"):
        validate_formal_provenance_snapshot(payload)
