import pytest

from motion_eval.contracts import (
    ContractValidationError,
    EvaluationErrorCode,
    GenerativePredictionRow,
    prediction_row_from_dict,
    validate_prediction_row,
)
from motion_eval.evaluation import make_generative_row


def _serialized_row():
    return make_generative_row(
        batch_id="batch-abc123",
        model_id="qwen3vl_4b_lora",
        sample_id="qa-0001",
        group_id="motion-001",
        modality="V",
        gold="B",
        raw_output="<answer>B</answer>",
        metadata={"attempt_id": "attempt-1"},
    ).to_dict()


def test_serialized_contract_has_explicit_kind_schema_and_error_state():
    row = _serialized_row()
    assert row["kind"] == "generative"
    assert row["schema_version"] == "1.0"
    assert row["error_code"] == "none"
    assert row["parse_status"] == "valid"


def test_serialized_mapping_is_revalidated_not_trusted():
    row = _serialized_row()
    row["correct"] = False
    with pytest.raises(ContractValidationError, match="correct must equal"):
        validate_prediction_row(row)


def test_unknown_field_is_rejected_to_catch_schema_typos():
    row = _serialized_row()
    row["eror_code"] = "timeout"
    with pytest.raises(ContractValidationError, match="unknown prediction row fields"):
        prediction_row_from_dict(row)


def test_missing_explicit_schema_or_error_field_is_rejected():
    for field_name in ("schema_version", "error_code"):
        row = _serialized_row()
        del row[field_name]
        with pytest.raises(ContractValidationError, match="missing prediction row fields"):
            prediction_row_from_dict(row)


def test_unknown_error_code_is_rejected():
    row = _serialized_row()
    row.update(
        prediction=None,
        correct=False,
        error_code="gpu_exploded",
        error_message="boom",
        parse_status="not_attempted",
    )
    with pytest.raises(ContractValidationError, match="unknown evaluation error code"):
        prediction_row_from_dict(row)


def test_invalid_output_cannot_be_disguised_as_success():
    with pytest.raises(ContractValidationError):
        GenerativePredictionRow(
            batch_id="batch",
            model_id="model",
            sample_id="sample",
            group_id="group",
            modality="V",
            gold="A",
            prediction="A",
            correct=True,
            error_code=EvaluationErrorCode.INVALID_OUTPUT,
            error_message="invalid",
            raw_output="A",
            parse_status="missing_answer_tag",
        )


def test_metadata_rejects_nonfinite_values():
    row = _serialized_row()
    row["metadata"] = {"latency": float("nan")}
    with pytest.raises(ContractValidationError, match="finite JSON"):
        prediction_row_from_dict(row)


def test_canonical_sample_and_group_ids_are_mandatory():
    row = _serialized_row()
    for field_name, bad_value in (("sample_id", ""), ("group_id", None)):
        candidate = dict(row)
        candidate[field_name] = bad_value
        with pytest.raises(ContractValidationError, match=field_name):
            prediction_row_from_dict(candidate)


def test_unserializable_unicode_is_rejected_at_contract_boundary():
    row = _serialized_row()
    row["raw_output"] = "<answer>B</answer>\ud800"
    with pytest.raises(ContractValidationError, match="Unicode surrogate"):
        prediction_row_from_dict(row)


def test_falsey_non_mapping_metadata_is_not_silently_replaced():
    row = _serialized_row()
    row["metadata"] = []
    with pytest.raises(ContractValidationError, match="metadata must be a JSON object"):
        prediction_row_from_dict(row)


def test_valid_status_and_prediction_are_recomputed_from_raw_output():
    row = _serialized_row()
    row["raw_output"] = "I choose B"
    with pytest.raises(ContractValidationError, match="parse_status does not match"):
        prediction_row_from_dict(row)

    row = _serialized_row()
    row["raw_output"] = "<answer>A</answer>"
    with pytest.raises(ContractValidationError, match="prediction does not match"):
        prediction_row_from_dict(row)


def test_metadata_is_recursively_frozen_and_detached_from_source():
    source = {"preprocessing": {"frames": [1, 2]}}
    row = make_generative_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="V",
        gold="B",
        raw_output="<answer>B</answer>",
        metadata=source,
    )
    source["preprocessing"]["frames"].append(float("nan"))
    assert row.to_dict()["metadata"] == {"preprocessing": {"frames": [1, 2]}}
    with pytest.raises(TypeError):
        row.metadata["preprocessing"]["frames"][0] = 9


def test_motion_and_eval_domain_enums_remain_value_compatible():
    from motionllm.contracts import Modality, OPTION_LABELS

    from motion_eval.contracts import ANSWER_CHOICES, InputModality

    assert tuple(item.value for item in Modality) == tuple(item.value for item in InputModality)
    assert tuple(item.value for item in OPTION_LABELS) == ANSWER_CHOICES
