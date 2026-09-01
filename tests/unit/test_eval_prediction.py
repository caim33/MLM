import math

import pytest

from motion_eval.contracts import (
    ContractValidationError,
    DiscriminativePredictionRow,
    EvaluationErrorCode,
    InputModality,
    ParseStatus,
    argmax_choice,
    prediction_row_from_dict,
)
from motion_eval.evaluation import (
    discriminative_failure_row,
    generative_failure_row,
    make_discriminative_row,
    make_generative_row,
)


def _scores(**overrides):
    values = {"A": 0.1, "B": 0.2, "C": 0.8, "D": -1.0}
    values.update(overrides)
    return values


def test_generative_builder_makes_strict_success_row():
    row = make_generative_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="VM",
        gold="C",
        raw_output="why\n<answer>C</answer>",
        metadata={"seed": 7},
    )

    assert row.prediction == "C"
    assert row.correct is True
    assert row.error_code is EvaluationErrorCode.NONE
    assert row.parse_status is ParseStatus.VALID
    assert row.modality is InputModality.VIDEO_MOTION


def test_invalid_generative_output_remains_in_denominator_as_incorrect():
    row = make_generative_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="V",
        gold="A",
        raw_output="I choose A",
    )

    assert row.prediction is None
    assert row.correct is False
    assert row.error_code is EvaluationErrorCode.INVALID_OUTPUT
    assert row.parse_status is ParseStatus.MISSING_ANSWER_TAG


def test_operational_generative_failure_never_publishes_prediction():
    row = generative_failure_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="V",
        gold="D",
        error_code="timeout",
        error_message="worker deadline exceeded",
    )

    assert row.prediction is None
    assert not row.correct
    assert row.parse_status is ParseStatus.NOT_ATTEMPTED


def test_discriminative_builder_records_all_scores_and_computed_argmax():
    row = make_discriminative_row(
        batch_id="batch",
        model_id="agcn_official",
        sample_id="sample",
        group_id="group",
        modality="M",
        gold="C",
        scores=_scores(),
    )

    assert row.prediction == "C"
    assert row.correct
    assert dict(row.scores or {}) == _scores()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("choice", ["A", "B", "C", "D"])
def test_discriminative_scores_reject_nan_and_infinity(choice, bad):
    with pytest.raises(ContractValidationError, match="must be finite"):
        make_discriminative_row(
            batch_id="batch",
            model_id="model",
            sample_id="sample",
            group_id="group",
            modality="M",
            gold="A",
            scores=_scores(**{choice: bad}),
        )


def test_discriminative_scores_reject_bool_and_non_numeric_values():
    for bad in (True, "0.4", None):
        with pytest.raises(ContractValidationError, match="real number"):
            make_discriminative_row(
                batch_id="batch",
                model_id="model",
                sample_id="sample",
                group_id="group",
                modality="M",
                gold="A",
                scores=_scores(A=bad),
            )


def test_discriminative_scores_reject_integer_too_large_for_finite_float():
    with pytest.raises(ContractValidationError, match="finite real number"):
        make_discriminative_row(
            batch_id="batch",
            model_id="model",
            sample_id="sample",
            group_id="group",
            modality="M",
            gold="A",
            scores=_scores(A=10**10000),
        )


@pytest.mark.parametrize(
    "scores",
    [
        {"A": 1.0, "B": 2.0, "C": 3.0},
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0},
    ],
)
def test_discriminative_scores_require_exact_canonical_keys(scores):
    with pytest.raises(ContractValidationError, match="exactly A/B/C/D"):
        argmax_choice(scores)


def test_explicit_prediction_must_match_argmax():
    with pytest.raises(ContractValidationError, match="does not match score argmax"):
        make_discriminative_row(
            batch_id="batch",
            model_id="model",
            sample_id="sample",
            group_id="group",
            modality="M",
            gold="A",
            scores=_scores(),
            prediction="A",
        )


def test_tied_argmax_uses_canonical_option_order():
    assert argmax_choice({"D": 1, "C": 0, "B": 1, "A": 1}) == "A"


def test_failed_discriminative_row_has_no_fake_scores():
    row = discriminative_failure_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="M",
        gold="A",
        error_code="media_error",
        error_message="motion file is missing",
    )
    assert row.scores is None
    assert row.prediction is None
    assert not row.correct


def test_correct_flag_cannot_disagree_with_prediction():
    with pytest.raises(ContractValidationError, match="correct must equal"):
        DiscriminativePredictionRow(
            batch_id="batch",
            model_id="model",
            sample_id="sample",
            group_id="group",
            modality="M",
            gold="A",
            prediction="C",
            correct=True,
            scores=_scores(),
        )


def test_prediction_round_trip_preserves_contract():
    row = make_discriminative_row(
        batch_id="batch",
        model_id="model",
        sample_id="sample",
        group_id="group",
        modality="M",
        gold="C",
        scores=_scores(),
        metadata={"frames": [0, 2], "normalization": {"mean": 0.0}},
    )
    restored = prediction_row_from_dict(row.to_dict())
    assert restored.to_dict() == row.to_dict()
