"""Safe constructors that turn adapter outputs into validated row contracts."""

from __future__ import annotations

from typing import Any, Mapping

from motion_eval.contracts.errors import EvaluationErrorCode
from motion_eval.contracts.prediction import (
    DiscriminativePredictionRow,
    GenerativePredictionRow,
    InputModality,
    ParseStatus,
    argmax_choice,
)

from .parser import parse_strict_answer


def make_generative_row(
    *,
    batch_id: str,
    model_id: str,
    sample_id: str,
    group_id: str,
    modality: InputModality | str,
    gold: str,
    raw_output: str,
    metadata: Mapping[str, Any] | None = None,
) -> GenerativePredictionRow:
    parsed = parse_strict_answer(raw_output)
    if parsed.is_valid:
        return GenerativePredictionRow(
            batch_id=batch_id,
            model_id=model_id,
            sample_id=sample_id,
            group_id=group_id,
            modality=modality,
            gold=gold,
            prediction=parsed.answer,
            correct=parsed.answer == gold,
            error_code=EvaluationErrorCode.NONE,
            raw_output=raw_output,
            parse_status=parsed.status,
            metadata={} if metadata is None else metadata,
        )
    return GenerativePredictionRow(
        batch_id=batch_id,
        model_id=model_id,
        sample_id=sample_id,
        group_id=group_id,
        modality=modality,
        gold=gold,
        prediction=None,
        correct=False,
        error_code=EvaluationErrorCode.INVALID_OUTPUT,
        error_message=f"strict answer parse failed: {parsed.status.value}",
        raw_output=raw_output,
        parse_status=parsed.status,
        metadata={} if metadata is None else metadata,
    )


def generative_failure_row(
    *,
    batch_id: str,
    model_id: str,
    sample_id: str,
    group_id: str,
    modality: InputModality | str,
    gold: str,
    error_code: EvaluationErrorCode | str,
    error_message: str,
    metadata: Mapping[str, Any] | None = None,
) -> GenerativePredictionRow:
    code = EvaluationErrorCode.coerce(error_code)
    if code in {EvaluationErrorCode.NONE, EvaluationErrorCode.INVALID_OUTPUT}:
        raise ValueError("operational failure requires media/timeout/oom/runtime error")
    return GenerativePredictionRow(
        batch_id=batch_id,
        model_id=model_id,
        sample_id=sample_id,
        group_id=group_id,
        modality=modality,
        gold=gold,
        prediction=None,
        correct=False,
        error_code=code,
        error_message=error_message,
        raw_output=None,
        parse_status=ParseStatus.NOT_ATTEMPTED,
        metadata={} if metadata is None else metadata,
    )


def make_discriminative_row(
    *,
    batch_id: str,
    model_id: str,
    sample_id: str,
    group_id: str,
    modality: InputModality | str,
    gold: str,
    scores: Mapping[str, float],
    prediction: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DiscriminativePredictionRow:
    computed = argmax_choice(scores)
    candidate = computed if prediction is None else prediction
    return DiscriminativePredictionRow(
        batch_id=batch_id,
        model_id=model_id,
        sample_id=sample_id,
        group_id=group_id,
        modality=modality,
        gold=gold,
        prediction=candidate,
        correct=candidate == gold,
        error_code=EvaluationErrorCode.NONE,
        scores=scores,
        metadata={} if metadata is None else metadata,
    )


def discriminative_failure_row(
    *,
    batch_id: str,
    model_id: str,
    sample_id: str,
    group_id: str,
    modality: InputModality | str,
    gold: str,
    error_code: EvaluationErrorCode | str,
    error_message: str,
    metadata: Mapping[str, Any] | None = None,
) -> DiscriminativePredictionRow:
    code = EvaluationErrorCode.coerce(error_code)
    if code is EvaluationErrorCode.NONE:
        raise ValueError("failure row cannot use error code 'none'")
    return DiscriminativePredictionRow(
        batch_id=batch_id,
        model_id=model_id,
        sample_id=sample_id,
        group_id=group_id,
        modality=modality,
        gold=gold,
        prediction=None,
        correct=False,
        error_code=code,
        error_message=error_message,
        scores=None,
        metadata=metadata or {},
    )
