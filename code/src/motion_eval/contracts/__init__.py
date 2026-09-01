"""Framework-independent contracts for evaluation artifacts."""

from .errors import EvaluationErrorCode
from .prediction import (
    ANSWER_CHOICES,
    ContractValidationError,
    DiscriminativePredictionRow,
    GenerativePredictionRow,
    InputModality,
    ParseStatus,
    PredictionRow,
    argmax_choice,
    prediction_row_from_dict,
    validate_prediction_row,
)

__all__ = [
    "ANSWER_CHOICES",
    "ContractValidationError",
    "DiscriminativePredictionRow",
    "EvaluationErrorCode",
    "GenerativePredictionRow",
    "InputModality",
    "ParseStatus",
    "PredictionRow",
    "argmax_choice",
    "prediction_row_from_dict",
    "validate_prediction_row",
]
