import subprocess

import pytest

from motion_eval.contracts import EvaluationErrorCode
from motion_eval.evaluation import EvaluationFailure, classify_evaluation_exception


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("late"), EvaluationErrorCode.TIMEOUT),
        (
            subprocess.TimeoutExpired(["worker"], timeout=30),
            EvaluationErrorCode.TIMEOUT,
        ),
        (MemoryError("allocation"), EvaluationErrorCode.OOM),
        (FakeCudaOutOfMemoryError("CUDA out of memory"), EvaluationErrorCode.OOM),
        (RuntimeError("boom"), EvaluationErrorCode.RUNTIME_ERROR),
    ],
)
def test_exception_classification_is_closed_and_deterministic(error, expected):
    assert classify_evaluation_exception(error) is expected


def test_media_context_is_explicit():
    error = FileNotFoundError("video.mp4")
    assert classify_evaluation_exception(error) is EvaluationErrorCode.RUNTIME_ERROR
    assert (
        classify_evaluation_exception(error, media_context=True)
        is EvaluationErrorCode.MEDIA_ERROR
    )


def test_explicit_failure_code_wins():
    error = EvaluationFailure(EvaluationErrorCode.INVALID_OUTPUT, "bad tag")
    assert classify_evaluation_exception(error) is EvaluationErrorCode.INVALID_OUTPUT


def test_none_is_not_a_failure_code():
    with pytest.raises(ValueError, match="cannot use"):
        EvaluationFailure(EvaluationErrorCode.NONE, "not an error")
