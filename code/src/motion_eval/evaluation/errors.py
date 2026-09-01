"""Deterministic mapping from worker failures to the closed taxonomy."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from motion_eval.contracts.errors import EvaluationErrorCode


@dataclass(eq=False)
class EvaluationFailure(Exception):
    """An adapter-raised failure carrying an already normalized code."""

    code: EvaluationErrorCode
    message: str

    def __post_init__(self) -> None:
        self.code = EvaluationErrorCode.coerce(self.code)
        if self.code is EvaluationErrorCode.NONE:
            raise ValueError("EvaluationFailure cannot use error code 'none'")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("EvaluationFailure message must be non-empty")
        Exception.__init__(self, self.message)


def classify_evaluation_exception(
    error: BaseException,
    *,
    media_context: bool = False,
) -> EvaluationErrorCode:
    """Map an exception to one stable row-level error code.

    ``media_context`` must be set explicitly by media loaders.  This prevents
    an unrelated missing config/checkpoint from being mislabeled as a media
    failure.  CUDA/vendor OOM classes are not imported; their conventional
    class names and messages are recognized after built-in ``MemoryError``.
    """

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if isinstance(error, EvaluationFailure):
        return error.code
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return EvaluationErrorCode.TIMEOUT
    if isinstance(error, MemoryError):
        return EvaluationErrorCode.OOM

    class_name = type(error).__name__.lower()
    message = str(error).lower()
    if "outofmemory" in class_name or "out of memory" in message:
        return EvaluationErrorCode.OOM
    if media_context and isinstance(error, (OSError, ValueError)):
        return EvaluationErrorCode.MEDIA_ERROR
    return EvaluationErrorCode.RUNTIME_ERROR
