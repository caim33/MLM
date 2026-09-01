"""Closed error taxonomy shared by every evaluation adapter.

The fixed benchmark denominator includes every row.  Consequently, an
adapter may not invent a private error label or omit a failed row: it must use
one of the values below and record the row as incorrect.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class EvaluationErrorCode(str, Enum):
    """Canonical terminal state for one prediction attempt."""

    NONE = "none"
    INVALID_OUTPUT = "invalid_output"
    MEDIA_ERROR = "media_error"
    TIMEOUT = "timeout"
    OOM = "oom"
    RUNTIME_ERROR = "runtime_error"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "EvaluationErrorCode":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("error_code must be a string or EvaluationErrorCode")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(
                f"unknown evaluation error code {value!r}; expected one of: {allowed}"
            ) from exc
