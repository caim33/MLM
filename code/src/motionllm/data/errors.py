"""Typed errors for the framework-free MotionLLM data boundary."""

from __future__ import annotations

from pathlib import Path


class DataContractError(ValueError):
    """Base class for malformed data-boundary inputs."""


class PathResolutionError(DataContractError):
    """Raised when a configured path cannot be resolved as requested."""


class UnsafePathError(PathResolutionError):
    """Raised when a path escapes its declared root or uses unsafe syntax."""


class MediaNotFoundError(PathResolutionError):
    """Raised when a required media file does not exist."""


class JsonlOpenError(DataContractError):
    """Raised when a JSONL source cannot be opened as a regular file."""

    def __init__(self, source: Path, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"cannot open JSONL source {source}: {reason}")


class JsonlLineError(DataContractError):
    """Raised for one invalid JSONL row without rendering its payload."""

    def __init__(
        self,
        source: Path,
        line_number: int,
        reason: str,
        *,
        sample_id: str | None = None,
    ) -> None:
        self.source = source
        self.line_number = line_number
        self.reason = reason
        self.sample_id = sample_id
        identity = f" (sample_id={sample_id!r})" if sample_id is not None else ""
        super().__init__(f"{source}: line {line_number}{identity}: {reason}")


class DuplicateSampleIdError(JsonlLineError):
    """Raised at the later row when a sample identity is repeated."""

    def __init__(
        self,
        source: Path,
        line_number: int,
        sample_id: str,
        first_line_number: int,
    ) -> None:
        self.first_line_number = first_line_number
        super().__init__(
            source,
            line_number,
            f"duplicate sample_id; first declared on line {first_line_number}",
            sample_id=sample_id,
        )


class MessageContractError(DataContractError):
    """Raised when a legacy conversation cannot be adapted losslessly."""


class CollationContractError(DataContractError):
    """Raised when physical tensors and logical sample ownership disagree."""
