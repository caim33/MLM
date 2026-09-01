"""Errors raised by the framework-light MotionLLM model facade."""

from __future__ import annotations


class MotionModelError(ValueError):
    """Base class for invalid motion-model configuration or inputs."""


class MotionModelConfigError(MotionModelError):
    """The explicit motion model configuration is incomplete or invalid."""


class MotionInjectionError(MotionModelError):
    """Motion inputs cannot be injected without changing sample semantics."""


class StateDictAuditError(MotionModelError):
    """A checkpoint failed the required key/shape audit."""

