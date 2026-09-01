"""Exceptions raised by the framework-independent motion pipeline."""


class MotionError(ValueError):
    """Base class for invalid motion data or configuration."""


class MotionIOError(MotionError):
    """A motion file exists but cannot be decoded unambiguously."""


class MotionValidationError(MotionError):
    """Motion values, shapes, or normalization statistics are invalid."""


class TemporalContractError(MotionValidationError):
    """A temporal padding/downsampling contract is inconsistent."""
