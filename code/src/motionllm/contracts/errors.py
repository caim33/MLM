"""Framework-free domain contract errors.

The exceptions in this module deliberately inherit from :class:`ValueError` so
callers can treat malformed external data as a validation failure without
depending on PyTorch, Transformers, or a validation framework.
"""

from __future__ import annotations


class ContractError(ValueError):
    """Base class for a value that violates a MotionLLM domain contract."""


class ModalityContractError(ContractError):
    """Raised when a modality or legacy branch value is not canonical."""


class OptionContractError(ContractError):
    """Raised when an option label or option set is malformed."""


class GoldSyntaxError(OptionContractError):
    """Raised when a gold answer is not an exact answer tag."""


class MediaContractError(ContractError):
    """Raised when media references do not match the declared modality."""


class SampleContractError(ContractError):
    """Raised when a canonical sample field is invalid."""
