"""Public framework-free contracts for MotionLLM."""

from .errors import (
    ContractError,
    GoldSyntaxError,
    MediaContractError,
    ModalityContractError,
    OptionContractError,
    SampleContractError,
)
from .modality import Modality
from .options import (
    OPTION_LABELS,
    STRICT_GOLD_PATTERN,
    GoldAnswer,
    Option,
    OptionLabel,
    format_gold_answer,
    parse_gold_answer,
)
from .sample import MediaReferences, Sample, validate_media_for_modality

__all__ = [
    "ContractError",
    "GoldAnswer",
    "GoldSyntaxError",
    "MediaContractError",
    "MediaReferences",
    "Modality",
    "ModalityContractError",
    "OPTION_LABELS",
    "Option",
    "OptionContractError",
    "OptionLabel",
    "STRICT_GOLD_PATTERN",
    "Sample",
    "SampleContractError",
    "format_gold_answer",
    "parse_gold_answer",
    "validate_media_for_modality",
]
