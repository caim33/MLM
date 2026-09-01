"""Framework-independent motion loading and preprocessing."""

from .errors import (
    MotionError,
    MotionIOError,
    MotionValidationError,
    TemporalContractError,
)
from .io import load_motion_array, load_normalization_stats
from .normalization import denormalize_motion, normalize_motion
from .temporal import (
    TemporalLengthContract,
    apply_temporal_contract,
    downsample_motion,
    plan_temporal_length,
    prepare_motion_temporal,
)
from .validation import validate_motion_array, validate_normalization_stats

__all__ = [
    "MotionError",
    "MotionIOError",
    "MotionValidationError",
    "TemporalContractError",
    "TemporalLengthContract",
    "apply_temporal_contract",
    "denormalize_motion",
    "downsample_motion",
    "load_motion_array",
    "load_normalization_stats",
    "normalize_motion",
    "plan_temporal_length",
    "prepare_motion_temporal",
    "validate_motion_array",
    "validate_normalization_stats",
]
