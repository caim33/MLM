"""Framework-light MotionLLM model configuration and validation facade."""

from .config import (
    MotionDevicePolicy,
    MotionDTypePolicy,
    MotionModelSpec,
    MotionResizePolicy,
    migrate_legacy_motion_config,
    resolve_motion_model_spec,
    validate_motion_encoder_downsample,
)
from .errors import (
    MotionInjectionError,
    MotionModelConfigError,
    MotionModelError,
    StateDictAuditError,
)
from .generation import is_generation_prefill, prefill_motion_payload
from .injection import (
    enumerate_motion_compute_placements,
    normalize_modalities,
    required_feature_length,
    validate_motion_compute_contract,
    validate_motion_presence,
    validate_motion_segment_ownership,
    validate_preembedded_motion_inputs,
)
from .state_dict import (
    StateDictAudit,
    VQCheckpointSelection,
    audit_state_dict,
    extract_state_dict,
    normalize_state_dict_keys,
    select_vq_checkpoint_state,
)

__all__ = [
    "MotionDevicePolicy",
    "MotionDTypePolicy",
    "MotionInjectionError",
    "MotionModelConfigError",
    "MotionModelError",
    "MotionModelSpec",
    "MotionResizePolicy",
    "StateDictAudit",
    "StateDictAuditError",
    "VQCheckpointSelection",
    "audit_state_dict",
    "enumerate_motion_compute_placements",
    "extract_state_dict",
    "is_generation_prefill",
    "migrate_legacy_motion_config",
    "normalize_modalities",
    "normalize_state_dict_keys",
    "prefill_motion_payload",
    "required_feature_length",
    "resolve_motion_model_spec",
    "select_vq_checkpoint_state",
    "validate_motion_compute_contract",
    "validate_motion_encoder_downsample",
    "validate_motion_presence",
    "validate_motion_segment_ownership",
    "validate_preembedded_motion_inputs",
]
