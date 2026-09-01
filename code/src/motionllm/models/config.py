"""Explicit, framework-free configuration for the motion model adapter.

Legacy checkpoints did not persist several motion-specific fields.  The
``migrate_legacy_motion_config`` function is the only place where those old
defaults are mapped.  Runtime code consumes :class:`MotionModelSpec` and never
guesses dimensions, token IDs, dtype, or resize behaviour itself.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from motionllm.fusion import ProjectorSpec, build_projector_spec

from .errors import MotionModelConfigError


_MISSING = object()
_NO_DEFAULT = object()
_READABLE_INTEGER_BITS = 128
_INTEGER_LOW_BITS = 16


def _bounded_int_summary(value: int) -> str:
    """Return a bounded diagnostic without decimalizing attacker-sized ints."""

    bit_length = value.bit_length()
    if bit_length <= _READABLE_INTEGER_BITS:
        return str(value)
    sign = "-" if value < 0 else "+"
    low_mask = (1 << _INTEGER_LOW_BITS) - 1
    low_bits = value & low_mask
    return (
        f"<int sign={sign} bit_length={bit_length} "
        f"low{_INTEGER_LOW_BITS}=0x{low_bits:04x}>"
    )


class MotionResizePolicy(str, Enum):
    """Feature/placeholder length mismatch policy."""

    ERROR = "error"
    LINEAR = "linear"


class MotionDTypePolicy(str, Enum):
    """How the torch adapter selects its motion compute dtype."""

    MODEL = "model"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class MotionDevicePolicy(str, Enum):
    """Which existing module owns motion tensors."""

    ENCODER = "encoder"
    MODEL = "model"


def _get(config: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(config, Mapping):
        if key in config:
            return config[key]
    elif hasattr(config, key):
        return getattr(config, key)
    if default is _NO_DEFAULT:
        raise MotionModelConfigError(f"missing required config field {key!r}")
    return default


def _set(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
        return
    try:
        setattr(config, key, value)
    except (AttributeError, TypeError) as exc:
        raise MotionModelConfigError(
            f"legacy config cannot persist migrated field {key!r}"
        ) from exc


def _set_missing(config: Any, key: str, value: Any) -> bool:
    current = _get(config, key, _MISSING)
    if current is not _MISSING and current is not None:
        return False
    _set(config, key, value)
    return True


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise MotionModelConfigError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MotionModelConfigError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise MotionModelConfigError(
            f"{name} must be > 0, got {_bounded_int_summary(parsed)}"
        )
    return parsed


def _optional_token_id(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MotionModelConfigError(f"{name} must be a non-negative integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MotionModelConfigError(
            f"{name} must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise MotionModelConfigError(
            f"{name} must be >= 0, got {_bounded_int_summary(parsed)}"
        )
    return parsed


def _enum(enum_type: type[Enum], value: Any, *, name: str) -> Any:
    if not isinstance(value, str):
        raise MotionModelConfigError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise MotionModelConfigError(
            f"{name} must be one of {allowed}; got {value!r}"
        ) from exc


def _optional_path(value: Any, *, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise MotionModelConfigError(f"{name} must be a filesystem path or null")
    path = Path(value)
    if not path.is_absolute():
        raise MotionModelConfigError(f"{name} must be absolute")
    return path


@dataclass(frozen=True, slots=True)
class MotionModelSpec:
    """Resolved motion-model contract used by framework adapters."""

    input_dim: int
    encoder_output_dim: int
    downsample_factor: int
    placeholder_token_id: int
    start_token_id: int | None
    end_token_id: int | None
    projector: ProjectorSpec
    resize_policy: MotionResizePolicy
    dtype_policy: MotionDTypePolicy
    device_policy: MotionDevicePolicy
    allow_batch_broadcast: bool = False
    normalization_mean_path: Path | None = None
    normalization_std_path: Path | None = None

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, name="motion_input_dim")
        _positive_int(self.encoder_output_dim, name="motion_encoder_output_dim")
        _positive_int(self.downsample_factor, name="motion_downsample_factor")
        _optional_token_id(
            self.placeholder_token_id, name="motion_placeholder_token_id"
        )
        _optional_token_id(self.start_token_id, name="motion_start_token_id")
        _optional_token_id(self.end_token_id, name="motion_end_token_id")
        if (self.start_token_id is None) != (self.end_token_id is None):
            raise MotionModelConfigError(
                "motion_start_token_id and motion_end_token_id must be set together"
            )
        configured_ids = {
            token
            for token in (
                self.placeholder_token_id,
                self.start_token_id,
                self.end_token_id,
            )
            if token is not None
        }
        configured_count = 1 + (2 if self.start_token_id is not None else 0)
        if len(configured_ids) != configured_count:
            raise MotionModelConfigError("motion protocol token IDs must be distinct")
        if not isinstance(self.projector, ProjectorSpec):
            raise MotionModelConfigError("projector must be a ProjectorSpec")
        if self.projector.input_dim != self.encoder_output_dim:
            raise MotionModelConfigError(
                "motion projector input dimension must equal encoder output dimension"
            )
        if not isinstance(self.allow_batch_broadcast, bool):
            raise MotionModelConfigError("motion_allow_batch_broadcast must be bool")
        if (self.normalization_mean_path is None) != (
            self.normalization_std_path is None
        ):
            raise MotionModelConfigError(
                "motion normalization mean/std paths must be set together"
            )

    @property
    def has_boundary_token_ids(self) -> bool:
        return self.start_token_id is not None


def migrate_legacy_motion_config(config: Any) -> tuple[str, ...]:
    """Persist old checkpoint defaults as explicit fields.

    The returned field names form an auditable compatibility receipt.  New
    configs should already contain these fields, in which case the tuple is
    empty.  This function intentionally does not guess boundary token IDs;
    tokenizer setup must persist them after adding the special tokens.
    """

    migrated: list[str] = []
    dataname = str(_get(config, "dataname", "kit"))
    legacy_input_dim = 251 if dataname == "kit" else 263
    encoder_dim = _get(config, "vqvae_output_emb_width", 512)
    text_config = _get(config, "text_config", None)
    if text_config is None:
        text_hidden = _get(config, "text_hidden_size", 2560)
    else:
        text_hidden = _get(text_config, "hidden_size")

    try:
        legacy_downsample = int(_get(config, "vqvae_stride_t", 2)) ** int(
            _get(config, "vqvae_down_t", 2)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionModelConfigError(
            "legacy vqvae_stride_t/down_t must define a valid downsample factor"
        ) from exc
    legacy_post_norm_value = _get(
        config, "motion_projector_post_norm", _MISSING
    )
    legacy_post_norm_missing = (
        legacy_post_norm_value is _MISSING or legacy_post_norm_value is None
    )
    defaults = {
        "motion_input_dim": legacy_input_dim,
        "motion_encoder_output_dim": encoder_dim,
        "motion_downsample_factor": legacy_downsample,
        "motion_placeholder_token_id": 160001,
        "motion_projector_input_dim": encoder_dim,
        "motion_projector_hidden_dims": (4096,),
        "motion_projector_output_dim": text_hidden,
        "motion_projector_activation": "gelu",
        "motion_projector_bias": True,
        "motion_projector_pre_norm": True,
        # Historical checkpoints persisted motion_postnorm parameters, but the
        # legacy forward path never applied that layer.  Preserve those state
        # keys separately without changing the checkpoint's computation.
        "motion_projector_post_norm": False,
        "motion_resize_policy": "error",
        "motion_compute_dtype": "model",
        "motion_device_policy": "encoder",
        "motion_allow_batch_broadcast": False,
    }
    for key, value in defaults.items():
        if _set_missing(config, key, value):
            migrated.append(key)
    if legacy_post_norm_missing and _set_missing(
        config, "motion_legacy_postnorm_state_compat", True
    ):
        migrated.append("motion_legacy_postnorm_state_compat")
    return tuple(migrated)


def validate_motion_encoder_downsample(
    spec: MotionModelSpec,
    *,
    vqvae_down_t: Any,
    vqvae_stride_t: Any,
) -> int:
    """Require the configured placeholder factor to match the real encoder."""

    if not isinstance(spec, MotionModelSpec):
        raise MotionModelConfigError("spec must be a MotionModelSpec")
    target = _positive_int(
        spec.downsample_factor, name="motion_downsample_factor"
    )
    down_t = _positive_int(vqvae_down_t, name="vqvae_down_t")
    stride_t = _positive_int(vqvae_stride_t, name="vqvae_stride_t")
    if stride_t % 2:
        raise MotionModelConfigError(
            "vqvae_stride_t must be even for the encoder padding to preserve "
            "the exact stride ** down_t temporal contract"
        )

    # For stride >= 2, stride ** down_t has at least
    # floor(log2(stride)) * down_t + 1 bits. Compare via a bounded threshold
    # rather than multiplying attacker-controlled integers or constructing the
    # power when its bit length already proves a mismatch.
    target_bits = target.bit_length()
    stride_floor_log2 = stride_t.bit_length() - 1
    mismatch_exponent = (
        target_bits + stride_floor_log2 - 1
    ) // stride_floor_log2
    if down_t >= mismatch_exponent:
        raise MotionModelConfigError(
            f"motion_downsample_factor={_bounded_int_summary(target)} disagrees with "
            "vqvae_stride_t ** vqvae_down_t "
            f"({_bounded_int_summary(stride_t)} ** "
            f"{_bounded_int_summary(down_t)})"
        )
    try:
        actual = stride_t**down_t
    except (ArithmeticError, MemoryError) as exc:
        raise MotionModelConfigError(
            "vqvae_stride_t ** vqvae_down_t could not be computed safely "
            f"({_bounded_int_summary(stride_t)} ** "
            f"{_bounded_int_summary(down_t)})"
        ) from exc
    if actual != target:
        raise MotionModelConfigError(
            f"motion_downsample_factor={_bounded_int_summary(target)} disagrees with "
            "vqvae_stride_t ** vqvae_down_t="
            f"{_bounded_int_summary(actual)}"
        )
    return actual


def resolve_motion_model_spec(
    config: Any, *, migrate_legacy: bool = False
) -> MotionModelSpec:
    """Resolve and validate all model-side motion fields."""

    if migrate_legacy:
        migrate_legacy_motion_config(config)
    projector = build_projector_spec(config)
    placeholder_token_id = _optional_token_id(
        _get(config, "motion_placeholder_token_id"),
        name="motion_placeholder_token_id",
    )
    if placeholder_token_id is None:
        raise MotionModelConfigError("motion_placeholder_token_id must not be null")
    return MotionModelSpec(
        input_dim=_positive_int(_get(config, "motion_input_dim"), name="motion_input_dim"),
        encoder_output_dim=_positive_int(
            _get(config, "motion_encoder_output_dim"),
            name="motion_encoder_output_dim",
        ),
        downsample_factor=_positive_int(
            _get(config, "motion_downsample_factor"),
            name="motion_downsample_factor",
        ),
        placeholder_token_id=placeholder_token_id,
        start_token_id=_optional_token_id(
            _get(config, "motion_start_token_id", None),
            name="motion_start_token_id",
        ),
        end_token_id=_optional_token_id(
            _get(config, "motion_end_token_id", None),
            name="motion_end_token_id",
        ),
        projector=projector,
        resize_policy=_enum(
            MotionResizePolicy,
            _get(config, "motion_resize_policy"),
            name="motion_resize_policy",
        ),
        dtype_policy=_enum(
            MotionDTypePolicy,
            _get(config, "motion_compute_dtype"),
            name="motion_compute_dtype",
        ),
        device_policy=_enum(
            MotionDevicePolicy,
            _get(config, "motion_device_policy"),
            name="motion_device_policy",
        ),
        allow_batch_broadcast=_get(config, "motion_allow_batch_broadcast"),
        normalization_mean_path=_optional_path(
            _get(config, "motion_normalization_mean_path", None),
            name="motion_normalization_mean_path",
        ),
        normalization_std_path=_optional_path(
            _get(config, "motion_normalization_std_path", None),
            name="motion_normalization_std_path",
        ),
    )
