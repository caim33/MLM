"""Torch-free, config-driven motion projector specifications."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ProjectorSpecError


_MISSING = object()
_NO_DEFAULT = object()
_SUPPORTED_ACTIVATIONS = frozenset({"gelu", "relu", "silu", "tanh", "identity"})


def _config_value(config: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(config, Mapping):
        if key in config:
            return config[key]
    elif hasattr(config, key):
        return getattr(config, key)
    if default is _NO_DEFAULT:
        raise ProjectorSpecError(f"missing required config field {key!r}")
    return default


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ProjectorSpecError(f"{name} must be a positive integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ProjectorSpecError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ProjectorSpecError(f"{name} must be > 0, got {result}")
    return result


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectorSpecError(f"{name} must be bool, got {value!r}")
    return value


def _prefix(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectorSpecError(f"{name} must be a non-empty string")
    return value


def _hidden_dims(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise ProjectorSpecError(
            "motion_projector_hidden_dims must be an integer sequence"
        )
    try:
        single = operator.index(value)
    except TypeError:
        if not isinstance(value, Sequence):
            raise ProjectorSpecError(
                "motion_projector_hidden_dims must be an integer sequence"
            )
        items = tuple(value)
    else:
        items = (single,)
    return tuple(
        _positive_int(item, name=f"motion_projector_hidden_dims[{index}]")
        for index, item in enumerate(items)
    )


@dataclass(frozen=True, slots=True)
class LinearLayerSpec:
    """Shape and stable Sequential index of one projector linear layer."""

    module_index: int
    in_features: int
    out_features: int
    bias: bool

    def __post_init__(self) -> None:
        if isinstance(self.module_index, bool):
            raise ProjectorSpecError("module_index must be a non-negative integer")
        try:
            module_index = operator.index(self.module_index)
        except TypeError as exc:
            raise ProjectorSpecError(
                "module_index must be a non-negative integer"
            ) from exc
        if module_index < 0:
            raise ProjectorSpecError("module_index must be >= 0")
        _positive_int(self.in_features, name="in_features")
        _positive_int(self.out_features, name="out_features")
        _bool(self.bias, name="bias")

    @property
    def parameter_count(self) -> int:
        return self.in_features * self.out_features + (
            self.out_features if self.bias else 0
        )


@dataclass(frozen=True, slots=True)
class ProjectorSpec:
    """Complete shape-level contract for the motion-to-text projector."""

    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int
    activation: str = "gelu"
    bias: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    state_dict_prefix: str = "motion_proj"
    pre_norm_prefix: str = "motion_prenorm"
    post_norm_prefix: str = "motion_postnorm"

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, name="input_dim")
        normalized_hidden = _hidden_dims(self.hidden_dims)
        if normalized_hidden != self.hidden_dims:
            raise ProjectorSpecError("hidden_dims must be a tuple of positive integers")
        _positive_int(self.output_dim, name="output_dim")
        if not isinstance(self.activation, str):
            raise ProjectorSpecError("activation must be a string")
        if self.activation.lower() not in _SUPPORTED_ACTIVATIONS:
            raise ProjectorSpecError(
                f"unsupported activation {self.activation!r}; expected one of "
                f"{sorted(_SUPPORTED_ACTIVATIONS)!r}"
            )
        if self.activation != self.activation.lower():
            raise ProjectorSpecError("activation must be lowercase")
        _bool(self.bias, name="bias")
        _bool(self.pre_norm, name="pre_norm")
        _bool(self.post_norm, name="post_norm")
        _prefix(self.state_dict_prefix, name="state_dict_prefix")
        _prefix(self.pre_norm_prefix, name="pre_norm_prefix")
        _prefix(self.post_norm_prefix, name="post_norm_prefix")

    @property
    def dimensions(self) -> tuple[int, ...]:
        return (self.input_dim, *self.hidden_dims, self.output_dim)

    @property
    def linear_layers(self) -> tuple[LinearLayerSpec, ...]:
        dimensions = self.dimensions
        # A Sequential implementation places an activation after every hidden
        # linear, so compatible linear module indices are 0, 2, 4, ... .
        return tuple(
            LinearLayerSpec(
                module_index=index * 2,
                in_features=dimensions[index],
                out_features=dimensions[index + 1],
                bias=self.bias,
            )
            for index in range(len(dimensions) - 1)
        )

    @property
    def parameter_count(self) -> int:
        count = sum(layer.parameter_count for layer in self.linear_layers)
        if self.pre_norm:
            count += 2 * self.input_dim
        if self.post_norm:
            count += 2 * self.output_dim
        return count

    def infer_output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        if isinstance(input_shape, (str, bytes, bytearray)):
            raise ProjectorSpecError("input_shape must be an integer sequence")
        try:
            shape = tuple(input_shape)
        except TypeError as exc:
            raise ProjectorSpecError("input_shape must be an integer sequence") from exc
        if not shape:
            raise ProjectorSpecError("input_shape must have at least one dimension")

        normalized: list[int] = []
        for index, value in enumerate(shape):
            if isinstance(value, bool):
                raise ProjectorSpecError(
                    f"input_shape[{index}] must be a non-negative integer"
                )
            try:
                dimension = operator.index(value)
            except TypeError as exc:
                raise ProjectorSpecError(
                    f"input_shape[{index}] must be a non-negative integer"
                ) from exc
            if dimension < 0:
                raise ProjectorSpecError(
                    f"input_shape[{index}] must be >= 0, got {dimension}"
                )
            normalized.append(dimension)
        if normalized[-1] != self.input_dim:
            raise ProjectorSpecError(
                f"projector expected final dimension {self.input_dim}; "
                f"got {normalized[-1]}"
            )
        normalized[-1] = self.output_dim
        return tuple(normalized)

    @classmethod
    def from_config(cls, config: Any) -> "ProjectorSpec":
        """Build a spec from explicit config, with dimension-only aliases.

        The input may come from ``motion_projector_input_dim`` or the legacy
        config's ``vqvae_output_emb_width``.  The output may come from
        ``motion_projector_output_dim`` or ``text_config.hidden_size``.  Hidden
        dimensions are intentionally mandatory so architecture cannot silently
        fall back to the old hard-coded 4096.
        """

        input_dim = _config_value(config, "motion_projector_input_dim", _MISSING)
        if input_dim is _MISSING:
            input_dim = _config_value(config, "vqvae_output_emb_width")

        output_dim = _config_value(config, "motion_projector_output_dim", _MISSING)
        if output_dim is _MISSING:
            text_config = _config_value(config, "text_config", _MISSING)
            if text_config is _MISSING:
                output_dim = _config_value(config, "text_hidden_size")
            else:
                output_dim = _config_value(text_config, "hidden_size")

        hidden = _config_value(config, "motion_projector_hidden_dims", _MISSING)
        if hidden is _MISSING:
            hidden = _config_value(config, "motion_projector_hidden_dim", _MISSING)
        if hidden is _MISSING:
            raise ProjectorSpecError(
                "missing required config field 'motion_projector_hidden_dims'; "
                "use an explicit empty list for a direct linear projector"
            )

        return cls(
            input_dim=_positive_int(input_dim, name="motion_projector_input_dim"),
            hidden_dims=_hidden_dims(hidden),
            output_dim=_positive_int(output_dim, name="motion_projector_output_dim"),
            activation=str(
                _config_value(config, "motion_projector_activation", "gelu")
            ).lower(),
            bias=_bool(
                _config_value(config, "motion_projector_bias", True),
                name="motion_projector_bias",
            ),
            pre_norm=_bool(
                _config_value(config, "motion_projector_pre_norm", True),
                name="motion_projector_pre_norm",
            ),
            post_norm=_bool(
                _config_value(config, "motion_projector_post_norm", True),
                name="motion_projector_post_norm",
            ),
            state_dict_prefix=_prefix(
                _config_value(config, "motion_projector_state_dict_prefix", "motion_proj"),
                name="motion_projector_state_dict_prefix",
            ),
            pre_norm_prefix=_prefix(
                _config_value(config, "motion_pre_norm_state_dict_prefix", "motion_prenorm"),
                name="motion_pre_norm_state_dict_prefix",
            ),
            post_norm_prefix=_prefix(
                _config_value(config, "motion_post_norm_state_dict_prefix", "motion_postnorm"),
                name="motion_post_norm_state_dict_prefix",
            ),
        )


def build_projector_spec(config: Any) -> ProjectorSpec:
    """Functional spelling of :meth:`ProjectorSpec.from_config`."""

    return ProjectorSpec.from_config(config)


def infer_projector_output_shape(
    input_shape: Sequence[int], spec: ProjectorSpec
) -> tuple[int, ...]:
    """Validate an input shape and return the projector output shape."""

    if not isinstance(spec, ProjectorSpec):
        raise ProjectorSpecError("spec must be ProjectorSpec")
    return spec.infer_output_shape(input_shape)
