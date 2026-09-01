"""Explicit model-family selection for SFT adapters.

Checkpoint paths are data, never a model-type discriminator.  This module has
no import-time dependency on torch or transformers so its contracts can be
tested in a CPU-only control environment.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping


class ModelFactoryError(ValueError):
    """Raised when an explicit model selection cannot be satisfied."""


class ModelFamily(str, Enum):
    QWEN2_VL = "qwen2_vl"
    QWEN2_5_VL = "qwen2_5_vl"
    QWEN3_VL = "qwen3_vl"
    QWEN3_VL_MOE = "qwen3_vl_moe"
    QWEN3_VL_MOTION = "qwen3_vl_motion"


@dataclass(frozen=True)
class ModelFactorySpec:
    family: ModelFamily
    class_path: str
    data_model_type: str
    is_moe: bool = False
    supports_motion: bool = False

    def __post_init__(self) -> None:
        if "." not in self.class_path:
            raise ValueError("class_path must be a fully qualified object name")
        if not self.data_model_type.strip():
            raise ValueError("data_model_type must be non-empty")


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    processor: Any
    spec: ModelFactorySpec


def _import_object(path: str) -> Any:
    module_name, _, object_name = path.rpartition(".")
    if not module_name or not object_name:
        raise ModelFactoryError(f"invalid class path: {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ModelFactoryError(f"model class is unavailable: {path}") from exc


DEFAULT_MODEL_SPECS = (
    ModelFactorySpec(
        ModelFamily.QWEN2_VL,
        "transformers.Qwen2VLForConditionalGeneration",
        "qwen2vl",
    ),
    ModelFactorySpec(
        ModelFamily.QWEN2_5_VL,
        "transformers.Qwen2_5_VLForConditionalGeneration",
        "qwen2.5vl",
    ),
    ModelFactorySpec(
        ModelFamily.QWEN3_VL,
        "transformers.Qwen3VLForConditionalGeneration",
        "qwen3vl",
    ),
    ModelFactorySpec(
        ModelFamily.QWEN3_VL_MOE,
        "transformers.Qwen3VLMoeForConditionalGeneration",
        "qwen3vl",
        is_moe=True,
    ),
    ModelFactorySpec(
        ModelFamily.QWEN3_VL_MOTION,
        "models.qwen3_vl_motion.Qwen3VlMotionForConditionalGeneration",
        "qwen3vl",
        supports_motion=True,
    ),
)


class ExplicitModelFactory:
    """Resolve and load a model from an explicit, immutable family registry."""

    def __init__(
        self,
        specs: tuple[ModelFactorySpec, ...] = DEFAULT_MODEL_SPECS,
        *,
        class_resolver: Callable[[str], Any] = _import_object,
        processor_loader: Callable[..., Any] | None = None,
    ) -> None:
        registry: dict[ModelFamily, ModelFactorySpec] = {}
        for spec in specs:
            if spec.family in registry:
                raise ValueError(f"duplicate model family: {spec.family.value}")
            registry[spec.family] = spec
        self._registry = registry
        self._class_resolver = class_resolver
        self._processor_loader = processor_loader

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(family.value for family in self._registry))

    def spec_for(self, family: ModelFamily | str) -> ModelFactorySpec:
        try:
            normalized = family if isinstance(family, ModelFamily) else ModelFamily(family)
        except (TypeError, ValueError) as exc:
            raise ModelFactoryError(
                f"unknown model family {family!r}; choose one of {self.families}"
            ) from exc
        try:
            return self._registry[normalized]
        except KeyError as exc:  # pragma: no cover - defensive custom-registry guard
            raise ModelFactoryError(f"unregistered model family: {normalized.value}") from exc

    def load_model(
        self,
        *,
        family: ModelFamily | str,
        model_name_or_path: str,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ModelFactorySpec]:
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ModelFactoryError("model_name_or_path must be a non-empty string")
        spec = self.spec_for(family)
        model_class = self._class_resolver(spec.class_path)
        loader = getattr(model_class, "from_pretrained", None)
        if not callable(loader):
            raise ModelFactoryError(f"{spec.class_path} has no callable from_pretrained")
        model = loader(model_name_or_path, **dict(model_kwargs or {}))
        return model, spec

    def load_bundle(
        self,
        *,
        family: ModelFamily | str,
        model_name_or_path: str,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> ModelBundle:
        model, spec = self.load_model(
            family=family,
            model_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs,
        )
        loader = self._processor_loader
        if loader is None:
            auto_processor = _import_object("transformers.AutoProcessor")
            loader = auto_processor.from_pretrained
        processor = loader(model_name_or_path, **dict(processor_kwargs or {}))
        return ModelBundle(model=model, processor=processor, spec=spec)


default_model_factory = ExplicitModelFactory()
