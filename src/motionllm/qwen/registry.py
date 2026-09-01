"""Explicit, file-backed dataset aliases for the Qwen compatibility layer.

The legacy project stored machine-specific absolute paths in Python globals.
This module deliberately resolves one immutable JSON document per alias instead.
Importing it does not inspect the filesystem; resolution happens only when
``data_list`` or ``load_dataset_config`` is called.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_CONFIG_DIR_ENV = "MOTIONLLM_DATASET_CONFIG_DIR"
_ALIAS = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAMPLING_SUFFIX = re.compile(r"%(\d+)\Z")
_ALLOWED_FIELDS = {
    "schema_version",
    "name",
    "annotation_path",
    "media_root",
    "split",
    "motion_mean_path",
    "motion_std_path",
    "expected_motion_dim",
}
_RUNTIME_CONFIGS: dict[str, "DatasetConfig"] = {}
_RUNTIME_LOCK = threading.RLock()


class DatasetRegistryError(ValueError):
    """A dataset alias or its configuration is invalid."""


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise DatasetRegistryError(f"dataset config contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_nonfinite(value: str) -> None:
    raise DatasetRegistryError(f"dataset config contains non-finite number {value}")


def _absolute_path(value: Any, *, field_name: str, kind: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DatasetRegistryError(f"{field_name} must be a non-empty absolute path")
    if "\x00" in value or "://" in value:
        raise DatasetRegistryError(f"{field_name} must be a local filesystem path")
    if value.startswith("~"):
        raise DatasetRegistryError(f"{field_name} must not depend on a user-home expansion")
    path = Path(value)
    if not path.is_absolute():
        raise DatasetRegistryError(f"{field_name} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DatasetRegistryError(f"{field_name} does not exist: {path}") from exc
    if kind == "file" and not resolved.is_file():
        raise DatasetRegistryError(f"{field_name} must be a regular file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise DatasetRegistryError(f"{field_name} must be a directory: {resolved}")
    return resolved


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetRegistryError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """One validated alias binding with no implicit machine defaults."""

    name: str
    annotation_path: Path
    media_root: Path
    split: str
    config_path: Path | None
    motion_mean_path: Path | None = None
    motion_std_path: Path | None = None
    expected_motion_dim: int | None = None

    def to_legacy_mapping(self, *, sampling_rate: float = 1.0) -> dict[str, Any]:
        """Return the narrow mapping expected by the historical loader API."""

        if not math.isfinite(sampling_rate) or not 0.0 < sampling_rate <= 1.0:
            raise DatasetRegistryError("sampling_rate must be in (0, 1]")
        value: dict[str, Any] = {
            "name": self.name,
            "annotation_path": str(self.annotation_path),
            "data_path": str(self.media_root),
            "media_root": str(self.media_root),
            "split": self.split,
            "sampling_rate": sampling_rate,
        }
        if self.config_path is not None:
            value["config_path"] = str(self.config_path)
        if self.motion_mean_path is not None:
            value["motion_mean_path"] = str(self.motion_mean_path)
            value["motion_std_path"] = str(self.motion_std_path)
        if self.expected_motion_dim is not None:
            value["expected_motion_dim"] = self.expected_motion_dim
        return value


def default_config_dir() -> Path:
    """Resolve the explicit environment override or repository config folder."""

    override = os.environ.get(DATASET_CONFIG_DIR_ENV)
    selected = (
        Path(override)
        if override
        else Path(__file__).resolve().parents[3] / "configs" / "datasets"
    )
    try:
        resolved = selected.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DatasetRegistryError(f"dataset config directory does not exist: {selected}") from exc
    if not resolved.is_dir():
        raise DatasetRegistryError(f"dataset config path is not a directory: {resolved}")
    return resolved


def parse_sampling_rate(dataset_name: str) -> float:
    """Parse the legacy ``alias%N`` suffix without weakening alias validation."""

    if not isinstance(dataset_name, str) or not dataset_name:
        raise DatasetRegistryError("dataset alias must be a non-empty string")
    match = _SAMPLING_SUFFIX.search(dataset_name)
    if match is None:
        return 1.0
    percentage = int(match.group(1))
    if not 1 <= percentage <= 100:
        raise DatasetRegistryError("dataset sampling percentage must be in [1, 100]")
    return percentage / 100.0


def _base_alias(dataset_name: str) -> str:
    alias = _SAMPLING_SUFFIX.sub("", dataset_name)
    if not _ALIAS.fullmatch(alias):
        raise DatasetRegistryError(
            "dataset alias may contain only letters, digits, dot, underscore, and hyphen"
        )
    return alias


def load_dataset_config(
    alias: str,
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> DatasetConfig:
    """Load and validate exactly ``<alias>.dataset.json``."""

    name = _base_alias(alias)
    with _RUNTIME_LOCK:
        runtime = _RUNTIME_CONFIGS.get(name)
    if runtime is not None:
        return runtime
    if config_dir is None:
        root = default_config_dir()
    else:
        try:
            root = Path(config_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DatasetRegistryError(
                f"dataset config directory does not exist: {config_dir}"
            ) from exc
    if not root.is_dir():
        raise DatasetRegistryError(f"dataset config path is not a directory: {root}")
    config_path = (root / f"{name}.dataset.json").resolve(strict=False)
    if config_path.parent != root:
        raise DatasetRegistryError("dataset alias escapes the config directory")
    if not config_path.is_file():
        raise DatasetRegistryError(
            f"dataset alias {name!r} has no config file {config_path}; "
            f"set {DATASET_CONFIG_DIR_ENV} or pass config_dir explicitly"
        )
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_nonfinite,
        )
    except DatasetRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetRegistryError(f"failed to read dataset config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise DatasetRegistryError("dataset config root must be an object")
    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        raise DatasetRegistryError(f"dataset config contains unknown fields: {sorted(unknown)!r}")
    if payload.get("schema_version") != 1:
        raise DatasetRegistryError("dataset config schema_version must be 1")
    if payload.get("name") != name:
        raise DatasetRegistryError(
            f"dataset config name must exactly match alias {name!r}"
        )
    split = payload.get("split")
    if not isinstance(split, str) or not split.strip() or split != split.strip():
        raise DatasetRegistryError("dataset split must be a non-empty trimmed string")

    mean_raw = payload.get("motion_mean_path")
    std_raw = payload.get("motion_std_path")
    if (mean_raw is None) != (std_raw is None):
        raise DatasetRegistryError(
            "motion_mean_path and motion_std_path must be configured together"
        )
    mean_path = (
        _absolute_path(mean_raw, field_name="motion_mean_path", kind="file")
        if mean_raw is not None
        else None
    )
    std_path = (
        _absolute_path(std_raw, field_name="motion_std_path", kind="file")
        if std_raw is not None
        else None
    )
    expected_dim_raw = payload.get("expected_motion_dim")
    expected_dim = (
        _positive_int(expected_dim_raw, field_name="expected_motion_dim")
        if expected_dim_raw is not None
        else None
    )
    if expected_dim is not None and mean_path is None:
        raise DatasetRegistryError(
            "expected_motion_dim requires explicit normalization asset paths"
        )
    return DatasetConfig(
        name=name,
        annotation_path=_absolute_path(
            payload.get("annotation_path"), field_name="annotation_path", kind="file"
        ),
        media_root=_absolute_path(
            payload.get("media_root"), field_name="media_root", kind="directory"
        ),
        split=split,
        config_path=config_path.resolve(strict=True),
        motion_mean_path=mean_path,
        motion_std_path=std_path,
        expected_motion_dim=expected_dim,
    )


def register_dataset(
    name: str,
    annotation_path: str | os.PathLike[str],
    data_path: str | os.PathLike[str],
    *,
    split: str | None = None,
    replace: bool = False,
    motion_mean_path: str | os.PathLike[str] | None = None,
    motion_std_path: str | os.PathLike[str] | None = None,
    expected_motion_dim: int | None = None,
) -> DatasetConfig:
    """Register one validated process-scoped alias for an explicit CLI path.

    This is the supported replacement for mutating the historical ``data_dict``.
    Registration is opt-in, rejects accidental replacement by default, and never
    persists machine paths into the source tree.
    """

    alias = _base_alias(name)
    if not isinstance(replace, bool):
        raise DatasetRegistryError("replace must be a boolean")
    selected_split = "unspecified" if split is None else split
    if (
        not isinstance(selected_split, str)
        or not selected_split.strip()
        or selected_split != selected_split.strip()
    ):
        raise DatasetRegistryError("split must be a non-empty trimmed string")
    if (motion_mean_path is None) != (motion_std_path is None):
        raise DatasetRegistryError(
            "motion_mean_path and motion_std_path must be registered together"
        )
    mean_path = (
        _absolute_path(str(motion_mean_path), field_name="motion_mean_path", kind="file")
        if motion_mean_path is not None
        else None
    )
    std_path = (
        _absolute_path(str(motion_std_path), field_name="motion_std_path", kind="file")
        if motion_std_path is not None
        else None
    )
    if expected_motion_dim is not None:
        expected_motion_dim = _positive_int(
            expected_motion_dim, field_name="expected_motion_dim"
        )
        if mean_path is None:
            raise DatasetRegistryError(
                "expected_motion_dim requires explicit normalization asset paths"
            )
    config = DatasetConfig(
        name=alias,
        annotation_path=_absolute_path(
            str(annotation_path), field_name="annotation_path", kind="file"
        ),
        media_root=_absolute_path(str(data_path), field_name="data_path", kind="directory"),
        split=selected_split,
        config_path=None,
        motion_mean_path=mean_path,
        motion_std_path=std_path,
        expected_motion_dim=expected_motion_dim,
    )
    with _RUNTIME_LOCK:
        if alias in _RUNTIME_CONFIGS and not replace:
            raise DatasetRegistryError(
                f"runtime dataset alias {alias!r} is already registered; pass replace=True explicitly"
            )
        _RUNTIME_CONFIGS[alias] = config
    return config


def data_list(
    dataset_names: Iterable[str],
    *,
    config_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility resolver used by Full/LoRA SFT and inference."""

    if isinstance(dataset_names, (str, bytes)):
        raise DatasetRegistryError("dataset_names must be an iterable of aliases, not one string")
    result: list[dict[str, Any]] = []
    for raw_name in dataset_names:
        rate = parse_sampling_rate(raw_name)
        config = load_dataset_config(_base_alias(raw_name), config_dir=config_dir)
        result.append(config.to_legacy_mapping(sampling_rate=rate))
    if not result:
        raise DatasetRegistryError("at least one dataset alias is required")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_motion_normalization_binding(
    dataset_uses: Iterable[str],
    *,
    motion_mean_path: str | os.PathLike[str] | None,
    motion_std_path: str | os.PathLike[str] | None,
    expected_motion_dim: int | None = None,
    config_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Bind every motion dataset and the model CLI to one exact stats pair."""

    if (motion_mean_path is None) != (motion_std_path is None):
        raise DatasetRegistryError(
            "motion normalization mean/std CLI paths must be provided together"
        )
    if motion_mean_path is None:
        raise DatasetRegistryError(
            "motion training requires explicit normalization mean/std CLI paths"
        )
    mean = _absolute_path(
        str(motion_mean_path), field_name="motion_mean_path", kind="file"
    )
    std = _absolute_path(
        str(motion_std_path), field_name="motion_std_path", kind="file"
    )
    aliases: list[str] = []
    for raw in dataset_uses:
        if not isinstance(raw, str):
            raise DatasetRegistryError("dataset_use entries must be strings")
        aliases.extend(part for part in raw.split(",") if part)
    if not aliases:
        raise DatasetRegistryError("motion training requires at least one dataset alias")

    mean_hash = _sha256(mean)
    std_hash = _sha256(std)
    for alias in aliases:
        config = load_dataset_config(_base_alias(alias), config_dir=config_dir)
        if config.motion_mean_path is None or config.motion_std_path is None:
            raise DatasetRegistryError(
                f"motion dataset {config.name!r} has no explicit normalization assets"
            )
        if (
            config.motion_mean_path != mean
            or config.motion_std_path != std
            or _sha256(config.motion_mean_path) != mean_hash
            or _sha256(config.motion_std_path) != std_hash
        ):
            raise DatasetRegistryError(
                f"motion dataset {config.name!r} normalization does not match the model CLI"
            )
        if expected_motion_dim is not None:
            expected = _positive_int(
                expected_motion_dim, field_name="expected_motion_dim"
            )
            if config.expected_motion_dim != expected:
                raise DatasetRegistryError(
                    f"motion dataset {config.name!r} expected_motion_dim does not match the model"
                )
    return mean, std


__all__ = [
    "DATASET_CONFIG_DIR_ENV",
    "DatasetConfig",
    "DatasetRegistryError",
    "data_list",
    "default_config_dir",
    "load_dataset_config",
    "parse_sampling_rate",
    "register_dataset",
    "validate_motion_normalization_binding",
]
