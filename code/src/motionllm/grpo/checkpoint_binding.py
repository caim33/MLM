"""Fail-closed binding between a live PEFT adapter and one saved checkpoint.

Formal GRPO publication must prove that the exact ``checkpoint-N`` payload on
disk is the state that existed in the training process at the final save
event.  This module deliberately supports only PEFT's safe-serialization
format.  It freezes the payload through a no-follow file descriptor before
``safetensors`` parses a private 0600 copy, then rechecks the original path to
detect replacement or mutation during inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADDITIONAL_CONFIG_NAME = "additional_config.json"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_UNSAFE_WEIGHTS_NAME = "adapter_model.bin"
_PINNED_PEFT_VERSION = "0.18.0"
_CONFIG_MAX_BYTES = 1024 * 1024


class CheckpointBindingError(RuntimeError):
    """Raised when live adapter state cannot be bound to exact disk bytes."""


@dataclass(frozen=True)
class AdapterConfigBinding:
    """Stable raw and normalized semantic evidence for adapter_config.json."""

    filename: str
    payload_sha256: str
    payload_size_bytes: int
    semantic_sha256: str
    semantics: Mapping[str, Any]


@dataclass(frozen=True)
class AdapterCheckpointBinding:
    """Frozen evidence for one exact safetensors adapter payload."""

    filename: str
    payload_sha256: str
    payload_size_bytes: int
    tensor_count: int
    state_sha256: str
    state: Mapping[str, Any]
    config: AdapterConfigBinding
    extension_config: AdapterConfigBinding | None


@dataclass(frozen=True)
class _FileCapture:
    path: Path
    identity: tuple[int, int, int, int]
    payload_sha256: str
    payload_size_bytes: int


def _path_exists_without_following(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def reject_symlink_components(path: Path) -> None:
    """Reject symlinks, including dangling leaves, in an absolute path."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise CheckpointBindingError("formal GRPO checkpoint path must be absolute")
    current = candidate
    while True:
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            metadata = None
        except OSError as exc:
            raise CheckpointBindingError(
                "formal GRPO checkpoint path metadata cannot be inspected"
            ) from exc
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise CheckpointBindingError(
                "formal GRPO checkpoint path contains a symlink"
            )
        if current.parent == current:
            return
        current = current.parent


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _open_regular_no_follow(path: Path) -> int:
    reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointBindingError(
            f"formal GRPO adapter payload cannot be opened: {path.name}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode):
            raise CheckpointBindingError(
                f"formal GRPO adapter payload is not a regular file: {path.name}"
            )
        if opened.st_size <= 0:
            raise CheckpointBindingError(
                f"formal GRPO adapter payload is empty: {path.name}"
            )
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise CheckpointBindingError(
                "formal GRPO adapter payload changed while it was opened"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:  # pragma: no cover - defensive OS contract guard
            raise CheckpointBindingError("formal GRPO frozen adapter copy was truncated")
        offset += written


@contextmanager
def _frozen_regular_file(
    path: Path, *, frozen_name: str
) -> Iterator[tuple[Path, _FileCapture]]:
    """Yield a private immutable-by-ownership copy of one stable source file."""

    source_descriptor = _open_regular_no_follow(path)
    before = os.fstat(source_descriptor)
    digest = hashlib.sha256()
    copied = 0
    with tempfile.TemporaryDirectory(prefix="motionllm-grpo-adapter-") as temporary:
        frozen_path = Path(temporary) / frozen_name
        destination_flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0)
        )
        destination_descriptor = os.open(frozen_path, destination_flags, 0o600)
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                _write_all(destination_descriptor, chunk)
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        try:
            after = os.fstat(source_descriptor)
            named = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise CheckpointBindingError(
                "formal GRPO adapter payload disappeared during capture"
            ) from exc
        finally:
            os.close(source_descriptor)
        if _identity(before) != _identity(after) or _identity(after) != _identity(named):
            raise CheckpointBindingError(
                "formal GRPO adapter payload changed during capture"
            )
        if copied != before.st_size or copied <= 0:
            raise CheckpointBindingError(
                "formal GRPO adapter payload capture has an invalid size"
            )
        try:
            os.chmod(frozen_path, 0o600)
        except OSError as exc:  # pragma: no cover - platform permission failure
            raise CheckpointBindingError(
                "formal GRPO frozen adapter copy permissions cannot be secured"
            ) from exc
        capture = _FileCapture(
            path=path,
            identity=_identity(after),
            payload_sha256=digest.hexdigest(),
            payload_size_bytes=copied,
        )
        yield frozen_path, capture


def _stable_file_digest(path: Path) -> tuple[tuple[int, int, int, int], str, int]:
    descriptor = _open_regular_no_follow(path)
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBindingError(
            "formal GRPO adapter payload changed during verification"
        ) from exc
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or _identity(after) != _identity(named):
        raise CheckpointBindingError(
            "formal GRPO adapter payload changed during verification"
        )
    if observed != before.st_size or observed <= 0:
        raise CheckpointBindingError(
            "formal GRPO adapter payload verification has an invalid size"
        )
    return _identity(after), digest.hexdigest(), observed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:  # pragma: no cover - normalized invariant
        raise CheckpointBindingError(
            "formal GRPO adapter config is not canonical finite JSON"
        ) from exc


def _normalize_config_value(value: Any, *, location: str) -> Any:
    """Normalize dataclass/Enum/set output into deterministic JSON semantics."""

    if isinstance(value, Enum):
        return _normalize_config_value(value.value, location=location)
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not __import__("math").isfinite(value):
            raise CheckpointBindingError(
                f"formal GRPO adapter config has non-finite value at {location}"
            )
        return value
    if isinstance(value, str):
        if value != value.strip() or not value:
            raise CheckpointBindingError(
                f"formal GRPO adapter config has invalid text at {location}"
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise CheckpointBindingError(
                f"formal GRPO adapter config has control text at {location}"
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or raw_key != raw_key.strip() or not raw_key:
                raise CheckpointBindingError(
                    f"formal GRPO adapter config has invalid object key at {location}"
                )
            if raw_key in normalized:
                raise CheckpointBindingError(
                    f"formal GRPO adapter config has duplicate key at {location}"
                )
            normalized[raw_key] = _normalize_config_value(
                child, location=f"{location}.{raw_key}"
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        normalized_items = [
            _normalize_config_value(child, location=f"{location}[]") for child in value
        ]
        return sorted(normalized_items, key=_canonical_json_bytes)
    if isinstance(value, (list, tuple)):
        return [
            _normalize_config_value(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CheckpointBindingError(
        f"formal GRPO adapter config has unsupported value at {location}"
    )


def _expected_lora_config_keys() -> set[str]:
    try:
        import peft
    except ImportError as exc:  # pragma: no cover - formal dependency
        raise CheckpointBindingError(
            "formal GRPO adapter config binding requires pinned PEFT"
        ) from exc
    if getattr(peft, "__version__", None) != _PINNED_PEFT_VERSION:
        raise CheckpointBindingError(
            "formal GRPO adapter config binding requires PEFT 0.18.0"
        )
    baseline = peft.LoraConfig().to_dict()
    if not isinstance(baseline, Mapping) or not baseline:
        raise CheckpointBindingError("pinned PEFT exposes no LoRA config schema")
    return {str(key) for key in baseline}


def _normalize_selector(
    value: Any, *, location: str, allow_string: bool
) -> Any:
    if value is None:
        return None
    if allow_string and isinstance(value, str):
        return _normalize_config_value(value, location=location)
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise CheckpointBindingError(
            f"formal GRPO adapter config {location} has invalid type"
        )
    values = [
        _normalize_config_value(item, location=f"{location}[]") for item in value
    ]
    if not all(isinstance(item, str) for item in values) or len(set(values)) != len(values):
        raise CheckpointBindingError(
            f"formal GRPO adapter config {location} must contain unique text"
        )
    return sorted(values)


def normalize_lora_config_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact pinned PEFT LoRA JSON schema and normalize semantics."""

    if not isinstance(config, Mapping):
        raise CheckpointBindingError(
            "formal GRPO adapter_config.json must contain one JSON object"
        )
    expected_keys = _expected_lora_config_keys()
    if set(config) != expected_keys:
        missing = sorted(expected_keys - set(config))
        extra = sorted(set(config) - expected_keys)
        raise CheckpointBindingError(
            "formal GRPO adapter config schema differs: "
            f"missing={missing}, extra={extra}"
        )
    normalized = _normalize_config_value(config, location="adapter_config")
    normalized["target_modules"] = _normalize_selector(
        config["target_modules"], location="target_modules", allow_string=True
    )
    normalized["exclude_modules"] = _normalize_selector(
        config["exclude_modules"], location="exclude_modules", allow_string=True
    )
    normalized["modules_to_save"] = _normalize_selector(
        config["modules_to_save"], location="modules_to_save", allow_string=False
    )
    if normalized["peft_type"] != "LORA":
        raise CheckpointBindingError("formal GRPO adapter config peft_type must be LORA")
    if normalized["peft_version"] != _PINNED_PEFT_VERSION:
        raise CheckpointBindingError(
            "formal GRPO adapter config peft_version differs from the runtime pin"
        )
    for key in ("r", "qalora_group_size"):
        value = normalized[key]
        if type(value) is not int or value <= 0:
            raise CheckpointBindingError(
                f"formal GRPO adapter config {key} must be a positive integer"
            )
    alpha = normalized["lora_alpha"]
    if type(alpha) not in (int, float) or alpha <= 0:
        raise CheckpointBindingError(
            "formal GRPO adapter config lora_alpha must be finite positive"
        )
    dropout = normalized["lora_dropout"]
    if type(dropout) not in (int, float) or not 0 <= float(dropout) <= 1:
        raise CheckpointBindingError(
            "formal GRPO adapter config lora_dropout must be in [0, 1]"
        )
    if normalized["bias"] not in {"none", "all", "lora_only"}:
        raise CheckpointBindingError("formal GRPO adapter config bias is invalid")
    for key in (
        "inference_mode",
        "fan_in_fan_out",
        "use_rslora",
        "use_dora",
        "use_qalora",
        "lora_bias",
        "ensure_weight_tying",
    ):
        if type(normalized[key]) is not bool:
            raise CheckpointBindingError(
                f"formal GRPO adapter config {key} must be boolean"
            )
    auto_mapping = normalized["auto_mapping"]
    if auto_mapping is not None:
        if set(auto_mapping) != {"base_model_class", "parent_library"} or not all(
            isinstance(auto_mapping[key], str)
            for key in ("base_model_class", "parent_library")
        ):
            raise CheckpointBindingError(
                "formal GRPO adapter config auto_mapping schema differs"
            )
    if normalized["target_modules"] in (None, [], ""):
        raise CheckpointBindingError(
            "formal GRPO adapter config target_modules must be non-empty"
        )
    if normalized["modules_to_save"] in (None, []):
        raise CheckpointBindingError(
            "formal GRPO adapter config modules_to_save must be non-empty"
        )
    return normalized


def adapter_config_semantic_sha256(config: Mapping[str, Any]) -> str:
    normalized = normalize_lora_config_semantics(config)
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def adapter_config_critical_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_lora_config_semantics(config)
    keys = (
        "peft_type",
        "peft_version",
        "task_type",
        "base_model_name_or_path",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "modules_to_save",
        "bias",
        "use_rslora",
        "use_dora",
        "lora_bias",
    )
    return {key: normalized[key] for key in keys}


def _parse_strict_lora_config(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointBindingError(
            "formal GRPO adapter_config.json is not strict UTF-8 JSON"
        ) from exc
    return normalize_lora_config_semantics(parsed)


def _default_peft_config(model: Any) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - formal dependency
        raise CheckpointBindingError(
            "formal GRPO adapter binding requires pinned PEFT"
        ) from exc
    if not isinstance(model, PeftModel):
        raise CheckpointBindingError(
            "formal GRPO adapter binding requires an explicit PeftModel"
        )
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or set(configs) != {"default"}:
        raise CheckpointBindingError(
            "formal GRPO adapter binding requires exactly the default PEFT adapter"
        )
    config = configs["default"]
    convert = getattr(config, "to_peft_config", None)
    if callable(convert):
        config = convert()
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise CheckpointBindingError("formal GRPO live PEFT config has no to_dict API")
    return config


def live_peft_adapter_config_semantics(
    model: Any, *, expect_trainable: bool
) -> dict[str, Any]:
    """Return the exact config semantics PEFT save_pretrained must write."""

    source_configs = getattr(model, "peft_config", None)
    if not isinstance(source_configs, Mapping) or set(source_configs) != {"default"}:
        raise CheckpointBindingError(
            "formal GRPO adapter config requires exactly the default live adapter"
        )
    source_config = source_configs["default"]
    source_inference = getattr(source_config, "inference_mode", None)
    if type(source_inference) is not bool or source_inference is expect_trainable:
        expected_text = "trainable" if expect_trainable else "frozen"
        raise CheckpointBindingError(
            f"formal GRPO live PEFT config is not in expected {expected_text} mode"
        )
    config = _default_peft_config(model)
    semantics = normalize_lora_config_semantics(config.to_dict())
    semantics["inference_mode"] = True
    if semantics["task_type"] is None:
        get_base_model = getattr(model, "get_base_model", None)
        if not callable(get_base_model):
            raise CheckpointBindingError(
                "formal GRPO cannot derive PEFT auto_mapping from the live base model"
            )
        base_class = get_base_model().__class__
        semantics["auto_mapping"] = {
            "base_model_class": base_class.__name__,
            "parent_library": base_class.__module__,
        }
    else:
        semantics["auto_mapping"] = None
    return normalize_lora_config_semantics(semantics)


def require_identical_adapter_configs(
    live_config: Mapping[str, Any], disk_config: Mapping[str, Any]
) -> str:
    live = normalize_lora_config_semantics(live_config)
    disk = normalize_lora_config_semantics(disk_config)
    if live != disk:
        differing = sorted(
            key for key in live if live.get(key) != disk.get(key)
        )
        raise CheckpointBindingError(
            "formal GRPO live/disk adapter config semantics differ: "
            f"fields={differing}"
        )
    digest = adapter_config_semantic_sha256(live)
    if digest != adapter_config_semantic_sha256(disk):  # pragma: no cover
        raise CheckpointBindingError(
            "formal GRPO live/disk adapter config semantic hash differs"
        )
    return digest


def capture_adapter_config(checkpoint: Path) -> AdapterConfigBinding:
    config_path = Path(checkpoint) / ADAPTER_CONFIG_NAME
    reject_symlink_components(config_path)
    with _frozen_regular_file(config_path, frozen_name=ADAPTER_CONFIG_NAME) as (
        frozen_path,
        capture,
    ):
        if capture.payload_size_bytes > _CONFIG_MAX_BYTES:
            raise CheckpointBindingError(
                "formal GRPO adapter_config.json exceeds the size limit"
            )
        raw = frozen_path.read_bytes()
        if len(raw) != capture.payload_size_bytes:
            raise CheckpointBindingError(
                "formal GRPO frozen adapter config copy changed during inspection"
            )
        semantics = _parse_strict_lora_config(raw)
        identity, digest, size = _stable_file_digest(config_path)
        if (
            identity != capture.identity
            or digest != capture.payload_sha256
            or size != capture.payload_size_bytes
        ):
            raise CheckpointBindingError(
                "formal GRPO adapter_config.json changed while it was inspected"
            )
    return AdapterConfigBinding(
        filename=ADAPTER_CONFIG_NAME,
        payload_sha256=capture.payload_sha256,
        payload_size_bytes=capture.payload_size_bytes,
        semantic_sha256=adapter_config_semantic_sha256(semantics),
        semantics=semantics,
    )


def normalize_swift_lora_extension_semantics(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize ms-swift 4.2.2's mandatory additional_config.json."""

    required = {"lora_dtype", "lorap_lr_ratio", "lorap_emb_lr"}
    if not isinstance(config, Mapping) or set(config) != required:
        missing = sorted(required - set(config)) if isinstance(config, Mapping) else sorted(required)
        extra = sorted(set(config) - required) if isinstance(config, Mapping) else []
        raise CheckpointBindingError(
            "formal GRPO Swift LoRA extension config schema differs: "
            f"missing={missing}, extra={extra}"
        )
    normalized = _normalize_config_value(config, location="additional_config")
    dtype = normalized["lora_dtype"]
    if dtype not in {None, "float16", "bfloat16", "float32"}:
        raise CheckpointBindingError(
            "formal GRPO Swift LoRA extension lora_dtype is invalid"
        )
    ratio = normalized["lorap_lr_ratio"]
    if ratio is not None and (
        type(ratio) not in (int, float) or float(ratio) <= 0
    ):
        raise CheckpointBindingError(
            "formal GRPO Swift LoRA extension lorap_lr_ratio is invalid"
        )
    embedding_lr = normalized["lorap_emb_lr"]
    if type(embedding_lr) not in (int, float) or float(embedding_lr) <= 0:
        raise CheckpointBindingError(
            "formal GRPO Swift LoRA extension lorap_emb_lr is invalid"
        )
    return normalized


def _parse_strict_swift_lora_extension(raw: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointBindingError(
            "formal GRPO additional_config.json is not strict UTF-8 JSON"
        ) from exc
    return normalize_swift_lora_extension_semantics(parsed)


def _extension_semantic_sha256(config: Mapping[str, Any]) -> str:
    normalized = normalize_swift_lora_extension_semantics(config)
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def capture_swift_lora_extension_config(checkpoint: Path) -> AdapterConfigBinding:
    path = Path(checkpoint) / ADDITIONAL_CONFIG_NAME
    reject_symlink_components(path)
    with _frozen_regular_file(path, frozen_name=ADDITIONAL_CONFIG_NAME) as (
        frozen_path,
        capture,
    ):
        if capture.payload_size_bytes > _CONFIG_MAX_BYTES:
            raise CheckpointBindingError(
                "formal GRPO additional_config.json exceeds the size limit"
            )
        raw = frozen_path.read_bytes()
        if len(raw) != capture.payload_size_bytes:
            raise CheckpointBindingError(
                "formal GRPO frozen Swift extension config changed during inspection"
            )
        semantics = _parse_strict_swift_lora_extension(raw)
        identity, digest, size = _stable_file_digest(path)
        if (
            identity != capture.identity
            or digest != capture.payload_sha256
            or size != capture.payload_size_bytes
        ):
            raise CheckpointBindingError(
                "formal GRPO additional_config.json changed while it was inspected"
            )
    return AdapterConfigBinding(
        filename=ADDITIONAL_CONFIG_NAME,
        payload_sha256=capture.payload_sha256,
        payload_size_bytes=capture.payload_size_bytes,
        semantic_sha256=_extension_semantic_sha256(semantics),
        semantics=semantics,
    )


def live_swift_lora_extension_semantics(model: Any) -> dict[str, Any]:
    source_configs = getattr(model, "peft_config", None)
    if not isinstance(source_configs, Mapping) or set(source_configs) != {"default"}:
        raise CheckpointBindingError(
            "formal GRPO Swift extension requires exactly the default live adapter"
        )
    source = source_configs["default"]
    to_dict = getattr(source, "to_dict", None)
    to_peft_config = getattr(source, "to_peft_config", None)
    if not callable(to_dict) or not callable(to_peft_config):
        raise CheckpointBindingError(
            "formal GRPO live adapter is not the pinned ms-swift LoRA config"
        )
    raw = to_dict()
    expected_keys = _expected_lora_config_keys() | {
        "lora_dtype",
        "lorap_lr_ratio",
        "lorap_emb_lr",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise CheckpointBindingError(
            "formal GRPO live Swift LoRA config schema differs from core plus extension"
        )
    return normalize_swift_lora_extension_semantics(
        {key: raw[key] for key in ("lora_dtype", "lorap_lr_ratio", "lorap_emb_lr")}
    )


def _tensor_storage_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().cpu().contiguous()
    if not torch.is_tensor(value) or not torch.is_floating_point(value):
        raise CheckpointBindingError(
            "formal GRPO adapter state must contain only floating tensors"
        )
    if value.numel() <= 0:
        raise CheckpointBindingError("formal GRPO adapter state contains an empty tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise CheckpointBindingError(
            "formal GRPO adapter state contains a non-finite tensor"
        )
    return value.view(torch.uint8).numpy().tobytes()


def normalize_adapter_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Detach, validate, and clone a complete PEFT saveable state mapping."""

    import torch

    if not isinstance(state, Mapping) or not state:
        raise CheckpointBindingError("formal GRPO adapter state is empty")
    normalized: dict[str, Any] = {}
    for raw_name, raw_tensor in state.items():
        if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
            raise CheckpointBindingError("formal GRPO adapter state has an invalid key")
        if any(ord(character) < 32 or ord(character) == 127 for character in raw_name):
            raise CheckpointBindingError("formal GRPO adapter state key contains control text")
        if raw_name in normalized:
            raise CheckpointBindingError("formal GRPO adapter state contains duplicate keys")
        if not torch.is_tensor(raw_tensor):
            raise CheckpointBindingError(
                f"formal GRPO adapter state value is not a tensor: {raw_name}"
            )
        tensor = raw_tensor.detach().cpu().contiguous().clone()
        _tensor_storage_bytes(tensor)
        normalized[raw_name] = tensor
    return normalized


def canonical_adapter_state_sha256(state: Mapping[str, Any]) -> str:
    normalized = normalize_adapter_state(state)
    digest = hashlib.sha256()
    for name in sorted(normalized):
        tensor = normalized[name]
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = _tensor_storage_bytes(tensor)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def require_identical_adapter_states(
    live_state: Mapping[str, Any], disk_state: Mapping[str, Any]
) -> str:
    """Require exact keys, dtype, shape, and storage bytes; return canonical hash."""

    live = normalize_adapter_state(live_state)
    disk = normalize_adapter_state(disk_state)
    if set(live) != set(disk):
        missing = sorted(set(live) - set(disk))
        extra = sorted(set(disk) - set(live))
        raise CheckpointBindingError(
            "formal GRPO checkpoint adapter key drift: "
            f"missing={missing}, extra={extra}"
        )
    for name in sorted(live):
        before = live[name]
        saved = disk[name]
        if before.dtype != saved.dtype or tuple(before.shape) != tuple(saved.shape):
            raise CheckpointBindingError(
                f"formal GRPO checkpoint adapter tensor metadata differs: {name}"
            )
        if _tensor_storage_bytes(before) != _tensor_storage_bytes(saved):
            raise CheckpointBindingError(
                f"formal GRPO checkpoint adapter tensor bytes differ: {name}"
            )
    live_hash = canonical_adapter_state_sha256(live)
    disk_hash = canonical_adapter_state_sha256(disk)
    if live_hash != disk_hash:  # pragma: no cover - byte comparison already proves this
        raise CheckpointBindingError(
            "formal GRPO checkpoint canonical adapter state hash differs"
        )
    return live_hash


def _saveable_name_for_trainable(name: str) -> str:
    """Map only default-adapter LoRA/modules_to_save names to PEFT save keys."""

    modules_marker = ".modules_to_save.default."
    if modules_marker in name:
        if name.count(modules_marker) != 1:
            raise CheckpointBindingError(
                "formal GRPO trainable modules_to_save name is ambiguous"
            )
        return name.replace(modules_marker, ".", 1)
    components = name.split(".")
    default_positions = [
        index
        for index, component in enumerate(components)
        if component == "default"
        and index > 0
        and components[index - 1].startswith("lora_")
    ]
    if len(default_positions) == 1:
        del components[default_positions[0]]
        return ".".join(components)
    raise CheckpointBindingError(
        f"formal GRPO trainable parameter is not default LoRA/modules_to_save: {name}"
    )


def _require_frozen_embedding_extras(
    model: Any, extras: set[str], saved: Mapping[str, Any]
) -> None:
    """Allow only PEFT auto-saved, frozen input/output embedding tensors."""

    if not extras:
        return
    embedding_layers: list[Any] = []
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, getter_name, None)
        if callable(getter):
            layer = getter()
            if layer is not None and all(layer is not value for value in embedding_layers):
                embedding_layers.append(layer)
    if not embedding_layers:
        raise CheckpointBindingError(
            "formal GRPO saveable state has unrecognized non-trainable extras"
        )
    try:
        named_modules = list(model.named_modules(remove_duplicate=False))
        named_parameters = dict(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - pinned torch API guard
        raise CheckpointBindingError(
            "formal GRPO runtime lacks duplicate-aware module enumeration"
        ) from exc
    embedding_prefixes: set[str] = set()
    for layer in embedding_layers:
        base_layer = getattr(layer, "base_layer", None)
        for name, module in named_modules:
            if name and (module is layer or module is base_layer):
                embedding_prefixes.add(str(name))
    if not embedding_prefixes:
        raise CheckpointBindingError(
            "formal GRPO cannot identify the auto-saved embedding module paths"
        )
    full_state = model.state_dict()
    for name in sorted(extras):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in embedding_prefixes):
            raise CheckpointBindingError(
                f"formal GRPO saveable state has a non-whitelisted extra tensor: {name}"
            )
        parameter = named_parameters.get(name)
        full_tensor = full_state.get(name)
        if parameter is None or full_tensor is None or parameter.requires_grad:
            raise CheckpointBindingError(
                f"formal GRPO auto-saved embedding extra is not one frozen parameter: {name}"
            )
        saved_tensor = saved[name]
        if parameter.dtype != saved_tensor.dtype or tuple(parameter.shape) != tuple(
            saved_tensor.shape
        ):
            raise CheckpointBindingError(
                f"formal GRPO frozen embedding metadata differs from saveable state: {name}"
            )
        if (
            _tensor_storage_bytes(parameter)
            != _tensor_storage_bytes(full_tensor)
            or _tensor_storage_bytes(full_tensor)
            != _tensor_storage_bytes(saved_tensor)
        ):
            raise CheckpointBindingError(
                f"formal GRPO frozen embedding bytes differ from saveable state: {name}"
            )


def require_trainable_saveable_coverage(
    model: Any, saveable_state: Mapping[str, Any]
) -> None:
    """Prove a one-to-one mapping from every trainable tensor to saved state."""

    trainable = [
        (str(name), parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise CheckpointBindingError(
            "formal GRPO PeftModel exposes no trainable parameters"
        )
    canonical: dict[str, Any] = {}
    for name, parameter in trainable:
        saved_name = _saveable_name_for_trainable(name)
        if saved_name in canonical:
            raise CheckpointBindingError(
                "formal GRPO trainable parameters map to a duplicate save key"
            )
        canonical[saved_name] = parameter
    saved = normalize_adapter_state(saveable_state)
    if not set(canonical).issubset(saved):
        missing = sorted(set(canonical) - set(saved))
        raise CheckpointBindingError(
            "formal GRPO trainable/saveable coverage differs: "
            f"missing={missing}"
        )
    for name in sorted(canonical):
        parameter = canonical[name].detach().cpu().contiguous()
        saved_tensor = saved[name]
        if parameter.dtype != saved_tensor.dtype or tuple(parameter.shape) != tuple(
            saved_tensor.shape
        ):
            raise CheckpointBindingError(
                f"formal GRPO trainable/saveable tensor metadata differs: {name}"
            )
        if _tensor_storage_bytes(parameter) != _tensor_storage_bytes(saved_tensor):
            raise CheckpointBindingError(
                f"formal GRPO trainable/saveable tensor bytes differ: {name}"
            )
    _require_frozen_embedding_extras(model, set(saved) - set(canonical), saved)


def live_peft_adapter_state(model: Any) -> dict[str, Any]:
    """Extract a trainable PEFT payload and prove trainable/saveable coverage."""

    normalized = extract_peft_adapter_state(model)
    require_trainable_saveable_coverage(model, normalized)
    return normalized


def extract_peft_adapter_state(model: Any) -> dict[str, Any]:
    """Extract PEFT saveable state without requiring inference reload trainables."""

    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - formal runtime dependency
        raise CheckpointBindingError(
            "formal GRPO checkpoint binding requires pinned PEFT"
        ) from exc
    _default_peft_config(model)
    state = get_peft_model_state_dict(
        model,
        adapter_name="default",
        unwrap_compiled=False,
        save_embedding_layers="auto",
    )
    return normalize_adapter_state(state)


def _read_frozen_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - formal runtime dependency
        raise CheckpointBindingError(
            "formal GRPO checkpoint binding requires pinned safetensors"
        ) from exc
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise CheckpointBindingError(
                    "formal GRPO safetensors adapter contains no tensors"
                )
            if len(set(keys)) != len(keys):  # pragma: no cover - safetensors invariant
                raise CheckpointBindingError(
                    "formal GRPO safetensors adapter contains duplicate keys"
                )
            state = {name: handle.get_tensor(name) for name in keys}
    except CheckpointBindingError:
        raise
    except Exception as exc:
        raise CheckpointBindingError(
            "formal GRPO safetensors adapter cannot be parsed"
        ) from exc
    return normalize_adapter_state(state)


def capture_adapter_checkpoint(
    checkpoint: Path, *, require_swift_extension: bool = False
) -> AdapterCheckpointBinding:
    """Capture one exact, non-symlinked PEFT safetensors checkpoint leaf."""

    artifact = Path(checkpoint)
    reject_symlink_components(artifact)
    try:
        artifact_metadata = os.stat(artifact, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBindingError(
            "formal GRPO exact checkpoint leaf does not exist"
        ) from exc
    if not stat.S_ISDIR(artifact_metadata.st_mode):
        raise CheckpointBindingError(
            "formal GRPO exact checkpoint leaf is not a directory"
        )
    adapter_config = artifact / ADAPTER_CONFIG_NAME
    reject_symlink_components(adapter_config)
    try:
        config_metadata = os.stat(adapter_config, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBindingError(
            "formal GRPO checkpoint lacks adapter_config.json"
        ) from exc
    if not stat.S_ISREG(config_metadata.st_mode) or config_metadata.st_size <= 0:
        raise CheckpointBindingError(
            "formal GRPO checkpoint adapter_config.json is not one non-empty regular file"
        )
    unsafe_weights = artifact / ADAPTER_UNSAFE_WEIGHTS_NAME
    if _path_exists_without_following(unsafe_weights):
        raise CheckpointBindingError(
            "formal GRPO checkpoint forbids adapter_model.bin; safetensors is mandatory"
        )
    weights = artifact / ADAPTER_SAFE_WEIGHTS_NAME
    if not _path_exists_without_following(weights):
        raise CheckpointBindingError(
            "formal GRPO checkpoint lacks adapter_model.safetensors"
        )
    config_binding = capture_adapter_config(artifact)
    extension_path = artifact / ADDITIONAL_CONFIG_NAME
    extension_binding: AdapterConfigBinding | None = None
    if _path_exists_without_following(extension_path):
        extension_binding = capture_swift_lora_extension_config(artifact)
    elif require_swift_extension:
        raise CheckpointBindingError(
            "formal GRPO checkpoint lacks additional_config.json from pinned ms-swift"
        )
    with _frozen_regular_file(
        weights, frozen_name=ADAPTER_SAFE_WEIGHTS_NAME
    ) as (frozen_path, capture):
        state = _read_frozen_safetensors(frozen_path)
        identity, digest, size = _stable_file_digest(weights)
        if (
            identity != capture.identity
            or digest != capture.payload_sha256
            or size != capture.payload_size_bytes
        ):
            raise CheckpointBindingError(
                "formal GRPO adapter payload changed while safetensors was inspected"
            )
    state_hash = canonical_adapter_state_sha256(state)
    return AdapterCheckpointBinding(
        filename=ADAPTER_SAFE_WEIGHTS_NAME,
        payload_sha256=capture.payload_sha256,
        payload_size_bytes=capture.payload_size_bytes,
        tensor_count=len(state),
        state_sha256=state_hash,
        state=state,
        config=config_binding,
        extension_config=extension_binding,
    )


def bind_live_peft_adapter_to_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    require_swift_extension: bool = False,
) -> AdapterCheckpointBinding:
    """Prove that live PEFT saveable state exactly equals the disk payload."""

    live_state = live_peft_adapter_state(model)
    live_config = live_peft_adapter_config_semantics(model, expect_trainable=True)
    binding = capture_adapter_checkpoint(
        checkpoint, require_swift_extension=require_swift_extension
    )
    state_hash = require_identical_adapter_states(live_state, binding.state)
    if state_hash != binding.state_sha256:  # pragma: no cover - defensive guard
        raise CheckpointBindingError(
            "formal GRPO live adapter hash differs from captured checkpoint hash"
        )
    config_hash = require_identical_adapter_configs(
        live_config, binding.config.semantics
    )
    if config_hash != binding.config.semantic_sha256:  # pragma: no cover
        raise CheckpointBindingError(
            "formal GRPO live adapter config hash differs from captured checkpoint"
        )
    if require_swift_extension:
        if binding.extension_config is None:  # pragma: no cover - capture invariant
            raise CheckpointBindingError(
                "formal GRPO checkpoint has no Swift extension config binding"
            )
        live_extension = live_swift_lora_extension_semantics(model)
        if live_extension != binding.extension_config.semantics:
            differing = sorted(
                key
                for key in live_extension
                if live_extension.get(key)
                != binding.extension_config.semantics.get(key)
            )
            raise CheckpointBindingError(
                "formal GRPO live/disk Swift LoRA extension semantics differ: "
                f"fields={differing}"
            )
        if (
            _extension_semantic_sha256(live_extension)
            != binding.extension_config.semantic_sha256
        ):
            raise CheckpointBindingError(
                "formal GRPO live/disk Swift extension semantic hash differs"
            )
    return binding


__all__ = [
    "ADDITIONAL_CONFIG_NAME",
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_SAFE_WEIGHTS_NAME",
    "AdapterConfigBinding",
    "AdapterCheckpointBinding",
    "CheckpointBindingError",
    "adapter_config_critical_fields",
    "adapter_config_semantic_sha256",
    "bind_live_peft_adapter_to_checkpoint",
    "canonical_adapter_state_sha256",
    "capture_adapter_config",
    "capture_adapter_checkpoint",
    "capture_swift_lora_extension_config",
    "extract_peft_adapter_state",
    "live_peft_adapter_state",
    "live_peft_adapter_config_semantics",
    "live_swift_lora_extension_semantics",
    "normalize_lora_config_semantics",
    "normalize_adapter_state",
    "normalize_swift_lora_extension_semantics",
    "reject_symlink_components",
    "require_identical_adapter_states",
    "require_identical_adapter_configs",
    "require_trainable_saveable_coverage",
]
