"""Strict data reading, path, message, dataset, and collation contracts."""

from .collation import CollationPlan, logical_sample_payload, plan_collation
from .dataset import SampleDataset
from .errors import (
    CollationContractError,
    DataContractError,
    DuplicateSampleIdError,
    JsonlLineError,
    JsonlOpenError,
    MediaNotFoundError,
    MessageContractError,
    PathResolutionError,
    UnsafePathError,
)
from .jsonl import read_jsonl
from .messages import (
    LegacySampleDescriptor,
    build_legacy_messages,
    describe_legacy_sample,
    infer_legacy_modality,
)
from .paths import resolve_media_path, resolve_path_within_root
from .samples import read_samples_jsonl

__all__ = [
    "CollationContractError",
    "CollationPlan",
    "DataContractError",
    "DuplicateSampleIdError",
    "JsonlLineError",
    "JsonlOpenError",
    "LegacySampleDescriptor",
    "MediaNotFoundError",
    "MessageContractError",
    "PathResolutionError",
    "SampleDataset",
    "UnsafePathError",
    "build_legacy_messages",
    "describe_legacy_sample",
    "infer_legacy_modality",
    "logical_sample_payload",
    "plan_collation",
    "read_jsonl",
    "read_samples_jsonl",
    "resolve_media_path",
    "resolve_path_within_root",
]
