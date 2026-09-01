"""Auditable filesystem primitives for the evaluation controller."""

from .atomic import atomic_write_json, atomic_write_jsonl
from .hashing import (
    DirectoryCapture,
    FileChangedDuringHashError,
    HashingError,
    PathDigest,
    SymlinkNotAllowedError,
    directory_merkle_sha256,
    capture_directory_bytes,
    hash_path,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from .paths import PathOutsideRootError, UnsafePathError, resolve_within_root
from .source_inventory import (
    SOURCE_MANIFEST_SCHEMA,
    bytecode_source,
    formal_source_role_files,
    is_link_or_reparse,
    stable_source_files,
)

__all__ = [
    "FileChangedDuringHashError",
    "HashingError",
    "PathDigest",
    "PathOutsideRootError",
    "SymlinkNotAllowedError",
    "SOURCE_MANIFEST_SCHEMA",
    "UnsafePathError",
    "atomic_write_json",
    "atomic_write_jsonl",
    "directory_merkle_sha256",
    "DirectoryCapture",
    "capture_directory_bytes",
    "bytecode_source",
    "formal_source_role_files",
    "hash_path",
    "is_link_or_reparse",
    "resolve_within_root",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "stable_source_files",
]
