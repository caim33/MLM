"""Root-confined path resolution for dataset and media references."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath
from typing import Literal

from .errors import MediaNotFoundError, PathResolutionError, UnsafePathError


PathKind = Literal["file", "directory", "any"]
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    try:
        raw_root = os.fspath(root)
    except TypeError as exc:
        raise PathResolutionError("root must be a filesystem path") from exc
    if isinstance(raw_root, bytes) or not raw_root or "\x00" in raw_root:
        raise PathResolutionError("root must be a non-empty text path")
    try:
        resolved = Path(raw_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathResolutionError("declared root does not exist or cannot be resolved") from exc
    if not resolved.is_dir():
        raise PathResolutionError(f"declared root is not a directory: {resolved}")
    return resolved


def _validate_windows_reference(text: str) -> None:
    if os.name != "nt":
        return
    normalized = text.replace("/", "\\")
    if normalized.startswith(("\\\\.\\", "\\\\?\\")):
        raise UnsafePathError("Windows device paths are not permitted")

    parsed = PureWindowsPath(text)
    if parsed.drive and not parsed.root:
        raise UnsafePathError("drive-relative paths are not permitted")

    drive = parsed.drive.casefold()
    anchor = parsed.anchor.casefold()
    for part in parsed.parts:
        if part.casefold() in {drive, anchor} or part in ("\\", "/"):
            continue
        if ":" in part:
            raise UnsafePathError("Windows alternate data stream paths are not permitted")
        stem = part.rstrip(" .").split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise UnsafePathError("Windows reserved device names are not permitted")


def _coerce_reference(reference: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(reference)
    except TypeError as exc:
        raise PathResolutionError("path reference must be a string or path-like value") from exc
    if isinstance(raw, bytes):
        raise PathResolutionError("path reference must be text, not bytes")
    if not raw or not raw.strip():
        raise PathResolutionError("path reference must not be empty")
    if "\x00" in raw:
        raise PathResolutionError("path reference contains a null byte")
    if _URI_PATTERN.match(raw) or raw.casefold().startswith("file:"):
        raise PathResolutionError("URI references are not filesystem paths")
    _validate_windows_reference(raw)
    return Path(raw)


def resolve_path_within_root(
    root: str | os.PathLike[str],
    reference: str | os.PathLike[str],
    *,
    expected_kind: PathKind = "file",
    must_exist: bool = True,
) -> Path:
    """Resolve ``reference`` while confining it to an existing directory root.

    Existing symlinks are resolved before containment is checked.  A missing
    leaf may be allowed explicitly, but traversal outside the root never is.
    """

    if expected_kind not in ("file", "directory", "any"):
        raise PathResolutionError("expected_kind must be file, directory, or any")
    if not isinstance(must_exist, bool):
        raise PathResolutionError("must_exist must be a boolean")

    resolved_root = _resolve_root(root)
    relative_or_absolute = _coerce_reference(reference)
    candidate = (
        relative_or_absolute
        if relative_or_absolute.is_absolute()
        else resolved_root / relative_or_absolute
    )
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PathResolutionError("path reference cannot be resolved") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafePathError(
            f"resolved path is outside the declared root: {resolved_root}"
        ) from exc

    exists = resolved.exists()
    if must_exist and not exists:
        raise MediaNotFoundError(f"required path does not exist: {resolved}")
    if exists:
        if expected_kind == "file" and not resolved.is_file():
            raise PathResolutionError(f"path must be a regular file: {resolved}")
        if expected_kind == "directory" and not resolved.is_dir():
            raise PathResolutionError(f"path must be a directory: {resolved}")
    return resolved


def resolve_media_path(
    root: str | os.PathLike[str],
    reference: str | os.PathLike[str],
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one media reference as a root-confined regular file."""

    return resolve_path_within_root(
        root,
        reference,
        expected_kind="file",
        must_exist=must_exist,
    )
