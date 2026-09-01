"""Root-confined path resolution used before controller filesystem writes."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | os.PathLike[str]
_URI_PREFIX = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")


class UnsafePathError(ValueError):
    """Base class for a path that cannot be safely used."""


class PathOutsideRootError(UnsafePathError):
    """Raised when a candidate resolves outside its declared root."""


def _as_path(value: PathLike, field_name: str) -> Path:
    try:
        raw = os.fspath(value)
    except (TypeError, ValueError) as exc:
        raise UnsafePathError(f"invalid {field_name}: {exc}") from exc
    if not isinstance(raw, str):
        raise UnsafePathError(f"{field_name} must not be a bytes path")
    if not raw or not raw.strip():
        raise UnsafePathError(f"{field_name} cannot be empty")
    if "\x00" in raw:
        raise UnsafePathError(f"{field_name} contains a null byte")
    if _URI_PREFIX.match(raw):
        raise UnsafePathError(f"{field_name} must be a filesystem path, not a URI")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in raw):
        raise UnsafePathError(f"{field_name} contains an invalid Unicode surrogate")
    candidate = Path(raw)
    if candidate.drive and not candidate.is_absolute():
        raise UnsafePathError(f"{field_name} must not use a drive-relative path")
    if candidate.is_reserved():
        raise UnsafePathError(f"{field_name} uses a reserved filesystem name")
    if os.name == "nt":
        without_drive = raw[len(candidate.drive) :]
        if ":" in without_drive:
            raise UnsafePathError(f"{field_name} must not use an NTFS alternate data stream")
        if raw.startswith(("\\\\?\\", "\\\\.\\")):
            raise UnsafePathError(f"{field_name} must not use a Windows device path")
    return candidate


def resolve_within_root(
    path: PathLike,
    root: PathLike,
    *,
    must_exist: bool = False,
    allow_root: bool = False,
) -> Path:
    """Resolve ``path`` and prove it is confined beneath ``root``.

    Relative candidates are interpreted relative to ``root``.  ``resolve``
    also accounts for existing symlink/junction components, so a link that
    escapes the root is rejected.  The check is suitable for validation and
    must be repeated immediately before security-sensitive writes to reduce
    time-of-check/time-of-use exposure.
    """

    root_path = _as_path(root, "root")
    try:
        resolved_root = root_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafePathError(f"root cannot be resolved: {root_path}") from exc
    if not resolved_root.is_dir():
        raise UnsafePathError(f"root is not a directory: {resolved_root}")

    candidate = _as_path(path, "path")
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafePathError(f"path cannot be resolved: {candidate}") from exc

    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PathOutsideRootError(
            f"path resolves outside declared root: {resolved}"
        ) from exc
    if not allow_root and not relative.parts:
        raise UnsafePathError("path must name an entry below root, not root itself")
    if must_exist and not resolved.exists():
        # Handles dangling paths on platforms where resolve(strict=True) has
        # implementation-specific behavior.
        raise FileNotFoundError(resolved)
    return resolved
