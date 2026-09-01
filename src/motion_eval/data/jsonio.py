"""Strict, dependency-free JSON readers for controller evidence.

The readers deliberately reject Python's permissive JSON extensions.  Error
messages identify the source and, for JSONL, the line without echoing input
payloads that may contain credentials or other sensitive data.
"""

from __future__ import annotations

import json
import math
import stat
from pathlib import Path
from typing import Any


class StrictJsonError(ValueError):
    """A JSON/JSONL source is unreadable or violates the strict contract."""


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & reparse_flag
    )


def _source_file(path: str | Path) -> Path:
    try:
        candidate = Path(path)
    except (TypeError, ValueError) as exc:
        raise StrictJsonError("JSON source path is invalid") from exc
    try:
        if _is_link_or_reparse(candidate):
            raise StrictJsonError(f"JSON source must not be a link/reparse point: {candidate}")
        resolved = candidate.resolve(strict=True)
    except StrictJsonError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise StrictJsonError(f"JSON source cannot be resolved: {candidate}") from exc
    if not resolved.is_file():
        raise StrictJsonError(f"JSON source must be a regular file: {resolved}")
    return resolved


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJsonError(f"non-finite JSON number is forbidden: {value}")


def _validate_finite_unicode(value: Any, *, location: str = "JSON") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJsonError(f"{location} contains a non-finite number")
        return
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise StrictJsonError(f"{location} contains an invalid Unicode surrogate")
        return
    if isinstance(value, list):
        for child in value:
            _validate_finite_unicode(child, location=location)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_unicode(key, location=location)
            _validate_finite_unicode(child, location=location)
        return
    raise StrictJsonError(f"{location} contains an unsupported decoded value")


def _decode(text: str, *, source: Path, line_number: int | None = None) -> Any:
    location = str(source) if line_number is None else f"{source}: line {line_number}"
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
        _validate_finite_unicode(value, location=location)
        return value
    except StrictJsonError as exc:
        raise StrictJsonError(f"{location}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StrictJsonError(
            f"{location}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except RecursionError as exc:
        raise StrictJsonError(f"{location}: JSON nesting exceeds the parser limit") from exc
    except (TypeError, ValueError) as exc:
        raise StrictJsonError(f"{location}: invalid JSON value") from exc


def _read_utf8(source: Path) -> str:
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise StrictJsonError(f"cannot read JSON source: {source}") from exc
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJsonError(f"JSON source is not valid UTF-8: {source}") from exc


def load_json_strict(path: str | Path) -> Any:
    """Load one finite RFC-style JSON value without duplicate object keys."""

    source = _source_file(path)
    text = _read_utf8(source)
    if not text.strip():
        raise StrictJsonError(f"JSON source is empty: {source}")
    return _decode(text, source=source)


def load_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL file whose every physical row is an object."""

    source = _source_file(path)
    text = _read_utf8(source)
    if text == "":
        raise StrictJsonError(f"JSONL source is empty: {source}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise StrictJsonError(f"{source}: blank JSONL row at line {line_number}")
        value = _decode(line, source=source, line_number=line_number)
        if not isinstance(value, dict):
            raise StrictJsonError(
                f"{source}: line {line_number}: JSONL row must be an object"
            )
        rows.append(value)
    if not rows:
        raise StrictJsonError(f"JSONL source is empty: {source}")
    return rows


__all__ = ["StrictJsonError", "load_json_strict", "load_jsonl_strict"]
