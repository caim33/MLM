"""Strict, payload-safe JSONL reading primitives."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import JsonlLineError, JsonlOpenError


def _reject_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON numbers are not permitted")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = child
    return value


def _source_path(source: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(source)
    except TypeError as exc:
        raise JsonlOpenError(Path("<invalid>"), "source must be a filesystem path") from exc
    if isinstance(raw, bytes) or not raw or "\x00" in raw:
        raise JsonlOpenError(Path("<invalid>"), "source must be a non-empty text path")
    try:
        return Path(raw).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise JsonlOpenError(Path(raw), "source path cannot be resolved") from exc


def _split_jsonl(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def read_jsonl(source: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file with strict JSON and object-only rows."""

    path = _source_path(source)
    if not path.exists():
        raise JsonlOpenError(path, "source does not exist")
    if not path.is_file():
        raise JsonlOpenError(path, "source is not a regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise JsonlOpenError(path, "source could not be read") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        line_number = payload[: exc.start].count(b"\n") + 1
        raise JsonlLineError(path, line_number, "row is not valid UTF-8") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_split_jsonl(text), start=1):
        if not line.strip():
            raise JsonlLineError(path, line_number, "blank JSONL row is not permitted")
        try:
            parsed = json.loads(
                line,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise JsonlLineError(
                path,
                line_number,
                f"malformed JSON at column {exc.colno}",
            ) from exc
        except RecursionError as exc:
            raise JsonlLineError(
                path,
                line_number,
                "JSON nesting exceeds the supported depth",
            ) from exc
        except ValueError as exc:
            raise JsonlLineError(path, line_number, str(exc)) from exc
        if not isinstance(parsed, dict):
            raise JsonlLineError(path, line_number, "JSONL row must be an object")
        rows.append(parsed)
    return rows
