"""Crash-safe JSON/JSONL writes using a same-directory temporary file."""

from __future__ import annotations

import json
import os
import random
import stat
import tempfile
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .paths import PathLike, UnsafePathError, resolve_within_root

_REPLACE_RETRY_TIMEOUT_SECONDS = 5.0
_REPLACE_MAX_ATTEMPTS = 64
_REPLACE_BASE_DELAY_SECONDS = 0.005
_REPLACE_MAX_DELAY_SECONDS = 0.25
_WINDOWS_RETRYABLE_WINERRORS = frozenset({5, 32})


@dataclass
class _LockEntry:
    lock: threading.Lock
    users: int = 0


_LOCKS_GUARD = threading.Lock()
_TARGET_LOCKS: dict[str, _LockEntry] = {}


def _normalized_target_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


@contextmanager
def _target_lock(path: Path) -> Iterator[None]:
    """Serialize in-process writers targeting the same normalized path."""

    key = _normalized_target_key(path)
    with _LOCKS_GUARD:
        entry = _TARGET_LOCKS.get(key)
        if entry is None:
            entry = _LockEntry(threading.Lock())
            _TARGET_LOCKS[key] = entry
        entry.users += 1

    acquired = False
    try:
        entry.lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            entry.lock.release()
        with _LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _TARGET_LOCKS.get(key) is entry:
                del _TARGET_LOCKS[key]


def _resolve_destination(path: PathLike, root: PathLike | None) -> tuple[Path, Path | None]:
    if root is not None:
        resolved_root = Path(root).resolve(strict=True)
        destination = resolve_within_root(path, root, must_exist=False)
        lexical = Path(path)
        if not lexical.is_absolute():
            lexical = resolved_root / lexical
        _reject_symlink_components(lexical, resolved_root)
    else:
        candidate = Path(path)
        if not candidate.name:
            raise UnsafePathError("destination must name a file")
        lexical = candidate if candidate.is_absolute() else Path.cwd() / candidate
        _reject_symlink_components(lexical, None)
        destination = candidate.resolve(strict=False)
        resolved_root = None
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    return destination, resolved_root


def _reject_symlink_components(path: Path, root: Path | None) -> None:
    """Reject existing links, junctions, and reparse points in a write path."""

    cursor = path
    while True:
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            info = None
        except OSError as exc:
            raise UnsafePathError(f"cannot verify path component: {cursor}") from exc
        if info is not None:
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(info.st_mode) or bool(
                getattr(info, "st_file_attributes", 0) & reparse_flag
            ):
                raise UnsafePathError(
                    f"refusing write through symlink/reparse point: {cursor}"
                )
        if root is not None and cursor == root:
            return
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def _sync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _is_retryable_windows_replace_error(error: OSError) -> bool:
    return getattr(error, "winerror", None) in _WINDOWS_RETRYABLE_WINERRORS


def _commit_temp(
    temp_path: Path,
    destination: Path,
    *,
    overwrite: bool,
    validate_target: Callable[[], None],
) -> None:
    if overwrite:
        deadline = time.monotonic() + _REPLACE_RETRY_TIMEOUT_SECONDS
        attempt = 0
        last_error: OSError | None = None
        while True:
            if last_error is not None and time.monotonic() >= deadline:
                raise last_error
            validate_target()
            try:
                os.replace(temp_path, destination)
                break
            except OSError as exc:
                attempt += 1
                last_error = exc
                now = time.monotonic()
                if (
                    not _is_retryable_windows_replace_error(exc)
                    or attempt >= _REPLACE_MAX_ATTEMPTS
                    or now >= deadline
                ):
                    raise
                exponential = _REPLACE_BASE_DELAY_SECONDS * (2 ** min(attempt - 1, 8))
                delay = min(exponential, _REPLACE_MAX_DELAY_SECONDS)
                delay *= random.uniform(0.75, 1.25)
                time.sleep(min(delay, max(0.0, deadline - now)))
    else:
        # A hard link provides an atomic create-if-absent operation.  Both
        # paths live in the same directory/filesystem by construction.
        validate_target()
        os.link(temp_path, destination)
        temp_path.unlink()
    _sync_directory(destination.parent)


def _atomic_write_lines(
    destination: Path,
    lines: Iterable[str],
    *,
    overwrite: bool,
    root: Path | None,
) -> Path:
    with _target_lock(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Re-resolve after mkdir to account for a pre-existing reparse point in
        # the parent chain.  Every later replace attempt repeats these checks.
        destination = destination.resolve(strict=False)
        if root is not None:
            revalidated = resolve_within_root(destination, root, must_exist=False)
            if revalidated != destination:
                raise UnsafePathError("destination changed during root validation")
            _reject_symlink_components(destination, root)
        else:
            _reject_symlink_components(destination, None)
        parent_identity = _path_identity(destination.parent)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(destination)
        if not overwrite and destination.exists():
            raise FileExistsError(destination)

        descriptor, raw_temp = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(raw_temp)

        def validate_target() -> None:
            if _path_identity(destination.parent) != parent_identity:
                raise UnsafePathError("destination parent changed during atomic write")
            _reject_symlink_components(destination, root)
            if root is not None:
                revalidated = resolve_within_root(destination, root, must_exist=False)
                if revalidated != destination:
                    raise UnsafePathError("destination escaped root during atomic write")
            if destination.exists() and destination.is_dir():
                raise IsADirectoryError(destination)

        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for line in lines:
                    handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            _commit_temp(
                temp_path,
                destination,
                overwrite=overwrite,
                validate_target=validate_target,
            )
        except BaseException:
            try:
                temp_path.unlink(missing_ok=True)
            finally:
                raise
    return destination


def _path_identity(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


def atomic_write_json(
    path: PathLike,
    value: Any,
    *,
    root: PathLike | None = None,
    overwrite: bool = True,
    indent: int | None = 2,
) -> Path:
    """Serialize finite JSON and atomically publish it at ``path``."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )
        payload.encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite JSON-serializable data") from exc
    destination, resolved_root = _resolve_destination(path, root)
    return _atomic_write_lines(
        destination, (payload, "\n"), overwrite=overwrite, root=resolved_root
    )


def atomic_write_jsonl(
    path: PathLike,
    rows: Iterable[Any],
    *,
    root: PathLike | None = None,
    overwrite: bool = True,
) -> Path:
    """Atomically publish newline-delimited finite JSON rows."""

    destination, resolved_root = _resolve_destination(path, root)

    def serialized_rows() -> Iterable[str]:
        for index, row in enumerate(rows):
            try:
                encoded = json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                encoded.encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {index} is not finite JSON-serializable data") from exc
            yield encoded + "\n"

    return _atomic_write_lines(
        destination, serialized_rows(), overwrite=overwrite, root=resolved_root
    )
