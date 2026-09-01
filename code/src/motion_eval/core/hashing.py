"""Content SHA-256 and deterministic directory Merkle hashing."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .paths import PathLike, resolve_within_root

_CHUNK_SIZE = 1024 * 1024
_MERKLE_DOMAIN = b"motion-eval-directory-merkle-v1\0"
SymlinkPolicy = Literal["reject", "link", "follow"]


class HashingError(RuntimeError):
    """Base class for evidence that could not be hashed safely."""


class SymlinkNotAllowedError(HashingError):
    pass


class FileChangedDuringHashError(HashingError):
    pass


@dataclass(frozen=True)
class PathDigest:
    algorithm: str
    kind: str
    digest: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "kind": self.kind,
            "digest": self.digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class DirectoryCapture:
    """One stable directory generation represented by its exact file bytes."""

    receipt: PathDigest
    files: tuple[tuple[str, bytes], ...]


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    return hashlib.sha256(data).hexdigest()


def _stable_stat(path: Path) -> tuple[int, int, int, int, int, int]:
    info = path.stat()
    return _stat_identity(info)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _is_link_stat(info: os.stat_result) -> bool:
    """Treat POSIX links and Windows reparse points/junctions as links."""

    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _link_info(path: Path) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return info if _is_link_stat(info) else None


def _canonical_link_entry(
    path: Path,
    *,
    allowed_root: PathLike | None,
    expected: os.stat_result,
) -> tuple[Path, os.stat_result]:
    """Resolve only the parent and prove the link entry itself is confined."""

    lexical = path if path.is_absolute() else Path.cwd() / path
    if lexical.name in {"", ".", ".."}:
        raise HashingError(f"invalid link entry path: {path}")
    try:
        parent = lexical.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileChangedDuringHashError(
            f"link parent cannot be resolved safely: {path}"
        ) from exc
    if allowed_root is not None:
        parent = resolve_within_root(
            parent, allowed_root, must_exist=True, allow_root=True
        )
    entry = parent / lexical.name
    current = _link_info(entry)
    if current is None or _stat_identity(current) != _stat_identity(expected):
        raise FileChangedDuringHashError(
            f"link identity changed during containment validation: {path}"
        )
    return entry, current


def _read_link_stably(path: Path, before: os.stat_result) -> bytes:
    parent_before = _stable_stat(path.parent)
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise HashingError(f"cannot read link/reparse target safely: {path}") from exc
    after = _link_info(path)
    if (
        after is None
        or _stat_identity(after) != _stat_identity(before)
        or _stable_stat(path.parent) != parent_before
    ):
        raise FileChangedDuringHashError(f"link changed while hashing: {path}")
    return os.fsencode(target)


def _sha256_file_details(path: Path, *, chunk_size: int) -> tuple[str, int]:
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise FileChangedDuringHashError(f"file disappeared before hashing: {path}") from exc
    if _is_link_stat(path_before):
        raise SymlinkNotAllowedError(f"refusing to hash symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HashingError(f"cannot open file safely for hashing: {path}") from exc
    try:
        before = _stat_identity(os.fstat(descriptor))
        if (path_before.st_dev, path_before.st_ino) != before[:2] or not stat.S_ISREG(
            before[2]
        ):
            raise FileChangedDuringHashError(f"file identity changed before hashing: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, chunk_size)
            if not block:
                break
            digest.update(block)
        after = _stat_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise FileChangedDuringHashError(f"file disappeared while hashing: {path}") from exc
    if (
        after != before
        or _is_link_stat(path_after)
        or (path_after.st_dev, path_after.st_ino) != before[:2]
    ):
        raise FileChangedDuringHashError(f"file changed while hashing: {path}")
    return digest.hexdigest(), before[3]


def _capture_file_details(path: Path, *, chunk_size: int) -> tuple[bytes, str, int]:
    """Read one stable regular-file generation and return its exact bytes."""

    try:
        path_before = path.lstat()
    except OSError as exc:
        raise FileChangedDuringHashError(f"file disappeared before capture: {path}") from exc
    if _is_link_stat(path_before):
        raise SymlinkNotAllowedError(f"refusing to capture symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HashingError(f"cannot open file safely for capture: {path}") from exc
    try:
        before = _stat_identity(os.fstat(descriptor))
        if (path_before.st_dev, path_before.st_ino) != before[:2] or not stat.S_ISREG(
            before[2]
        ):
            raise FileChangedDuringHashError(f"file identity changed before capture: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, chunk_size)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = _stat_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise FileChangedDuringHashError(f"file disappeared while being captured: {path}") from exc
    if (
        after != before
        or _is_link_stat(path_after)
        or (path_after.st_dev, path_after.st_ino) != before[:2]
    ):
        raise FileChangedDuringHashError(f"file changed while being captured: {path}")
    payload = b"".join(chunks)
    return payload, digest.hexdigest(), before[3]


def sha256_file(
    path: PathLike,
    *,
    chunk_size: int = _CHUNK_SIZE,
    follow_symlinks: bool = False,
) -> str:
    candidate = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if _link_info(candidate) is not None:
        if not follow_symlinks:
            raise SymlinkNotAllowedError(f"refusing to hash symlink: {candidate}")
        candidate = candidate.resolve(strict=True)
    if not candidate.is_file():
        raise FileNotFoundError(f"not a regular file: {candidate}")

    digest, _ = _sha256_file_details(candidate, chunk_size=chunk_size)
    return digest


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite canonical JSON") from exc
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _frame(*parts: bytes) -> bytes:
    framed = bytearray()
    for part in parts:
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def _directory_digest(
    root: Path,
    *,
    symlink_policy: SymlinkPolicy,
    allowed_root: Path | None,
) -> PathDigest:
    if symlink_policy not in {"reject", "link", "follow"}:
        raise ValueError("symlink_policy must be 'reject', 'link', or 'follow'")

    digest = hashlib.sha256(_MERKLE_DOMAIN)
    file_count = 0
    total_bytes = 0

    def record(kind: bytes, relative: str, size: int, content_digest: str) -> None:
        digest.update(
            _frame(
                kind,
                relative.encode("utf-8"),
                str(size).encode("ascii"),
                content_digest.encode("ascii"),
            )
        )

    def visit(directory: Path, logical_prefix: tuple[str, ...], ancestors: set[tuple[int, int]]) -> None:
        nonlocal file_count, total_bytes
        directory_before = _stable_stat(directory)
        info = directory.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in ancestors:
            raise HashingError(f"directory cycle detected while hashing: {directory}")
        branch_ancestors = ancestors | {identity}

        entries = sorted(os.scandir(directory), key=lambda item: item.name.encode("utf-8"))
        for entry in entries:
            relative_parts = logical_prefix + (entry.name,)
            relative = "/".join(relative_parts)
            entry_path = Path(entry.path)

            link_info = _link_info(entry_path)
            if link_info is not None:
                if symlink_policy == "reject":
                    raise SymlinkNotAllowedError(f"refusing directory symlink: {entry_path}")
                canonical_entry, link_info = _canonical_link_entry(
                    entry_path,
                    allowed_root=allowed_root,
                    expected=link_info,
                )
                target_bytes = _read_link_stably(canonical_entry, link_info)
                if symlink_policy == "link":
                    record(b"L", relative, len(target_bytes), sha256_bytes(target_bytes))
                    continue
                resolved = canonical_entry.resolve(strict=True)
                if allowed_root is not None:
                    resolve_within_root(resolved, allowed_root, must_exist=True, allow_root=True)
                target_info = resolved.stat()
                if stat.S_ISDIR(target_info.st_mode):
                    record(b"D", relative, 0, "")
                    visit(resolved, relative_parts, branch_ancestors)
                elif stat.S_ISREG(target_info.st_mode):
                    file_digest, stable_size = _sha256_file_details(
                        resolved, chunk_size=_CHUNK_SIZE
                    )
                    record(b"F", relative, stable_size, file_digest)
                    file_count += 1
                    total_bytes += stable_size
                else:
                    raise HashingError(f"unsupported symlink target type: {entry_path}")
                continue

            entry_info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_info.st_mode):
                record(b"D", relative, 0, "")
                visit(entry_path, relative_parts, branch_ancestors)
            elif stat.S_ISREG(entry_info.st_mode):
                file_digest, stable_size = _sha256_file_details(
                    entry_path, chunk_size=_CHUNK_SIZE
                )
                record(b"F", relative, stable_size, file_digest)
                file_count += 1
                total_bytes += stable_size
            else:
                raise HashingError(f"unsupported filesystem entry: {entry_path}")

        if _stable_stat(directory) != directory_before:
            raise FileChangedDuringHashError(
                f"directory changed while hashing: {directory}"
            )

    visit(root, (), set())
    return PathDigest(
        algorithm="sha256-directory-merkle-v1",
        kind="directory",
        digest=digest.hexdigest(),
        file_count=file_count,
        total_bytes=total_bytes,
    )


def directory_merkle_sha256(
    path: PathLike,
    *,
    symlink_policy: SymlinkPolicy = "reject",
    allowed_root: PathLike | None = None,
) -> str:
    candidate = Path(path)
    link_info = _link_info(candidate)
    if link_info is not None:
        if symlink_policy == "reject":
            raise SymlinkNotAllowedError(f"refusing root symlink: {candidate}")
        if symlink_policy == "link":
            raise HashingError("a root symlink cannot be represented as a directory tree")
        candidate, _ = _canonical_link_entry(
            candidate, allowed_root=allowed_root, expected=link_info
        )
        candidate = candidate.resolve(strict=True)
    if not candidate.is_dir():
        raise FileNotFoundError(f"not a directory: {candidate}")
    confined_root = Path(allowed_root).resolve(strict=True) if allowed_root is not None else None
    if confined_root is not None:
        candidate = resolve_within_root(
            candidate, confined_root, must_exist=True, allow_root=True
        )
    return _directory_digest(
        candidate,
        symlink_policy=symlink_policy,
        allowed_root=confined_root,
    ).digest


def capture_directory_bytes(
    path: PathLike,
    *,
    allowed_root: PathLike | None = None,
    max_files: int = 4096,
    max_total_bytes: int = 64 * 1024 * 1024,
) -> DirectoryCapture:
    """Capture and Merkle-hash the same exact directory generation.

    Unlike a separate ``hash_path`` followed by ordinary reads, the Merkle
    digest here is computed from the bytes returned to the caller.  This makes
    the capture suitable for a verified in-memory Python import bundle and
    closes the hash-to-import path replacement window.
    """

    if type(max_files) is not int or max_files <= 0:
        raise ValueError("max_files must be a positive integer")
    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be a positive integer")
    candidate = Path(path)
    if _link_info(candidate) is not None:
        raise SymlinkNotAllowedError(f"refusing root symlink: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"not a directory: {candidate}")
    confined_root = Path(allowed_root).resolve(strict=True) if allowed_root is not None else None
    if confined_root is not None:
        candidate = resolve_within_root(
            candidate, confined_root, must_exist=True, allow_root=True
        )
    else:
        candidate = candidate.resolve(strict=True)

    digest = hashlib.sha256(_MERKLE_DOMAIN)
    files: list[tuple[str, bytes]] = []
    total_bytes = 0

    def record(kind: bytes, relative: str, size: int, content_digest: str) -> None:
        digest.update(
            _frame(
                kind,
                relative.encode("utf-8"),
                str(size).encode("ascii"),
                content_digest.encode("ascii"),
            )
        )

    def visit(
        directory: Path,
        logical_prefix: tuple[str, ...],
        ancestors: set[tuple[int, int]],
    ) -> None:
        nonlocal total_bytes
        directory_before = _stable_stat(directory)
        info = directory.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in ancestors:
            raise HashingError(f"directory cycle detected while capturing: {directory}")
        branch_ancestors = ancestors | {identity}
        entries = sorted(os.scandir(directory), key=lambda item: item.name.encode("utf-8"))
        for entry in entries:
            relative_parts = logical_prefix + (entry.name,)
            relative = "/".join(relative_parts)
            entry_path = Path(entry.path)
            if _link_info(entry_path) is not None:
                raise SymlinkNotAllowedError(f"refusing directory symlink: {entry_path}")
            entry_info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_info.st_mode):
                record(b"D", relative, 0, "")
                visit(entry_path, relative_parts, branch_ancestors)
            elif stat.S_ISREG(entry_info.st_mode):
                payload, content_digest, stable_size = _capture_file_details(
                    entry_path, chunk_size=_CHUNK_SIZE
                )
                if len(files) + 1 > max_files:
                    raise HashingError("directory capture exceeds the file-count limit")
                if total_bytes + stable_size > max_total_bytes:
                    raise HashingError("directory capture exceeds the byte-size limit")
                record(b"F", relative, stable_size, content_digest)
                files.append((relative, payload))
                total_bytes += stable_size
            else:
                raise HashingError(f"unsupported filesystem entry: {entry_path}")
        if _stable_stat(directory) != directory_before:
            raise FileChangedDuringHashError(
                f"directory changed while being captured: {directory}"
            )

    visit(candidate, (), set())
    return DirectoryCapture(
        receipt=PathDigest(
            algorithm="sha256-directory-merkle-v1",
            kind="directory",
            digest=digest.hexdigest(),
            file_count=len(files),
            total_bytes=total_bytes,
        ),
        files=tuple(files),
    )


def hash_path(
    path: PathLike,
    *,
    symlink_policy: SymlinkPolicy = "reject",
    allowed_root: PathLike | None = None,
) -> PathDigest:
    candidate = Path(path)
    link_info = _link_info(candidate)
    if link_info is not None:
        if symlink_policy == "reject":
            raise SymlinkNotAllowedError(f"refusing root symlink: {candidate}")
        if symlink_policy == "link":
            candidate, link_info = _canonical_link_entry(
                candidate, allowed_root=allowed_root, expected=link_info
            )
            target = _read_link_stably(candidate, link_info)
            return PathDigest(
                algorithm="sha256-symlink-v1",
                kind="symlink",
                digest=sha256_bytes(target),
                file_count=0,
                total_bytes=len(target),
            )
        candidate, _ = _canonical_link_entry(
            candidate, allowed_root=allowed_root, expected=link_info
        )
        resolved = candidate.resolve(strict=True)
        if allowed_root is not None:
            resolve_within_root(resolved, allowed_root, must_exist=True, allow_root=True)
        candidate = resolved

    if candidate.is_file():
        if allowed_root is not None:
            candidate = resolve_within_root(
                candidate, allowed_root, must_exist=True, allow_root=True
            )
        digest, size = _sha256_file_details(candidate, chunk_size=_CHUNK_SIZE)
        return PathDigest(
            algorithm="sha256",
            kind="file",
            digest=digest,
            file_count=1,
            total_bytes=size,
        )
    if candidate.is_dir():
        confined_root = Path(allowed_root).resolve(strict=True) if allowed_root is not None else None
        if confined_root is not None:
            candidate = resolve_within_root(
                candidate, confined_root, must_exist=True, allow_root=True
            )
        return _directory_digest(
            candidate,
            symlink_policy=symlink_policy,
            allowed_root=confined_root,
        )
    raise FileNotFoundError(f"path does not exist or is unsupported: {candidate}")
