"""Trusted, deterministic source allowlists shared by capture and verification."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Sequence

from .hashing import hash_path


SOURCE_MANIFEST_SCHEMA = "motionllm-source-allowlist-v2"


def is_link_or_reparse(path: Path) -> bool:
    """Treat POSIX links and Windows reparse points/junctions as links."""

    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def bytecode_source(path: Path) -> Path | None:
    suffix = path.suffix.lower()
    if suffix not in {".pyc", ".pyo"}:
        return None
    if path.parent.name == "__pycache__":
        return path.parent.parent / (path.name.split(".", 1)[0] + ".py")
    return path.with_suffix(".py")


def _reject_lexical_link_chain(
    path: Path, *, allowed_root: Path, label: str
) -> Path:
    lexical_root = Path(os.path.abspath(allowed_root))
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the formal source root: {path}") from exc
    current = lexical_root
    if is_link_or_reparse(current):
        raise ValueError(f"{label} source root is linked/reparsed: {current}")
    for part in relative.parts:
        current = current / part
        if is_link_or_reparse(current):
            raise ValueError(f"{label} rejects linked/reparsed path: {current}")
    resolved = lexical.resolve(strict=True)
    if os.path.normcase(str(resolved)) != os.path.normcase(str(lexical)):
        raise ValueError(f"{label} lexical and resolved paths differ: {lexical}")
    return lexical


def stable_source_files(
    project_root: Path,
    *,
    roots: Sequence[Path],
    fixed_files: Sequence[Path] = (),
    record_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Rebuild a complete, source-backed, link-free source file inventory."""

    project_root = Path(os.path.abspath(project_root))
    record_base = Path(os.path.abspath(record_root or project_root))
    try:
        record_base.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("formal source record root escapes project root") from exc
    selected: dict[str, Path] = {}

    def select(child: Path) -> None:
        try:
            relative = child.relative_to(record_base).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"formal source file escapes its role root: {child}"
            ) from exc
        selected[relative] = child

    for raw_root in roots:
        source_root = _reject_lexical_link_chain(
            raw_root,
            allowed_root=project_root,
            label="formal source allowlist root",
        )
        if not source_root.is_dir():
            raise ValueError(
                f"formal source allowlist root is not a directory: {source_root}"
            )
        for directory, directory_names, file_names in os.walk(
            source_root, followlinks=False
        ):
            base = Path(directory)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                child = base / name
                if name == "__pycache__":
                    if is_link_or_reparse(child) or not child.is_dir():
                        raise ValueError(
                            "formal source rejects linked/non-directory bytecode "
                            f"cache: {child}"
                        )
                    for bytecode in sorted(child.iterdir()):
                        if (
                            is_link_or_reparse(bytecode)
                            or not bytecode.is_file()
                            or bytecode.suffix.lower() not in {".pyc", ".pyo"}
                        ):
                            raise ValueError(
                                "formal source rejects invalid bytecode cache "
                                f"entry: {bytecode}"
                            )
                        source = bytecode_source(bytecode)
                        if (
                            source is None
                            or not source.is_file()
                            or is_link_or_reparse(source)
                        ):
                            raise ValueError(
                                f"formal source rejects sourceless bytecode: {bytecode}"
                            )
                        select(bytecode)
                    continue
                if name in {".git", ".cache"}:
                    raise ValueError(
                        "formal source rejects excluded control directory inside "
                        f"an import tree: {child}"
                    )
                if is_link_or_reparse(child):
                    raise ValueError(
                        f"formal source allowlist rejects linked directory: {child}"
                    )
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                child = base / name
                if child.suffix.lower() in {".pyc", ".pyo"}:
                    if is_link_or_reparse(child) or not child.is_file():
                        raise ValueError(
                            "formal source rejects linked/non-regular bytecode: "
                            f"{child}"
                        )
                    source = bytecode_source(child)
                    if (
                        source is None
                        or not source.is_file()
                        or is_link_or_reparse(source)
                    ):
                        raise ValueError(
                            f"formal source rejects sourceless bytecode: {child}"
                        )
                    select(child)
                    continue
                if is_link_or_reparse(child) or not child.is_file():
                    raise ValueError(
                        f"formal source allowlist rejects non-regular file: {child}"
                    )
                select(child)

    for raw_file in fixed_files:
        child = _reject_lexical_link_chain(
            raw_file,
            allowed_root=project_root,
            label="formal source allowlist file",
        )
        if is_link_or_reparse(child) or not child.is_file():
            raise ValueError(f"formal source allowlist file is invalid: {child}")
        select(child)
    if not selected:
        raise ValueError("formal source allowlist is empty")

    records: list[dict[str, Any]] = []
    for relative, child in sorted(selected.items()):
        digest = hash_path(
            child, symlink_policy="reject", allowed_root=project_root
        )
        if digest.kind != "file" or digest.file_count != 1 or digest.total_bytes <= 0:
            raise ValueError(f"formal source file is empty or invalid: {child}")
        records.append(
            {
                "relative_path": relative,
                "sha256": digest.digest,
                "size": digest.total_bytes,
            }
        )
    return tuple(records)


def formal_source_role_files(
    project_root: Path, *, runner_root: Path, role: str
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    """Return the exact fixed inventory for a formal code or runner role."""

    project_root = Path(os.path.abspath(project_root))
    runner_root = Path(os.path.abspath(runner_root))
    if runner_root != project_root / "qwenvl":
        raise ValueError("formal runner role root must be the checkout qwenvl tree")
    if role == "code":
        return project_root, stable_source_files(
            project_root,
            roots=(
                project_root / "src",
                project_root / "models",
            ),
            fixed_files=(
                project_root / "pyproject.toml",
                project_root / "requirements" / "sft.txt",
                project_root / "scripts" / "full_sft.sh",
                project_root / "scripts" / "lora_sft.sh",
                project_root / "scripts" / "zero2.json",
            ),
        )
    if role == "runner_code":
        return runner_root, stable_source_files(
            project_root,
            roots=(runner_root,),
            record_root=runner_root,
        )
    raise ValueError(f"unsupported formal source role: {role}")


__all__ = [
    "SOURCE_MANIFEST_SCHEMA",
    "bytecode_source",
    "formal_source_role_files",
    "is_link_or_reparse",
    "stable_source_files",
]
