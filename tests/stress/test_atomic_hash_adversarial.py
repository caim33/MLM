from __future__ import annotations

import json
import multiprocessing
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import motion_eval.core.atomic as atomic_module
import motion_eval.core.hashing as hashing_module
from motion_eval.core import (
    FileChangedDuringHashError,
    HashingError,
    PathOutsideRootError,
    UnsafePathError,
    atomic_write_json,
    hash_path,
    sha256_file,
)


def _exclusive_writer(
    path: str, writer: int, start: multiprocessing.synchronize.Event, queue: object
) -> None:
    start.wait(10)
    try:
        atomic_write_json(
            path,
            {"writer": writer, "payload": f"writer-{writer}" * 256},
            overwrite=False,
        )
    except FileExistsError:
        queue.put(("exists", writer))
    except BaseException as exc:  # pragma: no cover - returned to parent for assertion
        queue.put(("error", writer, type(exc).__name__, str(exc)))
    else:
        queue.put(("created", writer))


def _overwrite_writer(
    path: str, writer: int, start: multiprocessing.synchronize.Event, iterations: int
) -> None:
    start.wait(10)
    for sequence in range(iterations):
        token = f"{writer}:{sequence}:"
        atomic_write_json(
            path,
            {
                "writer": writer,
                "sequence": sequence,
                "token": token,
                "payload": token * 2048,
            },
        )


def _join_processes(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        process.join(timeout=30)
    try:
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.stress
def test_multiprocess_create_if_absent_has_exactly_one_winner(tmp_path: Path) -> None:
    destination = tmp_path / "immutable-receipt.json"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(
            target=_exclusive_writer,
            args=(str(destination), writer, start, queue),
        )
        for writer in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    _join_processes(processes)

    outcomes = [queue.get(timeout=5) for _ in processes]
    assert sum(item[0] == "created" for item in outcomes) == 1
    assert sum(item[0] == "exists" for item in outcomes) == len(processes) - 1
    assert not [item for item in outcomes if item[0] == "error"]
    value = json.loads(destination.read_text(encoding="utf-8"))
    assert value["payload"] == f"writer-{value['writer']}" * 256
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.stress
def test_readers_never_observe_partial_json_during_process_race(tmp_path: Path) -> None:
    destination = tmp_path / "hot-state.json"
    atomic_write_json(
        destination,
        {"writer": -1, "sequence": -1, "token": "seed:", "payload": "seed:" * 2048},
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=_overwrite_writer,
            args=(str(destination), writer, start, 24),
        )
        for writer in range(4)
    ]
    for process in processes:
        process.start()
    start.set()

    observations = 0
    deadline = time.monotonic() + 30
    try:
        while any(process.is_alive() for process in processes):
            assert time.monotonic() < deadline, "writer processes exceeded stress-test deadline"
            try:
                text = destination.read_text(encoding="utf-8")
            except PermissionError:
                # Windows can deny a read for the tiny interval in which an
                # independent process publishes with ReplaceFile semantics.
                # That is a fail-closed availability event, not partial data.
                continue
            value = json.loads(text)
            token = value["token"]
            assert value["payload"] == token * 2048
            observations += 1
    finally:
        _join_processes(processes)
    assert observations > 0
    value = json.loads(destination.read_text(encoding="utf-8"))
    assert value["payload"] == value["token"] * 2048
    assert not list(tmp_path.glob("*.tmp"))


def test_file_mutation_during_open_descriptor_hash_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"A" * (2 * 1024 * 1024))
    real_read = hashing_module.os.read
    changed = False

    def mutate_after_first_block(descriptor: int, count: int) -> bytes:
        nonlocal changed
        block = real_read(descriptor, count)
        if block and not changed:
            changed = True
            with artifact.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"B" * 4096)
                handle.flush()
                os.fsync(handle.fileno())
        return block

    monkeypatch.setattr(hashing_module.os, "read", mutate_after_first_block)

    with pytest.raises(FileChangedDuringHashError, match="changed while hashing"):
        sha256_file(artifact)
    assert changed


def test_simulated_windows_reparse_component_is_rejected_without_following_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    slot = root / "slot"
    slot.mkdir(parents=True)
    real_lstat = Path.lstat
    ordinary = real_lstat(slot)
    simulated_reparse = SimpleNamespace(
        st_mode=ordinary.st_mode,
        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
    )

    def lstat_with_reparse(path: Path) -> object:
        if Path(path) == slot:
            return simulated_reparse
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)

    with pytest.raises(UnsafePathError, match="reparse"):
        atomic_write_json("slot/result.json", {"unsafe": True}, root=root)
    assert not (slot / "result.json").exists()


@pytest.mark.stress
def test_symlink_parent_swap_after_initial_validation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mandatory remote gate: this must execute (not skip) on Linux or on a
    # Windows worker granted directory-symlink privileges.
    root = tmp_path / "root"
    slot = root / "slot"
    outside = tmp_path / "outside"
    slot.mkdir(parents=True)
    outside.mkdir()
    parked = root / "slot-before-swap"
    real_resolve_destination = atomic_module._resolve_destination
    swapped = False

    def resolve_then_swap(path: object, declared_root: object):
        nonlocal swapped
        resolved = real_resolve_destination(path, declared_root)
        if not swapped:
            slot.rename(parked)
            try:
                os.symlink(outside, slot, target_is_directory=True)
            except (OSError, NotImplementedError):
                parked.rename(slot)
                pytest.skip("creating directory symlinks is not permitted")
            swapped = True
        return resolved

    monkeypatch.setattr(atomic_module, "_resolve_destination", resolve_then_swap)

    with pytest.raises((PathOutsideRootError, UnsafePathError)):
        atomic_write_json("slot/result.json", {"escaped": True}, root=root)
    assert swapped
    assert not (outside / "result.json").exists()


@pytest.mark.stress
def test_directory_entry_swap_to_symlink_is_not_hashed_as_regular_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mandatory remote gate: this must execute (not skip) on Linux or on a
    # Windows worker granted file-symlink privileges.
    tree = tmp_path / "tree"
    tree.mkdir()
    entry = tree / "entry.bin"
    entry.write_bytes(b"inside")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-secret")
    real_link_info = hashing_module._link_info
    swapped = False

    def link_info_with_swap(path: Path):
        nonlocal swapped
        if Path(path) == entry and not swapped:
            entry.unlink()
            try:
                os.symlink(outside, entry)
            except (OSError, NotImplementedError):
                entry.write_bytes(b"inside")
                pytest.skip("creating file symlinks is not permitted")
            swapped = True
            # Model a swap immediately after the caller's lstat returned a
            # regular file. The next filesystem operation must still reject.
            return None
        return real_link_info(path)

    monkeypatch.setattr(hashing_module, "_link_info", link_info_with_swap)

    with pytest.raises(HashingError):
        hash_path(tree)
    assert swapped
