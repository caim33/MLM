import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import motion_eval.core.atomic as atomic_module
from motion_eval.core import (
    PathOutsideRootError,
    UnsafePathError,
    atomic_write_json,
    atomic_write_jsonl,
    resolve_within_root,
)


class _SimulatedWindowsSharingError(PermissionError):
    def __init__(self, winerror):
        super().__init__(f"simulated Windows sharing error {winerror}")
        self.winerror = winerror


def _multiprocess_atomic_writer(path, writer, iterations):
    """Top-level worker so Windows spawn exercises real independent processes."""

    for sequence in range(iterations):
        token = f"{writer}:{sequence}"
        atomic_write_json(
            path,
            {"writer": writer, "sequence": sequence, "payload": token * 100},
        )


def test_atomic_json_writes_complete_utf8_document(tmp_path):
    destination = atomic_write_json(
        "results/summary.json",
        {"模型": "Motion-R1", "count": 500},
        root=tmp_path,
    )

    assert destination == (tmp_path / "results" / "summary.json").resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "模型": "Motion-R1",
        "count": 500,
    }
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_json_replaces_old_value_but_never_leaves_partial_json(tmp_path):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"generation": 1})
    atomic_write_json(destination, {"generation": 2, "payload": "x" * 1000})
    assert json.loads(destination.read_text(encoding="utf-8"))["generation"] == 2


def test_serialization_failure_does_not_damage_existing_file(tmp_path):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"stable": True})

    with pytest.raises(ValueError, match="finite JSON"):
        atomic_write_json(destination, {"value": float("nan")})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"stable": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_unicode_does_not_damage_existing_file(tmp_path):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"stable": True})

    with pytest.raises(ValueError, match="finite JSON"):
        atomic_write_json(destination, {"value": "\ud800"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"stable": True}


def test_jsonl_generator_failure_cleans_temp_and_preserves_destination(tmp_path):
    destination = tmp_path / "predictions.jsonl"
    atomic_write_jsonl(destination, [{"sample_id": "old"}])

    def bad_rows():
        yield {"sample_id": "new"}
        yield {"score": float("inf")}

    with pytest.raises(ValueError, match="row 1"):
        atomic_write_jsonl(destination, bad_rows())

    assert destination.read_text(encoding="utf-8") == '{"sample_id":"old"}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_no_overwrite_is_fail_closed(tmp_path):
    destination = tmp_path / "manifest.json"
    atomic_write_json(destination, {"version": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(destination, {"version": 2}, overwrite=False)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}


@pytest.mark.stress
def test_128_writer_thread_stress_leaves_one_whole_document(tmp_path):
    destination = tmp_path / "state.json"

    def write(index):
        atomic_write_json(destination, {"writer": index, "payload": str(index) * 200})

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(write, range(128)))

    value = json.loads(destination.read_text(encoding="utf-8"))
    assert value["payload"] == str(value["writer"]) * 200
    assert not list(tmp_path.glob("*.tmp"))
    assert not atomic_module._TARGET_LOCKS


@pytest.mark.stress
def test_real_spawned_processes_concurrently_publish_whole_json(tmp_path):
    destination = tmp_path / "multiprocess-state.json"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_multiprocess_atomic_writer,
            args=(destination, writer, 12),
        )
        for writer in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    try:
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    value = json.loads(destination.read_text(encoding="utf-8"))
    token = f'{value["writer"]}:{value["sequence"]}'
    assert value["payload"] == token * 100
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("winerror", [5, 32])
def test_windows_sharing_errors_retry_without_deleting_destination(
    tmp_path, monkeypatch, winerror
):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"generation": "old"})
    real_replace = atomic_module.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        assert Path(target).read_text(encoding="utf-8").find("old") >= 0
        if attempts <= 10:
            raise _SimulatedWindowsSharingError(winerror)
        real_replace(source, target)

    monkeypatch.setattr(atomic_module.os, "replace", flaky_replace)
    atomic_write_json(destination, {"generation": "new"})

    assert attempts == 11
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": "new"
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_retry_has_total_deadline_and_cleans_temp(tmp_path, monkeypatch):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"generation": "old"})
    clock = [0.0]
    attempts = 0

    def always_busy(source, target):
        nonlocal attempts
        attempts += 1
        raise _SimulatedWindowsSharingError(32)

    monkeypatch.setattr(atomic_module.os, "replace", always_busy)
    monkeypatch.setattr(atomic_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        atomic_module.time,
        "sleep",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(atomic_module.random, "uniform", lambda low, high: 1.0)
    monkeypatch.setattr(atomic_module, "_REPLACE_RETRY_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(_SimulatedWindowsSharingError):
        atomic_write_json(destination, {"generation": "new"})

    assert 1 < attempts < atomic_module._REPLACE_MAX_ATTEMPTS
    assert clock[0] <= 0.05
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_parent_identity_change_between_retries_fails_closed_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"generation": "old"})
    real_identity = atomic_module._path_identity
    identity_calls = 0
    replace_calls = 0

    def changing_identity(path):
        nonlocal identity_calls
        identity_calls += 1
        identity = real_identity(path)
        if identity_calls >= 3:
            return identity[0], identity[1] + 1
        return identity

    def busy_once(source, target):
        nonlocal replace_calls
        replace_calls += 1
        raise _SimulatedWindowsSharingError(5)

    monkeypatch.setattr(atomic_module, "_path_identity", changing_identity)
    monkeypatch.setattr(atomic_module.os, "replace", busy_once)

    with pytest.raises(UnsafePathError, match="parent changed"):
        atomic_write_json(destination, {"generation": "new"})

    assert replace_calls == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_root_escape_between_retries_fails_closed_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "state.json"
    atomic_write_json(destination, {"generation": "old"}, root=tmp_path)
    real_resolve = atomic_module.resolve_within_root
    resolve_calls = 0
    replace_calls = 0

    def changing_root(path, root, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        if resolve_calls >= 4:
            raise PathOutsideRootError("simulated root escape")
        return real_resolve(path, root, **kwargs)

    def busy_once(source, target):
        nonlocal replace_calls
        replace_calls += 1
        raise _SimulatedWindowsSharingError(32)

    monkeypatch.setattr(atomic_module, "resolve_within_root", changing_root)
    monkeypatch.setattr(atomic_module.os, "replace", busy_once)

    with pytest.raises(PathOutsideRootError, match="simulated root escape"):
        atomic_write_json(destination, {"generation": "new"}, root=tmp_path)

    assert replace_calls == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_root_confined_resolution_rejects_parent_traversal(tmp_path):
    with pytest.raises(PathOutsideRootError):
        resolve_within_root("../escape.json", tmp_path)
    with pytest.raises(PathOutsideRootError):
        atomic_write_json("../escape.json", {"bad": True}, root=tmp_path)


def test_root_itself_is_not_a_valid_output_file(tmp_path):
    with pytest.raises(UnsafePathError, match="below root"):
        resolve_within_root(".", tmp_path)


def test_absolute_path_inside_root_is_allowed(tmp_path):
    target = tmp_path / "nested" / "value.json"
    assert resolve_within_root(target, tmp_path) == target.resolve()


def test_absolute_path_outside_root_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside.json"
    with pytest.raises(PathOutsideRootError):
        resolve_within_root(outside, tmp_path)


@pytest.mark.parametrize("reference", ["https://example.test/out.json", "bad\x00name"])
def test_non_filesystem_and_null_paths_are_rejected(tmp_path, reference):
    with pytest.raises(UnsafePathError):
        resolve_within_root(reference, tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="NTFS path syntax is Windows-specific")
@pytest.mark.parametrize("reference", ["result.json:secret", "NUL", r"\\.\NUL"])
def test_windows_device_and_alternate_stream_paths_are_rejected(tmp_path, reference):
    with pytest.raises(UnsafePathError):
        resolve_within_root(reference, tmp_path)
