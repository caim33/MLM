from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from motion_eval.core import sha256_json
import motion_eval.runtime.gpu as gpu_module
from motion_eval.runtime import (
    GPUDevice,
    GPUInventory,
    GPUOwnershipError,
    GPUProcess,
    GPUQueryError,
    GPULeaseStore,
    KeepaliveController,
    KeepaliveStore,
    command_fingerprint,
    system_process_fingerprint,
    worker_code_sha256,
)
from motion_eval.runtime.keepalive import WORKER_MODULE, build_worker_command
from motion_eval.runtime.keepalive_worker import _torch_uuid
from motion_eval.runtime.keepalive_worker import build_parser as worker_parser
from motion_eval.runtime.keepalive_worker import run_worker


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.base = datetime(2026, 8, 20, tzinfo=timezone.utc)
        self.hook = None

    def monotonic(self) -> float:
        return self.value

    def now(self) -> datetime:
        return self.base + timedelta(seconds=self.value)

    def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0.001)
        if self.hook is not None:
            self.hook()


class FakeProbe:
    def __init__(self) -> None:
        self.device = GPUDevice(2, "GPU-unit-test", "fake", 24_000, 0, 0)
        self.processes: tuple[GPUProcess, ...] = ()
        self.fail = False
        self.queries = 0

    def preview(self):
        return {"device_query": ["nvidia-smi", "devices"], "process_query": ["nvidia-smi", "processes"]}

    def query(self) -> GPUInventory:
        self.queries += 1
        if self.fail:
            raise GPUQueryError("unknown")
        return GPUInventory((self.device,), self.processes)


class FakeRuntime:
    def __init__(self) -> None:
        self.pid = 4321
        self.running = False
        self.argv: tuple[str, ...] | None = None
        self.command_hash: str | None = None
        self.launches = 0
        self.terminated: list[int] = []
        self.killed: list[int] = []

    def launch(self, argv, *, cwd, env) -> int:
        assert isinstance(argv, (list, tuple))
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-unit-test"
        assert Path(cwd).is_dir()
        self.argv = tuple(argv)
        self.command_hash = command_fingerprint(self.argv)
        self.running = True
        self.launches += 1
        return self.pid

    def alive(self, pid: int) -> bool:
        return pid == self.pid and self.running

    def fingerprint(self, pid: int) -> str | None:
        return self.command_hash if self.alive(pid) else None

    def terminate(self, pid: int) -> None:
        assert pid == self.pid
        self.terminated.append(pid)
        self.running = False

    def kill(self, pid: int) -> None:
        assert pid == self.pid
        self.killed.append(pid)
        self.running = False


def make_controller(tmp_path):
    clock = FakeClock()
    probe = FakeProbe()
    runtime = FakeRuntime()
    controller = KeepaliveController(
        tmp_path,
        project_owner="motionllm-test",
        probe=probe,
        process_runtime=runtime,
        now=clock.now,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        python_executable=sys.executable,
    )
    return controller, clock, probe, runtime


def arrange_fake_worker_heartbeat(controller, clock, probe, runtime):
    sent = False

    def hook():
        nonlocal sent
        stop_files = list(controller.store.root.glob("*.stop"))
        if stop_files and runtime.running:
            runtime.running = False
            return
        if sent or not runtime.running:
            return
        records = list(controller.store.root.glob("*.json"))
        if records and controller.store.ready_exists(probe.device.uuid):
            record = controller.store.load_record(probe.device.uuid, require_active=False)
            probe.processes = (GPUProcess(probe.device.uuid, runtime.pid, 8),)
            controller.store.heartbeat(
                probe.device.uuid,
                pid=runtime.pid,
                reservation_id=str(record["reservation_id"]),
                command_fingerprint=str(record["command_fingerprint"]),
                worker_module=str(record["worker_module"]),
                worker_code_sha256=str(record["worker_code_sha256"]),
            )
            sent = True

    clock.hook = hook


def test_system_process_fingerprint_matches_exact_launched_argv():
    if os.name == "posix" and not Path("/proc").is_dir():
        pytest.skip("this POSIX platform does not expose /proc process argv")
    argv = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
        "argument with spaces",
        'argument-with-"quote',
        "trailing-backslash\\",
    ]
    child = subprocess.Popen(argv, shell=False)
    try:
        expected = command_fingerprint(argv)
        deadline = time.monotonic() + 3.0
        actual = None
        while time.monotonic() < deadline:
            actual = system_process_fingerprint(child.pid)
            if actual is not None:
                break
            time.sleep(0.01)
        assert actual == expected
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_start_handshake_records_identity_and_waits_for_real_gpu_process(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)

    record = controller.start(
        probe.device.uuid,
        heartbeat_interval_seconds=1,
        ready_timeout_seconds=1,
        wait_timeout_seconds=1,
    )

    assert record["state"] == "active"
    assert record["pid"] == runtime.pid
    assert record["owner"] == "motionllm-test"
    assert record["gpu_uuid"] == probe.device.uuid
    assert record["gpu_index"] == probe.device.index
    assert record["command_fingerprint"] == runtime.command_hash
    assert record["worker_module"] == WORKER_MODULE
    assert record["worker_code_sha256"] == worker_code_sha256()
    assert record["record_sha256"]
    assert record["heartbeat_at"] >= record["started_at"]
    assert runtime.argv is not None and runtime.argv[:3] == (sys.executable, "-m", WORKER_MODULE)
    assert runtime.launches == 1


def test_start_requires_idle_gpu_before_launch(tmp_path):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    probe.processes = (GPUProcess(probe.device.uuid, 999, 100),)
    with pytest.raises(GPUQueryError, match="not proven idle"):
        controller.start(probe.device.uuid, wait_timeout_seconds=0.2)
    assert runtime.launches == 0
    assert list(tmp_path.iterdir()) == []


def test_concurrent_reservation_is_atomic_and_prepare_worker_blocks_it(tmp_path):
    probe = FakeProbe()
    first = KeepaliveStore(tmp_path, project_owner="motionllm-test", probe=probe)
    second = KeepaliveStore(tmp_path, project_owner="motionllm-test", probe=probe)
    first.reserve(probe.device, reservation_id="first")
    with pytest.raises(GPUOwnershipError, match="lifecycle|reserved"):
        second.reserve(probe.device, reservation_id="second")
    with pytest.raises(GPUOwnershipError, match="must be stopped"):
        first.prepare_worker(probe.device.uuid)


def test_worker_role_held_after_prepare_blocks_concurrent_keepalive_start(tmp_path):
    """Deterministic interleaving at the old prepare->launch race window."""

    controller, _clock, probe, runtime = make_controller(tmp_path)
    roles = GPULeaseStore(
        tmp_path,
        project_owner="motionllm-test",
        probe=probe,
        pid_alive=lambda _pid: True,
    )
    lease = roles.acquire_role(
        probe.device,
        role="eval",
        pid=123,
        purpose="paused after worker prepare",
    )
    # The worker's lifecycle check has completed, but its role mutex remains
    # held.  A keepalive cannot reserve or launch into that gap.
    KeepaliveStore(tmp_path, project_owner="motionllm-test", probe=probe).prepare_worker(
        probe.device.uuid
    )
    with pytest.raises(GPUOwnershipError, match="role mutex"):
        controller.start(probe.device.uuid, wait_timeout_seconds=0.2)
    assert runtime.launches == 0
    assert not list(tmp_path.glob("*.reservation"))
    roles.release(
        probe.device.uuid,
        lease_id=str(lease["lease_id"]),
        pid=123,
        role="eval",
    )


def test_active_keepalive_role_blocks_worker_before_model_launch(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    controller.start(probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1)

    model_launches = 0
    roles = GPULeaseStore(
        tmp_path,
        project_owner="motionllm-test",
        probe=probe,
        pid_alive=lambda _pid: True,
    )
    with pytest.raises(GPUOwnershipError, match="role mutex"):
        roles.acquire_role(
            probe.device,
            role="finetune",
            pid=999,
            purpose="model worker",
        )
        model_launches += 1
    assert model_launches == 0
    controller.stop(probe.device.uuid, wait_timeout_seconds=1)


def test_role_mutex_is_per_uuid_and_stale_release_cannot_delete_new_owner(tmp_path):
    first = GPUDevice(0, "GPU-A", "fake-a", 1000, 0, 0)
    second = GPUDevice(1, "GPU-B", "fake-b", 1000, 0, 0)
    roles = GPULeaseStore(
        tmp_path,
        project_owner="motionllm-test",
        probe=FakeProbe(),
        pid_alive=lambda _pid: True,
    )
    lease_a = roles.acquire_role(first, role="eval", pid=1, purpose="gpu-a")
    lease_b = roles.acquire_role(second, role="keepalive", pid=2, purpose="gpu-b")
    assert roles.load(first.uuid)["lease_id"] == lease_a["lease_id"]
    assert roles.load(second.uuid)["lease_id"] == lease_b["lease_id"]

    roles.release(first.uuid, lease_id=str(lease_a["lease_id"]), pid=1, role="eval")
    replacement = roles.acquire_role(first, role="finetune", pid=3, purpose="replacement")
    with pytest.raises(GPUOwnershipError, match="refusing"):
        roles.release(first.uuid, lease_id=str(lease_a["lease_id"]), pid=1, role="eval")
    assert roles.load(first.uuid)["lease_id"] == replacement["lease_id"]

    roles.release(first.uuid, lease_id=str(replacement["lease_id"]), pid=3, role="finetune")
    roles.release(second.uuid, lease_id=str(lease_b["lease_id"]), pid=2, role="keepalive")


def test_failed_first_heartbeat_terminates_only_child_and_cleans_evidence(tmp_path):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    with pytest.raises(GPUOwnershipError, match="live heartbeat"):
        controller.start(
            probe.device.uuid,
            heartbeat_interval_seconds=1,
            ready_timeout_seconds=1,
            wait_timeout_seconds=0.25,
        )
    assert runtime.terminated == [runtime.pid]
    assert runtime.killed == []
    assert list(tmp_path.iterdir()) == []


def test_start_rolls_back_role_and_reservation_on_baseexception(tmp_path):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    original_query = probe.query

    def interrupt_second_query():
        if probe.queries == 1:
            raise KeyboardInterrupt("deterministic start interruption")
        return original_query()

    probe.query = interrupt_second_query
    with pytest.raises(KeyboardInterrupt, match="deterministic start interruption"):
        controller.start(probe.device.uuid, wait_timeout_seconds=0.2)

    assert runtime.launches == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_suffix", [".reservation", ".owner.json"])
def test_start_rolls_back_occupancy_published_just_before_write_error(
    tmp_path, monkeypatch, failed_suffix
):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    original_write = gpu_module.atomic_write_json
    injected = False

    def publish_then_fail(path, value, **kwargs):
        nonlocal injected
        result = original_write(path, value, **kwargs)
        candidate = Path(path)
        if not injected and (
            candidate.suffix == failed_suffix
            or candidate.name.endswith(failed_suffix)
        ):
            injected = True
            raise OSError("deterministic post-publication failure")
        return result

    monkeypatch.setattr(gpu_module, "atomic_write_json", publish_then_fail)
    with pytest.raises(OSError, match="post-publication failure"):
        controller.start(probe.device.uuid, wait_timeout_seconds=0.2)

    assert injected
    assert runtime.launches == 0
    assert list(tmp_path.iterdir()) == []


def test_failed_start_reaps_other_owned_files_when_one_unlink_fails(
    tmp_path, monkeypatch
):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    original_unlink = Path.unlink

    def fail_ready_unlink(path, *args, **kwargs):
        if path.suffix == ".ready":
            raise PermissionError("deterministic ready-file lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_ready_unlink)
    with pytest.raises(GPUOwnershipError, match="cleanup could not be proven"):
        controller.start(
            probe.device.uuid,
            heartbeat_interval_seconds=1,
            ready_timeout_seconds=1,
            wait_timeout_seconds=0.25,
        )

    assert runtime.terminated == [runtime.pid]
    assert list(tmp_path.glob("*.ready"))
    assert not list(tmp_path.glob("*.reservation"))
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.role"))


def test_failed_start_reaps_owned_state_but_preserves_replaced_reservation(
    tmp_path, monkeypatch
):
    controller, clock, probe, runtime = make_controller(tmp_path)
    original_terminate = controller._terminate_failed_start
    foreign_reservation: dict[str, object] = {}

    def terminate_then_replace_reservation(pid, fingerprint, timeout):
        nonlocal foreign_reservation
        original_terminate(pid, fingerprint, timeout)
        body = {
            "schema_version": "2.0",
            "gpu_uuid": probe.device.uuid,
            "gpu_index": probe.device.index,
            "owner": "motionllm-test",
            "reservation_id": "foreign-replacement",
            "reserved_at": clock.now().isoformat(),
        }
        foreign_reservation = controller.store._hashed(body, "reservation_sha256")
        controller.store._reservation_path(probe.device.uuid).write_text(
            json.dumps(foreign_reservation), encoding="utf-8"
        )

    monkeypatch.setattr(
        controller, "_terminate_failed_start", terminate_then_replace_reservation
    )
    with pytest.raises(GPUOwnershipError, match="foreign or invalid: reservation"):
        controller.start(
            probe.device.uuid,
            heartbeat_interval_seconds=1,
            ready_timeout_seconds=1,
            wait_timeout_seconds=0.25,
        )

    reservation_path = controller.store._reservation_path(probe.device.uuid)
    assert json.loads(reservation_path.read_text(encoding="utf-8")) == foreign_reservation
    assert not list(tmp_path.glob("*.ready"))
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.role"))


def test_failed_start_keeps_mutex_when_live_child_identity_becomes_unknown(tmp_path):
    controller, _clock, probe, runtime = make_controller(tmp_path)
    fingerprint_calls = 0
    original_fingerprint = runtime.fingerprint

    def fingerprint_once(pid):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        if fingerprint_calls == 1:
            return original_fingerprint(pid)
        return None

    runtime.fingerprint = fingerprint_once
    with pytest.raises(GPUOwnershipError, match="child is not proven stopped"):
        controller.start(probe.device.uuid, wait_timeout_seconds=0.25)

    assert runtime.running
    assert runtime.terminated == [] and runtime.killed == []
    assert list(tmp_path.glob("*.reservation"))
    assert list(tmp_path.glob("*.role"))


def test_status_fails_closed_on_tamper_pid_reuse_stale_or_query_failure(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    controller.start(probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1)

    runtime.command_hash = "0" * 64
    with pytest.raises(GPUOwnershipError, match="reused|fingerprint"):
        controller.status()
    runtime.command_hash = command_fingerprint(runtime.argv or ())

    clock.value += 181
    with pytest.raises(GPUOwnershipError, match="stale"):
        controller.status(stale_after_seconds=180)
    clock.value -= 181

    probe.fail = True
    with pytest.raises(GPUQueryError):
        controller.status()
    probe.fail = False

    path = next(tmp_path.glob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["pid"] = 9876
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(GPUOwnershipError, match="tampered"):
        controller.status()


def test_stop_uses_bound_proof_and_graceful_request_for_only_recorded_pid(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    controller.start(probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1)

    stopped = controller.stop(probe.device.uuid, wait_timeout_seconds=1)
    assert stopped == {
        "stopped": True,
        "gpu_uuid": probe.device.uuid,
        "gpu_index": probe.device.index,
        "pid": runtime.pid,
        "owner": "motionllm-test",
    }
    assert runtime.terminated == []
    assert runtime.killed == []
    assert list(tmp_path.iterdir()) == []


def test_stop_accepts_only_a_final_bound_heartbeat_after_stop_request(tmp_path):
    """Reproduce the request -> heartbeat -> worker-exit serialization race."""

    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    initial = controller.start(
        probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1
    )
    final_record = None
    stop_poll_count = 0

    def heartbeat_then_exit():
        nonlocal final_record, stop_poll_count
        if not list(controller.store.root.glob("*.stop")) or not runtime.running:
            return
        stop_poll_count += 1
        if stop_poll_count == 1:
            record = controller.store.load_record(probe.device.uuid)
            final_record = controller.store.heartbeat(
                probe.device.uuid,
                pid=runtime.pid,
                reservation_id=str(record["reservation_id"]),
                command_fingerprint=str(record["command_fingerprint"]),
                worker_module=str(record["worker_module"]),
                worker_code_sha256=str(record["worker_code_sha256"]),
            )
            return
        runtime.running = False

    clock.hook = heartbeat_then_exit
    stopped = controller.stop(probe.device.uuid, wait_timeout_seconds=1)

    assert final_record is not None
    assert final_record["record_sha256"] != initial["record_sha256"]
    assert stopped["pid"] == runtime.pid
    assert runtime.terminated == [] and runtime.killed == []
    assert list(tmp_path.iterdir()) == []


def test_stop_keeps_original_request_proof_when_final_heartbeat_precedes_terminate(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    initial = controller.start(
        probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1
    )
    final_record = None

    def final_heartbeat_but_ignore_graceful_stop():
        nonlocal final_record
        if (
            final_record is not None
            or not runtime.running
            or not list(controller.store.root.glob("*.stop"))
        ):
            return
        record = controller.store.load_record(probe.device.uuid)
        final_record = controller.store.heartbeat(
            probe.device.uuid,
            pid=runtime.pid,
            reservation_id=str(record["reservation_id"]),
            command_fingerprint=str(record["command_fingerprint"]),
            worker_module=str(record["worker_module"]),
            worker_code_sha256=str(record["worker_code_sha256"]),
        )

    clock.hook = final_heartbeat_but_ignore_graceful_stop
    stopped = controller.stop(probe.device.uuid, wait_timeout_seconds=1)

    assert final_record is not None
    assert final_record["record_sha256"] != initial["record_sha256"]
    assert stopped["pid"] == runtime.pid
    assert runtime.terminated == [runtime.pid]
    assert runtime.killed == []
    assert list(tmp_path.iterdir()) == []


def _prepare_dead_worker_with_stop_request(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    controller.start(probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1)
    proof = controller.store.prove_stop(probe.device.uuid)
    stop_request = controller.store.request_stop(proof)
    runtime.running = False
    return controller, probe, runtime, proof, str(stop_request["stop_sha256"])


def _rewrite_hash_bound_json(path: Path, value: dict, hash_field: str) -> None:
    body = {key: item for key, item in value.items() if key != hash_field}
    value[hash_field] = sha256_json(body)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_finalize_stop_fails_closed_if_code_binding_changes_with_valid_hashes(tmp_path):
    controller, probe, _runtime, proof, stop_sha256 = _prepare_dead_worker_with_stop_request(tmp_path)
    record_path = next(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["worker_code_sha256"] = "e" * 64
    binding = controller.store._binding(record)
    record["binding_sha256"] = sha256_json(binding)
    _rewrite_hash_bound_json(record_path, record, "record_sha256")

    with pytest.raises(GPUOwnershipError, match="identity changed"):
        controller.store.finalize_stop(proof, stop_sha256=stop_sha256)

    assert record_path.exists()
    assert next(tmp_path.glob("*.stop")).exists()
    controller.role_leases.assert_owned(
        probe.device.uuid, lease_id=proof.reservation_id, role="keepalive"
    )


def test_finalize_stop_fails_closed_if_nonheartbeat_field_is_added(tmp_path):
    controller, probe, _runtime, proof, stop_sha256 = _prepare_dead_worker_with_stop_request(tmp_path)
    record_path = next(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["security_mode"] = "disabled"
    _rewrite_hash_bound_json(record_path, record, "record_sha256")

    with pytest.raises(GPUOwnershipError, match="record shape"):
        controller.store.finalize_stop(proof, stop_sha256=stop_sha256)

    assert record_path.exists()
    controller.role_leases.assert_owned(
        probe.device.uuid, lease_id=proof.reservation_id, role="keepalive"
    )


def test_finalize_stop_fails_closed_if_heartbeat_moves_backwards(tmp_path):
    controller, probe, _runtime, proof, stop_sha256 = _prepare_dead_worker_with_stop_request(tmp_path)
    record_path = next(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["started_at"] < proof.heartbeat_at
    record["heartbeat_at"] = record["started_at"]
    _rewrite_hash_bound_json(record_path, record, "record_sha256")

    with pytest.raises(GPUOwnershipError, match="moved backwards"):
        controller.store.finalize_stop(proof, stop_sha256=stop_sha256)

    assert record_path.exists()
    controller.role_leases.assert_owned(
        probe.device.uuid, lease_id=proof.reservation_id, role="keepalive"
    )


def test_finalize_stop_fails_closed_if_final_heartbeat_is_from_future(tmp_path):
    controller, probe, _runtime, proof, stop_sha256 = _prepare_dead_worker_with_stop_request(tmp_path)
    record_path = next(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["heartbeat_at"] = datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat()
    _rewrite_hash_bound_json(record_path, record, "record_sha256")

    with pytest.raises(GPUOwnershipError, match="from the future"):
        controller.store.finalize_stop(proof, stop_sha256=stop_sha256)

    assert record_path.exists()
    controller.role_leases.assert_owned(
        probe.device.uuid, lease_id=proof.reservation_id, role="keepalive"
    )


def test_finalize_stop_fails_closed_if_stop_request_no_longer_matches_proof(tmp_path):
    controller, probe, _runtime, proof, stop_sha256 = _prepare_dead_worker_with_stop_request(tmp_path)
    stop_path = next(tmp_path.glob("*.stop"))
    request = json.loads(stop_path.read_text(encoding="utf-8"))
    request["requested_at"] = datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat()
    _rewrite_hash_bound_json(stop_path, request, "stop_sha256")

    with pytest.raises(GPUOwnershipError, match="stop request changed"):
        controller.store.finalize_stop(proof, stop_sha256=stop_sha256)

    assert next(tmp_path.glob("*.json")).exists()
    controller.role_leases.assert_owned(
        probe.device.uuid, lease_id=proof.reservation_id, role="keepalive"
    )


def test_stop_refuses_changed_process_fingerprint_without_signalling(tmp_path):
    controller, clock, probe, runtime = make_controller(tmp_path)
    arrange_fake_worker_heartbeat(controller, clock, probe, runtime)
    controller.start(probe.device.uuid, heartbeat_interval_seconds=1, wait_timeout_seconds=1)
    runtime.command_hash = "f" * 64
    with pytest.raises(GPUOwnershipError, match="reused|fingerprint"):
        controller.stop(probe.device.uuid, wait_timeout_seconds=0.2)
    assert runtime.terminated == [] and runtime.killed == []


class NoCuda:
    class cuda:
        @staticmethod
        def is_available():
            return False


class NoTorchUuid:
    class cuda:
        @staticmethod
        def get_device_properties(_logical_index):
            return object()


def test_worker_uuid_falls_back_only_to_one_exact_uuid_isolation() -> None:
    assert _torch_uuid(
        NoTorchUuid(),
        0,
        visible_devices="GPU-81610c7d-1bb5-844d-b2e1-1bd1097c1992",
    ) == "81610c7d1bb5844db2e11bd1097c1992"

    for invalid in (None, "0", "GPU-a,GPU-b", ""):
        with pytest.raises(RuntimeError, match="not UUID-isolated"):
            _torch_uuid(NoTorchUuid(), 0, visible_devices=invalid)


def test_worker_exits_immediately_when_cuda_is_unavailable_after_ready(tmp_path):
    raw = [
        "--root", str(tmp_path),
        "--owner", "motionllm-test",
        "--gpu-uuid", "GPU-unit-test",
        "--gpu-index", "2",
        "--reservation-id", "worker-test",
        "--heartbeat-interval-seconds", "1.0",
        "--ready-timeout-seconds", "1.0",
    ]
    args = worker_parser().parse_args(raw)
    fingerprint = command_fingerprint((sys.executable, "-m", WORKER_MODULE, *raw))
    pid = __import__("os").getpid()
    device = GPUDevice(2, "GPU-unit-test", "fake", 1000, 0, 0)
    store = KeepaliveStore(
        tmp_path,
        project_owner="motionllm-test",
        pid_alive=lambda candidate: candidate == pid,
        process_fingerprint=lambda candidate: fingerprint if candidate == pid else None,
    )
    store.reserve(device, reservation_id="worker-test")
    store.create_start_record(
        device=device,
        pid=pid,
        command_fingerprint=fingerprint,
        worker_module=WORKER_MODULE,
        worker_code_sha256=worker_code_sha256(),
        reservation_id="worker-test",
    )
    store.publish_ready(device.uuid, reservation_id="worker-test")

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run_worker(args, raw_argv=raw, torch_module=NoCuda())


def test_worker_command_is_fixed_argv_and_contains_explicit_uuid_and_index(tmp_path):
    device = GPUDevice(7, "GPU-explicit", "fake", 1000, 0, 0)
    argv = build_worker_command(
        python_executable=sys.executable,
        root=tmp_path,
        owner="motionllm-test",
        device=device,
        reservation_id="reservation",
        heartbeat_interval_seconds=30,
        ready_timeout_seconds=10,
    )
    assert argv[1:3] == ("-m", WORKER_MODULE)
    assert argv[argv.index("--gpu-uuid") + 1] == device.uuid
    assert argv[argv.index("--gpu-index") + 1] == str(device.index)
    assert ";" not in "".join(argv)
