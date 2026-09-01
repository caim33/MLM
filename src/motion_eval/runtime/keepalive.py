"""Controller-side lifecycle for the project-owned CUDA keepalive worker."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from motion_eval.core import sha256_file

from .gpu import (
    GPUDevice,
    GPUInventory,
    GPUOwnershipError,
    GPUQueryError,
    GPULeaseStore,
    KeepaliveStopProof,
    KeepaliveStore,
    NvidiaSmiProbe,
    command_fingerprint,
    system_process_fingerprint,
)

WORKER_MODULE = "motion_eval.runtime.keepalive_worker"
DEFAULT_OWNER = "motionllm"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_STALE_AFTER_SECONDS = 180.0
DEFAULT_START_TIMEOUT_SECONDS = 30.0
DEFAULT_STOP_TIMEOUT_SECONDS = 15.0


class KeepaliveProcessRuntime(Protocol):
    """Injectable shell-free process operations used by the controller."""

    def launch(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str]) -> int: ...

    def alive(self, pid: int) -> bool: ...

    def fingerprint(self, pid: int) -> str | None: ...

    def terminate(self, pid: int) -> None: ...

    def kill(self, pid: int) -> None: ...


class SystemKeepaliveProcessRuntime:
    """OS process implementation; every launch is an argv vector with shell off."""

    def __init__(self) -> None:
        self._children: dict[int, subprocess.Popen[bytes]] = {}

    def launch(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str]) -> int:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=os.name != "nt",
            start_new_session=os.name == "posix",
        )
        pid = int(process.pid)
        self._children[pid] = process
        return pid

    def alive(self, pid: int) -> bool:
        child = self._children.get(pid)
        if child is not None:
            if child.poll() is None:
                return True
            self._children.pop(pid, None)
            return False
        return KeepaliveStore._pid_alive(pid)

    def fingerprint(self, pid: int) -> str | None:
        return system_process_fingerprint(pid)

    def terminate(self, pid: int) -> None:
        if pid <= 0:
            raise GPUOwnershipError("invalid keepalive PID")
        os.kill(pid, signal.SIGTERM)

    def kill(self, pid: int) -> None:
        if pid <= 0:
            raise GPUOwnershipError("invalid keepalive PID")
        if os.name == "posix":
            os.kill(pid, signal.SIGKILL)
            return
        # Windows has no SIGKILL.  taskkill receives only a validated integer
        # in a fixed argv vector and is never invoked through a shell.
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 and KeepaliveStore._pid_alive(pid):
            raise GPUOwnershipError("failed to terminate proven keepalive PID")


def worker_source_path() -> Path:
    return Path(__file__).with_name("keepalive_worker.py").resolve(strict=True)


def worker_code_sha256() -> str:
    return sha256_file(worker_source_path())


def build_worker_command(
    *,
    python_executable: str,
    root: Path,
    owner: str,
    device: GPUDevice,
    reservation_id: str,
    heartbeat_interval_seconds: float,
    ready_timeout_seconds: float,
) -> tuple[str, ...]:
    if not python_executable:
        raise ValueError("python_executable is required")
    if heartbeat_interval_seconds <= 0 or ready_timeout_seconds <= 0:
        raise ValueError("worker intervals must be positive")
    return (
        python_executable,
        "-m",
        WORKER_MODULE,
        "--root",
        str(root),
        "--owner",
        owner,
        "--gpu-uuid",
        device.uuid,
        "--gpu-index",
        str(device.index),
        "--reservation-id",
        reservation_id,
        "--heartbeat-interval-seconds",
        str(float(heartbeat_interval_seconds)),
        "--ready-timeout-seconds",
        str(float(ready_timeout_seconds)),
    )


class KeepaliveController:
    """Starts and stops one explicit GPU keepalive with a fail-closed handshake."""

    def __init__(
        self,
        root: str | Path,
        *,
        project_owner: str = DEFAULT_OWNER,
        probe: NvidiaSmiProbe | None = None,
        process_runtime: KeepaliveProcessRuntime | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        python_executable: str | None = None,
        launch_cwd: str | Path | None = None,
    ) -> None:
        self.probe = probe or NvidiaSmiProbe()
        self.process_runtime = process_runtime or SystemKeepaliveProcessRuntime()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.sleep = sleep or time.sleep
        self.python_executable = python_executable or sys.executable
        self.launch_cwd = Path(launch_cwd or Path(__file__).resolve().parents[3]).resolve(
            strict=True
        )
        self.store = KeepaliveStore(
            root,
            project_owner=project_owner,
            probe=self.probe,
            pid_alive=self.process_runtime.alive,
            process_fingerprint=self.process_runtime.fingerprint,
            now=self.now,
        )
        self.role_leases = GPULeaseStore(
            root,
            project_owner=project_owner,
            probe=self.probe,
            # The controller owns the persistent role mutex; the separately
            # proven child PID remains in the keepalive lifecycle record.
            pid_alive=KeepaliveStore._pid_alive,
        )

    @property
    def project_owner(self) -> str:
        return self.store.project_owner

    @staticmethod
    def _resolve_device(inventory: GPUInventory, selector: str | int) -> GPUDevice:
        normalized: str | int = selector
        if isinstance(selector, str) and selector.isdecimal():
            normalized = int(selector)
        return inventory.device(normalized)

    def preview_start(self, selector: str | int) -> dict[str, object]:
        preview = self.probe.preview() if hasattr(self.probe, "preview") else {}
        return {
            "dry_run": True,
            "action": "start",
            "gpu_selector": selector,
            "root": str(self.store.root),
            "owner": self.project_owner,
            "worker_module": WORKER_MODULE,
            "protocol": ["query_idle", "reserve", "launch_waiting_child", "record", "ready", "heartbeat", "gpu_process_proof"],
            **preview,
        }

    def preview_stop(self, gpu_uuid: str) -> dict[str, object]:
        proof = self._prove_keepalive_stop(gpu_uuid)
        return {
            "dry_run": True,
            "action": "stop",
            "gpu_uuid": proof.gpu_uuid,
            "gpu_index": proof.gpu_index,
            "pid": proof.pid,
            "owner": proof.owner,
            "command_fingerprint": proof.command_fingerprint,
            "record_sha256": proof.record_sha256,
        }

    def status(self, *, stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS) -> list[dict[str, object]]:
        records = self.store.status(stale_after_seconds=stale_after_seconds)
        for record in records:
            lease = self.role_leases.assert_owned(
                str(record["gpu_uuid"]),
                lease_id=str(record["reservation_id"]),
                role="keepalive",
            )
            if lease.get("gpu_index") != record.get("gpu_index"):
                raise GPUOwnershipError("keepalive role mutex GPU binding mismatch")
        return records

    def _prove_keepalive_stop(self, gpu_uuid: str) -> KeepaliveStopProof:
        proof = self.store.prove_stop(gpu_uuid)
        lease = self.role_leases.assert_owned(
            gpu_uuid, lease_id=proof.reservation_id, role="keepalive"
        )
        if lease.get("gpu_index") != proof.gpu_index:
            raise GPUOwnershipError("keepalive role mutex GPU binding mismatch")
        return proof

    def _child_environment(self, device: GPUDevice) -> dict[str, str]:
        environment = dict(os.environ)
        # UUID binding is stronger than an ordinal when device ordering changes.
        # Inside this isolated view the worker intentionally uses logical cuda:0.
        environment["CUDA_VISIBLE_DEVICES"] = device.uuid
        environment["PYTHONUNBUFFERED"] = "1"
        source_root = str(Path(__file__).resolve().parents[2])
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source_root if not existing else os.pathsep.join((source_root, existing))
        return environment

    def _wait_for_fingerprint(self, pid: int, expected: str, deadline: float) -> None:
        while self.monotonic() < deadline:
            if not self.process_runtime.alive(pid):
                raise GPUOwnershipError("keepalive child exited before ownership registration")
            actual = self.process_runtime.fingerprint(pid)
            if actual == expected:
                return
            if actual is not None:
                raise GPUOwnershipError("launched PID command fingerprint mismatch")
            self.sleep(min(0.05, max(0.0, deadline - self.monotonic())))
        raise GPUOwnershipError("timed out proving launched keepalive PID")

    def _wait_dead(self, pid: int, deadline: float) -> bool:
        while self.process_runtime.alive(pid) and self.monotonic() < deadline:
            self.sleep(min(0.1, max(0.0, deadline - self.monotonic())))
        return not self.process_runtime.alive(pid)

    def _terminate_failed_start(self, pid: int, expected_fingerprint: str, timeout: float) -> None:
        if not self.process_runtime.alive(pid):
            return
        if self.process_runtime.fingerprint(pid) != expected_fingerprint:
            raise GPUOwnershipError("refusing to stop failed-start PID after fingerprint mismatch")
        deadline = self.monotonic() + max(0.1, timeout)
        self.process_runtime.terminate(pid)
        if self._wait_dead(pid, self.monotonic() + max(0.05, timeout * 0.6)):
            return
        if self.process_runtime.fingerprint(pid) != expected_fingerprint:
            raise GPUOwnershipError("refusing force-stop after keepalive PID identity changed")
        self.process_runtime.kill(pid)
        if not self._wait_dead(pid, deadline):
            raise GPUOwnershipError("failed-start keepalive PID did not exit within timeout")

    def start(
        self,
        selector: str | int,
        *,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        ready_timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
        wait_timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        if heartbeat_interval_seconds <= 0 or ready_timeout_seconds <= 0 or wait_timeout_seconds <= 0:
            raise ValueError("keepalive timeouts and intervals must be positive")
        initial = self.probe.query()
        device = self._resolve_device(initial, selector)
        reservation_id = os.urandom(16).hex()
        role_acquired = False
        reservation_attempted = False
        pid: int | None = None
        fingerprint: str | None = None
        try:
            # This create-if-absent mutex is the linearization point shared with
            # formal finetune/eval workers.  All probing and lifecycle
            # publication happens while the keepalive role remains owned.
            self.role_leases.acquire_role(
                device,
                role="keepalive",
                pid=os.getpid(),
                lease_id=reservation_id,
                purpose="project CUDA keepalive",
            )
            role_acquired = True
            if not initial.is_idle(device):
                raise GPUQueryError("GPU is not proven idle; keepalive start refused")
            reservation_attempted = True
            self.store.reserve(device, reservation_id=reservation_id)
            # Re-query after taking the cross-process reservation.  A process
            # could have appeared between the first observation and the lock.
            confirmed = self.probe.query()
            confirmed_device = confirmed.device(device.uuid)
            if confirmed_device.index != device.index or not confirmed.is_idle(confirmed_device):
                raise GPUQueryError("GPU ceased to be proven idle after reservation")

            argv = build_worker_command(
                python_executable=self.python_executable,
                root=self.store.root,
                owner=self.project_owner,
                device=device,
                reservation_id=reservation_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                ready_timeout_seconds=ready_timeout_seconds,
            )
            fingerprint = command_fingerprint(argv)
            pid = self.process_runtime.launch(
                argv,
                cwd=str(self.launch_cwd),
                env=self._child_environment(device),
            )
            registration_deadline = self.monotonic() + min(wait_timeout_seconds, 5.0)
            self._wait_for_fingerprint(pid, fingerprint, registration_deadline)
            self.store.create_start_record(
                device=device,
                pid=pid,
                command_fingerprint=fingerprint,
                worker_module=WORKER_MODULE,
                worker_code_sha256=worker_code_sha256(),
                reservation_id=reservation_id,
            )
            self.store.publish_ready(device.uuid, reservation_id=reservation_id)

            deadline = self.monotonic() + wait_timeout_seconds
            last_error: Exception | None = None
            while self.monotonic() < deadline:
                if not self.process_runtime.alive(pid):
                    raise GPUOwnershipError("keepalive child exited before its first heartbeat")
                try:
                    records = self.store.status(
                        stale_after_seconds=max(
                            DEFAULT_STALE_AFTER_SECONDS,
                            heartbeat_interval_seconds * 3,
                        )
                    )
                    for record in records:
                        if record.get("gpu_uuid") == device.uuid and record.get("pid") == pid:
                            return record
                except (GPUOwnershipError, GPUQueryError) as exc:
                    last_error = exc
                self.sleep(min(0.1, max(0.0, deadline - self.monotonic())))
            detail = f": {last_error}" if last_error is not None else ""
            raise GPUOwnershipError(
                f"keepalive did not produce a live heartbeat and GPU process proof{detail}"
            )
        except BaseException as start_error:
            cleanup_errors: list[str] = []
            child_gone = pid is None
            termination_error: str | None = None
            if pid is not None:
                if fingerprint is None:
                    termination_error = "child termination: command fingerprint is unavailable"
                else:
                    try:
                        self._terminate_failed_start(
                            pid, fingerprint, min(wait_timeout_seconds, 5.0)
                        )
                    except BaseException as exc:
                        termination_error = (
                            f"child termination: {type(exc).__name__}: {exc}"
                        )
                try:
                    child_gone = not self.process_runtime.alive(pid)
                except BaseException as exc:
                    cleanup_errors.append(f"child liveness: {type(exc).__name__}: {exc}")
                    child_gone = False
                if termination_error is not None and not child_gone:
                    cleanup_errors.append(termination_error)
            if reservation_attempted and child_gone:
                try:
                    self.store.cleanup_failed_start(
                        device.uuid,
                        reservation_id=reservation_id,
                        pid=pid,
                        command_fingerprint=fingerprint,
                    )
                except BaseException as exc:
                    cleanup_errors.append(f"lifecycle rollback: {type(exc).__name__}: {exc}")
            elif reservation_attempted:
                cleanup_errors.append(
                    "lifecycle rollback: keepalive child is not proven stopped"
                )
            # Once no launched child can touch CUDA, the common role capability
            # is independent of lifecycle cleanup.  Reap it even when foreign
            # or malformed lifecycle evidence correctly remains fail-closed.
            if role_acquired and child_gone:
                try:
                    self.role_leases.release(
                        device.uuid,
                        lease_id=reservation_id,
                        role="keepalive",
                    )
                except BaseException as exc:
                    cleanup_errors.append(f"role rollback: {type(exc).__name__}: {exc}")
            elif role_acquired:
                cleanup_errors.append("role rollback: keepalive child is not proven stopped")
            if cleanup_errors:
                raise GPUOwnershipError(
                    "keepalive start failed and cleanup could not be proven: "
                    + "; ".join(cleanup_errors)
                ) from start_error
            raise

    def _same_worker(self, left: KeepaliveStopProof, right: KeepaliveStopProof) -> bool:
        return (
            left.gpu_uuid == right.gpu_uuid
            and left.gpu_index == right.gpu_index
            and left.pid == right.pid
            and left.owner == right.owner
            and left.command_fingerprint == right.command_fingerprint
            and left.reservation_id == right.reservation_id
            and left.binding_sha256 == right.binding_sha256
        )

    def stop(
        self,
        gpu_uuid: str,
        *,
        wait_timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive")
        stop_proof = self._prove_keepalive_stop(gpu_uuid)
        current_proof = stop_proof
        stop_request = self.store.request_stop(stop_proof)
        deadline = self.monotonic() + wait_timeout_seconds
        graceful_deadline = min(deadline, self.monotonic() + min(2.0, wait_timeout_seconds * 0.4))
        if not self._wait_dead(current_proof.pid, graceful_deadline):
            current = self._prove_keepalive_stop(gpu_uuid)
            if not self._same_worker(current_proof, current):
                raise GPUOwnershipError("keepalive identity changed while stopping")
            current_proof = current
            self.process_runtime.terminate(current_proof.pid)
            terminate_deadline = min(deadline, self.monotonic() + max(0.1, wait_timeout_seconds * 0.35))
            if not self._wait_dead(current_proof.pid, terminate_deadline):
                current = self._prove_keepalive_stop(gpu_uuid)
                if not self._same_worker(current_proof, current):
                    raise GPUOwnershipError("keepalive identity changed before force-stop")
                current_proof = current
                self.process_runtime.kill(current_proof.pid)
                if not self._wait_dead(current_proof.pid, deadline):
                    raise GPUOwnershipError("keepalive PID did not exit within bounded stop timeout")
        self.store.finalize_stop(
            stop_proof, stop_sha256=str(stop_request["stop_sha256"])
        )
        self.role_leases.release(
            stop_proof.gpu_uuid,
            lease_id=stop_proof.reservation_id,
            role="keepalive",
        )
        return {
            "stopped": True,
            "gpu_uuid": stop_proof.gpu_uuid,
            "gpu_index": stop_proof.gpu_index,
            "pid": stop_proof.pid,
            "owner": stop_proof.owner,
        }
