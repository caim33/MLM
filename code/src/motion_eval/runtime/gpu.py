"""Fail-closed GPU inventory, project leases, and keepalive ownership records."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from motion_eval.core import atomic_write_json, resolve_within_root, sha256_json
from motion_eval.data.jsonio import load_json_strict


class GPUQueryError(RuntimeError):
    pass


class GPUOwnershipError(RuntimeError):
    pass


KEEPALIVE_SCHEMA_VERSION = "2.0"
_KEEPALIVE_BINDING_KEYS = (
    "schema_version",
    "gpu_uuid",
    "gpu_index",
    "pid",
    "owner",
    "command_fingerprint",
    "started_at",
    "worker_module",
    "worker_code_sha256",
    "reservation_id",
)
_KEEPALIVE_RECORD_KEYS = frozenset(
    (*_KEEPALIVE_BINDING_KEYS, "heartbeat_at", "state", "binding_sha256", "record_sha256")
)
_KEEPALIVE_STOP_REQUEST_KEYS = frozenset(
    (
        "schema_version",
        "gpu_uuid",
        "pid",
        "owner",
        "reservation_id",
        "binding_sha256",
        "record_sha256",
        "requested_at",
        "stop_sha256",
    )
)


@dataclass(frozen=True)
class KeepaliveStopProof:
    """Immutable ownership evidence captured immediately before a stop."""

    gpu_uuid: str
    gpu_index: int
    pid: int
    owner: str
    command_fingerprint: str
    reservation_id: str
    binding_sha256: str
    record_sha256: str
    heartbeat_at: str


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int

    @property
    def idle(self) -> bool:
        return self.utilization_percent == 0 and self.memory_used_mib <= 16


@dataclass(frozen=True)
class GPUProcess:
    gpu_uuid: str
    pid: int
    used_memory_mib: int


@dataclass(frozen=True)
class GPUInventory:
    devices: tuple[GPUDevice, ...]
    processes: tuple[GPUProcess, ...]

    def device(self, uuid_or_index: str | int) -> GPUDevice:
        for device in self.devices:
            if device.uuid == str(uuid_or_index) or device.index == uuid_or_index:
                return device
        raise GPUQueryError(f"GPU not found: {uuid_or_index}")

    def is_idle(self, device: GPUDevice) -> bool:
        return device.idle and not any(process.gpu_uuid == device.uuid for process in self.processes)


def _csv_rows(text: str) -> list[list[str]]:
    return [[field.strip() for field in row] for row in csv.reader(io.StringIO(text)) if row]


def command_fingerprint(argv: Sequence[str]) -> str:
    """Hash an argv vector without ever persisting its plaintext form."""

    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("command argv must contain non-empty strings")
    return sha256_json({"argv": list(argv)})


def _windows_process_argv(pid: int) -> list[str] | None:
    """Read a Windows process command line without invoking a shell or WMI.

    ``ProcessCommandLineInformation`` returns one ``UNICODE_STRING`` whose
    backing buffer is owned by the caller.  Parsing that raw command line with
    ``CommandLineToArgvW`` mirrors the quoting rules used by ``Popen`` for an
    argv-vector launch, so the resulting hash has the same meaning as the
    Linux ``/proc/<pid>/cmdline`` hash.
    """

    if pid <= 0 or pid > 0xFFFFFFFF:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        ntdll.NtQueryInformationProcess.argtypes = [
            wintypes.HANDLE,
            wintypes.ULONG,
            wintypes.LPVOID,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        ntdll.NtQueryInformationProcess.restype = wintypes.LONG
        shell32.CommandLineToArgvW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_int),
        ]
        shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)

        class _UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p),
            ]

        process_query_limited_information = 0x1000
        process_command_line_information = 60
        status_info_length_mismatch = 0xC0000004
        maximum_result_bytes = 1 << 20

        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return None
        try:
            needed = wintypes.ULONG()
            status = ntdll.NtQueryInformationProcess(
                handle,
                process_command_line_information,
                None,
                0,
                ctypes.byref(needed),
            )
            if ctypes.c_ulong(status).value != status_info_length_mismatch:
                return None

            result = None
            for _attempt in range(3):
                size = int(needed.value)
                if not ctypes.sizeof(_UnicodeString) <= size <= maximum_result_bytes:
                    return None
                result = ctypes.create_string_buffer(size)
                status = ntdll.NtQueryInformationProcess(
                    handle,
                    process_command_line_information,
                    result,
                    size,
                    ctypes.byref(needed),
                )
                if status == 0:
                    break
                if ctypes.c_ulong(status).value != status_info_length_mismatch:
                    return None
            else:
                return None

            command = _UnicodeString.from_buffer(result)
            length = int(command.Length)
            address = int(command.Buffer or 0)
            result_start = ctypes.addressof(result)
            result_end = result_start + ctypes.sizeof(result)
            if (
                not address
                or length <= 0
                or length % 2
                or length > int(command.MaximumLength)
                or address < result_start
                or address + length > result_end
            ):
                return None
            raw_command = ctypes.wstring_at(address, length // 2)
            argument_count = ctypes.c_int()
            arguments = shell32.CommandLineToArgvW(
                raw_command, ctypes.byref(argument_count)
            )
            if not arguments:
                return None
            try:
                return [arguments[index] for index in range(argument_count.value)]
            finally:
                kernel32.LocalFree(ctypes.cast(arguments, ctypes.c_void_p))
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def system_process_fingerprint(pid: int) -> str | None:
    """Return an exact OS process argv hash, or ``None`` when unknowable."""

    if pid <= 0:
        return None
    if os.name == "nt":
        argv = _windows_process_argv(pid)
    elif os.name == "posix":
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            values = raw.rstrip(b"\0").split(b"\0") if raw else []
            argv = [value.decode("utf-8", "surrogateescape") for value in values]
        except OSError:
            return None
    else:
        return None
    if not argv or any(not item for item in argv):
        return None
    return command_fingerprint(argv)


class NvidiaSmiProbe:
    device_query = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    process_query = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    )

    def preview(self) -> dict[str, list[str]]:
        return {"device_query": list(self.device_query), "process_query": list(self.process_query)}

    def _run(self, argv: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                list(argv),
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GPUQueryError(f"GPU query failed closed: {type(exc).__name__}") from exc
        if result.returncode != 0:
            raise GPUQueryError(f"GPU query failed closed with exit code {result.returncode}")
        return result.stdout

    def query(self) -> GPUInventory:
        try:
            device_rows = _csv_rows(self._run(self.device_query))
            process_text = self._run(self.process_query)
            devices = tuple(
                GPUDevice(int(row[0]), row[1], row[2], int(row[3]), int(row[4]), int(row[5]))
                for row in device_rows
                if len(row) == 6
            )
            if len(devices) != len(device_rows) or not devices:
                raise ValueError("invalid or empty device inventory")
            if (
                len({device.index for device in devices}) != len(devices)
                or len({device.uuid for device in devices}) != len(devices)
                or any(
                    device.index < 0
                    or not device.uuid
                    or device.memory_total_mib <= 0
                    or device.memory_used_mib < 0
                    or device.memory_used_mib > device.memory_total_mib
                    or not 0 <= device.utilization_percent <= 100
                    for device in devices
                )
            ):
                raise ValueError("invalid or duplicate device identity")
            processes: list[GPUProcess] = []
            for row in _csv_rows(process_text):
                if len(row) != 3:
                    raise ValueError("invalid process inventory")
                process = GPUProcess(row[0], int(row[1]), int(row[2]))
                if (
                    process.gpu_uuid not in {device.uuid for device in devices}
                    or process.pid <= 0
                    or process.used_memory_mib < 0
                ):
                    raise ValueError("invalid process identity")
                processes.append(process)
        except (ValueError, IndexError) as exc:
            raise GPUQueryError("GPU query output could not be parsed; status is unknown") from exc
        return GPUInventory(devices, tuple(processes))


class KeepaliveStore:
    """Hash-bound ownership records for a single project keepalive namespace.

    A lifecycle uses four files.  The reservation is the atomic, per-GPU
    create-if-absent lock.  The controller then writes the owner record and a
    ready signal.  A child must validate both before touching CUDA.  A stop
    request is likewise bound to the immutable owner identity.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        project_owner: str,
        probe: NvidiaSmiProbe | None = None,
        pid_alive: Callable[[int], bool] | None = None,
        process_fingerprint: Callable[[int], str | None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not project_owner or any(ord(char) < 32 for char in project_owner):
            raise ValueError("project_owner must be non-empty and control-free")
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve(strict=True)
        self.project_owner = project_owner
        self.probe = probe or NvidiaSmiProbe()
        self.pid_alive = pid_alive or self._pid_alive
        self.process_fingerprint = process_fingerprint or system_process_fingerprint
        self.now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            # ``os.kill(pid, 0)`` is not a harmless existence probe on
            # Windows; it can terminate the target.  Query a process handle
            # without requesting terminate rights instead.
            try:
                import ctypes

                process_query_limited_information = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(
                    process_query_limited_information, False, pid
                )
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            except (AttributeError, OSError, ValueError):
                return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _safe_gpu_name(gpu_uuid: str) -> str:
        if not isinstance(gpu_uuid, str):
            raise ValueError("GPU UUID must be a string")
        safe = gpu_uuid.replace("/", "_").replace("\\", "_")
        if not safe or safe in {".", ".."} or any(ord(char) < 32 for char in safe):
            raise ValueError("invalid GPU UUID")
        return safe

    def _lifecycle_path(self, gpu_uuid: str, suffix: str) -> Path:
        safe = self._safe_gpu_name(gpu_uuid)
        return resolve_within_root(self.root / f"{safe}{suffix}", self.root, must_exist=False)

    def _path(self, gpu_uuid: str) -> Path:
        # Keep the original record filename for compatibility with existing
        # deployments; auxiliary lifecycle files deliberately do not end in
        # .json so record enumeration cannot confuse them.
        return self._lifecycle_path(gpu_uuid, ".json")

    def _reservation_path(self, gpu_uuid: str) -> Path:
        return self._lifecycle_path(gpu_uuid, ".reservation")

    def _ready_path(self, gpu_uuid: str) -> Path:
        return self._lifecycle_path(gpu_uuid, ".ready")

    def _stop_path(self, gpu_uuid: str) -> Path:
        return self._lifecycle_path(gpu_uuid, ".stop")

    def _now(self) -> datetime:
        value = self.now()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise GPUOwnershipError("keepalive clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _read_hashed(path: Path, hash_field: str, label: str) -> dict[str, object]:
        try:
            value = load_json_strict(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise GPUOwnershipError(f"invalid {label}: {path.name}") from exc
        if not isinstance(value, Mapping):
            raise GPUOwnershipError(f"invalid {label}: {path.name}")
        record = dict(value)
        body = {key: item for key, item in record.items() if key != hash_field}
        digest = record.get(hash_field)
        if not isinstance(digest, str) or digest != sha256_json(body):
            raise GPUOwnershipError(f"tampered {label}: {path.name}")
        return record

    @staticmethod
    def _hashed(body: Mapping[str, object], hash_field: str) -> dict[str, object]:
        copied = dict(body)
        return {**copied, hash_field: sha256_json(copied)}

    @staticmethod
    def _binding(record: Mapping[str, object]) -> dict[str, object]:
        try:
            return {key: record[key] for key in _KEEPALIVE_BINDING_KEYS}
        except KeyError as exc:
            raise GPUOwnershipError(f"keepalive record missing {exc.args[0]}") from exc

    @staticmethod
    def _timestamp(value: object, *, label: str) -> datetime:
        if not isinstance(value, str) or not value:
            raise GPUOwnershipError(f"invalid {label} timestamp")
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                raise ValueError("timezone is required")
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError) as exc:
            raise GPUOwnershipError(f"invalid {label} timestamp") from exc

    def _validate_record_shape(self, record: Mapping[str, object]) -> dict[str, object]:
        if set(record) != _KEEPALIVE_RECORD_KEYS:
            raise GPUOwnershipError("invalid keepalive record shape")
        if record.get("schema_version") != KEEPALIVE_SCHEMA_VERSION:
            raise GPUOwnershipError("unsupported keepalive record schema")
        if record.get("owner") != self.project_owner:
            raise GPUOwnershipError("refusing to manage a non-project keepalive")
        for key in (
            "gpu_uuid",
            "command_fingerprint",
            "started_at",
            "heartbeat_at",
            "worker_module",
            "worker_code_sha256",
            "reservation_id",
            "binding_sha256",
            "record_sha256",
        ):
            if not isinstance(record.get(key), str) or not record[key]:
                raise GPUOwnershipError(f"invalid keepalive record field: {key}")
        if not isinstance(record.get("gpu_index"), int) or int(record["gpu_index"]) < 0:
            raise GPUOwnershipError("invalid keepalive GPU index")
        if not isinstance(record.get("pid"), int) or int(record["pid"]) <= 0:
            raise GPUOwnershipError("invalid keepalive PID")
        if record.get("state") != "active":
            raise GPUOwnershipError("keepalive has not produced a live heartbeat")
        binding = self._binding(record)
        if record.get("binding_sha256") != sha256_json(binding):
            raise GPUOwnershipError("tampered keepalive binding")
        started = self._timestamp(record.get("started_at"), label="keepalive start")
        heartbeat = self._timestamp(record.get("heartbeat_at"), label="keepalive heartbeat")
        if heartbeat < started:
            raise GPUOwnershipError("keepalive heartbeat predates worker start")
        return dict(record)

    def load_record(self, gpu_uuid: str, *, require_active: bool = True) -> dict[str, object]:
        record = self._read_hashed(self._path(gpu_uuid), "record_sha256", "keepalive record")
        if require_active:
            return self._validate_record_shape(record)
        if set(record) != _KEEPALIVE_RECORD_KEYS:
            raise GPUOwnershipError("invalid keepalive record shape")
        if record.get("schema_version") != KEEPALIVE_SCHEMA_VERSION:
            raise GPUOwnershipError("unsupported keepalive record schema")
        if record.get("owner") != self.project_owner:
            raise GPUOwnershipError("refusing to manage a non-project keepalive")
        binding = self._binding(record)
        if record.get("binding_sha256") != sha256_json(binding):
            raise GPUOwnershipError("tampered keepalive binding")
        if record.get("state") not in {"starting", "active"}:
            raise GPUOwnershipError("invalid keepalive state")
        started = self._timestamp(record.get("started_at"), label="keepalive start")
        heartbeat = self._timestamp(record.get("heartbeat_at"), label="keepalive heartbeat")
        if heartbeat < started:
            raise GPUOwnershipError("keepalive heartbeat predates worker start")
        return record

    def reserve(self, device: GPUDevice, *, reservation_id: str) -> dict[str, object]:
        if not reservation_id or any(ord(char) < 32 for char in reservation_id):
            raise ValueError("reservation_id must be non-empty and control-free")
        for path in (
            self._path(device.uuid),
            self._reservation_path(device.uuid),
            self._ready_path(device.uuid),
            self._stop_path(device.uuid),
        ):
            if path.exists():
                raise GPUOwnershipError(
                    f"GPU {device.uuid} already has keepalive lifecycle evidence"
                )
        body: dict[str, object] = {
            "schema_version": KEEPALIVE_SCHEMA_VERSION,
            "gpu_uuid": device.uuid,
            "gpu_index": device.index,
            "owner": self.project_owner,
            "reservation_id": reservation_id,
            "reserved_at": self._now().isoformat(),
        }
        reservation = self._hashed(body, "reservation_sha256")
        try:
            atomic_write_json(
                self._reservation_path(device.uuid),
                reservation,
                root=self.root,
                overwrite=False,
            )
        except FileExistsError as exc:
            raise GPUOwnershipError(
                f"GPU {device.uuid} was reserved by another keepalive start"
            ) from exc
        return reservation

    def _load_reservation(self, gpu_uuid: str) -> dict[str, object]:
        reservation = self._read_hashed(
            self._reservation_path(gpu_uuid), "reservation_sha256", "keepalive reservation"
        )
        if (
            reservation.get("schema_version") != KEEPALIVE_SCHEMA_VERSION
            or reservation.get("gpu_uuid") != gpu_uuid
            or reservation.get("owner") != self.project_owner
            or not isinstance(reservation.get("gpu_index"), int)
            or not isinstance(reservation.get("reservation_id"), str)
        ):
            raise GPUOwnershipError("keepalive reservation ownership mismatch")
        return reservation

    def release_reservation(self, gpu_uuid: str, *, reservation_id: str) -> None:
        reservation = self._load_reservation(gpu_uuid)
        if reservation.get("reservation_id") != reservation_id:
            raise GPUOwnershipError("refusing to release another keepalive reservation")
        if self._path(gpu_uuid).exists() or self._ready_path(gpu_uuid).exists():
            raise GPUOwnershipError("cannot release a reservation with published ownership")
        self._reservation_path(gpu_uuid).unlink()

    def create_start_record(
        self,
        *,
        device: GPUDevice,
        pid: int,
        command_fingerprint: str,
        worker_module: str,
        worker_code_sha256: str,
        reservation_id: str,
    ) -> dict[str, object]:
        reservation = self._load_reservation(device.uuid)
        if (
            reservation.get("reservation_id") != reservation_id
            or reservation.get("gpu_index") != device.index
        ):
            raise GPUOwnershipError("keepalive reservation does not match target GPU")
        if not self.pid_alive(pid):
            raise GPUOwnershipError("keepalive PID is not alive")
        actual = self.process_fingerprint(pid)
        if actual is None or actual != command_fingerprint:
            raise GPUOwnershipError("cannot prove launched keepalive command fingerprint")
        if not worker_module or len(worker_code_sha256) != 64:
            raise GPUOwnershipError("worker module and SHA-256 are required")
        started = self._now().isoformat()
        body: dict[str, object] = {
            "schema_version": KEEPALIVE_SCHEMA_VERSION,
            "gpu_uuid": device.uuid,
            "gpu_index": device.index,
            "pid": pid,
            "owner": self.project_owner,
            "command_fingerprint": command_fingerprint,
            "started_at": started,
            "heartbeat_at": started,
            "worker_module": worker_module,
            "worker_code_sha256": worker_code_sha256,
            "reservation_id": reservation_id,
            "state": "starting",
        }
        body["binding_sha256"] = sha256_json(self._binding(body))
        record = self._hashed(body, "record_sha256")
        atomic_write_json(self._path(device.uuid), record, root=self.root, overwrite=False)
        return record

    def publish_ready(self, gpu_uuid: str, *, reservation_id: str) -> dict[str, object]:
        record = self.load_record(gpu_uuid, require_active=False)
        reservation = self._load_reservation(gpu_uuid)
        if (
            record.get("reservation_id") != reservation_id
            or reservation.get("reservation_id") != reservation_id
        ):
            raise GPUOwnershipError("keepalive ready signal ownership mismatch")
        body: dict[str, object] = {
            "schema_version": KEEPALIVE_SCHEMA_VERSION,
            "gpu_uuid": gpu_uuid,
            "owner": self.project_owner,
            "reservation_id": reservation_id,
            "binding_sha256": record["binding_sha256"],
            "ready_at": self._now().isoformat(),
        }
        ready = self._hashed(body, "ready_sha256")
        atomic_write_json(self._ready_path(gpu_uuid), ready, root=self.root, overwrite=False)
        return ready

    def ready_exists(self, gpu_uuid: str) -> bool:
        return self._ready_path(gpu_uuid).is_file()

    def validate_ready(
        self,
        gpu_uuid: str,
        *,
        pid: int,
        reservation_id: str,
        command_fingerprint: str,
        worker_module: str,
        worker_code_sha256: str,
    ) -> dict[str, object]:
        record = self.load_record(gpu_uuid, require_active=False)
        ready = self._read_hashed(self._ready_path(gpu_uuid), "ready_sha256", "keepalive ready signal")
        reservation = self._load_reservation(gpu_uuid)
        expected = {
            "pid": pid,
            "reservation_id": reservation_id,
            "command_fingerprint": command_fingerprint,
            "worker_module": worker_module,
            "worker_code_sha256": worker_code_sha256,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise GPUOwnershipError("worker identity does not match keepalive record")
        if (
            ready.get("schema_version") != KEEPALIVE_SCHEMA_VERSION
            or ready.get("gpu_uuid") != gpu_uuid
            or ready.get("owner") != self.project_owner
            or ready.get("reservation_id") != reservation_id
            or ready.get("binding_sha256") != record.get("binding_sha256")
            or reservation.get("reservation_id") != reservation_id
        ):
            raise GPUOwnershipError("keepalive ready signal is not bound to this worker")
        return record

    @staticmethod
    def _process_on_device(inventory: GPUInventory, gpu_uuid: str, pid: int) -> bool:
        return any(
            process.gpu_uuid == gpu_uuid and process.pid == pid
            for process in inventory.processes
        )

    def heartbeat(
        self,
        gpu_uuid: str,
        *,
        pid: int,
        reservation_id: str | None = None,
        command_fingerprint: str | None = None,
        worker_module: str | None = None,
        worker_code_sha256: str | None = None,
    ) -> dict[str, object]:
        record = self.load_record(gpu_uuid, require_active=False)
        expected = {
            "pid": pid,
            "reservation_id": reservation_id or record.get("reservation_id"),
            "command_fingerprint": command_fingerprint or record.get("command_fingerprint"),
            "worker_module": worker_module or record.get("worker_module"),
            "worker_code_sha256": worker_code_sha256 or record.get("worker_code_sha256"),
        }
        self.validate_ready(gpu_uuid, **expected)  # type: ignore[arg-type]
        actual = self.process_fingerprint(pid)
        if actual is None or actual != record.get("command_fingerprint"):
            raise GPUOwnershipError("keepalive PID fingerprint changed before heartbeat")
        inventory = self.probe.query()
        device = inventory.device(gpu_uuid)
        if device.index != record.get("gpu_index") or not self._process_on_device(
            inventory, gpu_uuid, pid
        ):
            raise GPUOwnershipError("keepalive GPU process cannot be proven")
        body = {key: item for key, item in record.items() if key != "record_sha256"}
        body["heartbeat_at"] = self._now().isoformat()
        body["state"] = "active"
        updated = self._hashed(body, "record_sha256")
        atomic_write_json(self._path(gpu_uuid), updated, root=self.root, overwrite=True)
        return updated

    def _validate_liveness(
        self,
        record: Mapping[str, object],
        *,
        inventory: GPUInventory,
        stale_after_seconds: float | None,
    ) -> None:
        pid = int(record["pid"])
        if not self.pid_alive(pid):
            raise GPUOwnershipError("keepalive PID is no longer alive")
        actual = self.process_fingerprint(pid)
        if actual is None or actual != record.get("command_fingerprint"):
            raise GPUOwnershipError("keepalive PID was reused or command fingerprint changed")
        device = inventory.device(str(record["gpu_uuid"]))
        if device.index != record.get("gpu_index"):
            raise GPUOwnershipError("keepalive GPU index/UUID mapping changed")
        if not self._process_on_device(inventory, device.uuid, pid):
            raise GPUOwnershipError("nvidia-smi does not prove the keepalive PID on its GPU")
        if stale_after_seconds is not None:
            if stale_after_seconds <= 0:
                raise ValueError("stale_after_seconds must be positive")
            try:
                parsed = datetime.fromisoformat(str(record["heartbeat_at"]))
                if parsed.tzinfo is None:
                    raise ValueError("timezone is required")
                heartbeat = parsed.astimezone(timezone.utc)
            except (TypeError, ValueError) as exc:
                raise GPUOwnershipError("invalid keepalive heartbeat timestamp") from exc
            age = self._now() - heartbeat
            if age < timedelta(seconds=-5) or age > timedelta(seconds=stale_after_seconds):
                raise GPUOwnershipError("keepalive heartbeat is stale or from the future")

    def _validate_lifecycle_bindings(self, record: Mapping[str, object]) -> None:
        gpu_uuid = str(record["gpu_uuid"])
        reservation = self._load_reservation(gpu_uuid)
        ready = self._read_hashed(self._ready_path(gpu_uuid), "ready_sha256", "keepalive ready signal")
        if (
            reservation.get("reservation_id") != record.get("reservation_id")
            or reservation.get("gpu_index") != record.get("gpu_index")
            or ready.get("reservation_id") != record.get("reservation_id")
            or ready.get("binding_sha256") != record.get("binding_sha256")
            or ready.get("owner") != self.project_owner
        ):
            raise GPUOwnershipError("keepalive lifecycle binding mismatch")

    def status(self, *, stale_after_seconds: float = 180.0) -> list[dict[str, object]]:
        record_paths = sorted(self.root.glob("*.json"))
        auxiliary = [
            *self.root.glob("*.reservation"),
            *self.root.glob("*.ready"),
            *self.root.glob("*.stop"),
        ]
        if not record_paths:
            if auxiliary:
                raise GPUOwnershipError("orphaned keepalive lifecycle evidence exists")
            return []
        inventory = self.probe.query()  # Unknown GPU state is always fatal.
        records: list[dict[str, object]] = []
        known_auxiliary: set[Path] = set()
        for path in record_paths:
            gpu_uuid = path.name[:-5]
            record = self.load_record(gpu_uuid)
            if record.get("gpu_uuid") != gpu_uuid:
                raise GPUOwnershipError(f"keepalive filename/UUID mismatch: {path.name}")
            self._validate_lifecycle_bindings(record)
            self._validate_liveness(
                record, inventory=inventory, stale_after_seconds=stale_after_seconds
            )
            known_auxiliary.update(
                {
                    self._reservation_path(gpu_uuid),
                    self._ready_path(gpu_uuid),
                    self._stop_path(gpu_uuid),
                }
            )
            records.append({**record, "alive": True, "gpu_process_proven": True})
        unexpected = [path for path in auxiliary if path not in known_auxiliary]
        if unexpected:
            raise GPUOwnershipError("orphaned keepalive lifecycle evidence exists")
        return records

    def prove_stop(self, gpu_uuid: str) -> KeepaliveStopProof:
        record = self.load_record(gpu_uuid)
        self._validate_lifecycle_bindings(record)
        inventory = self.probe.query()
        self._validate_liveness(record, inventory=inventory, stale_after_seconds=None)
        return KeepaliveStopProof(
            gpu_uuid=str(record["gpu_uuid"]),
            gpu_index=int(record["gpu_index"]),
            pid=int(record["pid"]),
            owner=str(record["owner"]),
            command_fingerprint=str(record["command_fingerprint"]),
            reservation_id=str(record["reservation_id"]),
            binding_sha256=str(record["binding_sha256"]),
            record_sha256=str(record["record_sha256"]),
            heartbeat_at=str(record["heartbeat_at"]),
        )

    def prove_owned_for_stop(self, gpu_uuid: str) -> int:
        """Compatibility API returning the PID after full ownership proof."""

        return self.prove_stop(gpu_uuid).pid

    def request_stop(self, proof: KeepaliveStopProof) -> dict[str, object]:
        current = self.prove_stop(proof.gpu_uuid)
        if current != proof:
            raise GPUOwnershipError("keepalive record changed before stop request")
        body: dict[str, object] = {
            "schema_version": KEEPALIVE_SCHEMA_VERSION,
            "gpu_uuid": proof.gpu_uuid,
            "pid": proof.pid,
            "owner": proof.owner,
            "reservation_id": proof.reservation_id,
            "binding_sha256": proof.binding_sha256,
            "record_sha256": proof.record_sha256,
            "requested_at": self._now().isoformat(),
        }
        request = self._hashed(body, "stop_sha256")
        atomic_write_json(self._stop_path(proof.gpu_uuid), request, root=self.root, overwrite=False)
        return request

    def stop_requested(
        self,
        gpu_uuid: str,
        *,
        pid: int,
        reservation_id: str,
        binding_sha256: str,
    ) -> bool:
        path = self._stop_path(gpu_uuid)
        if not path.exists():
            return False
        request = self._read_hashed(path, "stop_sha256", "keepalive stop request")
        if (
            set(request) != _KEEPALIVE_STOP_REQUEST_KEYS
            or request.get("schema_version") != KEEPALIVE_SCHEMA_VERSION
            or request.get("gpu_uuid") != gpu_uuid
            or request.get("pid") != pid
            or request.get("owner") != self.project_owner
            or request.get("reservation_id") != reservation_id
            or request.get("binding_sha256") != binding_sha256
        ):
            raise GPUOwnershipError("keepalive stop request ownership mismatch")
        self._timestamp(request.get("requested_at"), label="keepalive stop request")
        return True

    def finalize_stop(self, proof: KeepaliveStopProof, *, stop_sha256: str) -> None:
        if self.pid_alive(proof.pid):
            raise GPUOwnershipError("keepalive PID is still alive")
        if not isinstance(stop_sha256, str) or len(stop_sha256) != 64:
            raise GPUOwnershipError("a bound keepalive stop request proof is required")
        request = self._read_hashed(
            self._stop_path(proof.gpu_uuid), "stop_sha256", "keepalive stop request"
        )
        if (
            set(request) != _KEEPALIVE_STOP_REQUEST_KEYS
            or request.get("stop_sha256") != stop_sha256
            or request.get("schema_version") != KEEPALIVE_SCHEMA_VERSION
            or request.get("gpu_uuid") != proof.gpu_uuid
            or request.get("pid") != proof.pid
            or request.get("owner") != proof.owner
            or request.get("reservation_id") != proof.reservation_id
            or request.get("binding_sha256") != proof.binding_sha256
            or request.get("record_sha256") != proof.record_sha256
        ):
            raise GPUOwnershipError("keepalive stop request changed before cleanup")
        requested_at = self._timestamp(
            request.get("requested_at"), label="keepalive stop request"
        )
        proof_heartbeat = self._timestamp(
            proof.heartbeat_at, label="keepalive stop proof heartbeat"
        )
        if requested_at < proof_heartbeat:
            raise GPUOwnershipError("keepalive stop request predates its ownership proof")

        # Reload only after the process has been observed dead/reaped.  A worker
        # can serialize one last heartbeat around publication of the stop file.
        # The exact schema, active state, and binding digest make heartbeat_at
        # the only mutable field that can differ from the pre-stop proof.
        record = self.load_record(proof.gpu_uuid)
        if (
            record.get("gpu_uuid") != proof.gpu_uuid
            or record.get("gpu_index") != proof.gpu_index
            or record.get("pid") != proof.pid
            or record.get("owner") != proof.owner
            or record.get("command_fingerprint") != proof.command_fingerprint
            or record.get("reservation_id") != proof.reservation_id
            or record.get("binding_sha256") != proof.binding_sha256
        ):
            raise GPUOwnershipError("keepalive identity changed after stop")
        if record.get("record_sha256") != proof.record_sha256:
            heartbeat = self._timestamp(
                record.get("heartbeat_at"), label="final keepalive heartbeat"
            )
            if heartbeat < proof_heartbeat:
                raise GPUOwnershipError("keepalive heartbeat moved backwards after stop")
            if heartbeat > self._now() + timedelta(seconds=5):
                raise GPUOwnershipError("keepalive heartbeat is from the future")
        reservation = self._load_reservation(proof.gpu_uuid)
        if reservation.get("reservation_id") != proof.reservation_id:
            raise GPUOwnershipError("keepalive reservation changed after stop")
        for path in (
            self._stop_path(proof.gpu_uuid),
            self._ready_path(proof.gpu_uuid),
            self._path(proof.gpu_uuid),
            self._reservation_path(proof.gpu_uuid),
        ):
            path.unlink(missing_ok=True)

    def mark_stopped(
        self,
        gpu_uuid: str,
        *,
        pid: int,
        proof: KeepaliveStopProof | None = None,
        stop_sha256: str | None = None,
    ) -> None:
        """Compatibility wrapper requiring both ownership and request proofs."""

        if proof is None:
            raise GPUOwnershipError("a pre-stop ownership proof is required")
        if proof.gpu_uuid != gpu_uuid or proof.pid != pid:
            raise GPUOwnershipError("PID or GPU does not match stop proof")
        if stop_sha256 is None:
            raise GPUOwnershipError("a bound keepalive stop request proof is required")
        self.finalize_stop(proof, stop_sha256=stop_sha256)

    def cleanup_failed_start(
        self,
        gpu_uuid: str,
        *,
        reservation_id: str,
        pid: int | None = None,
        command_fingerprint: str | None = None,
    ) -> None:
        """Remove only this invocation's evidence after its child is gone."""

        if pid is not None and self.pid_alive(pid):
            raise GPUOwnershipError("cannot clean a failed start while its PID is alive")
        owned_paths: list[Path] = []
        conflicts: list[str] = []
        record_path = self._path(gpu_uuid)
        record: dict[str, object] | None = None
        if record_path.exists():
            try:
                candidate = self.load_record(gpu_uuid, require_active=False)
            except FileNotFoundError:
                pass
            except GPUOwnershipError:
                conflicts.append("record")
            else:
                if (
                    candidate.get("reservation_id") == reservation_id
                    and (pid is None or candidate.get("pid") == pid)
                    and (
                        command_fingerprint is None
                        or candidate.get("command_fingerprint") == command_fingerprint
                    )
                ):
                    record = candidate
                    owned_paths.append(record_path)
                else:
                    conflicts.append("record")

        ready_path = self._ready_path(gpu_uuid)
        if ready_path.exists():
            try:
                ready = self._read_hashed(
                    ready_path, "ready_sha256", "keepalive ready signal"
                )
            except FileNotFoundError:
                pass
            except GPUOwnershipError:
                conflicts.append("ready")
            else:
                if (
                    set(ready)
                    == {
                        "schema_version",
                        "gpu_uuid",
                        "owner",
                        "reservation_id",
                        "binding_sha256",
                        "ready_at",
                        "ready_sha256",
                    }
                    and ready.get("schema_version") == KEEPALIVE_SCHEMA_VERSION
                    and ready.get("gpu_uuid") == gpu_uuid
                    and ready.get("owner") == self.project_owner
                    and ready.get("reservation_id") == reservation_id
                    and (
                        record is None
                        or ready.get("binding_sha256") == record.get("binding_sha256")
                    )
                ):
                    owned_paths.append(ready_path)
                else:
                    conflicts.append("ready")

        stop_path = self._stop_path(gpu_uuid)
        if stop_path.exists():
            try:
                request = self._read_hashed(
                    stop_path, "stop_sha256", "keepalive stop request"
                )
            except FileNotFoundError:
                pass
            except GPUOwnershipError:
                conflicts.append("stop")
            else:
                if (
                    set(request) == _KEEPALIVE_STOP_REQUEST_KEYS
                    and request.get("schema_version") == KEEPALIVE_SCHEMA_VERSION
                    and request.get("gpu_uuid") == gpu_uuid
                    and request.get("owner") == self.project_owner
                    and request.get("reservation_id") == reservation_id
                    and (pid is None or request.get("pid") == pid)
                    and (
                        record is None
                        or request.get("binding_sha256") == record.get("binding_sha256")
                    )
                ):
                    owned_paths.append(stop_path)
                else:
                    conflicts.append("stop")

        reservation_path = self._reservation_path(gpu_uuid)
        if reservation_path.exists():
            try:
                reservation = self._load_reservation(gpu_uuid)
            except FileNotFoundError:
                pass
            except GPUOwnershipError:
                conflicts.append("reservation")
            else:
                if reservation.get("reservation_id") == reservation_id:
                    owned_paths.append(reservation_path)
                else:
                    conflicts.append("reservation")

        failures: list[str] = []
        # Auxiliary evidence is removed before the owner record/reservation.
        # Every deletion was independently bound above, so one conflicting or
        # temporarily locked file never prevents reaping other files we own.
        for path in (stop_path, ready_path, record_path, reservation_path):
            if path not in owned_paths:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failures.append(path.suffix)
        if conflicts or failures:
            details: list[str] = []
            if conflicts:
                details.append("foreign or invalid: " + ", ".join(conflicts))
            if failures:
                details.append("not reaped: " + ", ".join(failures))
            raise GPUOwnershipError(
                "failed-start lifecycle evidence could not be fully reaped: "
                + "; ".join(details)
            )

    def register(
        self,
        *,
        gpu_uuid: str,
        pid: int,
        command_fingerprint: str,
    ) -> dict[str, object]:
        """Legacy registration API retained for compatibility.

        New callers must use :class:`KeepaliveController`, which supplies the
        child-ready handshake.  This method still refuses an unverified PID.
        """

        inventory = self.probe.query()
        device = inventory.device(gpu_uuid)
        if not inventory.is_idle(device):
            raise GPUQueryError("GPU is not proven idle; keepalive registration refused")
        actual = self.process_fingerprint(pid)
        if actual is None or actual != command_fingerprint:
            raise GPUOwnershipError("cannot prove legacy keepalive PID fingerprint")
        reservation_id = os.urandom(16).hex()
        self.reserve(device, reservation_id=reservation_id)
        try:
            record = self.create_start_record(
                device=device,
                pid=pid,
                command_fingerprint=command_fingerprint,
                worker_module="legacy.external.keepalive",
                worker_code_sha256=sha256_json({"legacy": True}),
                reservation_id=reservation_id,
            )
            self.publish_ready(device.uuid, reservation_id=reservation_id)
            return record
        except Exception:
            if not self._path(device.uuid).exists() and self._reservation_path(device.uuid).exists():
                self.release_reservation(device.uuid, reservation_id=reservation_id)
            raise

    def prepare_worker(self, gpu_uuid: str) -> None:
        """Block finetune/eval while any lifecycle evidence exists on this GPU."""

        lifecycle = (
            self._path(gpu_uuid),
            self._reservation_path(gpu_uuid),
            self._ready_path(gpu_uuid),
            self._stop_path(gpu_uuid),
        )
        if any(path.exists() for path in lifecycle):
            raise GPUOwnershipError(
                f"project keepalive on {gpu_uuid} must be stopped and reaped before worker launch"
            )
        inventory = self.probe.query()
        inventory.device(gpu_uuid)  # Query success and identity are mandatory.


class GPULeaseStore:
    """Atomic per-GPU role mutex shared by keepalive and model workers.

    The role file is deliberately separate from keepalive's four lifecycle
    files.  Creation is create-if-absent, so a keepalive and a finetune/eval
    controller can never both own the same GPU.  The lease remains present for
    the complete role lifetime and may only be removed by its owner/lease ID.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        project_owner: str,
        probe: NvidiaSmiProbe | None = None,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve(strict=True)
        if not project_owner:
            raise ValueError("project_owner is required")
        self.project_owner = project_owner
        self.probe = probe or NvidiaSmiProbe()
        self.pid_alive = pid_alive or KeepaliveStore._pid_alive

    def _path(self, gpu_uuid: str) -> Path:
        safe = KeepaliveStore._safe_gpu_name(gpu_uuid)
        # Do not use a .json suffix: KeepaliveStore enumerates those as owner
        # records and must not confuse the common mutex with lifecycle state.
        return resolve_within_root(self.root / f"{safe}.role", self.root, must_exist=False)

    @staticmethod
    def _safe_lease_id(lease_id: str) -> str:
        if (
            not isinstance(lease_id, str)
            or not lease_id
            or len(lease_id) > 128
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in lease_id)
        ):
            raise GPUOwnershipError("GPU role lease ID is invalid")
        return lease_id

    def _token_path(self, gpu_uuid: str, lease_id: str) -> Path:
        directory = self._path(gpu_uuid)
        token = self._safe_lease_id(lease_id)
        return resolve_within_root(
            directory / f"{token}.owner.json", self.root, must_exist=False
        )

    def acquire_role(
        self,
        device: GPUDevice,
        *,
        role: str,
        pid: int,
        lease_id: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, object]:
        """Atomically claim ``device`` before any role-specific probe/work."""

        if role not in {"keepalive", "finetune", "eval"}:
            raise GPUOwnershipError("GPU role must be keepalive, finetune, or eval")
        if not self.pid_alive(pid):
            raise GPUOwnershipError("GPU role mutex requires a live project PID")
        identifier = lease_id or os.urandom(16).hex()
        identifier = self._safe_lease_id(identifier)
        detail = purpose or role
        if not detail or any(ord(char) < 32 for char in detail):
            raise GPUOwnershipError("GPU role purpose is invalid")
        body: dict[str, object] = {
            "schema_version": "2.0",
            "lease_id": identifier,
            "gpu_uuid": device.uuid,
            "gpu_index": device.index,
            "pid": pid,
            "owner": self.project_owner,
            "role": role,
            "purpose": detail,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        record = {**body, "record_sha256": sha256_json(body)}
        directory = self._path(device.uuid)
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise GPUOwnershipError(
                f"GPU {device.uuid} already has an active project role mutex"
            ) from exc
        try:
            atomic_write_json(
                self._token_path(device.uuid, identifier),
                record,
                root=self.root,
                overwrite=False,
            )
        except BaseException as acquire_error:
            # A create-if-absent publisher can report an error after the hard
            # link became visible (for example, if temporary-file cleanup was
            # interrupted).  Reap that token only when its complete contents
            # are exactly the capability produced by this invocation.
            cleanup_error: BaseException | None = None
            try:
                token = self._token_path(device.uuid, identifier)
                if token.exists():
                    published = load_json_strict(token)
                    if published != record:
                        raise GPUOwnershipError(
                            "GPU role token changed during failed acquisition"
                        )
                    token.unlink()
                directory.rmdir()
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                raise GPUOwnershipError(
                    "GPU role acquisition failed and its token could not be rolled back"
                ) from acquire_error
            raise
        return record

    def acquire(self, gpu_uuid: str, *, pid: int, purpose: str) -> dict[str, object]:
        inventory = self.probe.query()
        device = inventory.device(gpu_uuid)
        if not inventory.is_idle(device):
            raise GPUQueryError("GPU is not proven idle; lease acquisition refused")
        role = "eval" if purpose.startswith("eval") else "finetune"
        return self.acquire_role(device, role=role, pid=pid, purpose=purpose)

    def load(self, gpu_uuid: str) -> dict[str, object]:
        directory = self._path(gpu_uuid)
        if not directory.is_dir():
            raise GPUOwnershipError("GPU role mutex is missing or not a directory")
        entries = list(directory.iterdir())
        if len(entries) != 1 or not entries[0].is_file() or not entries[0].name.endswith(".owner.json"):
            raise GPUOwnershipError("GPU role mutex has invalid ownership evidence")
        value = load_json_strict(entries[0])
        if not isinstance(value, Mapping):
            raise GPUOwnershipError("invalid GPU lease")
        record = dict(value)
        body = {key: item for key, item in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != sha256_json(body):
            raise GPUOwnershipError("tampered GPU lease")
        required = {
            "schema_version", "lease_id", "gpu_uuid", "gpu_index", "pid",
            "owner", "role", "purpose", "acquired_at", "record_sha256",
        }
        if (
            set(record) != required
            or record.get("schema_version") != "2.0"
            or record.get("gpu_uuid") != gpu_uuid
            or record.get("role") not in {"keepalive", "finetune", "eval"}
            or not isinstance(record.get("gpu_index"), int)
            or not isinstance(record.get("pid"), int)
            or not isinstance(record.get("lease_id"), str)
        ):
            raise GPUOwnershipError("invalid GPU role mutex")
        if entries[0] != self._token_path(gpu_uuid, str(record["lease_id"])):
            raise GPUOwnershipError("GPU role mutex token/lease mismatch")
        return record

    def assert_owned(self, gpu_uuid: str, *, lease_id: str, role: str) -> dict[str, object]:
        record = self.load(gpu_uuid)
        if (
            record.get("owner") != self.project_owner
            or record.get("lease_id") != lease_id
            or record.get("role") != role
        ):
            raise GPUOwnershipError("GPU role mutex ownership mismatch")
        return record

    def heartbeat(self, gpu_uuid: str, *, lease_id: str, pid: int) -> dict[str, object]:
        record = self.load(gpu_uuid)
        if (
            record.get("owner") != self.project_owner
            or record.get("lease_id") != lease_id
            or record.get("pid") != pid
            or not self.pid_alive(pid)
        ):
            raise GPUOwnershipError("GPU lease ownership/liveness mismatch")
        # The role mutex is immutable.  Liveness belongs to the keepalive
        # lifecycle or the synchronous controller scope, not this ownership
        # capability.
        return record

    def release(
        self,
        gpu_uuid: str,
        *,
        lease_id: str,
        pid: int | None = None,
        role: str | None = None,
    ) -> None:
        record = self.load(gpu_uuid)
        if (
            record.get("owner") != self.project_owner
            or record.get("lease_id") != lease_id
            or (pid is not None and record.get("pid") != pid)
            or (role is not None and record.get("role") != role)
        ):
            raise GPUOwnershipError("refusing to release another owner/process lease")
        token = self._token_path(gpu_uuid, lease_id)
        try:
            token.unlink()
        except FileNotFoundError as exc:
            raise GPUOwnershipError("GPU role mutex changed before release") from exc
        try:
            self._path(gpu_uuid).rmdir()
        except OSError as exc:
            # Do not remove any unexpected/new owner's evidence.  An orphaned
            # directory intentionally blocks future acquisition fail-closed.
            raise GPUOwnershipError("GPU role mutex could not be reaped safely") from exc
