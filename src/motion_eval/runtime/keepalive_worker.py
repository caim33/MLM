"""CUDA worker for :mod:`motion_eval.runtime.keepalive`.

The process deliberately waits for the controller's hash-bound ready signal
before importing or initializing CUDA.  Any missing CUDA support, UUID
mismatch, ownership mismatch, or failed heartbeat exits instead of sleeping.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Sequence

from motion_eval.core import sha256_file

from .gpu import GPUOwnershipError, KeepaliveStore, command_fingerprint

WORKER_MODULE = "motion_eval.runtime.keepalive_worker"
_POLL_SECONDS = 0.1
_TENSOR_ELEMENTS = 262_144  # 1 MiB float32 plus the CUDA context.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {WORKER_MODULE}",
        description="Internal project-owned CUDA keepalive worker.",
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--reservation-id", required=True)
    parser.add_argument("--heartbeat-interval-seconds", type=float, required=True)
    parser.add_argument("--ready-timeout-seconds", type=float, required=True)
    return parser


def _worker_hash() -> str:
    return sha256_file(Path(__file__).resolve(strict=True))


def _reconstructed_command(argv: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, "-m", WORKER_MODULE, *argv)


def _normalized_uuid(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return "".join(char for char in text if char.isalnum())


def _torch_uuid(
    torch_module: ModuleType,
    logical_index: int,
    *,
    visible_devices: str | None = None,
) -> str:
    properties = torch_module.cuda.get_device_properties(logical_index)
    value = getattr(properties, "uuid", None)
    if value is None:
        # PyTorch 2.4 CUDA device properties do not expose ``uuid`` on every
        # build.  The controller isolates the child with one exact GPU UUID;
        # accept that binding only when the environment still contains exactly
        # that one UUID.  Controller status then independently proves the PID
        # appears on the same nvidia-smi UUID before publishing success.
        value = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices is None
            else visible_devices
        )
        if value is None or "," in value or not value.strip().lower().startswith("gpu-"):
            raise RuntimeError(
                "CUDA device UUID is unavailable and the child is not UUID-isolated"
            )
    normalized = _normalized_uuid(value)
    if not normalized:
        raise RuntimeError("CUDA device UUID is empty")
    return normalized


def run_worker(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
    torch_module: ModuleType | None = None,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> int:
    if (
        args.gpu_index < 0
        or args.heartbeat_interval_seconds <= 0
        or args.ready_timeout_seconds <= 0
    ):
        raise ValueError("GPU index, heartbeat interval, and ready timeout are invalid")
    pid = os.getpid()
    fingerprint = command_fingerprint(_reconstructed_command(raw_argv))
    code_hash = _worker_hash()
    store = KeepaliveStore(
        args.root,
        project_owner=args.owner,
        pid_alive=lambda candidate: candidate == pid,
        process_fingerprint=lambda candidate: fingerprint if candidate == pid else None,
    )

    deadline = monotonic() + args.ready_timeout_seconds
    record: dict[str, object] | None = None
    while monotonic() < deadline:
        if store.ready_exists(args.gpu_uuid):
            record = store.validate_ready(
                args.gpu_uuid,
                pid=pid,
                reservation_id=args.reservation_id,
                command_fingerprint=fingerprint,
                worker_module=WORKER_MODULE,
                worker_code_sha256=code_hash,
            )
            break
        sleep(min(_POLL_SECONDS, max(0.0, deadline - monotonic())))
    if record is None:
        raise GPUOwnershipError("controller ready signal was not published before timeout")
    if record.get("gpu_index") != args.gpu_index:
        raise GPUOwnershipError("worker GPU index does not match controller record")

    # Import CUDA only after the owner record and ready signal are both proven.
    if torch_module is None:
        try:
            import torch as imported_torch
        except Exception as exc:  # pragma: no cover - exercised on the GPU host
            raise RuntimeError("PyTorch CUDA runtime is unavailable") from exc
        torch_module = imported_torch
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable; keepalive exits without sleeping")
    if int(torch_module.cuda.device_count()) != 1:
        raise RuntimeError("keepalive requires exactly one UUID-isolated visible CUDA device")

    logical_index = 0
    expected_uuid = _normalized_uuid(args.gpu_uuid)
    actual_uuid = _torch_uuid(torch_module, logical_index)
    if actual_uuid != expected_uuid:
        raise RuntimeError("CUDA UUID mismatch; refusing to touch the selected device")
    torch_module.cuda.set_device(logical_index)
    device = torch_module.device("cuda:0")
    tensor = torch_module.ones(
        (_TENSOR_ELEMENTS,),
        device=device,
        dtype=torch_module.float32,
    )
    tensor.add_(1.0)
    torch_module.cuda.synchronize(device)
    record = store.heartbeat(
        args.gpu_uuid,
        pid=pid,
        reservation_id=args.reservation_id,
        command_fingerprint=fingerprint,
        worker_module=WORKER_MODULE,
        worker_code_sha256=code_hash,
    )

    stopping = threading.Event()

    def request_exit(_signum: int, _frame: object) -> None:
        stopping.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, request_exit)
        except (OSError, ValueError):  # pragma: no cover - non-main embedded worker
            pass

    next_heartbeat = monotonic() + args.heartbeat_interval_seconds
    while not stopping.is_set():
        if store.stop_requested(
            args.gpu_uuid,
            pid=pid,
            reservation_id=args.reservation_id,
            binding_sha256=str(record["binding_sha256"]),
        ):
            return 0
        now = monotonic()
        if now >= next_heartbeat:
            tensor.add_(1.0)
            torch_module.cuda.synchronize(device)
            record = store.heartbeat(
                args.gpu_uuid,
                pid=pid,
                reservation_id=args.reservation_id,
                command_fingerprint=fingerprint,
                worker_module=WORKER_MODULE,
                worker_code_sha256=code_hash,
            )
            next_heartbeat = now + args.heartbeat_interval_seconds
        sleep(min(_POLL_SECONDS, max(0.0, next_heartbeat - monotonic())))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(raw)
        return run_worker(args, raw_argv=raw)
    except Exception as exc:
        # The controller consumes only exit/liveness/GPU evidence.  Keep this
        # diagnostic terse and never include environment variables or argv.
        print(f"keepalive worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
