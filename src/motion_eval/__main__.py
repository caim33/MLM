"""Thin command-line interface for the framework-independent controller."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from motion_eval.controller import BatchController, load_canonical_registry
from motion_eval.data import REQUIRED_INPUT_ROLES, load_json_strict
from motion_eval.runtime import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_OWNER,
    DEFAULT_STALE_AFTER_SECONDS,
    DEFAULT_START_TIMEOUT_SECONDS,
    DEFAULT_STOP_TIMEOUT_SECONDS,
    KEEPALIVE_WORKER_MODULE,
    KeepaliveController,
    NvidiaSmiProbe,
)


_KEEPALIVE_PUBLIC_RECORD_FIELDS = (
    "schema_version",
    "gpu_uuid",
    "gpu_index",
    "pid",
    "owner",
    "state",
    "started_at",
    "heartbeat_at",
    "worker_module",
    "worker_code_sha256",
    "command_fingerprint",
    "reservation_id",
    "binding_sha256",
    "record_sha256",
    "alive",
    "gpu_process_proven",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _defaults() -> tuple[Path, Path, Path, Path]:
    repo = _repository_root()
    agent = repo / "model_evaluation_agent"
    return (
        agent / "batches",
        agent / "model_registry.json",
        agent / "pretrained_registry.json",
        repo / "src" / "motion_eval",
    )


def _add_controller_options(parser: argparse.ArgumentParser) -> None:
    batches, registry, pretrained, code = _defaults()
    parser.add_argument("--workspace-root", default=str(batches), help="directory containing batch folders")
    parser.add_argument("--registry", default=str(registry), help="canonical model_registry.json")
    parser.add_argument(
        "--pretrained-registry", default=str(pretrained), help="canonical pretrained_registry.json"
    )
    parser.add_argument("--code-root", default=str(code), help="code tree bound into batch receipts")
    parser.add_argument("--runner-root", help="runtime controller root frozen into a new batch")
    parser.add_argument("--pretrained-root", help="runtime pretrained root frozen into a new batch")
    parser.add_argument(
        "--controller-interpreter",
        help="absolute interpreter to freeze at batch creation (defaults to this process)",
    )
    parser.add_argument("--keepalive-root", help="dedicated project keepalive evidence root")
    parser.add_argument("--keepalive-owner", default="motionllm")


def _controller(args: argparse.Namespace) -> BatchController:
    return BatchController(
        args.workspace_root,
        registry_path=args.registry,
        pretrained_registry_path=args.pretrained_registry,
        code_root=args.code_root,
        runner_root=args.runner_root,
        pretrained_root=args.pretrained_root,
        controller_interpreter=args.controller_interpreter,
        keepalive_root=args.keepalive_root,
        keepalive_owner=args.keepalive_owner,
    )


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parse_inputs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must use ROLE=PATH")
        role, path = value.split("=", 1)
        if not role or not path or role in result:
            raise ValueError("--input roles/paths must be non-empty and unique")
        result[role] = path
    if set(result) != REQUIRED_INPUT_ROLES:
        raise ValueError(
            f"--input roles must be exactly {sorted(REQUIRED_INPUT_ROLES)}"
        )
    return result


def _nonempty_text(value: str) -> str:
    if not value.strip() or any(ord(char) < 32 for char in value):
        raise argparse.ArgumentTypeError("value must be non-empty and control-free")
    return value


def _gpu_index(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU index must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("GPU index must be non-negative")
    return parsed


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return parsed


def _add_keepalive_identity_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        required=True,
        type=_nonempty_text,
        help="dedicated project keepalive evidence and selected-GPU role-mutex root",
    )
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        type=_nonempty_text,
        help=f"project ownership namespace (default: {DEFAULT_OWNER})",
    )


def _add_keepalive_status_options(parser: argparse.ArgumentParser) -> None:
    _add_keepalive_identity_options(parser)
    parser.add_argument(
        "--stale-after-seconds",
        type=_positive_seconds,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help=(
            "maximum heartbeat age while proving PID, nvidia-smi CUDA process, "
            f"and selected-GPU role mutex (default: {DEFAULT_STALE_AFTER_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the read-only proof plan; do not create directories or acquire a mutex",
    )


def _add_keepalive_commands(parent: argparse.ArgumentParser) -> None:
    commands = parent.add_subparsers(dest="keepalive_command", required=True)

    start = commands.add_parser(
        "start",
        help="start on one proven-idle real CUDA GPU under the shared role mutex",
        description=(
            "Start the project CUDA keepalive on exactly one selected GPU. The same "
            "per-GPU role mutex automatically blocks formal finetune/eval launch until "
            "the keepalive is stopped. Real nvidia-smi and CUDA worker proof are mandatory."
        ),
    )
    _add_keepalive_identity_options(start)
    selector = start.add_mutually_exclusive_group(required=True)
    selector.add_argument("--gpu-uuid", type=_nonempty_text, help="stable nvidia-smi GPU UUID")
    selector.add_argument("--gpu-index", type=_gpu_index, help="nvidia-smi GPU index")
    start.add_argument(
        "--heartbeat-interval-seconds",
        type=_positive_seconds,
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        help=f"worker heartbeat interval (default: {DEFAULT_HEARTBEAT_INTERVAL_SECONDS:g})",
    )
    start.add_argument(
        "--ready-timeout-seconds",
        type=_positive_seconds,
        default=DEFAULT_START_TIMEOUT_SECONDS,
        help=f"child ready-handshake timeout (default: {DEFAULT_START_TIMEOUT_SECONDS:g})",
    )
    start.add_argument(
        "--wait-timeout-seconds",
        type=_positive_seconds,
        default=DEFAULT_START_TIMEOUT_SECONDS,
        help=f"controller wait for first proven heartbeat (default: {DEFAULT_START_TIMEOUT_SECONDS:g})",
    )
    start.add_argument(
        "--dry-run",
        action="store_true",
        help="show queries and handshake; do not write, spawn, or acquire a mutex",
    )

    status = commands.add_parser(
        "status",
        help="verify lifecycle, heartbeat, CUDA process, and selected-GPU role mutex",
        description=(
            "Fail closed unless every project keepalive record has fresh heartbeat, "
            "real nvidia-smi process proof, and the matching selected-GPU role mutex."
        ),
    )
    _add_keepalive_status_options(status)

    stop = commands.add_parser(
        "stop",
        help="stop one UUID-bound keepalive and release its selected-GPU role mutex",
        description=(
            "Stop only the hash-bound project worker for the exact GPU UUID. The role "
            "mutex is released only after the worker exit and lifecycle cleanup are proven."
        ),
    )
    _add_keepalive_identity_options(stop)
    stop.add_argument("--gpu-uuid", required=True, type=_nonempty_text)
    stop.add_argument(
        "--wait-timeout-seconds",
        type=_positive_seconds,
        default=DEFAULT_STOP_TIMEOUT_SECONDS,
        help=f"bounded graceful/forced stop timeout (default: {DEFAULT_STOP_TIMEOUT_SECONDS:g})",
    )
    stop.add_argument(
        "--dry-run",
        action="store_true",
        help="show the stop target; do not write, signal, or release a mutex",
    )


def _public_keepalive_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist CLI output so process environments and secrets cannot leak."""

    if not isinstance(record, Mapping):
        raise ValueError("keepalive controller returned an invalid record")
    return {
        key: record[key]
        for key in _KEEPALIVE_PUBLIC_RECORD_FIELDS
        if key in record
    }


def _keepalive_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dry_run": True,
        "action": args.keepalive_command,
        "root": str(Path(args.root).resolve(strict=False)),
        "owner": args.owner,
        "selected_gpu_role_mutex": True,
        "automatic_finetune_eval_exclusion": True,
        "real_cuda_remote_gate_required": True,
    }
    if args.keepalive_command == "start":
        selector_kind = "gpu_uuid" if args.gpu_uuid is not None else "gpu_index"
        base.update(
            {
                selector_kind: args.gpu_uuid if args.gpu_uuid is not None else args.gpu_index,
                "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
                "ready_timeout_seconds": args.ready_timeout_seconds,
                "wait_timeout_seconds": args.wait_timeout_seconds,
                "worker_module": KEEPALIVE_WORKER_MODULE,
                "protocol": [
                    "query_nvidia_smi",
                    "acquire_selected_gpu_role_mutex",
                    "prove_idle",
                    "reserve_lifecycle",
                    "requery_idle",
                    "spawn_uuid_isolated_cuda_worker",
                    "verify_heartbeat_and_gpu_process",
                ],
                "queries": NvidiaSmiProbe().preview(),
            }
        )
    elif args.keepalive_command == "status":
        base.update(
            {
                "stale_after_seconds": args.stale_after_seconds,
                "proofs": [
                    "hashed_lifecycle",
                    "fresh_heartbeat",
                    "nvidia_smi_gpu_process",
                    "selected_gpu_role_mutex",
                ],
            }
        )
    elif args.keepalive_command == "stop":
        base.update(
            {
                "gpu_uuid": args.gpu_uuid,
                "wait_timeout_seconds": args.wait_timeout_seconds,
                "protocol": [
                    "prove_hash_bound_worker",
                    "request_stop",
                    "bounded_terminate",
                    "prove_exit_and_cleanup",
                    "release_selected_gpu_role_mutex",
                ],
            }
        )
    else:  # pragma: no cover - argparse constrains this field
        raise RuntimeError("unknown keepalive command")
    return base


def _run_keepalive_command(args: argparse.Namespace, *, deprecated: bool = False) -> None:
    if deprecated:
        print(
            "motion_eval: warning: `keepalive status` is deprecated; use `gpu keepalive status`",
            file=sys.stderr,
        )
    if args.dry_run:
        output = _keepalive_dry_run(args)
    else:
        controller = KeepaliveController(args.root, project_owner=args.owner)
        if args.keepalive_command == "start":
            selector = args.gpu_uuid if args.gpu_uuid is not None else args.gpu_index
            record = controller.start(
                selector,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                ready_timeout_seconds=args.ready_timeout_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
            )
            output = {"action": "start", "record": _public_keepalive_record(record)}
        elif args.keepalive_command == "status":
            records = controller.status(stale_after_seconds=args.stale_after_seconds)
            output = {
                "action": "status",
                "records": [_public_keepalive_record(record) for record in records],
            }
        elif args.keepalive_command == "stop":
            receipt = controller.stop(
                args.gpu_uuid,
                wait_timeout_seconds=args.wait_timeout_seconds,
            )
            output = {
                "action": "stop",
                "result": {
                    key: receipt[key]
                    for key in ("stopped", "gpu_uuid", "gpu_index", "pid", "owner")
                    if key in receipt
                },
            }
        else:  # pragma: no cover - argparse constrains this field
            raise RuntimeError("unknown keepalive command")
    if deprecated:
        output = {
            **output,
            "deprecated": True,
            "replacement": "gpu keepalive status",
        }
    _json(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m motion_eval",
        description="Fresh-finetune-first controller for all 15 MotionLLM evaluation models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("registry", help="validate canonical registries")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_validate = registry_sub.add_parser("validate", help="validate the exact 15-model matrix")
    _add_controller_options(registry_validate)
    registry_validate.add_argument("--dry-run", action="store_true", help="read-only; retained for scripting symmetry")

    batch = commands.add_parser("batch", help="create or validate immutable batches")
    batch_sub = batch.add_subparsers(dest="batch_command", required=True)
    batch_create = batch_sub.add_parser("create", help="freeze inputs and create a batch")
    _add_controller_options(batch_create)
    batch_create.add_argument("batch_id")
    batch_create.add_argument(
        "--input", action="append", default=[], metavar="ROLE=PATH", help="repeat for all six frozen input roles"
    )
    batch_create.add_argument("--config", help="finite JSON controller/training configuration")
    batch_create.add_argument("--description", default="")
    batch_create.add_argument("--dry-run", action="store_true", help="validate and hash without writing")
    batch_validate = batch_sub.add_parser("validate", help="recompute all hashes and evidence")
    _add_controller_options(batch_validate)
    batch_validate.add_argument("batch_id")
    batch_validate.add_argument("--dry-run", action="store_true", help="validation is always read-only")

    plan = commands.add_parser("plan", help="show 15 typed finetune/eval CommandSpecs")
    _add_controller_options(plan)
    plan.add_argument("batch_id")
    plan.add_argument("--python-executable")
    plan.add_argument("--controller-root")
    plan.add_argument("--dry-run", action="store_true", help="plan is always read-only")

    finetune = commands.add_parser("finetune", help="fresh finetune evidence and global barrier")
    finetune_sub = finetune.add_subparsers(dest="finetune_command", required=True)
    ft_barrier = finetune_sub.add_parser("barrier", help="show global terminal barrier status")
    _add_controller_options(ft_barrier)
    ft_barrier.add_argument("batch_id")
    ft_barrier.add_argument("--dry-run", action="store_true")
    ft_attempt = finetune_sub.add_parser("attempt", help="create an append-only typed finetune attempt")
    _add_controller_options(ft_attempt)
    ft_attempt.add_argument("batch_id")
    ft_attempt.add_argument("--model-id", required=True)
    ft_attempt.add_argument("--attempt-id")
    ft_attempt.add_argument("--python-executable")
    ft_attempt.add_argument("--controller-root")
    ft_attempt.add_argument("--gpu", required=True, help="GPU UUID or index resolved by nvidia-smi")
    ft_attempt.add_argument("--limit", type=int)
    ft_attempt.add_argument(
        "--purpose", choices=("production", "preflight"), default="production"
    )
    ft_attempt.add_argument("--dry-run", action="store_true")
    ft_run = finetune_sub.add_parser("run-attempt", help="execute frozen argv and record process-exit evidence")
    _add_controller_options(ft_run)
    ft_run.add_argument("batch_id")
    ft_run.add_argument("--model-id", required=True)
    ft_run.add_argument("--attempt-id", required=True)
    ft_run.add_argument("--dry-run", action="store_true")
    ft_complete = finetune_sub.add_parser("complete", help="validate a fresh attempt artifact")
    _add_controller_options(ft_complete)
    ft_complete.add_argument("batch_id")
    ft_complete.add_argument("--model-id", required=True)
    ft_complete.add_argument("--attempt-id", required=True)
    ft_complete.add_argument("--manifest", required=True)
    ft_complete.add_argument("--dry-run", action="store_true")
    ft_block = finetune_sub.add_parser("block", help="record a structured evidence-backed blocker")
    _add_controller_options(ft_block)
    ft_block.add_argument("batch_id")
    ft_block.add_argument("--model-id", required=True)
    ft_block.add_argument("--reason", required=True)
    ft_block.add_argument("--component", required=True)
    ft_block.add_argument("--detail", required=True)
    ft_block.add_argument("--attempt-id")
    ft_block.add_argument("--dry-run", action="store_true")
    ft_open = finetune_sub.add_parser("open-eval", help="open eval only after all 15 terminal")
    _add_controller_options(ft_open)
    ft_open.add_argument("batch_id")
    ft_open.add_argument("--dry-run", action="store_true")

    gate = commands.add_parser("gate", help="open global phase gates")
    gate_sub = gate.add_subparsers(dest="gate_command", required=True)
    for name, help_text in (
        ("open-eval", "open evaluation after the global finetune barrier"),
        ("open-full", "open full-500 after every evaluable model passes 1/8/32"),
    ):
        child = gate_sub.add_parser(name, help=help_text)
        _add_controller_options(child)
        child.add_argument("batch_id")
        child.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("eval", help="record smoke/full prediction evidence")
    eval_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    smoke = eval_sub.add_parser("smoke", help="validate one 1/8/32 prediction attempt")
    _add_controller_options(smoke)
    smoke.add_argument("batch_id")
    smoke.add_argument("--model-id", required=True)
    smoke.add_argument("--size", type=int, choices=(1, 8, 32), required=True)
    smoke.add_argument("--attempt-id", required=True)
    smoke.add_argument("--predictions", required=True)
    smoke.add_argument("--dry-run", action="store_true")
    eval_attempt = eval_sub.add_parser("attempt", help="create an append-only typed smoke/full attempt")
    _add_controller_options(eval_attempt)
    eval_attempt.add_argument("batch_id")
    eval_attempt.add_argument("--model-id", required=True)
    eval_attempt.add_argument("--stage", choices=("smoke_1", "smoke_8", "smoke_32", "full"), required=True)
    eval_attempt.add_argument("--attempt-id")
    eval_attempt.add_argument("--python-executable")
    eval_attempt.add_argument("--controller-root")
    eval_attempt.add_argument("--gpu", required=True, help="GPU UUID or index resolved by nvidia-smi")
    eval_attempt.add_argument("--dry-run", action="store_true")
    eval_run = eval_sub.add_parser("run-attempt", help="execute a frozen eval argv and attest predictions")
    _add_controller_options(eval_run)
    eval_run.add_argument("batch_id")
    eval_run.add_argument("--model-id", required=True)
    eval_run.add_argument("--stage", choices=("smoke_1", "smoke_8", "smoke_32", "full"), required=True)
    eval_run.add_argument("--attempt-id", required=True)
    eval_run.add_argument("--dry-run", action="store_true")
    eval_open = eval_sub.add_parser("open-full", help="alias for gate open-full")
    _add_controller_options(eval_open)
    eval_open.add_argument("batch_id")
    eval_open.add_argument("--dry-run", action="store_true")
    full = eval_sub.add_parser("full", help="validate one fixed-500 prediction attempt")
    _add_controller_options(full)
    full.add_argument("batch_id")
    full.add_argument("--model-id", required=True)
    full.add_argument("--attempt-id", required=True)
    full.add_argument("--predictions", required=True)
    full.add_argument("--dry-run", action="store_true")

    release = commands.add_parser("release", help="build or re-verify the current-batch release")
    release_sub = release.add_subparsers(dest="release_command", required=True)
    for name in ("build", "verify"):
        child = release_sub.add_parser(name, help=f"{name} the release manifest and tables")
        _add_controller_options(child)
        child.add_argument("batch_id")
        child.add_argument("--dry-run", action="store_true")

    gpu = commands.add_parser(
        "gpu",
        help="fail-closed GPU inventory and selected-GPU role-mutex operations",
        description=(
            "Inspect real nvidia-smi state or manage the project CUDA keepalive. "
            "Keepalive and formal finetune/eval workers share one selected-GPU role "
            "mutex, so training launch is automatically excluded while it is active."
        ),
    )
    gpu_sub = gpu.add_subparsers(dest="gpu_command", required=True)
    gpu_status = gpu_sub.add_parser(
        "status",
        help="query real nvidia-smi state without assuming failure means idle",
    )
    gpu_status.add_argument("--dry-run", action="store_true", help="print exact read-only queries")

    gpu_keepalive = gpu_sub.add_parser(
        "keepalive",
        help="manage the project CUDA keepalive under the shared selected-GPU role mutex",
    )
    _add_keepalive_commands(gpu_keepalive)

    keepalive = commands.add_parser(
        "keepalive",
        help="DEPRECATED compatibility alias; use `gpu keepalive status`",
        description=(
            "DEPRECATED compatibility path. Use `gpu keepalive status`; the new path "
            "also documents the selected-GPU role mutex and real CUDA proof gate."
        ),
    )
    keepalive_sub = keepalive.add_subparsers(dest="keepalive_command", required=True)
    keepalive_status = keepalive_sub.add_parser(
        "status",
        help="DEPRECATED alias for `gpu keepalive status`",
    )
    _add_keepalive_status_options(keepalive_status)
    return parser


def _dry_gate(controller: BatchController, batch_id: str, gate: str) -> dict[str, Any]:
    state = controller.validate_batch(batch_id)
    eligible = False
    if gate == "eval":
        eligible = all(
            item["finetune_status"] in {"finetune_complete", "blocked"}
            for item in state["models"].values()
        ) and not state["eval_open"]
    elif gate == "full":
        eligible = state["eval_open"] and not state["full_open"] and all(
            item["finetune_status"] != "finetune_complete"
            or all(value == "passed" for value in item["smoke"].values())
            for item in state["models"].values()
        )
    return {"dry_run": True, "gate": gate, "eligible": eligible, "revision": state["revision"]}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "registry":
            registry = load_canonical_registry(args.registry, args.pretrained_registry)
            _json(
                {
                    "valid": True,
                    "model_count": len(registry.models),
                    "model_ids": list(registry.ids),
                    "dry_run": args.dry_run,
                }
            )
            return 0
        if args.command == "gpu":
            if args.gpu_command == "keepalive":
                _run_keepalive_command(args)
                return 0
            probe = NvidiaSmiProbe()
            if args.dry_run:
                _json({"dry_run": True, **probe.preview()})
            else:
                inventory = probe.query()
                _json(
                    {
                        "known": True,
                        "devices": [device.__dict__ for device in inventory.devices],
                        "processes": [process.__dict__ for process in inventory.processes],
                    }
                )
            return 0
        if args.command == "keepalive":
            _run_keepalive_command(args, deprecated=True)
            return 0

        controller = _controller(args)
        if args.command == "batch" and args.batch_command == "create":
            inputs = _parse_inputs(args.input)
            config = {} if args.config is None else load_json_strict(args.config)
            if args.dry_run:
                with tempfile.TemporaryDirectory(prefix="motion_eval_batch_dryrun_") as temp:
                    preview = BatchController(
                        Path(temp) / "batches",
                        registry_path=args.registry,
                        pretrained_registry_path=args.pretrained_registry,
                        code_root=args.code_root,
                        runner_root=args.runner_root,
                        pretrained_root=args.pretrained_root,
                        controller_interpreter=args.controller_interpreter,
                        keepalive_root=args.keepalive_root,
                        keepalive_owner=args.keepalive_owner,
                    )
                    receipt = preview.create_batch(
                        args.batch_id,
                        inputs=inputs,
                        config=config,
                        description=args.description,
                    )
                    _json(
                        {
                            "dry_run": True,
                            "batch_id": args.batch_id,
                            "inputs_valid": True,
                            "receipt_sha256": receipt["receipt_sha256"],
                        }
                    )
            else:
                _json(
                    controller.create_batch(
                        args.batch_id,
                        inputs=inputs,
                        config=config,
                        description=args.description,
                    )
                )
        elif args.command == "batch":
            _json({"valid": True, "state": controller.validate_batch(args.batch_id)})
        elif args.command == "plan":
            _json(
                controller.plan(
                    args.batch_id,
                    python_executable=args.python_executable,
                    controller_root=args.controller_root,
                )
            )
        elif args.command == "finetune" and args.finetune_command == "barrier":
            _json(controller.barrier_status(args.batch_id))
        elif args.command == "finetune" and args.finetune_command == "attempt":
            if args.dry_run:
                _json(controller.preview_finetune_attempt(
                    args.batch_id,
                    model_id=args.model_id,
                    attempt_id=args.attempt_id or "dryrun",
                    python_executable=args.python_executable,
                    controller_root=args.controller_root,
                    limit=args.limit,
                    purpose=args.purpose,
                    gpu=args.gpu,
                ))
            else:
                _json(
                    controller.create_finetune_attempt(
                        args.batch_id,
                        model_id=args.model_id,
                        attempt_id=args.attempt_id,
                        python_executable=args.python_executable,
                        controller_root=args.controller_root,
                        limit=args.limit,
                        purpose=args.purpose,
                        gpu=args.gpu,
                    )
                )
        elif args.command == "finetune" and args.finetune_command == "run-attempt":
            _json(
                controller.execute_frozen_attempt(
                    args.batch_id,
                    model_id=args.model_id,
                    stage="finetune",
                    attempt_id=args.attempt_id,
                    dry_run=args.dry_run,
                )
            )
        elif args.command == "finetune" and args.finetune_command == "complete":
            if args.dry_run:
                _json({"dry_run": True, "manifest": str(Path(args.manifest).resolve(strict=False))})
            else:
                _json(
                    controller.complete_finetune(
                        args.batch_id,
                        model_id=args.model_id,
                        attempt_id=args.attempt_id,
                        run_manifest_path=args.manifest,
                    )
                )
        elif args.command == "finetune" and args.finetune_command == "block":
            if args.dry_run:
                _json({"dry_run": True, "model_id": args.model_id, "reason": args.reason, "component": args.component})
            else:
                _json(
                    controller.block_finetune(
                        args.batch_id,
                        model_id=args.model_id,
                        reason_code=args.reason,
                        component=args.component,
                        detail=args.detail,
                        attempt_id=args.attempt_id,
                    )
                )
        elif (
            args.command == "finetune" and args.finetune_command == "open-eval"
        ) or (args.command == "gate" and args.gate_command == "open-eval"):
            _json(_dry_gate(controller, args.batch_id, "eval") if args.dry_run else controller.open_evaluation(args.batch_id))
        elif (args.command == "gate" and args.gate_command == "open-full") or (
            args.command == "eval" and args.eval_command == "open-full"
        ):
            _json(_dry_gate(controller, args.batch_id, "full") if args.dry_run else controller.open_full_evaluation(args.batch_id))
        elif args.command == "eval" and args.eval_command in {"smoke", "full"}:
            stage = f"smoke_{args.size}" if args.eval_command == "smoke" else "full"
            if args.dry_run:
                _json({"dry_run": True, "stage": stage, "predictions": str(Path(args.predictions).resolve(strict=False))})
            else:
                _json(
                    controller.complete_evaluation(
                        args.batch_id,
                        model_id=args.model_id,
                        stage=stage,
                        attempt_id=args.attempt_id,
                        predictions_path=args.predictions,
                    )
                )
        elif args.command == "eval" and args.eval_command == "attempt":
            if args.dry_run:
                _json(controller.preview_evaluation_attempt(
                    args.batch_id,
                    model_id=args.model_id,
                    stage=args.stage,
                    attempt_id=args.attempt_id or "dryrun",
                    python_executable=args.python_executable,
                    controller_root=args.controller_root,
                    gpu=args.gpu,
                ))
            else:
                _json(
                    controller.create_evaluation_attempt(
                        args.batch_id,
                        model_id=args.model_id,
                        stage=args.stage,
                        attempt_id=args.attempt_id,
                        python_executable=args.python_executable,
                        controller_root=args.controller_root,
                        gpu=args.gpu,
                    )
                )
        elif args.command == "eval" and args.eval_command == "run-attempt":
            _json(
                controller.execute_frozen_attempt(
                    args.batch_id,
                    model_id=args.model_id,
                    stage=args.stage,
                    attempt_id=args.attempt_id,
                    dry_run=args.dry_run,
                )
            )
        elif args.command == "release" and args.release_command == "build":
            if args.dry_run:
                state = controller.validate_batch(args.batch_id)
                _json({"dry_run": True, "eligible": state["full_open"] and state["release_status"] == "pending"})
            else:
                _json(controller.build_release(args.batch_id))
        elif args.command == "release":
            _json(controller.verify_release(args.batch_id))
        else:  # pragma: no cover - argparse and branches cover every command
            raise RuntimeError("unhandled command")
        return 0
    except Exception as exc:
        print(f"motion_eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
