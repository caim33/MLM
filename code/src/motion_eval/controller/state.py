"""Append-only hash-chained controller events and materialized state cache."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from motion_eval.core import atomic_write_json, sha256_bytes, sha256_file, sha256_json
from motion_eval.data.jsonio import load_json_strict

STATE_SCHEMA_VERSION = "2.0"
EVENT_SCHEMA_VERSION = "2.0"
EVENT_TRUST_SCHEMA_VERSION = "2.0"
EVENT_ANCHOR_SCHEMA_VERSION = "1.0"
CONTROLLER_STATE_ROOT_ENV = "MOTION_EVAL_CONTROLLER_STATE_ROOT"
_ZERO_HASH = "0" * 64
_EVENT_TYPES = frozenset(
    {
        "GENESIS",
        "ATTEMPT_LEASED",
        "ATTEMPT_STARTED",
        "ATTEMPT_EXECUTED",
        "ATTEMPT_VERIFIED",
        "FINETUNE_COMPLETE",
        "FINETUNE_BLOCKED",
        "EVAL_OPENED",
        "SMOKE_PASSED",
        "FULL_OPENED",
        "FULL_COMPLETE",
        "RELEASE_BUILT",
    }
)


class StateError(RuntimeError):
    pass


class ConcurrentTransitionError(StateError):
    pass


def _event_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"event_sha256", "event_hmac_sha256"}
    }


def _state_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"state_sha256", "state_hmac_sha256"}
    }


def _default_controller_state_root() -> Path:
    configured = os.environ.get(CONTROLLER_STATE_ROOT_ENV)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise StateError(f"{CONTROLLER_STATE_ROOT_ENV} must be an absolute path")
        return candidate.resolve(strict=False)
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return (base / "MotionLLM" / "controller_state").resolve(strict=False)
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local" / "state"
    return (base / "motionllm" / "controller_state").resolve(strict=False)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows ACLs are not represented faithfully by chmod.  The public
        # trust proof therefore never claims same-principal protection.
        pass


def _event_trust_material(
    workspace_root: str | Path,
) -> tuple[dict[str, Any], bytes, Path]:
    """Return public proof, secret key bytes, and the external state namespace.

    The public proof is safe to freeze into receipts.  It deliberately omits
    the key path and key digest.  The namespace itself is outside the worker
    batch tree by default; a process that can act as the controller's OS
    principal and modify that namespace remains an explicit threat boundary.
    """

    workspace = Path(workspace_root).resolve(strict=True)
    controller_state_root = _default_controller_state_root()
    if _is_within(controller_state_root, workspace):
        raise StateError("controller state root must be outside the batch workspace")
    controller_state_root.mkdir(parents=True, exist_ok=True)
    controller_state_root = controller_state_root.resolve(strict=True)
    _chmod_best_effort(controller_state_root, 0o700)

    scope_id = sha256_json(
        {
            "schema_version": "1.0",
            "workspace_root": os.path.normcase(str(workspace)),
        }
    )
    # Keep Windows paths comfortably below legacy MAX_PATH while retaining a
    # 160-bit collision-resistant on-disk namespace.  The signed records carry
    # the complete 256-bit identities and reject any namespace collision.
    namespace_root = controller_state_root / "ws" / scope_id[:40]
    trust_root = namespace_root / "trust"
    anchors_root = namespace_root / "heads"
    trust_root.mkdir(parents=True, exist_ok=True)
    anchors_root.mkdir(parents=True, exist_ok=True)
    for directory in (namespace_root, trust_root, anchors_root):
        _chmod_best_effort(directory, 0o700)

    key_path = trust_root / "event_hmac.key"
    try:
        descriptor = os.open(
            key_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError:
        descriptor = None
    if descriptor is not None:
        try:
            os.write(descriptor, secrets.token_bytes(32))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _chmod_best_effort(key_path, 0o600)
    payload = key_path.read_bytes()
    if len(payload) != 32:
        raise StateError("controller event trust capability has an invalid length")
    key_id = sha256_bytes(b"motion-eval-event-key-id-v2\0" + payload)
    proof = {
        "schema_version": EVENT_TRUST_SCHEMA_VERSION,
        "key_id": f"sha256:{key_id}",
        "state_scope_id": scope_id,
        "storage_scope": "external_controller_state",
        "same_os_principal_protected": False,
        "protection_capability": (
            "HMAC-SHA256 authenticates controller events, cache, and the batch-external "
            "monotonic head; rollback confined to the batch/output tree is detected while "
            "external controller state remains intact"
        ),
        "threat_model": (
            "Protection depends on the batch-external controller state remaining outside an "
            "attacker's write authority. A worker or other process running as the same OS principal "
            "may be able to access the controller key/state; replay of both the batch and anchor "
            "by that principal is explicitly out of scope"
        ),
    }
    return proof, payload, namespace_root.resolve(strict=True)


def ensure_event_trust(workspace_root: str | Path) -> dict[str, Any]:
    """Create/load event trust and return only its non-sensitive public proof."""

    proof, _, _ = _event_trust_material(workspace_root)
    return proof


def _initial_state(batch_id: str, receipt_sha256: str, model_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "batch_id": batch_id,
        "batch_receipt_sha256": receipt_sha256,
        "revision": 0,
        "phase": "finetune",
        "eval_open": False,
        "full_open": False,
        "release_status": "pending",
        "attempts": {},
        "models": {
            model_id: {
                "finetune_status": "pending",
                "finetune_evidence": None,
                "smoke": {"1": "pending", "8": "pending", "32": "pending"},
                "smoke_evidence": {"1": None, "8": None, "32": None},
                "full_status": "pending",
                "full_evidence": None,
            }
            for model_id in model_ids
        },
        "last_event_sha256": _ZERO_HASH,
    }


def _require_model(state: Mapping[str, Any], model_id: Any) -> dict[str, Any]:
    models = state["models"]
    if model_id not in models:
        raise StateError(f"event references unknown model: {model_id!r}")
    return models[model_id]


def _attempt_key(model_id: Any, stage: Any, attempt_id: Any) -> str:
    if not all(isinstance(value, str) and value for value in (model_id, stage, attempt_id)):
        raise StateError("attempt identity must contain non-empty strings")
    return sha256_json([model_id, stage, attempt_id])


def _require_attempt(
    state: Mapping[str, Any], model_id: Any, stage: Any, attempt_id: Any
) -> dict[str, Any]:
    key = _attempt_key(model_id, stage, attempt_id)
    attempt = state["attempts"].get(key)
    if not isinstance(attempt, dict):
        raise StateError("attempt is not anchored in the controller event chain")
    if (attempt["model_id"], attempt["stage"], attempt["attempt_id"]) != (
        model_id,
        stage,
        attempt_id,
    ):
        raise StateError("attempt event identity collision")
    return attempt


def _require_stage_eligible(state: Mapping[str, Any], model_id: str, stage: str) -> None:
    model = _require_model(state, model_id)
    if stage == "finetune":
        if state["eval_open"] or model["finetune_status"] != "pending":
            raise StateError("finetune attempt is not eligible")
        return
    if stage not in {"smoke_1", "smoke_8", "smoke_32", "full"}:
        raise StateError(f"unsupported attempt stage: {stage}")
    if not state["eval_open"] or model["finetune_status"] != "finetune_complete":
        raise StateError("evaluation attempt is not eligible")
    if stage == "full":
        if not state["full_open"] or model["full_status"] != "pending":
            raise StateError("full evaluation attempt is not eligible")
        return
    if state["full_open"]:
        raise StateError("smoke phase is closed")
    size = stage.split("_", 1)[1]
    expected = next(
        (candidate for candidate in ("1", "8", "32") if model["smoke"][candidate] != "passed"),
        None,
    )
    if size != expected:
        raise StateError(f"smoke sequence violation; expected {expected}, got {size}")


def _apply_event(state: dict[str, Any] | None, event: Mapping[str, Any]) -> dict[str, Any]:
    event_type = event["event_type"]
    payload = event["payload"]
    if event_type == "GENESIS":
        if state is not None or event["sequence"] != 0:
            raise StateError("GENESIS must be the first and only genesis event")
        result = _initial_state(
            event["batch_id"], event["batch_receipt_sha256"], payload["model_ids"]
        )
    else:
        if state is None:
            raise StateError("event chain is missing GENESIS")
        result = json.loads(json.dumps(state))
        model_id = payload.get("model_id")
        if event_type == "ATTEMPT_LEASED":
            required = {
                "model_id", "stage", "attempt_id", "purpose", "lease_nonce",
                "command_sha256", "attempt_sha256", "attempt_reference", "leased_revision",
                "gpu_uuid", "gpu_index", "keepalive_root", "keepalive_owner",
            }
            if set(payload) != required:
                raise StateError("attempt lease payload schema is invalid")
            stage = payload["stage"]
            attempt_id = payload["attempt_id"]
            _require_stage_eligible(result, model_id, stage)
            if payload["leased_revision"] != state["revision"]:
                raise StateError("attempt lease was created against a stale state revision")
            key = _attempt_key(model_id, stage, attempt_id)
            if key in result["attempts"]:
                raise StateError("attempt identity has already been leased")
            if not all(
                isinstance(payload[name], str) and payload[name]
                for name in ("lease_nonce", "command_sha256", "attempt_sha256")
            ):
                raise StateError("attempt lease nonce/hash is invalid")
            if (
                not isinstance(payload["gpu_uuid"], str)
                or not payload["gpu_uuid"]
                or type(payload["gpu_index"]) is not int
                or not isinstance(payload["keepalive_root"], str)
                or not isinstance(payload["keepalive_owner"], str)
            ):
                raise StateError("attempt GPU/keepalive lease binding is invalid")
            result["attempts"][key] = {
                "model_id": model_id,
                "stage": stage,
                "attempt_id": attempt_id,
                "purpose": payload["purpose"],
                "lease_nonce": payload["lease_nonce"],
                "command_sha256": payload["command_sha256"],
                "attempt_sha256": payload["attempt_sha256"],
                "attempt_reference": payload["attempt_reference"],
                "gpu_uuid": payload["gpu_uuid"],
                "gpu_index": payload["gpu_index"],
                "keepalive_root": payload["keepalive_root"],
                "keepalive_owner": payload["keepalive_owner"],
                "leased_revision": payload["leased_revision"],
                "status": "leased",
                "execution": None,
                "verification": None,
            }
        elif event_type == "ATTEMPT_STARTED":
            required = {
                "model_id", "stage", "attempt_id", "lease_nonce", "command_sha256",
                "gpu_uuid",
            }
            if set(payload) != required:
                raise StateError("attempt start payload schema is invalid")
            stage, attempt_id = payload["stage"], payload["attempt_id"]
            attempt = _require_attempt(result, model_id, stage, attempt_id)
            _require_stage_eligible(result, model_id, stage)
            if attempt["status"] != "leased":
                raise StateError("attempt can only start once from its lease")
            if (
                payload["lease_nonce"] != attempt["lease_nonce"]
                or payload["command_sha256"] != attempt["command_sha256"]
                or payload["gpu_uuid"] != attempt["gpu_uuid"]
            ):
                raise StateError("attempt start nonce/command binding mismatch")
            attempt["status"] = "started"
            attempt["started_revision"] = event["sequence"]
        elif event_type == "ATTEMPT_EXECUTED":
            required = {
                "model_id", "stage", "attempt_id", "lease_nonce", "command_sha256",
                "status", "process_started", "execution",
                "gpu_uuid",
            }
            if set(payload) != required:
                raise StateError("attempt execution payload schema is invalid")
            stage, attempt_id = payload["stage"], payload["attempt_id"]
            attempt = _require_attempt(result, model_id, stage, attempt_id)
            if attempt["status"] != "started":
                raise StateError("attempt execution must follow exactly one start event")
            if (
                payload["lease_nonce"] != attempt["lease_nonce"]
                or payload["command_sha256"] != attempt["command_sha256"]
                or payload["status"] not in {"success", "failed"}
                or type(payload["process_started"]) is not bool
                or payload["gpu_uuid"] != attempt["gpu_uuid"]
            ):
                raise StateError("attempt execution binding/status is invalid")
            attempt["status"] = "executed"
            attempt["execution_status"] = payload["status"]
            attempt["process_started"] = payload["process_started"]
            attempt["execution"] = payload["execution"]
            attempt["executed_revision"] = event["sequence"]
        elif event_type == "ATTEMPT_VERIFIED":
            required = {
                "model_id", "stage", "attempt_id", "lease_nonce", "command_sha256",
                "verifier_status", "verification",
                "gpu_uuid",
            }
            if set(payload) != required:
                raise StateError("attempt verifier payload schema is invalid")
            stage, attempt_id = payload["stage"], payload["attempt_id"]
            attempt = _require_attempt(result, model_id, stage, attempt_id)
            if stage != "finetune" or attempt["status"] != "executed":
                raise StateError("only an executed finetune attempt can be verified")
            if attempt["verification"] is not None:
                raise StateError("attempt verifier result is append-only")
            if (
                payload["lease_nonce"] != attempt["lease_nonce"]
                or payload["command_sha256"] != attempt["command_sha256"]
                or payload["verifier_status"] not in {"passed", "failed"}
                or payload["gpu_uuid"] != attempt["gpu_uuid"]
            ):
                raise StateError("attempt verifier binding/status is invalid")
            attempt["verification_status"] = payload["verifier_status"]
            attempt["verification"] = payload["verification"]
            attempt["verified_revision"] = event["sequence"]
        elif event_type == "FINETUNE_COMPLETE":
            model = _require_model(result, model_id)
            if model["finetune_status"] != "pending" or result["eval_open"]:
                raise StateError("finetune completion is not valid in the current state")
            attempt = _require_attempt(result, model_id, "finetune", payload.get("attempt_id"))
            if (
                attempt["purpose"] != "production"
                or attempt["status"] != "executed"
                or attempt.get("execution_status") != "success"
                or attempt.get("process_started") is not True
                or attempt.get("verification_status") != "passed"
            ):
                raise StateError("finetune completion lacks controller execution/verifier chain")
            model["finetune_status"] = "finetune_complete"
            model["finetune_evidence"] = payload["evidence"]
        elif event_type == "FINETUNE_BLOCKED":
            model = _require_model(result, model_id)
            if model["finetune_status"] != "pending" or result["eval_open"]:
                raise StateError("blocked transition is not valid in the current state")
            model["finetune_status"] = "blocked"
            model["finetune_evidence"] = payload["evidence"]
        elif event_type == "EVAL_OPENED":
            if result["eval_open"]:
                raise StateError("evaluation phase is already open")
            statuses = [model["finetune_status"] for model in result["models"].values()]
            if any(status not in {"finetune_complete", "blocked"} for status in statuses):
                raise StateError("global finetune barrier is not terminal")
            result["eval_open"] = True
            result["phase"] = "smoke"
        elif event_type == "SMOKE_PASSED":
            model = _require_model(result, model_id)
            size = str(payload["size"])
            if not result["eval_open"] or result["full_open"]:
                raise StateError("smoke transition is outside the smoke phase")
            if model["finetune_status"] != "finetune_complete":
                raise StateError("blocked or unfinished models cannot run evaluation")
            expected_next = next(
                (candidate for candidate in ("1", "8", "32") if model["smoke"][candidate] != "passed"),
                None,
            )
            if size != expected_next:
                raise StateError(f"smoke sequence violation; expected {expected_next}, got {size}")
            attempt = _require_attempt(
                result, model_id, f"smoke_{size}", payload.get("attempt_id")
            )
            if (
                attempt["status"] != "executed"
                or attempt.get("execution_status") != "success"
                or attempt.get("process_started") is not True
            ):
                raise StateError("smoke completion lacks controller execution chain")
            model["smoke"][size] = "passed"
            model["smoke_evidence"][size] = payload["evidence"]
        elif event_type == "FULL_OPENED":
            if not result["eval_open"] or result["full_open"]:
                raise StateError("full evaluation gate is not eligible")
            for model in result["models"].values():
                if model["finetune_status"] == "finetune_complete" and any(
                    value != "passed" for value in model["smoke"].values()
                ):
                    raise StateError("all evaluable models must pass smoke 1/8/32")
            result["full_open"] = True
            result["phase"] = "full"
        elif event_type == "FULL_COMPLETE":
            model = _require_model(result, model_id)
            if not result["full_open"] or model["finetune_status"] != "finetune_complete":
                raise StateError("full result is not eligible")
            if model["full_status"] != "pending":
                raise StateError("full result is already terminal")
            attempt = _require_attempt(result, model_id, "full", payload.get("attempt_id"))
            if (
                attempt["status"] != "executed"
                or attempt.get("execution_status") != "success"
                or attempt.get("process_started") is not True
            ):
                raise StateError("full completion lacks controller execution chain")
            model["full_status"] = "complete"
            model["full_evidence"] = payload["evidence"]
        elif event_type == "RELEASE_BUILT":
            if not result["full_open"] or result["release_status"] != "pending":
                raise StateError("release cannot be built in the current state")
            for model in result["models"].values():
                if model["finetune_status"] == "finetune_complete" and model["full_status"] != "complete":
                    raise StateError("release requires every evaluable model full result")
            result["release_status"] = "built"
            result["release_evidence"] = payload["evidence"]
            result["phase"] = "released"
        else:  # pragma: no cover - guarded during event validation
            raise StateError(f"unknown event type: {event_type}")
    result["revision"] = event["sequence"]
    result["last_event_sha256"] = event["event_sha256"]
    return result


class EventStore:
    def __init__(self, batch_root: str | Path) -> None:
        self.batch_root = Path(batch_root).resolve(strict=True)
        trust, event_key, controller_namespace = _event_trust_material(
            self.batch_root.parent
        )
        self.trust_proof = trust
        self._event_key = event_key
        self.controller_namespace = controller_namespace
        self.state_root = self.batch_root / ".controller"
        self.events_root = self.state_root / "events"
        self.cache_path = self.state_root / "state.json"
        self.lock_path = self.state_root / "transition.lock"
        self.batch_root_id = sha256_json(
            {
                "schema_version": "1.0",
                "batch_root": os.path.normcase(str(self.batch_root)),
            }
        )
        self.anchor_path = (
            self.controller_namespace / "heads" / f"{self.batch_root_id[:40]}.json"
        )

    def _hmac(self, digest: str) -> str:
        return hmac.new(self._event_key, digest.encode("ascii"), hashlib.sha256).hexdigest()

    def _anchor_hmac(self, digest: str) -> str:
        message = f"motion-eval-event-anchor-v1:{digest}".encode("ascii")
        return hmac.new(self._event_key, message, hashlib.sha256).hexdigest()

    @staticmethod
    def _anchor_body(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"anchor_sha256", "anchor_hmac_sha256"}
        }

    def _read_anchor(self) -> dict[str, Any]:
        try:
            value = load_json_strict(self.anchor_path)
        except FileNotFoundError as exc:
            raise StateError("controller event head anchor is missing") from exc
        if not isinstance(value, Mapping):
            raise StateError("controller event head anchor must be an object")
        anchor = dict(value)
        required = {
            "schema_version",
            "workspace_scope_id",
            "batch_root_id",
            "batch_id",
            "batch_receipt_sha256",
            "sequence",
            "last_event_sha256",
            "anchor_sha256",
            "anchor_hmac_sha256",
        }
        if set(anchor) != required or anchor.get("schema_version") != EVENT_ANCHOR_SCHEMA_VERSION:
            raise StateError("controller event head anchor schema is invalid")
        if (
            anchor.get("workspace_scope_id") != self.trust_proof["state_scope_id"]
            or anchor.get("batch_root_id") != self.batch_root_id
        ):
            raise StateError("controller event head anchor scope mismatch")
        if type(anchor.get("sequence")) is not int or anchor["sequence"] < 0:
            raise StateError("controller event head anchor sequence is invalid")
        body_digest = sha256_json(self._anchor_body(anchor))
        if anchor.get("anchor_sha256") != body_digest:
            raise StateError("controller event head anchor content hash mismatch")
        if not hmac.compare_digest(
            str(anchor.get("anchor_hmac_sha256", "")), self._anchor_hmac(body_digest)
        ):
            raise StateError("controller event head anchor HMAC mismatch")
        return anchor

    def _validate_anchor(self, head_event: Mapping[str, Any]) -> None:
        anchor = self._read_anchor()
        if (
            anchor["batch_id"] != head_event["batch_id"]
            or anchor["batch_receipt_sha256"] != head_event["batch_receipt_sha256"]
        ):
            raise StateError("controller event head anchor batch identity mismatch")
        if (
            anchor["sequence"] != head_event["sequence"]
            or anchor["last_event_sha256"] != head_event["event_sha256"]
        ):
            raise StateError(
                "controller event rollback/replay detected: batch chain differs from external head"
            )

    def _publish_anchor(self, event: Mapping[str, Any], *, initialize: bool) -> None:
        if initialize:
            if self.anchor_path.exists():
                raise StateError("controller event head anchor already exists")
            if event["sequence"] != 0 or event["previous_event_sha256"] != _ZERO_HASH:
                raise StateError("initial controller anchor must bind GENESIS")
        else:
            previous = self._read_anchor()
            if (
                event["sequence"] != previous["sequence"] + 1
                or event["previous_event_sha256"] != previous["last_event_sha256"]
                or event["batch_id"] != previous["batch_id"]
                or event["batch_receipt_sha256"] != previous["batch_receipt_sha256"]
            ):
                raise StateError("controller event head anchor cannot move non-monotonically")
        body = {
            "schema_version": EVENT_ANCHOR_SCHEMA_VERSION,
            "workspace_scope_id": self.trust_proof["state_scope_id"],
            "batch_root_id": self.batch_root_id,
            "batch_id": event["batch_id"],
            "batch_receipt_sha256": event["batch_receipt_sha256"],
            "sequence": event["sequence"],
            "last_event_sha256": event["event_sha256"],
        }
        digest = sha256_json(body)
        anchor = {
            **body,
            "anchor_sha256": digest,
            "anchor_hmac_sha256": self._anchor_hmac(digest),
        }
        atomic_write_json(
            self.anchor_path,
            anchor,
            root=self.controller_namespace,
            overwrite=not initialize,
        )
        _chmod_best_effort(self.anchor_path, 0o600)

    @contextmanager
    def lock(self, *, timeout_seconds: float = 10.0) -> Iterator[None]:
        self.state_root.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        deadline = time.monotonic() + timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                    0o600,
                )
            except (FileExistsError, PermissionError):
                # Windows can report an existing, exclusively opened lock file as
                # EACCES instead of EEXIST.  Treat both as lock contention and
                # retain the same bounded, no-stale-takeover policy.
                if time.monotonic() >= deadline:
                    raise ConcurrentTransitionError(
                        "controller transition lock is held; refusing unsafe stale-lock takeover"
                    )
                time.sleep(0.01)
        try:
            os.write(descriptor, token.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            yield
        finally:
            try:
                current = self.lock_path.read_text(encoding="ascii")
                if current != token:
                    raise ConcurrentTransitionError("transition lock identity changed")
                self.lock_path.unlink()
            except FileNotFoundError as exc:
                raise ConcurrentTransitionError("transition lock disappeared") from exc

    def initialize(self, *, batch_id: str, receipt_sha256: str, model_ids: list[str]) -> dict[str, Any]:
        with self.lock():
            if self.events_root.exists() or self.cache_path.exists():
                raise StateError("controller state already exists")
            self.events_root.mkdir(parents=True, exist_ok=False)
            event = self._new_event(
                sequence=0,
                previous=_ZERO_HASH,
                batch_id=batch_id,
                receipt_sha256=receipt_sha256,
                event_type="GENESIS",
                payload={"model_ids": model_ids},
            )
            atomic_write_json(
                self.events_root / "00000000.json", event, root=self.batch_root, overwrite=False
            )
            state = _apply_event(None, event)
            self._write_cache(state)
            self._publish_anchor(event, initialize=True)
            return state

    def _new_event(
        self,
        *,
        sequence: int,
        previous: str,
        batch_id: str,
        receipt_sha256: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if event_type not in _EVENT_TYPES:
            raise StateError(f"unsupported event type: {event_type}")
        body = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": sequence,
            "batch_id": batch_id,
            "batch_receipt_sha256": receipt_sha256,
            "previous_event_sha256": previous,
            "event_type": event_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        digest = sha256_json(body)
        return {
            **body,
            "event_sha256": digest,
            "event_hmac_sha256": self._hmac(digest),
        }

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.events_root.is_dir():
            raise StateError("controller event store is missing")
        files = sorted(self.events_root.glob("*.json"))
        if not files:
            raise StateError("controller event chain is empty")
        events: list[dict[str, Any]] = []
        previous = _ZERO_HASH
        batch_id: str | None = None
        receipt_hash: str | None = None
        for sequence, path in enumerate(files):
            if path.name != f"{sequence:08d}.json":
                raise StateError("controller event sequence has a gap or unexpected file")
            value = load_json_strict(path)
            if not isinstance(value, Mapping):
                raise StateError(f"event is not an object: {path}")
            event = dict(value)
            required = {
                "schema_version", "sequence", "batch_id", "batch_receipt_sha256",
                "previous_event_sha256", "event_type", "created_at", "payload", "event_sha256",
                "event_hmac_sha256",
            }
            if set(event) != required or event.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise StateError(f"event schema is invalid: {path}")
            if event["sequence"] != sequence or event["previous_event_sha256"] != previous:
                raise StateError("event sequence/hash chain mismatch")
            if event["event_type"] not in _EVENT_TYPES or not isinstance(event["payload"], Mapping):
                raise StateError("event type/payload is invalid")
            if event["event_sha256"] != sha256_json(_event_body(event)):
                raise StateError("event content hash mismatch")
            if event["event_hmac_sha256"] != self._hmac(event["event_sha256"]):
                raise StateError("event controller HMAC mismatch")
            if batch_id is None:
                batch_id, receipt_hash = event["batch_id"], event["batch_receipt_sha256"]
            elif (event["batch_id"], event["batch_receipt_sha256"]) != (batch_id, receipt_hash):
                raise StateError("event changed batch or input receipt identity")
            previous = event["event_sha256"]
            events.append(event)
        self._validate_anchor(events[-1])
        return events

    def replay(self) -> dict[str, Any]:
        state: dict[str, Any] | None = None
        for event in self._load_events():
            state = _apply_event(state, event)
        assert state is not None
        return state

    def load(self) -> dict[str, Any]:
        expected = self.replay()
        value = load_json_strict(self.cache_path)
        if not isinstance(value, Mapping):
            raise StateError("state cache must be an object")
        cache = dict(value)
        if cache.get("state_sha256") != sha256_json(_state_body(cache)):
            raise StateError("state cache hash mismatch")
        if cache.get("state_hmac_sha256") != self._hmac(cache["state_sha256"]):
            raise StateError("state cache controller HMAC mismatch")
        if _state_body(cache) != expected:
            raise StateError("state cache differs from append-only event replay")
        return expected

    def _write_cache(self, state: Mapping[str, Any]) -> None:
        cache = dict(state)
        cache["state_sha256"] = sha256_json(state)
        cache["state_hmac_sha256"] = self._hmac(cache["state_sha256"])
        atomic_write_json(self.cache_path, cache, root=self.batch_root, overwrite=True)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            state = self.load()
            if expected_revision is not None and state["revision"] != expected_revision:
                raise ConcurrentTransitionError(
                    f"state revision changed: expected {expected_revision}, got {state['revision']}"
                )
            sequence = state["revision"] + 1
            event = self._new_event(
                sequence=sequence,
                previous=state["last_event_sha256"],
                batch_id=state["batch_id"],
                receipt_sha256=state["batch_receipt_sha256"],
                event_type=event_type,
                payload=payload,
            )
            candidate = _apply_event(state, event)
            atomic_write_json(
                self.events_root / f"{sequence:08d}.json",
                event,
                root=self.batch_root,
                overwrite=False,
            )
            self._write_cache(candidate)
            self._publish_anchor(event, initialize=False)
            return candidate

    def append_prepared(
        self,
        event_type: str,
        prepare: Callable[
            [dict[str, Any]],
            tuple[Mapping[str, Any], Callable[[], None], Callable[[], None]],
        ],
    ) -> dict[str, Any]:
        """Atomically validate state, prepare a side file, and append its event anchor.

        The caller's commit runs only after the candidate event has passed the
        state reducer while the transition lock is held.  Its rollback is used
        if publishing either the side file or event/cache fails.
        """

        with self.lock():
            state = self.load()
            payload, commit, rollback = prepare(state)
            sequence = state["revision"] + 1
            event = self._new_event(
                sequence=sequence,
                previous=state["last_event_sha256"],
                batch_id=state["batch_id"],
                receipt_sha256=state["batch_receipt_sha256"],
                event_type=event_type,
                payload=payload,
            )
            candidate = _apply_event(state, event)
            committed = False
            try:
                commit()
                committed = True
                atomic_write_json(
                    self.events_root / f"{sequence:08d}.json",
                    event,
                    root=self.batch_root,
                    overwrite=False,
                )
                self._write_cache(candidate)
                self._publish_anchor(event, initialize=False)
            except BaseException:
                if committed:
                    rollback()
                raise
            return candidate
