"""Runtime proof that every VM/V group reward call is actually co-located."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from motion_eval.core import atomic_write_json

from .schema import RewardBranch, RewardMetadata, RewardMetadataError

COLOCATION_PATH_ENV = "MOTION_GRPO_COLOCATION_RECEIPT_PATH"
COLOCATION_NONCE_ENV = "MOTION_GRPO_COLOCATION_RUN_NONCE"
COLOCATION_DATASET_ENV = "MOTION_GRPO_COLOCATION_DATASET_SHA256"
COLOCATION_CONFIG_ENV = "MOTION_GRPO_COLOCATION_CONFIG_SHA256"
COLOCATION_PLAN_ENV = "MOTION_GRPO_COLOCATION_PLAN_SHA256"
COLOCATION_ENV_KEYS = frozenset(
    {
        COLOCATION_PATH_ENV,
        COLOCATION_NONCE_ENV,
        COLOCATION_DATASET_ENV,
        COLOCATION_CONFIG_ENV,
        COLOCATION_PLAN_ENV,
    }
)
_HEX64 = frozenset("0123456789abcdef")
_PLAN_ALGORITHM = "motion-grpo-colocation-plan-v2"


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX64 for character in value)
    ):
        raise RewardMetadataError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_reward_call_colocation(
    metadata: Sequence[RewardMetadata],
) -> dict[str, dict[str, int]]:
    """Fail when a reward invocation does not contain balanced VM/V rollouts."""

    if not metadata:
        raise RewardMetadataError("group reward call must contain at least one completion")
    groups: dict[str, dict[str, int]] = {}
    for item in metadata:
        if not isinstance(item, RewardMetadata):
            raise RewardMetadataError("co-location metadata must contain RewardMetadata")
        if item.branch not in {RewardBranch.VM, RewardBranch.V}:
            raise RewardMetadataError("co-location accepts only vm/v branches")
        counts = groups.setdefault(item.group_id, {"vm": 0, "v": 0})
        counts[item.branch.value] += 1
    for group_id, counts in sorted(groups.items()):
        if counts["vm"] <= 0 or counts["v"] <= 0 or counts["vm"] != counts["v"]:
            raise RewardMetadataError(
                f"reward call does not co-locate balanced vm/v rollouts for group {group_id!r}: "
                f"{counts['vm']}/{counts['v']}"
            )
    return groups


def _call_evidence(metadata: Sequence[RewardMetadata]) -> dict[str, Any]:
    groups = validate_reward_call_colocation(metadata)
    rollout_keys = [item.rollout_key for item in metadata]
    if len(set(rollout_keys)) != len(rollout_keys):
        raise RewardMetadataError("co-location call contains duplicate rollout keys")
    call_payload = {
        "groups": [
            {"group_id": group_id, **groups[group_id]} for group_id in sorted(groups)
        ],
        "rollout_keys": [list(key) for key in sorted(rollout_keys)],
    }
    call_digest = hashlib.sha256(
        json.dumps(
            call_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "call_digest": call_digest,
        "group_count": len(groups),
        "rollout_count": len(metadata),
    }


def initialize_runtime_colocation_plan(
    path: str | Path,
    *,
    nonce: str,
    dataset_digest: str,
    config_digest: str,
    planned_calls: Sequence[Sequence[RewardMetadata]],
) -> dict[str, Any]:
    """Freeze every expected call/generation before the Swift process starts."""

    destination = Path(path).resolve(strict=False)
    if destination.exists():
        raise RewardMetadataError("runtime co-location receipt path must be fresh")
    if not destination.parent.is_dir():
        raise RewardMetadataError("runtime co-location receipt parent does not exist")
    if not isinstance(nonce, str) or len(nonce) != 32 or not nonce.isalnum():
        raise RewardMetadataError("runtime co-location nonce is invalid")
    if not planned_calls:
        raise RewardMetadataError("runtime co-location plan must contain at least one call")
    calls = []
    for index, metadata in enumerate(planned_calls):
        evidence = _call_evidence(metadata)
        calls.append({"planned_index": index, **evidence})
    plan_body = {
        "algorithm": _PLAN_ALGORITHM,
        "planned_call_count": len(calls),
        "planned_rollout_count": sum(item["rollout_count"] for item in calls),
        "calls": calls,
    }
    plan = {**plan_body, "plan_sha256": hashlib.sha256(
        json.dumps(
            plan_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()}
    payload = {
        "schema_version": "2.0",
        "status": "runtime_colocation_pending",
        "run_nonce": nonce,
        "dataset_sha256": _digest(dataset_digest, name="dataset_sha256"),
        "config_sha256": _digest(config_digest, name="config_sha256"),
        "plan": plan,
        "observed_calls": [],
    }
    atomic_write_json(destination, payload, root=destination.parent, overwrite=False)
    return payload


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RewardMetadataError(f"duplicate co-location receipt key: {key}")
        result[key] = value
    return result


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RewardMetadataError("runtime co-location receipt is unreadable") from exc
    if not isinstance(value, dict):
        raise RewardMetadataError("runtime co-location receipt must be an object")
    return value


@contextmanager
def _process_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except (FileExistsError, PermissionError):
            if time.monotonic() >= deadline:
                raise RewardMetadataError(
                    "runtime co-location receipt lock is held; refusing stale-lock takeover"
                ) from None
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
            if lock_path.read_text(encoding="ascii") != token:
                raise RewardMetadataError("runtime co-location lock identity changed")
            lock_path.unlink()
        except FileNotFoundError as exc:
            raise RewardMetadataError("runtime co-location lock disappeared") from exc


def _binding_from_environment() -> tuple[Path, str, str, str, str] | None:
    values = {name: os.environ.get(name) for name in COLOCATION_ENV_KEYS}
    if all(value in (None, "") for value in values.values()):
        return None
    missing = sorted(name for name, value in values.items() if value in (None, ""))
    if missing:
        raise RewardMetadataError(f"incomplete runtime co-location environment: {missing}")
    path = Path(values[COLOCATION_PATH_ENV]).resolve(strict=False)
    if not path.parent.is_dir():
        raise RewardMetadataError("runtime co-location receipt parent does not exist")
    nonce = values[COLOCATION_NONCE_ENV]
    if not isinstance(nonce, str) or len(nonce) != 32 or not nonce.isalnum():
        raise RewardMetadataError("runtime co-location nonce is invalid")
    return (
        path,
        nonce,
        _digest(values[COLOCATION_DATASET_ENV], name="dataset_sha256"),
        _digest(values[COLOCATION_CONFIG_ENV], name="config_sha256"),
        _digest(values[COLOCATION_PLAN_ENV], name="plan_sha256"),
    )


def record_runtime_colocation(metadata: Sequence[RewardMetadata]) -> Path | None:
    """Atomically accumulate call-level proof from the actual reward process."""

    call = _call_evidence(metadata)
    binding = _binding_from_environment()
    if binding is None:
        return None
    path, nonce, dataset_digest, config_digest, plan_digest = binding
    with _process_lock(path):
        if not path.is_file():
            raise RewardMetadataError(
                "runtime co-location plan was not initialized before reward execution"
            )
        payload = _read_receipt(path)
        _validate_receipt_payload(
            payload,
            nonce=nonce,
            dataset_digest=dataset_digest,
            config_digest=config_digest,
            plan_digest=plan_digest,
            require_complete=False,
        )
        observed_indices = {item["planned_index"] for item in payload["observed_calls"]}
        matching = [
            item
            for item in payload["plan"]["calls"]
            if item["planned_index"] not in observed_indices
            and all(item[key] == call[key] for key in call)
        ]
        if not matching:
            raise RewardMetadataError(
                "runtime co-location call is unexpected, duplicated, or outside the frozen plan"
            )
        planned = min(matching, key=lambda item: item["planned_index"])
        payload["observed_calls"].append(
            {"planned_index": planned["planned_index"], **call}
        )
        payload["observed_calls"].sort(key=lambda item: item["planned_index"])
        if len(payload["observed_calls"]) == payload["plan"]["planned_call_count"]:
            payload["status"] = "runtime_colocation_verified"
        atomic_write_json(path, payload, root=path.parent, overwrite=True)
    return path


def _validate_receipt_payload(
    value: Any,
    *,
    nonce: str,
    dataset_digest: str,
    config_digest: str,
    plan_digest: str,
    require_complete: bool,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "run_nonce",
        "dataset_sha256",
        "config_sha256",
        "plan",
        "observed_calls",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RewardMetadataError("runtime co-location receipt schema mismatch")
    if value["schema_version"] != "2.0" or value["status"] not in {
        "runtime_colocation_pending",
        "runtime_colocation_verified",
    }:
        raise RewardMetadataError("runtime co-location receipt status is invalid")
    if (
        value["run_nonce"] != nonce
        or value["dataset_sha256"] != dataset_digest
        or value["config_sha256"] != config_digest
    ):
        raise RewardMetadataError("runtime co-location receipt binding mismatch")
    plan = value["plan"]
    plan_keys = {
        "algorithm",
        "planned_call_count",
        "planned_rollout_count",
        "calls",
        "plan_sha256",
    }
    if not isinstance(plan, dict) or set(plan) != plan_keys:
        raise RewardMetadataError("runtime co-location plan schema mismatch")
    plan_body = {key: plan[key] for key in plan_keys if key != "plan_sha256"}
    actual_plan_digest = hashlib.sha256(
        json.dumps(
            plan_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        plan.get("algorithm") != _PLAN_ALGORITHM
        or plan.get("plan_sha256") != actual_plan_digest
        or plan.get("plan_sha256") != _digest(plan_digest, name="plan_sha256")
    ):
        raise RewardMetadataError("runtime co-location plan hash/algorithm mismatch")
    calls = plan.get("calls")
    if not isinstance(calls, list) or not calls:
        raise RewardMetadataError("runtime co-location plan contains no calls")
    planned_count = plan.get("planned_call_count")
    planned_rollouts = plan.get("planned_rollout_count")
    if (
        isinstance(planned_count, bool)
        or not isinstance(planned_count, int)
        or planned_count != len(calls)
        or isinstance(planned_rollouts, bool)
        or not isinstance(planned_rollouts, int)
        or planned_rollouts <= 0
    ):
        raise RewardMetadataError("runtime co-location planned counts are invalid")
    planned_by_index: dict[int, dict[str, Any]] = {}
    for call in calls:
        if not isinstance(call, dict) or set(call) != {
            "planned_index",
            "call_digest",
            "group_count",
            "rollout_count",
        }:
            raise RewardMetadataError("runtime co-location call schema mismatch")
        _digest(call["call_digest"], name="call_digest")
        index = call["planned_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RewardMetadataError("runtime co-location planned_index is invalid")
        if index in planned_by_index:
            raise RewardMetadataError("duplicate runtime co-location planned_index")
        planned_by_index[index] = call
        for count_name in ("group_count", "rollout_count"):
            count = call[count_name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RewardMetadataError(f"runtime co-location {count_name} must be positive")
        if call["rollout_count"] < 2 * call["group_count"]:
            raise RewardMetadataError("runtime co-location call cannot contain complete pairs")
    if set(planned_by_index) != set(range(planned_count)):
        raise RewardMetadataError("runtime co-location planned indices are not contiguous")
    if sum(item["rollout_count"] for item in calls) != planned_rollouts:
        raise RewardMetadataError("runtime co-location planned rollout count mismatch")

    observed = value["observed_calls"]
    if not isinstance(observed, list):
        raise RewardMetadataError("runtime co-location observed_calls must be a list")
    seen_indices: set[int] = set()
    for call in observed:
        if not isinstance(call, dict) or set(call) != {
            "planned_index",
            "call_digest",
            "group_count",
            "rollout_count",
        }:
            raise RewardMetadataError("runtime co-location observed call schema mismatch")
        index = call.get("planned_index")
        if index in seen_indices or index not in planned_by_index:
            raise RewardMetadataError("duplicate or unknown observed planned_index")
        seen_indices.add(index)
        if call != planned_by_index[index]:
            raise RewardMetadataError("observed co-location call differs from frozen plan")
    is_complete = seen_indices == set(planned_by_index)
    expected_status = (
        "runtime_colocation_verified" if is_complete else "runtime_colocation_pending"
    )
    if value["status"] != expected_status:
        raise RewardMetadataError("runtime co-location status/completeness mismatch")
    if require_complete and not is_complete:
        missing = sorted(set(planned_by_index) - seen_indices)
        raise RewardMetadataError(
            f"runtime co-location receipt is incomplete; missing planned calls: {missing}"
        )
    return value


def validate_runtime_colocation_receipt(
    path: str | Path,
    *,
    nonce: str,
    dataset_digest: str,
    config_digest: str,
    plan_digest: str,
) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file():
        raise RewardMetadataError("runtime co-location receipt must be a regular file")
    return _validate_receipt_payload(
        _read_receipt(candidate),
        nonce=nonce,
        dataset_digest=_digest(dataset_digest, name="dataset_sha256"),
        config_digest=_digest(config_digest, name="config_sha256"),
        plan_digest=_digest(plan_digest, name="plan_sha256"),
        require_complete=True,
    )
