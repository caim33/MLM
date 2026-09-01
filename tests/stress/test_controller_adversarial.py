from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import motion_eval.controller.batch as batch_module
import motion_eval.data.receipts as receipt_module
from motion_eval.controller import BatchController, ControllerValidationError, EventStore
from motion_eval.core import atomic_write_json, hash_path, sha256_bytes, sha256_file, sha256_json
from motion_eval.data import BatchReceiptError, BenchmarkItem
from motion_eval.evaluation import make_generative_row
from motion_eval.reporting import build_release_files
from motion_eval.runtime import CommandSpec


REPO = Path(__file__).resolve().parents[2]


def _controller(workspace: Path) -> BatchController:
    controller = BatchController(
        workspace,
        registry_path=REPO / "model_evaluation_agent" / "model_registry.json",
        pretrained_registry_path=REPO
        / "model_evaluation_agent"
        / "pretrained_registry.json",
    )
    # Stress the post-bootstrap state machine while the production default is
    # independently tested as fail-closed.
    controller._controller_verified_multi_root_bootstrap = lambda **_kwargs: True
    return controller


def _append_block_event_process(
    root: str,
    model_id: str,
    start: multiprocessing.synchronize.Event,
    queue: object,
) -> None:
    start.wait(10)
    try:
        EventStore(root).append(
            "FINETUNE_BLOCKED",
            {
                "model_id": model_id,
                "evidence": {"process": os.getpid(), "model_id": model_id},
            },
        )
    except BaseException as exc:  # pragma: no cover - reported to parent process
        queue.put(("error", model_id, type(exc).__name__, str(exc)))
        raise
    else:
        queue.put(("ok", model_id))


@pytest.mark.stress
def test_concurrent_batch_create_loser_never_deletes_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path / "batches")
    batch_id = "race_batch_deadbeef"
    target = controller.workspace_root / batch_id
    first_exists = threading.Barrier(2)
    root_created = threading.Event()
    losing_cleanup_finished = threading.Event()
    seen_threads: set[int] = set()
    seen_lock = threading.Lock()
    owner_thread: list[int] = []
    real_exists = Path.exists
    real_mkdir = Path.mkdir
    real_rmtree = shutil.rmtree

    def synchronized_initial_exists(path: Path) -> bool:
        if Path(path) == target:
            identity = threading.get_ident()
            with seen_lock:
                first = identity not in seen_threads
                if first:
                    seen_threads.add(identity)
            if first:
                first_exists.wait(timeout=10)
                return False
        return real_exists(path)

    def ordered_root_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) != target:
            return real_mkdir(path, *args, **kwargs)
        identity = threading.get_ident()
        with seen_lock:
            if not owner_thread:
                owner_thread.append(identity)
                owner = True
            else:
                owner = owner_thread[0] == identity
        if owner:
            real_mkdir(path, *args, **kwargs)
            root_created.set()
            # An unsafe loser removes this root and signals. A safe loser does
            # neither; the short timeout then lets the owner publish normally.
            losing_cleanup_finished.wait(timeout=0.5)
            return None
        assert root_created.wait(timeout=10)
        return real_mkdir(path, *args, **kwargs)

    def observed_rmtree(path: object, *args: Any, **kwargs: Any) -> None:
        try:
            real_rmtree(path, *args, **kwargs)
        finally:
            if Path(path) == target:
                losing_cleanup_finished.set()

    def minimal_receipt(destination: Path, **_: Any) -> dict[str, Any]:
        receipt = {"receipt_sha256": "a" * 64}
        atomic_write_json(destination, receipt, overwrite=False)
        return receipt

    monkeypatch.setattr(Path, "exists", synchronized_initial_exists)
    monkeypatch.setattr(Path, "mkdir", ordered_root_mkdir)
    monkeypatch.setattr(batch_module.shutil, "rmtree", observed_rmtree)
    monkeypatch.setattr(batch_module, "create_batch_receipt", minimal_receipt)

    def create() -> tuple[str, object]:
        try:
            return "created", controller.create_batch(batch_id, inputs={})
        except BaseException as exc:
            return "failed", exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: create(), range(2)))

    assert sum(status == "created" for status, _ in outcomes) == 1
    failures = [value for status, value in outcomes if status == "failed"]
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert target.is_dir()
    assert (target / "00_inputs" / "batch_receipt.json").is_file()
    assert (target / "02_finetune").is_dir()
    assert (target / ".controller" / "events" / "00000000.json").is_file()
    assert not (target / ".create_owner.json").exists()


@pytest.mark.stress
def test_concurrent_attempt_publish_is_append_only_and_collision_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path / "workspace")
    root = controller.workspace_root / "attempt_batch"
    root.mkdir()
    receipt = {"batch_id": "attempt_batch", "receipt_sha256": "b" * 64}
    EventStore(root).initialize(
        batch_id="attempt_batch",
        receipt_sha256=receipt["receipt_sha256"],
        model_ids=list(controller.registry.ids),
    )
    monkeypatch.setattr(controller, "_receipt", lambda _: (root, receipt))
    monkeypatch.setattr(controller, "_require_catalog_command", lambda *args, **kwargs: None)
    command = CommandSpec(argv=("python", "runner.py"), cwd=str(root))

    def publish(attempt_id: str) -> tuple[str, object]:
        try:
            value = controller.create_attempt(
                "attempt_batch",
                model_id="qwen36_27b_lora",
                stage="finetune",
                command=command,
                attempt_id=attempt_id,
                purpose="production",
                expected_training_steps=1,
            )
            return "created", value
        except BaseException as exc:
            return "failed", exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        unique = list(pool.map(publish, [f"unique_{index}" for index in range(16)]))
    unique_failures = [value for status, value in unique if status == "failed"]
    assert not unique_failures, [
        (type(value).__name__, str(value)) for value in unique_failures
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        collision = list(pool.map(publish, ["same_attempt", "same_attempt"]))
    assert sum(status == "created" for status, _ in collision) == 1
    failures = [value for status, value in collision if status == "failed"]
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)

    attempts = root / "02_finetune" / "qwen36_27b_lora" / "attempts"
    receipts = sorted(attempts.glob("*/attempt_receipt.json"))
    assert len(receipts) == 17
    assert len({json.loads(path.read_text())["attempt_sha256"] for path in receipts}) == 17


@pytest.mark.stress
def test_event_chain_serializes_independent_process_transitions(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    model_ids = [f"model_{index}" for index in range(6)]
    EventStore(root).initialize(
        batch_id="process_batch",
        receipt_sha256="9" * 64,
        model_ids=model_ids,
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(
            target=_append_block_event_process,
            args=(str(root), model_id, start, queue),
        )
        for model_id in model_ids
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
    try:
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    outcomes = [queue.get(timeout=5) for _ in processes]
    assert all(item[0] == "ok" for item in outcomes)
    state = EventStore(root).load()
    assert state["revision"] == len(model_ids)
    assert all(
        state["models"][model_id]["finetune_status"] == "blocked"
        for model_id in model_ids
    )
    assert len(list((root / ".controller" / "events").glob("*.json"))) == len(
        model_ids
    ) + 1


def test_forged_zero_leakage_counts_are_recomputed(tmp_path: Path) -> None:
    bindings = {
        "train_sha256": "1" * 64,
        "validation_sha256": "2" * 64,
        "benchmark_sha256": "3" * 64,
        "media_manifest_sha256": "4" * 64,
    }
    fake_checks = {
        "sample_id": 0,
        "group_id": 0,
        "media_sha256": 0,
        "normalized_question_options": 0,
        "near_duplicate": 0,
    }
    algorithm = {
        "version": receipt_module.LEAKAGE_ALGORITHM_VERSION,
        "sha256": receipt_module.LEAKAGE_ALGORITHM_SHA256,
    }
    computation = {"algorithm": algorithm, "bindings": bindings, "checks": fake_checks}
    audit = {
        "schema_version": receipt_module.LEAKAGE_AUDIT_SCHEMA_VERSION,
        "status": "passed",
        **computation,
        "computed_sha256": sha256_json(computation),
    }
    path = tmp_path / "leakage.json"
    atomic_write_json(path, audit)
    shared = {
        "sample_id": "same",
        "group_id": "same-group",
        "media_sha256": frozenset({"a" * 64}),
        "normalized_question_options": "b" * 64,
        "near_duplicate": "c" * 64,
    }

    with pytest.raises(BatchReceiptError, match="counts differ"):
        receipt_module._validate_leakage_audit(
            path,
            bindings=bindings,
            train_records=(shared,),
            validation_records=(),
            benchmark_records=(dict(shared),),
        )


def test_media_manifest_cannot_claim_an_unrelated_content_hash(tmp_path: Path) -> None:
    media = tmp_path / "video.bin"
    media.write_bytes(b"actual-video")
    manifest = {
        "schema_version": "1.0",
        "row_count": 1,
        "resources": [
            {
                "resource_id": "video-0",
                "kind": "video",
                "path": str(media),
                "sha256": sha256_bytes(b"different-video"),
            }
        ],
        "rows": [{"sample_id": "sample-0", "resource_ids": ["video-0"]}],
    }
    path = tmp_path / "media.json"
    atomic_write_json(path, manifest)

    with pytest.raises(BatchReceiptError, match="hash mismatch"):
        receipt_module._validate_media_manifest(
            path,
            (BenchmarkItem("sample-0", "group-0", "A"),),
            required_kinds=frozenset({"video"}),
        )


def test_frozen_pretrained_component_is_rehashed_after_same_size_tamper(
    tmp_path: Path,
) -> None:
    pretrained_root = tmp_path / "pretrained"
    component = pretrained_root / "by_model" / "model" / "base.bin"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"original-component")
    frozen, _ = receipt_module._freeze_pretrained_assets(
        pretrained_root,
        {
            "model": [
                {
                    "role": "base",
                    "path": "by_model/model/base.bin",
                    "kind": "checkpoint",
                    "expected_sha256": None,
                }
            ]
        },
    )
    assert frozen["model"][0]["state"] == "present"

    component.write_bytes(b"tampered-component")  # Same byte length as original.

    with pytest.raises(BatchReceiptError, match="changed after batch freeze"):
        receipt_module._verify_pretrained_assets(frozen)


def _blocked_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[BatchController, str, Path]:
    controller = _controller(tmp_path / "workspace")
    batch_id = "blocked_batch"
    root = controller.workspace_root / batch_id
    root.mkdir()
    model_id = "qwen3vl_8b_lora"
    role = controller.registry.pretrained_artifacts[model_id][0].role
    receipt = {
        "batch_id": batch_id,
        "receipt_sha256": "d" * 64,
        "pretrained_assets_sha256": "e" * 64,
        "runtime_roots": {
            "controller_root": str(tmp_path / "runners"),
            "pretrained_root": str(tmp_path / "pretrained"),
        },
        "pretrained_assets": {
            registry_id: [
                {
                    "role": controller.registry.pretrained_artifacts[registry_id][0].role,
                    "state": "missing",
                    "path": str(
                        tmp_path
                        / "pretrained"
                        / controller.registry.pretrained_artifacts[registry_id][0].path
                    ),
                }
            ]
            for registry_id in controller.registry.ids
        },
    }
    EventStore(root).initialize(
        batch_id=batch_id,
        receipt_sha256=receipt["receipt_sha256"],
        model_ids=list(controller.registry.ids),
    )
    monkeypatch.setattr(controller, "_receipt", lambda _: (root, receipt))
    # This isolated state-machine fixture intentionally has no frozen input
    # tree. Full-content receipt auditing remains covered by real batch
    # fixtures; substitute only this synthetic receipt in this helper.
    monkeypatch.setattr(
        batch_module,
        "load_and_validate_batch_receipt",
        lambda _path, *, verify_pretrained_content=False: receipt,
    )
    controller.block_finetune(
        batch_id,
        model_id=model_id,
        reason_code="missing_path",
        component=f"pretrained:{role}",
        detail="registered component is absent in the isolated test fixture",
    )
    return controller, batch_id, root


def test_blocker_tamper_rehash_cannot_bypass_event_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, batch_id, root = _blocked_controller(tmp_path, monkeypatch)
    untampered = controller.validate_batch(batch_id)
    assert (
        untampered["models"]["qwen3vl_8b_lora"]["finetune_status"]
        == "blocked"
    )

    state = EventStore(root).load()
    reference = state["models"]["qwen3vl_8b_lora"]["finetune_evidence"]
    evidence_path = root / reference["path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["detail"] = "attacker rewrote the explanation"
    body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence["evidence_sha256"] = sha256_json(body)
    atomic_write_json(evidence_path, evidence)

    with pytest.raises(ControllerValidationError, match="evidence file changed"):
        controller.validate_batch(batch_id)


def test_all_invalid_smoke_rows_never_count_as_a_pass(tmp_path: Path) -> None:
    controller = _controller(tmp_path / "workspace")
    model = controller.registry.model("qwen36_27b_lora")
    expected = tuple(
        BenchmarkItem(f"sample-{index}", f"group-{index}", "A")
        for index in range(32)
    )
    rows = [
        make_generative_row(
            batch_id="smoke_batch",
            model_id=model.model_id,
            sample_id=item.sample_id,
            group_id=item.group_id,
            modality=model.modality,
            gold=item.gold,
            raw_output="A",  # Deliberately lacks the strict answer tag.
        ).to_dict()
        for item in expected
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ControllerValidationError, match="strict valid"):
        controller._validate_prediction_rows(
            path,
            batch_id="smoke_batch",
            model=model,
            expected_items=expected,
            smoke=True,
        )


def test_rehashed_run_manifest_tamper_cannot_replace_process_exit_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real lease/run event is valid, but cannot bless a later self-rehash."""

    controller = _controller(tmp_path / "workspace")
    batch_id = "manifest_batch"
    model_id = "qwen36_27b_lora"
    attempt_id = "production_attempt"
    root = controller.workspace_root / batch_id
    root.mkdir()
    (root / "02_finetune" / model_id).mkdir(parents=True)
    attempt_root = root / "02_finetune" / model_id / "attempts" / attempt_id
    manifest_path = attempt_root / "run_manifest.json"
    original_body = {"schema_version": "test", "status": "success"}
    original = {**original_body, "manifest_sha256": sha256_json(original_body)}
    runner_code = (
        "from pathlib import Path; "
        f"Path({str(manifest_path)!r}).write_text({json.dumps(original, sort_keys=True)!r}, encoding='utf-8')"
    )
    runner_code_root = root / "runner_code"
    runner_code_root.mkdir()
    runner = runner_code_root / "frozen_test_runner.py"
    runner.write_text(runner_code + "\n", encoding="utf-8")
    interpreter_launcher = Path(os.path.abspath(sys.executable))
    interpreter_target = interpreter_launcher.resolve(strict=True)
    runner = runner.resolve(strict=True)
    command = CommandSpec(
        argv=(str(interpreter_launcher), str(runner)),
        cwd=str(root),
        env={"CUDA_VISIBLE_DEVICES": "UNBOUND-TEST"},
    )
    command_template = command.receipt()
    receipt = {
        "batch_id": batch_id,
        "receipt_sha256": "a" * 64,
        "runner_code": {
            "path": str(runner_code_root.resolve(strict=True)),
            **hash_path(runner_code_root, symlink_policy="reject").to_dict(),
        },
        "runtime_contract": {
            "interpreter": {
                "launcher_path": str(interpreter_launcher),
                "path": str(interpreter_target),
                **hash_path(
                    interpreter_target, symlink_policy="reject"
                ).to_dict(),
            },
            "models": {
                model_id: {
                    "finetune": {
                        "relative_path": runner.name,
                        "state": "present",
                        "absolute_path": str(runner),
                        "runner_receipt": {
                            "path": str(runner),
                            **hash_path(runner, symlink_policy="reject").to_dict(),
                        },
                        "command_template": command_template,
                        "command_template_sha256": sha256_json(command_template),
                    }
                }
            },
        },
    }
    EventStore(root).initialize(
        batch_id=batch_id,
        receipt_sha256=receipt["receipt_sha256"],
        model_ids=list(controller.registry.ids),
    )
    monkeypatch.setattr(controller, "_receipt", lambda _: (root, receipt))
    monkeypatch.setattr(controller, "_require_catalog_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_prepare_worker_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_run_finetune_verifier", lambda *args, **kwargs: None)

    controller.create_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        command=command,
        attempt_id=attempt_id,
        purpose="production",
        expected_training_steps=1,
        gpu_binding={
            "gpu_uuid": "UNBOUND-TEST",
            "gpu_index": -1,
            "keepalive_root": "UNBOUND-TEST",
            "keepalive_owner": "UNBOUND-TEST",
        },
    )
    execution = controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    assert execution["status"] == "success"
    loaded_root, attempt = controller._load_attempt(
        root, model_id=model_id, stage="finetune", attempt_id=attempt_id
    )
    controller._load_execution(
        root,
        loaded_root,
        attempt,
        batch_id=batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )

    tampered_body = {"schema_version": "test", "status": "attacker-rehashed"}
    atomic_write_json(
        manifest_path,
        {**tampered_body, "manifest_sha256": sha256_json(tampered_body)},
    )
    with pytest.raises(ControllerValidationError, match="worker output changed"):
        controller._load_execution(
            root,
            loaded_root,
            attempt,
            batch_id=batch_id,
            model_id=model_id,
            stage="finetune",
            attempt_id=attempt_id,
        )


def _legacy_rehashed_run_manifest_fixture(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path / "workspace")
    batch_id = "manifest_batch"
    model_id = "qwen36_27b_lora"
    attempt_id = "production_attempt"
    root = controller.workspace_root / batch_id
    attempt_root = (
        root / "02_finetune" / model_id / "attempts" / attempt_id
    )
    attempt_root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    attempt_created = now - timedelta(seconds=5)
    execution_started = now - timedelta(seconds=4)
    training_started = now - timedelta(seconds=3)
    training_finished = now - timedelta(seconds=2)
    reload_checked = now - timedelta(seconds=1)
    execution_finished = now

    command = CommandSpec(argv=("python", "runner.py"), cwd=str(root)).receipt()
    attempt_body = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "batch_id": batch_id,
        "batch_receipt_sha256": "a" * 64,
        "model_id": model_id,
        "stage": "finetune",
        "purpose": "production",
        "expected_training_steps": 1,
        "sample_limit": None,
        "created_at": attempt_created.isoformat(),
        "command": command,
        "command_sha256": sha256_json(command),
    }
    attempt = {**attempt_body, "attempt_sha256": sha256_json(attempt_body)}
    atomic_write_json(attempt_root / "attempt_receipt.json", attempt)

    asset_rows = [
        {
            "role": "base_and_processor",
            "state": "present",
            "path": str(tmp_path / "pretrained" / "base"),
        }
    ]
    training_config = {"training_steps": 1, "preflight_steps": 1}
    receipt = {
        "batch_id": batch_id,
        "receipt_sha256": attempt["batch_receipt_sha256"],
        "registry": {"sha256": "b" * 64},
        "pretrained_registry": {"sha256": "c" * 64},
        "pretrained_assets_sha256": "d" * 64,
        "pretrained_assets": {model_id: asset_rows},
        "config": {"model_training": {model_id: training_config}},
        "config_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "inputs": {
            "train": {"digest": "1" * 64},
            "validation": {"digest": "2" * 64},
            "leakage_audit": {"digest": "3" * 64},
        },
        "code": {"digest": "4" * 64},
        "runner_code": {"digest": "5" * 64},
    }

    artifact = attempt_root / "artifact"
    artifact.mkdir()
    (artifact / "adapter.bin").write_bytes(b"fresh-current-batch-adapter")
    artifact_receipt = {
        "path": str(artifact.resolve()),
        **hash_path(artifact).to_dict(),
    }
    reload_report = {
        "schema_version": "1.0",
        "status": "passed",
        "batch_id": batch_id,
        "model_id": model_id,
        "attempt_id": attempt_id,
        "artifact_digest": artifact_receipt["digest"],
        "checker": f"{model_id}:reload",
        "checked_at": reload_checked.isoformat(),
    }
    reload_path = attempt_root / "reload_report.json"
    atomic_write_json(reload_path, reload_report)

    bindings = {
        "batch_receipt_sha256": receipt["receipt_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "command_sha256": attempt["command_sha256"],
        "registry_sha256": receipt["registry"]["sha256"],
        "pretrained_registry_sha256": receipt["pretrained_registry"]["sha256"],
        "pretrained_assets_sha256": receipt["pretrained_assets_sha256"],
        "model_pretrained_assets_sha256": sha256_json(asset_rows),
        "model_training_config_sha256": sha256_json(training_config),
        "train_sha256": receipt["inputs"]["train"]["digest"],
        "validation_sha256": receipt["inputs"]["validation"]["digest"],
        "leakage_audit_sha256": receipt["inputs"]["leakage_audit"]["digest"],
        "code_sha256": receipt["code"]["digest"],
        "runner_code_sha256": receipt["runner_code"]["digest"],
        "config_sha256": receipt["config_sha256"],
        "environment_sha256": receipt["environment_sha256"],
    }
    manifest_body = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "model_id": model_id,
        "attempt_id": attempt_id,
        "purpose": "production",
        "status": "success",
        "exit_code": 0,
        "started_at": training_started.isoformat(),
        "finished_at": training_finished.isoformat(),
        "training_steps": 1,
        "bindings": bindings,
        "artifact": artifact_receipt,
        "reload_verified": {
            "status": "passed",
            "report_path": str(reload_path.resolve()),
            "report_sha256": sha256_file(reload_path),
        },
    }
    manifest = {**manifest_body, "manifest_sha256": sha256_json(manifest_body)}
    manifest_path = attempt_root / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)

    execution_body = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "model_id": model_id,
        "stage": "finetune",
        "attempt_id": attempt_id,
        "batch_receipt_sha256": receipt["receipt_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "command_sha256": attempt["command_sha256"],
        "status": "success",
        "exit_code": 0,
        "error_code": "none",
        "started_at": execution_started.isoformat(),
        "finished_at": execution_finished.isoformat(),
        "output_path": str(manifest_path.resolve()),
        "output_observed": True,
        "output_fresh": True,
        "output_sha256": sha256_file(manifest_path),
        "stdout_sha256": sha256_bytes(b""),
        "stderr_sha256": sha256_bytes(b""),
    }
    execution = {
        **execution_body,
        "execution_sha256": sha256_json(execution_body),
    }
    atomic_write_json(attempt_root / "execution_receipt.json", execution)

    controller._validate_production_manifest(
        root,
        receipt,
        model_id=model_id,
        attempt_id=attempt_id,
        manifest_path=manifest_path,
    )

    manifest["training_steps"] = 2
    tampered_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_json(tampered_body)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ControllerValidationError, match="worker output changed"):
        controller._validate_production_manifest(
            root,
            receipt,
            model_id=model_id,
            attempt_id=attempt_id,
            manifest_path=manifest_path,
        )


def test_release_table_tamper_is_detected_even_when_manifest_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path / "workspace")
    batch_id = "release_batch"
    root = controller.workspace_root / batch_id
    root.mkdir()
    receipt = {"receipt_sha256": "f" * 64}
    results = [
        {
            "model_id": "qwen36_27b_lora",
            "display_name": "Qwen",
            "modality": "V",
            "evaluation_mode": "generative",
            "correct": 250,
            "denominator": 500,
            "accuracy": 0.5,
            "invalid_output": 0,
            "media_error": 0,
            "timeout": 0,
            "oom": 0,
            "runtime_error": 0,
            "predictions_sha256": "1" * 64,
            "evaluation_evidence_sha256": "2" * 64,
        }
    ]
    manifest = build_release_files(
        root / "04_release",
        batch_root=root,
        batch_id=batch_id,
        batch_receipt_sha256=receipt["receipt_sha256"],
        model_results=results,
        blocked_models=[],
    )
    manifest_path = root / "04_release" / "evaluation_release_manifest.json"
    state = {
        "release_status": "built",
        "release_evidence": {
            "path": manifest_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_sha256"],
        },
    }
    monkeypatch.setattr(controller, "_receipt", lambda _: (root, receipt))
    monkeypatch.setattr(controller, "validate_batch", lambda _: state)
    monkeypatch.setattr(controller, "_release_sources", lambda *args: (results, []))
    controller.verify_release(batch_id)

    csv_path = root / "04_release" / "all_models_results.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "forged,row\n", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="release file changed"):
        controller.verify_release(batch_id)
