from __future__ import annotations

import json
import sys
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import motion_eval.controller.batch as batch_module
import motion_eval.data.receipts as receipt_module
from motion_eval.controller import BatchController, ControllerValidationError, EventStore, StateError
from motion_eval.core import atomic_write_json, hash_path, sha256_bytes, sha256_file, sha256_json
from motion_eval.data import BatchReceiptError, StrictJsonError, load_benchmark
from motion_eval.data.receipts import LEAKAGE_ALGORITHM_SHA256, LEAKAGE_ALGORITHM_VERSION
from motion_eval.runtime import CommandSpec, GPUDevice, GPUInventory


REPO = Path(__file__).resolve().parents[2]


class _GPUProbe:
    def query(self):
        return GPUInventory(
            (GPUDevice(0, "GPU-TEST", "test", 24000, 0, 0),),
            (),
        )


def _write_inputs(root: Path) -> dict[str, Path]:
    root.mkdir()
    benchmark = root / "benchmark.jsonl"
    benchmark_rows = [
        {
            "sample_id": f"s{index:03d}",
            "group_id": f"g{index:03d}",
            "gold": "A",
            "question": f"benchmark question {index}",
            "options": {key: f"benchmark {index} option {key}" for key in "ABCD"},
        }
        for index in range(500)
    ]
    train_video = root / "train.video"
    train_motion = root / "train.motion"
    validation_video = root / "validation.video"
    validation_motion = root / "validation.motion"
    train_video.write_bytes(b"unique-train-video")
    train_motion.write_bytes(b"unique-train-motion")
    validation_video.write_bytes(b"unique-validation-video")
    validation_motion.write_bytes(b"unique-validation-motion")
    train = root / "train.jsonl"
    train.write_text(
        json.dumps({
            "sample_id": "train-1", "group_id": "train-group",
            "gold": "A",
            "question": "unique train question",
            "options": {key: f"train option {key}" for key in "ABCD"},
            "video": str(train_video.resolve()), "motion": str(train_motion.resolve()),
        }) + "\n",
        encoding="utf-8",
    )
    validation = root / "validation.jsonl"
    validation.write_text(
        json.dumps({
            "sample_id": "val-1", "group_id": "val-group", "split": "validation",
            "gold": "B",
            "question": "unique validation question",
            "options": {key: f"validation option {key}" for key in "ABCD"},
            "video": str(validation_video.resolve()),
            "motion": str(validation_motion.resolve()),
        }) + "\n",
        encoding="utf-8",
    )
    media_file = root / "media.container"
    resources = []
    media_rows = []
    chunks = []
    pending = []
    benchmark_refs: dict[str, dict[str, object]] = {}
    offset = 0
    for index in range(500):
        sample_id = f"s{index:03d}"
        linked = []
        for kind in ("video", "motion"):
            payload = f"canonical:{sample_id}:{kind}\n".encode("ascii")
            resource_id = f"{sample_id}:{kind}"
            pending.append((resource_id, kind, offset, payload))
            chunks.append(payload)
            offset += len(payload)
            linked.append(resource_id)
        media_rows.append({"sample_id": sample_id, "resource_ids": linked})
    media_file.write_bytes(b"".join(chunks))
    container_sha256 = sha256_file(media_file)
    for resource_id, kind, offset, payload in pending:
        reference = {
            "resource_id": resource_id, "kind": kind,
            "path": str(media_file.resolve()), "sha256": container_sha256,
            "offset": offset, "length": len(payload),
            "content_sha256": sha256_bytes(payload),
        }
        resources.append(reference)
        sample_id = resource_id.split(":", 1)[0]
        benchmark_refs.setdefault(sample_id, {})[kind] = {
            key: reference[key]
            for key in ("path", "sha256", "offset", "length", "content_sha256")
        }
    for row in benchmark_rows:
        row.update(benchmark_refs[row["sample_id"]])
    benchmark.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in benchmark_rows),
        encoding="utf-8",
    )
    media_manifest = root / "media_manifest.json"
    media_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "row_count": 500,
                "resources": resources,
                "rows": media_rows,
            }
        ),
        encoding="utf-8",
    )
    derivation = root / "derivation_code.py"
    derivation.write_text("# frozen\n", encoding="utf-8")
    leakage = root / "leakage_audit.json"
    bindings = {
        "train_sha256": sha256_file(train),
        "validation_sha256": sha256_file(validation),
        "benchmark_sha256": sha256_file(benchmark),
        "media_manifest_sha256": sha256_file(media_manifest),
    }
    checks = {
        "sample_id": 0,
        "group_id": 0,
        "media_sha256": 0,
        "normalized_question_options": 0,
        "near_duplicate": 0,
    }
    computation = {
        "algorithm": {
            "version": LEAKAGE_ALGORITHM_VERSION,
            "sha256": LEAKAGE_ALGORITHM_SHA256,
        },
        "bindings": bindings,
        "checks": checks,
    }
    leakage.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "status": "passed",
                "algorithm": computation["algorithm"],
                "bindings": bindings,
                "checks": checks,
                "computed_sha256": sha256_json(computation),
            }
        ),
        encoding="utf-8",
    )
    result = {
        "benchmark": benchmark,
        "train": train,
        "validation": validation,
        "media_manifest": media_manifest,
        "derivation_code": derivation,
        "leakage_audit": leakage,
    }
    return result


@pytest.fixture
def controller_batch(tmp_path, monkeypatch):
    # Tests use an explicit controller-owned state directory that is outside
    # the batch workspace while still being cleaned up with tmp_path.
    monkeypatch.setenv(
        "MOTION_EVAL_CONTROLLER_STATE_ROOT", str((tmp_path / "controller-state").resolve())
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "version.py").write_text("VERSION='test'\n", encoding="utf-8")
    runner_root = tmp_path / "runners"
    (runner_root / "scripts").mkdir(parents=True)
    pretrained = tmp_path / "pretrained"
    for relative in (
        "by_model/qwen36_27b_lora/base",
        "by_model/qwen3vl_8b_lora/base",
    ):
        asset = pretrained / relative
        asset.mkdir(parents=True)
        (asset / "weights.bin").write_bytes(relative.encode("ascii"))
    _finetune_worker(tmp_path)
    _verifier_worker(tmp_path)
    for role in ("finetune", "evaluation", "verifier"):
        backend = (
            runner_root
            / "scripts"
            / "backends"
            / "missing"
            / "qwen36_27b_lora"
            / f"{role}.py"
        )
        backend.parent.mkdir(parents=True, exist_ok=True)
        backend.write_text("# fixture backend capability marker\n", encoding="utf-8")
    controller = BatchController(
        tmp_path / "batches",
        registry_path=REPO / "model_evaluation_agent" / "model_registry.json",
        pretrained_registry_path=REPO / "model_evaluation_agent" / "pretrained_registry.json",
        code_root=code,
        runner_root=runner_root,
        pretrained_root=pretrained,
        gpu_probe=_GPUProbe(),
    )
    # Legacy state-machine tests exercise the post-bootstrap controller logic
    # with fixture workers.  They explicitly simulate the future verified
    # multi-root executor; the fail-closed default is covered separately.
    controller._controller_verified_multi_root_bootstrap = lambda **_kwargs: True
    inputs = _write_inputs(tmp_path / "inputs")
    batch_id = "qa500v2_unit_deadbeef"
    controller.create_batch(
        batch_id,
        inputs=inputs,
        config={
            "seed": 7,
            "model_training": {
                model_id: {"training_steps": 2, "preflight_steps": 1}
                for model_id in controller.registry.ids
            },
        },
    )
    return controller, batch_id, inputs


def test_production_runtime_is_blocked_before_python_spawn_and_completion(
    controller_batch, monkeypatch
) -> None:
    controller, batch_id, _ = controller_batch
    controller._controller_verified_multi_root_bootstrap = (
        BatchController._controller_verified_multi_root_bootstrap.__get__(
            controller, BatchController
        )
    )
    calls: list[object] = []

    def forbidden_spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("production Python process was spawned")

    monkeypatch.setattr(batch_module, "run_verified_python", forbidden_spawn)
    controller.create_finetune_attempt(
        batch_id,
        model_id="qwen36_27b_lora",
        attempt_id="blocked_production",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    with pytest.raises(
        ControllerValidationError, match="blocker=verified-multi-root-bootstrap"
    ):
        controller.execute_frozen_attempt(
            batch_id,
            model_id="qwen36_27b_lora",
            stage="finetune",
            attempt_id="blocked_production",
        )
    assert calls == []

    controller._require_verified_multi_root_bootstrap(
        role="finetune", purpose="preflight"
    )
    for role in ("evaluation", "verifier"):
        with pytest.raises(
            ControllerValidationError,
            match="blocker=verified-multi-root-bootstrap",
        ):
            controller._require_verified_multi_root_bootstrap(
                role=role, purpose="production"
            )

    for completion in (
        lambda: controller.complete_finetune(
            batch_id,
            model_id="qwen36_27b_lora",
            attempt_id="blocked_production",
            run_manifest_path=controller.batch_root(batch_id) / "missing.json",
        ),
        lambda: controller.complete_evaluation(
            batch_id,
            model_id="qwen36_27b_lora",
            stage="smoke_1",
            attempt_id="blocked_evaluation",
            predictions_path=controller.batch_root(batch_id) / "missing.jsonl",
        ),
    ):
        with pytest.raises(
            ControllerValidationError,
            match="blocker=verified-multi-root-bootstrap",
        ):
            completion()

    blocked = controller.block_finetune(
        batch_id,
        model_id="qwen36_27b_lora",
        reason_code="unrecoverable_provenance",
        component="verified-multi-root-bootstrap",
        detail=(
            "The controller has no verified multi-root Python bootstrap for "
            "this batch."
        ),
        attempt_id="blocked_production",
    )
    assert blocked["diagnostic"]["diagnostic_type"] == (
        "unavailable_verified_multi_root_bootstrap"
    )
    state = controller.validate_batch(batch_id)
    assert state["models"]["qwen36_27b_lora"]["finetune_status"] == "blocked"


def test_pretrained_content_is_fully_hashed_at_freeze_and_explicit_phase_barriers(
    controller_batch,
) -> None:
    controller, batch_id, _ = controller_batch
    root = controller.batch_root(batch_id)
    receipt = receipt_module.load_and_validate_batch_receipt(
        root / "00_inputs" / "batch_receipt.json"
    )
    frozen = next(
        row
        for rows in receipt["pretrained_assets"].values()
        for row in rows
        if row["state"] == "present"
    )
    assert frozen["content"] == hash_path(
        frozen["path"], symlink_policy="follow"
    ).to_dict()

    asset_path = Path(frozen["path"])
    content_file = asset_path if asset_path.is_file() else next(
        path for path in asset_path.rglob("*") if path.is_file()
    )
    original = content_file.read_bytes()
    replacement = bytes((byte ^ 0xFF) for byte in original)
    assert len(replacement) == len(original) and replacement != original
    content_file.write_bytes(replacement)

    # A normal state transition consumes the HMAC-bound index and performs no
    # underlying asset content hash.  Every explicit phase barrier below must
    # perform the full audit and detect the same-size byte substitution.
    assert controller.state(batch_id)["batch_id"] == batch_id
    for method_name in (
        "validate_batch",
        "open_evaluation",
        "open_full_evaluation",
        "build_release",
        "verify_release",
    ):
        with pytest.raises(BatchReceiptError, match="changed after batch freeze"):
            getattr(controller, method_name)(batch_id)


def test_self_consistent_forged_receipt_and_index_need_external_hmac_state(
    controller_batch,
) -> None:
    controller, batch_id, _ = controller_batch
    root = controller.batch_root(batch_id)
    receipt_path = root / "00_inputs" / "batch_receipt.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged = deepcopy(original)
    forged_index = forged["pretrained_assets_index"]
    forged_index["generated_at"] = "2001-01-01T00:00:00+00:00"
    forged_index_body = {
        key: value for key, value in forged_index.items() if key != "index_sha256"
    }
    forged_index["index_sha256"] = sha256_json(forged_index_body)
    forged["description"] = "self-consistent forged receipt and trusted index"
    forged_body = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = sha256_json(forged_body)
    assert forged["receipt_sha256"] != original["receipt_sha256"]
    atomic_write_json(receipt_path, forged, overwrite=True)

    # All public/internal JSON hashes and path bindings are self-consistent.
    # The controller must still reject it because the external HMAC-protected
    # EventStore remains bound to the original receipt hash.
    validated = receipt_module.load_and_validate_batch_receipt(receipt_path)
    assert validated["receipt_sha256"] == forged["receipt_sha256"]
    with pytest.raises(
        ControllerValidationError,
        match="external HMAC state is bound to a different batch receipt",
    ):
        controller.state(batch_id)


def _verifier_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "runners" / "scripts" / "verify_artifact_reload.py"
    worker.write_text(
        """
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from motion_eval.core import hash_path
parser=argparse.ArgumentParser()
parser.add_argument('--batch-id', required=True); parser.add_argument('--model-id', required=True)
parser.add_argument('--attempt-id', required=True); parser.add_argument('--artifact', required=True)
parser.add_argument('--artifact-sha256', required=True); parser.add_argument('--report', required=True)
args=parser.parse_args()
assert hash_path(args.artifact).digest == args.artifact_sha256
report={'schema_version':'1.0','status':'passed','batch_id':args.batch_id,
        'model_id':args.model_id,'attempt_id':args.attempt_id,
        'artifact_digest':args.artifact_sha256,'checker':f'{args.model_id}:catalog-reload',
        'checked_at':datetime.now(timezone.utc).isoformat()}
Path(args.report).write_text(json.dumps(report, sort_keys=True), encoding='utf-8')
""".strip() + "\n",
        encoding="utf-8",
    )
    return worker


def _finetune_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "runners" / "scripts" / "finetune_qwen36_27b_lora.py"
    if worker.exists():
        return worker
    worker.write_text(
        """
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from motion_eval.core import hash_path, sha256_file, sha256_json
from motion_eval.training_receipt import make_training_receipt, write_training_receipt

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True)
parser.add_argument('--purpose', required=True)
parser.add_argument('--training-steps', type=int, required=True)
args, _ = parser.parse_known_args()
attempt_root = Path(args.output_dir).parent
batch_receipt = json.loads((attempt_root.parents[3] / '00_inputs' / 'batch_receipt.json').read_text(encoding='utf-8'))
attempt = json.loads((attempt_root / 'attempt_receipt.json').read_text(encoding='utf-8'))
model_id = attempt['model_id']
if attempt['attempt_id'] == 'failed_worker':
    raise SystemExit(7)
started = datetime.now(timezone.utc).isoformat()
artifact = attempt_root / 'artifact'
artifact.mkdir()
(artifact / 'adapter.bin').write_bytes(b'fresh-current-batch')
artifact_receipt = {'path': str(artifact.resolve()), **hash_path(artifact).to_dict()}
training = make_training_receipt(
    batch_id=attempt['batch_id'], model_id=model_id, backend_id='fixture_backend',
    model_family='fixture_family', modality='V', training_mode='lora_sft',
    planned_global_steps=args.training_steps, actual_global_steps=args.training_steps,
    planned_optimizer_steps=args.training_steps, actual_optimizer_steps=args.training_steps,
    finite_losses=[1.0], nonzero_finite_gradient_steps=args.training_steps,
    max_gradient=0.5, trainable_tensor_count=1, trainable_parameter_count=1,
    changed_trainable_tensor_count=1, initial_trainable_sha256='1'*64,
    final_trainable_sha256='2'*64, max_parameter_update=0.25,
    batch_receipt_sha256=batch_receipt['receipt_sha256'],
    attempt_sha256=attempt['attempt_sha256'],
    train_sha256=batch_receipt['inputs']['train']['digest'],
    validation_sha256=batch_receipt['inputs']['validation']['digest'],
    leakage_audit_sha256=batch_receipt['inputs']['leakage_audit']['digest'],
    base_artifact_sha256=sha256_json(batch_receipt['pretrained_assets'][model_id]),
    config_sha256=batch_receipt['config_sha256'], code_sha256=batch_receipt['code']['digest'],
    runner_code_sha256=batch_receipt['runner_code']['digest'],
    environment_sha256=batch_receipt['environment_sha256'],
    artifact_sha256=artifact_receipt['digest'])
training_path = attempt_root / 'training_receipt.json'
write_training_receipt(training_path, training, root=attempt_root)
training_reference = {'path': str(training_path.resolve()),
    'file_sha256': sha256_file(training_path), 'content_sha256': training['receipt_sha256']}
finished = datetime.now(timezone.utc).isoformat()
body = {
    'schema_version': '1.0', 'batch_id': attempt['batch_id'], 'model_id': model_id,
    'attempt_id': attempt['attempt_id'], 'purpose': args.purpose, 'status': 'success',
    'exit_code': 0, 'started_at': started, 'finished_at': finished,
    'training_steps': args.training_steps,
    'bindings': {
        'batch_receipt_sha256': batch_receipt['receipt_sha256'],
        'attempt_sha256': attempt['attempt_sha256'], 'command_sha256': attempt['command_sha256'],
        'registry_sha256': batch_receipt['registry']['sha256'],
        'pretrained_registry_sha256': batch_receipt['pretrained_registry']['sha256'],
        'pretrained_assets_sha256': batch_receipt['pretrained_assets_sha256'],
        'model_pretrained_assets_sha256': sha256_json(batch_receipt['pretrained_assets'][model_id]),
        'model_training_config_sha256': sha256_json(batch_receipt['config']['model_training'][model_id]),
        'train_sha256': batch_receipt['inputs']['train']['digest'],
        'validation_sha256': batch_receipt['inputs']['validation']['digest'],
        'leakage_audit_sha256': batch_receipt['inputs']['leakage_audit']['digest'],
        'code_sha256': batch_receipt['code']['digest'],
        'runner_code_sha256': batch_receipt['runner_code']['digest'],
        'config_sha256': batch_receipt['config_sha256'],
        'environment_sha256': batch_receipt['environment_sha256'],
    },
    'artifact': artifact_receipt,
    'training_receipt': training_reference,
}
manifest = {**body, 'manifest_sha256': sha256_json(body)}
(attempt_root / 'run_manifest.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return worker


def _complete_one(controller: BatchController, batch_id: str, model_id: str, tmp_path: Path):
    attempt_id = f"ft_{model_id}"
    attempt = controller.batch_root(batch_id) / "02_finetune" / model_id / "attempts" / attempt_id
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    controller.complete_finetune(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        run_manifest_path=attempt / "run_manifest.json",
    )
    return attempt / "artifact"


def _registered_component(controller: BatchController, model_id: str) -> str:
    return f"pretrained:{controller.registry.pretrained_artifacts[model_id][0].role}"


def test_benchmark_requires_explicit_group_id(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"sample_id": f"s{index:03d}", "gold": "A"}
                if index == 0
                else {"sample_id": f"s{index:03d}", "group_id": f"g{index:03d}", "gold": "A"}
            )
            + "\n"
            for index in range(500)
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrictJsonError, match="explicit group_id"):
        load_benchmark(path)


def test_eval_directory_is_not_created_before_global_barrier(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    root = controller.batch_root(batch_id)
    assert not (root / "03_eval").exists()
    _complete_one(controller, batch_id, "qwen36_27b_lora", tmp_path)
    (root / "status.md").write_text("all models finetune_complete\n", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="barrier"):
        controller.open_evaluation(batch_id)
    assert not (root / "03_eval").exists()


def test_unexecuted_random_artifact_and_fake_manifest_are_rejected(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    model_id = "qwen36_27b_lora"
    with pytest.raises(ControllerValidationError, match="canonical model adapter"):
        controller.create_attempt(
            batch_id,
            model_id=model_id,
            stage="finetune",
            attempt_id="arbitrary_command",
            command=CommandSpec(
                argv=(sys.executable, "-c", "print('fake training')"), cwd=str(tmp_path)
            ),
            purpose="production",
            expected_training_steps=2,
            gpu_binding=controller._gpu_binding(
                controller._receipt(batch_id)[1], "GPU-TEST"
            ),
        )
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id="attempt",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    attempt_root = (
        controller.batch_root(batch_id)
        / "02_finetune"
        / model_id
        / "attempts"
        / "attempt"
    )
    (attempt_root / "artifact").mkdir()
    (attempt_root / "artifact" / "random.bin").write_bytes(b"copied-history")
    fake_manifest = attempt_root / "run_manifest.json"
    fake_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="execution"):
        controller.complete_finetune(
            batch_id,
            model_id=model_id,
            attempt_id="attempt",
            run_manifest_path=fake_manifest,
        )


def test_nonzero_worker_exit_cannot_complete_finetune(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    model_id = "qwen36_27b_lora"
    attempt_id = "failed_worker"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    execution = controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    assert execution["status"] == "failed" and execution["exit_code"] != 0
    attempt_root = (
        controller.batch_root(batch_id)
        / "02_finetune"
        / model_id
        / "attempts"
        / attempt_id
    )
    (attempt_root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="not successfully executed"):
        controller.complete_finetune(
            batch_id,
            model_id=model_id,
            attempt_id=attempt_id,
            run_manifest_path=attempt_root / "run_manifest.json",
        )


def test_artifact_tamper_is_detected(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    artifact = _complete_one(controller, batch_id, "qwen36_27b_lora", tmp_path)
    (artifact / "adapter.bin").write_bytes(b"tampered")
    with pytest.raises(ControllerValidationError, match="artifact receipt"):
        controller.validate_batch(batch_id)


def test_manual_state_cache_tamper_is_detected(controller_batch):
    controller, batch_id, _ = controller_batch
    cache = controller.batch_root(batch_id) / ".controller" / "state.json"
    value = json.loads(cache.read_text(encoding="utf-8"))
    value["eval_open"] = True
    cache.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(StateError, match="hash mismatch"):
        controller.state(batch_id)


def test_recomputed_event_hash_without_controller_hmac_is_rejected(controller_batch):
    controller, batch_id, _ = controller_batch
    event_path = controller.batch_root(batch_id) / ".controller" / "events" / "00000000.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["created_at"] = "2099-01-01T00:00:00+00:00"
    body = {
        key: value
        for key, value in event.items()
        if key not in {"event_sha256", "event_hmac_sha256"}
    }
    event["event_sha256"] = sha256_json(body)
    event_path.write_text(json.dumps(event, sort_keys=True), encoding="utf-8")
    with pytest.raises(StateError, match="HMAC"):
        EventStore(controller.batch_root(batch_id)).load()


def test_batch_tree_signed_prefix_rollback_is_rejected_by_external_head(controller_batch):
    controller, batch_id, _ = controller_batch
    root = controller.batch_root(batch_id)
    store = EventStore(root)
    old_cache = store.cache_path.read_bytes()

    controller.block_finetune(
        batch_id,
        model_id="motionr1_vm_lora",
        reason_code="missing_path",
        component=_registered_component(controller, "motionr1_vm_lora"),
        detail="registered base is not staged",
    )
    assert store.load()["revision"] == 1
    assert not store.anchor_path.is_relative_to(controller.workspace_root)

    # Restore the complete old, correctly HMAC-signed batch-local prefix:
    # GENESIS plus its matching old state cache.  The external head remains at
    # revision 1 and must make both load() and replay() fail closed.
    (store.events_root / "00000001.json").unlink()
    store.cache_path.write_bytes(old_cache)
    with pytest.raises(StateError, match="rollback/replay"):
        store.replay()
    with pytest.raises(StateError, match="rollback/replay"):
        store.load()


def test_replaying_an_older_signed_external_head_is_rejected(controller_batch):
    controller, batch_id, _ = controller_batch
    store = EventStore(controller.batch_root(batch_id))
    old_anchor = store.anchor_path.read_bytes()
    controller.block_finetune(
        batch_id,
        model_id="motionr1_vm_lora",
        reason_code="missing_path",
        component=_registered_component(controller, "motionr1_vm_lora"),
        detail="registered base is not staged",
    )
    assert store.load()["revision"] == 1

    # A stale but valid signed anchor cannot be replayed while the event chain
    # is newer.  Replaying the batch tree and external state together requires
    # same-principal controller-state access and is explicitly out of scope.
    store.anchor_path.write_bytes(old_anchor)
    with pytest.raises(StateError, match="rollback/replay"):
        store.load()


def test_frozen_event_trust_proof_is_non_sensitive_and_honest(controller_batch):
    controller, batch_id, _ = controller_batch
    root, receipt = controller._receipt(batch_id)
    proof = receipt["runtime_contract"]["event_trust"]
    assert set(proof) == {
        "schema_version",
        "key_id",
        "state_scope_id",
        "storage_scope",
        "same_os_principal_protected",
        "protection_capability",
        "threat_model",
    }
    assert proof["storage_scope"] == "external_controller_state"
    assert proof["same_os_principal_protected"] is False
    assert "out of scope" in proof["threat_model"]
    serialized = (root / "00_inputs" / "batch_receipt.json").read_text(encoding="utf-8")
    assert "event_hmac.key" not in serialized
    assert str(EventStore(root).anchor_path) not in serialized


def test_attempt_nonce_rewrite_and_self_rehash_cannot_replay(controller_batch):
    controller, batch_id, _ = controller_batch
    controller.create_finetune_attempt(
        batch_id,
        model_id="qwen36_27b_lora",
        attempt_id="nonce_rewrite",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    receipt_path = (
        controller.batch_root(batch_id)
        / "02_finetune/qwen36_27b_lora/attempts/nonce_rewrite/attempt_receipt.json"
    )
    attempt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempt["lease_nonce"] = "attacker-recomputed-nonce"
    attempt["attempt_sha256"] = sha256_json(
        {key: value for key, value in attempt.items() if key != "attempt_sha256"}
    )
    receipt_path.write_text(json.dumps(attempt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="lease|nonce"):
        controller.execute_frozen_attempt(
            batch_id,
            model_id="qwen36_27b_lora",
            stage="finetune",
            attempt_id="nonce_rewrite",
        )


def test_concurrent_transitions_are_serialized(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    models = ["motionr1_vm_lora", "qwen3vl_4b_lora"]

    def block(model_id: str):
        controller.block_finetune(
            batch_id,
            model_id=model_id,
            reason_code="missing_path",
            component=_registered_component(controller, model_id),
            detail="registered pretrained component is not staged",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(block, models))
    state = controller.state(batch_id)
    assert [state["models"][model]["finetune_status"] for model in models] == [
        "blocked",
        "blocked",
    ]
    assert state["revision"] == 2


def test_blocked_condition_is_revalidated(controller_batch, tmp_path):
    controller, batch_id, _ = controller_batch
    component = _registered_component(controller, "agcn_official")
    controller.block_finetune(
        batch_id,
        model_id="agcn_official",
        reason_code="missing_path",
        component=component,
        detail="registered AGCN source is missing",
    )
    _, receipt = controller._receipt(batch_id)
    missing = Path(
        controller._expected_block_path(
            receipt, "agcn_official", "missing_path", component
        )
    )
    missing.parent.mkdir(parents=True, exist_ok=True)
    missing.write_text("# appeared later\n", encoding="utf-8")
    with pytest.raises(Exception, match="frozen missing|no longer holds"):
        controller.validate_batch(batch_id)
