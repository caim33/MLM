from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from motion_eval.controller import BatchController, ControllerValidationError
from motion_eval.core import sha256_bytes, sha256_file, sha256_json
from motion_eval.data.receipts import (
    BatchReceiptError,
    LEAKAGE_ALGORITHM_SHA256,
    LEAKAGE_ALGORITHM_VERSION,
)
from motion_eval.__main__ import main as cli_main
from motion_eval.runtime import GPUDevice, GPUInventory
import motion_eval.runtime.process as process_module


REPO = Path(__file__).resolve().parents[2]


class _GPUProbe:
    def query(self):
        return GPUInventory(
            (GPUDevice(0, "GPU-TEST", "test", 24000, 0, 0),),
            (),
        )


def _controller(
    tmp_path: Path,
    *,
    with_sibling_imports: bool = False,
    controller_interpreter: str | Path | None = None,
):
    code = tmp_path / "code"
    code.mkdir()
    (code / "controller.py").write_text("# version\n", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    benchmark = inputs / "benchmark.jsonl"
    benchmark_rows = [
        {
            "sample_id": f"s{i:03}", "group_id": f"g{i:03}", "gold": "A",
            "question": f"benchmark question {i}",
            "options": {key: f"benchmark {i} option {key}" for key in "ABCD"},
        }
        for i in range(500)
    ]
    train_video = inputs / "train.video"; train_video.write_bytes(b"train-video")
    train_motion = inputs / "train.motion"; train_motion.write_bytes(b"train-motion")
    val_video = inputs / "validation.video"; val_video.write_bytes(b"validation-video")
    val_motion = inputs / "validation.motion"; val_motion.write_bytes(b"validation-motion")
    train = inputs / "train.jsonl"
    train.write_text(json.dumps({
        "sample_id": "train-1", "group_id": "train-g",
        "gold": "A",
        "question": "unique train question",
        "options": {key: f"train option {key}" for key in "ABCD"},
        "video": str(train_video.resolve()), "motion": str(train_motion.resolve()),
    }) + "\n", encoding="utf-8")
    validation = inputs / "validation.jsonl"
    validation.write_text(json.dumps({
        "sample_id": "val-1", "group_id": "val-g", "split": "val",
        "gold": "B",
        "question": "unique validation question",
        "options": {key: f"validation option {key}" for key in "ABCD"},
        "video": str(val_video.resolve()), "motion": str(val_motion.resolve()),
    }) + "\n", encoding="utf-8")
    media = inputs / "media.container"
    resources = []
    media_rows = []
    chunks = []
    pending = []
    benchmark_refs = {}
    offset = 0
    for i in range(500):
        sample_id = f"s{i:03}"
        linked = []
        for kind in ("video", "motion"):
            payload = f"canonical:{sample_id}:{kind}\n".encode("ascii")
            resource_id = f"{sample_id}:{kind}"
            pending.append((resource_id, kind, offset, payload))
            chunks.append(payload)
            offset += len(payload)
            linked.append(resource_id)
        media_rows.append({"sample_id": sample_id, "resource_ids": linked})
    media.write_bytes(b"".join(chunks))
    container_sha256 = sha256_file(media)
    for resource_id, kind, offset, payload in pending:
        reference = {
            "resource_id": resource_id, "kind": kind,
            "path": str(media.resolve()), "sha256": container_sha256,
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
    media_manifest = inputs / "media_manifest.json"
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
    derivation = inputs / "derivation_code.py"
    derivation.write_text("# frozen\n", encoding="utf-8")
    leakage = inputs / "leakage_audit.json"
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
    paths = {
        "benchmark": benchmark,
        "train": train,
        "validation": validation,
        "media_manifest": media_manifest,
        "derivation_code": derivation,
        "leakage_audit": leakage,
    }
    runner_root = tmp_path / "runners"
    (runner_root / "scripts").mkdir(parents=True)
    pretrained = tmp_path / "pretrained"
    qwen_asset = pretrained / "by_model" / "qwen36_27b_lora" / "base"
    qwen_asset.mkdir(parents=True)
    (qwen_asset / "weights.bin").write_bytes(b"qwen36 canonical base")
    finetune_worker = _finetune_worker(tmp_path)
    _eval_worker(tmp_path)
    verifier_worker = _verifier_worker(tmp_path)
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
    if with_sibling_imports:
        sibling = runner_root / "scripts" / "verified_sibling.py"
        sibling.write_text("VALUE = 'SAFE-SIBLING'\n", encoding="utf-8")
        prefix = "import verified_sibling\nassert verified_sibling.VALUE == 'SAFE-SIBLING'\n"
        for worker in (finetune_worker, verifier_worker):
            worker.write_text(prefix + worker.read_text(encoding="utf-8"), encoding="utf-8")
    controller = BatchController(
        tmp_path / "batches",
        registry_path=REPO / "model_evaluation_agent" / "model_registry.json",
        pretrained_registry_path=REPO / "model_evaluation_agent" / "pretrained_registry.json",
        code_root=code,
        runner_root=runner_root,
        pretrained_root=pretrained,
        controller_interpreter=controller_interpreter,
        gpu_probe=_GPUProbe(),
    )
    # These integration fixtures exercise controller mechanics after a
    # hypothetical verified bootstrap.  Production defaults remain fail-closed
    # and are covered by the dedicated controller security test.
    controller._controller_verified_multi_root_bootstrap = lambda **_kwargs: True
    batch_id = "qa500v2_integration_deadbeef"
    controller.create_batch(
        batch_id,
        inputs=paths,
        config={
            "seed": 0,
            "model_training": {
                model_id: {"training_steps": 2, "preflight_steps": 1}
                for model_id in controller.registry.ids
            },
        },
    )
    return controller, batch_id, paths


def _interpreter_symlink_or_skip(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(Path(sys.executable).resolve(strict=True))
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create an interpreter symlink on this platform: {exc}")
    return path


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
parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', required=True)
parser.add_argument('--purpose', required=True); parser.add_argument('--training-steps', type=int, required=True)
args,_=parser.parse_known_args()
attempt_root=Path(args.output_dir).parent
batch_receipt=json.loads((attempt_root.parents[3]/'00_inputs'/'batch_receipt.json').read_text(encoding='utf-8'))
attempt = json.loads((attempt_root / 'attempt_receipt.json').read_text(encoding='utf-8'))
model_id = attempt['model_id']
reported_steps = args.training_steps + (1 if attempt['attempt_id'] == 'wrong_steps' else 0)
started = datetime.now(timezone.utc).isoformat()
artifact = attempt_root / 'artifact'; artifact.mkdir()
(artifact / 'adapter.bin').write_bytes(b'fresh-production')
artifact_receipt = {'path': str(artifact.resolve()), **hash_path(artifact).to_dict()}
training = make_training_receipt(
    batch_id=attempt['batch_id'], model_id=model_id, backend_id='fixture_backend',
    model_family='fixture_family', modality='V', training_mode='lora_sft',
    planned_global_steps=reported_steps, actual_global_steps=reported_steps,
    planned_optimizer_steps=reported_steps, actual_optimizer_steps=reported_steps,
    finite_losses=[1.0], nonzero_finite_gradient_steps=reported_steps,
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
training_reference = {'path':str(training_path.resolve()),
    'file_sha256':sha256_file(training_path),'content_sha256':training['receipt_sha256']}
finished = datetime.now(timezone.utc).isoformat()
body = {'schema_version':'1.0','batch_id':attempt['batch_id'],'model_id':model_id,
        'attempt_id':attempt['attempt_id'],'purpose':args.purpose,'status':'success','exit_code':0,
        'started_at':started,'finished_at':finished,'training_steps':reported_steps,
        'bindings':{'batch_receipt_sha256':batch_receipt['receipt_sha256'],
          'attempt_sha256':attempt['attempt_sha256'],'command_sha256':attempt['command_sha256'],
          'registry_sha256':batch_receipt['registry']['sha256'],
          'pretrained_registry_sha256':batch_receipt['pretrained_registry']['sha256'],
          'pretrained_assets_sha256':batch_receipt['pretrained_assets_sha256'],
          'model_pretrained_assets_sha256':sha256_json(batch_receipt['pretrained_assets'][model_id]),
          'model_training_config_sha256':sha256_json(batch_receipt['config']['model_training'][model_id]),
          'train_sha256':batch_receipt['inputs']['train']['digest'],
          'validation_sha256':batch_receipt['inputs']['validation']['digest'],
          'leakage_audit_sha256':batch_receipt['inputs']['leakage_audit']['digest'],
          'code_sha256':batch_receipt['code']['digest'],
          'runner_code_sha256':batch_receipt['runner_code']['digest'],
          'config_sha256':batch_receipt['config_sha256'],
          'environment_sha256':batch_receipt['environment_sha256']},
        'artifact':artifact_receipt,'training_receipt':training_reference}
if attempt['attempt_id'] == 'self_claim_only':
    body['reload_verified'] = {'status':'passed','report_path':'worker-owned','report_sha256':'0'*64}
manifest = {**body,'manifest_sha256':sha256_json(body)}
(attempt_root / 'run_manifest.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
""".strip() + "\n",
        encoding="utf-8",
    )
    return worker


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
if args.attempt_id == 'verifier_fails':
    raise SystemExit(9)
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


def _eval_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "runners" / "scripts" / "eval_qwen36_27b_generate.py"
    worker.write_text(
        """
import argparse, json
from pathlib import Path
from motion_eval.evaluation import make_generative_row
parser=argparse.ArgumentParser()
parser.add_argument('--predictions', required=True); parser.add_argument('--benchmark-manifest', required=True)
parser.add_argument('--batch-id', required=True); parser.add_argument('--model-id', required=True)
parser.add_argument('--modality', required=True); parser.add_argument('--limit', type=int)
args,_=parser.parse_known_args()
benchmark=[json.loads(line) for line in Path(args.benchmark_manifest).read_text(encoding='utf-8').splitlines()]
target=Path(args.predictions)
scenario=target.parent.name
count=args.limit if args.limit is not None else 500
if 'partial' in scenario: count=499
rows=[]
for item in benchmark[:count]:
    output='A' if 'all_invalid' in scenario else '<answer>A</answer>'
    rows.append(make_generative_row(batch_id=args.batch_id, model_id=args.model_id,
        sample_id=item['sample_id'], group_id=item['group_id'], modality=args.modality,
        gold=item['gold'], raw_output=output).to_dict())
if 'duplicate' in scenario: rows[-1]=dict(rows[0])
target.write_text(''.join(json.dumps(row,sort_keys=True)+'\\n' for row in rows),encoding='utf-8')
""".strip() + "\n",
        encoding="utf-8",
    )
    return worker


def _fresh_model_and_block_rest(controller, batch_id, tmp_path):
    model_id = "qwen36_27b_lora"
    attempt_root = controller.batch_root(batch_id) / "02_finetune" / model_id / "attempts" / "ft"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id="ft",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id="ft",
    )
    controller.complete_finetune(
        batch_id,
        model_id=model_id,
        attempt_id="ft",
        run_manifest_path=attempt_root / "run_manifest.json",
    )
    for other in controller.registry.ids[1:]:
        component = f"pretrained:{controller.registry.pretrained_artifacts[other][0].role}"
        controller.block_finetune(
            batch_id,
            model_id=other,
            reason_code="missing_path",
            component=component,
            detail="registered pretrained component unavailable in this fixture",
        )
    controller.open_evaluation(batch_id)
    return model_id


def _eval_attempt(controller, batch_id, model_id, stage, count, tmp_path, *, mutate=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenario = tmp_path.name.replace("-", "_")
    attempt_id = f"{stage}_{count}_{scenario}"
    root = controller.batch_root(batch_id) / "03_eval" / model_id / stage / "attempts" / attempt_id
    predictions = root / "predictions.jsonl"
    controller.create_evaluation_attempt(
        batch_id,
        model_id=model_id,
        stage=stage,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage=stage,
        attempt_id=attempt_id,
    )
    return attempt_id, predictions


def _frozen_config(controller, batch_id):
    return controller._receipt(batch_id)[1]["config"]


def _refresh_audit_bindings(paths):
    audit = json.loads(paths["leakage_audit"].read_text(encoding="utf-8"))
    bindings = {
        "train_sha256": sha256_file(paths["train"]),
        "validation_sha256": sha256_file(paths["validation"]),
        "benchmark_sha256": sha256_file(paths["benchmark"]),
        "media_manifest_sha256": sha256_file(paths["media_manifest"]),
    }
    audit["bindings"] = bindings
    computation = {
        "algorithm": audit["algorithm"],
        "bindings": bindings,
        "checks": audit["checks"],
    }
    audit["computed_sha256"] = sha256_json(computation)
    paths["leakage_audit"].write_text(json.dumps(audit), encoding="utf-8")


def test_batch_preserves_one_symlink_interpreter_launcher_for_every_runtime_role(
    tmp_path,
):
    target = Path(sys.executable).resolve(strict=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    launcher = _interpreter_symlink_or_skip(
        tmp_path / "venv-a" / "bin" / f"python{suffix}"
    )
    alternate = _interpreter_symlink_or_skip(
        tmp_path / "venv-b" / "bin" / f"python{suffix}"
    )
    controller, batch_id, _ = _controller(
        tmp_path, controller_interpreter=launcher
    )
    root, receipt = controller._receipt(batch_id)
    runtime_contract = receipt["runtime_contract"]
    assert runtime_contract["interpreter"]["path"] == str(target)
    assert runtime_contract["interpreter"]["launcher_path"] == str(launcher)

    frozen_launchers = {
        role["command_template"]["argv"][0]
        for roles in runtime_contract["models"].values()
        for role in roles.values()
    }
    assert frozen_launchers == {str(launcher)}
    plan = controller.plan(batch_id, python_executable=str(launcher))
    assert {
        command["argv"][0]
        for model in plan["models"]
        for command in (model["finetune"], model["evaluation"])
    } == {str(launcher)}
    with pytest.raises(ControllerValidationError, match="differs from the frozen"):
        controller.plan(batch_id, python_executable=str(alternate))

    # Even a fully rehashed receipt cannot bind one model/role to an alternate
    # launcher that happens to resolve to the same base interpreter target.
    first_model = next(iter(runtime_contract["models"].values()))
    changed_role = first_model["evaluation"]
    changed_role["command_template"]["argv"][0] = str(alternate)
    changed_role["command_template_sha256"] = sha256_json(
        changed_role["command_template"]
    )
    runtime_contract["runtime_contract_sha256"] = sha256_json(
        {
            key: value
            for key, value in runtime_contract.items()
            if key != "runtime_contract_sha256"
        }
    )
    receipt["receipt_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    (root / "00_inputs" / "batch_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(BatchReceiptError, match="different interpreter launcher"):
        controller.state(batch_id)


def test_smoke_sequence_partial_duplicate_full_and_release_tamper(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = _fresh_model_and_block_rest(controller, batch_id, tmp_path)

    with pytest.raises(Exception, match="sequence"):
        _eval_attempt(
            controller, batch_id, model_id, "smoke_8", 8, tmp_path / "wrong-order"
        )

    for size in (1, 8, 32):
        stage = f"smoke_{size}"
        attempt, predictions = _eval_attempt(
            controller, batch_id, model_id, stage, size, tmp_path / stage
        )
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage=stage,
            attempt_id=attempt,
            predictions_path=predictions,
        )
    controller.open_full_evaluation(batch_id)

    partial_attempt, partial = _eval_attempt(
        controller, batch_id, model_id, "full", 499, tmp_path / "partial"
    )
    with pytest.raises(ControllerValidationError, match="denominator"):
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage="full",
            attempt_id=partial_attempt,
            predictions_path=partial,
        )

    def duplicate(rows):
        rows[-1] = dict(rows[0])
        return rows

    duplicate_attempt, duplicate_path = _eval_attempt(
        controller, batch_id, model_id, "full", 500, tmp_path / "duplicate", mutate=duplicate
    )
    with pytest.raises(ControllerValidationError, match="identity"):
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage="full",
            attempt_id=duplicate_attempt,
            predictions_path=duplicate_path,
        )

    attempt, predictions = _eval_attempt(
        controller, batch_id, model_id, "full", 500, tmp_path / "valid"
    )
    controller.complete_evaluation(
        batch_id,
        model_id=model_id,
        stage="full",
        attempt_id=attempt,
        predictions_path=predictions,
    )
    manifest = controller.build_release(batch_id)
    assert manifest["policy"]["historical_results_allowed"] is False
    assert manifest["policy"]["proxy_results_allowed"] is False
    assert manifest["models"][0]["denominator"] == 500
    controller.verify_release(batch_id)
    predictions.write_text(predictions.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="worker output changed"):
        controller.verify_release(batch_id)


def test_p0_input_overlap_and_fake_leakage_leave_no_batch_root(tmp_path):
    controller, _, paths = _controller(tmp_path)
    overlap = dict(paths)
    overlap["train"] = overlap["benchmark"]
    with pytest.raises(Exception, match="train|different"):
        controller.create_batch("qa500v2_bad_overlap", inputs=overlap)
    assert not (controller.workspace_root / "qa500v2_bad_overlap").exists()

    paths["leakage_audit"].write_text('{"status":"passed"}\n', encoding="utf-8")
    with pytest.raises(Exception, match="leakage"):
        controller.create_batch("qa500v2_fake_leakage", inputs=paths)
    assert not (controller.workspace_root / "qa500v2_fake_leakage").exists()


def test_p0_arbitrary_blocker_and_all_invalid_smoke_cannot_release(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    with pytest.raises(ControllerValidationError, match="registered"):
        controller.block_finetune(
            batch_id,
            model_id="motionr1_vm_lora",
            reason_code="missing_path",
            component="pretrained:not-a-real-role",
            detail="caller selected an arbitrary missing path",
        )
    assert controller.state(batch_id)["models"]["motionr1_vm_lora"]["finetune_status"] == "pending"

    model_id = _fresh_model_and_block_rest(controller, batch_id, tmp_path)

    attempt, predictions = _eval_attempt(
        controller,
        batch_id,
        model_id,
        "smoke_1",
        1,
        tmp_path / "all-invalid",
    )
    with pytest.raises(ControllerValidationError, match="strict valid"):
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage="smoke_1",
            attempt_id=attempt,
            predictions_path=predictions,
        )
    assert controller.state(batch_id)["models"][model_id]["smoke"]["1"] == "pending"
    with pytest.raises(ControllerValidationError):
        controller.build_release(batch_id)


def test_p0_controller_recomputes_semantic_leakage_instead_of_trusting_zeroes(tmp_path):
    controller, batch_id, paths = _controller(tmp_path)
    config = _frozen_config(controller, batch_id)
    original_train = json.loads(paths["train"].read_text(encoding="utf-8"))
    leaked = {
        "sample_id": "s000",
        "group_id": "g000",
        "gold": original_train["gold"],
        "question": "benchmark question 0",
        "options": {key: f"benchmark 0 option {key}" for key in "ABCD"},
        "video": original_train["video"],
        "motion": original_train["motion"],
    }
    paths["train"].write_text(json.dumps(leaked) + "\n", encoding="utf-8")
    _refresh_audit_bindings(paths)  # Deliberately retains caller-declared zero counts.
    new_id = "qa500v2_semantic_leak_rejected"
    with pytest.raises(Exception, match="recomputation|leakage"):
        controller.create_batch(
            new_id, inputs=paths, config=config
        )
    assert not (controller.workspace_root / new_id).exists()


def test_p0_media_requires_one_video_and_motion_identity_per_benchmark_row(tmp_path):
    controller, batch_id, paths = _controller(tmp_path)
    config = _frozen_config(controller, batch_id)
    manifest = json.loads(paths["media_manifest"].read_text(encoding="utf-8"))
    manifest["resources"] = [
        resource for resource in manifest["resources"] if resource["kind"] == "video"
    ]
    for row in manifest["rows"]:
        row["resource_ids"] = [item for item in row["resource_ids"] if item.endswith(":video")]
    paths["media_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_audit_bindings(paths)
    new_id = "qa500v2_missing_motion_rejected"
    with pytest.raises(Exception, match="video.*motion|motion|each"):
        controller.create_batch(
            new_id, inputs=paths, config=config
        )
    assert not (controller.workspace_root / new_id).exists()


def test_p0_preflight_limit_and_missing_pretrain_cannot_complete_production(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    with pytest.raises(ControllerValidationError, match="production.*limit"):
        controller.create_finetune_attempt(
            batch_id,
            model_id="qwen36_27b_lora",
            attempt_id="bad_production_limit",
            python_executable=sys.executable,
            gpu="GPU-TEST",
            purpose="production",
            limit=1,
        )
    controller.create_finetune_attempt(
        batch_id,
        model_id="qwen36_27b_lora",
        attempt_id="preflight_limit1",
        python_executable=sys.executable,
        gpu="GPU-TEST",
        purpose="preflight",
        limit=1,
    )
    controller.execute_frozen_attempt(
        batch_id,
        model_id="qwen36_27b_lora",
        stage="finetune",
        attempt_id="preflight_limit1",
    )
    manifest = (
        controller.batch_root(batch_id)
        / "02_finetune/qwen36_27b_lora/attempts/preflight_limit1/run_manifest.json"
    )
    with pytest.raises(ControllerValidationError, match="production attempt"):
        controller.complete_finetune(
            batch_id,
            model_id="qwen36_27b_lora",
            attempt_id="preflight_limit1",
            run_manifest_path=manifest,
        )
    with pytest.raises(ControllerValidationError, match="not ready"):
        controller.create_finetune_attempt(
            batch_id,
            model_id="motionr1_vm_lora",
            attempt_id="missing_base",
            python_executable=sys.executable,
            gpu="GPU-TEST",
        )


def test_p0_manifest_cannot_overreport_frozen_production_steps(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    attempt_id = "wrong_steps"
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
    manifest = (
        controller.batch_root(batch_id)
        / "02_finetune"
        / model_id
        / "attempts"
        / attempt_id
        / "run_manifest.json"
    )
    with pytest.raises(ControllerValidationError, match="production training"):
        controller.complete_finetune(
            batch_id,
            model_id=model_id,
            attempt_id=attempt_id,
            run_manifest_path=manifest,
        )


def test_p0_runtime_failure_is_retryable_and_cannot_become_terminal_blocker(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    with pytest.raises(ControllerValidationError, match="blocked reason"):
        controller.block_finetune(
            batch_id,
            model_id="motionr1_vm_lora",
            reason_code="incompatible_environment",
            component="environment",
            detail="caller-authored blocker",
        )
    with pytest.raises(ControllerValidationError, match="missing|retryable"):
        controller.create_finetune_attempt(
            batch_id,
            model_id="qwen36_27b_lora",
            attempt_id="missing_interpreter",
            python_executable=str(tmp_path / "missing-python.exe"),
            gpu="GPU-TEST",
        )
    with pytest.raises(ControllerValidationError, match="blocked reason"):
        controller.block_finetune(
            batch_id,
            model_id="qwen36_27b_lora",
            reason_code="command_failed",
            component="runner",
            detail="runtime failures remain retryable",
            attempt_id="missing_interpreter",
        )
    assert controller.state(batch_id)["models"]["qwen36_27b_lora"]["finetune_status"] == "pending"


def test_missing_catalog_backend_is_a_verifiable_component_blocker(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    evidence = controller.block_finetune(
        batch_id,
        model_id="motionr1_vm_lora",
        reason_code="missing_code",
        component="backend:finetune",
        detail="reviewed production finetune backend is not installed",
    )
    assert evidence["diagnostic"]["component"] == "backend:finetune"
    assert evidence["diagnostic"]["observed_state"] == "missing"
    assert evidence["diagnostic"]["expected_path"].replace("\\", "/").endswith(
        "scripts/backends/missing/motionr1_vm_lora/finetune.py"
    )


def test_p0_competing_batch_creator_never_deletes_winner(tmp_path):
    controller, batch_id, paths = _controller(tmp_path)
    config = _frozen_config(controller, batch_id)
    new_id = "qa500v2_concurrent_create"
    entered = threading.Event()
    release = threading.Event()
    from motion_eval.controller import batch as batch_module

    original = batch_module.create_batch_receipt
    first = True
    first_lock = threading.Lock()

    def delayed(*args, **kwargs):
        nonlocal first
        with first_lock:
            should_wait = first
            first = False
        if should_wait:
            entered.set()
            assert release.wait(timeout=10)
        return original(*args, **kwargs)

    with patch.object(batch_module, "create_batch_receipt", delayed):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            winner = pool.submit(
                controller.create_batch,
                new_id,
                inputs=paths,
                config=config,
            )
            assert entered.wait(timeout=10)
            loser = pool.submit(
                controller.create_batch,
                new_id,
                inputs=paths,
                config=config,
            )
            with pytest.raises(FileExistsError):
                loser.result(timeout=10)
            assert (controller.workspace_root / new_id).is_dir()
            release.set()
            winner.result(timeout=30)
    assert controller.validate_batch(new_id)["phase"] == "finetune"


def test_p0_unbound_media_claim_and_decoy_benchmark_reference_are_rejected(tmp_path):
    claim_root = tmp_path / "claim"; claim_root.mkdir()
    controller, batch_id, paths = _controller(claim_root)
    config = _frozen_config(controller, batch_id)
    train = json.loads(paths["train"].read_text(encoding="utf-8"))
    train.pop("video"); train.pop("motion")
    train["media_sha256"] = "0" * 64
    paths["train"].write_text(json.dumps(train) + "\n", encoding="utf-8")
    _refresh_audit_bindings(paths)
    with pytest.raises(Exception, match="unbound|actual.*reference"):
        controller.create_batch(
            "qa500v2_unbound_media_claim", inputs=paths, config=config
        )

    decoy_root = tmp_path / "decoy"; decoy_root.mkdir()
    controller, batch_id, paths = _controller(decoy_root)
    config = _frozen_config(controller, batch_id)
    decoy = decoy_root / "decoy.video"; decoy.write_bytes(b"decoy-video")
    rows = [json.loads(line) for line in paths["benchmark"].read_text(encoding="utf-8").splitlines()]
    rows[0]["video"] = {"path": str(decoy.resolve()), "sha256": sha256_file(decoy)}
    paths["benchmark"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _refresh_audit_bindings(paths)
    with pytest.raises(Exception, match="canonical linked media resource"):
        controller.create_batch(
            "qa500v2_decoy_media_rejected", inputs=paths, config=config
        )


def test_p0_worker_dir_self_hashes_cannot_forge_finetune_or_eval_receipts(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id="forged_finetune",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    ft_root = controller.batch_root(batch_id) / "02_finetune" / model_id / "attempts" / "forged_finetune"
    fake_body = {"status": "success", "output_sha256": "0" * 64}
    (ft_root / "execution_receipt.json").write_text(
        json.dumps({**fake_body, "execution_sha256": sha256_json(fake_body)}),
        encoding="utf-8",
    )
    (ft_root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="anchored|run-attempt"):
        controller.complete_finetune(
            batch_id,
            model_id=model_id,
            attempt_id="forged_finetune",
            run_manifest_path=ft_root / "run_manifest.json",
        )

    model_id = _fresh_model_and_block_rest(controller, batch_id, tmp_path)
    created = controller.create_evaluation_attempt(
        batch_id,
        model_id=model_id,
        stage="smoke_1",
        attempt_id="forged_eval",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    argv = created["command"]["argv"]
    receipt = controller._receipt(batch_id)[1]
    assert argv[argv.index("--media-manifest") + 1] == receipt["inputs"]["media_manifest"]["path"]
    assert argv[argv.index("--media-manifest-sha256") + 1] == receipt["inputs"]["media_manifest"]["digest"]
    eval_root = controller.batch_root(batch_id) / "03_eval" / model_id / "smoke_1" / "attempts" / "forged_eval"
    (eval_root / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (eval_root / "execution_receipt.json").write_text(
        json.dumps({**fake_body, "execution_sha256": sha256_json(fake_body)}),
        encoding="utf-8",
    )
    with pytest.raises(ControllerValidationError, match="anchored|run-attempt"):
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage="smoke_1",
            attempt_id="forged_eval",
            predictions_path=eval_root / "predictions.jsonl",
        )


def test_p0_worker_reload_claim_cannot_replace_catalog_verifier(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    for attempt_id, expected in (
        ("verifier_fails", "passed controller-launched verifier"),
        ("self_claim_only", "manifest schema"),
    ):
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
        assert execution["status"] == "success"
        manifest = controller.batch_root(batch_id) / "02_finetune" / model_id / "attempts" / attempt_id / "run_manifest.json"
        with pytest.raises(ControllerValidationError, match=expected):
            controller.complete_finetune(
                batch_id,
                model_id=model_id,
                attempt_id=attempt_id,
                run_manifest_path=manifest,
            )


def test_p1_runner_and_verifier_path_swap_at_spawn_execute_frozen_bytes(
    tmp_path, monkeypatch
):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    attempt_id = "spawn_path_swap"
    marker = tmp_path / "malicious_path_executed.txt"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    original_run = process_module.subprocess.run
    swapped: list[str] = []

    def replace_original_path_during_spawn(argv, **kwargs):
        if len(argv) >= 5 and argv[1:3] == ["-I", "-c"]:
            script = Path(argv[4])
            if script.name in {
                "finetune_qwen36_27b_lora.py",
                "verify_artifact_reload.py",
            }:
                source = script.read_bytes()
                script.write_text(
                    f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                    encoding="utf-8",
                )
                swapped.append(script.name)
                try:
                    return original_run(argv, **kwargs)
                finally:
                    script.write_bytes(source)
        return original_run(argv, **kwargs)

    monkeypatch.setattr(
        process_module.subprocess, "run", replace_original_path_during_spawn
    )
    execution = controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    assert execution["status"] == "success"
    assert set(swapped) == {
        "finetune_qwen36_27b_lora.py",
        "verify_artifact_reload.py",
    }
    assert not marker.exists()
    manifest = (
        controller.batch_root(batch_id)
        / "02_finetune"
        / model_id
        / "attempts"
        / attempt_id
        / "run_manifest.json"
    )
    controller.complete_finetune(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        run_manifest_path=manifest,
    )


def test_p1_runner_changed_at_verified_execution_entry_is_rejected(
    tmp_path, monkeypatch
):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    attempt_id = "entry_path_swap"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    from motion_eval.controller import batch as batch_module

    original_verified_run = batch_module.run_verified_python

    def replace_before_verified_read(spec, **kwargs):
        script = Path(spec.argv[1])
        source = script.read_bytes()
        script.write_text("print('replacement')\n", encoding="utf-8")
        try:
            return original_verified_run(spec, **kwargs)
        finally:
            script.write_bytes(source)

    monkeypatch.setattr(
        batch_module, "run_verified_python", replace_before_verified_read
    )
    execution = controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    assert execution["status"] == "failed"
    assert execution["process_started"] is False
    assert execution["error_code"] == "runtime_error"


def test_p1_controller_transitive_import_swap_uses_verified_bundle(
    tmp_path, monkeypatch
):
    controller, batch_id, _ = _controller(tmp_path, with_sibling_imports=True)
    model_id = "qwen36_27b_lora"
    attempt_id = "transitive_import_swap"
    marker = tmp_path / "malicious_sibling_executed.txt"
    sibling = Path(controller.runner_root) / "scripts" / "verified_sibling.py"
    original_sibling = sibling.read_bytes()
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    original_run = process_module.subprocess.run
    attacks = 0

    def replace_sibling_during_spawn(argv, **kwargs):
        nonlocal attacks
        if len(argv) >= 7 and argv[1:3] == ["-I", "-c"]:
            sibling.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ATTACKED')\n"
                "VALUE = 'MALICIOUS-SIBLING'\n",
                encoding="utf-8",
            )
            attacks += 1
            try:
                return original_run(argv, **kwargs)
            finally:
                sibling.write_bytes(original_sibling)
        return original_run(argv, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "run", replace_sibling_during_spawn)
    execution = controller.execute_frozen_attempt(
        batch_id,
        model_id=model_id,
        stage="finetune",
        attempt_id=attempt_id,
    )
    assert execution["status"] == "success"
    assert attacks == 2  # finetune worker and independent reload verifier
    assert not marker.exists()
    assert not (sibling.parent / "__pycache__").exists()
    manifest = (
        controller.batch_root(batch_id)
        / "02_finetune"
        / model_id
        / "attempts"
        / attempt_id
        / "run_manifest.json"
    )
    controller.complete_finetune(
        batch_id,
        model_id=model_id,
        attempt_id=attempt_id,
        run_manifest_path=manifest,
    )


def test_p0_keepalive_blocks_launch_and_stale_smoke_lease_cannot_run(tmp_path):
    controller, batch_id, _ = _controller(tmp_path)
    model_id = "qwen36_27b_lora"
    controller.create_finetune_attempt(
        batch_id,
        model_id=model_id,
        attempt_id="gpu_retry",
        python_executable=sys.executable,
        gpu="GPU-TEST",
    )
    lifecycle = controller.keepalive_root / "GPU-TEST.reservation"
    lifecycle.write_text("occupied", encoding="utf-8")
    with pytest.raises(ControllerValidationError, match="keepalive|retryable"):
        controller.execute_frozen_attempt(
            batch_id,
            model_id=model_id,
            stage="finetune",
            attempt_id="gpu_retry",
        )
    assert controller.state(batch_id)["attempts"][sha256_json([model_id, "finetune", "gpu_retry"])]["status"] == "leased"
    lifecycle.unlink()
    controller.execute_frozen_attempt(
        batch_id, model_id=model_id, stage="finetune", attempt_id="gpu_retry"
    )
    manifest = controller.batch_root(batch_id) / "02_finetune" / model_id / "attempts" / "gpu_retry" / "run_manifest.json"
    controller.complete_finetune(
        batch_id, model_id=model_id, attempt_id="gpu_retry", run_manifest_path=manifest
    )
    for other in controller.registry.ids[1:]:
        controller.block_finetune(
            batch_id,
            model_id=other,
            reason_code="missing_path",
            component=f"pretrained:{controller.registry.pretrained_artifacts[other][0].role}",
            detail="registered component unavailable in fixture",
        )
    controller.open_evaluation(batch_id)
    for attempt_id in ("stale", "winner"):
        controller.create_evaluation_attempt(
            batch_id,
            model_id=model_id,
            stage="smoke_1",
            attempt_id=attempt_id,
            python_executable=sys.executable,
            gpu="GPU-TEST",
        )
    controller.execute_frozen_attempt(
        batch_id, model_id=model_id, stage="smoke_1", attempt_id="winner"
    )
    winner = controller.batch_root(batch_id) / "03_eval" / model_id / "smoke_1" / "attempts" / "winner" / "predictions.jsonl"
    controller.complete_evaluation(
        batch_id, model_id=model_id, stage="smoke_1", attempt_id="winner", predictions_path=winner
    )
    with pytest.raises(Exception, match="sequence"):
        controller.execute_frozen_attempt(
            batch_id, model_id=model_id, stage="smoke_1", attempt_id="stale"
        )


@patch(
    "motion_eval.controller.batch.NvidiaSmiProbe.query",
    return_value=_GPUProbe().query(),
)
def test_p1_cli_dry_run_uses_real_limit_and_stage_without_placeholders(
    _probe_query, tmp_path, capsys
):
    controller, batch_id, _ = _controller(tmp_path)
    common = [
        "--workspace-root", str(controller.workspace_root),
        "--registry", str(controller.registry.registry_path),
        "--pretrained-registry", str(controller.registry.pretrained_registry_path),
        "--code-root", str(controller.code_root),
        "--runner-root", str(controller.runner_root),
        "--pretrained-root", str(controller.pretrained_root),
    ]
    assert cli_main([
        "finetune", "attempt", *common, batch_id,
        "--model-id", "qwen36_27b_lora", "--attempt-id", "cli_preflight",
        "--python-executable", sys.executable, "--purpose", "preflight",
        "--limit", "7", "--gpu", "GPU-TEST", "--dry-run",
    ]) == 0
    finetune = json.loads(capsys.readouterr().out)
    argv = finetune["command"]["argv"]
    assert argv[argv.index("--limit") + 1] == "7"
    assert argv[argv.index("--purpose") + 1] == "preflight"

    model_id = _fresh_model_and_block_rest(controller, batch_id, tmp_path)
    assert cli_main([
        "eval", "attempt", *common, batch_id,
        "--model-id", model_id, "--attempt-id", "cli_smoke32_rejected",
        "--stage", "smoke_32", "--python-executable", sys.executable,
        "--gpu", "GPU-TEST", "--dry-run",
    ]) == 2
    assert "sequence" in capsys.readouterr().err
    for size in (1, 8):
        stage = f"smoke_{size}"
        attempt, predictions = _eval_attempt(
            controller, batch_id, model_id, stage, size, tmp_path / f"cli_{stage}"
        )
        controller.complete_evaluation(
            batch_id,
            model_id=model_id,
            stage=stage,
            attempt_id=attempt,
            predictions_path=predictions,
        )
    assert cli_main([
        "eval", "attempt", *common, batch_id,
        "--model-id", model_id, "--attempt-id", "cli_smoke32",
        "--stage", "smoke_32", "--python-executable", sys.executable,
        "--gpu", "GPU-TEST", "--dry-run",
    ]) == 0
    evaluation = json.loads(capsys.readouterr().out)
    argv = evaluation["command"]["argv"]
    assert argv[argv.index("--limit") + 1] == "32"
    assert "<stage>" not in " ".join(argv)
