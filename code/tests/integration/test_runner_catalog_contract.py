from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from motion_eval.adapters import build_adapter_catalog
from motion_eval.adapters.catalog import FROZEN_ADAPTER_SPECS
from motion_eval.controller import load_canonical_registry
from motion_eval.core import (
    formal_source_role_files,
    hash_path,
    sha256_file,
    sha256_json,
)


REPO = Path(__file__).resolve().parents[2]
RUNNER_ROOT = REPO / "model_evaluation_agent"


def _runner_specs():
    path = RUNNER_ROOT / "scripts" / "runner_specs.py"
    spec = importlib.util.spec_from_file_location("integration_runner_specs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_support(monkeypatch):
    scripts = RUNNER_ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "runner_support.py"
    spec = importlib.util.spec_from_file_location("integration_runner_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("facade", "arguments"),
    [
        ("eval_qwen3vl_generate.py", ()),
        ("verify_artifact_reload.py", ()),
        ("finetune_qwen3vl_lora.py", ("--purpose", "production")),
    ],
)
def test_production_facade_blocks_before_shadow_motion_eval_import(
    tmp_path: Path, facade: str, arguments: tuple[str, ...]
) -> None:
    marker = tmp_path / "shadow-imported.txt"
    shadow_package = tmp_path / "shadow" / "motion_eval"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(shadow_package.parent)
    result = subprocess.run(
        (
            sys.executable,
            str(RUNNER_ROOT / "scripts" / facade),
            *arguments,
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "blocker=verified-multi-root-bootstrap" in result.stderr
    assert not marker.exists()


def _catalog_formal_snapshot_case(tmp_path):
    attempt_root = tmp_path / "attempt"
    artifact = attempt_root / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "adapter.bin").write_bytes(b"adapter")
    training_path = attempt_root / "training_receipt.json"
    training_path.write_text("{}\n", encoding="utf-8")
    snapshot_path = attempt_root / "formal_provenance_snapshot.json"
    snapshot_path.write_text("{}\n", encoding="utf-8")

    code_root = tmp_path / "code"
    runner_root = code_root / "qwenvl"
    environment_root = tmp_path / "venv"
    base_environment_root = tmp_path / "base_python"
    for directory in (runner_root, environment_root, base_environment_root):
        directory.mkdir(parents=True)
    base_artifact = tmp_path / "base" / "config.json"
    base_artifact.parent.mkdir()
    base_artifact.write_text("{}\n", encoding="utf-8")

    role_paths = {
        "base_artifact": base_artifact,
        "train_data": tmp_path / "train.jsonl",
        "validation_data": tmp_path / "validation.jsonl",
        "benchmark": tmp_path / "benchmark.jsonl",
        "leakage_audit": tmp_path / "leakage.json",
        "config": tmp_path / "config.json",
        "code": code_root,
        "runner_code": runner_root,
        "environment": environment_root,
    }
    for role, path in role_paths.items():
        if not path.exists():
            path.write_text(f"{role}\n", encoding="utf-8")
    provenance = {
        role: {
            "path": str(path.resolve()),
            "algorithm": "test-sha256-v1",
            "kind": "directory" if path.is_dir() else "file",
            "digest": _digest(role),
            "file_count": 1,
            "total_bytes": 1,
        }
        for role, path in role_paths.items()
    }
    snapshot_sha256 = _digest("snapshot")
    snapshot = {
        "schema_version": "motionllm-inprocess-provenance-v2",
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": "batch_1",
        "model_id": "qwen3vl_4b_lora",
        "training_mode": "lora_sft",
        "canonical_identity": {
            "model_id": "qwen3vl_4b_lora",
            "model_family": "qwen3_vl",
            "modality": "V",
            "base_artifact": provenance["base_artifact"]["path"],
            "code_root": str(code_root.resolve()),
            "runner_code_root": str(runner_root.resolve()),
            "environment_root": str(environment_root.resolve()),
            "interpreter": str((environment_root / "python").resolve()),
            "base_environment_root": str(base_environment_root.resolve()),
        },
        "provenance": provenance,
        "manifests": {},
        "snapshot_sha256": snapshot_sha256,
    }
    training = {
        "schema_version": "2.0",
        "batch_id": "batch_1",
        "model_id": "qwen3vl_4b_lora",
        "training_mode": "lora_sft",
        "provenance_snapshot_path": str(snapshot_path.resolve()),
        "provenance_snapshot_file_sha256": sha256_file(snapshot_path),
        "provenance_pre_sha256": snapshot_sha256,
        "provenance_post_sha256": snapshot_sha256,
        "provenance_unchanged": True,
        "base_artifact_sha256": provenance["base_artifact"]["digest"],
        "train_sha256": provenance["train_data"]["digest"],
        "validation_sha256": provenance["validation_data"]["digest"],
        "leakage_audit_sha256": provenance["leakage_audit"]["digest"],
        "config_sha256": provenance["config"]["digest"],
        "code_sha256": provenance["code"]["digest"],
        "runner_code_sha256": provenance["runner_code"]["digest"],
        "environment_sha256": provenance["environment"]["digest"],
    }
    manifest = {
        "bindings": {
            "model_pretrained_assets_sha256": provenance["base_artifact"]["digest"],
            "train_sha256": provenance["train_data"]["digest"],
            "validation_sha256": provenance["validation_data"]["digest"],
            "leakage_audit_sha256": provenance["leakage_audit"]["digest"],
            "config_sha256": provenance["config"]["digest"],
            "code_sha256": provenance["code"]["digest"],
            "runner_code_sha256": provenance["runner_code"]["digest"],
            "environment_sha256": provenance["environment"]["digest"],
        }
    }
    batch_receipt = {
        "inputs": {
            input_role: dict(provenance[snapshot_role])
            for snapshot_role, input_role in {
                "train_data": "train",
                "validation_data": "validation",
                "benchmark": "benchmark",
                "leakage_audit": "leakage_audit",
            }.items()
        },
        "code": {
            "path": provenance["code"]["path"],
            "digest": provenance["code"]["digest"],
        },
        "runner_code": {
            "path": provenance["runner_code"]["path"],
            "digest": provenance["runner_code"]["digest"],
        },
        "config_sha256": provenance["config"]["digest"],
        "environment_sha256": provenance["environment"]["digest"],
        "pretrained_assets": {
            "qwen3vl_4b_lora": [
                {
                    "path": provenance["base_artifact"]["path"],
                    "state": "present",
                    "content": {
                        key: provenance["base_artifact"][key]
                        for key in (
                            "algorithm",
                            "kind",
                            "digest",
                            "file_count",
                            "total_bytes",
                        )
                    },
                }
            ]
        },
    }
    return {
        "attempt_root": attempt_root.resolve(),
        "artifact": artifact.resolve(),
        "training_path": training_path.resolve(),
        "snapshot_path": snapshot_path.resolve(),
        "snapshot": snapshot,
        "training": training,
        "manifest": manifest,
        "batch_receipt": batch_receipt,
    }


def _validate_catalog_snapshot(runner_support, case):
    return runner_support._validate_catalog_formal_snapshot(
        attempt_root=case["attempt_root"],
        artifact=case["artifact"],
        training_path=case["training_path"],
        batch_receipt=case["batch_receipt"],
        artifact_manifest=case["manifest"],
        training_receipt=case["training"],
        model_id="qwen3vl_4b_lora",
    )


def test_controller_receipt_paths_and_runner_imports_have_one_backend_identity():
    registry = load_canonical_registry(
        RUNNER_ROOT / "model_registry.json",
        RUNNER_ROOT / "pretrained_registry.json",
    )
    catalog = build_adapter_catalog(registry)
    runner_specs = _runner_specs()

    for model_id, descriptor in catalog.items():
        frozen = FROZEN_ADAPTER_SPECS[model_id]
        for role in ("finetune", "evaluation", "verifier"):
            receipt_path = descriptor.backend_for(role)
            module_name = runner_specs.backend_for(model_id, role)
            if module_name is None:
                assert receipt_path == (
                    f"scripts/backends/missing/{model_id}/{role}.py"
                )
                assert not (RUNNER_ROOT / receipt_path).exists()
            else:
                assert receipt_path == f"scripts/{module_name.replace('.', '/')}.py"
                assert receipt_path == frozen.implemented_backend_for(role)
                assert (RUNNER_ROOT / receipt_path).is_file()


def test_runner_support_rejects_project_modules_from_unfrozen_roots(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    receipt = {
        "runner_code": {"path": str((RUNNER_ROOT / "scripts").resolve())},
        "code": {"path": str((REPO / "src" / "motion_eval").resolve())},
    }
    runner_support._verify_project_code_origins(receipt)

    receipt["runner_code"] = {"path": str(tmp_path.resolve())}
    with pytest.raises(RuntimeError, match="not imported from frozen runner_code"):
        runner_support._verify_project_code_origins(receipt)


def test_runner_support_rejects_snapshot_validator_from_unfrozen_root(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    receipt = {
        "runner_code": {"path": str((RUNNER_ROOT / "scripts").resolve())},
        "code": {"path": str((REPO / "src" / "motion_eval").resolve())},
    }
    spoofed = tmp_path / "training_receipt.py"
    spoofed.write_text("# spoofed validator\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_support.training_receipt_module, "__file__", str(spoofed)
    )

    with pytest.raises(RuntimeError, match="snapshot validator.*frozen controller code"):
        runner_support._verify_project_code_origins(receipt)


def test_runner_support_rejects_source_inventory_verifier_from_unfrozen_root(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    receipt = {
        "runner_code": {"path": str((RUNNER_ROOT / "scripts").resolve())},
        "code": {"path": str((REPO / "src" / "motion_eval").resolve())},
    }
    spoofed = tmp_path / "source_inventory.py"
    spoofed.write_text("# spoofed inventory verifier\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_support.source_inventory_module, "__file__", str(spoofed)
    )

    with pytest.raises(RuntimeError, match="source inventory verifier.*frozen controller code"):
        runner_support._verify_project_code_origins(receipt)


def test_catalog_reload_follows_snapshot_but_blocks_aliased_qwen_code_roles(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    calls = []

    def strict_loader(path, *, expected_file_sha256, expected):
        calls.append((path, expected_file_sha256, expected))
        return case["snapshot"]

    monkeypatch.setattr(
        runner_support, "load_and_validate_formal_provenance_snapshot", strict_loader
    )
    with pytest.raises(ValueError, match="controller_code/catalog_runner_code"):
        _validate_catalog_snapshot(runner_support, case)
    assert calls == [
        (
            case["snapshot_path"],
            case["training"]["provenance_snapshot_file_sha256"],
            {
                "batch_id": "batch_1",
                "model_id": "qwen3vl_4b_lora",
                "training_mode": "lora_sft",
                "snapshot_sha256": case["training"]["provenance_pre_sha256"],
            },
        )
    ]


def test_catalog_real_snapshot_rejects_incompatible_controller_and_training_roles(
    monkeypatch, tmp_path
):
    from tests.unit.test_sft_snapshot_security import (
        _build_environment_manifest,
        _plain_evidence,
        _write_bytes,
    )

    runner_support = _runner_support(monkeypatch)
    attempt_root = tmp_path / "attempt"
    artifact = attempt_root / "artifact"
    _write_bytes(artifact / "adapter.bin", b"adapter")
    training_path = _write_bytes(attempt_root / "training_receipt.json", b"{}\n")
    snapshot_path = attempt_root / "formal_provenance_snapshot.json"

    code_root = REPO.resolve()
    training_runner_root = (REPO / "qwenvl").resolve()
    environment_root = (tmp_path / "venv").resolve()
    base_environment_root = (tmp_path / "base-python").resolve()
    interpreter = _write_bytes(
        environment_root
        / ("Scripts/python.exe" if runner_support.os.name == "nt" else "bin/python"),
        b"python-interpreter",
    )
    base_environment_root.mkdir(parents=True)

    source_manifests = {}
    source_evidence = {}
    for role in ("code", "runner_code"):
        root, records = formal_source_role_files(
            code_root, runner_root=training_runner_root, role=role
        )
        body = {
            "schema_version": "motionllm-source-allowlist-v2",
            "role": role,
            "root": str(root),
            "files": list(records),
        }
        source_manifests[role] = {
            **body,
            "manifest_sha256": sha256_json(body),
        }
        source_evidence[role] = {
            "path": str(root),
            "algorithm": "motionllm-source-allowlist-v2",
            "kind": "source-allowlist",
            "digest": source_manifests[role]["manifest_sha256"],
            "file_count": len(records),
            "total_bytes": sum(int(row["size"]) for row in records),
        }

    environment_manifest, environment_evidence, _ = _build_environment_manifest(
        tmp_path,
        code_root=code_root,
        runner_root=training_runner_root,
        environment_root=environment_root,
        base_root=base_environment_root,
        interpreter=interpreter,
        attack=None,
    )
    plain_paths = {
        role: _write_bytes(
            tmp_path / "inputs" / f"{role}.bin", role.encode("utf-8")
        )
        for role in (
            "base_artifact",
            "train_data",
            "validation_data",
            "benchmark",
            "leakage_audit",
            "config",
        )
    }
    provenance = {
        role: _plain_evidence(path) for role, path in plain_paths.items()
    }
    provenance.update(source_evidence)
    provenance["environment"] = environment_evidence
    snapshot_body = {
        "schema_version": "motionllm-inprocess-provenance-v2",
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": "batch_1",
        "model_id": "motionr1_vm_lora",
        "training_mode": "full_sft",
        "canonical_identity": {
            "model_id": "motionr1_vm_lora",
            "model_family": "qwen3_vl_motion",
            "modality": "VM",
            "base_artifact": str(plain_paths["base_artifact"]),
            "code_root": str(code_root),
            "runner_code_root": str(training_runner_root),
            "environment_root": str(environment_root),
            "interpreter": str(interpreter),
            "base_environment_root": str(base_environment_root),
        },
        "provenance": provenance,
        "manifests": {
            **source_manifests,
            "environment": environment_manifest,
        },
    }
    snapshot = {
        **snapshot_body,
        "snapshot_sha256": sha256_json(snapshot_body),
    }
    snapshot_path.write_text(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    training = {
        "schema_version": "2.0",
        "batch_id": "batch_1",
        "model_id": "motionr1_vm_lora",
        "training_mode": "full_sft",
        "provenance_snapshot_path": str(snapshot_path.resolve()),
        "provenance_snapshot_file_sha256": sha256_file(snapshot_path),
        "provenance_pre_sha256": snapshot["snapshot_sha256"],
        "provenance_post_sha256": snapshot["snapshot_sha256"],
        "provenance_unchanged": True,
    }
    controller_code = (REPO / "src" / "motion_eval").resolve()
    catalog_runner_code = (RUNNER_ROOT / "scripts").resolve()
    batch_receipt = {
        "code": {"path": str(controller_code), **hash_path(controller_code).to_dict()},
        "runner_code": {
            "path": str(catalog_runner_code),
            **hash_path(catalog_runner_code).to_dict(),
        },
    }
    assert batch_receipt["code"]["path"] != provenance["code"]["path"]
    assert (
        batch_receipt["runner_code"]["path"]
        != provenance["runner_code"]["path"]
    )

    with pytest.raises(ValueError, match="controller_code/catalog_runner_code"):
        runner_support._validate_catalog_formal_snapshot(
            attempt_root=attempt_root.resolve(),
            artifact=artifact.resolve(),
            training_path=training_path,
            batch_receipt=batch_receipt,
            artifact_manifest={"bindings": {}},
            training_receipt=training,
            model_id="motionr1_vm_lora",
        )


def test_bound_artifact_loader_invokes_independent_snapshot_validation(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    model_id = "qwen3vl_4b_lora"
    attempt_root = (
        tmp_path
        / "batches"
        / "batch_1"
        / "02_finetune"
        / model_id
        / "attempts"
        / "attempt_1"
    )
    artifact = attempt_root / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "adapter.bin").write_bytes(b"adapter")
    training_path = attempt_root / "training_receipt.json"
    training_path.write_text("{}\n", encoding="utf-8")
    pretrained_root = tmp_path / "pretrained"
    pretrained_root.mkdir()
    attempt = {
        "attempt_id": "attempt_1",
        "batch_id": "batch_1",
        "model_id": model_id,
        "stage": "finetune",
        "purpose": "production",
        "sample_limit": None,
        "expected_training_steps": 2,
        "batch_receipt_sha256": "1" * 64,
        "attempt_sha256": "2" * 64,
        "command_sha256": "3" * 64,
    }
    receipt = {
        "batch_id": "batch_1",
        "receipt_sha256": "1" * 64,
        "registry": {"sha256": "4" * 64},
        "pretrained_registry": {"sha256": "5" * 64},
        "pretrained_assets": {model_id: [{"frozen": True}]},
        "pretrained_assets_sha256": "6" * 64,
        "inputs": {
            "train": {"digest": "7" * 64},
            "validation": {"digest": "8" * 64},
            "leakage_audit": {"digest": "9" * 64},
        },
        "code": {"digest": "a" * 64},
        "runner_code": {"digest": "b" * 64},
        "config": {"model_training": {model_id: {"steps": 2}}},
        "config_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
        "runtime_roots": {"pretrained_root": str(pretrained_root.resolve())},
    }
    artifact_info = {
        "path": str(artifact.resolve()),
        **hash_path(
            artifact, symlink_policy="reject", allowed_root=attempt_root
        ).to_dict(),
    }
    reference = {
        "path": str(training_path.resolve()),
        "file_sha256": sha256_file(training_path),
        "content_sha256": "e" * 64,
    }
    manifest_body = {
        "schema_version": "1.0",
        "batch_id": "batch_1",
        "model_id": model_id,
        "attempt_id": "attempt_1",
        "purpose": "production",
        "status": "success",
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "training_steps": 2,
        "bindings": runner_support._manifest_bindings(receipt, attempt, model_id),
        "artifact": artifact_info,
        "training_receipt": reference,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": sha256_json(manifest_body),
    }
    training = {"schema_version": "2.0", "receipt_sha256": "e" * 64}
    backend = SimpleNamespace(
        BACKEND_ID="qwen:test",
        MODEL_FAMILY="qwen3_vl",
        TRAINING_MODE="lora_sft",
    )
    calls = []
    monkeypatch.setattr(runner_support, "_load_attempt", lambda *args, **kwargs: attempt)
    monkeypatch.setattr(runner_support, "_load_batch", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(runner_support, "_load_object", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(runner_support, "_require_backend", lambda *args: backend)
    monkeypatch.setattr(
        runner_support,
        "load_and_validate_training_receipt",
        lambda *args, **kwargs: training,
    )
    monkeypatch.setattr(
        runner_support,
        "_validate_catalog_formal_snapshot",
        lambda **kwargs: calls.append(kwargs),
    )

    runner_support._load_bound_finetune_artifact(
        artifact.resolve(),
        batch_id="batch_1",
        model_id=model_id,
        require_evidence=False,
    )
    assert len(calls) == 1
    assert calls[0]["training_receipt"] is training
    assert calls[0]["artifact_manifest"] is manifest
    assert calls[0]["batch_receipt"] is receipt


def test_catalog_reload_rejects_tampered_formal_snapshot_file(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    case["snapshot_path"].write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot is invalid") as caught:
        _validate_catalog_snapshot(runner_support, case)
    assert "file hash changed" in str(caught.value.__cause__)


def test_catalog_reload_rejects_self_hashed_snapshot_forgery(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    forged = {
        "schema_version": "motionllm-inprocess-provenance-v2",
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": "batch_1",
        "model_id": "qwen3vl_4b_lora",
        "training_mode": "lora_sft",
        "canonical_identity": {},
        "provenance": {},
        "manifests": {},
        "snapshot_sha256": "0" * 64,
    }
    case["snapshot_path"].write_text(json.dumps(forged), encoding="utf-8")
    case["training"]["provenance_snapshot_file_sha256"] = sha256_file(
        case["snapshot_path"]
    )

    with pytest.raises(ValueError, match="snapshot is invalid") as caught:
        _validate_catalog_snapshot(runner_support, case)
    assert "self-hash" in str(caught.value.__cause__)


def test_catalog_reload_rejects_snapshot_artifact_manifest_mismatch(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    monkeypatch.setattr(
        runner_support,
        "load_and_validate_formal_provenance_snapshot",
        lambda *args, **kwargs: case["snapshot"],
    )
    case["manifest"]["bindings"]["environment_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="artifact manifest provenance.*environment"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_snapshot_benchmark_not_in_batch_receipt(
    monkeypatch, tmp_path
):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    monkeypatch.setattr(
        runner_support,
        "load_and_validate_formal_provenance_snapshot",
        lambda *args, **kwargs: case["snapshot"],
    )
    case["snapshot"]["provenance"]["benchmark"] = {
        **case["snapshot"]["provenance"]["benchmark"],
        "digest": "0" * 64,
    }

    with pytest.raises(ValueError, match="frozen batch inputs.*benchmark"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_unfrozen_optional_motion_asset(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    motion = tmp_path / "motion_vqvae.pth"
    motion.write_bytes(b"motion-vqvae")
    case["snapshot"]["provenance"]["motion_vqvae"] = {
        "path": str(motion.resolve()),
        "algorithm": "test-sha256-v1",
        "kind": "file",
        "digest": _digest("motion_vqvae"),
        "file_count": 1,
        "total_bytes": len(b"motion-vqvae"),
    }
    monkeypatch.setattr(
        runner_support,
        "load_and_validate_formal_provenance_snapshot",
        lambda *args, **kwargs: case["snapshot"],
    )

    with pytest.raises(ValueError, match="motion_vqvae.*frozen asset"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_snapshot_outside_attempt(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    escaped = tmp_path / "escaped_snapshot.json"
    escaped.write_text("{}\n", encoding="utf-8")
    case["training"]["provenance_snapshot_path"] = str(escaped.resolve())
    case["training"]["provenance_snapshot_file_sha256"] = sha256_file(escaped)

    with pytest.raises(ValueError, match="escapes the finetune attempt"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_snapshot_inside_artifact(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    embedded = case["artifact"] / "formal_provenance_snapshot.json"
    embedded.write_text("{}\n", encoding="utf-8")
    case["training"]["provenance_snapshot_path"] = str(embedded.resolve())
    case["training"]["provenance_snapshot_file_sha256"] = sha256_file(embedded)

    with pytest.raises(ValueError, match="outside the artifact"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_snapshot_link_or_reparse_path(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    original = runner_support._is_link_or_reparse
    monkeypatch.setattr(
        runner_support,
        "_is_link_or_reparse",
        lambda path: path == case["snapshot_path"] or original(path),
    )

    with pytest.raises(ValueError, match="links/reparse points"):
        _validate_catalog_snapshot(runner_support, case)


def test_catalog_reload_rejects_qwen_snapshot_schema_downgrade(monkeypatch, tmp_path):
    runner_support = _runner_support(monkeypatch)
    case = _catalog_formal_snapshot_case(tmp_path)
    case["training"]["schema_version"] = "1.0"

    with pytest.raises(ValueError, match="schema-2 provenance snapshot"):
        _validate_catalog_snapshot(runner_support, case)
