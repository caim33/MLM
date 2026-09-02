from __future__ import annotations

import json
import importlib.machinery
import os
import subprocess
from pathlib import Path

import pytest

from motion_eval.core import hash_path, sha256_file, sha256_json
from motion_eval.training_receipt import make_training_receipt, write_training_receipt
from motionllm.training import (
    ArtifactProvenancePaths,
    ArtifactValidationError,
    ReloadVerificationReceipt,
    write_finetune_artifact_manifest,
    write_reload_verification_receipt,
)


BATCH_ID = "qa500v2_deadbeef"
MODEL_ID = "motionr1_vm_lora"
TRAINING_MODE = "full_sft"


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path.resolve()


def _plain_evidence(path: Path) -> dict[str, object]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), **hash_path(candidate).to_dict()}


def _file_row(path: Path, root: Path) -> dict[str, object]:
    candidate = path.resolve(strict=True)
    return {
        "relative_path": candidate.relative_to(root).as_posix(),
        "sha256": sha256_file(candidate),
        "size": candidate.stat().st_size,
    }


def _source_manifest(
    root: Path, *, role: str, files: tuple[Path, ...]
) -> tuple[dict[str, object], dict[str, object]]:
    root = root.resolve(strict=True)
    rows = sorted(
        (_file_row(path, root) for path in files),
        key=lambda row: str(row["relative_path"]),
    )
    body: dict[str, object] = {
        "schema_version": "motionllm-source-allowlist-v2",
        "role": role,
        "root": str(root),
        "files": rows,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    evidence = {
        "path": str(root),
        "algorithm": "motionllm-source-allowlist-v2",
        "kind": "source-allowlist",
        "digest": manifest["manifest_sha256"],
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
    }
    return manifest, evidence


def _installed_file_row(
    path: Path,
    root: Path,
    *,
    owner: str = "unowned-direct-content",
    record_sha256: str | None = None,
) -> dict[str, object]:
    row = _file_row(path, root)
    return {**row, "owner": owner, "record_sha256": record_sha256}


def _build_environment_manifest(
    tmp_path: Path,
    *,
    code_root: Path,
    runner_root: Path,
    environment_root: Path,
    base_root: Path,
    interpreter: Path,
    attack: str | None,
) -> tuple[dict[str, object], dict[str, object], Path]:
    pyvenv = _write_bytes(
        environment_root / "pyvenv.cfg",
        b"include-system-site-packages = false\n",
    )
    package = _write_bytes(
        environment_root / "Lib" / "site-packages" / "demo.py",
        b"VERSION = '1.0'\n",
    )
    dist_info = environment_root / "Lib" / "site-packages" / "demo-1.0.dist-info"
    metadata = _write_bytes(
        dist_info / "METADATA", b"Name: demo\nVersion: 1.0\n"
    )
    record = _write_bytes(dist_info / "RECORD", b"demo.py,,\n")
    stdlib_root = (base_root / "Lib").resolve()
    stdlib_file = _write_bytes(stdlib_root / "os.py", b"name = 'posix'\n")
    native = _write_bytes(base_root / "DLLs" / "runtime.dll", b"native-runtime")

    pth_rows: list[dict[str, object]] = []
    installed = [
        _installed_file_row(interpreter, environment_root),
        _installed_file_row(pyvenv, environment_root),
        _installed_file_row(
            package, environment_root, owner="demo==1.0", record_sha256=None
        ),
        _installed_file_row(metadata, environment_root),
        _installed_file_row(record, environment_root),
    ]
    loading_variables: dict[str, str | None] = {
        "CUDA_HOME": None,
        "CUDA_PATH": None,
        "CUDA_VISIBLE_DEVICES": None,
        "LD_LIBRARY_PATH": None,
        "NVIDIA_VISIBLE_DEVICES": None,
    }
    sys_path = [
        str((code_root / "src").resolve()),
        str((code_root / "models").resolve()),
        str(runner_root.resolve()),
        str(package.parent.resolve()),
        str(stdlib_root),
    ]
    native_rows: list[dict[str, object]] = [
        {
            "logical_path": str(native),
            "link_sha256": None,
            "resolved_target": str(native),
            "sha256": sha256_file(native),
            "size": native.stat().st_size,
        }
    ]
    meta_rows: list[dict[str, object]] = [
        {
            "module": "_frozen_importlib",
            "qualname": "BuiltinImporter",
            "origin": None,
            "origin_sha256": None,
        }
    ]
    internal_links: list[dict[str, object]] = []

    external = tmp_path / "external-runtime"
    if attack == "external_pth":
        external.mkdir()
        pth = _write_bytes(
            environment_root / "Lib" / "site-packages" / "escape.pth",
            (str(external.resolve()) + "\n").encode("utf-8"),
        )
        pth_rows.append(
            {
                "relative_path": pth.relative_to(environment_root).as_posix(),
                "sha256": sha256_file(pth),
                "accepted_paths": [str(external.resolve())],
            }
        )
        installed.append(_installed_file_row(pth, environment_root))
    elif attack == "external_sys_path":
        external.mkdir()
        sys_path.append(str(external.resolve()))
    elif attack == "internal_environment_sys_path":
        injected = environment_root / "injected-path"
        _write_bytes(injected / "evil_loader.py", b"LOADED = True\n")
        sys_path.append(str(injected.resolve()))
    elif attack == "code_root_sys_path":
        _write_bytes(
            tmp_path / "checkout" / "evil_loader.py", b"LOADED = True\n"
        )
        sys_path.append(str((tmp_path / "checkout").resolve()))
    elif attack == "external_native":
        escaped_native = _write_bytes(external / "runtime.dll", b"escaped-native")
        native_rows = [
            {
                "logical_path": str(escaped_native),
                "link_sha256": None,
                "resolved_target": str(escaped_native),
                "sha256": sha256_file(escaped_native),
                "size": escaped_native.stat().st_size,
            }
        ]
    elif attack == "loader_environment":
        loading_variables["LD_PRELOAD"] = str(external / "inject.dll")
    elif attack == "external_meta_path":
        escaped_loader = _write_bytes(external / "loader.py", b"class Loader: pass\n")
        meta_rows = [
            {
                "module": "external_loader",
                "qualname": "Loader",
                "origin": str(escaped_loader),
                "origin_sha256": sha256_file(escaped_loader),
            }
        ]
    elif attack == "spoofed_originless_finder":
        meta_rows = [
            {
                "module": "_frozen_importlib",
                "qualname": "InjectedFinder",
                "origin": None,
                "origin_sha256": None,
            }
        ]
    elif attack == "excluded_cache_sys_path":
        cache_root = code_root / "src" / ".cache"
        sys_path.append(str(cache_root.resolve()))
        assert (
            importlib.machinery.PathFinder.find_spec(
                "evil_loader", [str(cache_root.resolve())]
            )
            is not None
        )
    elif attack == "internal_directory_link":
        hidden_package = environment_root / "hidden-runtime" / "evilpkg"
        _write_bytes(hidden_package / "__init__.py", b"LOADED = True\n")
        linked_package = package.parent / "evilpkg"
        try:
            linked_package.symlink_to(hidden_package, target_is_directory=True)
        except OSError as exc:
            if os.name != "nt":
                pytest.skip(f"directory links are unavailable: {exc}")
            created = subprocess.run(
                (
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(linked_package),
                    str(hidden_package),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if created.returncode != 0:
                pytest.skip(
                    "directory symlinks/junctions are unavailable: "
                    f"{exc}; {created.stderr.strip()}"
                )
        assert (
            importlib.machinery.PathFinder.find_spec(
                "evilpkg", [str(package.parent.resolve())]
            )
            is not None
        )
        internal_links.append(
            {
                "relative_path": linked_package.relative_to(
                    environment_root
                ).as_posix(),
                "link_sha256": hash_path(
                    linked_package, symlink_policy="link"
                ).digest,
                "resolved_target": str(hidden_package.resolve(strict=True)),
            }
        )
    elif attack is not None:
        raise AssertionError(f"unknown attack fixture: {attack}")

    stdlib_rows = [_file_row(stdlib_file, base_root)]
    native_allowed_roots = sorted(
        (str(environment_root), str(base_root)), key=os.path.normcase
    )
    body: dict[str, object] = {
        "schema_version": "motionllm-installed-environment-v2",
        "environment_root": str(environment_root),
        "base_environment_root": str(base_root),
        "python_version": "3.12.0 (test)",
        "python_implementation": "cpython",
        "interpreter_entry": str(interpreter),
        "interpreter_entry_link_sha256": None,
        "interpreter_target": str(interpreter),
        "interpreter_target_sha256": sha256_file(interpreter),
        "interpreter_target_size": interpreter.stat().st_size,
        "pyvenv_sha256": sha256_file(pyvenv),
        "pyvenv": {"include-system-site-packages": "false"},
        "pth_files": pth_rows,
        "loading_environment": {
            "variables": loading_variables,
            "normalized_ld_library_path": [],
        },
        "sys_path": sys_path,
        "meta_path": meta_rows,
        "stdlib_root": str(stdlib_root),
        "stdlib_files": stdlib_rows,
        "native_runtime_files": native_rows,
        "native_allowed_roots": native_allowed_roots,
        "internal_links": internal_links,
        "distributions": [
            {
                "name": "demo",
                "version": "1.0",
                "dist_info": dist_info.relative_to(environment_root).as_posix(),
                "record_sha256": sha256_file(record),
                "record_file_count": 1,
            }
        ],
        "files": installed,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    evidence = {
        "path": str(environment_root),
        "algorithm": "motionllm-installed-environment-v2",
        "kind": "installed-environment-manifest",
        "digest": manifest["manifest_sha256"],
        "file_count": (
            len(installed)
            + len(internal_links)
            + len(stdlib_rows)
            + len(native_rows)
            + 1
        ),
        "total_bytes": (
            sum(int(row["size"]) for row in installed)
            + sum(int(row["size"]) for row in stdlib_rows)
            + sum(int(row["size"]) for row in native_rows)
            + interpreter.stat().st_size
        ),
    }
    return manifest, evidence, stdlib_root


def _write_formal_bundle(tmp_path: Path, *, attack: str | None) -> Path:
    code_root = (tmp_path / "checkout").resolve()
    runner_root = (code_root / "qwenvl").resolve()
    code_file = _write_bytes(code_root / "src" / "motionllm" / "entry.py", b"CODE = 1\n")
    eval_file = _write_bytes(
        code_root / "src" / "motion_eval" / "controller.py", b"VERIFY = 1\n"
    )
    model_file = _write_bytes(code_root / "models" / "model.py", b"MODEL = 1\n")
    fixed_code_files = tuple(
        _write_bytes(code_root / relative, value)
        for relative, value in (
            ("pyproject.toml", b"[project]\nname = 'fixture'\n"),
            ("requirements/sft.txt", b"torch==0\n"),
            ("scripts/full_sft.sh", b"#!/bin/sh\n"),
            ("scripts/lora_sft.sh", b"#!/bin/sh\n"),
            ("scripts/zero2.json", b"{}\n"),
        )
    )
    runner_data_file = _write_bytes(
        runner_root / "data" / "dataset.py", b"DATA = 1\n"
    )
    runner_file = _write_bytes(runner_root / "train" / "runner.py", b"RUN = 1\n")
    if attack == "excluded_cache_sys_path":
        _write_bytes(
            code_root / "src" / ".cache" / "evil_loader.py",
            b"LOADED = True\n",
        )
    environment_root = (tmp_path / "venv").resolve()
    base_root = (tmp_path / "base-python").resolve()
    interpreter = _write_bytes(
        environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        b"python-interpreter",
    )
    base_root.mkdir(parents=True, exist_ok=True)

    plain_paths: dict[str, Path] = {}
    for role in (
        "base_artifact",
        "train_data",
        "validation_data",
        "benchmark",
        "leakage_audit",
        "config",
    ):
        plain_paths[role] = _write_bytes(
            tmp_path / "inputs" / f"{role}.bin", f"{role}-content".encode()
        )

    code_files = (code_file, eval_file, model_file, *fixed_code_files)
    runner_files = (runner_data_file, runner_file)
    code_manifest, code_evidence = _source_manifest(
        code_root,
        role="code",
        files=code_files[:-1] if attack == "omit_code" else code_files,
    )
    runner_manifest, runner_evidence = _source_manifest(
        runner_root,
        role="runner_code",
        files=runner_files[:-1] if attack == "omit_runner" else runner_files,
    )
    environment_manifest, environment_evidence, _ = _build_environment_manifest(
        tmp_path,
        code_root=code_root,
        runner_root=runner_root,
        environment_root=environment_root,
        base_root=base_root,
        interpreter=interpreter,
        attack=(
            attack
            if attack
            in {
                "external_pth",
                "external_sys_path",
                "internal_environment_sys_path",
                "code_root_sys_path",
                "external_native",
                "loader_environment",
                "external_meta_path",
                "spoofed_originless_finder",
                "excluded_cache_sys_path",
                "internal_directory_link",
            }
            else None
        ),
    )
    provenance: dict[str, dict[str, object]] = {
        role: _plain_evidence(path) for role, path in plain_paths.items()
    }
    provenance.update(
        {
            "code": code_evidence,
            "runner_code": runner_evidence,
            "environment": environment_evidence,
        }
    )
    snapshot_body: dict[str, object] = {
        "schema_version": "motionllm-inprocess-provenance-v2",
        "status": "captured_before_model_data_load_after_entrypoint_imports",
        "batch_id": BATCH_ID,
        "model_id": MODEL_ID,
        "training_mode": TRAINING_MODE,
        "canonical_identity": {
            "model_id": MODEL_ID,
            "model_family": "qwen3_vl_motion",
            "modality": "VM",
            "base_artifact": str(plain_paths["base_artifact"]),
            "code_root": str(code_root),
            "runner_code_root": str(runner_root),
            "environment_root": str(environment_root),
            "interpreter": str(interpreter),
            "base_environment_root": str(base_root),
        },
        "provenance": provenance,
        "manifests": {
            "code": code_manifest,
            "runner_code": runner_manifest,
            "environment": environment_manifest,
        },
    }
    snapshot = {**snapshot_body, "snapshot_sha256": sha256_json(snapshot_body)}
    snapshot_path = tmp_path / "formal_provenance_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    artifact = tmp_path / "checkpoint"
    _write_bytes(artifact / "weights.bin", b"fresh-weights")
    artifact_sha256 = hash_path(artifact).digest
    paths = ArtifactProvenancePaths(
        base_artifact=plain_paths["base_artifact"],
        train_data=plain_paths["train_data"],
        validation_data=plain_paths["validation_data"],
        benchmark=plain_paths["benchmark"],
        leakage_audit=plain_paths["leakage_audit"],
        config=plain_paths["config"],
        code=code_root,
        runner_code=runner_root,
        environment=environment_root,
    )
    training = make_training_receipt(
        batch_id=BATCH_ID,
        model_id=MODEL_ID,
        backend_id="test_backend",
        model_family="qwen3_vl_motion",
        modality="VM",
        training_mode=TRAINING_MODE,
        planned_global_steps=1,
        actual_global_steps=1,
        planned_optimizer_steps=1,
        actual_optimizer_steps=1,
        finite_losses=[1.0],
        nonzero_finite_gradient_steps=1,
        max_gradient=0.5,
        trainable_tensor_count=1,
        trainable_parameter_count=1,
        changed_trainable_tensor_count=1,
        initial_trainable_sha256="1" * 64,
        final_trainable_sha256="2" * 64,
        max_parameter_update=0.25,
        batch_receipt_sha256="3" * 64,
        attempt_sha256="4" * 64,
        train_sha256=str(provenance["train_data"]["digest"]),
        validation_sha256=str(provenance["validation_data"]["digest"]),
        leakage_audit_sha256=str(provenance["leakage_audit"]["digest"]),
        base_artifact_sha256=str(provenance["base_artifact"]["digest"]),
        config_sha256=str(provenance["config"]["digest"]),
        code_sha256=str(provenance["code"]["digest"]),
        runner_code_sha256=str(provenance["runner_code"]["digest"]),
        environment_sha256=str(provenance["environment"]["digest"]),
        artifact_sha256=artifact_sha256,
        provenance_snapshot_path=str(snapshot_path.resolve()),
        provenance_snapshot_file_sha256=sha256_file(snapshot_path),
        provenance_pre_sha256=snapshot["snapshot_sha256"],
        provenance_post_sha256=snapshot["snapshot_sha256"],
        provenance_unchanged=True,
    )
    training_path = tmp_path / "training_receipt.json"
    write_training_receipt(training_path, training, root=tmp_path)
    reload_path = tmp_path / "reload_receipt.json"
    write_reload_verification_receipt(
        reload_path,
        ReloadVerificationReceipt(
            batch_id=BATCH_ID,
            model_id=MODEL_ID,
            artifact_hash=artifact_sha256,
            expected_modules=("__test__",),
            reloaded_modules=("__test__",),
            motion_start_token_id=None,
            motion_end_token_id=None,
            state_hash_before="5" * 64,
            state_hash_after="5" * 64,
            processor_state_hash_before="6" * 64,
            processor_state_hash_after="6" * 64,
            processor_assets_hash="7" * 64,
        ),
        allowed_root=tmp_path,
    )
    manifest_path = tmp_path / "run_manifest.json"
    write_finetune_artifact_manifest(
        manifest_path,
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id=BATCH_ID,
        model_id=MODEL_ID,
        training_mode=TRAINING_MODE,
        allowed_root=tmp_path,
        reload_receipt_path=reload_path,
        training_receipt_path=training_path,
        provenance_evidence=provenance,
    )
    return manifest_path


def test_self_consistent_formal_snapshot_and_receipt_are_accepted(tmp_path):
    manifest = _write_formal_bundle(tmp_path, attack=None)
    assert manifest.is_file()


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("external_pth", "accepted_paths escapes its allowed roots"),
        ("external_sys_path", r"sys.path\[5\] escapes its allowed roots"),
        (
            "internal_environment_sys_path",
            r"sys.path\[5\] escapes its allowed roots",
        ),
        ("code_root_sys_path", r"sys.path\[5\] escapes its allowed roots"),
        (
            "external_native",
            r"native_runtime_files\[0\].logical_path escapes its allowed roots",
        ),
        ("loader_environment", "rejects loader/Python variable: LD_PRELOAD"),
        ("external_meta_path", r"meta_path\[0\].origin escapes its allowed roots"),
        ("spoofed_originless_finder", r"meta_path\[0\] hash is invalid"),
        (
            "excluded_cache_sys_path",
            "rejects excluded control directory inside an import tree",
        ),
        (
            "internal_directory_link",
            "rejects every directory link/reparse",
        ),
        ("omit_code", "snapshot code source inventory is incomplete"),
        ("omit_runner", "snapshot runner_code source inventory is incomplete"),
    ],
)
def test_artifact_rejects_self_consistent_malicious_snapshot_and_receipt(
    tmp_path, attack, message
):
    with pytest.raises(ArtifactValidationError, match=message):
        _write_formal_bundle(tmp_path, attack=attack)
