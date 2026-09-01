from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import motionllm.training.sft as sft
import motion_eval.core.source_inventory as source_inventory


def _formal_fixture(tmp_path, monkeypatch):
    environment = tmp_path / "venv"
    interpreter = environment / ("Scripts/python.exe" if sft.os.name == "nt" else "bin/python")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python")
    base_environment = tmp_path / "base-python"
    base_environment.mkdir()

    remote_root = tmp_path / "canonical-pretrained"
    base = remote_root / "by_model" / "motionr1_vm_lora" / "base"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}", encoding="utf-8")

    project_root = Path(sft.__file__).resolve().parents[3]
    model_registry = {
        "schema_version": "1.0",
        "updated_at": "now",
        "fresh_finetune_required_per_batch": True,
        "global_finetune_barrier_before_eval": True,
        "models": [
            {
                "id": "motionr1_vm_lora",
                "main_modality": "VM",
            }
        ],
    }
    pretrained_registry = {
        "schema_version": "1.0",
        "updated_at": "now",
        "purpose": "test",
        "remote_root": str(remote_root),
        "policy": {},
        "models": [
            {
                "id": "motionr1_vm_lora",
                "artifacts": [
                    {
                        "role": "base_and_processor",
                        "path": "by_model/motionr1_vm_lora/base",
                        "kind": "hf_model_dir",
                    }
                ],
            }
        ],
    }

    def registry_loader(path):
        return (
            model_registry
            if Path(path).name == "model_registry.json"
            else pretrained_registry
        )

    monkeypatch.setattr(sft, "load_json_strict", registry_loader)
    monkeypatch.setattr(sft.sys, "prefix", str(environment))
    monkeypatch.setattr(sft.sys, "base_prefix", str(base_environment))
    monkeypatch.setattr(sft.sys, "executable", str(interpreter))
    model_arguments = SimpleNamespace(model_family="qwen3_vl_motion")
    artifact_arguments = SimpleNamespace(
        model_registry_id="motionr1_vm_lora",
        base_artifact_path=str(base),
        code_path=str(project_root),
        runner_code_path=str(project_root / "qwenvl"),
        environment_path=str(environment),
    )
    model_spec = SimpleNamespace(
        family=SimpleNamespace(value="qwen3_vl_motion"), supports_motion=True
    )
    return model_arguments, artifact_arguments, model_spec, base, environment, project_root


def test_canonical_sft_identity_binds_environment_root_and_actual_runner_tree(
    tmp_path, monkeypatch
):
    model, artifact, spec, base, environment, project_root = _formal_fixture(
        tmp_path, monkeypatch
    )
    identity = sft.bind_canonical_formal_identity(
        model, artifact, model_spec=spec, formal_artifact=True
    )
    assert identity.base_artifact == base
    assert identity.environment_root == environment.resolve()
    assert identity.runner_code_root == project_root / "qwenvl"

    artifact.environment_path = str(Path(sft.sys.executable))
    with pytest.raises(ValueError, match="environment root"):
        sft.bind_canonical_formal_identity(
            model, artifact, model_spec=spec, formal_artifact=True
        )


def test_canonical_base_rejects_a_spoofed_matching_suffix(tmp_path, monkeypatch):
    model, artifact, spec, _, _, _ = _formal_fixture(tmp_path, monkeypatch)
    spoof = tmp_path / "spoof" / "by_model" / "motionr1_vm_lora" / "base"
    spoof.mkdir(parents=True)
    (spoof / "config.json").write_text("{}", encoding="utf-8")
    artifact.base_artifact_path = str(spoof)
    with pytest.raises(ValueError, match="exactly equal remote_root"):
        sft.bind_canonical_formal_identity(
            model, artifact, model_spec=spec, formal_artifact=True
        )


def test_canonical_runner_code_cannot_alias_the_broader_code_root(
    tmp_path, monkeypatch
):
    model, artifact, spec, _, _, project_root = _formal_fixture(
        tmp_path, monkeypatch
    )
    artifact.runner_code_path = str(project_root)
    with pytest.raises(ValueError, match="actual checkout's qwenvl tree"):
        sft.bind_canonical_formal_identity(
            model, artifact, model_spec=spec, formal_artifact=True
        )


def test_formal_qwen_bootstrap_is_explicitly_blocked_but_legacy_is_unchanged(
    monkeypatch,
):
    for name in ("NNODES", "GROUP_WORLD_SIZE", "NODE_RANK", "GROUP_RANK"):
        monkeypatch.delenv(name, raising=False)
    sft.require_controller_verified_formal_bootstrap(formal_artifact=False)
    with pytest.raises(RuntimeError, match="external-HMAC-bound pre-spawn"):
        sft.require_controller_verified_formal_bootstrap(formal_artifact=True)


def test_formal_qwen_rejects_multinode_before_bootstrap(monkeypatch):
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("GROUP_WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("GROUP_RANK", "0")
    with pytest.raises(RuntimeError, match="one node only"):
        sft.require_controller_verified_formal_bootstrap(formal_artifact=True)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LOCAL_WORLD_SIZE", "2", "zero-restart local-rank topology"),
        ("TORCHELASTIC_RESTART_COUNT", "1", "zero-restart local-rank topology"),
        ("MASTER_ADDR", "10.0.0.5", "loopback MASTER_ADDR"),
    ],
)
def test_formal_qwen_rejects_unbound_local_topology(
    monkeypatch, name, value, message
):
    for field in (
        "NNODES", "GROUP_WORLD_SIZE", "NODE_RANK", "GROUP_RANK", "WORLD_SIZE",
        "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "TORCHELASTIC_RESTART_COUNT",
        "MASTER_ADDR",
    ):
        monkeypatch.delenv(field, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        sft.require_controller_verified_formal_bootstrap(formal_artifact=True)


def test_formal_environment_rejects_python_and_loader_injection(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "C:/untrusted")
    with pytest.raises(ValueError, match="injection variable: PYTHONPATH"):
        sft._runtime_loading_environment()
    monkeypatch.delenv("PYTHONPATH")
    monkeypatch.setenv("LD_PRELOAD", "C:/untrusted.dll")
    with pytest.raises(ValueError, match="injection variable: LD_PRELOAD"):
        sft._runtime_loading_environment()


def test_formal_environment_rejects_executable_pth_and_legacy_egg_link(tmp_path):
    environment = tmp_path / "venv"
    site = environment / "Lib" / "site-packages"
    site.mkdir(parents=True)
    executable = site / "inject.pth"
    executable.write_text("import injected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable .pth"):
        sft._validate_pth_files((site,), environment)
    executable.unlink()
    (site / "legacy.egg-link").write_text("../source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy editable egg-link"):
        sft._validate_pth_files((site,), environment)


def test_formal_source_manifest_hashes_source_backed_bytecode_bytes(tmp_path):
    project_root = tmp_path / "checkout"
    package = project_root / "src" / "package"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    bytecode = cache / "module.cpython-312.pyc"
    bytecode.write_bytes(b"first-bytecode-image")

    first = sft._stable_source_files(project_root, roots=(package,))
    first_rows = {row["relative_path"]: row for row in first}
    bytecode_relative = "src/package/__pycache__/module.cpython-312.pyc"
    assert bytecode_relative in first_rows

    bytecode.write_bytes(b"second-bytecode-image")
    second = sft._stable_source_files(project_root, roots=(package,))
    second_rows = {row["relative_path"]: row for row in second}
    assert second_rows[bytecode_relative]["sha256"] != first_rows[bytecode_relative][
        "sha256"
    ]
    assert second_rows[bytecode_relative]["size"] == len(b"second-bytecode-image")


def test_formal_source_rejects_linked_bytecode_cache_before_special_case(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "checkout"
    package = project_root / "src" / "package"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    original = source_inventory.is_link_or_reparse
    monkeypatch.setattr(
        source_inventory,
        "is_link_or_reparse",
        lambda path: Path(path) == cache or original(Path(path)),
    )
    with pytest.raises(ValueError, match="linked/non-directory bytecode cache"):
        sft._stable_source_files(project_root, roots=(package,))
