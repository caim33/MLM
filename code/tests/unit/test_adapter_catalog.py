from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from motion_eval.adapters import AdapterContext, build_adapter_catalog
from motion_eval.adapters.catalog import FROZEN_ADAPTER_SPECS, FrozenAdapterSpec
from motion_eval.contracts import InputModality
from motion_eval.controller import EXPECTED_MODEL_IDS, load_canonical_registry
from motion_eval.runtime import CommandSpec, CommandValidationError


REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "model_evaluation_agent" / "model_registry.json"
PRETRAINED = REPO / "model_evaluation_agent" / "pretrained_registry.json"


def test_registry_and_adapter_catalog_exactly_cover_15_models():
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    catalog = build_adapter_catalog(registry)
    assert registry.ids == EXPECTED_MODEL_IDS
    assert tuple(catalog) == EXPECTED_MODEL_IDS
    assert len(catalog) == 15


def _load_runner_specs_module():
    path = REPO / "model_evaluation_agent" / "scripts" / "runner_specs.py"
    spec = importlib.util.spec_from_file_location("test_runner_specs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_views_are_derived_exactly_from_the_frozen_controller_catalog():
    runner_specs = _load_runner_specs_module()
    assert tuple(runner_specs.MODEL_SPECS) == tuple(FROZEN_ADAPTER_SPECS)
    assert tuple(runner_specs.BACKENDS) == tuple(FROZEN_ADAPTER_SPECS)
    for model_id, frozen in FROZEN_ADAPTER_SPECS.items():
        assert runner_specs.MODEL_SPECS[model_id] == (
            frozen.modality,
            frozen.evaluation_mode,
            frozen.initialization,
        )
        assert runner_specs.dependencies_for(model_id) == frozen.dependencies
        for role in ("finetune", "evaluation", "verifier"):
            assert runner_specs.backend_for(model_id, role) == frozen.backend_import_for(role)


def test_frozen_catalog_cannot_be_mutated_or_queried_with_unknown_identity():
    first = FROZEN_ADAPTER_SPECS["qwen36_27b_lora"]
    with pytest.raises(TypeError):
        FROZEN_ADAPTER_SPECS["qwen36_27b_lora"] = first
    with pytest.raises(FrozenInstanceError):
        first.modality = "M"
    runner_specs = _load_runner_specs_module()
    with pytest.raises(ValueError, match="unknown catalog model"):
        runner_specs.backend_for("invented_model", "finetune")
    with pytest.raises(ValueError, match="unknown adapter role"):
        runner_specs.backend_for("qwen36_27b_lora", "invented_role")


@pytest.mark.parametrize(
    "backend_path",
    (
        "../backends/evil.py",
        "scripts/../evil.py",
        "scripts/backends/../../evil.py",
        "/scripts/backends/evil.py",
        "scripts/not_backends/evil.py",
        "scripts/backends/evil.txt",
    ),
)
def test_backend_path_to_import_translation_fails_closed(backend_path):
    spec = FrozenAdapterSpec(
        "adversarial_model",
        "V",
        "generative",
        "pretrained",
        "scripts/finetune_adversarial.py",
        "scripts/eval_adversarial.py",
        finetune_backend=backend_path,
    )
    with pytest.raises(RuntimeError, match="not a safe scripts/backend file"):
        spec.backend_import_for("finetune")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("modality", InputModality.MOTION),
        ("evaluation_mode", "discriminative_abcd_scores"),
    ),
)
def test_registry_metadata_drift_fails_before_catalog_commands_are_built(field, value):
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    tampered_model = replace(registry.models[0], **{field: value})
    tampered_registry = replace(
        registry, models=(tampered_model, *registry.models[1:])
    )
    with pytest.raises(ValueError, match="registry/catalog contract drift"):
        build_adapter_catalog(tampered_registry)


def test_all_models_produce_typed_finetune_and_eval_specs():
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    catalog = build_adapter_catalog(registry)
    for model in registry.models:
        context = AdapterContext(
            batch_id="qa500v2_test_deadbeef",
            python_executable="python3",
            controller_root="/controller",
            batch_root="/controller/batches/qa500v2_test_deadbeef",
            train_manifest="/data/train.jsonl",
            validation_manifest="/data/validation.jsonl",
            benchmark_manifest="/data/benchmark.jsonl",
            media_manifest="/data/media_manifest.json",
            media_manifest_sha256="0" * 64,
            leakage_audit="/data/leakage.json",
            pretrained_root="/controller/shared_assets/pretrained",
            output_path=f"/batch/02_finetune/{model.model_id}/artifact",
            artifact_path=f"/batch/02_finetune/{model.model_id}/artifact",
            artifact_digest="0" * 64,
            attempt_id="attempt",
            training_steps=2,
        )
        finetune = catalog[model.model_id].finetune_spec(context)
        evaluate = catalog[model.model_id].evaluation_spec(context)
        verify = catalog[model.model_id].verification_spec(context)
        assert isinstance(finetune, CommandSpec)
        assert isinstance(evaluate, CommandSpec)
        assert isinstance(verify, CommandSpec)
        assert finetune.receipt()["shell"] is False
        assert evaluate.receipt()["shell"] is False
        assert verify.receipt()["shell"] is False
        assert model.modality.value in finetune.argv
        assert model.evaluation_mode in evaluate.argv
        assert "--media-manifest" in evaluate.argv
        assert "--media-manifest-sha256" in evaluate.argv


def test_agcn_and_motionclip_are_formal_finetunes_never_proxies():
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    catalog = build_adapter_catalog(registry)
    agcn = catalog["agcn_official"]
    motionclip = catalog["motionclip_official"]
    assert agcn.official_finetune and motionclip.official_finetune
    assert agcn.initialization == "random"
    assert "agcn_official" in agcn.finetune_runner
    assert "motionclip_official" in motionclip.finetune_runner
    assert "proxy" not in agcn.finetune_runner.lower()
    assert "proxy" not in motionclip.finetune_runner.lower()


def test_every_active_catalog_facade_exists_and_exposes_its_standard_cli():
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    catalog = build_adapter_catalog(registry)
    runner_root = REPO / "model_evaluation_agent"
    paths = {
        runner_root / relative
        for adapter in catalog.values()
        for relative in (
            adapter.finetune_runner,
            adapter.evaluation_runner,
            adapter.verifier_runner,
        )
    }
    assert paths
    for path in sorted(paths):
        assert path.is_file() and not path.is_symlink(), path
        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (path, completed.stderr)
        help_text = completed.stdout
        if path.name.startswith("finetune_"):
            for flag in (
                "--batch-id", "--model-id", "--train-manifest",
                "--validation-manifest", "--leakage-audit", "--pretrained-root",
                "--output-dir", "--modality", "--initialization", "--purpose",
                "--training-steps",
            ):
                assert flag in help_text
        elif path.name.startswith("eval_"):
            for flag in (
                "--batch-id", "--model-id", "--benchmark-manifest",
                "--media-manifest", "--media-manifest-sha256", "--artifact",
                "--predictions", "--modality", "--evaluation-mode",
            ):
                assert flag in help_text
        else:
            for flag in (
                "--batch-id", "--model-id", "--attempt-id", "--artifact",
                "--artifact-sha256", "--report",
            ):
                assert flag in help_text


def test_backend_inventory_is_explicit_and_missing_integrations_are_not_facades():
    registry = load_canonical_registry(REGISTRY, PRETRAINED)
    catalog = build_adapter_catalog(registry)
    runner_root = REPO / "model_evaluation_agent"
    implemented = {"videollama_lora", "motionllm_official"}
    for model_id, adapter in catalog.items():
        finetune = runner_root / adapter.finetune_backend
        evaluation = runner_root / adapter.evaluation_backend
        verifier = runner_root / adapter.verifier_backend
        assert finetune.is_file() is (model_id in implemented)
        assert verifier.is_file() is (model_id in implemented)
        assert not evaluation.exists()


def test_unimplemented_evaluation_facade_fails_closed_without_predictions(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    media = tmp_path / "media.json"
    artifact = tmp_path / "artifact"
    predictions = tmp_path / "predictions.jsonl"
    benchmark.write_text("{}\n", encoding="utf-8")
    media.write_text("{}\n", encoding="utf-8")
    artifact.mkdir()
    command = [
        sys.executable,
        str(REPO / "model_evaluation_agent/scripts/eval_qwen36_27b_generate.py"),
        "--batch-id", "qa500v2_test_deadbeef",
        "--model-id", "qwen36_27b_lora",
        "--benchmark-manifest", str(benchmark),
        "--media-manifest", str(media),
        "--media-manifest-sha256", hashlib.sha256(media.read_bytes()).hexdigest(),
        "--artifact", str(artifact),
        "--predictions", str(predictions),
        "--modality", "V",
        "--evaluation-mode", "generative",
        "--do-sample", "false",
        "--temperature", "0",
        "--strict-answer-tags",
    ]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "blocker=verified-multi-root-bootstrap" in completed.stderr
    assert not predictions.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ("worker", "--password", "bad"),
        ("worker", "--api-key=bad"),
        ("worker", "https://user:password@example.invalid/path"),
    ],
)
def test_command_spec_rejects_secrets_in_argv(argv):
    with pytest.raises(CommandValidationError):
        CommandSpec(argv=argv, cwd=".")


def test_command_spec_rejects_unapproved_environment():
    with pytest.raises(CommandValidationError, match="allowlisted"):
        CommandSpec(argv=("worker",), cwd=".", env={"RANDOM_SECRET": "value"})
