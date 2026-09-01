from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from motion_eval.core import hash_path
from motion_eval.training_receipt import make_training_receipt, write_training_receipt
from motionllm.training import (
    ArtifactProvenancePaths,
    ArtifactValidationError,
    ReloadVerificationReceipt,
    publish_artifact_distributed,
    resolve_resume_from_arguments,
    resolve_validated_resume_checkpoint,
    setup_motion_tokens,
    validate_lora_adapter_pairs,
    validate_artifact_policy,
    validate_resume_artifact,
    verify_lora_save_reload,
    verify_processor_save_reload,
    write_finetune_artifact_manifest,
    write_reload_verification_receipt,
    compute_verified_provenance,
)


def provenance(tmp_path):
    values = {}
    for role in (
        "base_artifact",
        "train_data",
        "validation_data",
        "benchmark",
        "leakage_audit",
        "config",
        "code",
        "runner_code",
        "environment",
    ):
        path = tmp_path / f"{role}.txt"
        path.write_text(f"fresh {role}\n", encoding="utf-8")
        values[role] = path
    return ArtifactProvenancePaths(**values)


def validate(manifest, paths, tmp_path, *, mode="full_sft"):
    return validate_resume_artifact(
        manifest,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode=mode,
        allowed_root=tmp_path,
    )


def training_receipt(tmp_path, paths, artifact, *, batch_id, model_id, mode):
    binding, _ = compute_verified_provenance(
        paths,
        batch_id=batch_id,
        model_id=model_id,
        training_mode=mode,
    )
    artifact_hash = hash_path(artifact).digest
    payload = make_training_receipt(
        batch_id=batch_id,
        model_id=model_id,
        backend_id="test_backend",
        model_family="test_family",
        modality="VM",
        training_mode=mode,
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
        train_sha256=binding.train_data_hash,
        validation_sha256=binding.validation_data_hash,
        leakage_audit_sha256=binding.leakage_audit_hash,
        base_artifact_sha256=binding.base_artifact_hash,
        config_sha256=binding.config_hash,
        code_sha256=binding.code_hash,
        runner_code_sha256=binding.runner_code_hash,
        environment_sha256=binding.environment_hash,
        artifact_sha256=artifact_hash,
    )
    destination = tmp_path / f"training_{artifact_hash[:12]}.json"
    if not destination.exists():
        write_training_receipt(destination, payload, root=tmp_path)
    return destination


def reload_receipt(tmp_path, artifact, *, batch_id, model_id):
    artifact_hash = hash_path(artifact).digest
    destination = tmp_path / f"reload_{artifact_hash[:12]}.json"
    if destination.exists():
        return destination
    receipt = ReloadVerificationReceipt(
        batch_id=batch_id,
        model_id=model_id,
        artifact_hash=artifact_hash,
        expected_modules=("__test__",),
        reloaded_modules=("__test__",),
        motion_start_token_id=None,
        motion_end_token_id=None,
        state_hash_before="5" * 64,
        state_hash_after="5" * 64,
        processor_state_hash_before="6" * 64,
        processor_state_hash_after="6" * 64,
        processor_assets_hash="7" * 64,
    )
    return write_reload_verification_receipt(destination, receipt, allowed_root=tmp_path)


def write_formal_manifest(
    manifest, *, artifact_path, provenance_paths, batch_id, model_id,
    training_mode, allowed_root, reload_receipt_path=None,
):
    reload_path = reload_receipt_path or reload_receipt(
        allowed_root, artifact_path, batch_id=batch_id, model_id=model_id
    )
    training_path = training_receipt(
        allowed_root,
        provenance_paths,
        artifact_path,
        batch_id=batch_id,
        model_id=model_id,
        mode=training_mode,
    )
    return write_finetune_artifact_manifest(
        manifest,
        artifact_path=artifact_path,
        provenance_paths=provenance_paths,
        batch_id=batch_id,
        model_id=model_id,
        training_mode=training_mode,
        allowed_root=allowed_root,
        reload_receipt_path=reload_path,
        training_receipt_path=training_path,
    )


def test_artifact_manifest_recomputes_actual_paths_and_content(tmp_path):
    paths = provenance(tmp_path)
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"fresh full-sft weights")
    manifest = tmp_path / "run_manifest.json"
    written = write_formal_manifest(
        manifest,
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    loaded = validate(manifest, paths, tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert loaded.artifact_digest == written.artifact_digest
    assert value["provenance"]["train_data"]["path"] == str(paths.train_data.resolve())

    paths.train_data.write_text("tampered data\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="provenance changed"):
        validate(manifest, paths, tmp_path)


def test_sft_runner_code_is_independent_provenance_and_receipt_binding(tmp_path):
    paths = provenance(tmp_path)
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"fresh weights")
    manifest = tmp_path / "run_manifest.json"
    write_formal_manifest(
        manifest,
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["provenance"]["runner_code"]["path"] == str(
        paths.runner_code.resolve()
    )
    assert (
        payload["provenance"]["runner_code"]["digest"]
        != payload["provenance"]["code"]["digest"]
    )

    paths.runner_code.write_text("tampered runner\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="provenance changed"):
        validate(manifest, paths, tmp_path)


def test_formal_environment_provenance_hashes_dependency_tree_not_only_python(tmp_path):
    paths = provenance(tmp_path)
    paths.environment.unlink()
    site_packages = paths.environment / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    dependency = site_packages / "transformers.py"
    dependency.write_text("version = 'frozen'\n", encoding="utf-8")
    interpreter = paths.environment / "bin" / "python"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"python")

    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"fresh weights")
    manifest = tmp_path / "run_manifest.json"
    write_formal_manifest(
        manifest,
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["provenance"]["environment"]["kind"] == "directory"
    assert payload["provenance"]["environment"]["file_count"] == 2

    dependency.write_text("version = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="provenance changed"):
        validate(manifest, paths, tmp_path)


def test_motion_vqvae_asset_is_hashed_as_first_class_provenance(tmp_path):
    base_paths = provenance(tmp_path)
    motion_vqvae = tmp_path / "motion_vqvae.pth"
    motion_vqvae.write_bytes(b"exact-loaded-motion-asset")
    paths = ArtifactProvenancePaths(
        **base_paths.to_dict(), motion_vqvae=motion_vqvae
    )
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    write_formal_manifest(
        manifest,
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["provenance"]["motion_vqvae"]["path"] == str(
        motion_vqvae.resolve()
    )
    motion_vqvae.write_bytes(b"tampered")
    with pytest.raises(ArtifactValidationError, match="provenance changed"):
        validate(manifest, paths, tmp_path)


def test_empty_provenance_or_artifact_is_rejected(tmp_path):
    paths = provenance(tmp_path)
    paths.code.write_text("", encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(ArtifactValidationError, match="code must contain"):
        write_finetune_artifact_manifest(
            tmp_path / "manifest.json",
            artifact_path=artifact,
            provenance_paths=paths,
            batch_id="batch_1",
            model_id="model_1",
            training_mode="full_sft",
            allowed_root=tmp_path,
        )
    paths.code.write_text("code", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="artifact must contain"):
        write_finetune_artifact_manifest(
            tmp_path / "manifest.json",
            artifact_path=artifact,
            provenance_paths=paths,
            batch_id="batch_1",
            model_id="model_1",
            training_mode="full_sft",
            allowed_root=tmp_path,
        )


def test_lora_manifest_requires_matching_external_reload_receipt(tmp_path):
    paths = provenance(tmp_path)
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter.bin").write_bytes(b"adapter")
    artifact_hash = hash_path(artifact).digest
    receipt = ReloadVerificationReceipt(
        batch_id="batch_1",
        model_id="model_1",
        artifact_hash=artifact_hash,
        expected_modules=("motion_proj",),
        reloaded_modules=("motion_proj",),
        motion_start_token_id=1,
        motion_end_token_id=2,
        state_hash_before="b" * 64,
        state_hash_after="b" * 64,
        processor_state_hash_before="c" * 64,
        processor_state_hash_after="c" * 64,
        processor_assets_hash="d" * 64,
    )
    reload_path = write_reload_verification_receipt(
        tmp_path / "reload.json", receipt, allowed_root=tmp_path
    )
    write_formal_manifest(
        tmp_path / "manifest.json",
        artifact_path=artifact,
        provenance_paths=paths,
        batch_id="batch_1",
        model_id="model_1",
        training_mode="lora_sft",
        allowed_root=tmp_path,
        reload_receipt_path=reload_path,
    )
    with pytest.raises(ArtifactValidationError, match="inside the artifact"):
        nested = artifact / "nested_reload.json"
        nested.write_text(reload_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_finetune_artifact_manifest(
            tmp_path / "second.json",
            artifact_path=artifact,
            provenance_paths=paths,
            batch_id="batch_1",
            model_id="model_1",
            training_mode="lora_sft",
            allowed_root=tmp_path,
            reload_receipt_path=nested,
        )


def test_motion_reload_receipt_rejects_empty_modules_on_construct_and_read(tmp_path):
    fields = {
        "batch_id": "batch_1",
        "model_id": "motion_model",
        "artifact_hash": "a" * 64,
        "expected_modules": (),
        "reloaded_modules": (),
        "motion_start_token_id": 1,
        "motion_end_token_id": 2,
        "state_hash_before": "b" * 64,
        "state_hash_after": "b" * 64,
        "processor_state_hash_before": "c" * 64,
        "processor_state_hash_after": "c" * 64,
        "processor_assets_hash": "d" * 64,
    }
    with pytest.raises(ArtifactValidationError, match="non-empty expected modules"):
        ReloadVerificationReceipt(**fields)

    paths = provenance(tmp_path)
    artifact = tmp_path / "motion_adapter"
    artifact.mkdir()
    (artifact / "adapter.bin").write_bytes(b"adapter")
    fields["artifact_hash"] = hash_path(artifact).digest
    invalid_payload = {
        "schema_version": "2.0",
        "status": "reload_verified",
        **fields,
        "expected_modules": [],
        "reloaded_modules": [],
    }
    receipt_path = tmp_path / "invalid_motion_reload.json"
    receipt_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="non-empty expected modules"):
        write_finetune_artifact_manifest(
            tmp_path / "invalid_motion_manifest.json",
            artifact_path=artifact,
            provenance_paths=paths,
            batch_id="batch_1",
            model_id="motion_model",
            training_mode="lora_sft",
            allowed_root=tmp_path,
            reload_receipt_path=receipt_path,
        )


def test_formal_policy_requires_manifest_and_only_explicit_unsafe_escape(tmp_path):
    missing = SimpleNamespace(artifact_manifest_path=None, unsafe_legacy_no_manifest=False)
    with pytest.raises(ValueError, match="formal training requires"):
        validate_artifact_policy(missing, training_mode="full_sft")
    unsafe = SimpleNamespace(
        artifact_manifest_path=None,
        unsafe_legacy_no_manifest=True,
        resume_manifest=None,
    )
    assert validate_artifact_policy(unsafe, training_mode="full_sft") is False
    assert resolve_resume_from_arguments(unsafe, training_mode="full_sft") is None


class FakeDistributed:
    def __init__(self, rank):
        self.rank = rank
        self.barriers = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return 1

    def broadcast_object_list(self, value, src=0):
        assert src == 0

    def all_gather_object(self, output, value):
        output[0] = value

    def barrier(self):
        self.barriers += 1


def test_only_primary_rank_publishes_then_all_ranks_revalidate(tmp_path):
    paths = provenance(tmp_path)
    artifact = tmp_path / "full"
    artifact.mkdir()
    (artifact / "weights").write_text("weights", encoding="utf-8")
    reload_path = reload_receipt(
        tmp_path, artifact, batch_id="batch_1", model_id="model_1"
    )
    training_path = training_receipt(
        tmp_path,
        paths,
        artifact,
        batch_id="batch_1",
        model_id="model_1",
        mode="full_sft",
    )
    arguments = SimpleNamespace(
        artifact_manifest_path=str(tmp_path / "manifest.json"),
        unsafe_legacy_no_manifest=False,
        artifact_root=str(tmp_path),
        batch_id="batch_1",
        model_registry_id="model_1",
        base_artifact_path=paths.base_artifact,
        train_data_path=paths.train_data,
        validation_data_path=paths.validation_data,
        benchmark_path=paths.benchmark,
        leakage_audit_path=paths.leakage_audit,
        config_path=paths.config,
        code_path=paths.code,
        runner_code_path=paths.runner_code,
        environment_path=paths.environment,
        reload_receipt_path=str(reload_path),
        training_receipt_path=str(training_path),
        batch_receipt_sha256="3" * 64,
        attempt_sha256="4" * 64,
        resume_manifest=None,
    )
    primary_dist = FakeDistributed(rank=0)
    publish_artifact_distributed(
        arguments,
        training_mode="full_sft",
        artifact_path=artifact,
        torch_module=SimpleNamespace(distributed=primary_dist),
    )
    original = (tmp_path / "manifest.json").read_bytes()
    secondary_dist = FakeDistributed(rank=1)
    publish_artifact_distributed(
        arguments,
        training_mode="full_sft",
        artifact_path=artifact,
        torch_module=SimpleNamespace(distributed=secondary_dist),
    )
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert primary_dist.barriers == secondary_dist.barriers == 2


class ReloadTokenizer:
    def __init__(self):
        self.vocab = {"base": 0}
        self.additional_special_tokens = []

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_ids_to_tokens(self, token_id):
        return next((key for key, value in self.vocab.items() if value == token_id), None)

    def add_special_tokens(self, payload, **kwargs):
        del kwargs
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added


class ReloadModule:
    def __init__(self, values):
        self.values = values

    def state_dict(self):
        return {"weight": self.values}


class ReloadModel:
    def __init__(self, values):
        self.config = type("Config", (), {})()
        self.embedding = type("Embedding", (), {"num_embeddings": 1})()
        self.motion_proj = ReloadModule(values)
        self.visual = ReloadModule(values)

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size):
        self.embedding.num_embeddings = size


class ReloadParameter:
    def __init__(self, values, *, requires_grad):
        self.values = values
        self.requires_grad = requires_grad

    def numpy(self):
        import numpy as np

        return np.asarray(self.values, dtype=np.float32)


class ReloadPeftModel(ReloadModel):
    def __init__(self, module_values, *, lora_a=(1.0, 2.0), lora_b=(3.0, 4.0), trainable=True):
        super().__init__(module_values)
        self._named = {
            "base.lora_A.default.weight": ReloadParameter(
                lora_a, requires_grad=trainable
            ),
            "base.lora_B.default.weight": ReloadParameter(
                lora_b, requires_grad=trainable
            ),
            "base.frozen.weight": ReloadParameter((9.0,), requires_grad=False),
        }

    def named_parameters(self):
        return iter(self._named.items())


class ReloadProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def processor_artifact(tmp_path):
    artifact = tmp_path / "processor_artifact"
    artifact.mkdir(exist_ok=True)
    (artifact / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (artifact / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return artifact


def test_lora_save_reload_receipt_hashes_custom_state_without_pickle(tmp_path):
    original = ReloadPeftModel([1.0, 2.0])
    reloaded = ReloadPeftModel([1.0, 2.0], trainable=False)
    tokenizer = ReloadTokenizer()
    reloaded_tokenizer = ReloadTokenizer()
    setup_motion_tokens(tokenizer, original)
    setup_motion_tokens(reloaded_tokenizer, reloaded)
    receipt = verify_lora_save_reload(
        original,
        reloaded,
        tokenizer=tokenizer,
        reloaded_tokenizer=reloaded_tokenizer,
        processor=ReloadProcessor(tokenizer),
        reloaded_processor=ReloadProcessor(reloaded_tokenizer),
        processor_artifact_path=processor_artifact(tmp_path),
        module_names=("motion_proj",),
        batch_id="batch_1",
        model_id="model_1",
        artifact_hash="a" * 64,
    )
    assert receipt.state_hash_before == receipt.state_hash_after
    reloaded.motion_proj.values = [1.0, 3.0]
    with pytest.raises(ArtifactValidationError, match="differ"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=("motion_proj",),
            batch_id="batch_1",
            model_id="model_1",
            artifact_hash="a" * 64,
        )


def test_video_only_lora_reload_does_not_require_motion_tokens(tmp_path):
    original = ReloadPeftModel([1.0, 2.0])
    reloaded = ReloadPeftModel([1.0, 2.0], trainable=False)
    tokenizer = ReloadTokenizer()
    reloaded_tokenizer = ReloadTokenizer()
    artifact = processor_artifact(tmp_path)
    (artifact / "adapter_model.safetensors").write_bytes(b"adapter")
    artifact_hash = hash_path(artifact).digest

    receipt = verify_lora_save_reload(
        original,
        reloaded,
        tokenizer=tokenizer,
        reloaded_tokenizer=reloaded_tokenizer,
        processor=ReloadProcessor(tokenizer),
        reloaded_processor=ReloadProcessor(reloaded_tokenizer),
        processor_artifact_path=artifact,
        module_names=(),
        batch_id="batch_1",
        model_id="video_model",
        artifact_hash=artifact_hash,
        supports_motion=False,
    )

    assert receipt.motion_start_token_id is None
    assert receipt.motion_end_token_id is None
    receipt_path = write_reload_verification_receipt(
        tmp_path / "video_reload.json", receipt, allowed_root=tmp_path
    )
    write_formal_manifest(
        tmp_path / "video_manifest.json",
        artifact_path=artifact,
        provenance_paths=provenance(tmp_path),
        batch_id="batch_1",
        model_id="video_model",
        training_mode="lora_sft",
        allowed_root=tmp_path,
        reload_receipt_path=receipt_path,
    )


def test_motion_lora_reload_still_requires_motion_tokens(tmp_path):
    original = ReloadPeftModel([1.0, 2.0])
    reloaded = ReloadPeftModel([1.0, 2.0], trainable=False)
    tokenizer = ReloadTokenizer()
    reloaded_tokenizer = ReloadTokenizer()

    with pytest.raises(ValueError, match="motion_start"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=("motion_proj",),
            batch_id="batch_1",
            model_id="motion_model",
            artifact_hash="f" * 64,
            supports_motion=True,
        )

    setup_motion_tokens(tokenizer, original)
    setup_motion_tokens(reloaded_tokenizer, reloaded)
    with pytest.raises(ArtifactValidationError, match="modules_to_save"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=(),
            batch_id="batch_1",
            model_id="motion_model",
            artifact_hash="f" * 64,
            supports_motion=True,
        )


def test_lora_reload_compares_every_adapter_ab_and_modules_to_save_entry(tmp_path):
    original = ReloadPeftModel([5.0, 6.0], trainable=True)
    reloaded = ReloadPeftModel([5.0, 6.0], trainable=False)
    tokenizer = ReloadTokenizer()
    reloaded_tokenizer = ReloadTokenizer()
    setup_motion_tokens(tokenizer, original)
    setup_motion_tokens(reloaded_tokenizer, reloaded)
    receipt = verify_lora_save_reload(
        original,
        reloaded,
        tokenizer=tokenizer,
        reloaded_tokenizer=reloaded_tokenizer,
        processor=ReloadProcessor(tokenizer),
        reloaded_processor=ReloadProcessor(reloaded_tokenizer),
        processor_artifact_path=processor_artifact(tmp_path),
        module_names=("motion_proj",),
        batch_id="batch_1",
        model_id="model_1",
        artifact_hash="b" * 64,
    )
    assert receipt.state_hash_before == receipt.state_hash_after

    reloaded._named["base.lora_A.default.weight"] = ReloadParameter(
        (1.0, 99.0), requires_grad=False
    )
    with pytest.raises(ArtifactValidationError, match="content differs"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=("motion_proj",),
            batch_id="batch_1",
            model_id="model_1",
            artifact_hash="b" * 64,
        )

    reloaded = ReloadPeftModel([5.0, 6.0], trainable=False)
    setup_motion_tokens(reloaded_tokenizer, reloaded)
    reloaded._named["extra.lora_A.default.weight"] = ReloadParameter(
        (7.0,), requires_grad=False
    )
    reloaded._named["extra.lora_B.default.weight"] = ReloadParameter(
        (8.0,), requires_grad=False
    )
    with pytest.raises(ArtifactValidationError, match="parameter names differ"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=("motion_proj",),
            batch_id="batch_1",
            model_id="model_1",
            artifact_hash="b" * 64,
        )


def test_lora_pair_validation_is_per_exact_adapter_prefix():
    assert validate_lora_adapter_pairs(
        ["q.lora_A.default.weight", "q.lora_B.default.weight"]
    )
    with pytest.raises(ArtifactValidationError, match="every LoRA adapter prefix"):
        validate_lora_adapter_pairs(
            ["q.lora_A.default.weight", "k.lora_B.default.weight"]
        )
    with pytest.raises(ArtifactValidationError, match="exactly one A and one B"):
        validate_lora_adapter_pairs(
            [
                "q.lora_A.default.weight",
                "q.lora_A.default.weight",
                "q.lora_B.default.weight",
            ]
        )


def test_processor_reload_is_part_of_lora_receipt(tmp_path):
    original = ReloadPeftModel([1.0])
    reloaded = ReloadPeftModel([1.0], trainable=False)
    tokenizer = ReloadTokenizer()
    reloaded_tokenizer = ReloadTokenizer()
    setup_motion_tokens(tokenizer, original)
    setup_motion_tokens(reloaded_tokenizer, reloaded)
    reloaded_tokenizer.special_tokens_map = {"drift": "<drift>"}
    with pytest.raises(ArtifactValidationError, match="processor or tokenizer state differs"):
        verify_lora_save_reload(
            original,
            reloaded,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=ReloadProcessor(tokenizer),
            reloaded_processor=ReloadProcessor(reloaded_tokenizer),
            processor_artifact_path=processor_artifact(tmp_path),
            module_names=("motion_proj",),
            batch_id="batch_1",
            model_id="model_1",
            artifact_hash="e" * 64,
        )


def test_processor_reload_requires_saved_tokenizer_and_processor_assets(tmp_path):
    artifact = tmp_path / "missing_processor_assets"
    artifact.mkdir()
    processor = ReloadProcessor(ReloadTokenizer())
    with pytest.raises(ArtifactValidationError, match="no processor/tokenizer assets"):
        verify_processor_save_reload(
            processor,
            ReloadProcessor(ReloadTokenizer()),
            artifact_path=artifact,
        )


def test_formal_resume_requires_one_complete_hf_checkpoint(tmp_path):
    paths = provenance(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "resume_manifest.json"
    write_formal_manifest(
        manifest,
        artifact_path=checkpoint,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    with pytest.raises(ValueError, match="exactly one complete checkpoint"):
        resolve_validated_resume_checkpoint(
            manifest,
            provenance_paths=paths,
            batch_id="qa500v2_deadbeef",
            model_id="motionr1_vm_lora",
            training_mode="full_sft",
            allowed_root=tmp_path,
        )

    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth",
    ):
        (checkpoint / name).write_bytes(b"state")
    manifest.unlink()
    write_formal_manifest(
        manifest,
        artifact_path=checkpoint,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    )
    assert resolve_validated_resume_checkpoint(
        manifest,
        provenance_paths=paths,
        batch_id="qa500v2_deadbeef",
        model_id="motionr1_vm_lora",
        training_mode="full_sft",
        allowed_root=tmp_path,
    ) == checkpoint.resolve()


def test_formal_grpo_resume_is_rejected_until_state_proof_exists(tmp_path):
    with pytest.raises(ValueError, match="GRPO resume is not implemented"):
        resolve_validated_resume_checkpoint(
            tmp_path / "unused.json",
            provenance_paths=provenance(tmp_path),
            batch_id="batch_1",
            model_id="model_1",
            training_mode="grpo",
            allowed_root=tmp_path,
        )
