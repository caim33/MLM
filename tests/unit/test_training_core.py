from __future__ import annotations

from types import SimpleNamespace

import pytest

from motionllm.training import (
    ExplicitModelFactory,
    FreezePolicy,
    LoraSavePolicy,
    ModelFamily,
    apply_freeze_policy,
    bind_model_base_provenance,
    bind_motion_length_divisor,
    bind_motion_vqvae_provenance,
    bind_supervised_data_provenance,
    make_provenance_bound_supervised_data_module,
    require_explicit_formal_seed,
    resolve_modules_to_save,
    resolve_eval_enabled,
    resolve_motion_length_divisor,
    setup_motion_tokens,
    validate_fresh_formal_output_directory,
    verify_motion_tokens,
)
from motionllm.training.model_factory import ModelFactoryError
from motionllm.training.runtime import distributed_rank, is_primary_process
from motionllm.training.artifact import processor_assets_sha256
from motionllm.training.tokens import (
    MotionTokenError,
    bind_model_to_motion_tokens,
    verify_motion_tokenizer_tokens,
)


class FakeParameter:
    def __init__(self, size=1):
        self.requires_grad = None
        self._size = size

    def numel(self):
        return self._size


class FakeModule:
    def __init__(self, count=1):
        self._parameters = [FakeParameter(index + 1) for index in range(count)]

    def parameters(self):
        return iter(self._parameters)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"base": 0}
        self.additional_special_tokens = []

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_ids_to_tokens(self, token_id):
        return next((token for token, value in self.vocab.items() if value == token_id), None)

    def add_special_tokens(self, payload, replace_additional_special_tokens=False):
        del replace_additional_special_tokens
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.additional_special_tokens.append(token)
                added += 1
        return added


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace()
        self.embedding = SimpleNamespace(num_embeddings=1)
        self.resize_calls = []
        self.language_model = FakeModule(2)
        self.lm_head = FakeModule(2)
        self.visual = FakeModule(2)
        self.visual.merger = FakeModule(1)
        self.motion_encoder = FakeModule(2)
        self.motion_prenorm = FakeModule(1)
        self.motion_proj = FakeModule(2)
        self.motion_postnorm = FakeModule(1)
        self.motion_boundary_embed = FakeModule(1)
        self.unclassified = FakeModule(1)

    def parameters(self):
        modules = (
            self.language_model,
            self.lm_head,
            self.visual,
            self.visual.merger,
            self.motion_encoder,
            self.motion_prenorm,
            self.motion_proj,
            self.motion_postnorm,
            self.motion_boundary_embed,
            self.unclassified,
        )
        for module in modules:
            yield from module.parameters()

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size):
        self.resize_calls.append(size)
        self.embedding.num_embeddings = size


def test_explicit_factory_never_guesses_from_checkpoint_path():
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(path=path)

    resolved = []

    def resolver(path):
        resolved.append(path)
        return Loader

    factory = ExplicitModelFactory(class_resolver=resolver)
    model, spec = factory.load_model(
        family="qwen3_vl",
        model_name_or_path="C:/models/a3b-moe-looking-name",
        model_kwargs={"dtype": "fake"},
    )
    assert spec.family is ModelFamily.QWEN3_VL
    assert spec.is_moe is False
    assert resolved == ["transformers.Qwen3VLForConditionalGeneration"]
    assert model.path.endswith("a3b-moe-looking-name")
    assert calls[0][1] == {"dtype": "fake"}


def test_explicit_factory_rejects_missing_or_unknown_family():
    factory = ExplicitModelFactory(class_resolver=lambda _: object)
    for family in (None, "", "qwen3-contains-a"):
        with pytest.raises(ModelFactoryError):
            factory.spec_for(family)


def test_motion_token_setup_is_idempotent_and_resizes_exactly_once():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    first = setup_motion_tokens(tokenizer, model)
    second = setup_motion_tokens(tokenizer, model)
    assert first.added_count == 2
    assert first.resized is True
    assert second.added_count == 0
    assert second.resized is False
    assert model.resize_calls == [3]
    assert verify_motion_tokens(tokenizer, model) == (
        model.config.motion_start_token_id,
        model.config.motion_end_token_id,
    )


def test_reload_binding_never_mutates_a_preverified_disk_tokenizer():
    tokenizer = FakeTokenizer()
    setup_motion_tokens(tokenizer, FakeModel())
    expected_ids = verify_motion_tokenizer_tokens(tokenizer)
    frozen_vocab = tokenizer.get_vocab()

    def forbidden_add(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reload binding must never add tokenizer tokens")

    tokenizer.add_special_tokens = forbidden_add
    fresh_base = FakeModel()
    receipt = bind_model_to_motion_tokens(
        tokenizer, fresh_base, expected_token_ids=expected_ids
    )
    assert receipt.added_count == 0
    assert receipt.resized is True
    assert tokenizer.get_vocab() == frozen_vocab
    assert fresh_base.resize_calls == [len(tokenizer)]
    assert verify_motion_tokens(tokenizer, fresh_base) == expected_ids


def test_nonempty_saved_processor_files_cannot_mask_missing_motion_tokens(tmp_path):
    artifact = tmp_path / "saved_adapter"
    artifact.mkdir()
    (artifact / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (artifact / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert processor_assets_sha256(artifact)

    disk_tokenizer = FakeTokenizer()
    frozen_vocab = disk_tokenizer.get_vocab()
    untouched_base = FakeModel()
    with pytest.raises(MotionTokenError, match="not registered exactly"):
        verify_motion_tokenizer_tokens(disk_tokenizer)
    assert disk_tokenizer.get_vocab() == frozen_vocab
    assert untouched_base.resize_calls == []
    assert not hasattr(untouched_base.config, "motion_start_token_id")


def test_freeze_policy_updates_every_parameter_not_module_attribute():
    model = FakeModel()
    receipt = apply_freeze_policy(
        model,
        FreezePolicy(
            language_model=False,
            lm_head=False,
            visual_encoder=False,
            visual_merger=True,
            motion_encoder=False,
            motion_adapters=True,
            motion_boundary=True,
        ),
    )
    assert all(parameter.requires_grad is False for parameter in model.lm_head.parameters())
    assert all(parameter.requires_grad is False for parameter in model.visual.parameters())
    assert all(parameter.requires_grad is True for parameter in model.visual.merger.parameters())
    assert all(parameter.requires_grad is True for parameter in model.motion_proj.parameters())
    assert all(parameter.requires_grad is False for parameter in model.unclassified.parameters())
    assert receipt.by_component("lm_head").parameter_tensors == 2


def test_lora_modules_to_save_cover_all_present_motion_state():
    model = FakeModel()
    modules = resolve_modules_to_save(
        model,
        LoraSavePolicy(requested_modules=("lm_head",), require_motion_modules=True),
        freeze_policy=FreezePolicy(lm_head=True),
    )
    assert modules == (
        "lm_head",
        "motion_prenorm",
        "motion_proj",
        "motion_postnorm",
        "motion_boundary_embed",
    )
    with pytest.raises(ValueError, match="freeze-policy"):
        resolve_modules_to_save(
            model,
            LoraSavePolicy(requested_modules=("visual",)),
            freeze_policy=FreezePolicy(visual_encoder=False),
        )
    with pytest.raises(ValueError, match="motion_embed no longer exists"):
        resolve_modules_to_save(
            model,
            LoraSavePolicy(requested_modules=("motion_embed",)),
            freeze_policy=FreezePolicy(),
        )


def test_fake_peft_modules_to_save_cannot_unfreeze_forbidden_blocks():
    model = FakeModel()
    policy = FreezePolicy(
        language_model=False,
        lm_head=False,
        visual_encoder=False,
        visual_merger=False,
        motion_encoder=False,
        motion_adapters=True,
        motion_boundary=True,
    )
    apply_freeze_policy(model, policy)
    modules = resolve_modules_to_save(
        model,
        LoraSavePolicy(require_motion_modules=True),
        freeze_policy=policy,
    )
    # Fake PEFT's modules_to_save behavior: every selected module becomes trainable.
    for name in modules:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        for parameter in module.parameters():
            parameter.requires_grad = True
    assert all(not p.requires_grad for p in model.visual.parameters())
    assert all(not p.requires_grad for p in model.motion_encoder.parameters())
    assert all(p.requires_grad for p in model.motion_proj.parameters())


class FakeDistributed:
    def __init__(self, *, available=True, initialized=False, rank=7):
        self.available = available
        self.initialized = initialized
        self.rank = rank

    def is_available(self):
        return self.available

    def is_initialized(self):
        return self.initialized

    def get_rank(self):
        return self.rank


def test_distributed_rank_is_safe_before_initialization():
    torch = SimpleNamespace(distributed=FakeDistributed(initialized=False))
    assert distributed_rank(torch) == 0
    assert is_primary_process(torch) is True
    torch.distributed.initialized = True
    assert distributed_rank(torch) == 7
    assert is_primary_process(torch) is False


def test_eval_switch_honors_arguments_and_dataset_presence():
    dataset = object()
    assert resolve_eval_enabled(SimpleNamespace(do_eval=True, eval_strategy="no"), dataset)
    assert resolve_eval_enabled(SimpleNamespace(do_eval=False, eval_strategy="steps"), dataset)
    assert not resolve_eval_enabled(SimpleNamespace(do_eval=False, eval_strategy="no"), dataset)
    assert not resolve_eval_enabled(SimpleNamespace(do_eval=True, eval_strategy="steps"), None)


def test_stride_four_downsample_bridge_uses_verified_factor_not_two_to_depth():
    model = SimpleNamespace(
        config=SimpleNamespace(
            vqvae_stride_t=4,
            vqvae_down_t=2,
            motion_downsample_factor=16,
        ),
        motion_spec=SimpleNamespace(downsample_factor=16),
    )
    data = SimpleNamespace(motion_length_divisor=None)
    assert resolve_motion_length_divisor(model) == 16
    assert bind_motion_length_divisor(data, model) == 16
    assert data.motion_length_divisor == 16
    assert data.motion_length_divisor != 2**model.config.vqvae_down_t


def test_downsample_bridge_fallback_computes_stride_power_and_rejects_mismatch():
    compatible = SimpleNamespace(
        config=SimpleNamespace(
            vqvae_stride_t=4,
            vqvae_down_t=2,
            motion_downsample_factor=16,
        )
    )
    assert resolve_motion_length_divisor(compatible) == 16

    incompatible = SimpleNamespace(
        config=SimpleNamespace(
            vqvae_stride_t=4,
            vqvae_down_t=2,
            motion_downsample_factor=4,
        )
    )
    with pytest.raises(ValueError, match="disagree"):
        resolve_motion_length_divisor(incompatible)

    explicit_bad_data = SimpleNamespace(motion_length_divisor=4)
    with pytest.raises(ValueError, match="data motion_length_divisor"):
        bind_motion_length_divisor(explicit_bad_data, compatible)


def test_formal_supervised_loader_is_bound_to_exact_train_and_validation_paths(tmp_path):
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    other = tmp_path / "hardcoded.json"
    for path in (train, validation, other):
        path.write_text("[]", encoding="utf-8")
    registry = {
        "train_alias": {"annotation_path": str(train), "data_path": ""},
        "validation_alias": {"annotation_path": str(validation), "data_path": ""},
        "hardcoded": {"annotation_path": str(other), "data_path": ""},
    }

    def resolver(names):
        return [{**registry[name], "sampling_rate": 1.0} for name in names]

    data = SimpleNamespace(dataset_use="train_alias", eval_dataset_use="validation_alias")
    provenance = SimpleNamespace(train_data_path=train, validation_data_path=validation)
    receipt = bind_supervised_data_provenance(
        data, provenance, dataset_resolver=resolver
    )
    assert receipt.train_data_path == train.resolve()
    module = make_provenance_bound_supervised_data_module(
        object(),
        data,
        provenance,
        dataset_resolver=resolver,
        data_module_factory=lambda processor, data_args: {
            "train_dataset": [object()],
            "eval_dataset": [object()],
        },
    )
    assert module["train_dataset"] is not None

    data.dataset_use = "hardcoded"
    with pytest.raises(ValueError, match="train loader path"):
        bind_supervised_data_provenance(data, provenance, dataset_resolver=resolver)
    data.dataset_use = "train_alias,hardcoded"
    with pytest.raises(ValueError, match="combine"):
        bind_supervised_data_provenance(data, provenance, dataset_resolver=resolver)


def test_supervised_loader_detects_alias_drift_during_real_factory_call(tmp_path):
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    replacement = tmp_path / "replacement.json"
    for path in (train, validation, replacement):
        path.write_text("[]", encoding="utf-8")
    registry = {
        "train": train,
        "validation": validation,
    }

    def resolver(names):
        return [
            {"annotation_path": str(registry[name]), "data_path": "", "sampling_rate": 1.0}
            for name in names
        ]

    def mutating_factory(processor, data_args):
        registry["train"] = replacement
        return {"train_dataset": [object()], "eval_dataset": [object()]}

    with pytest.raises(ValueError, match="actual SFT train loader path|resolver changed"):
        make_provenance_bound_supervised_data_module(
            object(),
            SimpleNamespace(dataset_use="train", eval_dataset_use="validation"),
            SimpleNamespace(train_data_path=train, validation_data_path=validation),
            dataset_resolver=resolver,
            data_module_factory=mutating_factory,
        )


def test_motion_vqvae_loader_path_must_equal_formal_provenance(tmp_path):
    actual = tmp_path / "motion_vqvae.pth"
    stale = tmp_path / "stale.pth"
    actual.write_bytes(b"fresh-vqvae")
    stale.write_bytes(b"stale-vqvae")
    motion = SimpleNamespace(motion_vqvae_path=str(actual))
    artifact = SimpleNamespace(motion_vqvae_asset_path=str(actual))
    assert bind_motion_vqvae_provenance(
        motion, artifact, formal_artifact=True, supports_motion=True
    ) == actual.resolve()
    artifact.motion_vqvae_asset_path = str(stale)
    with pytest.raises(ValueError, match="exactly equal"):
        bind_motion_vqvae_provenance(
            motion, artifact, formal_artifact=True, supports_motion=True
        )
    artifact.motion_vqvae_asset_path = None
    with pytest.raises(ValueError, match="requires motion_vqvae_asset_path"):
        bind_motion_vqvae_provenance(
            motion, artifact, formal_artifact=True, supports_motion=True
        )


def test_formal_base_loader_path_is_local_nonempty_and_exactly_bound(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}", encoding="utf-8")
    model = SimpleNamespace(model_name_or_path=str(base))
    artifact = SimpleNamespace(base_artifact_path=str(base))
    receipt = bind_model_base_provenance(model, artifact, formal_artifact=True)
    assert receipt is not None
    assert receipt.model_path == base.resolve()
    assert model.model_name_or_path == str(base.resolve())

    artifact.base_artifact_path = str(other)
    with pytest.raises(ValueError, match="exactly equal"):
        bind_model_base_provenance(model, artifact, formal_artifact=True)
    model.model_name_or_path = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="must exist locally"):
        bind_model_base_provenance(model, artifact, formal_artifact=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    model.model_name_or_path = str(empty)
    artifact.base_artifact_path = str(empty)
    with pytest.raises(ValueError, match="non-empty files"):
        bind_model_base_provenance(model, artifact, formal_artifact=True)


def test_formal_sft_seed_must_be_explicit_once_and_valid():
    arguments = SimpleNamespace(seed=123)
    assert require_explicit_formal_seed(
        arguments, formal_artifact=True, argv=["--seed", "123"]
    ) == 123
    assert require_explicit_formal_seed(
        arguments, formal_artifact=True, argv=["--seed=123"]
    ) == 123
    for argv in ([], ["--seed", "123", "--seed=123"], ["--seed", "--output_dir", "x"]):
        with pytest.raises(ValueError, match="seed"):
            require_explicit_formal_seed(arguments, formal_artifact=True, argv=argv)
    with pytest.raises(ValueError, match="integer"):
        require_explicit_formal_seed(
            SimpleNamespace(seed=True), formal_artifact=True, argv=["--seed", "1"]
        )


def test_formal_supervised_loader_rejects_empty_aliases_same_split_and_empty_data(tmp_path):
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text("[]", encoding="utf-8")
    validation.write_text("[]", encoding="utf-8")
    registry = {"train": train, "validation": validation}

    def resolver(names):
        return [
            {"annotation_path": str(registry[name]), "sampling_rate": 1.0}
            for name in names
        ]

    provenance = SimpleNamespace(
        train_data_path=train, validation_data_path=validation
    )
    for field in ("dataset_use", "eval_dataset_use"):
        data = SimpleNamespace(dataset_use="train", eval_dataset_use="validation")
        setattr(data, field, "")
        with pytest.raises(ValueError, match="one explicit dataset name"):
            bind_supervised_data_provenance(
                data, provenance, dataset_resolver=resolver
            )

    with pytest.raises(ValueError, match="must be distinct"):
        bind_supervised_data_provenance(
            SimpleNamespace(dataset_use="train", eval_dataset_use="train"),
            SimpleNamespace(train_data_path=train, validation_data_path=train),
            dataset_resolver=resolver,
        )
    with pytest.raises(ValueError, match="at least one sample"):
        make_provenance_bound_supervised_data_module(
            object(),
            SimpleNamespace(dataset_use="train", eval_dataset_use="validation"),
            provenance,
            dataset_resolver=resolver,
            data_module_factory=lambda processor, data_args: {
                "train_dataset": [],
                "eval_dataset": [object()],
            },
        )


def test_formal_sft_output_must_be_fresh_and_disjoint_from_sources(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    values = {"base_artifact_path": str(base)}
    for name in (
        "train_data_path",
        "validation_data_path",
        "benchmark_path",
        "leakage_audit_path",
        "config_path",
        "code_path",
        "runner_code_path",
        "environment_path",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        values[name] = str(path)
    arguments = SimpleNamespace(
        **values,
        artifact_root=str(tmp_path),
        output_dir=str(tmp_path / "fresh_output"),
        artifact_manifest_path=str(tmp_path / "manifest.json"),
        reload_receipt_path=str(tmp_path / "reload.json"),
        resume_manifest=None,
        motion_vqvae_asset_path=None,
    )
    assert validate_fresh_formal_output_directory(arguments) == (
        tmp_path / "fresh_output"
    ).resolve()
    stale = tmp_path / "stale_output"
    stale.mkdir()
    (stale / "old.bin").write_bytes(b"old")
    arguments.output_dir = str(stale)
    with pytest.raises(ValueError, match="fresh and empty"):
        validate_fresh_formal_output_directory(arguments)
    arguments.output_dir = str(base / "nested_output")
    with pytest.raises(ValueError, match="overlaps"):
        validate_fresh_formal_output_directory(arguments)
