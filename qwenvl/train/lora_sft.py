import os
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="backslashreplace")

import torch
import numpy as np
import transformers
from transformers import Trainer, TrainerCallback

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: E402
from qwenvl.data.data_processor import make_supervised_data_module  # noqa: E402
from qwenvl.data import data_list, validate_motion_normalization_binding  # noqa: E402
from qwenvl.train.argument import (  # noqa: E402
    DataArguments,
    ModelArguments,
    MotionArguments,
    TrainingArguments,
)
from motionllm.training import (  # noqa: E402
    FreezePolicy,
    LoraSavePolicy,
    OptimizerEvidenceTracker,
    apply_freeze_policy,
    bind_canonical_formal_identity,
    bind_model_base_provenance,
    bind_motion_length_divisor,
    bind_motion_vqvae_provenance,
    capture_formal_provenance_snapshot,
    changed_trainable_tensor_count,
    collect_finite_training_losses,
    completed_step_counts,
    default_model_factory,
    is_primary_process,
    make_provenance_bound_supervised_data_module,
    publish_artifact_distributed,
    require_controller_verified_formal_bootstrap,
    require_explicit_formal_seed,
    resume_starting_global_step,
    resolve_modules_to_save,
    resolve_resume_from_arguments,
    setup_motion_tokens,
    snapshot_trainable_state,
    validate_artifact_policy,
    validate_formal_deepspeed_zero2,
    validate_fresh_formal_output_directory,
    verify_lora_save_reload,
    verify_formal_provenance_unchanged,
    verify_processor_save_reload,
    write_reload_verification_receipt,
    write_training_proof_from_arguments,
)
from motionllm.training.tokens import (
    bind_model_to_motion_tokens,
    verify_motion_tokenizer_tokens,
)
from motion_eval.core import hash_path  # noqa: E402

LoraConfig = None
PeftModel = None
TaskType = None
get_peft_model = None


def _require_peft():
    global LoraConfig, PeftModel, TaskType, get_peft_model
    if LoraConfig is not None:
        return
    try:
        from peft import (
            LoraConfig as _LoraConfig,
            PeftModel as _PeftModel,
            TaskType as _TaskType,
            get_peft_model as _get_peft_model,
        )
    except ImportError as exc:  # pragma: no cover - production-only dependency
        raise RuntimeError(
            "LoRA training requires peft; install the production SFT dependencies."
        ) from exc
    LoraConfig = _LoraConfig
    PeftModel = _PeftModel
    TaskType = _TaskType
    get_peft_model = _get_peft_model

local_rank = None


class TrainingEvidenceCallback(TrainerCallback):
    def __init__(self, tracker: OptimizerEvidenceTracker):
        self.tracker = tracker
        self.accelerator = None

    def bind_accelerator(self, accelerator):
        self.accelerator = accelerator

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        self.tracker.record_pre_optimizer_step(kwargs.get("model"))
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.tracker.record_optimizer_step(self.accelerator)
        return control


@dataclass
class LoraArguments:
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "以逗号分隔的 target modules"},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias 配置，可选 none/lora_only/all"},
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "需要额外保存梯度的模块，逗号分隔，如 visual,visual.merger"
        },
    )
    lora_use_dora: bool = field(
        default=False,
        metadata={"help": "是否启用 DoRA (PEFT use_dora)"},
    )


def rank0_print(*args):
    if is_primary_process(torch):
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """收集 state dict 并保存。"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """Legacy facade over the shared, parameter-level freeze policy."""
    return apply_freeze_policy(model, FreezePolicy.from_legacy_arguments(model_args))


def print_trainable_layers(model, model_args):
    """打印所有可训练的层和参数"""
    rank0_print("=" * 80)
    rank0_print("训练参数配置检查")
    rank0_print("=" * 80)
    
    # 统计可训练参数
    trainable_params = []
    total_params = 0
    trainable_params_count = 0
    
    # 检查 Visual 模块
    if hasattr(model, 'visual'):
        visual_trainable = []
        visual_frozen = []
        for name, param in model.visual.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_count += param.numel()
                visual_trainable.append(name)
                trainable_params.append(('visual', name, param.numel()))
            else:
                visual_frozen.append(name)
        
        rank0_print(f"\n[Visual Module]")
        rank0_print(f"  可训练: {len(visual_trainable)} 个参数组")
        rank0_print(f"  冻结: {len(visual_frozen)} 个参数组")
        if visual_trainable:
            rank0_print(f"  可训练参数组示例 (前5个): {visual_trainable[:5]}")
    
    # 检查 Merger/MLP 模块
    if hasattr(model, 'visual') and hasattr(model.visual, 'merger'):
        merger_trainable = []
        merger_frozen = []
        for name, param in model.visual.merger.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_count += param.numel()
                merger_trainable.append(name)
                trainable_params.append(('merger', name, param.numel()))
            else:
                merger_frozen.append(name)
        
        rank0_print(f"\n[Merger/MLP Module]")
        rank0_print(f"  可训练: {len(merger_trainable)} 个参数组")
        rank0_print(f"  冻结: {len(merger_frozen)} 个参数组")
        if merger_trainable:
            rank0_print(f"  可训练参数组: {merger_trainable}")
    
    # 检查 Language Model 模块
    if hasattr(model, 'language_model'):
        llm_trainable = []
        llm_frozen = []
        for name, param in model.language_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_count += param.numel()
                llm_trainable.append(name)
                trainable_params.append(('language_model', name, param.numel()))
            else:
                llm_frozen.append(name)
        
        rank0_print(f"\n[Language Model Module]")
        rank0_print(f"  可训练: {len(llm_trainable)} 个参数组")
        rank0_print(f"  冻结: {len(llm_frozen)} 个参数组")
        if llm_trainable:
            rank0_print(f"  可训练参数组示例 (前5个): {llm_trainable[:5]}")
    
    # 检查 lm_head
    if hasattr(model, 'lm_head'):
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        lm_head_trainable = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
        total_params += lm_head_params
        if lm_head_trainable > 0:
            trainable_params_count += lm_head_trainable
            trainable_params.append(('lm_head', 'lm_head', lm_head_trainable))
            rank0_print(f"\n[LM Head]")
            rank0_print(f"  可训练: 是 ({lm_head_trainable:,} 参数)")
        else:
            rank0_print(f"\n[LM Head]")
            rank0_print(f"  可训练: 否")
    
    # 检查 Motion Encoder (VQ-VAE)
    if hasattr(model, 'motion_encoder'):
        motion_trainable = []
        motion_frozen = []
        for name, param in model.motion_encoder.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_count += param.numel()
                motion_trainable.append(name)
                trainable_params.append(('motion_encoder', name, param.numel()))
            else:
                motion_frozen.append(name)
        
        rank0_print(f"\n[Motion Encoder (VQ-VAE)]")
        rank0_print(f"  可训练: {len(motion_trainable)} 个参数组")
        rank0_print(f"  冻结: {len(motion_frozen)} 个参数组")
        if motion_trainable:
            rank0_print(f"  可训练参数组示例 (前5个): {motion_trainable[:5]}")
    
    # 检查其他模块（如 motion_embed, motion_proj）
    if hasattr(model, 'motion_embed'):
        motion_embed_params = sum(p.numel() for p in model.motion_embed.parameters())
        motion_embed_trainable = sum(p.numel() for p in model.motion_embed.parameters() if p.requires_grad)
        total_params += motion_embed_params
        if motion_embed_trainable > 0:
            trainable_params_count += motion_embed_trainable
            trainable_params.append(('motion_embed', 'motion_embed', motion_embed_trainable))
            rank0_print(f"\n[Motion Embed]")
            rank0_print(f"  可训练: 是 ({motion_embed_trainable:,} 参数)")
        else:
            rank0_print(f"\n[Motion Embed]")
            rank0_print(f"  可训练: 否")
    
    if hasattr(model, 'motion_proj'):
        motion_proj_trainable = []
        motion_proj_frozen = []
        for name, param in model.motion_proj.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_count += param.numel()
                motion_proj_trainable.append(name)
                trainable_params.append(('motion_proj', name, param.numel()))
            else:
                motion_proj_frozen.append(name)
        
        rank0_print(f"\n[Motion Projection]")
        rank0_print(f"  可训练: {len(motion_proj_trainable)} 个参数组")
        rank0_print(f"  冻结: {len(motion_proj_frozen)} 个参数组")
        if motion_proj_trainable:
            rank0_print(f"  可训练参数组: {motion_proj_trainable}")
    
    # 检查 LoRA 适配器
    if hasattr(model, 'peft_config') or hasattr(model, 'get_peft_model'):
        try:
            from peft import get_peft_model_state_dict
            peft_state_dict = get_peft_model_state_dict(model)
            lora_params = len(peft_state_dict)
            lora_param_count = sum(p.numel() for p in peft_state_dict.values())
            trainable_params_count += lora_param_count
            rank0_print(f"\n[LoRA Adapters]")
            rank0_print(f"  可训练参数组数量: {lora_params}")
            rank0_print(f"  可训练参数示例 (前5个): {list(peft_state_dict.keys())[:5]}")
        except:
            pass
    
    # 打印配置摘要
    rank0_print(f"\n" + "=" * 80)
    rank0_print("训练配置摘要:")
    rank0_print(f"  tune_mm_vision: {model_args.tune_mm_vision}")
    rank0_print(f"  tune_mm_mlp: {model_args.tune_mm_mlp}")
    rank0_print(f"  tune_mm_llm: {model_args.tune_mm_llm}")
    if hasattr(model_args, 'tune_mm_motion'):
        rank0_print(f"  tune_mm_motion: {model_args.tune_mm_motion}")
    
    rank0_print(f"\n参数统计:")
    rank0_print(f"  总参数数量: {total_params:,}")
    rank0_print(f"  可训练参数数量: {trainable_params_count:,}")
    rank0_print(f"  可训练参数比例: {100 * trainable_params_count / total_params if total_params > 0 else 0:.2f}%")
    rank0_print("=" * 80 + "\n")


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else None


def apply_lora(model, lora_args: LoraArguments, freeze_policy: FreezePolicy):
    _require_peft()
    target_modules = _parse_str_list(lora_args.lora_target_modules)
    requested_modules = tuple(_parse_str_list(lora_args.lora_modules_to_save) or ())
    modules_to_save = resolve_modules_to_save(
        model,
        LoraSavePolicy(
            requested_modules=requested_modules,
            preserve_motion_modules=True,
            require_motion_modules=hasattr(model, "motion_proj"),
        ),
        freeze_policy=freeze_policy,
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=list(modules_to_save) or None,
        use_dora=lora_args.lora_use_dora,
    )
    model = get_peft_model(model, lora_config)
    setattr(model, "_motionllm_modules_to_save", tuple(modules_to_save))
    rank0_print("LoRA 参数：")
    model.print_trainable_parameters()
    return model


def _publication_modules_to_save(model, *, supports_motion: bool) -> tuple[str, ...]:
    modules_to_save = tuple(getattr(model, "_motionllm_modules_to_save", ()))
    if supports_motion and not modules_to_save:
        raise ValueError(
            "formal motion LoRA publication requires verified motion modules_to_save"
        )
    return modules_to_save


def train(attn_implementation: str = "flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, MotionArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, motion_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    if lora_args.lora_r <= 0 or lora_args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not math.isfinite(lora_args.lora_dropout) or not 0.0 <= lora_args.lora_dropout < 1.0:
        raise ValueError("LoRA dropout must be finite and in [0, 1)")
    if not math.isfinite(training_args.learning_rate) or training_args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    _require_peft()
    formal_artifact = validate_artifact_policy(training_args, training_mode="lora_sft")
    require_controller_verified_formal_bootstrap(formal_artifact=formal_artifact)
    validate_formal_deepspeed_zero2(
        training_args, formal_artifact=formal_artifact
    )
    if not formal_artifact and is_primary_process(torch):
        rank0_print(
            "WARNING: UNSAFE LEGACY NO-MANIFEST MODE; this output is ineligible for formal evaluation."
        )

    # Seed before model, tokenizer, motion-token, or PEFT initialization.
    seed = require_explicit_formal_seed(
        training_args, formal_artifact=formal_artifact
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rank0_print(f"Random seed set to {seed} for reproducibility.")

    if hasattr(motion_args, "vqvae_norm") and motion_args.vqvae_norm is not None:
        norm = motion_args.vqvae_norm.strip().lower()
        if norm in ("", "none"):
            motion_args.vqvae_norm = None

    local_rank = training_args.local_rank
    if not model_args.model_family:
        raise ValueError(
            "--model_family is required; checkpoint paths are never used to guess model type"
        )
    spec = default_model_factory.spec_for(model_args.model_family)
    canonical_identity = bind_canonical_formal_identity(
        model_args,
        training_args,
        model_spec=spec,
        formal_artifact=formal_artifact,
    )
    if formal_artifact:
        validate_fresh_formal_output_directory(training_args)
    provenance_snapshot = capture_formal_provenance_snapshot(
        training_args,
        canonical_identity=canonical_identity,
        training_mode="lora_sft",
        formal_artifact=formal_artifact,
        torch_module=torch,
    )
    bind_model_base_provenance(
        model_args,
        training_args,
        formal_artifact=formal_artifact,
        provenance_snapshot=provenance_snapshot,
    )
    bind_motion_vqvae_provenance(
        motion_args,
        training_args,
        formal_artifact=formal_artifact,
        supports_motion=spec.supports_motion,
        provenance_snapshot=provenance_snapshot,
    )
    resume_checkpoint = resolve_resume_from_arguments(
        training_args,
        training_mode="lora_sft",
        provenance_snapshot=provenance_snapshot,
    )
    if formal_artifact:
        validate_fresh_formal_output_directory(
            training_args, resume_checkpoint=resume_checkpoint
        )
    os.makedirs(training_args.output_dir, exist_ok=True)
    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "attn_implementation": attn_implementation,
        "dtype": (torch.bfloat16 if training_args.bf16 else None),
    }
    if spec.supports_motion:
        dataset_uses = [data_args.dataset_use]
        if data_args.eval_dataset_use:
            dataset_uses.append(data_args.eval_dataset_use)
        normalization_mean, normalization_std = validate_motion_normalization_binding(
            dataset_uses,
            motion_mean_path=motion_args.motion_normalization_mean_path,
            motion_std_path=motion_args.motion_normalization_std_path,
        )
        normalization_overrides = {
            "motion_normalization_mean_path": str(normalization_mean),
            "motion_normalization_std_path": str(normalization_std),
        }
        vqvae_kwargs = {
            "vqvae_nb_code": motion_args.vqvae_nb_code,
            "vqvae_code_dim": motion_args.vqvae_code_dim,
            "vqvae_output_emb_width": motion_args.vqvae_output_emb_width,
            "vqvae_down_t": motion_args.vqvae_down_t,
            "vqvae_stride_t": motion_args.vqvae_stride_t,
            "vqvae_width": motion_args.vqvae_width,
            "vqvae_depth": motion_args.vqvae_depth,
            "vqvae_dilation_growth_rate": motion_args.vqvae_dilation_growth_rate,
            "vqvae_activation": motion_args.vqvae_activation,
            "vqvae_norm": motion_args.vqvae_norm
            if motion_args.vqvae_norm and motion_args.vqvae_norm.strip()
            else None,
        }
        vqvae_kwargs = {k: v for k, v in vqvae_kwargs.items() if v is not None}
        model_kwargs.update(
            {
                "vqvae_path": motion_args.motion_vqvae_path,
                "motion_dataname": motion_args.motion_dataname,
                "motion_quantizer": motion_args.motion_quantizer,
                "motion_config_overrides": normalization_overrides or None,
                **vqvae_kwargs,
            }
        )
    bundle = default_model_factory.load_bundle(
        family=model_args.model_family,
        model_name_or_path=model_args.model_name_or_path,
        model_kwargs=model_kwargs,
    )
    model, processor, spec = bundle.model, bundle.processor, bundle.spec
    data_args.model_type = spec.data_model_type
    if spec.supports_motion:
        bind_motion_length_divisor(data_args, model)
        validate_motion_normalization_binding(
            dataset_uses,
            motion_mean_path=normalization_mean,
            motion_std_path=normalization_std,
            expected_motion_dim=int(model.motion_spec.input_dim),
        )

    print(
        f"the initialized family is {spec.family.value}; class={model.__class__.__name__}"
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor must expose tokenizer")
    if spec.supports_motion:
        token_receipt = setup_motion_tokens(tokenizer, model)
        rank0_print(f"Motion token setup: {token_receipt.to_dict()}")
        placeholder_id = getattr(model.config, "motion_placeholder_token_id", None)
        if placeholder_id is None:
            raise ValueError("model config did not bind motion_placeholder_token_id")
        data_args.motion_placeholder_token_id = int(placeholder_id)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"

    freeze_policy = FreezePolicy.from_legacy_arguments(model_args)
    apply_freeze_policy(model, freeze_policy)
    model = apply_lora(model, lora_args, freeze_policy)
    initial_trainable_snapshot = (
        snapshot_trainable_state(model) if formal_artifact else None
    )

    if is_primary_process(torch):
        print_trainable_layers(model, model_args)
        rank0_print("\nLoRA 参数统计:")
        model.print_trainable_parameters()

    if formal_artifact:
        data_module = make_provenance_bound_supervised_data_module(
            processor,
            data_args,
            training_args,
            dataset_resolver=data_list,
            data_module_factory=make_supervised_data_module,
        )
    else:
        data_module = make_supervised_data_module(processor, data_args=data_args)
    evidence_tracker = OptimizerEvidenceTracker(torch)
    evidence_callback = TrainingEvidenceCallback(evidence_tracker)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[evidence_callback],
        **data_module,
    )
    evidence_callback.bind_accelerator(trainer.accelerator)

    starting_global_step = resume_starting_global_step(resume_checkpoint)
    if resume_checkpoint is not None:
        logging.info("validated hash-bound checkpoint found; resuming training")
        train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        train_result = trainer.train()
    planned_steps, actual_steps = completed_step_counts(
        trainer,
        evidence_tracker,
        starting_global_step=starting_global_step,
    )
    finite_losses = collect_finite_training_losses(train_result, trainer)
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir
    )
    if is_primary_process(torch):
        processor.save_pretrained(training_args.output_dir)

    distributed = getattr(torch, "distributed", None)
    if (
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    ):
        distributed.barrier()

    provenance_post_sha256 = verify_formal_provenance_unchanged(
        training_args,
        snapshot=provenance_snapshot,
        canonical_identity=canonical_identity,
        training_mode="lora_sft",
        formal_artifact=formal_artifact,
        torch_module=torch,
    )

    if formal_artifact and is_primary_process(torch):
        artifact_info = hash_path(
            training_args.output_dir,
            symlink_policy="reject",
        )
        reloaded_bundle = default_model_factory.load_bundle(
            family=model_args.model_family,
            model_name_or_path=model_args.model_name_or_path,
            model_kwargs=model_kwargs,
        )
        reloaded_base = reloaded_bundle.model
        reloaded_processor = transformers.AutoProcessor.from_pretrained(
            training_args.output_dir
        )
        reloaded_tokenizer = getattr(reloaded_processor, "tokenizer", None)
        if reloaded_tokenizer is None:
            raise ValueError("saved processor must expose tokenizer during reload verification")
        if spec.supports_motion:
            # These checks are intentionally read-only.  A missing on-disk token must
            # fail before any model mutation or PEFT adapter load can mask the defect.
            original_motion_ids = verify_motion_tokenizer_tokens(tokenizer)
            disk_motion_ids = verify_motion_tokenizer_tokens(reloaded_tokenizer)
            if disk_motion_ids != original_motion_ids:
                raise ValueError(
                    "saved processor motion boundary token IDs differ from training state"
                )
        verify_processor_save_reload(
            processor,
            reloaded_processor,
            artifact_path=training_args.output_dir,
        )
        if spec.supports_motion:
            bind_model_to_motion_tokens(
                reloaded_tokenizer,
                reloaded_base,
                expected_token_ids=original_motion_ids,
            )
        reloaded_model = PeftModel.from_pretrained(
            reloaded_base,
            training_args.output_dir,
            is_trainable=False,
        )
        if initial_trainable_snapshot is None or canonical_identity is None:
            raise RuntimeError("formal LoRA training state evidence was not initialized")
        final_trainable_snapshot = snapshot_trainable_state(
            reloaded_model,
            parameter_names=tuple(initial_trainable_snapshot.tensor_sha256),
        )
        changed_tensors = changed_trainable_tensor_count(
            initial_trainable_snapshot, final_trainable_snapshot
        )
        modules_to_save = _publication_modules_to_save(
            model, supports_motion=spec.supports_motion
        )
        reload_receipt = verify_lora_save_reload(
            model,
            reloaded_model,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=processor,
            reloaded_processor=reloaded_processor,
            processor_artifact_path=training_args.output_dir,
            module_names=modules_to_save,
            batch_id=training_args.batch_id,
            model_id=training_args.model_registry_id,
            artifact_hash=artifact_info.digest,
            supports_motion=spec.supports_motion,
        )
        write_reload_verification_receipt(
            training_args.reload_receipt_path,
            reload_receipt,
            allowed_root=training_args.artifact_root,
            overwrite=False,
        )
        write_training_proof_from_arguments(
            training_args,
            training_mode="lora_sft",
            backend_id="qwenvl.train.lora_sft",
            canonical_identity=canonical_identity,
            artifact_path=training_args.output_dir,
            planned_steps=planned_steps,
            actual_steps=actual_steps,
            finite_losses=finite_losses,
            nonzero_finite_gradient_steps=(
                evidence_tracker.nonzero_finite_gradient_steps
            ),
            max_gradient=evidence_tracker.max_gradient,
            initial_snapshot=initial_trainable_snapshot,
            final_snapshot=final_trainable_snapshot,
            changed_tensor_count=changed_tensors,
            provenance_snapshot=provenance_snapshot,
            provenance_post_sha256=provenance_post_sha256,
        )

    publish_artifact_distributed(
        training_args,
        training_mode="lora_sft",
        artifact_path=training_args.output_dir,
        torch_module=torch,
        provenance_snapshot=provenance_snapshot,
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
