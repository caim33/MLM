# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import math
import random
import torch
import transformers
import sys
import numpy as np
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="backslashreplace")

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.data import data_list, validate_motion_normalization_binding
from qwenvl.train.argument import (
    ModelArguments,
    MotionArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
from motionllm.training import (
    FreezePolicy,
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
    resolve_resume_from_arguments,
    setup_motion_tokens,
    snapshot_trainable_state,
    validate_artifact_policy,
    validate_formal_deepspeed_zero2,
    validate_fresh_formal_output_directory,
    verify_full_save_reload,
    verify_formal_provenance_unchanged,
    write_reload_verification_receipt,
    write_training_proof_from_arguments,
)
from motion_eval.core import hash_path

local_rank = None


def _is_primary_process_for_logging() -> bool:
    """
    是否应在终端/日志里打印「主进程」信息。
    注意：DeepSpeed + HF 时 `training_args.local_rank` 常为 -1，不能只用 local_rank==0。
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # 未初始化分布式：单进程
    lr = local_rank
    return lr in (None, -1, 0)


def rank0_print(*args, **kwargs):
    """主进程打印（与 print_trainable_layers 使用的 rank0 一致）；默认 flush 便于 tee。"""
    kwargs.setdefault("flush", True)
    if _is_primary_process_for_logging():
        print(*args, **kwargs)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

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


def _unwrap_model(model):
    """DDP / DeepSpeed 等包装时取内层 module。"""
    return model.module if hasattr(model, "module") else model


def _is_global_rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _grad_l2_norm_sq(param_iter):
    """
    各子模块梯度 L2 范数的平方（全局）。

    单机：对本 rank 上 param_iter 里各 param.grad 求和 ||g||^2。
    分布式 / DeepSpeed ZeRO-3：每个 rank 只持有部分参数的梯度分片，对本地分片求 ||g_local||^2 后
    再 all_reduce(SUM)，得到全局 ||g||^2（与整向量拼接后求范数一致）。
    """
    total_sq_local = 0.0
    n_with_grad = 0
    n_requires_grad = 0
    device = None
    for p in param_iter:
        if device is None:
            device = p.device
        if p.requires_grad:
            n_requires_grad += 1
        if p.grad is not None:
            total_sq_local += float(p.grad.detach().float().pow(2).sum().item())
            n_with_grad += 1

    total_sq = total_sq_local
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([total_sq_local], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_sq = float(t.item())

    return total_sq, n_with_grad, n_requires_grad


def format_grad_line(name, total_sq, n_with_grad, n_requires_grad):
    norm = total_sq ** 0.5
    if n_requires_grad == 0:
        return f"  {name}: no trainable params (frozen or missing)"
    if n_with_grad == 0:
        return (
            f"  {name}: ||g||_2=0.000e+00 (no grad tensors; likely frozen or unused in graph) "
            f"[requires_grad params: {n_requires_grad}]"
        )
    return f"  {name}: ||g||_2={norm:.4e}  [params with grad: {n_with_grad}/{n_requires_grad}]"


def log_module_gradient_norms(model, log_fn=print):
    """
    打印 language_model、vision encoder / adapter、motion encoder / adapter 的梯度 L2 范数。
    vision encoder = visual 中除 merger 外的参数；vision adapter = visual.merger。
    motion adapter = motion_prenorm + motion_proj (+ motion_postnorm 若存在)。

    注意：含分布式 all_reduce，必须在所有 rank 上调用；仅 global rank 0 会 log_fn 输出。
    DeepSpeed ZeRO-3 下为全局范数（分片梯度平方和跨 rank 相加）。
    """
    m = _unwrap_model(model)
    if _is_global_rank0():
        log_fn("--- grad norm (pre optimizer step, global ||g||_2 if multi-GPU / ZeRO-3) ---")
    def _log(line):
        if _is_global_rank0():
            log_fn(line)

    # 1) LLM
    if hasattr(m, "language_model"):
        sq, nw, nr = _grad_l2_norm_sq(m.language_model.parameters())
        _log(format_grad_line("language_model", sq, nw, nr))
    else:
        _log("  language_model: N/A")
    if hasattr(m, "lm_head") and m.lm_head is not None:
        sq, nw, nr = _grad_l2_norm_sq(m.lm_head.parameters())
        _log(format_grad_line("lm_head", sq, nw, nr))
    # 2) Vision
    if hasattr(m, "visual") and m.visual is not None:
        merger_ids = set()
        if hasattr(m.visual, "merger") and m.visual.merger is not None:
            merger_ids = {id(p) for p in m.visual.merger.parameters()}
        enc_params = [p for p in m.visual.parameters() if id(p) not in merger_ids]
        if enc_params:
            sq, nw, nr = _grad_l2_norm_sq(enc_params)
            _log(format_grad_line("vision_encoder (visual minus merger)", sq, nw, nr))
        else:
            _log("  vision_encoder: N/A (empty)")
        if hasattr(m.visual, "merger") and m.visual.merger is not None:
            sq, nw, nr = _grad_l2_norm_sq(m.visual.merger.parameters())
            _log(format_grad_line("vision_adapter (visual.merger)", sq, nw, nr))
        else:
            _log("  vision_adapter (merger): N/A")
    else:
        _log("  vision_encoder: N/A")
        _log("  vision_adapter: N/A")
    # 3) Motion
    if hasattr(m, "motion_encoder") and m.motion_encoder is not None:
        sq, nw, nr = _grad_l2_norm_sq(m.motion_encoder.parameters())
        _log(format_grad_line("motion_encoder (VQ-VAE)", sq, nw, nr))
    else:
        _log("  motion_encoder: N/A")
    motion_adapter_modules = []
    for attr in ("motion_prenorm", "motion_proj", "motion_postnorm"):
        if hasattr(m, attr) and getattr(m, attr) is not None:
            motion_adapter_modules.append(getattr(m, attr))
    if motion_adapter_modules:
        def _adapter_params():
            for mod in motion_adapter_modules:
                yield from mod.parameters()

        sq, nw, nr = _grad_l2_norm_sq(_adapter_params())
        _log(format_grad_line("motion_adapter (prenorm+proj+postnorm)", sq, nw, nr))
    else:
        _log("  motion_adapter: N/A")
    _log("--- end grad norm ---")


class GradientNormCallback(TrainerCallback):
    """在 optimizer.step 之前打印各子模块梯度范数，用于确认是否冻结。"""

    def __init__(self, every_n_steps: int):
        self.every_n_steps = max(1, int(every_n_steps))
        self._opt_step = 0

    def on_pre_optimizer_step(
        self,
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # HF CallbackHandler.call_event 会传入 model=trainer.model（含 DeepSpeed 包装）
        model = kwargs.get("model")
        if model is None:
            if _is_global_rank0():
                rank0_print(
                    "[grad norm] on_pre_optimizer_step: kwargs 中无 model，跳过（请检查 transformers 版本）"
                )
            return control
        self._opt_step += 1
        if self._opt_step % self.every_n_steps != 0:
            return control
        # 含 all_reduce：所有 rank 必须执行 log_module_gradient_norms；仅 global rank 0 打印
        try:
            log_module_gradient_norms(model, log_fn=rank0_print)
        except Exception as e:
            if _is_global_rank0():
                rank0_print(f"[grad norm] failed to log: {e}")
        return control


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


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, MotionArguments, DataArguments, TrainingArguments)
    )
    model_args, motion_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not math.isfinite(training_args.learning_rate) or training_args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    formal_artifact = validate_artifact_policy(training_args, training_mode="full_sft")
    require_controller_verified_formal_bootstrap(formal_artifact=formal_artifact)
    validate_formal_deepspeed_zero2(
        training_args, formal_artifact=formal_artifact
    )
    if not formal_artifact and is_primary_process(torch):
        rank0_print(
            "WARNING: UNSAFE LEGACY NO-MANIFEST MODE; this output is ineligible for formal evaluation."
        )
    local_rank = training_args.local_rank

    # data_processor 中的 grouped sampling 需要 data_flatten 或 data_packing
    # 为避免忘记设置导致训练直接报错，这里做一个兜底：两者都没开时默认开启 data_packing
    if hasattr(data_args, "data_flatten") and hasattr(data_args, "data_packing"):
        if (not data_args.data_flatten) and (not data_args.data_packing):
            data_args.data_packing = True
            rank0_print(
                "Auto-enabled data_packing=True (required for grouped sampling). "
                "Set --data_flatten/--data_packing explicitly to override."
            )

    # 设置随机种子，确保可复现
    seed = require_explicit_formal_seed(
        training_args, formal_artifact=formal_artifact
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rank0_print(f"Random seed set to {seed} for reproducibility.")

    # Convert placeholder strings to None for optional VQ-VAE args
    if hasattr(motion_args, 'vqvae_norm') and motion_args.vqvae_norm is not None:
        norm = motion_args.vqvae_norm.strip().lower()
        if norm in ("", "none"):
            motion_args.vqvae_norm = None

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
        training_mode="full_sft",
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
        training_mode="full_sft",
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
            "vqvae_norm": motion_args.vqvae_norm if motion_args.vqvae_norm and motion_args.vqvae_norm.strip() else None,
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

    print(f'the initialized family is {spec.family.value}; class={model.__class__.__name__}')

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

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    set_model(model_args, model)
    initial_trainable_snapshot = (
        snapshot_trainable_state(model) if formal_artifact else None
    )

    if is_primary_process(torch):
        print_trainable_layers(model, model_args)
        rank0_print("\n详细参数统计:")
        if hasattr(model, 'visual'):
            model.visual.print_trainable_parameters()
        if hasattr(model, 'model'):
            model.model.print_trainable_parameters()
    
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
    if data_module.get("eval_dataset") is not None:
        rank0_print(f"Eval enabled on dataset: {getattr(data_args, 'eval_dataset_use', 'N/A')}")
    evidence_tracker = OptimizerEvidenceTracker(torch)
    evidence_callback = TrainingEvidenceCallback(evidence_tracker)
    callbacks = [evidence_callback]
    if getattr(training_args, "log_grad_norm_every", 0) and training_args.log_grad_norm_every > 0:
        callbacks.append(GradientNormCallback(training_args.log_grad_norm_every))
        rank0_print(
            f"[grad norm] 已开启：每 {training_args.log_grad_norm_every} 次 optimizer step "
            f"在 clip 之后、optimizer.step 之前打印各子模块 ||g||_2（与日志里 grad_norm 同一时刻附近）。"
        )
    else:
        rank0_print(
            "[grad norm] 未开启。请在启动参数中加 --log_grad_norm_every 1 "
            "或设置环境变量 LOG_GRAD_NORM_EVERY=1（见 scripts/full_sft.sh）。"
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
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

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
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
        training_mode="full_sft",
        formal_artifact=formal_artifact,
        torch_module=torch,
    )

    if formal_artifact and is_primary_process(torch):
        if initial_trainable_snapshot is None or canonical_identity is None:
            raise RuntimeError("formal full-SFT state evidence was not initialized")
        artifact_info = hash_path(
            training_args.output_dir, symlink_policy="reject"
        )
        reloaded_bundle = default_model_factory.load_bundle(
            family=model_args.model_family,
            model_name_or_path=training_args.output_dir,
            model_kwargs=model_kwargs,
        )
        reloaded_model = reloaded_bundle.model
        reloaded_processor = reloaded_bundle.processor
        reloaded_tokenizer = getattr(reloaded_processor, "tokenizer", None)
        if reloaded_tokenizer is None:
            raise ValueError("saved full-SFT processor must expose tokenizer")
        final_trainable_snapshot = snapshot_trainable_state(
            reloaded_model,
            parameter_names=tuple(initial_trainable_snapshot.tensor_sha256),
        )
        changed_tensors = changed_trainable_tensor_count(
            initial_trainable_snapshot, final_trainable_snapshot
        )
        reload_receipt = verify_full_save_reload(
            model,
            reloaded_model,
            tokenizer=tokenizer,
            reloaded_tokenizer=reloaded_tokenizer,
            processor=processor,
            reloaded_processor=reloaded_processor,
            processor_artifact_path=training_args.output_dir,
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
            training_mode="full_sft",
            backend_id="qwenvl.train.full_sft",
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
        training_mode="full_sft",
        artifact_path=training_args.output_dir,
        torch_module=torch,
        provenance_snapshot=provenance_snapshot,
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
