import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_family: Optional[str] = field(
        default=None,
        metadata={
            "help": "Required explicit family: qwen2_vl, qwen2_5_vl, qwen3_vl, "
            "qwen3_vl_moe, or qwen3_vl_motion. It is never inferred from a path."
        },
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_motion: bool = field(default=False)


@dataclass
class MotionArguments:
    motion_vqvae_path: Optional[str] = field(
        default=None, metadata={"help": "VQ-VAE ckpt path used by Qwen3VlMotion model."}
    )
    motion_dataname: Optional[str] = field(
        default=None, metadata={"help": "Motion dataset tag stored in config.dataname."}
    )
    motion_quantizer: Optional[str] = field(
        default=None, metadata={"help": "Motion quantizer tag stored in config.quantizer."}
    )
    motion_normalization_mean_path: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit Mean.npy used by the motion model; configure together with std."},
    )
    motion_normalization_std_path: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit Std.npy used by the motion model; configure together with mean."},
    )
    vqvae_nb_code: Optional[int] = field(
        default=None, metadata={"help": "Number of VQ-VAE codes (config.vqvae_nb_code)."}
    )
    vqvae_code_dim: Optional[int] = field(
        default=None, metadata={"help": "Dimension of each code vector (config.vqvae_code_dim)."}
    )
    vqvae_output_emb_width: Optional[int] = field(
        default=None, metadata={"help": "Output embedding width (config.vqvae_output_emb_width)."}
    )
    vqvae_down_t: Optional[int] = field(
        default=None, metadata={"help": "Temporal down-sampling factor (config.vqvae_down_t)."}
    )
    vqvae_stride_t: Optional[int] = field(
        default=None, metadata={"help": "Temporal stride (config.vqvae_stride_t)."}
    )
    vqvae_width: Optional[int] = field(
        default=None, metadata={"help": "Base channel width (config.vqvae_width)."}
    )
    vqvae_depth: Optional[int] = field(
        default=None, metadata={"help": "Number of residual blocks (config.vqvae_depth)."}
    )
    vqvae_dilation_growth_rate: Optional[int] = field(
        default=None, metadata={"help": "Dilation growth rate (config.vqvae_dilation_growth_rate)."}
    )
    vqvae_activation: Optional[str] = field(
        default=None, metadata={"help": "Activation name for VQ-VAE (config.vqvae_activation)."}
    )
    vqvae_norm: Optional[str] = field(
        default=None, metadata={"help": "Normalization type for VQ-VAE (config.vqvae_norm)."}
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name for evaluation (e.g. motionX_v0_1_test). If None, no eval during training."}
    )
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2
    motion_length_divisor: Optional[int] = field(
        default=None,
        metadata={"help": "Divisor used for motion placeholders; must match the verified VQ stride**depth factor."}
    )
    motion_placeholder_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Bound at runtime from model.config; do not guess this token ID."},
    )
    motion_fps: float = field(
        default=30.0,
        metadata={
            "help": "Frames per second for motion time axis; used in <t> seconds> strings (like video metadata.fps)."
        },
    )
    motion_placeholders_per_timestamp: Optional[int] = field(
        default=None,
        metadata={
            "help": "How many motion pad tokens (160001) share one '<t> seconds>' prefix, analogous to video where "
            "one timestamp precedes frame_seqlen video_pad tokens. If None, use video_processor.merge_size (fallback 1)."
        },
    )
    motion_timestamps_sync_with_video: bool = field(
        default=False,
        metadata={
            "help": "If True and a sample has both video and motion, reuse the exact '<t seconds>' token sequences "
            "already inserted for video by Qwen3-VL native mechanism, and insert them into the motion placeholder "
            "segment as well. This guarantees motion timestamp times/count match video exactly."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    log_grad_norm_every: int = field(
        default=0,
        metadata={
            "help": "If > 0, on rank 0 print L2 grad norms per module every N optimizer steps "
            "(after backward, before optimizer.step). 0 disables."
        },
    )
    batch_id: Optional[str] = field(default=None)
    model_registry_id: Optional[str] = field(default=None)
    base_artifact_path: Optional[str] = field(default=None)
    train_data_path: Optional[str] = field(default=None)
    validation_data_path: Optional[str] = field(default=None)
    benchmark_path: Optional[str] = field(default=None)
    leakage_audit_path: Optional[str] = field(default=None)
    config_path: Optional[str] = field(default=None)
    code_path: Optional[str] = field(default=None)
    runner_code_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Formal provenance root for the strict Qwen SFT runner source allowlist."
        },
    )
    environment_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Current isolated Python environment root for the secondary in-process diagnostic; formal Qwen publication remains blocked until controller pre-spawn verification exists."
        },
    )
    motion_vqvae_asset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Formal provenance path for the exact motion VQ-VAE checkpoint loaded by the model."
        },
    )
    artifact_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root that confines artifact, manifest and resume paths."},
    )
    artifact_manifest_path: Optional[str] = field(
        default=None,
        metadata={"help": "Write a fresh hash-bound finetune manifest after save."},
    )
    reload_receipt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Verified fresh save/reload receipt; mandatory for formal SFT."},
    )
    training_receipt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Strict self-hashed optimizer/gradient/weight-change proof; formal Qwen publication additionally requires the currently unavailable verified controller bootstrap."},
    )
    batch_receipt_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Controller-frozen batch receipt SHA-256."},
    )
    attempt_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Controller-frozen finetune attempt receipt SHA-256."},
    )
    resume_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume only from this validated manifest; checkpoint discovery is disabled."
        },
    )
    unsafe_legacy_no_manifest: bool = field(
        default=False,
        metadata={
            "help": "Explicitly allow a non-release legacy smoke without a manifest. "
            "Forbidden for controller/formal batches."
        },
    )
