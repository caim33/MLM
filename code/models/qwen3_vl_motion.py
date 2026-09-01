from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from models.vqvae import VQVAE_251
import numpy as np
import torch
from typing import Any, List, Optional, Sequence, Union
from contextlib import nullcontext
import os
from pathlib import Path

from transformers import AutoConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging

from motionllm.contracts import Modality
from motionllm.fusion import MotionTokenIds, parse_motion_spans
from motionllm.models import (
    MotionDevicePolicy,
    MotionDTypePolicy,
    MotionResizePolicy,
    enumerate_motion_compute_placements,
    extract_state_dict,
    is_generation_prefill,
    migrate_legacy_motion_config,
    normalize_modalities,
    normalize_state_dict_keys,
    prefill_motion_payload,
    required_feature_length,
    resolve_motion_model_spec,
    select_vq_checkpoint_state,
    validate_motion_compute_contract,
    validate_motion_encoder_downsample,
    validate_motion_presence,
    validate_motion_segment_ownership,
    validate_preembedded_motion_inputs,
)
from motionllm.motion import (
    load_motion_array,
    load_normalization_stats,
    MotionValidationError,
    normalize_motion,
)

try:
    from typing import Unpack  # Python 3.11+
except ImportError:
    from typing_extensions import Unpack

from transformers.cache_utils import Cache

try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    from typing import Any, Dict
    TransformersKwargs = Dict[str, Any]

import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast

logger = logging.get_logger(__name__)


def _is_motion_grpo_debug_enabled() -> bool:
    flag = os.environ.get("MOTION_GRPO_DEBUG", "")
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def _debug_value_summary(value: Any, max_items: int = 4) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        prefix = type(value).__name__
        sample = [repr(value[i]) for i in range(min(len(value), max_items))]
        if len(value) > max_items:
            sample.append("...")
        return f"{prefix}(len={len(value)}, sample=[{', '.join(sample)}])"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        if len(keys) > max_items:
            keys = keys[:max_items] + ["..."]
        return f"dict(keys={keys})"
    return repr(value)


# GRPO/inference metadata keys that may be attached to generation requests.
# These fields are for reward bookkeeping only and must not be forwarded to
# the underlying language model forward pass.
_GENERATION_METADATA_KEYS = {
    "solution",
    "answer",
    "group_id",
    "branch",
    "sample_id",
    "rollout_id",
    "prompt_id",
    "request_id",
    "question_id",
    "difficulty",
    "packed_branch",
    "packed_sample_id",
    "packed_group_id",
    "motion_owner_indices",
}


class Qwen3VlMotionForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        migrated_fields = migrate_legacy_motion_config(config)
        motion_spec = resolve_motion_model_spec(config)
        super().__init__(config)
        self.motion_spec = motion_spec
        if migrated_fields:
            logger.warning(
                "Migrated legacy motion config fields into explicit config: %s",
                migrated_fields,
            )
        # 从 config 读取所有 vqvae 参数
        vqvae_nb_code = getattr(config, "vqvae_nb_code", 512)
        vqvae_code_dim = getattr(config, "vqvae_code_dim", 512)
        vqvae_output_emb_width = getattr(config, "vqvae_output_emb_width", 512)
        vqvae_down_t = getattr(config, "vqvae_down_t", 2)
        vqvae_stride_t = getattr(config, "vqvae_stride_t", 2)
        vqvae_width = getattr(config, "vqvae_width", 512)
        vqvae_depth = getattr(config, "vqvae_depth", 3)
        vqvae_dilation_growth_rate = getattr(config, "vqvae_dilation_growth_rate", 3)
        vqvae_activation = getattr(config, "vqvae_activation", "relu")
        vqvae_norm = getattr(config, "vqvae_norm", None)
        encoder_input_dim = 251 if getattr(config, "dataname", "kit") == "kit" else 263
        if encoder_input_dim != motion_spec.input_dim:
            raise ValueError(
                f"VQ encoder input dimension {encoder_input_dim} disagrees with "
                f"motion_input_dim={motion_spec.input_dim}"
            )
        if int(vqvae_output_emb_width) != motion_spec.encoder_output_dim:
            raise ValueError(
                f"VQ encoder output dimension {vqvae_output_emb_width} disagrees "
                f"with motion_encoder_output_dim={motion_spec.encoder_output_dim}"
            )
        if int(config.text_config.hidden_size) != motion_spec.projector.output_dim:
            raise ValueError(
                f"projector output dimension {motion_spec.projector.output_dim} "
                f"must equal text hidden size {config.text_config.hidden_size}"
            )
        validate_motion_encoder_downsample(
            motion_spec,
            vqvae_down_t=vqvae_down_t,
            vqvae_stride_t=vqvae_stride_t,
        )

        self.motion_encoder = VQVAE_251(
            config,
            nb_code=vqvae_nb_code,
            code_dim=vqvae_code_dim,
            output_emb_width=vqvae_output_emb_width,
            down_t=vqvae_down_t,
            stride_t=vqvae_stride_t,
            width=vqvae_width,
            depth=vqvae_depth,
            dilation_growth_rate=vqvae_dilation_growth_rate,
            activation=vqvae_activation,
            norm=vqvae_norm
        )

        activation_factories = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }
        projector_layers: list[nn.Module] = []
        for index, layer in enumerate(motion_spec.projector.linear_layers):
            projector_layers.append(
                nn.Linear(layer.in_features, layer.out_features, bias=layer.bias)
            )
            if index + 1 < len(motion_spec.projector.linear_layers):
                projector_layers.append(
                    activation_factories[motion_spec.projector.activation]()
                )
        self.motion_proj = nn.Sequential(*projector_layers)
        self.motion_prenorm = (
            nn.LayerNorm(motion_spec.encoder_output_dim)
            if motion_spec.projector.pre_norm
            else nn.Identity()
        )
        self._apply_motion_postnorm = bool(motion_spec.projector.post_norm)
        retain_legacy_postnorm_state = bool(
            getattr(config, "motion_legacy_postnorm_state_compat", False)
        )
        self.motion_postnorm = (
            nn.LayerNorm(motion_spec.projector.output_dim)
            if self._apply_motion_postnorm or retain_legacy_postnorm_state
            else nn.Identity()
        )

        # Motion boundary special-token embeddings (trainable even when LLM is frozen).
        # These correspond to tokenizer-added special tokens: <motion_start>, <motion_end>.
        self.motion_boundary_embed = nn.Embedding(
            2, motion_spec.projector.output_dim
        )
        self._motion_mean = None
        self._motion_std = None
        # Motion text-anchor patterns are dynamically derived from tokenizer via config.
        # Keep legacy fallbacks only for backward compatibility when config is missing.
        self._motion_text_token_patterns = self._build_motion_text_token_patterns()
        if _is_motion_grpo_debug_enabled():
            logger.warning(
                "[motion-debug][init] motion_text_token_patterns=%s motion_start_token_id=%r motion_end_token_id=%r",
                self._motion_text_token_patterns,
                getattr(self.config, "motion_start_token_id", None),
                getattr(self.config, "motion_end_token_id", None),
            )

    def _load_motion_norm_stats(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._motion_mean is None or self._motion_std is None:
            if self.motion_spec.normalization_mean_path is None:
                raise MotionValidationError(
                    "motion normalization is required for path-based motion input; "
                    "set motion_normalization_mean_path and "
                    "motion_normalization_std_path explicitly"
                )
            mean_path = self.motion_spec.normalization_mean_path
            std_path = self.motion_spec.normalization_std_path
            mean, std = load_normalization_stats(
                mean_path,
                std_path,
                expected_feature_dim=self.motion_spec.input_dim,
            )
            self._motion_mean = torch.from_numpy(mean.copy())
            self._motion_std = torch.from_numpy(std.copy())
        return self._motion_mean.to(device=device), self._motion_std.to(device=device)

    def _load_motion_from_path(self, path_like: Union[str, Path]) -> torch.Tensor:
        motion_path = Path(str(path_like))
        arr = load_motion_array(
            motion_path,
            expected_feature_dim=self.motion_spec.input_dim,
        )
        mean, std = self._load_motion_norm_stats(device=torch.device("cpu"))
        normalized = normalize_motion(
            arr,
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
        )
        return torch.from_numpy(normalized.copy())

    @staticmethod
    def _find_subsequence(tokens: list[int], pattern: tuple[int, ...]) -> int:
        if not pattern:
            return -1
        size = len(pattern)
        for idx in range(len(tokens) - size + 1):
            if tuple(tokens[idx : idx + size]) == pattern:
                return idx
        return -1

    @staticmethod
    def _normalize_token_pattern(value: Any) -> Optional[tuple[int, ...]]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return None
        if len(value) == 0:
            return None

        pattern: List[int] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().item()
            if isinstance(item, np.ndarray):
                item = np.asarray(item).item()
            try:
                pattern.append(int(item))
            except (TypeError, ValueError):
                return None
        return tuple(pattern)

    @classmethod
    def _normalize_token_patterns(cls, value: Any) -> List[tuple[int, ...]]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()

        patterns: List[tuple[int, ...]] = []
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return patterns
            if all(isinstance(item, (list, tuple, torch.Tensor, np.ndarray)) for item in value):
                for item in value:
                    pattern = cls._normalize_token_pattern(item)
                    if pattern is not None:
                        patterns.append(pattern)
            else:
                pattern = cls._normalize_token_pattern(value)
                if pattern is not None:
                    patterns.append(pattern)
            return patterns

        pattern = cls._normalize_token_pattern(value)
        return [pattern] if pattern is not None else []

    def _build_motion_text_token_patterns(self) -> List[tuple[int, ...]]:
        patterns: List[tuple[int, ...]] = []
        # Preferred dynamic patterns injected by loader.
        for key in (
            "motion_text_token_patterns",
            "motion_text_token_ids",
            "motion_boundary_inline_token_ids",
            "motion_boundary_spaced_token_ids",
        ):
            patterns.extend(self._normalize_token_patterns(getattr(self.config, key, None)))

        unique_patterns: List[tuple[int, ...]] = []
        seen = set()
        for pattern in patterns:
            if not pattern:
                continue
            if pattern in seen:
                continue
            seen.add(pattern)
            unique_patterns.append(pattern)
        return unique_patterns

    def _find_motion_text_span(self, sample_input_ids: torch.LongTensor) -> Optional[tuple[int, int]]:
        if not self._motion_text_token_patterns:
            return None
        token_list = sample_input_ids.detach().cpu().tolist()
        for pattern in self._motion_text_token_patterns:
            start = self._find_subsequence(token_list, pattern)
            if start >= 0:
                return int(start), int(start + len(pattern))
        return None

    def _find_motion_boundary_span(self, sample_input_ids: torch.LongTensor) -> Optional[tuple[int, int]]:
        motion_start_id = getattr(self.config, "motion_start_token_id", None)
        motion_end_id = getattr(self.config, "motion_end_token_id", None)
        if (motion_start_id is None) != (motion_end_id is None):
            raise ValueError(
                "motion_start_token_id and motion_end_token_id must be configured together"
            )
        if motion_start_id is None or motion_end_id is None:
            return None

        start_positions = (sample_input_ids == int(motion_start_id)).nonzero(as_tuple=True)[0]
        end_positions = (sample_input_ids == int(motion_end_id)).nonzero(as_tuple=True)[0]
        if start_positions.numel() == 0 or end_positions.numel() == 0:
            return None

        for start in start_positions.tolist():
            valid_ends = end_positions[end_positions > start]
            if valid_ends.numel() == 0:
                continue
            end = int(valid_ends[0].item())
            if end > start + 1:
                return int(start), int(end)
        return None

    def _strict_motion_layout(
        self,
        sample_input_ids: torch.LongTensor,
        *,
        expected_segment_feature_counts: Optional[Sequence[int]] = None,
    ) -> Optional[tuple[torch.LongTensor, tuple[int, ...]]]:
        start_id = getattr(self.config, "motion_start_token_id", None)
        end_id = getattr(self.config, "motion_end_token_id", None)
        if start_id is None and end_id is None:
            return None
        if start_id is None or end_id is None:
            raise ValueError(
                "motion_start_token_id and motion_end_token_id must be configured together"
            )
        token_spec = MotionTokenIds(
            start=int(start_id),
            placeholder=int(self.motion_spec.placeholder_token_id),
            end=int(end_id),
        )
        allowed = getattr(
            self.config, "motion_allowed_interstitial_token_ids", ()
        )
        spans = parse_motion_spans(
            sample_input_ids.detach().cpu().tolist(),
            token_spec,
            allowed_interstitial_token_ids=allowed,
        )
        if expected_segment_feature_counts is not None:
            validate_motion_segment_ownership(
                expected_segment_feature_counts,
                tuple(span.placeholder_count for span in spans),
                allow_per_segment_resize=(
                    self.motion_spec.resize_policy is MotionResizePolicy.LINEAR
                ),
            )
        if not spans:
            return (
                torch.empty((0,), dtype=torch.long, device=sample_input_ids.device),
                (),
            )
        positions = [
            position for span in spans for position in span.placeholder_positions
        ]
        return (
            torch.tensor(positions, dtype=torch.long, device=sample_input_ids.device),
            tuple(span.placeholder_count for span in spans),
        )

    def _strict_motion_placeholder_positions(
        self,
        sample_input_ids: torch.LongTensor,
        *,
        expected_segment_feature_counts: Optional[Sequence[int]] = None,
    ) -> Optional[torch.LongTensor]:
        layout = self._strict_motion_layout(
            sample_input_ids,
            expected_segment_feature_counts=expected_segment_feature_counts,
        )
        return None if layout is None else layout[0]

    def _resize_motion_embeds(self, motion_embeds: torch.Tensor, target_len: int) -> torch.Tensor:
        target_len = required_feature_length(
            int(motion_embeds.shape[0]),
            target_len,
            policy=self.motion_spec.resize_policy,
        )
        if motion_embeds.shape[0] == target_len:
            return motion_embeds
        resized = F.interpolate(
            motion_embeds.transpose(0, 1).unsqueeze(0),
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1)

    @staticmethod
    def _normalize_motion_length_entry(entry: Any) -> Optional[List[int]]:
        if entry is None:
            return None
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu().tolist()
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()

        if isinstance(entry, (int, float)):
            value = int(entry)
            return [value] if value > 0 else []

        if isinstance(entry, (list, tuple)):
            values: List[int] = []
            for item in entry:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().item()
                if isinstance(item, np.ndarray):
                    item = np.asarray(item).item()
                value = int(item)
                if value > 0:
                    values.append(value)
            return values

        raise ValueError(f"Unsupported motion length entry type: {type(entry)}")

    @classmethod
    def _normalize_motion_lengths_per_sample(
        cls,
        motion_lengths: Optional[Any],
        batch_size: int,
    ) -> List[Optional[List[int]]]:
        if motion_lengths is None:
            return [None] * batch_size

        if isinstance(motion_lengths, torch.Tensor):
            motion_lengths = motion_lengths.detach().cpu().tolist()
        if isinstance(motion_lengths, np.ndarray):
            motion_lengths = motion_lengths.tolist()

        if isinstance(motion_lengths, (int, float)):
            normalized = cls._normalize_motion_length_entry(motion_lengths)
            return [normalized] * batch_size

        if not isinstance(motion_lengths, (list, tuple)):
            raise ValueError(f"Unsupported motion_lengths type: {type(motion_lengths)}")

        if batch_size == 1:
            if len(motion_lengths) == 0:
                return [None]
            if any(isinstance(x, (list, tuple, torch.Tensor, np.ndarray)) for x in motion_lengths):
                if len(motion_lengths) == 1:
                    return [cls._normalize_motion_length_entry(motion_lengths[0])]
                return [cls._normalize_motion_length_entry(motion_lengths)]
            return [cls._normalize_motion_length_entry(motion_lengths)]

        if len(motion_lengths) == batch_size:
            return [cls._normalize_motion_length_entry(x) for x in motion_lengths]
        if len(motion_lengths) == 1:
            normalized = cls._normalize_motion_length_entry(motion_lengths[0])
            return [normalized] * batch_size

        raise ValueError(
            f"motion_lengths length mismatch for batch_size={batch_size}: "
            f"got {len(motion_lengths)} entries."
        )

    def _convert_motion_item_to_tensor(self, motion_item: Any) -> Optional[torch.Tensor]:
        if motion_item is None:
            return None

        if isinstance(motion_item, (str, Path)):
            motion_tensor = self._load_motion_from_path(motion_item)
        elif isinstance(motion_item, np.ndarray):
            motion_tensor = torch.tensor(motion_item, dtype=torch.float32)
        elif isinstance(motion_item, torch.Tensor):
            motion_tensor = motion_item.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported motion item type: {type(motion_item)}")

        if motion_tensor.dim() != 2:
            raise ValueError(f"Expected 2D motion tensor, got shape={tuple(motion_tensor.shape)}")
        if motion_tensor.shape[0] <= 0 or motion_tensor.shape[1] != self.motion_spec.input_dim:
            raise ValueError(
                "Motion tensor must have non-empty (time, features) shape with "
                f"features={self.motion_spec.input_dim}; got {tuple(motion_tensor.shape)}"
            )
        if not bool(torch.isfinite(motion_tensor).all().item()):
            raise ValueError("Motion tensor contains non-finite values")
        return motion_tensor

    @staticmethod
    def _split_concatenated_motion_tensor(
        motion_tensor: torch.Tensor,
        per_sample_lengths: Sequence[Optional[List[int]]],
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if motion_tensor.dim() != 2:
            return None
        if any(lengths is None for lengths in per_sample_lengths):
            return None

        token_lengths = [sum(lengths or []) for lengths in per_sample_lengths]
        total = sum(token_lengths)
        if total <= 0:
            return None
        if total != motion_tensor.shape[0]:
            raise ValueError(
                f"Concatenated motion length mismatch: expected total={total}, "
                f"got motion.shape[0]={motion_tensor.shape[0]}"
            )

        out: List[Optional[torch.Tensor]] = []
        offset = 0
        for length in token_lengths:
            if length <= 0:
                out.append(None)
                continue
            out.append(motion_tensor[offset : offset + length])
            offset += length
        return out

    def _normalize_batch_motion_inputs(
        self,
        motion: Optional[Any],
        motion_lengths: Optional[Any],
        batch_size: int,
    ) -> tuple[List[Optional[torch.Tensor]], List[Optional[List[int]]]]:
        per_sample_lengths = self._normalize_motion_lengths_per_sample(motion_lengths, batch_size)

        if motion is None:
            raw_items: List[Any] = [None] * batch_size
        elif isinstance(motion, (list, tuple)):
            if len(motion) == 0:
                raw_items = [None] * batch_size
            elif len(motion) == batch_size:
                raw_items = list(motion)
            elif len(motion) == 1:
                if batch_size > 1 and not self.motion_spec.allow_batch_broadcast:
                    raise ValueError(
                        "Refusing to broadcast one motion across multiple batch rows; "
                        "pass one explicit entry per row"
                    )
                raw_items = [motion[0]] * batch_size
            else:
                raise ValueError(
                    f"motion length mismatch for batch_size={batch_size}: got {len(motion)} entries."
                )
        elif isinstance(motion, torch.Tensor):
            if motion.dim() == 3 and motion.shape[0] == batch_size:
                raw_items = [motion[i] for i in range(batch_size)]
            else:
                split_items = self._split_concatenated_motion_tensor(motion, per_sample_lengths)
                if split_items is not None:
                    raw_items = split_items
                elif batch_size == 1 or self.motion_spec.allow_batch_broadcast:
                    raw_items = [motion] * batch_size
                else:
                    raise ValueError(
                        "A 2D motion tensor for batch_size>1 requires exact nested "
                        "motion_lengths or explicit broadcast policy"
                    )
        elif isinstance(motion, np.ndarray):
            if motion.ndim == 3 and motion.shape[0] == batch_size:
                raw_items = [motion[i] for i in range(batch_size)]
            else:
                motion_tensor = torch.tensor(motion, dtype=torch.float32)
                split_items = self._split_concatenated_motion_tensor(motion_tensor, per_sample_lengths)
                if split_items is not None:
                    raw_items = split_items
                elif batch_size == 1 or self.motion_spec.allow_batch_broadcast:
                    raw_items = [motion] * batch_size
                else:
                    raise ValueError(
                        "A 2D motion array for batch_size>1 requires exact nested "
                        "motion_lengths or explicit broadcast policy"
                    )
        elif isinstance(motion, (str, Path)):
            if batch_size > 1 and not self.motion_spec.allow_batch_broadcast:
                raise ValueError(
                    "Refusing to broadcast one motion path across multiple batch rows"
                )
            raw_items = [motion] * batch_size
        else:
            raise ValueError(f"Unsupported motion input type: {type(motion)}")

        if len(raw_items) != batch_size:
            raise ValueError(
                f"Internal motion normalization mismatch: expected {batch_size}, got {len(raw_items)}"
            )

        per_sample_motion: List[Optional[torch.Tensor]] = []
        for idx, item in enumerate(raw_items):
            motion_tensor = self._convert_motion_item_to_tensor(item)
            per_sample_motion.append(motion_tensor)
            if motion_tensor is None:
                per_sample_lengths[idx] = None

        return per_sample_motion, per_sample_lengths

    def _encode_motion_tensor(
        self,
        motion_tensor: Optional[torch.Tensor],
        sample_lengths: Optional[Sequence[int]],
    ) -> tuple[Optional[torch.Tensor], Optional[tuple[int, ...]]]:
        if motion_tensor is None:
            return None, None

        encoder_module = getattr(self.motion_encoder, "encoder", None)
        if encoder_module is None:
            encoder_module = self.motion_encoder
            encoder_name = "motion_encoder"
        else:
            encoder_name = "motion_encoder.encoder"
        model_reference = self.get_input_embeddings().weight
        dtype_map = {
            MotionDTypePolicy.FLOAT32: torch.float32,
            MotionDTypePolicy.FLOAT16: torch.float16,
            MotionDTypePolicy.BFLOAT16: torch.bfloat16,
        }
        expected_dtype = dtype_map.get(
            self.motion_spec.dtype_policy, model_reference.dtype
        )

        placements = enumerate_motion_compute_placements(
            encoder_name=encoder_name,
            encoder=encoder_module,
            motion_prenorm=self.motion_prenorm,
            motion_proj=self.motion_proj,
            motion_postnorm=self.motion_postnorm,
            apply_postnorm=self._apply_motion_postnorm,
            motion_boundary_embed=self.motion_boundary_embed,
        )
        encoder_placements = tuple(
            placement
            for placement in placements
            if placement[0] == encoder_name
            or placement[0].startswith(f"{encoder_name}.")
        )
        expected_device = (
            encoder_placements[0][2]
            if self.motion_spec.device_policy is MotionDevicePolicy.ENCODER
            and encoder_placements
            else model_reference.device
        )
        validate_motion_compute_contract(
            expected_dtype=expected_dtype,
            expected_device=expected_device,
            module_placements=placements,
        )
        motion_tensor = motion_tensor.to(
            device=expected_device, dtype=expected_dtype
        )
        autocast_ctx = nullcontext()
        if expected_device.type == "cuda" and expected_dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            autocast_ctx = torch.amp.autocast("cuda", dtype=expected_dtype)

        with autocast_ctx:
            if sample_lengths is not None:
                segs = []
                encoded_segment_lengths: List[int] = []
                offset = 0
                for length in sample_lengths:
                    if length <= 0:
                        continue
                    if offset + int(length) > motion_tensor.shape[0]:
                        raise ValueError(
                            f"Motion length overflow: offset={offset}, length={length}, "
                            f"motion_rows={motion_tensor.shape[0]}"
                        )
                    motion_i = motion_tensor[offset : offset + int(length)]
                    offset += int(length)
                    feat = self.motion_encoder.encode(motion_i.unsqueeze(0))
                    if feat.shape[-1] != self.motion_spec.encoder_output_dim:
                        raise ValueError(
                            "motion encoder output dimension mismatch: "
                            f"expected {self.motion_spec.encoder_output_dim}, "
                            f"got {feat.shape[-1]}"
                        )
                    if feat.dim() == 3 and feat.shape[0] == 1:
                        feat = feat.squeeze(0)
                    if feat.dim() != 2 or feat.shape[0] <= 0:
                        raise ValueError(
                            "motion encoder must return non-empty (time, features) "
                            f"embeddings; got {tuple(feat.shape)}"
                        )
                    feat = self.motion_prenorm(feat)
                    projected = self.motion_proj(feat)
                    if self._apply_motion_postnorm:
                        projected = self.motion_postnorm(projected)
                    segs.append(projected)
                    encoded_segment_lengths.append(int(projected.shape[0]))

                if offset != motion_tensor.shape[0]:
                    raise ValueError(
                        f"Motion length under-consumed: consumed={offset}, total={motion_tensor.shape[0]}"
                    )
                if not segs:
                    return None, None
                motion_embeds = torch.cat(segs, dim=0)
            else:
                motion_feat = self.motion_encoder.encode(motion_tensor.unsqueeze(0))
                if motion_feat.shape[-1] != self.motion_spec.encoder_output_dim:
                    raise ValueError(
                        "motion encoder output dimension mismatch: "
                        f"expected {self.motion_spec.encoder_output_dim}, "
                        f"got {motion_feat.shape[-1]}"
                    )
                if motion_feat.dim() == 3 and motion_feat.shape[0] == 1:
                    motion_feat = motion_feat.squeeze(0)
                if motion_feat.dim() != 2 or motion_feat.shape[0] <= 0:
                    raise ValueError(
                        "motion encoder must return non-empty (time, features) "
                        f"embeddings; got {tuple(motion_feat.shape)}"
                    )
                motion_feat = self.motion_prenorm(motion_feat)
                motion_embeds = self.motion_proj(motion_feat)
                if self._apply_motion_postnorm:
                    motion_embeds = self.motion_postnorm(motion_embeds)
                encoded_segment_lengths = [int(motion_embeds.shape[0])]

        if motion_embeds.dim() == 3:
            motion_embeds = motion_embeds.squeeze(0)
        return motion_embeds, tuple(encoded_segment_lengths)

    def _build_single_sample_inputs_embeds(
        self,
        sample_input_ids: torch.LongTensor,
        sample_motion_embeds: Optional[torch.Tensor],
        motion_pad_token: int,
        expected_segment_feature_counts: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if sample_motion_embeds is None:
            return self.get_input_embeddings()(sample_input_ids.unsqueeze(0)).squeeze(0)

        motion_indices = (sample_input_ids == motion_pad_token).nonzero(as_tuple=True)[0]
        strict_layout = self._strict_motion_layout(
            sample_input_ids,
            expected_segment_feature_counts=expected_segment_feature_counts,
        )
        strict_positions = None if strict_layout is None else strict_layout[0]
        if strict_positions is not None:
            if not torch.equal(motion_indices, strict_positions):
                raise ValueError(
                    "motion boundary/placeholder state machine disagrees with "
                    "placeholder token positions"
                )
            motion_indices = strict_positions
            if (
                expected_segment_feature_counts is not None
                and self.motion_spec.resize_policy is MotionResizePolicy.LINEAR
            ):
                feature_counts = tuple(
                    int(value) for value in expected_segment_feature_counts
                )
                placeholder_counts = strict_layout[1]
                if sum(feature_counts) != int(sample_motion_embeds.shape[0]):
                    raise ValueError(
                        "encoded motion segment counts do not sum to the motion "
                        "embedding tensor length"
                    )
                resized_segments = []
                offset = 0
                for feature_count, placeholder_count in zip(
                    feature_counts, placeholder_counts, strict=True
                ):
                    segment = sample_motion_embeds[
                        offset : offset + feature_count
                    ]
                    offset += feature_count
                    resized_segments.append(
                        self._resize_motion_embeds(segment, placeholder_count)
                    )
                sample_motion_embeds = torch.cat(resized_segments, dim=0)
        input_ids_wo_motion = sample_input_ids[sample_input_ids != motion_pad_token]

        if motion_indices.numel() == 0:
            text_span = self._find_motion_text_span(sample_input_ids)
            if text_span is not None:
                start, end = text_span
                target_len = end - start
                sample_motion_embeds = self._resize_motion_embeds(sample_motion_embeds, target_len)
                motion_indices = torch.arange(
                    start,
                    start + target_len,
                    device=sample_input_ids.device,
                    dtype=torch.long,
                )
                input_ids_wo_motion = torch.cat(
                    [sample_input_ids[:start], sample_input_ids[start + target_len :]],
                    dim=0,
                )

        if motion_indices.numel() == 0:
            boundary_span = self._find_motion_boundary_span(sample_input_ids)
            if boundary_span is not None:
                start, end = boundary_span
                motion_indices = torch.arange(
                    start + 1,
                    end,
                    device=sample_input_ids.device,
                    dtype=torch.long,
                )
                input_ids_wo_motion = torch.cat(
                    [sample_input_ids[: start + 1], sample_input_ids[end:]],
                    dim=0,
                )

        if motion_indices.numel() == 0:
            raise ValueError(
                "Motion input provided but no motion placeholder found in input_ids. "
                f"Expected configured token {motion_pad_token}, explicit textual "
                "anchor pattern, or <motion_start>...<motion_end> span."
            )

        sample_motion_embeds = self._resize_motion_embeds(
            sample_motion_embeds, int(motion_indices.numel())
        )

        token_embeds = self.get_input_embeddings()(
            input_ids_wo_motion.unsqueeze(0)
        ).squeeze(0)
        sample_motion_embeds = sample_motion_embeds.to(
            device=token_embeds.device, dtype=token_embeds.dtype
        )

        motion_indices_set = set(motion_indices.detach().cpu().tolist())
        output_embeds: List[torch.Tensor] = []
        text_cursor = 0
        motion_cursor = 0
        for idx in range(sample_input_ids.shape[0]):
            if idx in motion_indices_set:
                output_embeds.append(sample_motion_embeds[motion_cursor])
                motion_cursor += 1
            else:
                output_embeds.append(token_embeds[text_cursor])
                text_cursor += 1
        return torch.stack(output_embeds, dim=0)

    def _sample_has_motion_placeholder(self, sample_input_ids: torch.LongTensor, motion_pad_token: int) -> bool:
        if bool((sample_input_ids == motion_pad_token).any().item()):
            return True
        if self._find_motion_text_span(sample_input_ids) is not None:
            return True
        return self._find_motion_boundary_span(sample_input_ids) is not None

    @staticmethod
    def _normalize_branch_per_sample(
        branch: Any,
        batch_size: int,
        *,
        strict: bool = False,
    ) -> Optional[List[str]]:
        if branch is None:
            return None
        if isinstance(branch, torch.Tensor):
            branch = branch.detach().cpu().tolist()
        if isinstance(branch, np.ndarray):
            branch = branch.tolist()
        modalities = normalize_modalities(branch, batch_size=batch_size)
        return [item.branch for item in modalities] if modalities is not None else None

    @staticmethod
    def _normalize_meta_per_sample(value: Any, batch_size: int) -> List[Any]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                return list(value)
            if len(value) == 1:
                return [value[0]] * batch_size
            raise ValueError(
                f"Metadata length mismatch for batch_size={batch_size}: got {len(value)} entries."
            )
        return [value] * batch_size

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        motion: Optional[torch.Tensor] = None,
        motion_lengths: Optional[list] = None,
        **kwargs: Unpack['TransformersKwargs'],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Accept generic-ms-swift payloads where `motion` may be scalar/list/path/tensor.
        if motion_lengths is None:
            motion_lengths = kwargs.pop("motion_lengths", None)
        validate_preembedded_motion_inputs(
            inputs_embeds_present=inputs_embeds is not None,
            motion_present=motion is not None,
            motion_lengths_present=motion_lengths is not None,
        )
        debug_enabled = _is_motion_grpo_debug_enabled()
        if debug_enabled and (motion is not None or kwargs.get("branch") is not None):
            logger.warning(
                "[motion-debug][forward/entry] has_input_ids=%s has_inputs_embeds=%s kwargs_keys=%s "
                "branch=%s motion=%s motion_lengths=%s",
                input_ids is not None,
                inputs_embeds is not None,
                sorted(kwargs.keys()),
                _debug_value_summary(kwargs.get("branch")),
                _debug_value_summary(motion),
                _debug_value_summary(motion_lengths),
            )

        if inputs_embeds is None:
            motion_pad_token = self.motion_spec.placeholder_token_id
            input_ids_tensor = input_ids if input_ids is not None else kwargs.get("input_ids")
            if input_ids_tensor is None:
                raise ValueError("input_ids must be provided when using motion inputs.")
            if input_ids_tensor.dim() == 1:
                input_ids_2d = input_ids_tensor.unsqueeze(0)
            elif input_ids_tensor.dim() == 2:
                input_ids_2d = input_ids_tensor
            else:
                raise ValueError(f"Unsupported input_ids rank with motion: {input_ids_tensor.dim()}")

            batch_size = input_ids_2d.size(0)
            per_sample_motion, per_sample_lengths = self._normalize_batch_motion_inputs(
                motion=motion,
                motion_lengths=motion_lengths,
                batch_size=batch_size,
            )
            branch_raw = kwargs.get("branch")
            strict_group_mode = any(
                kwargs.get(k) is not None for k in ("group_id", "sample_id", "rollout_id", "prompt_id", "request_id")
            )
            prefill_phase = is_generation_prefill(
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
            branch_per_sample = self._normalize_branch_per_sample(
                branch_raw,
                batch_size,
                strict=strict_group_mode,
            )
            sample_ids = self._normalize_meta_per_sample(kwargs.get("sample_id"), batch_size)
            group_ids = self._normalize_meta_per_sample(kwargs.get("group_id"), batch_size)
            request_ids = self._normalize_meta_per_sample(kwargs.get("request_id"), batch_size)

            if strict_group_mode and branch_per_sample is None:
                raise ValueError(
                    "Strict group mode requires `branch` for each sample, but branch is missing."
                )

            if branch_per_sample is None:
                inject_motion_mask = []
                for i in range(batch_size):
                    has_anchor = self._sample_has_motion_placeholder(
                        input_ids_2d[i], motion_pad_token
                    )
                    has_motion = per_sample_motion[i] is not None
                    if not prefill_phase and has_motion:
                        raise ValueError(
                            f"decode phase must not receive motion at batch_index={i}"
                        )
                    if prefill_phase and has_anchor != has_motion:
                        raise ValueError(
                            "motion payload/anchor presence disagree and no explicit "
                            f"branch was supplied at batch_index={i}"
                        )
                    inject_motion_mask.append(
                        bool(prefill_phase and has_anchor and has_motion)
                    )
            else:
                modalities = tuple(
                    Modality.from_branch(item) for item in branch_per_sample
                )
                inject_motion_mask = list(
                    validate_motion_presence(
                        modalities,
                        tuple(item is not None for item in per_sample_motion),
                        prefill=prefill_phase,
                    )
                )
                expects_video = any(item.requires_video for item in modalities)
                has_video_payload = pixel_values_videos is not None
                if prefill_phase and expects_video != has_video_payload:
                    raise ValueError(
                        "video tensor presence disagrees with V/M/VM/T branch metadata"
                    )
                if not prefill_phase and has_video_payload:
                    raise ValueError("decode phase must not receive video pixels again")
                for i, (modality, inject) in enumerate(
                    zip(modalities, inject_motion_mask, strict=True)
                ):
                    has_anchor = self._sample_has_motion_placeholder(
                        input_ids_2d[i], motion_pad_token
                    )
                    if inject and not has_anchor:
                        raise ValueError(
                            f"{modality.value} sample missing strict motion anchor; "
                            f"sample_id={sample_ids[i]!r}, group_id={group_ids[i]!r}, "
                            f"request_id={request_ids[i]!r}, batch_index={i}"
                        )
                    if prefill_phase and not modality.requires_motion and has_anchor:
                        raise ValueError(
                            f"{modality.value} sample must not contain a motion anchor; "
                            f"sample_id={sample_ids[i]!r}, batch_index={i}"
                        )
            if debug_enabled:
                motion_present = [item is not None for item in per_sample_motion]
                has_placeholder = [
                    self._sample_has_motion_placeholder(input_ids_2d[i], motion_pad_token) for i in range(batch_size)
                ]
                has_motion_pad = [bool((input_ids_2d[i] == motion_pad_token).any().item()) for i in range(batch_size)]
                logger.warning(
                    "[motion-debug][forward/pre-mask] batch_size=%s branch_raw=%s branch_norm=%s "
                    "motion_input=%s has_placeholder=%s has_motion_pad=%s inject_motion_mask=%s",
                    batch_size,
                    _debug_value_summary(branch_raw),
                    _debug_value_summary(branch_per_sample),
                    motion_present,
                    has_placeholder,
                    has_motion_pad,
                    inject_motion_mask,
                )

            for i, inject in enumerate(inject_motion_mask):
                if not inject:
                    if per_sample_motion[i] is not None:
                        raise ValueError(
                            f"motion validation refused injection but row {i} still "
                            "owns a motion payload"
                        )
                    per_sample_motion[i] = None
                    per_sample_lengths[i] = None
            if debug_enabled:
                motion_after_mask = [item is not None for item in per_sample_motion]
                logger.warning(
                    "[motion-debug][forward/post-mask] motion_after_mask=%s",
                    motion_after_mask,
                )

            encoded_motion = [
                self._encode_motion_tensor(motion_tensor, sample_lengths)
                for motion_tensor, sample_lengths in zip(per_sample_motion, per_sample_lengths)
            ]
            sample_embeds = [
                self._build_single_sample_inputs_embeds(
                    sample_input_ids=input_ids_2d[i],
                    sample_motion_embeds=encoded_motion[i][0],
                    motion_pad_token=motion_pad_token,
                    expected_segment_feature_counts=encoded_motion[i][1],
                )
                for i in range(batch_size)
            ]

            inputs_embeds = torch.stack(sample_embeds, dim=0)
            input_ids = input_ids_2d

        # Replace embeddings for motion boundary tokens with a dedicated trainable embedding table.
        # This allows alignment training of <motion_start>/<motion_end> without unfreezing the whole LLM embedding.
        motion_start_id = getattr(self.config, "motion_start_token_id", None)
        motion_end_id = getattr(self.config, "motion_end_token_id", None)
        if input_ids is not None and (motion_start_id is not None or motion_end_id is not None):
            # Ensure 2D ids for masking
            input_ids_2d = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
            if motion_start_id is not None:
                ms_mask = input_ids_2d == int(motion_start_id)
                if ms_mask.any():
                    ms_emb = self.motion_boundary_embed(
                        torch.zeros(ms_mask.sum(), device=inputs_embeds.device, dtype=torch.long)
                    ).to(dtype=inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[ms_mask] = ms_emb
            if motion_end_id is not None:
                me_mask = input_ids_2d == int(motion_end_id)
                if me_mask.any():
                    me_emb = self.motion_boundary_embed(
                        torch.ones(me_mask.sum(), device=inputs_embeds.device, dtype=torch.long)
                    ).to(dtype=inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[me_mask] = me_emb

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # print('inputs_embeds after image mask: ', inputs_embeds)
            # print('inputs_embeds shape after image mask: ', inputs_embeds.shape)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None

        # print('video mask shape: ', video_mask.shape)
        # image_mask is none
        # video_mask is not none
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]  # 拉成一维
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if inputs_embeds.dtype == torch.long:
            raise ValueError(
                "inputs_embeds has dtype Long; it must be float (e.g. bfloat16). "
                "Check that the batch does not pass input_ids as inputs_embeds."
            )
        model_dtype = self.get_input_embeddings().weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        # Strip reward bookkeeping metadata before passing kwargs to language_model.
        for meta_key in _GENERATION_METADATA_KEYS:
            kwargs.pop(meta_key, None)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        outputs = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=None,
        )


        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        motion=None,
        motion_lengths=None,
        solution=None,
        answer=None,
        group_id=None,
        branch=None,
        sample_id=None,
        rollout_id=None,
        prompt_id=None,
        request_id=None,
        **kwargs,
    ):
        """与父类一致，首步传入 motion；后续步不再传 motion，避免每步重跑 motion encoder。"""
        debug_enabled = _is_motion_grpo_debug_enabled()
        # Keep rollout metadata out of parent generation prep.
        # Note: explicit args above are to satisfy HF generate() kwargs validation.
        for meta_key in _GENERATION_METADATA_KEYS:
            kwargs.pop(meta_key, None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        selected_motion, selected_lengths = prefill_motion_payload(
            motion,
            motion_lengths,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        model_inputs["motion"] = selected_motion
        model_inputs["motion_lengths"] = selected_lengths

        model_inputs["branch"] = branch
        model_inputs["sample_id"] = sample_id
        model_inputs["group_id"] = group_id
        model_inputs["rollout_id"] = rollout_id
        model_inputs["prompt_id"] = prompt_id
        model_inputs["request_id"] = request_id

        if debug_enabled:
            prefill = is_generation_prefill(
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
            logger.warning(
                "[motion-debug][prepare_inputs_for_generation] prefill=%s branch_arg=%s "
                "motion_arg=%s motion_lengths_arg=%s model_input_keys=%s",
                prefill,
                _debug_value_summary(branch),
                _debug_value_summary(motion),
                _debug_value_summary(motion_lengths),
                sorted(model_inputs.keys()),
            )

        return model_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        vqvae_path=None,
        motion_dataname=None,
        motion_quantizer=None,
        vqvae_nb_code=None,
        vqvae_code_dim=None,
        vqvae_output_emb_width=None,
        vqvae_down_t=None,
        vqvae_stride_t=None,
        vqvae_width=None,
        vqvae_depth=None,
        vqvae_dilation_growth_rate=None,
        vqvae_activation=None,
        vqvae_norm=None,
        *model_args,
        **kwargs,
        ):
        # ---- 1. 加载 / 准备配置 ----
        config = kwargs.pop("config", None)
        motion_config_overrides = kwargs.pop("motion_config_overrides", None)
        cache_dir = kwargs.get("cache_dir", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_path, cache_dir=cache_dir)
        if motion_config_overrides is not None:
            if not isinstance(motion_config_overrides, dict):
                raise TypeError("motion_config_overrides must be a dict")
            allowed_override_fields = {
                "motion_input_dim",
                "motion_encoder_output_dim",
                "motion_downsample_factor",
                "motion_placeholder_token_id",
                "motion_start_token_id",
                "motion_end_token_id",
                "motion_projector_input_dim",
                "motion_projector_hidden_dims",
                "motion_projector_output_dim",
                "motion_projector_activation",
                "motion_projector_bias",
                "motion_projector_pre_norm",
                "motion_projector_post_norm",
                "motion_resize_policy",
                "motion_compute_dtype",
                "motion_device_policy",
                "motion_allow_batch_broadcast",
                "motion_normalization_mean_path",
                "motion_normalization_std_path",
                "motion_allowed_interstitial_token_ids",
            }
            unknown = set(motion_config_overrides) - allowed_override_fields
            if unknown:
                raise ValueError(
                    f"Unknown motion config override fields: {sorted(unknown)!r}"
                )
            for key, value in motion_config_overrides.items():
                setattr(config, key, value)

        if motion_dataname is not None:
            setattr(config, "dataname", motion_dataname)
        elif not hasattr(config, "dataname"):
            setattr(config, "dataname", "kit")

        if motion_quantizer is not None:
            setattr(config, "quantizer", motion_quantizer)
        elif not hasattr(config, "quantizer"):
            setattr(config, "quantizer", "ema")

        # 设置来自函数参数的 vqvae config
        if vqvae_nb_code is not None:
            setattr(config, "vqvae_nb_code", vqvae_nb_code)
        if vqvae_code_dim is not None:
            setattr(config, "vqvae_code_dim", vqvae_code_dim)
        if vqvae_output_emb_width is not None:
            setattr(config, "vqvae_output_emb_width", vqvae_output_emb_width)
            if "motion_encoder_output_dim" not in (motion_config_overrides or {}):
                setattr(config, "motion_encoder_output_dim", vqvae_output_emb_width)
            if "motion_projector_input_dim" not in (motion_config_overrides or {}):
                setattr(config, "motion_projector_input_dim", vqvae_output_emb_width)
        if vqvae_down_t is not None:
            setattr(config, "vqvae_down_t", vqvae_down_t)
        if vqvae_stride_t is not None:
            setattr(config, "vqvae_stride_t", vqvae_stride_t)
        if vqvae_width is not None:
            setattr(config, "vqvae_width", vqvae_width)
        if vqvae_depth is not None:
            setattr(config, "vqvae_depth", vqvae_depth)
        if vqvae_dilation_growth_rate is not None:
            setattr(config, "vqvae_dilation_growth_rate", vqvae_dilation_growth_rate)
        if vqvae_activation is not None:
            setattr(config, "vqvae_activation", vqvae_activation)
        if vqvae_norm is not None:
            setattr(config, "vqvae_norm", vqvae_norm)

        # 设置 vqvae 参数的默认值（如果 config 中没有）
        if not hasattr(config, "vqvae_nb_code") or getattr(config, "vqvae_nb_code") is None:
            setattr(config, "vqvae_nb_code", 512)
        if not hasattr(config, "vqvae_code_dim") or getattr(config, "vqvae_code_dim") is None:
            setattr(config, "vqvae_code_dim", 512)
        if not hasattr(config, "vqvae_output_emb_width") or getattr(config, "vqvae_output_emb_width") is None:
            setattr(config, "vqvae_output_emb_width", 512)
        if not hasattr(config, "vqvae_down_t") or getattr(config, "vqvae_down_t") is None:
            setattr(config, "vqvae_down_t", 2)
        if not hasattr(config, "vqvae_stride_t") or getattr(config, "vqvae_stride_t") is None:
            setattr(config, "vqvae_stride_t", 2)
        if not hasattr(config, "vqvae_width") or getattr(config, "vqvae_width") is None:
            setattr(config, "vqvae_width", 512)
        if not hasattr(config, "vqvae_depth") or getattr(config, "vqvae_depth") is None:
            setattr(config, "vqvae_depth", 3)
        if not hasattr(config, "vqvae_dilation_growth_rate") or getattr(config, "vqvae_dilation_growth_rate") is None:
            setattr(config, "vqvae_dilation_growth_rate", 3)
        if not hasattr(config, "vqvae_activation") or getattr(config, "vqvae_activation") is None:
            setattr(config, "vqvae_activation", "relu")
        if not hasattr(config, "vqvae_norm") or getattr(config, "vqvae_norm") is None:
            setattr(config, "vqvae_norm", None)
        migrate_legacy_motion_config(config)
        resolve_motion_model_spec(config)

        # ---- 2. 初始化并加载 Qwen3-VL 权重 ----
        model = super(Qwen3VlMotionForConditionalGeneration, cls).from_pretrained(
            pretrained_model_path, *model_args, config=config, **kwargs
        )

        # ---- 3. 加载 motion encoder (VQVAE) 的权重 ----
        if vqvae_path is not None:
            ckpt_file = Path(vqvae_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(f"VQ-VAE .pth checkpoint not found at {ckpt_file}")
            try:
                checkpoint_payload = torch.load(
                    ckpt_file, map_location="cpu", weights_only=True
                )
            except TypeError:
                logger.warning(
                    "Installed torch lacks weights_only=True; loading the trusted "
                    "project checkpoint with the legacy API"
                )
                checkpoint_payload = torch.load(ckpt_file, map_location="cpu")
            motion_state = normalize_state_dict_keys(
                extract_state_dict(checkpoint_payload)
            )

            # ---- 进入 ZeRO-3 gather context（可选） ----
            gather_context = nullcontext()
            if is_deepspeed_zero3_enabled():
                try:
                    from deepspeed import zero as ds_zero
                    gather_context = ds_zero.GatheredParameters(model.motion_encoder.parameters(), modifier_rank=None)
                except Exception as exc:
                    logger.warning("Failed to enter ZeRO-3 gather context: %s", exc)
                    gather_context = nullcontext()

            with gather_context:
                selection = select_vq_checkpoint_state(
                    full_expected=model.motion_encoder.state_dict(),
                    encoder_expected=model.motion_encoder.encoder.state_dict(),
                    candidate=motion_state,
                )
                load_module = (
                    model.motion_encoder
                    if selection.target == "full_vqvae"
                    else model.motion_encoder.encoder
                )
                load_result = load_module.load_state_dict(
                    selection.state_dict, strict=True
                )
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "strict VQ checkpoint load contradicted the pre-load audit: "
                    f"missing={load_result.missing_keys!r}, "
                    f"unexpected={load_result.unexpected_keys!r}"
                )
            logger.info(
                "Strictly loaded and shape-audited %s motion keys (%s) from %s",
                len(selection.audit.matched_keys),
                selection.target,
                ckpt_file,
            )

        return model
