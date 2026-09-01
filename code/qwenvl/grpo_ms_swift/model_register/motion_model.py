"""Custom ms-swift model registration for Motion-r1 Qwen3-VL-Motion."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template.register import register_template
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta

logger = logging.getLogger(__name__)

QWEN_VL_FINETUNE_ROOT = Path(__file__).resolve().parents[3]
if str(QWEN_VL_FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_VL_FINETUNE_ROOT))
SRC_ROOT = QWEN_VL_FINETUNE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motionllm.training import setup_motion_tokens  # noqa: E402


MOTION_MODEL_TYPE = "motion_r1_qwen3_vl_motion"
MOTION_TEMPLATE_TYPE = "motion_r1_qwen3_vl"


class MotionR1Qwen3VLTemplate(Qwen3VLTemplate):
    """Qwen3-VL template with motion-anchor-preserving truncation for GRPO left-truncate."""

    _MOTION_TEXT_PATTERNS = (
        "<motion_start><motion><motion_end>",
        "<motion_start> <motion> <motion_end>",
        "<motion>",
    )
    _PREFIX_QID_MARKERS = ("[QID=",)
    _PREFIX_QUESTION_MARKERS = ("\nQuestion:", "Question:")
    _SUFFIX_MESSAGE_END_TOKEN = "<|im_end|>"

    def _resolve_exact_token_id(self, token: str):
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            return None
        recovered = self.tokenizer.convert_ids_to_tokens(token_id)
        if recovered != token:
            return None
        return int(token_id)

    @staticmethod
    def _find_subsequence(token_list: Sequence[int], pattern: Sequence[int], start_idx: int = 0) -> int:
        if not pattern:
            return -1
        if start_idx < 0:
            start_idx = 0
        window = len(pattern)
        upper = len(token_list) - window + 1
        for i in range(start_idx, max(upper, 0)):
            if token_list[i:i + window] == pattern:
                return i
        return -1

    def _find_motion_anchor_span(self, input_ids_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
        token_list = input_ids_tensor.tolist()

        motion_start_id = self._resolve_exact_token_id("<motion_start>")
        motion_end_id = self._resolve_exact_token_id("<motion_end>")
        if motion_start_id is not None and motion_end_id is not None:
            starts = (input_ids_tensor == motion_start_id).nonzero(as_tuple=True)[0]
            ends = (input_ids_tensor == motion_end_id).nonzero(as_tuple=True)[0]
            if starts.numel() > 0 and ends.numel() > 0:
                for start in starts.tolist():
                    valid_ends = ends[ends > start]
                    if valid_ends.numel() == 0:
                        continue
                    end = int(valid_ends[0].item())
                    return int(start), int(end + 1)

        for text in self._MOTION_TEXT_PATTERNS:
            pattern = self.tokenizer.encode(text, add_special_tokens=False)
            if not pattern:
                continue
            start = self._find_subsequence(token_list, pattern)
            if start >= 0:
                return int(start), int(start + len(pattern))
        return None

    def _find_first_text_marker_span(
        self,
        input_ids_tensor: torch.Tensor,
        markers: Sequence[str],
        start_idx: int = 0,
    ) -> Optional[Tuple[int, int]]:
        token_list = input_ids_tensor.tolist()
        for marker in markers:
            marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)
            if not marker_tokens:
                continue
            pos = self._find_subsequence(token_list, marker_tokens, start_idx=start_idx)
            if pos >= 0:
                return int(pos), int(pos + len(marker_tokens))
        return None

    def _find_qid_branch_span(self, input_ids_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
        qid_span = self._find_first_text_marker_span(input_ids_tensor, self._PREFIX_QID_MARKERS)
        if qid_span is None:
            return None
        qid_start, _ = qid_span
        question_span = self._find_first_text_marker_span(
            input_ids_tensor,
            self._PREFIX_QUESTION_MARKERS,
            start_idx=qid_start,
        )
        if question_span is None:
            # Fallback: preserve enough tokens to keep QID/BRANCH section.
            end = min(qid_start + 192, int(input_ids_tensor.shape[0]))
            return int(qid_start), int(end)
        question_start, _ = question_span
        if question_start <= qid_start:
            return None
        return int(qid_start), int(question_start)

    def _find_question_span(self, input_ids_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
        question_span = self._find_first_text_marker_span(input_ids_tensor, self._PREFIX_QUESTION_MARKERS)
        if question_span is None:
            return None
        question_start, _ = question_span

        end = int(input_ids_tensor.shape[0])
        end_token_id = self._resolve_exact_token_id(self._SUFFIX_MESSAGE_END_TOKEN)
        if end_token_id is not None:
            ends = (input_ids_tensor == int(end_token_id)).nonzero(as_tuple=True)[0]
            valid_ends = ends[ends > question_start]
            if valid_ends.numel() > 0:
                end = int(valid_ends[0].item())
        if end <= question_start:
            end = min(question_start + 768, int(input_ids_tensor.shape[0]))
        return int(question_start), int(end)

    @staticmethod
    def _merge_spans(spans: Sequence[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
        normalized: List[Tuple[int, int]] = []
        for start, end in spans:
            start = max(0, int(start))
            end = min(length, int(end))
            if end > start:
                normalized.append((start, end))
        if not normalized:
            return []
        normalized.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [normalized[0]]
        for start, end in normalized[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _collect_protected_spans(self, input_ids_tensor: torch.Tensor) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        motion_span = self._find_motion_anchor_span(input_ids_tensor)
        if motion_span is not None:
            spans.append(motion_span)
        qid_branch_span = self._find_qid_branch_span(input_ids_tensor)
        if qid_branch_span is not None:
            spans.append(qid_branch_span)
        question_span = self._find_question_span(input_ids_tensor)
        if question_span is not None:
            spans.append(question_span)
        return self._merge_spans(spans, int(input_ids_tensor.shape[0]))

    def _truncate(self, input_ids, labels, loss_scale, truncation_strategy):
        placeholder_tokens = torch.tensor(self.placeholder_tokens, dtype=torch.long)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
        n_placeholder = int(protected.sum().item())

        protected_spans = self._collect_protected_spans(input_ids_tensor)
        for start, end in protected_spans:
            protected[start:end] = True

        n_protected = int(protected.sum().item())
        if n_protected > self.max_length:
            raise ValueError(
                "Protected token span exceeds max_length. "
                f"max_length={self.max_length}, protected={n_protected}, placeholders={n_placeholder}, "
                f"protected_spans={protected_spans}"
            )

        if os.getenv("MOTION_GRPO_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info(
                "[motion-truncate] seq_len=%s max_length=%s protected=%s placeholders=%s protected_spans=%s",
                int(input_ids_tensor.shape[0]),
                int(self.max_length),
                n_protected,
                n_placeholder,
                protected_spans,
            )

        if n_protected < self.max_length:
            non_protected = (~protected).nonzero(as_tuple=True)[0]
            keep = self.max_length - n_protected
            if truncation_strategy == "left":
                idx = non_protected[-keep:]
            else:
                idx = non_protected[:keep]
            protected[idx] = True

        input_ids = input_ids_tensor[protected].tolist()
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)[protected].tolist()
            labels[0] = -100
        if loss_scale is not None:
            loss_scale = torch.tensor(loss_scale, dtype=torch.float32)[protected].tolist()
            loss_scale[0] = 0
        return input_ids, labels, loss_scale


register_template(
    QwenTemplateMeta(
        MOTION_TEMPLATE_TYPE,
        template_cls=MotionR1Qwen3VLTemplate,
        default_system=None,
        thinking_prefix="<think>\n",
    ),
    exist_ok=True,
)


class MotionR1Qwen3VLMotionLoader(ModelLoader):
    """Use Motion-r1 model class while keeping ms-swift loading pipeline."""

    _DEBUG_LOGGED = False
    _MOTION_ANCHOR_PATTERNS = (
        ("motion_text_token_ids", "<motion>"),
        ("motion_boundary_inline_token_ids", "<motion_start><motion><motion_end>"),
        ("motion_boundary_spaced_token_ids", "<motion_start> <motion> <motion_end>"),
    )

    _INT_KEYS = {
        "vqvae_nb_code",
        "vqvae_code_dim",
        "vqvae_output_emb_width",
        "vqvae_down_t",
        "vqvae_stride_t",
        "vqvae_width",
        "vqvae_depth",
        "vqvae_dilation_growth_rate",
    }
    _STR_KEYS = {
        "vqvae_path",
        "motion_dataname",
        "motion_quantizer",
        "vqvae_activation",
        "vqvae_norm",
    }

    @staticmethod
    def _normalize_none_like(value: str):
        text = value.strip()
        if text.lower() in {"", "none", "null", "nil"}:
            return None
        return text

    @classmethod
    def _load_motion_kwargs_from_env(cls):
        if os.getenv("MOTION_GRPO_FORMAL_BOUND_INPUTS") == "1":
            return {}
        merged = {}
        for key in cls._STR_KEYS.union(cls._INT_KEYS):
            env_key = key.upper()
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            normalized = cls._normalize_none_like(raw_value)
            if key in cls._INT_KEYS:
                if normalized is None:
                    continue
                try:
                    merged[key] = int(normalized)
                except ValueError:
                    logger.warning(
                        "[motion_loader] ignore invalid int env %s=%r",
                        env_key,
                        raw_value,
                    )
                continue
            # string key
            merged[key] = normalized
        return merged

    @staticmethod
    def _safe_get_tokenizer(processor):
        if processor is None:
            return None
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        # Fallback for processor wrappers with inner processors list.
        for item in getattr(processor, "processors", []) or []:
            inner = getattr(item, "tokenizer", None)
            if inner is not None:
                return inner
        return None

    @staticmethod
    def _encode_text_pattern(tokenizer, text: str):
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:  # pragma: no cover - tokenizer compatibility guard
            logger.warning("[motion_loader] failed to encode text pattern %r: %s", text, exc)
            return None
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            return None
        return [int(x) for x in token_ids]

    @staticmethod
    def _resolve_exact_token_id(tokenizer, token: str):
        token_id = None
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict) and token in vocab:
            token_id = int(vocab[token])
        if token_id is not None:
            return token_id

        try:
            raw_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(raw_id, int) and raw_id >= 0:
                recovered = tokenizer.convert_ids_to_tokens(raw_id)
                if recovered == token:
                    token_id = int(raw_id)
        except Exception:
            token_id = None
        return token_id

    @classmethod
    def _build_motion_anchor_config(cls, processor):
        tokenizer = cls._safe_get_tokenizer(processor)
        if tokenizer is None:
            return {}

        anchor_config = {}
        text_patterns = []
        for key, text in cls._MOTION_ANCHOR_PATTERNS:
            token_ids = cls._encode_text_pattern(tokenizer, text)
            if token_ids:
                anchor_config[key] = token_ids
                text_patterns.append(token_ids)
        if text_patterns:
            anchor_config["motion_text_token_patterns"] = text_patterns

        anchor_config["motion_token_id"] = cls._resolve_exact_token_id(tokenizer, "<motion>")
        anchor_config["motion_start_token_id"] = cls._resolve_exact_token_id(tokenizer, "<motion_start>")
        anchor_config["motion_end_token_id"] = cls._resolve_exact_token_id(tokenizer, "<motion_end>")
        return anchor_config

    def get_model(self, model_dir, config, processor, model_kwargs):
        from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration

        effective_kwargs = dict(model_kwargs or {})
        env_kwargs = self._load_motion_kwargs_from_env()
        merged_from_env = []
        for key, value in env_kwargs.items():
            if key not in effective_kwargs:
                effective_kwargs[key] = value
                merged_from_env.append(key)
        if os.getenv("MOTION_GRPO_FORMAL_BOUND_INPUTS") == "1" and not effective_kwargs.get(
            "vqvae_path"
        ):
            raise ValueError(
                "formal GRPO motion loader requires config-bound vqvae_path; env fallback is disabled"
            )

        anchor_cfg = self._build_motion_anchor_config(processor)
        for key, value in anchor_cfg.items():
            setattr(config, key, value)

        if not self.__class__._DEBUG_LOGGED:
            self.__class__._DEBUG_LOGGED = True
            logger.info(
                "[motion_loader] effective motion kwargs: motion_dataname=%r, motion_quantizer=%r, "
                "vqvae_path_set=%s, merged_from_env=%s, motion_anchor_keys=%s",
                effective_kwargs.get("motion_dataname"),
                effective_kwargs.get("motion_quantizer"),
                bool(effective_kwargs.get("vqvae_path")),
                sorted(merged_from_env),
                sorted(anchor_cfg.keys()),
            )

        self.auto_model_cls = self.auto_model_cls or Qwen3VlMotionForConditionalGeneration
        model = super().get_model(model_dir, config, processor, effective_kwargs)
        tokenizer = self._safe_get_tokenizer(processor)
        if tokenizer is None:
            raise ValueError("motion GRPO processor must expose a tokenizer")
        setup_motion_tokens(tokenizer, model)
        return model


register_model(
    ModelMeta(
        model_type=MOTION_MODEL_TYPE,
        model_groups=[
            ModelGroup(
                [Model(ms_model_id="motion-r1-qwen3-vl-motion", hf_model_id="motion-r1-qwen3-vl-motion")],
                template=MOTION_TEMPLATE_TYPE,
            )
        ],
        loader=MotionR1Qwen3VLMotionLoader,
        architectures=["Qwen3VLForConditionalGeneration"],
        is_multimodal=True,
        tags=["motion", "vision", "video"],
    ),
    exist_ok=True,
)
