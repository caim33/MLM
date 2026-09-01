import json
import random
import logging
import re
import time
import itertools
import torch
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
# motion 占位符与 prompt 中 "<motion>" 的 tokenizer 子词序列（需在模板里写 <motion>，以便插在 video 之后）
MOTION_PLACEHOLDER_TOKEN_ID = 160001
MOTION_PROMPT_TOKEN_IDS = (13748, 5956, 397)
# 占位符两侧以普通文本形式 tokenize 的边界标记（非 added_tokens，走 encode 子词）
MOTION_BOUNDARY_START_TEXT = "<motion_start>"
MOTION_BOUNDARY_END_TEXT = "<motion_end>"


def _extract_video_timestamp_token_spans(input_ids_1d: List[int]) -> List[List[int]]:
    """
    从 Qwen3-VL 原生 video 展开后的 input_ids 中提取每个 '<t seconds>' 的 token 切片（不重新 encode），
    以保证后续复用时数量/时间/rounding 与 video 完全一致。

    匹配规则：在每个 <|vision_start|>(151652) 之前，向前找形如
      '<' ... '.' ... ' seconds' ... '>'
    的 token 序列（其中 '<'=27, '.'=13, ' seconds'=6486, '>'=29）。
    """
    VISION_START = 151652
    LT = 27
    DOT = 13
    SECONDS = 6486
    GT = 29

    spans: List[List[int]] = []
    L = len(input_ids_1d)
    for i in range(L):
        if input_ids_1d[i] != VISION_START:
            continue
        if i - 1 < 0 or input_ids_1d[i - 1] != GT:
            continue
        j = max(0, i - 16)
        lt_pos = -1
        for p in range(i - 1, j - 1, -1):
            if input_ids_1d[p] == LT:
                lt_pos = p
                break
        if lt_pos < 0:
            continue
        cand = input_ids_1d[lt_pos:i]
        if (SECONDS in cand) and (DOT in cand) and (len(cand) >= 6) and (cand[-1] == GT):
            spans.append(cand)
    return spans


def _find_token_subsequence(input_ids_list: List[int], pattern: Tuple[int, ...]) -> int:
    """返回 pattern 在 input_ids_list 中首次出现的起始下标，未找到返回 -1。"""
    if not pattern or len(input_ids_list) < len(pattern):
        return -1
    plen = len(pattern)
    first = pattern[0]
    for i in range(len(input_ids_list) - plen + 1):
        if input_ids_list[i] != first:
            continue
        if tuple(input_ids_list[i : i + plen]) == pattern:
            return i
    return -1


def _calculate_timestamps_like_qwen3vl(
    indices: List[int], video_fps: float, merge_size: int = 2
) -> List[float]:
    """
    与 transformers Qwen3VLProcessor._calculate_timestamps 一致：
    帧下标 / fps 得秒级时间，再按 merge_size 对相邻帧做平均（与 video 占位符一致）。
    """
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
    timestamps = [idx / video_fps for idx in indices]
    timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2
        for i in range(0, len(timestamps), merge_size)
    ]
    return timestamps

# 为了排查长度不均导致的多卡问题，这里给输入序列一个统一上限
MAX_DEBUG_SEQ_LEN = 4096

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args, dataset_use_override=None, shuffle=True):
        super(LazySupervisedDataset, self).__init__()

        dataset_use = dataset_use_override or data_args.dataset_use
        dataset = dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        
        # Motion placeholder divisor
        self.motion_length_divisor = getattr(data_args, "motion_length_divisor", None)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total {'training' if shuffle else 'eval'} samples: {len(list_data_dict)}")

        if shuffle:
            random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        # -------- 按模态分桶，并预先构造组合采样的 group --------
        # motion-only:      只含 motion（且 motion 路径非空）
        # video-only:       只含 video（且 video 路径非空）
        # motion-video:     同时含 motion 与 video（且两者路径均非空）
        # text-only:        不含 motion 且不含 video（或路径为空）
        self.groups = None
        self.grouped_sampling = False
        m_indices, v_indices, mv_indices, t_indices = [], [], [], []

        for idx, ann in enumerate(list_data_dict):
            sample = ann[0] if isinstance(ann, list) and ann else ann
            if not isinstance(sample, dict):
                continue
            # NOTE: 某些数据会包含空字符串字段（例如 {"motion": "", "video": ""}），
            # 这里按“路径是否非空”判断模态是否存在。
            motion_path = sample.get("motion")
            video_path = sample.get("video")
            has_motion = bool(motion_path)
            has_video = bool(video_path)
            if has_motion and has_video:
                mv_indices.append(idx)
            elif has_motion:
                m_indices.append(idx)
            elif has_video:
                v_indices.append(idx)
            else:
                t_indices.append(idx)

        # 默认配比：2(motion-only) : 1(motion-video)，不使用 video-only / text-only
        # 可通过环境变量从 bash 控制：
        #   GROUP_NUM_MV, GROUP_NUM_MOTION, GROUP_NUM_VIDEO, GROUP_NUM_TEXT
        # 例如：GROUP_NUM_MV=1 GROUP_NUM_MOTION=1 GROUP_NUM_VIDEO=1 GROUP_NUM_TEXT=1
        group_num_mv = int(os.getenv("GROUP_NUM_MV", "1"))
        group_num_m = int(os.getenv("GROUP_NUM_MOTION", "2"))
        group_num_v = int(os.getenv("GROUP_NUM_VIDEO", "0"))
        group_num_t = int(os.getenv("GROUP_NUM_TEXT", "0"))

        # 兼容：全为 0 时直接不启用
        if (group_num_mv + group_num_m + group_num_v + group_num_t) <= 0:
            rank0_print("Grouped sampling disabled (all GROUP_NUM_* are 0).")
        else:
            caps = []
            if group_num_m > 0:
                caps.append(len(m_indices) // group_num_m)
            if group_num_v > 0:
                caps.append(len(v_indices) // group_num_v)
            if group_num_mv > 0:
                caps.append(len(mv_indices) // group_num_mv)
            if group_num_t > 0:
                caps.append(len(t_indices) // group_num_t)

            max_groups = min(caps) if caps else 0
            if max_groups > 0:
                random.shuffle(m_indices)
                random.shuffle(v_indices)
                random.shuffle(mv_indices)
                random.shuffle(t_indices)

                groups = []
                for i in range(max_groups):
                    group = []
                    if group_num_m > 0:
                        start = i * group_num_m
                        group.extend(m_indices[start : start + group_num_m])
                    if group_num_v > 0:
                        start = i * group_num_v
                        group.extend(v_indices[start : start + group_num_v])
                    if group_num_mv > 0:
                        start = i * group_num_mv
                        group.extend(mv_indices[start : start + group_num_mv])
                    if group_num_t > 0:
                        start = i * group_num_t
                        group.extend(t_indices[start : start + group_num_t])
                    groups.append(tuple(group))

                self.groups = groups
                self.grouped_sampling = True
                rank0_print(
                    f"Enabled grouped sampling (m={group_num_m}, v={group_num_v}, mv={group_num_mv}, t={group_num_t}), "
                    f"total groups: {len(self.groups)}"
                )
            else:
                rank0_print(
                    "Grouped sampling not enabled (insufficient modality variety for requested GROUP_NUM_*)."
                )

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # motion 段两侧边界：作为 special tokens（单 token id），不再走文本子词 encode
        tok = self.tokenizer
        ms_id = tok.convert_tokens_to_ids(MOTION_BOUNDARY_START_TEXT)
        me_id = tok.convert_tokens_to_ids(MOTION_BOUNDARY_END_TEXT)
        if ms_id is None or me_id is None:
            raise ValueError("Motion boundary tokens must exist in tokenizer as special tokens.")
        # Some tokenizers return unk id for missing tokens; guard against that.
        unk_id = getattr(tok, "unk_token_id", None)
        if unk_id is not None and (ms_id == unk_id or me_id == unk_id):
            raise ValueError(
                "Motion boundary tokens were mapped to unk_token_id; "
                "ensure full_sft.py adds <motion_start>/<motion_end> as special tokens before dataset init."
            )
        self._motion_start_ids = torch.tensor([int(ms_id)], dtype=torch.long)
        self._motion_end_ids = torch.tensor([int(me_id)], dtype=torch.long)
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        # 加载motion归一化参数
        mean_path = Path(__file__).parent / "Mean.npy"
        std_path = Path(__file__).parent / "Std.npy"
        if mean_path.exists() and std_path.exists():
            self.motion_mean = torch.tensor(np.load(mean_path))
            self.motion_std = torch.tensor(np.load(std_path))
            rank0_print(f"Loaded motion normalization: mean shape {self.motion_mean.shape}, std shape {self.motion_std.shape}")
        else:
            raise FileNotFoundError(f"Motion normalization files not found: {mean_path} or {std_path}")

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        if getattr(self, "grouped_sampling", False) and self.groups is not None:
            return len(self.groups)
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 如果启用了按模态分组的采样，这里 i 表示第 i 个 group
        if getattr(self, "grouped_sampling", False) and self.groups is not None:
            group = self.groups[i]
            # 兼容不同 group 结构（例如 1:1:2 m:v:mv 或 2:1 m:mv）
            sources = [self.list_data_dict[idx] for idx in group]
            # 分组采样只在 data_flatten 或 data_packing 模式下有意义
            if not (self.data_args.data_flatten or self.data_args.data_packing):
                raise ValueError(
                    "Grouped sampling requires data_flatten or data_packing to be True."
                )

            return self._get_packed_item(sources)

        # 默认行为：单条样本 lazy 读取
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _build_motion_core_sequences(
        self,
        num_placeholders: int,
        motion_tensor: torch.Tensor,
        divisor: int,
        dev: torch.device,
        id_dtype: torch.dtype,
        orig_labels: torch.Tensor,
        orig_attention_mask: torch.Tensor,
        video_timestamp_token_spans: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Motion 占位符序列（撤销“为 motion 插入 <t seconds> 时间戳文本”的改动）：
        - Video 时间戳仍由 Qwen3-VL processor 原生机制处理；
        - Motion 段只保留：motion boundary start + N 个 motion placeholder + motion boundary end。
        """
        ms = self._motion_start_ids.to(device=dev, dtype=id_dtype)
        me = self._motion_end_ids.to(device=dev, dtype=id_dtype)

        ids_parts: List[torch.Tensor] = [ms]
        lb_parts: List[torch.Tensor] = [
            torch.full(
                (ms.numel(),),
                -100,
                dtype=orig_labels.dtype,
                device=orig_labels.device,
            )
        ]
        attn_parts: List[torch.Tensor] = [
            torch.ones(
                (ms.numel(),),
                dtype=orig_attention_mask.dtype,
                device=orig_attention_mask.device,
            )
        ]

        use_sync = bool(getattr(self.data_args, "motion_timestamps_sync_with_video", False))

        # --- timestamps: prefer exact video spans; otherwise compute like Qwen3-VL ---
        tok = self.tokenizer
        motion_fps = float(getattr(self.data_args, "motion_fps", 30.0))
        g = getattr(self.data_args, "motion_placeholders_per_timestamp", None)
        if g is None:
            vp = getattr(self.processor, "video_processor", None)
            g = getattr(vp, "merge_size", None) if vp is not None else None
        if g is None:
            g = 1
        group = max(1, int(g))

        def _append_ts_tokens(ts_tokens: List[int]):
            ts_ids = torch.tensor(ts_tokens, dtype=id_dtype, device=dev)
            ids_parts.append(ts_ids)
            lb_parts.append(
                torch.full(
                    (ts_ids.numel(),),
                    -100,
                    dtype=orig_labels.dtype,
                    device=orig_labels.device,
                )
            )
            attn_parts.append(
                torch.ones(
                    (ts_ids.numel(),),
                    dtype=orig_attention_mask.dtype,
                    device=orig_attention_mask.device,
                )
            )

        if use_sync and video_timestamp_token_spans:
            n_ts = len(video_timestamp_token_spans)
            n_ph = int(num_placeholders)
            base = n_ph // n_ts
            rem = n_ph % n_ts
            ph_left = n_ph
            for ti, ts_tokens in enumerate(video_timestamp_token_spans):
                _append_ts_tokens(ts_tokens)
                k = base + (1 if ti < rem else 0)
                if k > ph_left:
                    k = ph_left
                for _ in range(k):
                    ph = torch.tensor([MOTION_PLACEHOLDER_TOKEN_ID], dtype=id_dtype, device=dev)
                    ids_parts.append(ph)
                    lb_parts.append(
                        torch.full(
                            (1,),
                            -100,
                            dtype=orig_labels.dtype,
                            device=orig_labels.device,
                        )
                    )
                    attn_parts.append(
                        torch.ones(
                            (1,),
                            dtype=orig_attention_mask.dtype,
                            device=orig_attention_mask.device,
                        )
                    )
                ph_left -= k
            for _ in range(ph_left):
                ph = torch.tensor([MOTION_PLACEHOLDER_TOKEN_ID], dtype=id_dtype, device=dev)
                ids_parts.append(ph)
                lb_parts.append(
                    torch.full(
                        (1,),
                        -100,
                        dtype=orig_labels.dtype,
                        device=orig_labels.device,
                    )
                )
                attn_parts.append(
                    torch.ones(
                        (1,),
                        dtype=orig_attention_mask.dtype,
                        device=orig_attention_mask.device,
                    )
                )
        elif use_sync:
            # motion-only fallback: follow Qwen3-VL video logic to decide timestamp count/spacing.
            # 1) Treat motion steps as frames with fps=motion_fps (default 30).
            # 2) Uniformly sample frames at target_fps (reuse data_args.video_fps, default 2).
            # 3) Convert sampled frames to temporal patches with temporal_patch_size (default 2) -> T patches.
            # 4) Generate exactly T '<t seconds>' tokens, and distribute motion pads across T.
            L = int(motion_tensor.shape[0])
            vp = getattr(self.processor, "video_processor", None)
            merge_size_ts = int(getattr(vp, "merge_size", 2)) if vp is not None else 2
            temporal_patch_size = int(getattr(vp, "temporal_patch_size", 2)) if vp is not None else 2
            target_fps = float(getattr(self.data_args, "video_fps", 2))

            total_num_frames = max(1, L)
            # Same spirit as qwen_vl_utils.smart_nframes():
            # - compute desired frame count by fps
            # - clamp to [min_frames, max_frames] (and <= total_num_frames)
            # - align to a factor for later temporal merging
            min_frames = int(getattr(self.data_args, "video_min_frames", 1) or 1)
            max_frames = getattr(self.data_args, "video_max_frames", None)
            max_frames = int(max_frames) if max_frames is not None else total_num_frames
            max_frames = min(max_frames, total_num_frames)

            # desired frames from fps (use floor like int())
            n_fps_raw = (total_num_frames / max(float(motion_fps), 1e-6)) * max(float(target_fps), 1e-6)
            num_frames = int(n_fps_raw)
            # clamp (still before factor alignment)
            num_frames = min(max(num_frames, min_frames), max_frames)
            num_frames = min(max(num_frames, 1), total_num_frames)

            # Align to factor=2 by default (matches Qwen video merge behavior).
            # Also ensure divisibility for timestamp merge/temporal patching when possible.
            frame_factor = 2
            if total_num_frames >= frame_factor:
                num_frames = (num_frames // frame_factor) * frame_factor
                num_frames = max(frame_factor, min(num_frames, total_num_frames))
            else:
                num_frames = 1

            # emulate sample_frames(): linspace + round
            if num_frames == 1:
                indices = [0]
            else:
                indices = (
                    np.linspace(0, total_num_frames - 1, num_frames).round().astype(int).tolist()
                )

            # timestamps are calculated from sampled indices (same as processor)
            ts_list = _calculate_timestamps_like_qwen3vl(indices, motion_fps, merge_size=merge_size_ts)

            # Compute T patches (video_grid_thw[0]) logic:
            # pad frames to divisible by temporal_patch_size then // temporal_patch_size
            n_frames_padded = int(num_frames)
            if temporal_patch_size > 1:
                r = n_frames_padded % temporal_patch_size
                if r != 0:
                    n_frames_padded += (temporal_patch_size - r)
            T = max(1, n_frames_padded // max(int(temporal_patch_size), 1))

            # make sure we have exactly T timestamps; if mismatch, pad/trim using last value
            if len(ts_list) < T:
                ts_list = ts_list + [ts_list[-1]] * (T - len(ts_list))
            elif len(ts_list) > T:
                ts_list = ts_list[:T]
            n_ts = T
            n_ph = int(num_placeholders)
            base = n_ph // n_ts
            rem = n_ph % n_ts
            ph_left = n_ph

            for ti in range(n_ts):
                t_sec = float(ts_list[ti])
                ts_str = f"<{t_sec:.1f} seconds>"
                ts_tokens = tok.encode(ts_str, add_special_tokens=False)
                _append_ts_tokens(ts_tokens)

                k = base + (1 if ti < rem else 0)
                if k > ph_left:
                    k = ph_left
                for _ in range(k):
                    ph = torch.tensor([MOTION_PLACEHOLDER_TOKEN_ID], dtype=id_dtype, device=dev)
                    ids_parts.append(ph)
                    lb_parts.append(
                        torch.full(
                            (1,),
                            -100,
                            dtype=orig_labels.dtype,
                            device=orig_labels.device,
                        )
                    )
                    attn_parts.append(
                        torch.ones(
                            (1,),
                            dtype=orig_attention_mask.dtype,
                            device=orig_attention_mask.device,
                        )
                    )
                ph_left -= k

            for _ in range(ph_left):
                ph = torch.tensor([MOTION_PLACEHOLDER_TOKEN_ID], dtype=id_dtype, device=dev)
                ids_parts.append(ph)
                lb_parts.append(
                    torch.full(
                        (1,),
                        -100,
                        dtype=orig_labels.dtype,
                        device=orig_labels.device,
                    )
                )
                attn_parts.append(
                    torch.ones(
                        (1,),
                        dtype=orig_attention_mask.dtype,
                        device=orig_attention_mask.device,
                    )
                )
        else:
            for _ in range(int(num_placeholders)):
                ph = torch.tensor([MOTION_PLACEHOLDER_TOKEN_ID], dtype=id_dtype, device=dev)
                ids_parts.append(ph)
                lb_parts.append(
                    torch.full(
                        (1,),
                        -100,
                        dtype=orig_labels.dtype,
                        device=orig_labels.device,
                    )
                )
                attn_parts.append(
                    torch.ones(
                        (1,),
                        dtype=orig_attention_mask.dtype,
                        device=orig_attention_mask.device,
                    )
                )

        ids_parts.append(me)
        lb_parts.append(
            torch.full(
                (me.numel(),),
                -100,
                dtype=orig_labels.dtype,
                device=orig_labels.device,
            )
        )
        attn_parts.append(
            torch.ones(
                (me.numel(),),
                dtype=orig_attention_mask.dtype,
                device=orig_attention_mask.device,
            )
        )

        return (
            torch.cat(ids_parts, dim=0),
            torch.cat(lb_parts, dim=0),
            torch.cat(attn_parts, dim=0),
        )

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )

        # 获取基础路径和 motion 文件路径（motion 可选：支持仅 video / 仅 text）
        if isinstance(sources, dict):
            source = sources
            motion_path = source.get("motion")
        elif isinstance(sources, list):
            source = sources[0]
            motion_path = source.get("motion")
        else:
            raise TypeError(f"Unexpected type for sources: {type(sources)}")

        # 统一把空字符串视为缺失
        if motion_path == "":
            motion_path = None
        
        if motion_path is not None:
            # 从 npy 文件加载 motion 数据
            base_path = Path(source.get("data_path", ""))
            if base_path:
                motion_file = base_path / motion_path if not Path(motion_path).is_absolute() else Path(motion_path)
            else:
                motion_file = Path(motion_path)
            
            # 加载 npy 文件并转换为 tensor
            motion_data = np.load(motion_file)
            motion_tensor = torch.tensor(motion_data)

            # 使用 Mean 和 Std 进行归一化
            motion_tensor = (motion_tensor - self.motion_mean) / self.motion_std
            data_dict["motion"] = motion_tensor
            # 记录 debug 信息：motion 文件路径与原始长度
            data_dict["motion_path"] = str(motion_file)
            data_dict["motion_raw_length"] = int(motion_data.shape[0])

        # 获取labels，找到最后一个-100的索引（即输入部分的末尾），这个位置插入占位符
        labels_tensor = data_dict["labels"][0]
        input_ids_tensor = data_dict["input_ids"][0]

        motion_tensor = data_dict.get("motion", None)
        if torch.is_tensor(motion_tensor):
            divisor = self.motion_length_divisor
            if divisor is None:
                divisor = 4  # default fallback
            # 将 motion 长度补齐到能被 divisor 整除的长度（不足部分用最后一帧重复）
            length = motion_tensor.shape[0]
            remainder = length % divisor
            if remainder != 0:
                pad_len = divisor - remainder
                last_frame = motion_tensor[-1:].clone()
                pad_frames = last_frame.repeat(pad_len, *([1] * (motion_tensor.dim() - 1)))
                motion_tensor = torch.cat([motion_tensor, pad_frames], dim=0)
                data_dict["motion"] = motion_tensor
            num_placeholders = max(1, motion_tensor.shape[0] // divisor)
        else:
            # 无 motion：不插入 motion 占位符，保持纯文本 / 仅 video 输入
            num_placeholders = 0

        # 找到输入数据的分界点（最后一个-100位置），输出部分就在其后
        labels_np = labels_tensor.cpu().numpy()
        # -100为输入label的部分
        input_mask = (labels_np == -100)
        if not input_mask.any():
            insert_pos = 0
        else:
            # 输入部分的最后位置就是最后一个-100
            insert_pos = int((input_mask).nonzero()[0][-1])

        # 在input_ids/labels/attention_mask插入/填充
        # 插入位置 = 输入部分的最后一个-100后面（即insert_pos+1）
        insert_index = insert_pos + 1

        # 构造新的input_ids、labels、attention_mask，插入前/后分割
        orig_input_ids = input_ids_tensor
        orig_labels = labels_tensor
        orig_attention_mask = torch.ones_like(orig_input_ids)

        if num_placeholders > 0:
            dev = orig_input_ids.device
            id_dtype = orig_input_ids.dtype
            div = self.motion_length_divisor
            if div is None:
                div = 4
            video_ts_spans: Optional[List[List[int]]] = None
            if getattr(self.data_args, "motion_timestamps_sync_with_video", False):
                # 只在该样本确实包含 video 时同步（避免 motion-only 样本莫名多出时间戳）
                if "video_grid_thw" in data_dict and data_dict.get("video_grid_thw") is not None:
                    video_ts_spans = _extract_video_timestamp_token_spans(
                        orig_input_ids.cpu().tolist()
                    )
                    if not video_ts_spans:
                        raise ValueError(
                            "motion_timestamps_sync_with_video=True but failed to extract any video '<t seconds>' tokens. "
                            "Please check processor/video settings or disable sync."
                        )
            motion_core_ids, motion_core_labels, motion_core_attn = (
                self._build_motion_core_sequences(
                    num_placeholders,
                    data_dict["motion"],
                    div,
                    dev,
                    id_dtype,
                    orig_labels,
                    orig_attention_mask,
                    video_timestamp_token_spans=video_ts_spans,
                )
            )

            ids_list = orig_input_ids.cpu().tolist()
            motion_span = _find_token_subsequence(ids_list, MOTION_PROMPT_TOKEN_IDS)
            if motion_span >= 0:
                # 用 motion 占位符替换 prompt 中的 <motion>（位于 video token 之后），不再追加到全文末尾
                span_len = len(MOTION_PROMPT_TOKEN_IDS)
                new_input_ids = torch.cat(
                    [
                        orig_input_ids[:motion_span],
                        motion_core_ids,
                        orig_input_ids[motion_span + span_len :],
                    ]
                )
                new_labels = torch.cat(
                    [
                        orig_labels[:motion_span],
                        motion_core_labels,
                        orig_labels[motion_span + span_len :],
                    ]
                )
                new_attention_mask = torch.cat(
                    [
                        orig_attention_mask[:motion_span],
                        motion_core_attn,
                        orig_attention_mask[motion_span + span_len :],
                    ]
                )
            else:
                # 模板中无 <motion> 时保持旧行为：插在输入段末尾（最后一个 -100 之后）
                new_input_ids = torch.cat(
                    [orig_input_ids[:insert_index], motion_core_ids, orig_input_ids[insert_index:]]
                )
                new_labels = torch.cat(
                    [orig_labels[:insert_index], motion_core_labels, orig_labels[insert_index:]]
                )
                new_attention_mask = torch.cat(
                    [orig_attention_mask[:insert_index], motion_core_attn, orig_attention_mask[insert_index:]]
                )
        else:
            # 不插入占位符，保持原始序列
            new_input_ids = orig_input_ids
            new_labels = orig_labels
            new_attention_mask = orig_attention_mask

        # data_dict赋值
        data_dict["input_ids"] = new_input_ids.detach().clone().unsqueeze(0)
        data_dict["labels"] = new_labels.detach().clone().unsqueeze(0)
        data_dict["attention_mask"] = new_attention_mask.detach().clone().unsqueeze(0)

        # 统一截断到固定上限（再也不超过 MAX_DEBUG_SEQ_LEN）
        max_len = min(self.tokenizer.model_max_length, MAX_DEBUG_SEQ_LEN)
        data_dict["input_ids"] = data_dict["input_ids"][:, :max_len]
        data_dict["labels"] = data_dict["labels"][:, :max_len]
        data_dict["attention_mask"] = data_dict["attention_mask"][:, :max_len]

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        # 对 sources 中的每一条样本，先走一遍 _get_item，然后再把结果拼接起来。
        if isinstance(sources, dict):
            sources = [sources]

        assert isinstance(sources, list), f"Unexpected type for sources in _get_packed_item: {type(sources)}"

        data_list: List[Dict[str, torch.Tensor]] = []
        for source in sources:
            # _get_item 期望传入 [dict]
            if isinstance(source, dict):
                wrapped = [source]
            else:
                wrapped = source
            assert len(wrapped) == 1, f"Don't know why it is wrapped to a list.\n {wrapped}"
            data_list.append(self._get_item(wrapped))

        '''for data_d in data_list:
            print(f"data: {data_d['motion_path']}")'''

        # 文本 / RoPE 等在各自子样本内部已经处理，这里只做拼接
        input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
        labels = torch.cat([d["labels"] for d in data_list], dim=1)
        position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
        attention_mask = [
            d["attention_mask"][0] for d in data_list if "attention_mask" in d
        ]

        new_data_dict: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask if attention_mask else None,
        }

        # 图像模态
        if any("pixel_values" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values": torch.cat(
                        [d["pixel_values"] for d in data_list if "pixel_values" in d],
                        dim=0,
                    ),
                    "image_grid_thw": torch.cat(
                        [d["image_grid_thw"] for d in data_list if "image_grid_thw" in d],
                        dim=0,
                    ),
                }
            )

        # 视频模态
        if any("pixel_values_videos" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values_videos": torch.cat(
                        [
                            d["pixel_values_videos"]
                            for d in data_list
                            if "pixel_values_videos" in d
                        ],
                        dim=0,
                    ),
                    "video_grid_thw": torch.cat(
                        [
                            d["video_grid_thw"]
                            for d in data_list
                            if "video_grid_thw" in d
                        ],
                        dim=0,
                    ),
                }
            )

        # motion 模态：完全复用 _get_item 里的归一化、补齐、占位符逻辑，
        # 这里只在时间维度上拼接，并保留每个子样本的路径、原始长度和拼接前每条 motion 的长度。
        if any("motion" in d for d in data_list):
            motion_tensors = [d["motion"] for d in data_list if "motion" in d]
            new_data_dict["motion"] = torch.cat(motion_tensors, dim=0)
            # 记录该 packed 样本内部每一条 motion 的长度，供 collator 在 batch 维度上进一步拼接
            new_data_dict["motion_lengths"] = [
                d["motion"].shape[0] for d in data_list if "motion" in d
            ]
            new_data_dict["motion_path"] = [
                d.get("motion_path", None) for d in data_list if "motion" in d
            ]
            new_data_dict["motion_raw_length"] = [
                d.get("motion_raw_length", None) for d in data_list if "motion" in d
            ]
        return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        # 支持同一 batch 内混合有 motion / 无 motion 的样本：
        # 仅对包含 "motion" 字段的样本收集并拼接，其余样本按纯 text / video 处理。
        motion_list = []
        motion_lengths = []
        motion_paths = []
        motion_raw_lengths = []
        for instance in instances:
            if "motion" in instance:
                motion_tensor = instance["motion"]
                motion_list.append(motion_tensor)
                # 如果样本（例如 packed 样本）已经在 new_data_dict 中提供了每条 motion 的长度列表，
                # 这里在 batch 维度上进行“拼接”，得到 batch 内所有 motion 的长度序列。
                if "motion_lengths" in instance:
                    lengths = instance["motion_lengths"]
                    if isinstance(lengths, (list, tuple)):
                        motion_lengths.extend(int(l) for l in lengths)
                    else:
                        motion_lengths.append(int(lengths))
                else:
                    # 普通未打包样本：直接使用当前 motion 的时间步长度
                    motion_lengths.append(int(motion_tensor.shape[0]))
                motion_paths.append(instance.get("motion_path", None))
                motion_raw_lengths.append(instance.get("motion_raw_length", None))

        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        if motion_list:
            motion = torch.cat(motion_list, dim=0)
            # 记录每条含 motion 样本的时间步数及 debug 信息
            batch["motion"] = motion
            batch["motion_lengths"] = motion_lengths
            batch["motion_path"] = motion_paths
            batch["motion_raw_length"] = motion_raw_lengths
        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args, shuffle=True)
    eval_dataset = None
    if getattr(data_args, "eval_dataset_use", None):
        eval_dataset = LazySupervisedDataset(
            processor, data_args=data_args,
            dataset_use_override=data_args.eval_dataset_use,
            shuffle=False,
        )
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
