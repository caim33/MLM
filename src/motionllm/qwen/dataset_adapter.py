"""Dataset bridge from explicit Qwen configs to canonical MotionLLM contracts."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from motionllm.data import (
    build_legacy_messages,
    describe_legacy_sample,
    logical_sample_payload,
    read_jsonl,
    resolve_media_path,
)
from motionllm.motion import (
    load_motion_array,
    load_normalization_stats,
    normalize_motion,
    prepare_motion_temporal,
)

from .collators import (
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
)
from .processor import QwenDataAdapterError, preprocess_qwen_visual, update_processor_pixels
from .registry import data_list
from .rope2d import build_position_ids


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise QwenDataAdapterError(f"annotation JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _nonfinite(value: str) -> None:
    raise QwenDataAdapterError(f"annotation JSON contains non-finite number {value}")


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_key,
            parse_constant=_nonfinite,
        )
    except QwenDataAdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenDataAdapterError(f"failed to read annotation JSON: {path}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise QwenDataAdapterError("annotation JSON root must be an array of objects")
    return payload


def _annotation_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [dict(row) for row in read_jsonl(path)]
    if path.suffix.lower() == ".json":
        return _read_json_array(path)
    raise QwenDataAdapterError("annotation file must use .json or .jsonl")


def _split_aliases(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise QwenDataAdapterError("dataset_use must be a non-empty alias string")
    aliases = value.split(",")
    if any(not alias or alias != alias.strip() for alias in aliases):
        raise QwenDataAdapterError("dataset_use aliases must be comma-separated without whitespace")
    return aliases


def _sample_rows(
    rows: list[dict[str, Any]], rate: float, *, seed: int
) -> list[dict[str, Any]]:
    if rate == 1.0:
        return rows
    if not math.isfinite(rate) or not 0.0 < rate < 1.0:
        raise QwenDataAdapterError("sampling_rate must be in (0, 1]")
    count = int(len(rows) * rate)
    if count <= 0:
        raise QwenDataAdapterError("sampling_rate selects no rows")
    selected = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in selected]


class LazySupervisedDataset(Dataset):
    """Identity-preserving Qwen dataset with same-index-only retries."""

    def __init__(
        self,
        processor: Any,
        data_args: Any,
        dataset_use_override: str | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        dataset_use = dataset_use_override or getattr(data_args, "dataset_use", None)
        configs = data_list(_split_aliases(dataset_use))
        seed = getattr(data_args, "data_seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise QwenDataAdapterError("data_seed must be an integer")
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for config_index, config in enumerate(configs):
            annotation = Path(config["annotation_path"])
            annotated_rows = []
            for source_row, raw in enumerate(_annotation_rows(annotation), start=1):
                row = dict(raw)
                row["_source_row"] = source_row
                annotated_rows.append(row)
            selected = _sample_rows(
                annotated_rows,
                float(config.get("sampling_rate", 1.0)),
                seed=seed + config_index,
            )
            for raw in selected:
                row = dict(raw)
                row_index = row["_source_row"]
                row["data_path"] = config["data_path"]
                row["media_root"] = config["media_root"]
                row["_source_path"] = str(annotation)
                for key in (
                    "motion_mean_path",
                    "motion_std_path",
                    "expected_motion_dim",
                ):
                    if key in config:
                        row[f"_{key}"] = config[key]
                try:
                    descriptor = describe_legacy_sample(row)
                    build_legacy_messages(row, media_root=config["media_root"])
                except Exception as exc:
                    sample_id = row.get("sample_id", "<missing>")
                    raise QwenDataAdapterError(
                        f"invalid sample {sample_id!r} at {annotation}:{row_index}"
                    ) from exc
                if descriptor.sample_id in seen_ids:
                    raise QwenDataAdapterError(
                        f"duplicate sample_id {descriptor.sample_id!r} at {annotation}:{row_index}"
                    )
                seen_ids.add(descriptor.sample_id)
                rows.append(row)
        if not rows:
            raise QwenDataAdapterError("resolved dataset contains no samples")
        if shuffle:
            random.Random(seed).shuffle(rows)
        self.groups: None = None
        self.grouped_sampling = False
        self.data_args = data_args
        self.processor = update_processor_pixels(processor, data_args)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise QwenDataAdapterError("processor must expose tokenizer")
        self.tokenizer = tokenizer
        self.list_data_dict = rows
        self.item_fn = self._get_packed_item if (
            bool(getattr(data_args, "data_packing", False))
        ) else self._get_item

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.list_data_dict)
        if index < 0 or index >= len(self.list_data_dict):
            raise IndexError(index)
        if getattr(self, "grouped_sampling", False) and self.groups is not None:
            sources = [self.list_data_dict[value] for value in self.groups[index]]
        else:
            source = self.list_data_dict[index]
            sources = [source] if isinstance(source, Mapping) else source
        retries = getattr(self.data_args, "sample_read_retries", 1)
        delay = getattr(self.data_args, "sample_read_retry_delay_seconds", 0.0)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries <= 0:
            raise QwenDataAdapterError("sample_read_retries must be a positive integer")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
            raise QwenDataAdapterError(
                "sample_read_retry_delay_seconds must be a non-negative number"
            )
        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                return self.item_fn(sources)
            except OSError as exc:
                last_error = exc
                if attempt + 1 < retries and delay:
                    time.sleep(float(delay))
            except Exception as exc:
                sample_id = sources[0].get("sample_id", "<missing>")
                raise RuntimeError(
                    f"sample {sample_id!r} failed at index {index}; no substitute row"
                ) from exc
        sample_id = sources[0].get("sample_id", "<missing>")
        raise RuntimeError(
            f"sample {sample_id!r} failed after {retries} same-index attempt(s); no substitute row"
        ) from last_error

    def _prepare_motion(
        self, source: Mapping[str, Any]
    ) -> tuple[torch.Tensor | None, Any | None]:
        descriptor = describe_legacy_sample(source)
        if not descriptor.modality.requires_motion:
            return None, None
        mean_path = source.get("_motion_mean_path")
        std_path = source.get("_motion_std_path")
        if mean_path is None or std_path is None:
            raise QwenDataAdapterError(
                f"motion sample {descriptor.sample_id!r} requires explicit normalization assets"
            )
        expected_dim = source.get("_expected_motion_dim")
        mean, std = load_normalization_stats(
            mean_path,
            std_path,
            expected_feature_dim=expected_dim,
            minimum_std=0.0,
        )
        motion_path = resolve_media_path(source["media_root"], source["motion"])
        motion = load_motion_array(
            motion_path, expected_feature_dim=int(mean.shape[0]), dtype=np.float32
        )
        normalized = normalize_motion(motion, mean, std, dtype=np.float32)
        divisor = getattr(self.data_args, "motion_length_divisor", None)
        if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
            raise QwenDataAdapterError(
                "motion_length_divisor must be bound from the verified model config"
            )
        prepared, contract = prepare_motion_temporal(
            normalized, downsample_factor=divisor, pad_mode="edge"
        )
        return torch.from_numpy(prepared), contract

    def _get_item(self, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if len(sources) != 1:
            raise QwenDataAdapterError("_get_item expects exactly one source")
        source = sources[0]
        descriptor = describe_legacy_sample(source)
        motion, temporal = self._prepare_motion(source)
        placeholder_count = temporal.placeholder_count if temporal is not None else None
        data = preprocess_qwen_visual(
            sources,
            self.processor,
            self.data_args,
            motion_placeholder_count=placeholder_count,
        )
        limit = getattr(self.tokenizer, "model_max_length", None)
        length = int(data["input_ids"].shape[1])
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise QwenDataAdapterError("tokenizer.model_max_length must be configured")
        if length > limit:
            raise QwenDataAdapterError(
                f"sample {descriptor.sample_id!r} exceeds model_max_length; refusing unsafe truncation"
            )
        data["position_ids"] = build_position_ids(
            self.processor,
            data,
            model_type=getattr(self.data_args, "model_type", "qwen3vl"),
        )
        data["attention_mask"] = [length]
        data["sample_id"] = descriptor.sample_id
        data["group_id"] = descriptor.group_id
        data["branch"] = descriptor.modality.branch
        motion_length = None
        if motion is not None:
            motion_length = int(motion.shape[0])
            data["motion"] = motion
            data["motion_lengths"] = [motion_length]
            data["motion_path"] = str(
                resolve_media_path(source["media_root"], source["motion"])
            )
            data["motion_raw_length"] = temporal.raw_length
        data["logical_samples"] = (
            logical_sample_payload(
                sample_id=descriptor.sample_id,
                group_id=descriptor.group_id,
                modality=descriptor.modality,
                motion_length=motion_length,
            ),
        )
        return data

    def _get_packed_item(
        self, sources: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if not sources:
            raise QwenDataAdapterError("packed source list must not be empty")
        items = [self._get_item([source]) for source in sources]
        packed: dict[str, Any] = {
            "input_ids": torch.cat([item["input_ids"] for item in items], dim=1),
            "labels": torch.cat([item["labels"] for item in items], dim=1),
            "position_ids": torch.cat([item["position_ids"] for item in items], dim=2),
            "attention_mask": [item["attention_mask"][0] for item in items],
            "logical_samples": tuple(
                logical
                for item in items
                for logical in item["logical_samples"]
            ),
        }
        for value_key, grid_key in (
            ("pixel_values", "image_grid_thw"),
            ("pixel_values_videos", "video_grid_thw"),
        ):
            values = [item[value_key] for item in items if item.get(value_key) is not None]
            grids = [item[grid_key] for item in items if item.get(grid_key) is not None]
            if values:
                packed[value_key] = torch.cat(values, dim=0)
                packed[grid_key] = torch.cat(grids, dim=0)
        motions = [item["motion"] for item in items if item.get("motion") is not None]
        if motions:
            packed["motion"] = torch.cat(motions, dim=0)
            packed["motion_lengths"] = [
                length
                for item in items
                for length in item.get("motion_lengths", ())
            ]
        return packed


def make_supervised_data_module(processor: Any, data_args: Any) -> dict[str, Any]:
    """Preserve the historical SFT factory while using explicit aliases."""

    train_dataset = LazySupervisedDataset(
        processor, data_args=data_args, shuffle=True
    )
    eval_alias = getattr(data_args, "eval_dataset_use", None)
    eval_dataset = (
        LazySupervisedDataset(
            processor,
            data_args=data_args,
            dataset_use_override=eval_alias,
            shuffle=False,
        )
        if eval_alias
        else None
    )
    collator_type = (
        FlattenedDataCollatorForSupervisedDataset
        if bool(getattr(data_args, "data_flatten", False))
        or bool(getattr(data_args, "data_packing", False))
        else DataCollatorForSupervisedDataset
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator_type(processor.tokenizer),
    }


__all__ = ["LazySupervisedDataset", "make_supervised_data_module"]
