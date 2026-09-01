"""Torch collators that preserve the framework-neutral ownership contract."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from motionllm.data import CollationPlan, plan_collation

from .processor import IGNORE_INDEX, QwenDataAdapterError


def _tokenizer_limit(tokenizer: Any) -> int:
    limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise QwenDataAdapterError("tokenizer.model_max_length must be a positive integer")
    return limit


def _pad_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QwenDataAdapterError("tokenizer.pad_token_id must be configured")
    return value


def _sequence_tensors(
    instances: Sequence[Mapping[str, Any]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    ids: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    for index, instance in enumerate(instances):
        try:
            input_ids = instance["input_ids"]
            label_ids = instance["labels"]
            position_ids = instance["position_ids"]
        except KeyError as exc:
            raise QwenDataAdapterError(
                f"instance {index} is missing required tensor {exc.args[0]!r}"
            ) from exc
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise QwenDataAdapterError(f"instance {index} input_ids must have shape (1, L)")
        if not torch.is_tensor(label_ids) or label_ids.shape != input_ids.shape:
            raise QwenDataAdapterError(f"instance {index} labels must match input_ids")
        if (
            not torch.is_tensor(position_ids)
            or position_ids.ndim != 3
            or position_ids.shape[:2] != (3, 1)
            or position_ids.shape[2] != input_ids.shape[1]
        ):
            raise QwenDataAdapterError(
                f"instance {index} position_ids must have shape (3, 1, L)"
            )
        ids.append(input_ids.squeeze(0))
        labels.append(label_ids.squeeze(0))
        positions.append(position_ids)
    return ids, labels, positions


def _pad_positions(position_ids: Sequence[torch.Tensor]) -> torch.Tensor:
    maximum = max(value.shape[2] for value in position_ids)
    padded = [
        torch.nn.functional.pad(value, (0, maximum - value.shape[2]), "constant", 1)
        for value in position_ids
    ]
    return torch.cat(padded, dim=1)


def _cat_optional(
    instances: Sequence[Mapping[str, Any]],
    value_key: str,
    grid_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    values: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    for index, instance in enumerate(instances):
        value = instance.get(value_key)
        grid = instance.get(grid_key)
        if (value is None) != (grid is None):
            raise QwenDataAdapterError(
                f"instance {index} must provide {value_key} and {grid_key} together"
            )
        if value is None:
            continue
        if not torch.is_tensor(value) or not torch.is_tensor(grid):
            raise QwenDataAdapterError(
                f"instance {index} {value_key}/{grid_key} must be tensors"
            )
        values.append(value)
        grids.append(grid)
    if not values:
        return None, None
    return torch.cat(values, dim=0), torch.cat(grids, dim=0)


def _validate_video_presence(
    instances: Sequence[Mapping[str, Any]], plan: CollationPlan
) -> None:
    for index, (instance, branch) in enumerate(
        zip(instances, plan.physical_branches, strict=True)
    ):
        has_values = instance.get("pixel_values_videos") is not None
        has_grid = instance.get("video_grid_thw") is not None
        expected = branch in {"v", "vm"}
        if has_values != has_grid or has_values != expected:
            raise ValueError(
                f"instance {index} video tensor presence disagrees with branch {branch!r}"
            )


def _physical_identities(
    instances: Sequence[Mapping[str, Any]], field_name: str
) -> list[str | tuple[str, ...]]:
    result: list[str | tuple[str, ...]] = []
    for instance in instances:
        logical = instance.get("logical_samples")
        if logical is None:
            result.append(instance[field_name])
        else:
            values = tuple(row[field_name] for row in logical)
            result.append(values[0] if len(values) == 1 else values)
    return result


def _aggregate_branch(branches: Sequence[str]) -> str:
    has_video = any(value in {"v", "vm"} for value in branches)
    has_motion = any(value in {"m", "vm"} for value in branches)
    return {
        (True, True): "vm",
        (True, False): "v",
        (False, True): "m",
        (False, False): "t",
    }[(has_video, has_motion)]


@dataclass
class DataCollatorForSupervisedDataset:
    """Pad physical rows while preserving per-row motion ownership."""

    tokenizer: Any

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        plan = plan_collation(instances)
        _validate_video_presence(instances, plan)
        input_rows, label_rows, positions = _sequence_tensors(instances)
        limit = _tokenizer_limit(self.tokenizer)
        if any(row.numel() > limit for row in input_rows):
            raise QwenDataAdapterError(
                "preprocessed sequence exceeds model_max_length; refusing unsafe multimodal truncation"
            )
        pad_id = _pad_token_id(self.tokenizer)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_rows, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            label_rows, batch_first=True, padding_value=IGNORE_INDEX
        )
        pixel_values, image_grid = _cat_optional(
            instances, "pixel_values", "image_grid_thw"
        )
        video_values, video_grid = _cat_optional(
            instances, "pixel_values_videos", "video_grid_thw"
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_id),
            "position_ids": _pad_positions(positions),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid,
            "pixel_values_videos": video_values,
            "video_grid_thw": video_grid,
            "motion": list(plan.motions),
            "motion_lengths": list(plan.motion_lengths),
            "branch": list(plan.physical_branches),
            "sample_id": _physical_identities(instances, "sample_id"),
            "group_id": _physical_identities(instances, "group_id"),
            "packed_branch": plan.packed_branches,
            "motion_owner_indices": plan.motion_owner_indices,
        }


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Flatten physical rows into one packed row without losing logical owners."""

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        plan = plan_collation(instances)
        _validate_video_presence(instances, plan)
        input_rows, label_rows, positions = _sequence_tensors(instances)
        total_length = sum(row.numel() for row in input_rows)
        if total_length > _tokenizer_limit(self.tokenizer):
            raise QwenDataAdapterError(
                "packed sequence exceeds model_max_length; refusing unsafe multimodal truncation"
            )
        lengths: list[int] = []
        for index, instance in enumerate(instances):
            raw = instance.get("attention_mask")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise QwenDataAdapterError(
                    f"instance {index} attention_mask must be a sequence of segment lengths"
                )
            for value in raw:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise QwenDataAdapterError("packed attention lengths must be positive integers")
                lengths.append(value)
        if sum(lengths) != total_length:
            raise QwenDataAdapterError(
                "packed attention lengths do not equal concatenated token length"
            )
        cumulative = torch.cumsum(
            torch.tensor([0, *lengths], dtype=torch.int32), dim=0, dtype=torch.int32
        )
        pixel_values, image_grid = _cat_optional(
            instances, "pixel_values", "image_grid_thw"
        )
        video_values, video_grid = _cat_optional(
            instances, "pixel_values_videos", "video_grid_thw"
        )
        motion_values = [value for value in plan.motions if value is not None]
        flat_motion_lengths = tuple(
            itertools.chain.from_iterable(
                value for value in plan.motion_lengths if value is not None
            )
        )
        batch: dict[str, Any] = {
            "input_ids": torch.cat(input_rows, dim=0).unsqueeze(0),
            "labels": torch.cat(label_rows, dim=0).unsqueeze(0),
            "attention_mask": cumulative,
            "position_ids": torch.cat(positions, dim=2),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid,
            "pixel_values_videos": video_values,
            "video_grid_thw": video_grid,
            "motion_lengths": (flat_motion_lengths,) if flat_motion_lengths else None,
            "branch": _aggregate_branch(plan.packed_branches),
            "sample_id": (plan.packed_sample_ids,),
            "group_id": (plan.packed_group_ids,),
            "packed_branch": plan.packed_branches,
            "motion_owner_indices": plan.motion_owner_indices,
        }
        if motion_values:
            batch["motion"] = torch.cat(motion_values, dim=0)
        else:
            batch["motion"] = None
        return batch


__all__ = [
    "DataCollatorForSupervisedDataset",
    "FlattenedDataCollatorForSupervisedDataset",
]
