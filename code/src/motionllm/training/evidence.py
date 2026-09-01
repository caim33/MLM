"""Runtime-only SFT evidence collection for Hugging Face trainer entrypoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from motion_eval.data import load_json_strict


class OptimizerEvidenceTracker:
    """Count optimizer events whose trainable gradients are finite and nonzero."""

    _HAS_GRADIENT = 0
    _HAS_NONFINITE = 1
    _MAX_ABSOLUTE = 2

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self.pre_optimizer_events = 0
        self.optimizer_steps = 0
        self.skipped_optimizer_steps = 0
        self.nonzero_finite_gradient_steps = 0
        self.max_gradient = 0.0

    def record_pre_optimizer_step(self, model: Any) -> None:
        if model is None:
            raise RuntimeError("optimizer evidence callback received no model")

        # Keep all per-parameter checks on the gradient's device.  The summary
        # layout permits one MAX collective to implement global OR for the two
        # flags and global maximum for the magnitude.  Non-finite magnitudes
        # are masked to zero because their dedicated flag is authoritative.
        device_summaries: dict[Any, Any] = {}
        fallback_device: Any = None
        with self._torch.no_grad():
            for parameter in model.parameters():
                parameter_device = getattr(parameter, "device", None)
                if fallback_device is None and parameter_device is not None:
                    fallback_device = parameter_device
                if not bool(getattr(parameter, "requires_grad", False)):
                    continue
                gradient = getattr(parameter, "grad", None)
                if gradient is None:
                    continue
                gradient = gradient.detach()
                device = gradient.device
                summary = device_summaries.get(device)
                if summary is None:
                    summary = self._torch.zeros(
                        3,
                        dtype=self._torch.float32,
                        device=device,
                    )
                    summary[self._HAS_GRADIENT].fill_(1.0)
                    device_summaries[device] = summary
                if gradient.numel() == 0:
                    continue

                finite = self._torch.isfinite(gradient).all()
                nonfinite = self._torch.logical_not(finite).to(
                    dtype=self._torch.float32
                )
                summary[self._HAS_NONFINITE].copy_(
                    self._torch.maximum(
                        summary[self._HAS_NONFINITE],
                        nonfinite,
                    )
                )
                max_absolute = gradient.abs().amax().to(dtype=self._torch.float32)
                max_absolute = self._torch.where(
                    finite,
                    max_absolute,
                    self._torch.zeros_like(max_absolute),
                )
                summary[self._MAX_ABSOLUTE].copy_(
                    self._torch.maximum(
                        summary[self._MAX_ABSOLUTE],
                        max_absolute,
                    )
                )

            summaries = list(device_summaries.values())
            if not summaries:
                summary = self._torch.zeros(
                    3,
                    dtype=self._torch.float32,
                    device=fallback_device if fallback_device is not None else "cpu",
                )
            else:
                summary = next(
                    (
                        candidate
                        for candidate in summaries
                        if candidate.device.type != "cpu"
                    ),
                    summaries[0],
                )
                for other in summaries:
                    if other is summary:
                        continue
                    summary = self._torch.maximum(
                        summary,
                        other.to(device=summary.device),
                    )

            distributed = getattr(self._torch, "distributed", None)
            if (
                distributed is not None
                and distributed.is_available()
                and distributed.is_initialized()
            ):
                world_size = distributed.get_world_size()
                if type(world_size) is not int or world_size < 1:
                    raise RuntimeError(
                        "formal SFT distributed world size is invalid"
                    )
                if world_size > 1:
                    distributed.all_reduce(
                        summary,
                        op=distributed.ReduceOp.MAX,
                    )

        # This is the only device-to-host synchronization performed by the
        # gradient evidence path: one three-scalar transfer per optimizer step.
        has_gradient, has_nonfinite, step_max = (
            float(value) for value in summary.detach().cpu().tolist()
        )
        if (
            not math.isfinite(has_gradient)
            or not math.isfinite(has_nonfinite)
            or has_nonfinite != 0.0
        ):
            raise RuntimeError("formal SFT observed a non-finite trainable gradient")
        self.pre_optimizer_events += 1
        if has_gradient <= 0.0 or not math.isfinite(step_max) or step_max <= 0.0:
            raise RuntimeError(
                "formal SFT optimizer step has no nonzero finite trainable gradient"
            )
        self.nonzero_finite_gradient_steps += 1
        self.max_gradient = max(self.max_gradient, step_max)

    def record_optimizer_step(self, accelerator: Any) -> None:
        """Record only a real update; AMP-overflow skips abort formal SFT.

        Transformers 4.57 invokes ``on_optimizer_step`` immediately after
        ``optimizer.step()``. Accelerate 1.13 has populated
        ``optimizer_step_was_skipped`` by then, before Trainer advances its
        global step. Requiring that signal prevents a skipped scaler update
        from being counted as fresh training evidence.
        """

        if self.pre_optimizer_events != self.optimizer_steps + 1:
            raise RuntimeError(
                "formal SFT optimizer callback order/count is inconsistent"
            )
        skipped = getattr(accelerator, "optimizer_step_was_skipped", None)
        if type(skipped) is not bool:
            raise RuntimeError(
                "formal SFT cannot verify whether the optimizer step was skipped"
            )
        if skipped:
            self.skipped_optimizer_steps += 1
            raise RuntimeError(
                "formal SFT optimizer step was skipped after AMP overflow"
            )
        self.optimizer_steps += 1


def resume_starting_global_step(checkpoint: str | Path | None) -> int:
    if checkpoint is None:
        return 0
    value = load_json_strict(Path(checkpoint) / "trainer_state.json")
    if not isinstance(value, Mapping):
        raise ValueError("resume trainer_state.json must be an object")
    step = value.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("resume trainer_state global_step must be non-negative")
    return step


def collect_finite_training_losses(train_result: Any, trainer: Any) -> list[float]:
    candidates: list[Any] = []
    metrics = getattr(train_result, "metrics", None)
    if isinstance(metrics, Mapping) and "train_loss" in metrics:
        candidates.append(metrics["train_loss"])
    state = getattr(trainer, "state", None)
    history = getattr(state, "log_history", None)
    if isinstance(history, list):
        for row in history:
            if isinstance(row, Mapping) and "loss" in row:
                candidates.append(row["loss"])
    losses: list[float] = []
    for index, value in enumerate(candidates):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"formal SFT loss {index} is not numeric")
        loss = float(value)
        if not math.isfinite(loss):
            raise RuntimeError(f"formal SFT loss {index} is non-finite")
        losses.append(loss)
    if not losses:
        raise RuntimeError("formal SFT produced no finite loss metric")
    return losses


def completed_step_counts(
    trainer: Any,
    tracker: OptimizerEvidenceTracker,
    *,
    starting_global_step: int,
) -> tuple[int, int]:
    state = getattr(trainer, "state", None)
    global_step = getattr(state, "global_step", None)
    max_steps = getattr(state, "max_steps", None)
    if type(global_step) is not int or type(max_steps) is not int:
        raise RuntimeError("trainer did not publish integer global_step/max_steps")
    actual = global_step - starting_global_step
    planned = max_steps - starting_global_step
    if planned <= 0 or actual != planned:
        raise RuntimeError(
            f"formal SFT did not complete planned fresh steps: actual={actual}, planned={planned}"
        )
    if tracker.optimizer_steps != actual:
        raise RuntimeError(
            "optimizer callback count differs from the trainer's fresh global steps"
        )
    if tracker.pre_optimizer_events != actual or tracker.skipped_optimizer_steps:
        raise RuntimeError("formal SFT includes an unverified or skipped optimizer step")
    if tracker.nonzero_finite_gradient_steps != actual:
        raise RuntimeError("not every fresh optimizer step had a nonzero finite gradient")
    return planned, actual


__all__ = [
    "OptimizerEvidenceTracker",
    "collect_finite_training_losses",
    "completed_step_counts",
    "resume_starting_global_step",
]
