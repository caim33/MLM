"""Frozen ms-swift reward and training-evidence registrations.

The reward implementations remain framework independent.  The one callback in
this module records evidence that the pinned Trainer actually observed finite
losses and gradients and changed trainable tensors.  The parent runner binds
that receipt to a fresh, unpredictable nonce before it can publish an
artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from motionllm.grpo import (  # noqa: E402
    GroupBonusConfig,
    format_rewards,
    option_accuracy_rewards,
    semantic_rewards,
    vm_v_group_bonus_rewards,
)
from motionllm.grpo.checkpoint_binding import (  # noqa: E402
    ADDITIONAL_CONFIG_NAME,
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    adapter_config_critical_fields,
    bind_live_peft_adapter_to_checkpoint,
    reject_symlink_components,
)
from motionllm.grpo.rubric_adapter import (  # noqa: E402
    motion_rubric_v2_rewards,
    qa_rubric_rewards,
)
from motionllm.grpo.rubric_online import OnlineJudgeConfig, OnlineRubricJudge  # noqa: E402
from swift.callbacks import TrainerCallback, callbacks_map  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402


_TRAINING_RECEIPT_CALLBACK = "motion_training_receipt"
_TRAINING_RECEIPT_SCHEMA = "motionllm.grpo.training_update.v2"
_TRAINING_NONCE_ENV = "MOTION_GRPO_TRAINING_NONCE"
_TRAINING_BATCH_ENV = "MOTION_GRPO_BATCH_ID"
_TRAINING_STEPS_ENV = "MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS"
_TRAINING_ARTIFACT_ENV = "MOTION_GRPO_ARTIFACT_PATH"
_FORMAL_MODEL_REGISTRY_ID = "motionr1_vm_lora"
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _world_process_zero(state: Any) -> bool:
    return bool(getattr(state, "is_world_process_zero", True))


def _required_training_env(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"formal GRPO callback requires {name}")
    return value


def _callback_model(callback: "MotionTrainingReceiptCallback", kwargs: Mapping[str, Any]) -> Any:
    model = kwargs.get("model")
    if model is None:
        model = getattr(callback.trainer, "model", None)
    if model is None or not callable(getattr(model, "named_parameters", None)):
        raise RuntimeError("formal GRPO callback received no trainable model")
    return model


def _callback_peft_model(
    callback: "MotionTrainingReceiptCallback", kwargs: Mapping[str, Any]
) -> Any:
    """Return only an explicit PEFT model or the pinned Accelerator unwrap."""

    from peft import PeftModel

    model = _callback_model(callback, kwargs)
    if isinstance(model, PeftModel):
        return model
    accelerator = getattr(callback.trainer, "accelerator", None)
    unwrap = getattr(accelerator, "unwrap_model", None)
    if not callable(unwrap):
        raise RuntimeError(
            "formal GRPO callback cannot unwrap the live model through pinned Accelerator"
        )
    try:
        unwrapped = unwrap(
            model,
            keep_fp32_wrapper=True,
            keep_torch_compile=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "formal GRPO callback failed to unwrap the live model through pinned Accelerator"
        ) from exc
    if not isinstance(unwrapped, PeftModel):
        raise RuntimeError(
            "formal GRPO callback live model is not an explicit PeftModel after unwrap"
        )
    return unwrapped


def _trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    values = sorted(
        ((str(name), parameter) for name, parameter in model.named_parameters() if parameter.requires_grad),
        key=lambda item: item[0],
    )
    if not values:
        raise RuntimeError("formal GRPO callback found no trainable tensors")
    if len({name for name, _ in values}) != len(values):
        raise RuntimeError("formal GRPO callback found duplicate trainable tensor names")
    return values


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().cpu().contiguous()
    if not torch.is_floating_point(value):
        raise RuntimeError("formal GRPO trainable tensors must use floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("formal GRPO trainable tensor contains a non-finite value")
    return value.view(torch.uint8).numpy().tobytes()


def _state_sha256(values: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = _tensor_bytes(tensor)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _reject_symlink_ancestors(path: Path) -> None:
    reject_symlink_components(path)


class MotionTrainingReceiptCallback(TrainerCallback):
    """Produce nonce-bound, fail-closed evidence of real optimizer updates."""

    def __init__(self, args: Any, trainer: Any):
        super().__init__(args, trainer)
        self._active = False
        self._expected_steps = 0
        self._batch_id = ""
        self._nonce = ""
        self._artifact_path: Path | None = None
        self._initial: dict[str, Any] = {}
        self._initial_hash = ""
        self._optimizer_step_count = 0
        self._completed_step_count = 0
        self._last_completed_step = 0
        self._finite_loss_count = 0
        self._gradient_observation_count = 0
        self._max_abs_gradient = 0.0
        self._pending_gradient_max: float | None = None
        self._optimizer_committed_for_step = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del control
        if not _world_process_zero(state):
            return
        if self._active:
            raise RuntimeError("formal GRPO callback cannot begin training twice")
        nonce = _required_training_env(_TRAINING_NONCE_ENV)
        if _LOWER_SHA256.fullmatch(nonce) is None:
            raise RuntimeError("formal GRPO callback nonce must be 64 lowercase hex characters")
        batch_id = _required_training_env(_TRAINING_BATCH_ENV)
        if _BATCH_ID.fullmatch(batch_id) is None:
            raise RuntimeError("formal GRPO callback batch id is invalid")
        expected_raw = _required_training_env(_TRAINING_STEPS_ENV)
        if not expected_raw.isascii() or not expected_raw.isdecimal() or expected_raw.startswith("0"):
            raise RuntimeError("formal GRPO callback expected steps must be canonical positive decimal")
        expected_steps = int(expected_raw)
        if expected_steps <= 0 or str(expected_steps) != expected_raw:
            raise RuntimeError("formal GRPO callback expected steps must be canonical positive decimal")
        artifact = Path(_required_training_env(_TRAINING_ARTIFACT_ENV))
        if not artifact.is_absolute() or artifact.name != f"checkpoint-{expected_steps}":
            raise RuntimeError("formal GRPO callback artifact path must be the exact checkpoint leaf")
        output = Path(str(getattr(args, "output_dir", "")))
        if not output.is_absolute() or artifact.parent != output:
            raise RuntimeError("formal GRPO callback artifact path differs from Trainer output_dir")
        if getattr(args, "max_steps", None) != expected_steps:
            raise RuntimeError("formal GRPO callback max_steps differs from the runner binding")
        if getattr(state, "global_step", None) != 0 or getattr(state, "max_steps", None) != expected_steps:
            raise RuntimeError("formal GRPO callback forbids resumed or differently-sized training")
        if str(getattr(args, "tuner_type", "")).strip().casefold() not in {"lora", "peft"}:
            raise RuntimeError("formal GRPO callback only supports the frozen LoRA/PEFT mode")
        if bool(getattr(self.trainer, "is_deepspeed_enabled", False)) or bool(
            getattr(self.trainer, "is_fsdp_enabled", False)
        ):
            raise RuntimeError(
                "formal GRPO callback requires replicated trainable parameters; DeepSpeed/FSDP is unsupported"
            )
        _reject_symlink_ancestors(output)
        model = _callback_model(self, kwargs)
        parameters = _trainable_parameters(model)
        self._initial = {name: parameter.detach().cpu().clone() for name, parameter in parameters}
        self._initial_hash = _state_sha256([(name, self._initial[name]) for name in sorted(self._initial)])
        self._expected_steps = expected_steps
        self._batch_id = batch_id
        self._nonce = nonce
        self._artifact_path = artifact
        self._pending_gradient_max = None
        self._optimizer_committed_for_step = False
        self._active = True

    def on_pre_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        del args, state, control
        if not self._active:
            return
        if self._pending_gradient_max is not None or self._optimizer_committed_for_step:
            raise RuntimeError(
                "formal GRPO callback received duplicate or out-of-order pre-optimizer event"
            )
        model = _callback_model(self, kwargs)
        import torch

        observed = False
        local_max = 0.0
        for _, parameter in _trainable_parameters(model):
            gradient = parameter.grad
            if gradient is None:
                continue
            value = gradient.detach()
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError("formal GRPO callback observed a non-finite gradient")
            if value.numel() == 0:
                continue
            observed = True
            local_max = max(local_max, float(value.abs().max().item()))
        if not observed:
            raise RuntimeError(
                "formal GRPO callback observed no gradient before optimizer step"
            )
        if not math.isfinite(local_max) or local_max <= 0:
            raise RuntimeError(
                "formal GRPO callback observed only zero gradients before optimizer step"
            )
        self._pending_gradient_max = local_max

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        if not self._active:
            return
        if self._optimizer_committed_for_step:
            raise RuntimeError("formal GRPO callback received a duplicate optimizer event")
        if self._pending_gradient_max is None:
            raise RuntimeError(
                "formal GRPO callback optimizer event has no pending gradient proof"
            )
        accelerator = getattr(self.trainer, "accelerator", None)
        missing = object()
        skipped = getattr(accelerator, "optimizer_step_was_skipped", missing)
        if skipped is missing or type(skipped) is not bool:
            raise RuntimeError(
                "formal GRPO callback requires boolean accelerator.optimizer_step_was_skipped"
            )
        if skipped:
            raise RuntimeError(
                "formal GRPO callback rejects a skipped AMP optimizer step"
            )
        self._optimizer_step_count += 1
        self._gradient_observation_count += 1
        self._max_abs_gradient = max(
            self._max_abs_gradient, self._pending_gradient_max
        )
        self._pending_gradient_max = None
        self._optimizer_committed_for_step = True

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if not self._active:
            return
        if self._pending_gradient_max is not None or not self._optimizer_committed_for_step:
            raise RuntimeError(
                "formal GRPO callback step ended without one successful optimizer event"
            )
        step = getattr(state, "global_step", None)
        expected_next = self._completed_step_count + 1
        if step != expected_next or step > self._expected_steps:
            raise RuntimeError("formal GRPO callback observed non-sequential optimizer steps")
        self._completed_step_count += 1
        self._last_completed_step = step
        if (
            self._optimizer_step_count != self._completed_step_count
            or self._gradient_observation_count != self._completed_step_count
        ):
            raise RuntimeError(
                "formal GRPO callback optimizer/gradient event counts diverged"
            )
        self._optimizer_committed_for_step = False

    def on_log(
        self, args: Any, state: Any, control: Any, logs: Optional[Mapping[str, Any]] = None, **kwargs: Any
    ) -> None:
        del args, control, kwargs
        if not self._active or getattr(state, "global_step", 0) <= 0 or not isinstance(logs, Mapping):
            return
        observed = False
        for key in ("loss", "train_loss"):
            value = logs.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError("formal GRPO callback observed a non-finite loss")
            observed = True
        if observed:
            self._finite_loss_count += 1

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control
        if not self._active or not _world_process_zero(state):
            return
        if getattr(state, "global_step", None) != self._expected_steps:
            return
        if self._optimizer_step_count != self._expected_steps:
            raise RuntimeError("formal GRPO callback optimizer event count differs from expected steps")
        if self._pending_gradient_max is not None or self._optimizer_committed_for_step:
            raise RuntimeError(
                "formal GRPO callback final save has an incomplete optimizer event"
            )
        if (
            self._completed_step_count != self._expected_steps
            or self._last_completed_step != self._expected_steps
        ):
            raise RuntimeError("formal GRPO callback completed-step sequence is incomplete")
        if self._finite_loss_count != self._expected_steps:
            raise RuntimeError(
                "formal GRPO callback lacks one finite loss proof per optimizer step"
            )
        if (
            self._gradient_observation_count != self._expected_steps
            or not math.isfinite(self._max_abs_gradient)
            or self._max_abs_gradient <= 0
        ):
            raise RuntimeError(
                "formal GRPO callback lacks one finite positive gradient proof per optimizer step"
            )
        model = _callback_model(self, kwargs)
        final_parameters = _trainable_parameters(model)
        final = {name: parameter.detach().cpu().clone() for name, parameter in final_parameters}
        if set(final) != set(self._initial):
            raise RuntimeError("formal GRPO trainable tensor set changed during training")
        import torch

        changed = 0
        max_abs_delta = 0.0
        for name in sorted(self._initial):
            before = self._initial[name]
            after = final[name]
            if before.dtype != after.dtype or before.shape != after.shape:
                raise RuntimeError("formal GRPO trainable tensor metadata changed during training")
            if not bool(torch.isfinite(after).all().item()):
                raise RuntimeError("formal GRPO final trainable tensor is non-finite")
            if not torch.equal(before, after):
                changed += 1
            if before.numel():
                delta = float((after.to(torch.float64) - before.to(torch.float64)).abs().max().item())
                max_abs_delta = max(max_abs_delta, delta)
        final_hash = _state_sha256([(name, final[name]) for name in sorted(final)])
        if changed <= 0 or final_hash == self._initial_hash:
            raise RuntimeError("formal GRPO callback proves no trainable tensor change")
        if not math.isfinite(max_abs_delta) or max_abs_delta <= 0:
            raise RuntimeError("formal GRPO callback proves no finite positive parameter delta")
        artifact = self._artifact_path
        if artifact is None or not artifact.is_dir():
            raise RuntimeError("formal GRPO final checkpoint does not exist at callback save time")
        _reject_symlink_ancestors(artifact)
        output = Path(str(getattr(self.args, "output_dir", ""))).resolve(strict=True)
        if artifact.resolve(strict=True).parent != output:
            raise RuntimeError("formal GRPO final checkpoint escaped Trainer output_dir")
        peft_model = _callback_peft_model(self, kwargs)
        peft_trainable_parameters = _trainable_parameters(peft_model)
        if {
            id(parameter) for _, parameter in peft_trainable_parameters
        } != {id(parameter) for _, parameter in final_parameters}:
            raise RuntimeError(
                "formal GRPO callback wrapped and unwrapped trainable tensor sets differ"
            )
        disk_binding = bind_live_peft_adapter_to_checkpoint(
            peft_model,
            artifact,
            require_swift_extension=True,
        )
        if disk_binding.extension_config is None:  # pragma: no cover - binding invariant
            raise RuntimeError(
                "formal GRPO checkpoint lacks its bound Swift LoRA extension config"
            )
        frozen_embedding_tensor_count = disk_binding.tensor_count - len(
            peft_trainable_parameters
        )
        if frozen_embedding_tensor_count < 0:
            raise RuntimeError(
                "formal GRPO checkpoint has fewer saveable tensors than trainable tensors"
            )
        adapter_config_critical = adapter_config_critical_fields(
            disk_binding.config.semantics
        )
        receipt = artifact / "grpo_training_receipt.json"
        payload = {
            "schema": _TRAINING_RECEIPT_SCHEMA,
            "status": "optimizer_and_checkpoint_verified",
            "nonce": self._nonce,
            "batch_id": self._batch_id,
            "model_registry_id": _FORMAL_MODEL_REGISTRY_ID,
            "expected_optimizer_steps": self._expected_steps,
            "observed_global_step": getattr(state, "global_step", None),
            "optimizer_step_count": self._optimizer_step_count,
            "finite_loss_count": self._finite_loss_count,
            "gradient_observation_count": self._gradient_observation_count,
            "trainable_tensor_count": len(final),
            "changed_tensor_count": changed,
            "initial_trainable_state_sha256": self._initial_hash,
            "final_trainable_state_sha256": final_hash,
            "adapter_filename": ADAPTER_SAFE_WEIGHTS_NAME,
            "adapter_payload_sha256": disk_binding.payload_sha256,
            "adapter_payload_size_bytes": disk_binding.payload_size_bytes,
            "adapter_tensor_count": disk_binding.tensor_count,
            "frozen_embedding_tensor_count": frozen_embedding_tensor_count,
            "final_saveable_adapter_state_sha256": disk_binding.state_sha256,
            "adapter_config_filename": ADAPTER_CONFIG_NAME,
            "adapter_config_payload_sha256": disk_binding.config.payload_sha256,
            "adapter_config_payload_size_bytes": disk_binding.config.payload_size_bytes,
            "adapter_config_semantic_sha256": disk_binding.config.semantic_sha256,
            "adapter_config_critical": adapter_config_critical,
            "adapter_extension_filename": ADDITIONAL_CONFIG_NAME,
            "adapter_extension_payload_sha256": disk_binding.extension_config.payload_sha256,
            "adapter_extension_payload_size_bytes": disk_binding.extension_config.payload_size_bytes,
            "adapter_extension_semantic_sha256": disk_binding.extension_config.semantic_sha256,
            "adapter_extension_semantics": disk_binding.extension_config.semantics,
            "max_abs_gradient": self._max_abs_gradient,
            "max_abs_delta": max_abs_delta,
        }
        encoded = (json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(receipt, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _columns(
    *,
    sample_id: Any,
    group_id: Any,
    branch: Any,
    rollout_id: Any,
    request_id: Any,
    answer: Any,
    solution: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id if sample_id is not None else kwargs.get("sample_id"),
        "group_id": group_id if group_id is not None else kwargs.get("group_id"),
        "branch": branch if branch is not None else kwargs.get("branch"),
        "rollout_id": rollout_id if rollout_id is not None else kwargs.get("rollout_id"),
        "request_id": request_id if request_id is not None else kwargs.get("request_id"),
        "answer": answer if answer is not None else kwargs.get("gold_answer"),
        "solution": solution if solution is not None else kwargs.get("solution"),
        "num_generations": kwargs.get("num_generations"),
    }


def _strict_env_float(name: str, fallback: Any) -> float:
    raw = os.getenv(name)
    value = fallback if raw is None else raw
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _online_judge(prefix: str, args: Any) -> Optional[OnlineRubricJudge]:
    endpoint = os.getenv(f"{prefix}_URL") or getattr(
        args, f"{prefix.lower()}_url", None
    )
    if endpoint is None:
        return None
    timeout = _strict_env_float(
        f"{prefix}_TIMEOUT_SECONDS",
        getattr(args, f"{prefix.lower()}_timeout_seconds", 60.0),
    )
    # The token value is never rendered in repr/errors/manifests by this
    # adapter. It remains process-scoped and is sent only as an HTTP header.
    token = os.getenv(f"{prefix}_TOKEN")
    return OnlineRubricJudge(
        OnlineJudgeConfig(
            endpoint=str(endpoint),
            timeout_seconds=timeout,
            bearer_token=token,
        )
    )


class MotionSemanticORM(ORM):
    def __call__(self, completions: Sequence[Any], answer=None, solution=None, **kwargs) -> List[float]:
        return semantic_rewards(
            list(completions),
            **_columns(answer=answer, solution=solution, kwargs=kwargs, **{
                name: kwargs.get(name) for name in ("sample_id", "group_id", "branch", "rollout_id", "request_id")
            }),
        )


class MotionFormatORM(ORM):
    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        return format_rewards(
            list(completions),
            **_columns(
                answer=kwargs.get("answer"),
                solution=kwargs.get("solution"),
                kwargs=kwargs,
                **{
                    name: kwargs.get(name)
                    for name in ("sample_id", "group_id", "branch", "rollout_id", "request_id")
                },
            ),
        )


class MotionOptionAccuracyORM(ORM):
    def __call__(self, completions: Sequence[Any], answer=None, solution=None, **kwargs) -> List[float]:
        return option_accuracy_rewards(
            list(completions),
            **_columns(answer=answer, solution=solution, kwargs=kwargs, **{
                name: kwargs.get(name) for name in ("sample_id", "group_id", "branch", "rollout_id", "request_id")
            }),
        )


class MotionVMVGroupBonusORM(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.config = GroupBonusConfig(
            threshold=_strict_env_float(
                "MOTION_GRPO_VM_V_THRESHOLD",
                getattr(args, "motion_grpo_vm_v_threshold", 1.0),
            ),
            bonus_value=_strict_env_float(
                "MOTION_GRPO_VM_V_BONUS",
                getattr(args, "motion_grpo_vm_v_bonus", 0.1),
            ),
            qualify_threshold=_strict_env_float(
                "MOTION_GRPO_VM_V_QUALIFY_THRESHOLD",
                getattr(args, "motion_grpo_vm_v_qualify_threshold", 0.1),
            ),
        )

    def __call__(self, completions: Sequence[Any], answer=None, solution=None, **kwargs) -> List[float]:
        return vm_v_group_bonus_rewards(
            list(completions),
            config=self.config,
            **_columns(answer=answer, solution=solution, kwargs=kwargs, **{
                name: kwargs.get(name) for name in ("sample_id", "group_id", "branch", "rollout_id", "request_id")
            }),
        )


class QAMCRubricORM(ORM):
    """QA Rubric ORM with strict precomputed or online judge input."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.judge = _online_judge("MOTION_GRPO_QA_RUBRIC_JUDGE", args)

    def __call__(self, completions: Sequence[Any], answer=None, solution=None, **kwargs) -> List[float]:
        return qa_rubric_rewards(
            list(completions),
            qa_rubric_criteria=kwargs.get("qa_rubric_criteria"),
            qa_rubric_judgment=kwargs.get("qa_rubric_judgment"),
            judge_client=self.judge,
            sample_id=kwargs.get("sample_id"),
            group_id=kwargs.get("group_id"),
            branch=kwargs.get("branch"),
            rollout_id=kwargs.get("rollout_id"),
            answer=answer if answer is not None else kwargs.get("gold_answer"),
            solution=solution,
            request_id=kwargs.get("request_id"),
            num_generations=kwargs.get("num_generations"),
        )


class MotionRubricV2ORM(ORM):
    """Motion-description V2 ORM; Stage 1 artifacts are rejected by mode."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.judge = _online_judge("MOTION_GRPO_MOTION_RUBRIC_V2_JUDGE", args)

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        return motion_rubric_v2_rewards(
            list(completions),
            motion_rubric_v2_criteria=kwargs.get("motion_rubric_v2_criteria"),
            motion_rubric_v2_id=kwargs.get("motion_rubric_v2_id"),
            motion_rubric_v2_judgment=kwargs.get("motion_rubric_v2_judgment"),
            judge_client=self.judge,
            sample_id=kwargs.get("sample_id"),
            num_generations=kwargs.get("num_generations"),
        )


orms["motion_semantic"] = MotionSemanticORM
orms["motion_option_accuracy"] = MotionOptionAccuracyORM
orms["motion_format"] = MotionFormatORM
orms["motion_vm_v_bonus"] = MotionVMVGroupBonusORM
orms["qa_mc_rubric"] = QAMCRubricORM
orms["motion_rubric_v2"] = MotionRubricV2ORM

if _TRAINING_RECEIPT_CALLBACK in callbacks_map:
    raise RuntimeError(
        f"ms-swift callback name {_TRAINING_RECEIPT_CALLBACK!r} is already registered"
    )
callbacks_map[_TRAINING_RECEIPT_CALLBACK] = MotionTrainingReceiptCallback
