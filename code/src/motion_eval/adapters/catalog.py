"""Framework-free command adapters for all canonical models.

The controller imports only this descriptor layer.  Heavy model frameworks
are loaded, if at all, by the isolated runner subprocess named in each spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from motion_eval.runtime import CommandSpec

if TYPE_CHECKING:
    from motion_eval.controller.registry import CanonicalRegistry


_ADAPTER_ROLES = ("finetune", "evaluation", "verifier")


@dataclass(frozen=True)
class FrozenAdapterSpec:
    """Single immutable source for controller and runner-facade identities.

    Backend paths are relative to the controller root.  ``None`` means that
    the role is deliberately unavailable and must fail closed; the controller
    still freezes a deterministic missing-path marker for blocker evidence.
    """

    model_id: str
    modality: str
    evaluation_mode: str
    initialization: str
    finetune_runner: str
    evaluation_runner: str
    verifier_runner: str = "scripts/verify_artifact_reload.py"
    finetune_backend: str | None = None
    evaluation_backend: str | None = None
    verifier_backend: str | None = None
    dependencies: tuple[str, ...] = ()
    official_finetune: bool = False

    def _value_for_role(self, role: str, suffix: str) -> str | None:
        if role not in _ADAPTER_ROLES:
            raise ValueError(f"unknown adapter role: {role}")
        return getattr(self, f"{role}_{suffix}")

    def runner_for(self, role: str) -> str:
        value = self._value_for_role(role, "runner")
        if not isinstance(value, str):  # pragma: no cover - frozen constants guard this
            raise RuntimeError(f"{self.model_id}.{role} runner contract is invalid")
        return value

    def implemented_backend_for(self, role: str) -> str | None:
        return self._value_for_role(role, "backend")

    def backend_receipt_path_for(self, role: str) -> str:
        implemented = self.implemented_backend_for(role)
        if implemented is not None:
            return implemented
        return f"scripts/backends/missing/{self.model_id}/{role}.py"

    def backend_import_for(self, role: str) -> str | None:
        """Translate the frozen controller-relative file into its import name."""

        implemented = self.implemented_backend_for(role)
        if implemented is None:
            return None
        path = PurePosixPath(implemented)
        if (
            path.is_absolute()
            or path.suffix != ".py"
            or path.parts[:2] != ("scripts", "backends")
            or ".." in path.parts
        ):
            raise RuntimeError(
                f"{self.model_id}.{role} backend contract is not a safe scripts/backend file"
            )
        return ".".join(path.with_suffix("").parts[1:])


def _freeze_adapter_specs(
    specs: tuple[FrozenAdapterSpec, ...],
) -> Mapping[str, FrozenAdapterSpec]:
    result = {spec.model_id: spec for spec in specs}
    if len(result) != len(specs):
        raise RuntimeError("frozen adapter contract repeats a model_id")
    for model_id, spec in result.items():
        if (
            not model_id
            or spec.modality not in {"V", "M", "VM"}
            or spec.evaluation_mode
            not in {"generative", "discriminative_abcd_scores"}
            or spec.initialization not in {"pretrained", "random"}
            or not spec.finetune_runner.startswith("scripts/")
            or not spec.evaluation_runner.startswith("scripts/")
            or not spec.verifier_runner.startswith("scripts/")
        ):
            raise RuntimeError(f"invalid frozen adapter contract for {model_id}")
        for role in _ADAPTER_ROLES:
            spec.backend_import_for(role)
    return MappingProxyType(result)


FROZEN_ADAPTER_SPECS: Mapping[str, FrozenAdapterSpec] = _freeze_adapter_specs(
    (
        FrozenAdapterSpec(
            "qwen36_27b_lora", "V", "generative", "pretrained",
            "scripts/finetune_qwen36_27b_lora.py", "scripts/eval_qwen36_27b_generate.py",
        ),
        FrozenAdapterSpec(
            "motionr1_vm_lora", "VM", "generative", "pretrained",
            "scripts/finetune_motionr1_vm_lora.py", "scripts/eval_motionr1_vm_generate.py",
        ),
        FrozenAdapterSpec(
            "qwen3vl_8b_lora", "V", "generative", "pretrained",
            "scripts/finetune_qwen3vl_lora.py", "scripts/eval_qwen3vl_generate.py",
        ),
        FrozenAdapterSpec(
            "qwen3vl_4b_lora", "V", "generative", "pretrained",
            "scripts/finetune_qwen3vl_lora.py", "scripts/eval_qwen3vl_generate.py",
        ),
        FrozenAdapterSpec(
            "qwen35_4b_lora", "V", "generative", "pretrained",
            "scripts/finetune_qwen35_lora.py", "scripts/eval_qwen35_generate.py",
        ),
        FrozenAdapterSpec(
            "videollava_7b_lora", "V", "generative", "pretrained",
            "scripts/finetune_videollava_lora.py", "scripts/eval_videollava_generate.py",
        ),
        FrozenAdapterSpec(
            "videochatgpt_lora", "V", "generative", "pretrained",
            "scripts/finetune_videochatgpt_lora.py", "scripts/eval_videochatgpt_generate.py",
        ),
        FrozenAdapterSpec(
            "videochat2_lora", "V", "generative", "pretrained",
            "scripts/finetune_videochat2_lora.py", "scripts/eval_videochat2_generate.py",
        ),
        FrozenAdapterSpec(
            "videollama_trainables", "V", "generative", "pretrained",
            "scripts/finetune_videollama_trainables.py", "scripts/eval_videollama_generate.py",
        ),
        FrozenAdapterSpec(
            "videollama_lora", "V", "generative", "pretrained",
            "scripts/finetune_videollama_lora.py", "scripts/eval_videollama_generate.py",
            finetune_backend="scripts/backends/finetune_videollama_lora.py",
            verifier_backend="scripts/backends/finetune_videollama_lora.py",
            dependencies=("torch", "yaml", "peft", "transformers", "decord", "cv2"),
        ),
        FrozenAdapterSpec(
            "mplug_owl_video_lora", "V", "generative", "pretrained",
            "scripts/finetune_mplug_owl_video_lora.py", "scripts/eval_mplug_owl_generate.py",
        ),
        FrozenAdapterSpec(
            "otter_video_lora", "V", "generative", "pretrained",
            "scripts/finetune_otter_video_lora.py", "scripts/eval_otter_generate.py",
        ),
        FrozenAdapterSpec(
            "agcn_official", "M", "discriminative_abcd_scores", "random",
            "scripts/finetune_agcn_official.py", "scripts/eval_agcn_official.py",
            official_finetune=True,
        ),
        FrozenAdapterSpec(
            "motionclip_official", "M", "discriminative_abcd_scores", "pretrained",
            "scripts/finetune_motionclip_official.py", "scripts/eval_motionclip_official.py",
            official_finetune=True,
        ),
        FrozenAdapterSpec(
            "motionllm_official", "V", "generative", "pretrained",
            "scripts/finetune_motionllm_lora.py", "scripts/eval_motionllm_generate.py",
            finetune_backend="scripts/backends/finetune_motionllm_lora.py",
            verifier_backend="scripts/backends/finetune_motionllm_lora.py",
            dependencies=("torch", "torchvision", "transformers", "pytorchvideo", "einops"),
        ),
    )
)


@dataclass(frozen=True)
class AdapterContext:
    batch_id: str
    python_executable: str
    controller_root: str
    batch_root: str
    train_manifest: str
    validation_manifest: str
    benchmark_manifest: str
    media_manifest: str
    media_manifest_sha256: str
    leakage_audit: str
    pretrained_root: str
    output_path: str
    artifact_path: str | None = None
    artifact_digest: str | None = None
    attempt_id: str | None = None
    limit: int | None = None
    purpose: str = "production"
    training_steps: int | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class AdapterDescriptor:
    model_id: str
    modality: str
    evaluation_mode: str
    finetune_runner: str
    evaluation_runner: str
    verifier_runner: str = "scripts/verify_artifact_reload.py"
    finetune_backend: str = "scripts/backends/missing/finetune.py"
    evaluation_backend: str = "scripts/backends/missing/evaluation.py"
    verifier_backend: str = "scripts/backends/missing/verifier.py"
    official_finetune: bool = False
    initialization: str = "pretrained"

    def backend_for(self, role: str) -> str:
        """Return the implementation file behind a catalog-facing runner.

        Runner facades are intentionally lightweight and uniform.  This second
        path prevents a facade that merely parses arguments from being mistaken
        for an implemented model integration.
        """

        try:
            return {
                "finetune": self.finetune_backend,
                "evaluation": self.evaluation_backend,
                "verifier": self.verifier_backend,
            }[role]
        except KeyError as exc:
            raise ValueError(f"unknown adapter role: {role}") from exc

    def _base(self, context: AdapterContext, runner: str) -> list[str]:
        if context.controller_root.startswith("/"):
            runner_path = str(PurePosixPath(context.controller_root) / runner)
        else:
            runner_path = str(
                (Path(context.controller_root) / Path(runner)).resolve(strict=False)
            )
        return [context.python_executable, runner_path]

    def finetune_spec(self, context: AdapterContext) -> CommandSpec:
        if context.purpose not in {"production", "preflight"}:
            raise ValueError("finetune purpose must be production or preflight")
        if type(context.training_steps) is not int or context.training_steps <= 0:
            raise ValueError("finetune training_steps must be a positive integer")
        if context.purpose == "production" and context.limit is not None:
            raise ValueError("production finetune cannot use a sample limit")
        if context.purpose == "preflight" and (
            type(context.limit) is not int or context.limit <= 0
        ):
            raise ValueError("preflight finetune requires a positive sample limit")
        argv = self._base(context, self.finetune_runner)
        argv.extend(
            [
                "--batch-id", context.batch_id,
                "--model-id", self.model_id,
                "--train-manifest", context.train_manifest,
                "--validation-manifest", context.validation_manifest,
                "--leakage-audit", context.leakage_audit,
                "--pretrained-root", context.pretrained_root,
                "--output-dir", context.output_path,
                "--modality", self.modality,
                "--initialization", self.initialization,
                "--purpose", context.purpose,
                "--training-steps", str(context.training_steps),
            ]
        )
        if context.limit is not None:
            argv.extend(["--limit", str(context.limit)])
        return CommandSpec(
            argv=tuple(argv),
            cwd=context.controller_root,
            env={} if context.env is None else context.env,
            timeout_seconds=172800,
            label=f"finetune:{self.model_id}",
        )

    def evaluation_spec(self, context: AdapterContext) -> CommandSpec:
        if not context.artifact_path:
            raise ValueError("current-batch artifact_path is required for evaluation")
        argv = self._base(context, self.evaluation_runner)
        argv.extend(
            [
                "--batch-id", context.batch_id,
                "--model-id", self.model_id,
                "--benchmark-manifest", context.benchmark_manifest,
                "--media-manifest", context.media_manifest,
                "--media-manifest-sha256", context.media_manifest_sha256,
                "--artifact", context.artifact_path,
                "--predictions", context.output_path,
                "--modality", self.modality,
                "--evaluation-mode", self.evaluation_mode,
            ]
        )
        if self.evaluation_mode == "generative":
            argv.extend(["--do-sample", "false", "--temperature", "0", "--strict-answer-tags"])
        else:
            argv.extend(["--score-order", "A,B,C,D"])
        if context.limit is not None:
            argv.extend(["--limit", str(context.limit)])
        return CommandSpec(
            argv=tuple(argv),
            cwd=context.controller_root,
            env={} if context.env is None else context.env,
            timeout_seconds=86400,
            label=f"evaluate:{self.model_id}",
        )

    def verification_spec(self, context: AdapterContext) -> CommandSpec:
        """Build the catalog-owned, controller-launched artifact verifier."""

        if not context.artifact_path or not context.artifact_digest or not context.attempt_id:
            raise ValueError("verification requires attempt, artifact path, and artifact digest")
        argv = self._base(context, self.verifier_runner)
        argv.extend(
            [
                "--batch-id", context.batch_id,
                "--model-id", self.model_id,
                "--attempt-id", context.attempt_id,
                "--artifact", context.artifact_path,
                "--artifact-sha256", context.artifact_digest,
                "--report", context.output_path,
            ]
        )
        return CommandSpec(
            argv=tuple(argv),
            cwd=context.controller_root,
            env={} if context.env is None else context.env,
            timeout_seconds=3600,
            label=f"verify:{self.model_id}",
        )


def build_adapter_catalog(registry: "CanonicalRegistry") -> Mapping[str, AdapterDescriptor]:
    if tuple(registry.ids) != tuple(FROZEN_ADAPTER_SPECS):
        raise ValueError("adapter catalog does not exactly cover the canonical 15 models")
    result: dict[str, AdapterDescriptor] = {}
    for model in registry.models:
        frozen = FROZEN_ADAPTER_SPECS[model.model_id]
        if (
            model.modality.value != frozen.modality
            or model.evaluation_mode != frozen.evaluation_mode
        ):
            raise ValueError(
                f"registry/catalog contract drift for {model.model_id}: "
                "modality or evaluation mode differs"
            )
        result[model.model_id] = AdapterDescriptor(
            model_id=model.model_id,
            modality=frozen.modality,
            evaluation_mode=frozen.evaluation_mode,
            finetune_runner=frozen.finetune_runner,
            evaluation_runner=frozen.evaluation_runner,
            verifier_runner=frozen.verifier_runner,
            finetune_backend=frozen.backend_receipt_path_for("finetune"),
            evaluation_backend=frozen.backend_receipt_path_for("evaluation"),
            verifier_backend=frozen.backend_receipt_path_for("verifier"),
            official_finetune=frozen.official_finetune,
            initialization=frozen.initialization,
        )
    return MappingProxyType(result)
