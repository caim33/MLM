"""High-level batch controller enforcing fresh-finetune and evaluation gates."""

from __future__ import annotations

import os
import re
import sys
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from motion_eval.adapters import AdapterContext, build_adapter_catalog
from motion_eval.contracts import (
    EvaluationErrorCode,
    InputModality,
    PredictionRow,
    prediction_row_from_dict,
)
from motion_eval.core import (
    atomic_write_json,
    hash_path,
    resolve_within_root,
    sha256_file,
    sha256_bytes,
    sha256_json,
)
from motion_eval.data import (
    BatchReceiptError,
    create_batch_receipt,
    load_and_validate_batch_receipt,
    load_benchmark,
    load_json_strict,
    load_jsonl_strict,
    smoke_items,
)
from motion_eval.data.receipts import validate_batch_id
from motion_eval.reporting import build_release_files
from motion_eval.training_receipt import load_and_validate_training_receipt
from motion_eval.runtime import (
    CommandSpec,
    GPUDevice,
    GPULeaseStore,
    KeepaliveStore,
    NvidiaSmiProbe,
    run_verified_python,
)

from .registry import CanonicalRegistry, ModelSpec, load_canonical_registry
from .state import EventStore, StateError, ensure_event_trust

_SAFE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SECRET_TEXT = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]",
    re.IGNORECASE,
)
_BLOCK_REASONS = frozenset(
    {
        "missing_path",
        "missing_code",
        "missing_weight",
        "unrecoverable_provenance",
    }
)
_MISSING_REASONS = frozenset({"missing_path", "missing_code", "missing_weight"})
_VERIFIED_MULTI_ROOT_BOOTSTRAP_BLOCKER = (
    "blocker=verified-multi-root-bootstrap: production Python execution is "
    "fail-closed until the controller verifies an isolated -I -S -B bootstrap "
    "covering controller code, catalog runner code, training code, training "
    "runner code, and the runtime environment; -I alone can execute system "
    "sitecustomize before the current stdin bootstrap"
)


class ControllerValidationError(RuntimeError):
    pass


def _safe_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ControllerValidationError(f"{name} is not a safe identifier")
    return value


def _evidence_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "evidence_sha256"}


def _attempt_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "attempt_sha256"}


def _absolute_path_preserving_links(value: str | Path) -> Path:
    """Return an absolute lexical path without dereferencing a venv launcher."""

    return Path(os.path.abspath(os.fspath(value)))


def _same_lexical_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


class BatchController:
    """One controller for all fifteen registry models.

    Workers remain isolated subprocesses.  This class validates only immutable
    receipts, command specifications, artifacts, predictions, and state gates.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        registry_path: str | Path | None = None,
        pretrained_registry_path: str | Path | None = None,
        code_root: str | Path | None = None,
        runner_root: str | Path | None = None,
        pretrained_root: str | Path | None = None,
        controller_interpreter: str | Path | None = None,
        keepalive_root: str | Path | None = None,
        keepalive_owner: str = "motionllm",
        gpu_probe: NvidiaSmiProbe | None = None,
    ) -> None:
        repository = Path(__file__).resolve().parents[3]
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = self.workspace_root.resolve(strict=True)
        registry_path = registry_path or repository / "model_evaluation_agent" / "model_registry.json"
        pretrained_registry_path = (
            pretrained_registry_path
            or repository / "model_evaluation_agent" / "pretrained_registry.json"
        )
        self.code_root = Path(code_root or repository / "src" / "motion_eval").resolve(strict=True)
        self.registry: CanonicalRegistry = load_canonical_registry(
            registry_path, pretrained_registry_path
        )
        self.adapters = build_adapter_catalog(self.registry)
        self.runner_root = str(
            Path(runner_root).resolve(strict=False)
            if runner_root is not None
            else self.registry.registry_path.parent.resolve(strict=True)
        )
        self.pretrained_root = (
            str(Path(pretrained_root).resolve(strict=False))
            if pretrained_root is not None
            else self.registry.pretrained_root
        )
        interpreter = _absolute_path_preserving_links(
            controller_interpreter or sys.executable
        )
        try:
            interpreter_target = interpreter.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ControllerValidationError(
                "controller interpreter must resolve to an existing file"
            ) from exc
        if not interpreter.is_file() or not interpreter_target.is_file():
            raise ControllerValidationError("controller interpreter must be an existing file")
        self.controller_interpreter = str(interpreter)
        self.keepalive_root = Path(
            keepalive_root or (self.workspace_root / ".keepalive")
        ).resolve(strict=False)
        self.keepalive_root.mkdir(parents=True, exist_ok=True)
        self.keepalive_root = self.keepalive_root.resolve(strict=True)
        if not isinstance(keepalive_owner, str) or not keepalive_owner.strip():
            raise ControllerValidationError("keepalive_owner must be non-empty")
        self.keepalive_owner = keepalive_owner
        self.gpu_probe = gpu_probe or NvidiaSmiProbe()
        self.event_trust = ensure_event_trust(self.workspace_root)

    def batch_root(self, batch_id: str, *, must_exist: bool = True) -> Path:
        validate_batch_id(batch_id)
        candidate = resolve_within_root(
            self.workspace_root / batch_id,
            self.workspace_root,
            must_exist=must_exist,
        )
        if must_exist and not candidate.is_dir():
            raise ControllerValidationError(f"batch is not a directory: {batch_id}")
        return candidate

    def _runtime_contract(self) -> dict[str, Any]:
        """Freeze the executable, catalog runners, and complete typed command templates."""

        interpreter_launcher = Path(self.controller_interpreter)
        try:
            interpreter_path = interpreter_launcher.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ControllerValidationError(
                "controller interpreter launcher no longer resolves to an existing file"
            ) from exc
        interpreter = {
            "launcher_path": str(interpreter_launcher),
            "path": str(interpreter_path),
            **hash_path(interpreter_path, symlink_policy="reject").to_dict(),
        }
        try:
            target_after_hash = interpreter_launcher.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ControllerValidationError(
                "controller interpreter launcher changed while it was frozen"
            ) from exc
        if target_after_hash != interpreter_path:
            raise ControllerValidationError(
                "controller interpreter launcher changed while it was frozen"
            )
        models: dict[str, Any] = {}
        placeholder = {
            "batch_id": "__BATCH_ID__",
            "python_executable": str(interpreter_launcher),
            "controller_root": self.runner_root,
            "batch_root": "__BATCH_ROOT__",
            "train_manifest": "__TRAIN_MANIFEST__",
            "validation_manifest": "__VALIDATION_MANIFEST__",
            "benchmark_manifest": "__BENCHMARK_MANIFEST__",
            "media_manifest": "__MEDIA_MANIFEST__",
            "media_manifest_sha256": "0" * 64,
            "leakage_audit": "__LEAKAGE_AUDIT__",
            "pretrained_root": self.pretrained_root,
            "env": {"CUDA_VISIBLE_DEVICES": "__GPU_UUID__"},
        }
        for model_id, adapter in self.adapters.items():
            contexts = {
                "finetune": AdapterContext(
                    **placeholder,
                    output_path="__ARTIFACT_OUTPUT__",
                    purpose="production",
                    training_steps=1,
                ),
                "evaluation": AdapterContext(
                    **placeholder,
                    output_path="__PREDICTIONS_OUTPUT__",
                    artifact_path="__ARTIFACT__",
                    limit=32,
                ),
                "verifier": AdapterContext(
                    **placeholder,
                    output_path="__VERIFIER_REPORT__",
                    artifact_path="__ARTIFACT__",
                    artifact_digest="0" * 64,
                    attempt_id="__ATTEMPT_ID__",
                ),
            }
            specs = {
                "finetune": adapter.finetune_spec(contexts["finetune"]),
                "evaluation": adapter.evaluation_spec(contexts["evaluation"]),
                "verifier": adapter.verification_spec(contexts["verifier"]),
            }
            roles: dict[str, Any] = {}
            runners = {
                "finetune": adapter.finetune_runner,
                "evaluation": adapter.evaluation_runner,
                "verifier": adapter.verifier_runner,
            }
            for role, relative in runners.items():
                absolute = self._join_runtime_path(self.runner_root, relative)
                candidate = Path(absolute)
                present = candidate.is_file() and not candidate.is_symlink()
                runner_receipt = None
                if present:
                    resolved = candidate.resolve(strict=True)
                    runner_receipt = {
                        "path": str(resolved),
                        **hash_path(resolved, symlink_policy="reject").to_dict(),
                    }
                    absolute = str(resolved)
                template = specs[role].receipt()
                backend_relative = adapter.backend_for(role)
                backend_absolute = self._join_runtime_path(
                    self.runner_root, backend_relative
                )
                backend_candidate = Path(backend_absolute)
                backend_present = (
                    backend_candidate.is_file() and not backend_candidate.is_symlink()
                )
                backend_receipt = None
                if backend_present:
                    backend_resolved = backend_candidate.resolve(strict=True)
                    backend_receipt = {
                        "path": str(backend_resolved),
                        **hash_path(backend_resolved, symlink_policy="reject").to_dict(),
                    }
                    backend_absolute = str(backend_resolved)
                roles[role] = {
                    "relative_path": relative,
                    "absolute_path": absolute,
                    "state": "present" if present else "missing",
                    "runner_receipt": runner_receipt,
                    "backend_relative_path": backend_relative,
                    "backend_absolute_path": backend_absolute,
                    "backend_state": "present" if backend_present else "missing",
                    "backend_receipt": backend_receipt,
                    "command_template": template,
                    "command_template_sha256": sha256_json(template),
                }
            models[model_id] = roles
        try:
            final_interpreter_target = interpreter_launcher.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ControllerValidationError(
                "controller interpreter launcher changed while runtime commands were frozen"
            ) from exc
        if final_interpreter_target != interpreter_path:
            raise ControllerValidationError(
                "controller interpreter launcher changed while runtime commands were frozen"
            )
        body = {
            "schema_version": "1.0",
            "interpreter": interpreter,
            "gpu_policy": {
                "selector_required": True,
                "identity": "nvidia-smi-uuid",
                "cuda_visible_devices_binding": "gpu_uuid",
                "keepalive_root": str(self.keepalive_root),
                "keepalive_owner": self.keepalive_owner,
                "prepare_worker_required": True,
            },
            "event_trust": self.event_trust,
            "models": models,
        }
        return {**body, "runtime_contract_sha256": sha256_json(body)}

    @staticmethod
    def _runtime_role(
        receipt: Mapping[str, Any], model_id: str, role: str, *, require_present: bool = True
    ) -> Mapping[str, Any]:
        try:
            value = receipt["runtime_contract"]["models"][model_id][role]
        except (KeyError, TypeError) as exc:
            raise ControllerValidationError("frozen runtime role is missing") from exc
        if not isinstance(value, Mapping):
            raise ControllerValidationError("frozen runtime role is invalid")
        if require_present and value.get("state") != "present":
            raise ControllerValidationError(
                f"frozen catalog {role} runner is missing for {model_id}"
            )
        return value

    @staticmethod
    def _runtime_backend(
        receipt: Mapping[str, Any], model_id: str, role: str, *, require_present: bool = True
    ) -> Mapping[str, Any]:
        runtime_role = BatchController._runtime_role(
            receipt, model_id, role, require_present=False
        )
        required = {
            "backend_relative_path", "backend_absolute_path",
            "backend_state", "backend_receipt",
        }
        if not required.issubset(runtime_role):
            raise ControllerValidationError(
                f"frozen catalog {role} backend contract is unavailable for {model_id}"
            )
        if require_present and runtime_role.get("backend_state") != "present":
            raise ControllerValidationError(
                f"frozen catalog {role} backend is missing for {model_id}; "
                f"formally block component=backend:{role}"
            )
        return runtime_role

    def _run_frozen_runtime_role(
        self,
        receipt: Mapping[str, Any],
        *,
        model_id: str,
        role: str,
        purpose: str,
        command: CommandSpec,
    ):
        """Execute exactly the source bytes whose digests were frozen in the batch."""

        self._require_verified_multi_root_bootstrap(role=role, purpose=purpose)
        runtime_role = self._runtime_role(receipt, model_id, role)
        interpreter = receipt["runtime_contract"]["interpreter"]
        interpreter_launcher = self._frozen_interpreter(receipt)
        runner = runtime_role.get("runner_receipt")
        runner_code = receipt.get("runner_code")
        if (
            not isinstance(runner, Mapping)
            or not isinstance(runner_code, Mapping)
            or runner.get("kind") != "file"
            or runner_code.get("kind") != "directory"
            or interpreter.get("kind") != "file"
            or tuple(command.argv[:2])
            != (interpreter_launcher, runtime_role.get("absolute_path"))
        ):
            raise ControllerValidationError(
                f"{role} command is not bound to the frozen Python runtime"
            )
        return run_verified_python(
            command,
            expected_interpreter_sha256=str(interpreter.get("digest")),
            expected_interpreter_target_path=str(interpreter.get("path")),
            expected_script_sha256=str(runner.get("digest")),
            import_root=str(runner_code.get("path")),
            expected_import_root_receipt=runner_code,
        )

    def _controller_verified_multi_root_bootstrap(
        self, *, role: str, purpose: str
    ) -> bool:
        """Return whether the not-yet-implemented production bootstrap is active.

        This intentionally remains false.  Keeping the decision in the trusted
        controller gives a later dedicated implementation one narrow contract
        to replace without treating a post-import ``__file__`` check as proof.
        """

        del role, purpose
        return False

    def _require_verified_multi_root_bootstrap(
        self, *, role: str, purpose: str
    ) -> None:
        if role == "finetune" and purpose == "preflight":
            return
        if self._controller_verified_multi_root_bootstrap(
            role=role, purpose=purpose
        ):
            return
        raise ControllerValidationError(_VERIFIED_MULTI_ROOT_BOOTSTRAP_BLOCKER)

    @staticmethod
    def _frozen_interpreter(
        receipt: Mapping[str, Any], requested: str | Path | None = None
    ) -> str:
        try:
            runtime_contract = receipt["runtime_contract"]
            interpreter_receipt = runtime_contract["interpreter"]
            interpreter_target = interpreter_receipt["path"]
            recorded_launcher = interpreter_receipt.get("launcher_path")
            models = runtime_contract["models"]
        except (KeyError, TypeError) as exc:
            raise ControllerValidationError("batch lacks a frozen controller interpreter") from exc
        if (
            not isinstance(interpreter_target, str)
            or not Path(interpreter_target).is_absolute()
            or not isinstance(models, Mapping)
        ):
            raise ControllerValidationError("batch frozen controller interpreter is invalid")

        if recorded_launcher is not None and (
            not isinstance(recorded_launcher, str)
            or not Path(recorded_launcher).is_absolute()
        ):
            raise ControllerValidationError(
                "batch frozen controller interpreter launcher is invalid"
            )

        frozen: str | None = recorded_launcher
        for roles in models.values():
            if not isinstance(roles, Mapping):
                raise ControllerValidationError(
                    "batch frozen controller interpreter templates are invalid"
                )
            for runtime_role in roles.values():
                if not isinstance(runtime_role, Mapping):
                    raise ControllerValidationError(
                        "batch frozen controller interpreter templates are invalid"
                    )
                template = runtime_role.get("command_template")
                argv = template.get("argv") if isinstance(template, Mapping) else None
                launcher = argv[0] if isinstance(argv, list) and argv else None
                if not isinstance(launcher, str) or not Path(launcher).is_absolute():
                    raise ControllerValidationError(
                        "batch frozen controller interpreter launcher is invalid"
                    )
                if frozen is None:
                    frozen = launcher
                elif not _same_lexical_path(launcher, frozen):
                    raise ControllerValidationError(
                        "batch command templates bind different interpreter launchers"
                    )
        if frozen is None:
            raise ControllerValidationError(
                "batch lacks a frozen controller interpreter launcher"
            )

        try:
            frozen_target = Path(frozen).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ControllerValidationError(
                "frozen interpreter launcher is missing; batch remains retryable"
            ) from exc
        if (
            not Path(frozen).is_file()
            or not _same_lexical_path(frozen_target, interpreter_target)
        ):
            raise ControllerValidationError(
                "frozen interpreter launcher no longer resolves to its frozen target"
            )
        if requested is not None:
            try:
                candidate = _absolute_path_preserving_links(requested)
                candidate_target = candidate.resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ControllerValidationError(
                    "requested interpreter is missing; batch remains retryable"
                ) from exc
            if (
                not candidate.is_file()
                or not _same_lexical_path(candidate, frozen)
                or not _same_lexical_path(candidate_target, interpreter_target)
            ):
                raise ControllerValidationError(
                    "requested interpreter differs from the frozen controller interpreter"
                )
        return frozen

    def _gpu_binding(
        self, receipt: Mapping[str, Any], selector: str | int | None
    ) -> dict[str, Any]:
        if selector is None:
            raise ControllerValidationError(
                "formal finetune/evaluation attempts require an explicit GPU UUID or index"
            )
        normalized: str | int = selector
        if isinstance(selector, str) and selector.isdecimal():
            normalized = int(selector)
        try:
            inventory = self.gpu_probe.query()
            device = inventory.device(normalized)
        except Exception as exc:
            raise ControllerValidationError(
                "GPU identity could not be proven; attempt remains retryable"
            ) from exc
        policy = receipt["runtime_contract"]["gpu_policy"]
        if (
            policy.get("keepalive_root") != str(self.keepalive_root)
            or policy.get("keepalive_owner") != self.keepalive_owner
        ):
            raise ControllerValidationError(
                "controller keepalive root/owner differs from the frozen batch policy"
            )
        return {
            "gpu_uuid": device.uuid,
            "gpu_index": device.index,
            "keepalive_root": policy["keepalive_root"],
            "keepalive_owner": policy["keepalive_owner"],
        }

    def _prepare_worker_gpu(
        self, receipt: Mapping[str, Any], attempt: Mapping[str, Any]
    ) -> None:
        policy = receipt["runtime_contract"]["gpu_policy"]
        expected = {
            "gpu_uuid": attempt.get("gpu_uuid"),
            "gpu_index": attempt.get("gpu_index"),
            "keepalive_root": attempt.get("keepalive_root"),
            "keepalive_owner": attempt.get("keepalive_owner"),
        }
        if (
            expected["keepalive_root"] != policy.get("keepalive_root")
            or expected["keepalive_owner"] != policy.get("keepalive_owner")
            or not isinstance(expected["gpu_uuid"], str)
            or type(expected["gpu_index"]) is not int
        ):
            raise ControllerValidationError("attempt GPU/keepalive binding is invalid")
        try:
            inventory = self.gpu_probe.query()
            device = inventory.device(expected["gpu_uuid"])
            if device.index != expected["gpu_index"]:
                raise ControllerValidationError("GPU UUID/index mapping changed")
            if not inventory.is_idle(device):
                raise ControllerValidationError(
                    "GPU is not proven idle after role acquisition"
                )
            KeepaliveStore(
                expected["keepalive_root"],
                project_owner=expected["keepalive_owner"],
                probe=self.gpu_probe,
            ).prepare_worker(expected["gpu_uuid"])
        except ControllerValidationError:
            raise
        except Exception as exc:
            raise ControllerValidationError(
                "GPU/keepalive pre-launch proof failed; attempt remains retryable"
            ) from exc

    @staticmethod
    def _gpu_command_env(
        env: Mapping[str, str] | None, gpu_uuid: str
    ) -> dict[str, str]:
        result = dict(env or {})
        existing = result.get("CUDA_VISIBLE_DEVICES")
        if existing is not None and existing != gpu_uuid:
            raise ControllerValidationError(
                "CUDA_VISIBLE_DEVICES conflicts with the controller-proven GPU UUID"
            )
        result["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        return result

    def create_batch(
        self,
        batch_id: str,
        *,
        inputs: Mapping[str, str | Path],
        config: Mapping[str, Any] | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        root = self.batch_root(batch_id, must_exist=False)
        if root.exists():
            raise FileExistsError(f"batch ID is immutable and already exists: {batch_id}")
        owner_token = f"{os.getpid()}:{uuid.uuid4().hex}"
        owner_marker = root / ".create_owner.json"
        owns_root = False
        published = False
        try:
            root.mkdir(parents=False, exist_ok=False)
            owns_root = True
            atomic_write_json(
                owner_marker,
                {"owner_token": owner_token},
                root=root,
                overwrite=False,
            )
            (root / "00_inputs").mkdir()
            (root / "02_finetune").mkdir()
            receipt = create_batch_receipt(
                root / "00_inputs" / "batch_receipt.json",
                batch_id=batch_id,
                inputs=inputs,
                registry_path=self.registry.registry_path,
                pretrained_registry_path=self.registry.pretrained_registry_path,
                code_root=self.code_root,
                runner_code_root=Path(self.runner_root) / "scripts",
                model_ids=list(self.registry.ids),
                model_modalities={
                    model.model_id: model.modality.value for model in self.registry.models
                },
                pretrained_artifact_specs={
                    model_id: [
                        {
                            "role": artifact.role,
                            "path": artifact.path,
                            "kind": artifact.kind,
                            "expected_sha256": artifact.expected_sha256,
                        }
                        for artifact in self.registry.pretrained_artifacts[model_id]
                    ]
                    for model_id in self.registry.ids
                },
                runtime_roots={
                    "controller_root": self.runner_root,
                    "pretrained_root": self.pretrained_root,
                },
                runtime_contract=self._runtime_contract(),
                config=config,
                description=description,
            )
            EventStore(root).initialize(
                batch_id=batch_id,
                receipt_sha256=receipt["receipt_sha256"],
                model_ids=list(self.registry.ids),
            )
            published = True
            owner_marker.unlink()
            return receipt
        except BaseException:
            # Atomic mkdir is the ownership boundary.  A competing creator
            # that loses mkdir must never remove the winner's directory.
            # Before publication, cleanup additionally requires our unique
            # marker so a changed/replaced directory fails closed.
            if owns_root and not published and root.is_dir():
                try:
                    marker = load_json_strict(owner_marker)
                except Exception:
                    marker = None
                if (
                    isinstance(marker, Mapping)
                    and marker.get("owner_token") == owner_token
                    and root.parent == self.workspace_root
                    and root.name == batch_id
                ):
                    shutil.rmtree(root)
            raise

    def _receipt(self, batch_id: str) -> tuple[Path, dict[str, Any]]:
        root = self.batch_root(batch_id)
        receipt = load_and_validate_batch_receipt(root / "00_inputs" / "batch_receipt.json")
        if receipt["batch_id"] != batch_id:
            raise ControllerValidationError("batch directory/receipt ID mismatch")
        if tuple(receipt["registry"]["model_ids"]) != self.registry.ids:
            raise ControllerValidationError("batch receipt canonical model coverage changed")
        if tuple(receipt["pretrained_registry"]["model_ids"]) != self.registry.ids:
            raise ControllerValidationError("batch pretrain coverage changed")
        if receipt["registry"]["sha256"] != sha256_file(self.registry.registry_path):
            raise ControllerValidationError("controller registry differs from frozen batch")
        if receipt["pretrained_registry"]["sha256"] != sha256_file(
            self.registry.pretrained_registry_path
        ):
            raise ControllerValidationError("controller pretrained registry differs from frozen batch")
        self._frozen_interpreter(receipt)
        gpu_policy = receipt["runtime_contract"]["gpu_policy"]
        if (
            gpu_policy["keepalive_root"] != str(self.keepalive_root)
            or gpu_policy["keepalive_owner"] != self.keepalive_owner
        ):
            raise ControllerValidationError(
                "controller keepalive root/owner differs from frozen batch"
            )
        frozen_trust = receipt["runtime_contract"]["event_trust"]
        if (
            frozen_trust.get("key_id") != self.event_trust["key_id"]
            or frozen_trust.get("state_scope_id")
            != self.event_trust["state_scope_id"]
            or frozen_trust.get("storage_scope") != "external_controller_state"
        ):
            raise ControllerValidationError("controller event trust root differs from frozen batch")
        # The immutable receipt and its cheap pretrained index are trusted only
        # through the external HMAC-protected event state.  Checking this in the
        # common transition path prevents a forged, self-consistent receipt and
        # index from being accepted without rewriting the external trust root.
        state = EventStore(root).load()
        if state.get("batch_receipt_sha256") != receipt.get("receipt_sha256"):
            raise ControllerValidationError(
                "external HMAC state is bound to a different batch receipt"
            )
        return root, receipt

    def state(self, batch_id: str) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        if state["batch_receipt_sha256"] != receipt["receipt_sha256"]:
            raise ControllerValidationError("state is bound to a different input receipt")
        return state

    def barrier_status(self, batch_id: str) -> dict[str, Any]:
        state = self.state(batch_id)
        counts = {status: 0 for status in ("pending", "finetune_complete", "blocked")}
        for model in state["models"].values():
            counts[model["finetune_status"]] += 1
        return {
            "batch_id": batch_id,
            "models": len(state["models"]),
            "counts": counts,
            "barrier_terminal": counts["pending"] == 0,
            "eval_open": state["eval_open"],
        }

    def plan(
        self,
        batch_id: str,
        *,
        python_executable: str | None = None,
        controller_root: str | None = None,
    ) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        frozen_controller_root = receipt["runtime_roots"]["controller_root"]
        if controller_root is not None and controller_root != frozen_controller_root:
            raise ControllerValidationError("controller_root differs from the frozen batch root")
        controller_root = frozen_controller_root
        interpreter = self._frozen_interpreter(receipt, python_executable)
        rows: list[dict[str, Any]] = []
        for model in self.registry.models:
            adapter = self.adapters[model.model_id]
            training_config = self._model_training_config(receipt, model.model_id)
            artifact = str(root / "02_finetune" / model.model_id / "attempts" / "<attempt>" / "artifact")
            context = AdapterContext(
                batch_id=batch_id,
                python_executable=interpreter,
                controller_root=controller_root,
                batch_root=str(root),
                train_manifest=inputs["train"],
                validation_manifest=inputs["validation"],
                benchmark_manifest=inputs["benchmark"],
                media_manifest=inputs["media_manifest"],
                media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
                leakage_audit=inputs["leakage_audit"],
                pretrained_root=receipt["runtime_roots"]["pretrained_root"],
                output_path=artifact,
                purpose="production",
                training_steps=training_config["training_steps"],
                env={"CUDA_VISIBLE_DEVICES": "__GPU_UUID__"},
            )
            finetune = adapter.finetune_spec(context)
            eval_context = AdapterContext(
                **{**context.__dict__, "output_path": str(root / "03_eval" / model.model_id / "<stage>" / "predictions.jsonl"), "artifact_path": artifact}
            )
            evaluate = adapter.evaluation_spec(eval_context)
            rows.append(
                {
                    "model_id": model.model_id,
                    "modality": model.modality.value,
                    "prediction_kind": model.prediction_kind,
                    "finetune_status": state["models"][model.model_id]["finetune_status"],
                    "finetune": finetune.receipt(),
                    "evaluation": evaluate.receipt(),
                    "runtime_contract": receipt["runtime_contract"]["models"][model.model_id],
                }
            )
        return {
            "batch_id": batch_id,
            "global_finetune_barrier": True,
            "models": rows,
        }

    def _model(self, model_id: str) -> ModelSpec:
        return self.registry.model(model_id)

    def _model_training_config(
        self, receipt: Mapping[str, Any], model_id: str
    ) -> Mapping[str, Any]:
        config = receipt.get("config", {}).get("model_training", {}).get(model_id)
        if not isinstance(config, Mapping):
            raise ControllerValidationError(
                f"frozen production training config is missing for {model_id}"
            )
        return config

    def _model_pretrained_assets(
        self, receipt: Mapping[str, Any], model_id: str
    ) -> list[dict[str, Any]]:
        assets = receipt.get("pretrained_assets", {}).get(model_id)
        if not isinstance(assets, list) or not assets:
            raise ControllerValidationError(
                f"frozen pretrained asset inventory is missing for {model_id}"
            )
        return [dict(item) for item in assets]

    def _require_ready_pretrained_assets(
        self, receipt: Mapping[str, Any], model_id: str
    ) -> tuple[list[dict[str, Any]], str]:
        assets = self._model_pretrained_assets(receipt, model_id)
        unavailable = [item["role"] for item in assets if item.get("state") != "present"]
        if unavailable:
            raise ControllerValidationError(
                f"registered pretrained components are not ready for {model_id}: {unavailable}"
            )
        return assets, sha256_json(assets)

    def _attempt_root(self, root: Path, model_id: str, stage: str, attempt_id: str) -> Path:
        _safe_id(attempt_id, "attempt_id")
        if stage == "finetune":
            base = root / "02_finetune" / model_id
        elif stage in {"smoke_1", "smoke_8", "smoke_32", "full"}:
            base = root / "03_eval" / model_id / stage
        else:
            raise ControllerValidationError(f"unsupported attempt stage: {stage}")
        return resolve_within_root(base / "attempts" / attempt_id, root, must_exist=False)

    def create_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        command: CommandSpec,
        attempt_id: str | None = None,
        purpose: str | None = None,
        expected_training_steps: int | None = None,
        sample_limit: int | None = None,
        gpu_binding: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        attempt_id = attempt_id or uuid.uuid4().hex
        attempt_root = self._attempt_root(root, model_id, stage, attempt_id)
        command_receipt = command.receipt()
        command_sha256 = sha256_json(command_receipt)
        if gpu_binding is None and "runtime_contract" in receipt:
            raise ControllerValidationError("formal attempt lacks a controller-proven GPU binding")
        frozen_gpu = dict(
            gpu_binding
            or {
                "gpu_uuid": "UNBOUND-TEST",
                "gpu_index": -1,
                "keepalive_root": "UNBOUND-TEST",
                "keepalive_owner": "UNBOUND-TEST",
            }
        )
        published: dict[str, Any] = {}

        def prepare(current: dict[str, Any]):
            nonlocal purpose, expected_training_steps, sample_limit, published
            if attempt_root.exists():
                raise FileExistsError(f"attempt ID is append-only and already exists: {attempt_id}")
            model_state = current["models"][model_id]
            if stage == "finetune":
                if current["eval_open"] or model_state["finetune_status"] != "pending":
                    raise ControllerValidationError("finetune attempt is not eligible")
                if purpose not in {"production", "preflight"}:
                    raise ControllerValidationError("finetune attempt purpose is invalid")
                if type(expected_training_steps) is not int or expected_training_steps <= 0:
                    raise ControllerValidationError("finetune expected_training_steps is invalid")
                if purpose == "production" and sample_limit is not None:
                    raise ControllerValidationError("production finetune cannot use a sample limit")
                if purpose == "preflight" and (
                    type(sample_limit) is not int or sample_limit <= 0
                ):
                    raise ControllerValidationError("preflight requires a positive sample limit")
            else:
                if stage not in {"smoke_1", "smoke_8", "smoke_32", "full"}:
                    raise ControllerValidationError("unsupported evaluation stage")
                if (
                    not current["eval_open"]
                    or model_state["finetune_status"] != "finetune_complete"
                ):
                    raise ControllerValidationError("evaluation attempt is not eligible")
                if stage == "full" and not current["full_open"]:
                    raise ControllerValidationError("full evaluation gate is closed")
                if stage.startswith("smoke_") and current["full_open"]:
                    raise ControllerValidationError("smoke phase is closed")
                purpose = "evaluation"
                expected_training_steps = None
                sample_limit = None if stage == "full" else int(stage.split("_", 1)[1])
            self._require_catalog_command(
                root,
                receipt,
                current,
                model_id=model_id,
                stage=stage,
                attempt_root=attempt_root,
                command=command,
                purpose=purpose,
                expected_training_steps=expected_training_steps,
                sample_limit=sample_limit,
            )
            lease_nonce = uuid.uuid4().hex
            body: dict[str, Any] = {
                "schema_version": "2.0",
                "attempt_id": attempt_id,
                "batch_id": batch_id,
                "batch_receipt_sha256": receipt["receipt_sha256"],
                "model_id": model_id,
                "stage": stage,
                "purpose": purpose,
                "expected_training_steps": expected_training_steps,
                "sample_limit": sample_limit,
                **frozen_gpu,
                "lease_nonce": lease_nonce,
                "leased_revision": current["revision"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": command_receipt,
                "command_sha256": command_sha256,
            }
            published = {**body, "attempt_sha256": sha256_json(body)}
            relative = (attempt_root / "attempt_receipt.json").relative_to(root).as_posix()
            payload = {
                "model_id": model_id,
                "stage": stage,
                "attempt_id": attempt_id,
                "purpose": purpose,
                "lease_nonce": lease_nonce,
                "command_sha256": command_sha256,
                "attempt_sha256": published["attempt_sha256"],
                "attempt_reference": {
                    "path": relative,
                    "content_sha256": published["attempt_sha256"],
                },
                "leased_revision": current["revision"],
                **frozen_gpu,
            }

            def commit() -> None:
                attempt_root.mkdir(parents=True, exist_ok=False)
                atomic_write_json(
                    attempt_root / "attempt_receipt.json",
                    published,
                    root=root,
                    overwrite=False,
                )

            def rollback() -> None:
                if attempt_root.is_dir():
                    shutil.rmtree(attempt_root)

            return payload, commit, rollback

        EventStore(root).append_prepared("ATTEMPT_LEASED", prepare)
        return published

    def _require_catalog_command(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        model_id: str,
        stage: str,
        attempt_root: Path,
        command: CommandSpec,
        purpose: str,
        expected_training_steps: int | None,
        sample_limit: int | None,
    ) -> None:
        """Reject arbitrary executables masquerading as a model attempt."""

        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        interpreter = self._frozen_interpreter(receipt)
        if stage == "finetune":
            runtime_role = self._runtime_role(receipt, model_id, "finetune")
            context = AdapterContext(
                batch_id=receipt["batch_id"],
                python_executable=interpreter,
                controller_root=receipt["runtime_roots"]["controller_root"],
                batch_root=str(root),
                train_manifest=inputs["train"],
                validation_manifest=inputs["validation"],
                benchmark_manifest=inputs["benchmark"],
                media_manifest=inputs["media_manifest"],
                media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
                leakage_audit=inputs["leakage_audit"],
                pretrained_root=receipt["runtime_roots"]["pretrained_root"],
                output_path=str(attempt_root / "artifact"),
                limit=sample_limit,
                purpose=purpose,
                training_steps=expected_training_steps,
                env=command.env,
            )
            expected = self.adapters[model_id].finetune_spec(context)
        else:
            runtime_role = self._runtime_role(receipt, model_id, "evaluation")
            reference = state["models"][model_id]["finetune_evidence"]
            evidence = self._validate_finetune_evidence(
                root, receipt, model_id, reference
            )
            context = AdapterContext(
                batch_id=receipt["batch_id"],
                python_executable=interpreter,
                controller_root=receipt["runtime_roots"]["controller_root"],
                batch_root=str(root),
                train_manifest=inputs["train"],
                validation_manifest=inputs["validation"],
                benchmark_manifest=inputs["benchmark"],
                media_manifest=inputs["media_manifest"],
                media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
                leakage_audit=inputs["leakage_audit"],
                pretrained_root=receipt["runtime_roots"]["pretrained_root"],
                output_path=str(attempt_root / "predictions.jsonl"),
                artifact_path=evidence["artifact"]["path"],
                limit=sample_limit,
                env=command.env,
            )
            expected = self.adapters[model_id].evaluation_spec(context)
        if (
            tuple(expected.argv[:2])
            != (interpreter, runtime_role["absolute_path"])
            or command.receipt() != expected.receipt()
        ):
            raise ControllerValidationError(
                "attempt command must exactly match the canonical model adapter spec"
            )

    def create_finetune_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        python_executable: str | None = None,
        controller_root: str | None = None,
        attempt_id: str | None = None,
        env: Mapping[str, str] | None = None,
        limit: int | None = None,
        purpose: str = "production",
        gpu: str | int | None = None,
    ) -> dict[str, Any]:
        """Build the model adapter's typed spec and append an attempt receipt."""

        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        self._require_ready_pretrained_assets(receipt, model_id)
        gpu_binding = self._gpu_binding(receipt, gpu)
        command_env = self._gpu_command_env(env, gpu_binding["gpu_uuid"])
        training_config = self._model_training_config(receipt, model_id)
        if purpose == "production":
            if limit is not None:
                raise ControllerValidationError(
                    "production finetune cannot use --limit; create a preflight attempt instead"
                )
            training_steps = training_config["training_steps"]
            self._runtime_role(receipt, model_id, "verifier")
            for role in ("finetune", "verifier", "evaluation"):
                self._runtime_backend(receipt, model_id, role)
        elif purpose == "preflight":
            if type(limit) is not int or limit <= 0:
                raise ControllerValidationError("preflight finetune requires a positive --limit")
            training_steps = training_config.get("preflight_steps", 1)
            self._runtime_backend(receipt, model_id, "finetune")
        else:
            raise ControllerValidationError("finetune purpose must be production or preflight")
        attempt_id = attempt_id or uuid.uuid4().hex
        attempt_root = self._attempt_root(root, model_id, "finetune", attempt_id)
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        frozen_controller_root = receipt["runtime_roots"]["controller_root"]
        if controller_root is not None and controller_root != frozen_controller_root:
            raise ControllerValidationError("controller_root differs from the frozen batch root")
        controller_root = frozen_controller_root
        interpreter = self._frozen_interpreter(receipt, python_executable)
        context = AdapterContext(
            batch_id=batch_id,
            python_executable=interpreter,
            controller_root=controller_root,
            batch_root=str(root),
            train_manifest=inputs["train"],
            validation_manifest=inputs["validation"],
            benchmark_manifest=inputs["benchmark"],
            media_manifest=inputs["media_manifest"],
            media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
            leakage_audit=inputs["leakage_audit"],
            pretrained_root=receipt["runtime_roots"]["pretrained_root"],
            output_path=str(attempt_root / "artifact"),
            limit=limit,
            purpose=purpose,
            training_steps=training_steps,
            env=command_env,
        )
        command = self.adapters[model_id].finetune_spec(context)
        attempt = self.create_attempt(
            batch_id,
            model_id=model_id,
            stage="finetune",
            command=command,
            attempt_id=attempt_id,
            purpose=purpose,
            expected_training_steps=training_steps,
            sample_limit=limit,
            gpu_binding=gpu_binding,
        )
        return {
            "attempt": attempt,
            "command": command.receipt(),
            "artifact_path": str(attempt_root / "artifact"),
        }

    def preview_finetune_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        python_executable: str | None = None,
        controller_root: str | None = None,
        attempt_id: str = "dryrun",
        env: Mapping[str, str] | None = None,
        limit: int | None = None,
        purpose: str = "production",
        gpu: str | int | None = None,
    ) -> dict[str, Any]:
        """Build the exact no-placeholder command without writing an attempt."""

        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        if state["eval_open"] or state["models"][model_id]["finetune_status"] != "pending":
            raise ControllerValidationError("finetune attempt is not eligible")
        self._require_ready_pretrained_assets(receipt, model_id)
        gpu_binding = self._gpu_binding(receipt, gpu)
        command_env = self._gpu_command_env(env, gpu_binding["gpu_uuid"])
        training_config = self._model_training_config(receipt, model_id)
        if purpose == "production":
            if limit is not None:
                raise ControllerValidationError("production finetune cannot use --limit")
            training_steps = training_config["training_steps"]
            self._runtime_role(receipt, model_id, "verifier")
            for role in ("finetune", "verifier", "evaluation"):
                self._runtime_backend(receipt, model_id, role)
        elif purpose == "preflight":
            if type(limit) is not int or limit <= 0:
                raise ControllerValidationError("preflight finetune requires a positive --limit")
            training_steps = training_config.get("preflight_steps", 1)
            self._runtime_backend(receipt, model_id, "finetune")
        else:
            raise ControllerValidationError("finetune purpose must be production or preflight")
        attempt_root = self._attempt_root(root, model_id, "finetune", attempt_id)
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        frozen_controller_root = receipt["runtime_roots"]["controller_root"]
        if controller_root is not None and controller_root != frozen_controller_root:
            raise ControllerValidationError("controller_root differs from the frozen batch root")
        interpreter = self._frozen_interpreter(receipt, python_executable)
        context = AdapterContext(
            batch_id=batch_id,
            python_executable=interpreter,
            controller_root=frozen_controller_root,
            batch_root=str(root),
            train_manifest=inputs["train"],
            validation_manifest=inputs["validation"],
            benchmark_manifest=inputs["benchmark"],
            media_manifest=inputs["media_manifest"],
            media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
            leakage_audit=inputs["leakage_audit"],
            pretrained_root=receipt["runtime_roots"]["pretrained_root"],
            output_path=str(attempt_root / "artifact"),
            limit=limit,
            purpose=purpose,
            training_steps=training_steps,
            env=command_env,
        )
        command = self.adapters[model_id].finetune_spec(context)
        self._require_catalog_command(
            root,
            receipt,
            state,
            model_id=model_id,
            stage="finetune",
            attempt_root=attempt_root,
            command=command,
            purpose=purpose,
            expected_training_steps=training_steps,
            sample_limit=limit,
        )
        return {
            "dry_run": True,
            "attempt_id": attempt_id,
            "purpose": purpose,
            "expected_training_steps": training_steps,
            "sample_limit": limit,
            "command": command.receipt(),
            "artifact_path": str(attempt_root / "artifact"),
        }

    def create_evaluation_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        python_executable: str | None = None,
        controller_root: str | None = None,
        attempt_id: str | None = None,
        env: Mapping[str, str] | None = None,
        gpu: str | int | None = None,
    ) -> dict[str, Any]:
        """Build one smoke/full spec bound to the current-batch artifact."""

        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        gpu_binding = self._gpu_binding(receipt, gpu)
        command_env = self._gpu_command_env(env, gpu_binding["gpu_uuid"])
        finetune_reference = state["models"][model_id]["finetune_evidence"]
        evidence = self._validate_finetune_evidence(
            root, receipt, model_id, finetune_reference
        )
        attempt_id = attempt_id or uuid.uuid4().hex
        attempt_root = self._attempt_root(root, model_id, stage, attempt_id)
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        frozen_controller_root = receipt["runtime_roots"]["controller_root"]
        if controller_root is not None and controller_root != frozen_controller_root:
            raise ControllerValidationError("controller_root differs from the frozen batch root")
        controller_root = frozen_controller_root
        interpreter = self._frozen_interpreter(receipt, python_executable)
        limit = None if stage == "full" else int(stage.split("_", 1)[1])
        context = AdapterContext(
            batch_id=batch_id,
            python_executable=interpreter,
            controller_root=controller_root,
            batch_root=str(root),
            train_manifest=inputs["train"],
            validation_manifest=inputs["validation"],
            benchmark_manifest=inputs["benchmark"],
            media_manifest=inputs["media_manifest"],
            media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
            leakage_audit=inputs["leakage_audit"],
            pretrained_root=receipt["runtime_roots"]["pretrained_root"],
            output_path=str(attempt_root / "predictions.jsonl"),
            artifact_path=evidence["artifact"]["path"],
            limit=limit,
            env=command_env,
        )
        command = self.adapters[model_id].evaluation_spec(context)
        attempt = self.create_attempt(
            batch_id,
            model_id=model_id,
            stage=stage,
            command=command,
            attempt_id=attempt_id,
            gpu_binding=gpu_binding,
        )
        return {
            "attempt": attempt,
            "command": command.receipt(),
            "predictions_path": str(attempt_root / "predictions.jsonl"),
        }

    def preview_evaluation_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        python_executable: str | None = None,
        controller_root: str | None = None,
        attempt_id: str = "dryrun",
        env: Mapping[str, str] | None = None,
        gpu: str | int | None = None,
    ) -> dict[str, Any]:
        """Build the exact requested smoke/full command without placeholders."""

        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        gpu_binding = self._gpu_binding(receipt, gpu)
        command_env = self._gpu_command_env(env, gpu_binding["gpu_uuid"])
        model_state = state["models"][model_id]
        if not state["eval_open"] or model_state["finetune_status"] != "finetune_complete":
            raise ControllerValidationError("evaluation attempt is not eligible")
        if stage == "full" and not state["full_open"]:
            raise ControllerValidationError("full evaluation gate is closed")
        if stage.startswith("smoke_") and state["full_open"]:
            raise ControllerValidationError("smoke phase is closed")
        if stage not in {"smoke_1", "smoke_8", "smoke_32", "full"}:
            raise ControllerValidationError("unsupported evaluation stage")
        if stage.startswith("smoke_"):
            requested = stage.split("_", 1)[1]
            expected = next(
                (
                    size
                    for size in ("1", "8", "32")
                    if model_state["smoke"][size] != "passed"
                ),
                None,
            )
            if requested != expected:
                raise ControllerValidationError(
                    f"smoke sequence violation; expected {expected}, got {requested}"
                )
        evidence = self._validate_finetune_evidence(
            root, receipt, model_id, model_state["finetune_evidence"]
        )
        attempt_root = self._attempt_root(root, model_id, stage, attempt_id)
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        frozen_controller_root = receipt["runtime_roots"]["controller_root"]
        if controller_root is not None and controller_root != frozen_controller_root:
            raise ControllerValidationError("controller_root differs from the frozen batch root")
        limit = None if stage == "full" else int(stage.split("_", 1)[1])
        interpreter = self._frozen_interpreter(receipt, python_executable)
        context = AdapterContext(
            batch_id=batch_id,
            python_executable=interpreter,
            controller_root=frozen_controller_root,
            batch_root=str(root),
            train_manifest=inputs["train"],
            validation_manifest=inputs["validation"],
            benchmark_manifest=inputs["benchmark"],
            media_manifest=inputs["media_manifest"],
            media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
            leakage_audit=inputs["leakage_audit"],
            pretrained_root=receipt["runtime_roots"]["pretrained_root"],
            output_path=str(attempt_root / "predictions.jsonl"),
            artifact_path=evidence["artifact"]["path"],
            limit=limit,
            env=command_env,
        )
        command = self.adapters[model_id].evaluation_spec(context)
        self._require_catalog_command(
            root,
            receipt,
            state,
            model_id=model_id,
            stage=stage,
            attempt_root=attempt_root,
            command=command,
            purpose="evaluation",
            expected_training_steps=None,
            sample_limit=limit,
        )
        return {
            "dry_run": True,
            "attempt_id": attempt_id,
            "stage": stage,
            "sample_limit": limit,
            "command": command.receipt(),
            "predictions_path": str(attempt_root / "predictions.jsonl"),
        }

    def _load_attempt(
        self, root: Path, *, model_id: str, stage: str, attempt_id: str
    ) -> tuple[Path, dict[str, Any]]:
        attempt_root = self._attempt_root(root, model_id, stage, attempt_id)
        if not attempt_root.is_dir():
            raise ControllerValidationError("attempt directory does not exist")
        value = load_json_strict(attempt_root / "attempt_receipt.json")
        if not isinstance(value, Mapping):
            raise ControllerValidationError("attempt receipt must be an object")
        attempt = dict(value)
        expected_fields = {
            "schema_version", "attempt_id", "batch_id", "batch_receipt_sha256",
            "model_id", "stage", "purpose", "expected_training_steps", "sample_limit",
            "gpu_uuid", "gpu_index", "keepalive_root", "keepalive_owner",
            "lease_nonce", "leased_revision", "created_at", "command", "command_sha256",
            "attempt_sha256",
        }
        if set(attempt) != expected_fields or attempt.get("schema_version") != "2.0":
            raise ControllerValidationError("attempt receipt schema is invalid")
        if attempt.get("attempt_sha256") != sha256_json(_attempt_body(attempt)):
            raise ControllerValidationError("attempt receipt hash mismatch")
        expected = (model_id, stage, attempt_id)
        if (attempt.get("model_id"), attempt.get("stage"), attempt.get("attempt_id")) != expected:
            raise ControllerValidationError("attempt path/identity mismatch")
        if attempt.get("command_sha256") != sha256_json(attempt.get("command")):
            raise ControllerValidationError("attempt command hash mismatch")
        command = attempt.get("command")
        if not isinstance(command, Mapping) or command.get("shell") is not False:
            raise ControllerValidationError("attempt does not prove shell-free execution")
        state = EventStore(root).load()
        anchor = state.get("attempts", {}).get(sha256_json([model_id, stage, attempt_id]))
        expected_reference = {
            "path": (attempt_root / "attempt_receipt.json").relative_to(root).as_posix(),
            "content_sha256": attempt["attempt_sha256"],
        }
        if (
            not isinstance(anchor, Mapping)
            or anchor.get("lease_nonce") != attempt.get("lease_nonce")
            or anchor.get("command_sha256") != attempt.get("command_sha256")
            or anchor.get("attempt_sha256") != attempt.get("attempt_sha256")
            or anchor.get("leased_revision") != attempt.get("leased_revision")
            or anchor.get("attempt_reference") != expected_reference
            or anchor.get("gpu_uuid") != attempt.get("gpu_uuid")
            or anchor.get("gpu_index") != attempt.get("gpu_index")
            or anchor.get("keepalive_root") != attempt.get("keepalive_root")
            or anchor.get("keepalive_owner") != attempt.get("keepalive_owner")
        ):
            raise ControllerValidationError(
                "attempt is not bound to its controller lease/event nonce"
            )
        return attempt_root, attempt

    def execute_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        attempt_id: str,
        command: CommandSpec,
    ) -> dict[str, Any]:
        """Hold the shared per-GPU role mutex for worker and verifier."""

        root, receipt = self._receipt(batch_id)
        _attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage=stage, attempt_id=attempt_id
        )
        if stage != "finetune" and stage not in {"smoke_1", "smoke_8", "smoke_32", "full"}:
            raise ControllerValidationError("worker GPU role is invalid")
        runtime_role = "finetune" if stage == "finetune" else "evaluation"
        self._require_verified_multi_root_bootstrap(
            role=runtime_role, purpose=str(attempt.get("purpose"))
        )
        gpu_role = "finetune" if stage == "finetune" else "eval"
        if sha256_json(command.receipt()) != attempt["command_sha256"]:
            raise ControllerValidationError("execution command differs from frozen attempt")
        if command.env.get("CUDA_VISIBLE_DEVICES") != attempt.get("gpu_uuid"):
            raise ControllerValidationError("frozen command is not bound to its proven GPU UUID")
        device = GPUDevice(
            int(attempt["gpu_index"]),
            str(attempt["gpu_uuid"]),
            "controller-bound",
            1,
            0,
            0,
        )
        role_store = GPULeaseStore(
            attempt["keepalive_root"],
            project_owner=attempt["keepalive_owner"],
            probe=self.gpu_probe,
        )
        lease = role_store.acquire_role(
            device,
            role=gpu_role,
            pid=os.getpid(),
            purpose=f"{batch_id}:{model_id}:{stage}:{attempt_id}",
        )
        try:
            # Probe and lifecycle checks happen only after the atomic claim, so
            # a concurrent keepalive start cannot enter between check/launch.
            self._prepare_worker_gpu(receipt, attempt)
            return self._execute_attempt_with_owned_gpu(
                batch_id,
                model_id=model_id,
                stage=stage,
                attempt_id=attempt_id,
                command=command,
            )
        finally:
            role_store.release(
                device.uuid,
                lease_id=str(lease["lease_id"]),
                pid=os.getpid(),
                role=gpu_role,
            )

    def _execute_attempt_with_owned_gpu(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        attempt_id: str,
        command: CommandSpec,
    ) -> dict[str, Any]:
        """Execute the frozen argv and attest the output observed at process exit.

        Completion never accepts a receipt invented after the fact: the
        expected worker output must not exist before execution and its exact
        hash is captured by the controller immediately after subprocess exit.
        """

        root, receipt = self._receipt(batch_id)
        attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage=stage, attempt_id=attempt_id
        )
        runtime_role = "finetune" if stage == "finetune" else "evaluation"
        self._require_verified_multi_root_bootstrap(
            role=runtime_role, purpose=str(attempt.get("purpose"))
        )
        if sha256_json(command.receipt()) != attempt["command_sha256"]:
            raise ControllerValidationError("execution command differs from frozen attempt")
        if command.env.get("CUDA_VISIBLE_DEVICES") != attempt.get("gpu_uuid"):
            raise ControllerValidationError("frozen command is not bound to its proven GPU UUID")
        state = EventStore(root).load()
        EventStore(root).append(
            "ATTEMPT_STARTED",
            {
                "model_id": model_id,
                "stage": stage,
                "attempt_id": attempt_id,
                "lease_nonce": attempt["lease_nonce"],
                "command_sha256": attempt["command_sha256"],
                "gpu_uuid": attempt["gpu_uuid"],
            },
            expected_revision=state["revision"],
        )
        execution_path = (
            root
            / ".controller"
            / "attestations"
            / "executions"
            / model_id
            / stage
            / f"{attempt_id}.json"
        )
        if execution_path.exists():
            raise ControllerValidationError("controller execution attestation already exists")
        output_name = "run_manifest.json" if stage == "finetune" else "predictions.jsonl"
        output_path = attempt_root / output_name
        if output_path.exists():
            raise ControllerValidationError("expected worker output existed before command execution")
        started_epoch_ns = time.time_ns()
        started_at = datetime.now(timezone.utc)
        result = self._run_frozen_runtime_role(
            receipt,
            model_id=model_id,
            role=runtime_role,
            purpose=str(attempt.get("purpose")),
            command=command,
        )
        finished_at = datetime.now(timezone.utc)
        output_observed = output_path.is_file()
        output_sha256 = sha256_file(output_path) if output_observed else None
        output_is_fresh = output_observed and output_path.stat().st_mtime_ns >= started_epoch_ns
        success = (
            result.process_started
            and result.succeeded
            and output_observed
            and output_is_fresh
        )
        body: dict[str, Any] = {
            "schema_version": "1.0",
            "batch_id": batch_id,
            "model_id": model_id,
            "stage": stage,
            "attempt_id": attempt_id,
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "command_sha256": attempt["command_sha256"],
            "lease_nonce": attempt["lease_nonce"],
            "gpu_uuid": attempt["gpu_uuid"],
            "gpu_index": attempt["gpu_index"],
            "status": "success" if success else "failed",
            "process_started": result.process_started,
            "exit_code": result.returncode,
            "error_code": result.error_code.value,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "output_path": str(output_path),
            "output_observed": output_observed,
            "output_fresh": output_is_fresh,
            "output_sha256": output_sha256,
            "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
        }
        execution = {**body, "execution_sha256": sha256_json(body)}
        execution_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(execution_path, execution, root=root, overwrite=False)
        reference = self._reference(root, execution_path, execution["execution_sha256"])
        EventStore(root).append(
            "ATTEMPT_EXECUTED",
            {
                "model_id": model_id,
                "stage": stage,
                "attempt_id": attempt_id,
                "lease_nonce": attempt["lease_nonce"],
                "command_sha256": attempt["command_sha256"],
                "status": execution["status"],
                "process_started": result.process_started,
                "gpu_uuid": attempt["gpu_uuid"],
                "execution": reference,
            },
        )
        if stage == "finetune" and success and attempt.get("purpose") == "production":
            self._run_finetune_verifier(
                root,
                receipt,
                model_id=model_id,
                attempt_id=attempt_id,
                attempt_root=attempt_root,
                attempt=attempt,
                manifest_path=output_path,
            )
        return execution

    def execute_frozen_attempt(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        attempt_id: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reconstruct a typed spec; secrets, if declared, come only from process env."""

        root, _ = self._receipt(batch_id)
        _, attempt = self._load_attempt(
            root, model_id=model_id, stage=stage, attempt_id=attempt_id
        )
        serialized = attempt["command"]
        if not isinstance(serialized, Mapping) or set(serialized) != {
            "argv", "cwd", "env", "timeout_seconds", "label", "shell"
        } or serialized.get("shell") is not False:
            raise ControllerValidationError("frozen command receipt schema is invalid")
        stored_env = serialized.get("env")
        if not isinstance(stored_env, Mapping):
            raise ControllerValidationError("frozen command environment is invalid")
        env: dict[str, str] = {}
        for key, value in stored_env.items():
            if isinstance(value, str):
                env[key] = value
            elif value == {"present": True, "redacted": True}:
                secret = os.environ.get(key)
                if secret is None:
                    raise ControllerValidationError(
                        f"required secret environment key {key} is not present in this process"
                    )
                env[key] = secret
            else:
                raise ControllerValidationError("frozen command environment value is invalid")
        command = CommandSpec(
            argv=tuple(serialized["argv"]),
            cwd=serialized["cwd"],
            env=env,
            timeout_seconds=serialized["timeout_seconds"],
            label=serialized["label"],
        )
        if sha256_json(command.receipt()) != attempt["command_sha256"]:
            raise ControllerValidationError("reconstructed command differs from frozen receipt")
        if dry_run:
            return {"dry_run": True, "command": command.receipt()}
        return self.execute_attempt(
            batch_id,
            model_id=model_id,
            stage=stage,
            attempt_id=attempt_id,
            command=command,
        )

    def _load_recorded_execution(
        self,
        root: Path,
        attempt_root: Path,
        attempt: Mapping[str, Any],
        *,
        batch_id: str,
        model_id: str,
        stage: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        state = EventStore(root).load()
        anchor = state.get("attempts", {}).get(
            sha256_json([model_id, stage, attempt_id])
        )
        if not isinstance(anchor, Mapping) or anchor.get("status") != "executed":
            raise ControllerValidationError(
                "execution is not anchored by a controller run-attempt event"
            )
        reference = anchor.get("execution")
        if not isinstance(reference, Mapping) or set(reference) != {
            "path", "file_sha256", "content_sha256"
        }:
            raise ControllerValidationError("controller execution reference is invalid")
        path = resolve_within_root(reference["path"], root, must_exist=True)
        if not path.is_file() or sha256_file(path) != reference["file_sha256"]:
            raise ControllerValidationError("controller execution attestation changed")
        value = load_json_strict(path)
        if not isinstance(value, Mapping):
            raise ControllerValidationError("execution receipt must be an object")
        execution = dict(value)
        expected_fields = {
            "schema_version", "batch_id", "model_id", "stage", "attempt_id",
            "batch_receipt_sha256", "attempt_sha256", "command_sha256", "status",
            "lease_nonce", "gpu_uuid", "gpu_index", "process_started", "exit_code",
            "error_code", "started_at",
            "finished_at", "output_path",
            "output_observed", "output_fresh", "output_sha256", "stdout_sha256",
            "stderr_sha256", "execution_sha256",
        }
        if set(execution) != expected_fields or execution.get("schema_version") != "1.0":
            raise ControllerValidationError("execution receipt schema is invalid")
        body = {key: item for key, item in execution.items() if key != "execution_sha256"}
        if execution.get("execution_sha256") != sha256_json(body):
            raise ControllerValidationError("execution receipt hash mismatch")
        if (
            execution.get("batch_id") != batch_id
            or execution.get("model_id") != model_id
            or execution.get("stage") != stage
            or execution.get("attempt_id") != attempt_id
            or execution.get("batch_receipt_sha256") != attempt["batch_receipt_sha256"]
            or execution.get("attempt_sha256") != attempt["attempt_sha256"]
            or execution.get("command_sha256") != attempt["command_sha256"]
            or execution.get("lease_nonce") != attempt["lease_nonce"]
            or execution.get("gpu_uuid") != attempt["gpu_uuid"]
            or execution.get("gpu_index") != attempt["gpu_index"]
            or execution.get("execution_sha256") != reference["content_sha256"]
            or anchor.get("execution_status") != execution.get("status")
            or anchor.get("process_started") != execution.get("process_started")
        ):
            raise ControllerValidationError("execution receipt identity/bindings are invalid")
        started = self._timestamp(execution.get("started_at"), "execution.started_at")
        finished = self._timestamp(execution.get("finished_at"), "execution.finished_at")
        if finished < started:
            raise ControllerValidationError("execution timestamps are reversed")
        return execution

    def _load_execution(
        self,
        root: Path,
        attempt_root: Path,
        attempt: Mapping[str, Any],
        *,
        batch_id: str,
        model_id: str,
        stage: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        execution = self._load_recorded_execution(
            root,
            attempt_root,
            attempt,
            batch_id=batch_id,
            model_id=model_id,
            stage=stage,
            attempt_id=attempt_id,
        )
        if (
            execution.get("status") != "success"
            or execution.get("process_started") is not True
            or execution.get("exit_code") != 0
            or execution.get("error_code") != "none"
            or execution.get("output_observed") is not True
            or execution.get("output_fresh") is not True
        ):
            raise ControllerValidationError("attempt was not successfully executed by the controller")
        output = resolve_within_root(execution.get("output_path"), attempt_root, must_exist=True)
        if not output.is_file() or sha256_file(output) != execution.get("output_sha256"):
            raise ControllerValidationError("worker output changed after execution")
        return execution

    @staticmethod
    def _timestamp(value: Any, field: str) -> datetime:
        if not isinstance(value, str):
            raise ControllerValidationError(f"{field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ControllerValidationError(f"{field} must be an ISO timestamp") from exc
        if parsed.tzinfo is None:
            raise ControllerValidationError(f"{field} must include a timezone")
        return parsed.astimezone(timezone.utc)

    def _run_finetune_verifier(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        *,
        model_id: str,
        attempt_id: str,
        attempt_root: Path,
        attempt: Mapping[str, Any],
        manifest_path: Path,
    ) -> dict[str, Any]:
        """Launch the frozen independent verifier and anchor its controller receipt."""

        self._require_verified_multi_root_bootstrap(
            role="verifier", purpose="production"
        )
        runtime_role = self._runtime_role(receipt, model_id, "verifier")
        value = load_json_strict(manifest_path)
        if not isinstance(value, Mapping) or not isinstance(value.get("artifact"), Mapping):
            raise ControllerValidationError(
                "production manifest lacks an artifact for independent verification"
            )
        artifact = value["artifact"]
        expected_artifact_fields = {
            "path", "algorithm", "kind", "digest", "file_count", "total_bytes"
        }
        if set(artifact) != expected_artifact_fields:
            raise ControllerValidationError("verifier artifact receipt schema is invalid")
        artifact_path = resolve_within_root(artifact["path"], attempt_root, must_exist=True)
        actual = hash_path(
            artifact_path, symlink_policy="reject", allowed_root=attempt_root
        ).to_dict()
        if actual != {key: artifact[key] for key in actual} or str(artifact_path) != artifact["path"]:
            raise ControllerValidationError("verifier artifact receipt differs from current files")
        if actual["file_count"] <= 0 or actual["total_bytes"] <= 0:
            raise ControllerValidationError("verifier refuses an empty artifact")
        report_path = attempt_root / "independent_reload_report.json"
        if report_path.exists():
            raise ControllerValidationError("verifier report existed before controller launch")
        inputs = {key: item["path"] for key, item in receipt["inputs"].items()}
        context = AdapterContext(
            batch_id=receipt["batch_id"],
            python_executable=self._frozen_interpreter(receipt),
            controller_root=receipt["runtime_roots"]["controller_root"],
            batch_root=str(root),
            train_manifest=inputs["train"],
            validation_manifest=inputs["validation"],
            benchmark_manifest=inputs["benchmark"],
            media_manifest=inputs["media_manifest"],
            media_manifest_sha256=receipt["inputs"]["media_manifest"]["digest"],
            leakage_audit=inputs["leakage_audit"],
            pretrained_root=receipt["runtime_roots"]["pretrained_root"],
            output_path=str(report_path),
            artifact_path=str(artifact_path),
            artifact_digest=actual["digest"],
            attempt_id=attempt_id,
            env={"CUDA_VISIBLE_DEVICES": attempt["gpu_uuid"]},
        )
        command = self.adapters[model_id].verification_spec(context)
        if tuple(command.argv[:2]) != (
            self._frozen_interpreter(receipt),
            runtime_role["absolute_path"],
        ):
            raise ControllerValidationError("verifier command differs from frozen catalog runner")
        self._prepare_worker_gpu(receipt, attempt)
        started_ns = time.time_ns()
        started_at = datetime.now(timezone.utc)
        result = self._run_frozen_runtime_role(
            receipt,
            model_id=model_id,
            role="verifier",
            purpose="production",
            command=command,
        )
        finished_at = datetime.now(timezone.utc)
        report_observed = report_path.is_file()
        report_fresh = report_observed and report_path.stat().st_mtime_ns >= started_ns
        report_sha256 = sha256_file(report_path) if report_observed else None
        passed = result.process_started and result.succeeded and report_fresh
        if passed:
            report = load_json_strict(report_path)
            report_fields = {
                "schema_version", "status", "batch_id", "model_id", "attempt_id",
                "artifact_digest", "checker", "checked_at",
            }
            if not isinstance(report, Mapping) or set(report) != report_fields:
                passed = False
            elif (
                report.get("schema_version") != "1.0"
                or report.get("status") != "passed"
                or report.get("batch_id") != receipt["batch_id"]
                or report.get("model_id") != model_id
                or report.get("attempt_id") != attempt_id
                or report.get("artifact_digest") != actual["digest"]
                or report.get("checker") != f"{model_id}:catalog-reload"
            ):
                passed = False
            else:
                checked = self._timestamp(report["checked_at"], "verifier.checked_at")
                if not (started_at <= checked <= finished_at):
                    passed = False
        body: dict[str, Any] = {
            "schema_version": "1.0",
            "batch_id": receipt["batch_id"],
            "model_id": model_id,
            "stage": "finetune",
            "attempt_id": attempt_id,
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "lease_nonce": attempt["lease_nonce"],
            "gpu_uuid": attempt["gpu_uuid"],
            "gpu_index": attempt["gpu_index"],
            "artifact_path": str(artifact_path),
            "artifact_digest": actual["digest"],
            "command": command.receipt(),
            "command_sha256": sha256_json(command.receipt()),
            "process_started": result.process_started,
            "exit_code": result.returncode,
            "error_code": result.error_code.value,
            "status": "passed" if passed else "failed",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "report_path": str(report_path),
            "report_observed": report_observed,
            "report_fresh": report_fresh,
            "report_sha256": report_sha256,
            "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
        }
        verification = {**body, "verification_sha256": sha256_json(body)}
        attestation_path = (
            root
            / ".controller"
            / "attestations"
            / "verifiers"
            / model_id
            / f"{attempt_id}.json"
        )
        if attestation_path.exists():
            raise ControllerValidationError("controller verifier attestation already exists")
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(attestation_path, verification, root=root, overwrite=False)
        reference = self._reference(
            root, attestation_path, verification["verification_sha256"]
        )
        EventStore(root).append(
            "ATTEMPT_VERIFIED",
            {
                "model_id": model_id,
                "stage": "finetune",
                "attempt_id": attempt_id,
                "lease_nonce": attempt["lease_nonce"],
                "command_sha256": attempt["command_sha256"],
                "gpu_uuid": attempt["gpu_uuid"],
                "verifier_status": verification["status"],
                "verification": reference,
            },
        )
        return verification

    def _load_verification(
        self,
        root: Path,
        attempt: Mapping[str, Any],
        *,
        model_id: str,
        attempt_id: str,
        artifact_digest: str,
    ) -> dict[str, Any]:
        state = EventStore(root).load()
        anchor = state.get("attempts", {}).get(
            sha256_json([model_id, "finetune", attempt_id])
        )
        if (
            not isinstance(anchor, Mapping)
            or anchor.get("verification_status") != "passed"
            or not isinstance(anchor.get("verification"), Mapping)
        ):
            raise ControllerValidationError(
                "finetune lacks a passed controller-launched verifier event"
            )
        reference = anchor["verification"]
        path = resolve_within_root(reference["path"], root, must_exist=True)
        if not path.is_file() or sha256_file(path) != reference.get("file_sha256"):
            raise ControllerValidationError("controller verifier attestation changed")
        value = load_json_strict(path)
        if not isinstance(value, Mapping):
            raise ControllerValidationError("controller verifier attestation is invalid")
        verification = dict(value)
        body = {
            key: item for key, item in verification.items() if key != "verification_sha256"
        }
        if (
            verification.get("verification_sha256") != sha256_json(body)
            or verification.get("verification_sha256") != reference.get("content_sha256")
            or verification.get("status") != "passed"
            or verification.get("process_started") is not True
            or verification.get("exit_code") != 0
            or verification.get("error_code") != "none"
            or verification.get("model_id") != model_id
            or verification.get("attempt_id") != attempt_id
            or verification.get("attempt_sha256") != attempt["attempt_sha256"]
            or verification.get("lease_nonce") != attempt["lease_nonce"]
            or verification.get("gpu_uuid") != attempt["gpu_uuid"]
            or verification.get("gpu_index") != attempt["gpu_index"]
            or verification.get("artifact_digest") != artifact_digest
        ):
            raise ControllerValidationError("controller verifier binding/result is invalid")
        report_path = resolve_within_root(
            verification.get("report_path"),
            self._attempt_root(root, model_id, "finetune", attempt_id),
            must_exist=True,
        )
        if sha256_file(report_path) != verification.get("report_sha256"):
            raise ControllerValidationError("independent verifier report changed")
        return verification

    def _validate_production_manifest(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        *,
        model_id: str,
        attempt_id: str,
        manifest_path: str | Path,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage="finetune", attempt_id=attempt_id
        )
        manifest_file = resolve_within_root(manifest_path, attempt_root, must_exist=True)
        if manifest_file != attempt_root / "run_manifest.json":
            raise ControllerValidationError("production run manifest must be attempt/run_manifest.json")
        execution = self._load_execution(
            root,
            attempt_root,
            attempt,
            batch_id=receipt["batch_id"],
            model_id=model_id,
            stage="finetune",
            attempt_id=attempt_id,
        )
        if (
            attempt.get("purpose") != "production"
            or attempt.get("sample_limit") is not None
            or type(attempt.get("expected_training_steps")) is not int
        ):
            raise ControllerValidationError(
                "only an unlimited production attempt can become finetune evidence"
            )
        if execution["output_path"] != str(manifest_file):
            raise ControllerValidationError("execution did not attest this run manifest")
        value = load_json_strict(manifest_file)
        if not isinstance(value, Mapping):
            raise ControllerValidationError("run manifest must be an object")
        manifest = dict(value)
        expected_fields = {
            "schema_version", "batch_id", "model_id", "attempt_id", "purpose", "status",
            "exit_code", "started_at", "finished_at", "training_steps", "bindings", "artifact",
            "training_receipt", "manifest_sha256",
        }
        if set(manifest) != expected_fields:
            raise ControllerValidationError("production run manifest schema is invalid")
        manifest_body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != sha256_json(manifest_body):
            raise ControllerValidationError("production run manifest hash mismatch")
        if sha256_file(manifest_file) != execution["output_sha256"]:
            raise ControllerValidationError("run manifest differs from process-exit evidence")
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("batch_id") != receipt["batch_id"]
            or manifest.get("model_id") != model_id
            or manifest.get("attempt_id") != attempt_id
            or manifest.get("purpose") != attempt["purpose"]
            or manifest.get("status") != "success"
            or manifest.get("exit_code") != 0
            or type(manifest.get("training_steps")) is not int
            or manifest["training_steps"] != attempt["expected_training_steps"]
        ):
            raise ControllerValidationError("run manifest does not prove successful production training")
        _, model_assets_sha256 = self._require_ready_pretrained_assets(
            receipt, model_id
        )
        training_config = self._model_training_config(receipt, model_id)
        bindings = manifest.get("bindings")
        expected_bindings = {
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "command_sha256": attempt["command_sha256"],
            "registry_sha256": receipt["registry"]["sha256"],
            "pretrained_registry_sha256": receipt["pretrained_registry"]["sha256"],
            "pretrained_assets_sha256": receipt["pretrained_assets_sha256"],
            "model_pretrained_assets_sha256": model_assets_sha256,
            "model_training_config_sha256": sha256_json(training_config),
            "train_sha256": receipt["inputs"]["train"]["digest"],
            "validation_sha256": receipt["inputs"]["validation"]["digest"],
            "leakage_audit_sha256": receipt["inputs"]["leakage_audit"]["digest"],
            "code_sha256": receipt["code"]["digest"],
            "runner_code_sha256": receipt["runner_code"]["digest"],
            "config_sha256": receipt["config_sha256"],
            "environment_sha256": receipt["environment_sha256"],
        }
        if not isinstance(bindings, Mapping) or dict(bindings) != expected_bindings:
            raise ControllerValidationError("run manifest provenance bindings are incomplete or wrong")
        attempt_time = self._timestamp(attempt["created_at"], "attempt.created_at")
        execution_start = self._timestamp(execution["started_at"], "execution.started_at")
        execution_finish = self._timestamp(execution["finished_at"], "execution.finished_at")
        training_start = self._timestamp(manifest["started_at"], "manifest.started_at")
        training_finish = self._timestamp(manifest["finished_at"], "manifest.finished_at")
        if not (attempt_time <= execution_start <= training_start <= training_finish <= execution_finish):
            raise ControllerValidationError("training timestamps are outside the controller-observed attempt")
        artifact = manifest.get("artifact")
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "path", "algorithm", "kind", "digest", "file_count", "total_bytes"
        }:
            raise ControllerValidationError("run manifest artifact receipt is invalid")
        artifact_path = resolve_within_root(artifact["path"], attempt_root, must_exist=True)
        actual = hash_path(artifact_path, symlink_policy="reject", allowed_root=attempt_root).to_dict()
        if actual != {key: artifact[key] for key in actual} or str(artifact_path) != artifact["path"]:
            raise ControllerValidationError("artifact receipt does not match current content")
        if actual["file_count"] <= 0 or actual["total_bytes"] <= 0:
            raise ControllerValidationError("production artifact is empty")
        training_reference = manifest.get("training_receipt")
        if not isinstance(training_reference, Mapping) or set(training_reference) != {
            "path", "file_sha256", "content_sha256"
        }:
            raise ControllerValidationError("run manifest lacks strict training evidence")
        training_path = resolve_within_root(
            training_reference.get("path"), attempt_root, must_exist=True
        )
        if training_path != attempt_root / "training_receipt.json":
            raise ControllerValidationError(
                "training receipt must be attempt/training_receipt.json"
            )
        if sha256_file(training_path) != training_reference.get("file_sha256"):
            raise ControllerValidationError("training receipt changed after publication")
        try:
            training_receipt = load_and_validate_training_receipt(
                training_path,
                expected={
                    "batch_id": receipt["batch_id"],
                    "model_id": model_id,
                    "modality": self.adapters[model_id].modality,
                    "planned_global_steps": manifest["training_steps"],
                    "actual_global_steps": manifest["training_steps"],
                    "planned_optimizer_steps": manifest["training_steps"],
                    "actual_optimizer_steps": manifest["training_steps"],
                    "batch_receipt_sha256": receipt["receipt_sha256"],
                    "attempt_sha256": attempt["attempt_sha256"],
                    "train_sha256": receipt["inputs"]["train"]["digest"],
                    "validation_sha256": receipt["inputs"]["validation"]["digest"],
                    "leakage_audit_sha256": receipt["inputs"]["leakage_audit"]["digest"],
                    "base_artifact_sha256": model_assets_sha256,
                    "config_sha256": receipt["config_sha256"],
                    "code_sha256": receipt["code"]["digest"],
                    "runner_code_sha256": receipt["runner_code"]["digest"],
                    "environment_sha256": receipt["environment_sha256"],
                    "artifact_sha256": actual["digest"],
                },
            )
        except ValueError as exc:
            raise ControllerValidationError(f"invalid training receipt: {exc}") from exc
        if training_receipt["receipt_sha256"] != training_reference.get(
            "content_sha256"
        ):
            raise ControllerValidationError(
                "training receipt content hash differs from run manifest"
            )
        execution_start_ns = int(execution_start.timestamp() * 1_000_000_000)
        artifact_files = [artifact_path] if artifact_path.is_file() else [
            path for path in artifact_path.rglob("*") if path.is_file()
        ]
        if any(path.stat().st_mtime_ns < execution_start_ns for path in artifact_files):
            raise ControllerValidationError("artifact contains files older than this execution attempt")
        verification = self._load_verification(
            root,
            attempt,
            model_id=model_id,
            attempt_id=attempt_id,
            artifact_digest=actual["digest"],
        )
        return manifest, dict(artifact), dict(training_reference), execution, verification

    def complete_finetune(
        self,
        batch_id: str,
        *,
        model_id: str,
        attempt_id: str,
        run_manifest_path: str | Path,
    ) -> dict[str, Any]:
        self._model(model_id)
        root, receipt = self._receipt(batch_id)
        self._require_verified_multi_root_bootstrap(
            role="finetune", purpose="production"
        )
        state = EventStore(root).load()
        if state["models"][model_id]["finetune_status"] != "pending" or state["eval_open"]:
            raise ControllerValidationError("finetune completion is not eligible")
        attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage="finetune", attempt_id=attempt_id
        )
        (
            manifest,
            artifact_receipt,
            training_reference,
            execution,
            verification,
        ) = self._validate_production_manifest(
            root,
            receipt,
            model_id=model_id,
            attempt_id=attempt_id,
            manifest_path=run_manifest_path,
        )
        # Close the preflight-to-publication window: every frozen batch path,
        # the attempt receipt, artifact, and training receipt are rehashed
        # immediately before the durable completion evidence is written.
        current_receipt = load_and_validate_batch_receipt(
            root / "00_inputs" / "batch_receipt.json"
        )
        _, current_attempt = self._load_attempt(
            root, model_id=model_id, stage="finetune", attempt_id=attempt_id
        )
        if current_receipt != receipt or current_attempt != attempt:
            raise ControllerValidationError(
                "batch or attempt evidence changed before finetune publication"
            )
        current_artifact = hash_path(
            artifact_receipt["path"], symlink_policy="reject", allowed_root=attempt_root
        ).to_dict()
        if current_artifact != {
            key: artifact_receipt[key] for key in current_artifact
        }:
            raise ControllerValidationError(
                "artifact changed before finetune publication"
            )
        if sha256_file(training_reference["path"]) != training_reference["file_sha256"]:
            raise ControllerValidationError(
                "training receipt changed before finetune publication"
            )
        body: dict[str, Any] = {
            "schema_version": "1.0",
            "evidence_type": "fresh_finetune",
            "batch_id": batch_id,
            "model_id": model_id,
            "attempt_id": attempt_id,
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "verification_sha256": verification["verification_sha256"],
            "run_manifest_path": str(Path(run_manifest_path).resolve(strict=True)),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "artifact": artifact_receipt,
            "training_receipt": training_reference,
            "fresh_current_batch": True,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence = {**body, "evidence_sha256": sha256_json(body)}
        evidence_path = attempt_root / "finetune_evidence.json"
        atomic_write_json(evidence_path, evidence, root=root, overwrite=False)
        reference = self._reference(root, evidence_path, evidence["evidence_sha256"])
        EventStore(root).append(
            "FINETUNE_COMPLETE",
            {"model_id": model_id, "attempt_id": attempt_id, "evidence": reference},
        )
        return evidence

    def block_finetune(
        self,
        batch_id: str,
        *,
        model_id: str,
        reason_code: str,
        component: str,
        detail: str,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        self._model(model_id)
        if reason_code not in _BLOCK_REASONS:
            raise ControllerValidationError(f"blocked reason must be one of {sorted(_BLOCK_REASONS)}")
        if not isinstance(detail, str) or not detail.strip() or _SECRET_TEXT.search(detail):
            raise ControllerValidationError("blocked detail must be non-empty and credential-free")
        root, receipt = self._receipt(batch_id)
        state = EventStore(root).load()
        if state["models"][model_id]["finetune_status"] != "pending" or state["eval_open"]:
            raise ControllerValidationError("blocked transition is not eligible")
        checked_at = datetime.now(timezone.utc).isoformat()
        if reason_code in _MISSING_REASONS:
            expected_path = self._expected_block_path(receipt, model_id, reason_code, component)
            frozen_state = "missing"
            if reason_code == "missing_code":
                if Path(expected_path).exists():
                    raise ControllerValidationError(
                        f"registered blocker target exists: {expected_path}"
                    )
            else:
                role = component.split(":", 1)[1]
                frozen = next(
                    item
                    for item in self._model_pretrained_assets(receipt, model_id)
                    if item["role"] == role
                )
                frozen_state = frozen["state"]
                if frozen_state == "present":
                    raise ControllerValidationError(
                        f"registered pretrained component is ready: {component}"
                    )
            diagnostic: dict[str, Any] = {
                "diagnostic_type": "registered_missing_component",
                "component": component,
                "expected_path": expected_path,
                "observed_state": frozen_state,
                "frozen_contract_sha256": (
                    receipt["runtime_contract"]["runtime_contract_sha256"]
                    if reason_code == "missing_code"
                    else receipt["pretrained_assets_sha256"]
                ),
                "checked_at": checked_at,
            }
        elif reason_code == "unrecoverable_provenance":
            if component != "verified-multi-root-bootstrap":
                raise ControllerValidationError(
                    "unrecoverable provenance must name "
                    "component=verified-multi-root-bootstrap"
                )
            if self._controller_verified_multi_root_bootstrap(
                role="finetune", purpose="production"
            ):
                raise ControllerValidationError(
                    "verified multi-root bootstrap is available; provenance "
                    "blocker no longer holds"
                )
            diagnostic = {
                "diagnostic_type": "unavailable_verified_multi_root_bootstrap",
                "component": component,
                "controller_code_sha256": receipt["code"]["digest"],
                "catalog_runner_code_sha256": receipt["runner_code"]["digest"],
                "runtime_contract_sha256": receipt["runtime_contract"][
                    "runtime_contract_sha256"
                ],
                "checked_at": checked_at,
            }
        block_id = uuid.uuid4().hex
        block_root = resolve_within_root(
            root / "02_finetune" / model_id / "blocked" / block_id,
            root,
            must_exist=False,
        )
        block_root.mkdir(parents=True, exist_ok=False)
        body: dict[str, Any] = {
            "schema_version": "1.0",
            "evidence_type": "finetune_blocked",
            "batch_id": batch_id,
            "model_id": model_id,
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "reason_code": reason_code,
            "component": component,
            "detail": detail,
            "diagnostic": diagnostic,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence = {**body, "evidence_sha256": sha256_json(body)}
        evidence_path = block_root / "blocked_evidence.json"
        atomic_write_json(evidence_path, evidence, root=root, overwrite=False)
        reference = self._reference(root, evidence_path, evidence["evidence_sha256"])
        EventStore(root).append(
            "FINETUNE_BLOCKED", {"model_id": model_id, "evidence": reference}
        )
        return evidence

    @staticmethod
    def _join_runtime_path(root: str, relative: str) -> str:
        if root.startswith("/"):
            return str(PurePosixPath(root) / PurePosixPath(relative))
        return str((Path(root) / Path(relative)).resolve(strict=False))

    def _expected_block_path(
        self,
        receipt: Mapping[str, Any],
        model_id: str,
        reason_code: str,
        component: str,
    ) -> str:
        if reason_code == "missing_code":
            if component.startswith("backend:"):
                role = component.split(":", 1)[1]
                if role not in {"finetune", "evaluation", "verifier"}:
                    raise ControllerValidationError(
                        "missing_code backend component must name finetune, "
                        "evaluation, or verifier"
                    )
                runtime_role = self._runtime_backend(
                    receipt, model_id, role, require_present=False
                )
                if runtime_role.get("backend_state") != "missing":
                    raise ControllerValidationError(
                        f"frozen catalog backend is present: {component}"
                    )
                return str(runtime_role["backend_absolute_path"])
            role = {"runner": "finetune", "verifier": "verifier"}.get(component)
            if role is None:
                raise ControllerValidationError(
                    "missing_code requires component='runner', 'verifier', or "
                    "'backend:<finetune|evaluation|verifier>'"
                )
            runtime_role = self._runtime_role(
                receipt, model_id, role, require_present=False
            )
            if runtime_role.get("state") != "missing":
                raise ControllerValidationError(
                    f"frozen catalog component is present: {component}"
                )
            return runtime_role["absolute_path"]
        if reason_code not in {"missing_path", "missing_weight"} or not component.startswith(
            "pretrained:"
        ):
            raise ControllerValidationError(
                "missing_path/missing_weight require component=pretrained:<registered-role>"
            )
        role = component.split(":", 1)[1]
        artifact = next(
            (item for item in self.registry.pretrained_artifacts[model_id] if item.role == role),
            None,
        )
        if artifact is None:
            raise ControllerValidationError(
                f"{component!r} is not a registered pretrained artifact for {model_id}"
            )
        frozen = next(
            item
            for item in self._model_pretrained_assets(receipt, model_id)
            if item["role"] == role
        )
        return frozen["path"]

    def _reference(self, root: Path, path: Path, content_sha256: str) -> dict[str, str]:
        resolved = resolve_within_root(path, root, must_exist=True)
        return {
            "path": resolved.relative_to(root).as_posix(),
            "file_sha256": sha256_file(resolved),
            "content_sha256": content_sha256,
        }

    def _load_reference(self, root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(reference, Mapping) or set(reference) != {
            "path", "file_sha256", "content_sha256"
        }:
            raise ControllerValidationError("evidence reference schema is invalid")
        path = resolve_within_root(reference["path"], root, must_exist=True)
        if not path.is_file() or sha256_file(path) != reference["file_sha256"]:
            raise ControllerValidationError("evidence file changed after transition")
        value = load_json_strict(path)
        if not isinstance(value, Mapping):
            raise ControllerValidationError("evidence must be a JSON object")
        evidence = dict(value)
        if evidence.get("evidence_sha256") != reference["content_sha256"]:
            raise ControllerValidationError("evidence content reference mismatch")
        if evidence.get("evidence_sha256") != sha256_json(_evidence_body(evidence)):
            raise ControllerValidationError("evidence internal hash mismatch")
        return evidence

    def _validate_finetune_evidence(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        model_id: str,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = self._load_reference(root, reference)
        if (
            evidence.get("evidence_type") != "fresh_finetune"
            or evidence.get("batch_id") != receipt["batch_id"]
            or evidence.get("model_id") != model_id
            or evidence.get("batch_receipt_sha256") != receipt["receipt_sha256"]
            or evidence.get("fresh_current_batch") is not True
        ):
            raise ControllerValidationError("finetune evidence identity/freshness mismatch")
        attempt_root, attempt = self._load_attempt(
            root,
            model_id=model_id,
            stage="finetune",
            attempt_id=evidence.get("attempt_id"),
        )
        if attempt["attempt_sha256"] != evidence.get("attempt_sha256"):
            raise ControllerValidationError("finetune attempt receipt changed")
        manifest_path = resolve_within_root(
            evidence.get("run_manifest_path"), attempt_root, must_exist=True
        )
        if sha256_file(manifest_path) != evidence.get("run_manifest_sha256"):
            raise ControllerValidationError("production run manifest changed after completion")
        (
            _,
            artifact,
            training_reference,
            execution,
            verification,
        ) = self._validate_production_manifest(
            root,
            receipt,
            model_id=model_id,
            attempt_id=evidence["attempt_id"],
            manifest_path=manifest_path,
        )
        if evidence.get("artifact") != artifact:
            raise ControllerValidationError("finetune artifact evidence differs from run manifest")
        if evidence.get("training_receipt") != training_reference:
            raise ControllerValidationError(
                "finetune training receipt evidence differs from run manifest"
            )
        if evidence.get("execution_sha256") != execution["execution_sha256"]:
            raise ControllerValidationError("finetune execution evidence changed")
        if evidence.get("verification_sha256") != verification["verification_sha256"]:
            raise ControllerValidationError("finetune verifier evidence changed")
        return evidence

    def _validate_blocked_evidence(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        model_id: str,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = self._load_reference(root, reference)
        if (
            evidence.get("evidence_type") != "finetune_blocked"
            or evidence.get("batch_id") != receipt["batch_id"]
            or evidence.get("model_id") != model_id
            or evidence.get("batch_receipt_sha256") != receipt["receipt_sha256"]
            or evidence.get("reason_code") not in _BLOCK_REASONS
        ):
            raise ControllerValidationError("blocked evidence identity/reason mismatch")
        component = evidence.get("component")
        diagnostic = evidence.get("diagnostic")
        if not isinstance(component, str) or not isinstance(diagnostic, Mapping):
            raise ControllerValidationError("blocked component/diagnostic is invalid")
        reason = evidence["reason_code"]
        if reason in _MISSING_REASONS:
            expected_path = self._expected_block_path(receipt, model_id, reason, component)
            frozen_state = "missing"
            condition_holds = not Path(expected_path).exists()
            if reason != "missing_code":
                role = component.split(":", 1)[1]
                frozen = next(
                    item
                    for item in self._model_pretrained_assets(receipt, model_id)
                    if item["role"] == role
                )
                frozen_state = frozen["state"]
                condition_holds = frozen_state != "present"
            expected_diagnostic = {
                "diagnostic_type": "registered_missing_component",
                "component": component,
                "expected_path": expected_path,
                "observed_state": frozen_state,
                "frozen_contract_sha256": (
                    receipt["runtime_contract"]["runtime_contract_sha256"]
                    if reason == "missing_code"
                    else receipt["pretrained_assets_sha256"]
                ),
                "checked_at": diagnostic.get("checked_at"),
            }
            if dict(diagnostic) != expected_diagnostic or not condition_holds:
                raise ControllerValidationError("registered missing-component condition no longer holds")
            self._timestamp(diagnostic["checked_at"], "blocked.checked_at")
        elif reason == "unrecoverable_provenance":
            expected_diagnostic = {
                "diagnostic_type": "unavailable_verified_multi_root_bootstrap",
                "component": "verified-multi-root-bootstrap",
                "controller_code_sha256": receipt["code"]["digest"],
                "catalog_runner_code_sha256": receipt["runner_code"]["digest"],
                "runtime_contract_sha256": receipt["runtime_contract"][
                    "runtime_contract_sha256"
                ],
                "checked_at": diagnostic.get("checked_at"),
            }
            if (
                component != "verified-multi-root-bootstrap"
                or dict(diagnostic) != expected_diagnostic
                or self._controller_verified_multi_root_bootstrap(
                    role="finetune", purpose="production"
                )
            ):
                raise ControllerValidationError(
                    "verified multi-root bootstrap blocker no longer holds"
                )
            self._timestamp(diagnostic["checked_at"], "blocked.checked_at")
        return evidence

    def validate_batch(self, batch_id: str) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        audited_receipt = load_and_validate_batch_receipt(
            root / "00_inputs" / "batch_receipt.json",
            verify_pretrained_content=True,
        )
        if audited_receipt != receipt:
            raise ControllerValidationError(
                "explicit batch audit differs from trusted transition index"
            )
        state = EventStore(root).load()
        if state["batch_receipt_sha256"] != receipt["receipt_sha256"]:
            raise ControllerValidationError("state/batch receipt mismatch")
        for model_id, model_state in state["models"].items():
            status = model_state["finetune_status"]
            if status == "finetune_complete":
                self._validate_finetune_evidence(
                    root, receipt, model_id, model_state["finetune_evidence"]
                )
            elif status == "blocked":
                self._validate_blocked_evidence(
                    root, receipt, model_id, model_state["finetune_evidence"]
                )
            elif status != "pending":
                raise ControllerValidationError(f"unknown finetune status: {status}")
            for size in (1, 8, 32):
                if model_state["smoke"][str(size)] == "passed":
                    self._validate_evaluation_evidence(
                        root,
                        receipt,
                        model_id,
                        f"smoke_{size}",
                        model_state["smoke_evidence"][str(size)],
                    )
            if model_state["full_status"] == "complete":
                self._validate_evaluation_evidence(
                    root, receipt, model_id, "full", model_state["full_evidence"]
                )
        return state

    def open_evaluation(self, batch_id: str) -> dict[str, Any]:
        root, _ = self._receipt(batch_id)
        state = self.validate_batch(batch_id)
        if any(
            model["finetune_status"] not in {"finetune_complete", "blocked"}
            for model in state["models"].values()
        ):
            raise ControllerValidationError("global finetune barrier is not terminal")
        eval_root = root / "03_eval"
        if eval_root.exists():
            raise ControllerValidationError("unmanaged eval directory exists before gate transition")
        new_state = EventStore(root).append("EVAL_OPENED", {})
        eval_root.mkdir(exist_ok=False)
        for model_id, model_state in new_state["models"].items():
            if model_state["finetune_status"] == "finetune_complete":
                (eval_root / model_id).mkdir()
        return new_state

    def _expected_items(self, receipt: Mapping[str, Any], stage: str):
        benchmark = load_benchmark(receipt["inputs"]["benchmark"]["path"])
        if stage == "full":
            return benchmark
        if stage.startswith("smoke_"):
            return smoke_items(benchmark, int(stage.split("_", 1)[1]))
        raise ControllerValidationError(f"invalid evaluation stage: {stage}")

    def _validate_prediction_rows(
        self,
        path: Path,
        *,
        batch_id: str,
        model: ModelSpec,
        expected_items,
        smoke: bool,
    ) -> tuple[list[PredictionRow], dict[str, Any]]:
        serialized = load_jsonl_strict(path)
        if len(serialized) != len(expected_items):
            raise ControllerValidationError(
                f"prediction denominator mismatch: expected {len(expected_items)}, got {len(serialized)}"
            )
        rows: list[PredictionRow] = []
        errors = {code.value: 0 for code in EvaluationErrorCode if code is not EvaluationErrorCode.NONE}
        correct = 0
        for index, (raw, expected) in enumerate(zip(serialized, expected_items, strict=True)):
            try:
                row = prediction_row_from_dict(raw)
            except Exception as exc:
                raise ControllerValidationError(f"prediction row {index + 1} is invalid: {exc}") from exc
            if (
                row.batch_id != batch_id
                or row.model_id != model.model_id
                or row.sample_id != expected.sample_id
                or row.group_id != expected.group_id
                or row.gold != expected.gold
                or row.modality != model.modality
                or row.KIND != model.prediction_kind
            ):
                raise ControllerValidationError(f"prediction row {index + 1} identity/modality mismatch")
            if row.error_code is not EvaluationErrorCode.NONE:
                errors[row.error_code.value] += 1
                if smoke:
                    raise ControllerValidationError(
                        "smoke requires every row to be a strict valid prediction with no errors"
                    )
            if row.correct:
                correct += 1
            rows.append(row)
        if len({row.sample_id for row in rows}) != len(rows):
            raise ControllerValidationError("duplicate prediction sample_id")
        denominator = len(rows)
        summary = {
            "correct": correct,
            "denominator": denominator,
            "accuracy": correct / denominator,
            "errors": errors,
        }
        return rows, summary

    def complete_evaluation(
        self,
        batch_id: str,
        *,
        model_id: str,
        stage: str,
        attempt_id: str,
        predictions_path: str | Path,
    ) -> dict[str, Any]:
        model = self._model(model_id)
        root, receipt = self._receipt(batch_id)
        self._require_verified_multi_root_bootstrap(
            role="evaluation", purpose="production"
        )
        state = EventStore(root).load()
        if not state["eval_open"] or state["models"][model_id]["finetune_status"] != "finetune_complete":
            raise ControllerValidationError("model is not eligible for evaluation")
        if stage == "full" and not state["full_open"]:
            raise ControllerValidationError("full gate is closed")
        if stage.startswith("smoke_") and state["full_open"]:
            raise ControllerValidationError("smoke gate is closed")
        attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage=stage, attempt_id=attempt_id
        )
        predictions = resolve_within_root(predictions_path, attempt_root, must_exist=True)
        if not predictions.is_file():
            raise ControllerValidationError("predictions must be a JSONL file")
        execution = self._load_execution(
            root,
            attempt_root,
            attempt,
            batch_id=batch_id,
            model_id=model_id,
            stage=stage,
            attempt_id=attempt_id,
        )
        if execution["output_path"] != str(predictions):
            raise ControllerValidationError("execution did not attest this predictions file")
        expected = self._expected_items(receipt, stage)
        _, summary = self._validate_prediction_rows(
            predictions,
            batch_id=batch_id,
            model=model,
            expected_items=expected,
            smoke=stage != "full",
        )
        finetune_reference = state["models"][model_id]["finetune_evidence"]
        self._validate_finetune_evidence(root, receipt, model_id, finetune_reference)
        body: dict[str, Any] = {
            "schema_version": "1.0",
            "evidence_type": "evaluation",
            "batch_id": batch_id,
            "model_id": model_id,
            "stage": stage,
            "attempt_id": attempt_id,
            "batch_receipt_sha256": receipt["receipt_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "finetune_evidence_sha256": finetune_reference["content_sha256"],
            "media_manifest_sha256": receipt["inputs"]["media_manifest"]["digest"],
            "predictions_path": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "prediction_kind": model.prediction_kind,
            "modality": model.modality.value,
            "summary": summary,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence = {**body, "evidence_sha256": sha256_json(body)}
        evidence_path = attempt_root / "evaluation_evidence.json"
        atomic_write_json(evidence_path, evidence, root=root, overwrite=False)
        reference = self._reference(root, evidence_path, evidence["evidence_sha256"])
        if stage == "full":
            EventStore(root).append(
                "FULL_COMPLETE",
                {"model_id": model_id, "attempt_id": attempt_id, "evidence": reference},
            )
        else:
            EventStore(root).append(
                "SMOKE_PASSED",
                {
                    "model_id": model_id,
                    "size": int(stage.split("_", 1)[1]),
                    "attempt_id": attempt_id,
                    "evidence": reference,
                },
            )
        return evidence

    def _validate_evaluation_evidence(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        model_id: str,
        stage: str,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        model = self._model(model_id)
        evidence = self._load_reference(root, reference)
        if (
            evidence.get("evidence_type") != "evaluation"
            or evidence.get("batch_id") != receipt["batch_id"]
            or evidence.get("model_id") != model_id
            or evidence.get("stage") != stage
            or evidence.get("batch_receipt_sha256") != receipt["receipt_sha256"]
            or evidence.get("prediction_kind") != model.prediction_kind
            or evidence.get("modality") != model.modality.value
            or evidence.get("media_manifest_sha256")
            != receipt["inputs"]["media_manifest"]["digest"]
        ):
            raise ControllerValidationError("evaluation evidence identity mismatch")
        attempt_root, attempt = self._load_attempt(
            root, model_id=model_id, stage=stage, attempt_id=evidence.get("attempt_id")
        )
        if attempt["attempt_sha256"] != evidence.get("attempt_sha256"):
            raise ControllerValidationError("evaluation attempt receipt changed")
        execution = self._load_execution(
            root,
            attempt_root,
            attempt,
            batch_id=receipt["batch_id"],
            model_id=model_id,
            stage=stage,
            attempt_id=evidence["attempt_id"],
        )
        if execution["execution_sha256"] != evidence.get("execution_sha256"):
            raise ControllerValidationError("evaluation execution evidence changed")
        predictions = resolve_within_root(evidence.get("predictions_path"), attempt_root, must_exist=True)
        if sha256_file(predictions) != evidence.get("predictions_sha256"):
            raise ControllerValidationError("predictions changed after evaluation completion")
        if execution["output_path"] != str(predictions):
            raise ControllerValidationError("execution did not attest current predictions")
        expected = self._expected_items(receipt, stage)
        _, summary = self._validate_prediction_rows(
            predictions,
            batch_id=receipt["batch_id"],
            model=model,
            expected_items=expected,
            smoke=stage != "full",
        )
        if summary != evidence.get("summary"):
            raise ControllerValidationError("evaluation summary does not match prediction rows")
        state = EventStore(root).replay()
        finetune_reference = state["models"][model_id]["finetune_evidence"]
        if evidence.get("finetune_evidence_sha256") != finetune_reference["content_sha256"]:
            raise ControllerValidationError("evaluation did not use current-batch finetune evidence")
        self._validate_finetune_evidence(root, receipt, model_id, finetune_reference)
        return evidence

    def open_full_evaluation(self, batch_id: str) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        state = self.validate_batch(batch_id)
        if not state["eval_open"]:
            raise ControllerValidationError("evaluation phase is closed")
        for model_id, model_state in state["models"].items():
            if model_state["finetune_status"] == "finetune_complete":
                for size in (1, 8, 32):
                    if model_state["smoke"][str(size)] != "passed":
                        raise ControllerValidationError(
                            f"{model_id} has not passed smoke 1/8/32"
                        )
        full_paths = [
            root / "03_eval" / model_id / "full"
            for model_id, model_state in state["models"].items()
            if model_state["finetune_status"] == "finetune_complete"
        ]
        if any(path.exists() for path in full_paths):
            raise ControllerValidationError("unmanaged full-eval directory exists before gate transition")
        new_state = EventStore(root).append("FULL_OPENED", {})
        for model_id, model_state in new_state["models"].items():
            if model_state["finetune_status"] == "finetune_complete":
                (root / "03_eval" / model_id / "full").mkdir(parents=True, exist_ok=False)
        return new_state

    def _release_sources(
        self, root: Path, receipt: Mapping[str, Any], state: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for model in self.registry.models:
            model_state = state["models"][model.model_id]
            if model_state["finetune_status"] == "blocked":
                evidence = self._validate_blocked_evidence(
                    root, receipt, model.model_id, model_state["finetune_evidence"]
                )
                blocked.append(
                    {
                        "model_id": model.model_id,
                        "display_name": model.display_name,
                        "modality": model.modality.value,
                        "reason_code": evidence["reason_code"],
                        "evidence_sha256": evidence["evidence_sha256"],
                    }
                )
                continue
            if model_state["finetune_status"] != "finetune_complete" or model_state["full_status"] != "complete":
                raise ControllerValidationError("release has unfinished models")
            evidence = self._validate_evaluation_evidence(
                root, receipt, model.model_id, "full", model_state["full_evidence"]
            )
            summary = evidence["summary"]
            if summary["denominator"] != 500:
                raise ControllerValidationError("release denominator must be exactly 500")
            errors = summary["errors"]
            results.append(
                {
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "modality": model.modality.value,
                    "evaluation_mode": model.evaluation_mode,
                    "correct": summary["correct"],
                    "denominator": 500,
                    "accuracy": summary["correct"] / 500,
                    "invalid_output": errors["invalid_output"],
                    "media_error": errors["media_error"],
                    "timeout": errors["timeout"],
                    "oom": errors["oom"],
                    "runtime_error": errors["runtime_error"],
                    "predictions_sha256": evidence["predictions_sha256"],
                    "evaluation_evidence_sha256": evidence["evidence_sha256"],
                }
            )
        return results, blocked

    def build_release(self, batch_id: str) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        state = self.validate_batch(batch_id)
        if not state["full_open"] or state["release_status"] != "pending":
            raise ControllerValidationError("release gate is closed or already used")
        results, blocked = self._release_sources(root, receipt, state)
        release_root = root / "04_release"
        if release_root.exists():
            raise ControllerValidationError("release directory already exists; refusing overwrite")
        manifest = build_release_files(
            release_root,
            batch_root=root,
            batch_id=batch_id,
            batch_receipt_sha256=receipt["receipt_sha256"],
            model_results=results,
            blocked_models=blocked,
        )
        manifest_path = release_root / "evaluation_release_manifest.json"
        reference = {
            "path": manifest_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_sha256"],
        }
        EventStore(root).append("RELEASE_BUILT", {"evidence": reference})
        self.verify_release(batch_id)
        return manifest

    def verify_release(self, batch_id: str) -> dict[str, Any]:
        root, receipt = self._receipt(batch_id)
        state = self.validate_batch(batch_id)
        if state["release_status"] != "built":
            raise ControllerValidationError("release has not been built")
        reference = state.get("release_evidence")
        if not isinstance(reference, Mapping):
            raise ControllerValidationError("release event reference is missing")
        manifest_path = resolve_within_root(reference["path"], root, must_exist=True)
        if sha256_file(manifest_path) != reference.get("file_sha256"):
            raise ControllerValidationError("release manifest file changed")
        value = load_json_strict(manifest_path)
        if not isinstance(value, Mapping):
            raise ControllerValidationError("release manifest is invalid")
        manifest = dict(value)
        body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != sha256_json(body):
            raise ControllerValidationError("release manifest internal hash mismatch")
        if manifest["manifest_sha256"] != reference.get("content_sha256"):
            raise ControllerValidationError("release event/manifest hash mismatch")
        if (
            manifest.get("batch_id") != batch_id
            or manifest.get("batch_receipt_sha256") != receipt["receipt_sha256"]
            or manifest.get("policy", {}).get("historical_results_allowed") is not False
            or manifest.get("policy", {}).get("proxy_results_allowed") is not False
        ):
            raise ControllerValidationError("release policy/batch binding mismatch")
        results, blocked = self._release_sources(root, receipt, state)
        if manifest.get("models") != results or manifest.get("blocked_models") != blocked:
            raise ControllerValidationError("release sources differ from current evidence")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or set(files) != {
            "all_models_results.csv", "all_models_results.md", "blocked_models.md"
        }:
            raise ControllerValidationError("release file inventory is invalid")
        for name, digest in files.items():
            path = resolve_within_root(root / "04_release" / name, root, must_exist=True)
            if sha256_file(path) != digest:
                raise ControllerValidationError(f"release file changed: {name}")
        return manifest
