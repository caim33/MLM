"""Task-8 MVP runner: config-driven ms-swift GRPO launcher for Motion-r1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shlex
import secrets
import shutil
import subprocess
import sys
import unicodedata
from netrc import NetrcParseError, netrc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional until config parsing
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motionllm.grpo import (  # noqa: E402
    COLOCATION_CONFIG_ENV,
    COLOCATION_DATASET_ENV,
    COLOCATION_ENV_KEYS,
    COLOCATION_NONCE_ENV,
    COLOCATION_PATH_ENV,
    COLOCATION_PLAN_ENV,
    RewardBranch,
    RewardMetadata,
    describe_environment_overrides,
    initialize_runtime_colocation_plan,
    redact_command_for_log,
    validate_runtime_colocation_receipt,
    validate_semantic_reference,
)
from motionllm.grpo.checkpoint_binding import (  # noqa: E402
    ADDITIONAL_CONFIG_NAME,
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    AdapterCheckpointBinding,
    CheckpointBindingError,
    adapter_config_critical_fields,
    capture_adapter_checkpoint,
    extract_peft_adapter_state,
    live_peft_adapter_config_semantics,
    require_identical_adapter_configs,
    require_identical_adapter_states,
)
from motionllm.training import (  # noqa: E402
    ArtifactValidationError,
    ReloadVerificationReceipt,
    default_model_factory,
    setup_motion_tokens,
    validate_artifact_policy,
    validate_lora_adapter_pairs,
    verify_processor_save_reload,
    verify_motion_tokens,
    verify_state_mapping_reload,
    write_artifact_from_arguments,
    write_reload_verification_receipt,
)
from motion_eval.core import hash_path, resolve_within_root  # noqa: E402
DEFAULT_MAX_COMPLETION_LENGTH = 1024
DEFAULT_FORBIDDEN_CONTEXT_LIMIT_ENV_KEYS = (
    "VIDEO_MAX_TOKEN_NUM",
    "VIDEO_MIN_TOKEN_NUM",
    "IMAGE_MAX_TOKEN_NUM",
    "IMAGE_MIN_TOKEN_NUM",
    "FPS_MAX_FRAMES",
    "FPS_MIN_FRAMES",
    "MAX_RATIO",
    "FRAME_FACTOR",
)
DEFAULT_WANDB_PROJECT = "motion-r1-grpo"
FORMAL_BOUND_INPUTS_ENV = "MOTION_GRPO_FORMAL_BOUND_INPUTS"
FORMAL_PYTHON_ENV = "MOTION_GRPO_PYTHON"
FORMAL_LAUNCHER_ENV = "MOTION_GRPO_FORMAL_LAUNCHER"
FORMAL_TRAINING_NONCE_ENV = "MOTION_GRPO_TRAINING_NONCE"
FORMAL_TRAINING_BATCH_ENV = "MOTION_GRPO_BATCH_ID"
FORMAL_TRAINING_STEPS_ENV = "MOTION_GRPO_EXPECTED_OPTIMIZER_STEPS"
FORMAL_TRAINING_ARTIFACT_ENV = "MOTION_GRPO_ARTIFACT_PATH"
FORMAL_TRAINING_CALLBACK = "motion_training_receipt"
FORMAL_TRAINING_RECEIPT_SCHEMA = "motionllm.grpo.training_update.v2"
FORMAL_RUNTIME_SCHEMA = "motionllm.grpo.runtime.v1"
FORMAL_MODEL_REGISTRY_ID = "motionr1_vm_lora"
FORMAL_MODEL_FAMILY = "qwen3_vl_motion"
FORMAL_MODEL_TYPE = "motion_r1_qwen3_vl_motion"
FORMAL_PURPOSE = "formal_finetune"
FORMAL_LEAKAGE_SCHEMA = "motionllm.grpo.leakage_audit.v1"
FORMAL_NORMALIZATION_ALGORITHM = "nfkc_casefold_whitespace_v1"
FORMAL_REGISTRY_PATH = REPO_ROOT / "model_evaluation_agent" / "model_registry.json"
FORMAL_PRETRAINED_REGISTRY_PATH = (
    REPO_ROOT / "model_evaluation_agent" / "pretrained_registry.json"
)
FORMAL_PRETRAINED_INVENTORY_PATH = (
    REPO_ROOT
    / "model_evaluation_agent"
    / "server_audit"
    / "20260730_pretrained_inventory.json"
)
FORMAL_RUNTIME_CONTRACT_PATH = (
    REPO_ROOT / "qwenvl" / "grpo_ms_swift" / "runtime" / "grpo_api_contract.json"
)
FORMAL_EXTERNAL_PLUGIN_PATH = (
    REPO_ROOT / "qwenvl" / "grpo_ms_swift" / "plugins" / "swift_external_rewards.py"
)
FORMAL_CUSTOM_REGISTER_PATH = (
    REPO_ROOT / "qwenvl" / "grpo_ms_swift" / "model_register" / "motion_model.py"
)
_FORMAL_CODE_PATHS = (
    Path(__file__).resolve(),
    FORMAL_EXTERNAL_PLUGIN_PATH,
    FORMAL_CUSTOM_REGISTER_PATH,
    FORMAL_RUNTIME_CONTRACT_PATH,
    FORMAL_REGISTRY_PATH,
    FORMAL_PRETRAINED_REGISTRY_PATH,
    FORMAL_PRETRAINED_INVENTORY_PATH,
    REPO_ROOT / "scripts" / "train_grpo_ms_swift.sh",
    REPO_ROOT / "requirements-grpo.lock",
    REPO_ROOT / "models",
    # Freeze complete package trees, not a hand-maintained approximation of
    # their transitive import closure.  ``-B`` and the child environment below
    # prevent runtime pycache generation from destabilizing these snapshots.
    REPO_ROOT / "src" / "motionllm",
    REPO_ROOT / "src" / "motion_eval",
)
_FORMAL_DANGEROUS_ENV_KEYS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "LD_PRELOAD",
        "BASH_ENV",
        "ENV",
        "PATH",
    }
)
_FORMAL_RECORD_COMMON_KEYS = frozenset(
    {
        "messages",
        "video_sha256",
        "solution",
        "answer",
        "group_id",
        "branch",
        "sample_id",
        "rollout_id",
    }
)
_FORMAL_REGISTRY_IDS = frozenset(
    {
        "qwen36_27b_lora",
        "motionr1_vm_lora",
        "qwen3vl_8b_lora",
        "qwen3vl_4b_lora",
        "qwen35_4b_lora",
        "videollava_7b_lora",
        "videochatgpt_lora",
        "videochat2_lora",
        "videollama_trainables",
        "videollama_lora",
        "mplug_owl_video_lora",
        "otter_video_lora",
        "agcn_official",
        "motionclip_official",
        "motionllm_official",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")
_FORMAL_VQ_ENV_KEYS = frozenset(
    {
        "VQVAE_PATH",
        "MOTION_DATANAME",
        "MOTION_QUANTIZER",
        "VQVAE_NB_CODE",
        "VQVAE_CODE_DIM",
        "VQVAE_OUTPUT_EMB_WIDTH",
        "VQVAE_DOWN_T",
        "VQVAE_STRIDE_T",
        "VQVAE_WIDTH",
        "VQVAE_DEPTH",
        "VQVAE_DILATION_GROWTH_RATE",
        "VQVAE_ACTIVATION",
        "VQVAE_NORM",
    }
)


def _resolve_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("${REPO_ROOT}", str(REPO_ROOT))
    if isinstance(value, list):
        return [_resolve_placeholders(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_placeholders(v) for v in value)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v) for k, v in value.items()}
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load GRPO configs. Install with: pip install pyyaml")
    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> Dict[Any, Any]:
        if getattr(node, "tag", None) == "tag:yaml.org,2002:merge" or any(
            getattr(key_node, "value", None) == "<<" for key_node, _ in node.value
        ):
            raise ValueError("YAML merge keys are forbidden in GRPO configs")
        result: Dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML mapping key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=_UniqueKeyLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return _resolve_placeholders(raw)


def _append_cli_arg(command: List[str], key: str, value: Any) -> None:
    if value is None:
        return

    flag = f"--{key}"
    if isinstance(value, bool):
        command.extend([flag, "true" if value else "false"])
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        command.append(flag)
        command.extend(str(v) for v in value)
        return
    if isinstance(value, Mapping):
        command.extend([flag, json.dumps(value, ensure_ascii=False)])
        return
    command.extend([flag, str(value)])


def _get_section(config: Mapping[str, Any], key: str) -> MutableMapping[str, Any]:
    section = config.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, MutableMapping):
        raise ValueError(f"`{key}` section must be a mapping, got: {type(section)}")
    return dict(section)


def _apply_section(
    command: List[str],
    section: Mapping[str, Any],
    keys: Iterable[str],
) -> None:
    for key in keys:
        if key in section:
            _append_cli_arg(command, key, section[key])


def _build_command(
    config: Mapping[str, Any], *, swift_executable: str = "swift"
) -> List[str]:
    run_cfg = _get_section(config, "run")
    model_cfg = _get_section(config, "model")
    data_cfg = _get_section(config, "data")
    train_cfg = _get_section(config, "training")
    grpo_cfg = _get_section(config, "grpo")
    reward_cfg = _get_section(config, "rewards")
    plugin_cfg = _get_section(config, "plugins")

    # User requirement: inference/output length default is 1024.
    if "max_completion_length" not in grpo_cfg:
        grpo_cfg["max_completion_length"] = DEFAULT_MAX_COMPLETION_LENGTH

    command = [swift_executable, "rlhf", "--rlhf_type", "grpo"]

    _apply_section(
        command,
        run_cfg,
        ["output_dir", "run_name", "seed", "report_to", "add_version"],
    )
    _apply_section(
        command,
        model_cfg,
        [
            "model",
            "model_type",
            "torch_dtype",
            "max_length",
            "max_pixels",
            "min_pixels",
            "max_frames",
            "min_frames",
            "video_fps",
            "video_max_frames",
            "video_min_frames",
            "motion_length_divisor",
            "motion_timestamps_sync_with_video",
            "new_special_tokens",
            "model_kwargs",
        ],
    )
    _apply_section(
        command,
        data_cfg,
        [
            "dataset",
            "val_dataset",
            "custom_dataset_info",
            "columns",
            "dataset_num_proc",
            "load_from_cache_file",
            "dataset_shuffle",
            "strict",
            "remove_unused_columns",
        ],
    )
    _apply_section(
        command,
        train_cfg,
        [
            "tuner_type",
            "tuner_backend",
            "use_swift_lora",
            "target_modules",
            "target_regex",
            "target_parameters",
            "modules_to_save",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_bias",
            "use_rslora",
            "use_dora",
            "lora_dtype",
            "lorap_lr_ratio",
            "lorap_emb_lr",
            "freeze_llm",
            "freeze_vit",
            "freeze_aligner",
            "num_train_epochs",
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "logging_steps",
            "save_steps",
            "save_total_limit",
            "save_strategy",
            "save_only_model",
            "save_safetensors",
            "eval_strategy",
            "eval_steps",
            "dataloader_num_workers",
            "bf16",
            "gradient_checkpointing",
            "deepspeed",
            "callbacks",
        ],
    )
    _apply_section(
        command,
        grpo_cfg,
        [
            "use_vllm",
            "vllm_mode",
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
            "vllm_pipeline_parallel_size",
            "vllm_enable_expert_parallel",
            "vllm_max_num_seqs",
            "vllm_max_model_len",
            "vllm_disable_custom_all_reduce",
            "vllm_enforce_eager",
            "vllm_limit_mm_per_prompt",
            "vllm_max_lora_rank",
            "vllm_enable_prefix_caching",
            "vllm_use_async_engine",
            "vllm_quantization",
            "vllm_reasoning_parser",
            "vllm_disable_cascade_attn",
            "vllm_mm_processor_cache_gb",
            "vllm_speculative_config",
            "vllm_engine_kwargs",
            "vllm_data_parallel_size",
            "vllm_enable_lora",
            "vllm_server_base_url",
            "vllm_server_host",
            "vllm_server_port",
            "vllm_server_timeout",
            "vllm_server_group_port",
            "enable_flattened_weight_sync",
            "vllm_server_pass_dataset",
            "async_generate",
            "sleep_level",
            "move_model_batches",
            "offload_optimizer",
            "offload_model",
            "steps_per_generation",
            "generation_batch_size",
            "num_generations",
            "num_generations_eval",
            "num_iterations",
            "max_completion_length",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "stop_words",
            "beta",
            "epsilon",
            "epsilon_high",
            "advantage_estimator",
            "log_completions",
            "log_entropy",
            "top_entropy_quantile",
        ],
    )
    _apply_section(command, reward_cfg, ["reward_funcs", "reward_weights"])
    _apply_section(command, plugin_cfg, ["external_plugins", "custom_register_path"])

    return command


def _validate_required(config: Mapping[str, Any]) -> None:
    run_cfg = _get_section(config, "run")
    model_cfg = _get_section(config, "model")
    data_cfg = _get_section(config, "data")
    reward_cfg = _get_section(config, "rewards")
    if "model" not in model_cfg:
        raise ValueError("`model.model` is required.")
    if "dataset" not in data_cfg:
        raise ValueError("`data.dataset` is required.")
    if "reward_funcs" not in reward_cfg:
        raise ValueError("`rewards.reward_funcs` is required.")
    if "seed" not in run_cfg:
        raise ValueError("`run.seed` is required as the single deterministic GRPO seed.")
    seed = run_cfg["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("`run.seed` must be an integer in [0, 2**32-1].")


def _validate_grpo_training_mode(
    config: Mapping[str, Any], *, config_path: Path | None = None
) -> None:
    """Keep historical full-GRPO configs outside the publishable runner."""

    if config_path is not None:
        inventory_path = REPO_ROOT / "configs" / "grpo" / "legacy_full_inventory.json"
        inventory = _strict_json(
            inventory_path.read_text(encoding="utf-8"), location=str(inventory_path)
        )
        if (
            not isinstance(inventory, Mapping)
            or inventory.get("status") != "quarantined_non_formal"
        ):
            raise ValueError("legacy GRPO quarantine inventory is invalid")
        names = inventory.get("configs")
        digests = inventory.get("sha256")
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) for name in names)
            or not isinstance(digests, Mapping)
            or set(digests) != set(names)
        ):
            raise ValueError("legacy GRPO quarantine inventory lacks exact hashes")
        observed_digest = hash_path(config_path, symlink_policy="reject").digest
        canonical_parent = (
            REPO_ROOT
            / "legacy"
            / "refactor_snapshot"
            / "configs"
            / "grpo_personal"
        ).resolve(strict=True)
        if (
            config_path.parent.resolve(strict=True) == canonical_parent
            and config_path.name in names
        ):
            if digests.get(config_path.name) != observed_digest:
                raise ValueError("quarantined legacy GRPO config bytes differ from inventory")
            raise ValueError("quarantined legacy full-GRPO config cannot use the formal runner")
        if observed_digest in set(str(value) for value in digests.values()):
            raise ValueError(
                "copied quarantined legacy full-GRPO config cannot use the formal runner"
            )

    tuner_type = (
        str(_get_section(config, "training").get("tuner_type", ""))
        .strip()
        .casefold()
    )
    if tuner_type == "full":
        raise ValueError(
            "legacy full-GRPO configs are quarantined and cannot be launched by this runner; "
            "formal full publication remains disabled. Copy the formal LoRA template and "
            "produce a fresh current-batch adapter instead"
        )
    if tuner_type not in {"lora", "peft"}:
        raise ValueError("GRPO training.training.tuner_type must be explicitly lora or peft")


def _reward_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _get_section(config, "rewards").get("reward_funcs", ())
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError("`rewards.reward_funcs` must be a string or list of strings")
    names = tuple(str(value).strip() for value in values if str(value).strip())
    if not names:
        raise ValueError("`rewards.reward_funcs` must not be empty")
    return names


def _semantic_reward_enabled(config: Mapping[str, Any]) -> bool:
    if "reward_funcs" not in _get_section(config, "rewards"):
        return False
    return any(name == "motion_semantic" or "semantic" in name.casefold() for name in _reward_names(config))


def _validate_formal_reward_policy(config: Mapping[str, Any]) -> None:
    names = _reward_names(config)
    if len(set(names)) != len(names):
        raise ValueError("formal GRPO reward_funcs must be unique")
    required = {"motion_semantic", "motion_format", "motion_vm_v_bonus"}
    if set(names) != required or len(names) != len(required):
        raise ValueError(
            "formal GRPO reward_funcs must equal exactly motion_semantic, motion_format, "
            "and motion_vm_v_bonus"
        )
    weights = _get_section(config, "rewards").get("reward_weights")
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != len(names):
            raise ValueError("formal GRPO reward_weights must align exactly with reward_funcs")
        for value in weights:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("formal GRPO reward weights must be finite positive numbers")
            parsed = float(value)
            if not (parsed > 0.0 and parsed < float("inf")):
                raise ValueError("formal GRPO reward weights must be finite positive numbers")


def _enforce_formal_ambient_environment(config: Mapping[str, Any]) -> None:
    """Forbid ambient variables that can silently change data/model/reward behavior."""

    configured = _get_section(config, "run").get("env", {}) or {}
    if not isinstance(configured, Mapping):
        raise ValueError("run.env must be a mapping")
    configured_values = sorted(str(key) for key, value in configured.items() if value is not None)
    if configured_values:
        raise ValueError(
            "formal GRPO run.env may only explicitly unset inherited variables; "
            f"configured values are forbidden: {configured_values}"
        )
    effective = dict(os.environ)
    for key, value in configured.items():
        name = str(key)
        if value is None:
            effective.pop(name, None)
        else:
            effective[name] = str(value)
    active = sorted(
        name
        for name, value in effective.items()
        if value not in (None, "")
        and (
            name in _FORMAL_VQ_ENV_KEYS
            or name == FORMAL_BOUND_INPUTS_ENV
            or name in {
                FORMAL_TRAINING_NONCE_ENV,
                FORMAL_TRAINING_BATCH_ENV,
                FORMAL_TRAINING_STEPS_ENV,
                FORMAL_TRAINING_ARTIFACT_ENV,
            }
            or name in (_FORMAL_DANGEROUS_ENV_KEYS - {"PATH"})
            or name.startswith("GROUP_NUM_")
            or name.startswith("MOTION_GRPO_VM_V_")
            or name.startswith("MOTION_GRPO_QA_RUBRIC_")
            or name.startswith("MOTION_GRPO_MOTION_RUBRIC_V2_")
            or name == "MOTION_GRPO_DEBUG"
        )
    )
    if active:
        raise ValueError(
            "formal GRPO forbids ambient VQ/data/reward overrides; unset: "
            f"{active}"
        )


def _strict_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(raw: str, *, location: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid strict JSON at {location}: {exc}") from exc


def _reject_formal_config_secrets(config: Mapping[str, Any]) -> None:
    """Reject credentials and command-injection environment knobs in formal YAML."""

    sensitive_keys = {
        "password",
        "passwd",
        "secret",
        "api_key",
        "token",
        "access_token",
        "private_key",
        "wandb_api_key",
        "hf_token",
    }
    credential_uri = re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/@\s]+@", re.I)

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                name = str(key).strip().casefold()
                if name in sensitive_keys:
                    raise ValueError(
                        f"formal GRPO config must not contain credential field {location}.{key}"
                    )
                walk(nested, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, f"{location}[{index}]")
        elif isinstance(value, str) and (
            credential_uri.search(value) or value.lstrip().casefold().startswith("bearer ")
        ):
            raise ValueError(f"formal GRPO config contains credential-like text at {location}")

    walk(config, "config")


def _validate_formal_model_identity(config: Mapping[str, Any]) -> str:
    """Hard-bind this formal runner to the one registry model it implements."""

    run_cfg = _get_section(config, "run")
    model_cfg = _get_section(config, "model")
    if run_cfg.get("model_registry_id") != FORMAL_MODEL_REGISTRY_ID:
        raise ValueError(
            f"formal GRPO model_registry_id must be {FORMAL_MODEL_REGISTRY_ID!r}"
        )
    if model_cfg.get("model_family") != FORMAL_MODEL_FAMILY:
        raise ValueError(f"formal GRPO model_family must be {FORMAL_MODEL_FAMILY!r}")
    if model_cfg.get("model_type") != FORMAL_MODEL_TYPE:
        raise ValueError(f"formal GRPO model_type must be {FORMAL_MODEL_TYPE!r}")

    payload = _strict_json(
        FORMAL_REGISTRY_PATH.read_text(encoding="utf-8"),
        location=str(FORMAL_REGISTRY_PATH),
    )
    if not isinstance(payload, Mapping):
        raise ValueError("canonical model registry must be a JSON object")
    if payload.get("fresh_finetune_required_per_batch") is not True:
        raise ValueError("canonical model registry must require fresh per-batch finetuning")
    if payload.get("global_finetune_barrier_before_eval") is not True:
        raise ValueError("canonical model registry must require the global finetune barrier")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("canonical model registry models must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(models):
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError(f"canonical model registry models[{index}] is invalid")
        model_id = str(item["id"])
        if model_id in by_id:
            raise ValueError(f"canonical model registry has duplicate id {model_id!r}")
        by_id[model_id] = item
    if set(by_id) != set(_FORMAL_REGISTRY_IDS):
        raise ValueError("canonical model registry IDs differ from the frozen formal contract")
    entry = by_id[FORMAL_MODEL_REGISTRY_ID]
    if entry.get("main_modality") != "VM":
        raise ValueError("motionr1_vm_lora registry modality must be VM")
    if str(entry.get("finetune_kind", "")).casefold() != "lora":
        raise ValueError("motionr1_vm_lora registry finetune_kind must be LoRA")
    if entry.get("evaluation_mode") != "generative":
        raise ValueError("motionr1_vm_lora registry evaluation_mode must be generative")
    return hash_path(FORMAL_REGISTRY_PATH, symlink_policy="reject").digest


def _validate_canonical_pretrained_assets(arguments: SimpleNamespace) -> None:
    """Bind base/VQ bytes to the canonical pretrained registry and inventory."""

    registry = _strict_json(
        FORMAL_PRETRAINED_REGISTRY_PATH.read_text(encoding="utf-8"),
        location=str(FORMAL_PRETRAINED_REGISTRY_PATH),
    )
    inventory = _strict_json(
        FORMAL_PRETRAINED_INVENTORY_PATH.read_text(encoding="utf-8"),
        location=str(FORMAL_PRETRAINED_INVENTORY_PATH),
    )
    if not isinstance(registry, Mapping) or not isinstance(inventory, Mapping):
        raise ValueError("canonical pretrained registry/inventory must be JSON objects")
    policy = registry.get("policy")
    if not isinstance(policy, Mapping) or policy.get("hash_manifest_required") is not True:
        raise ValueError("canonical pretrained registry must require a hash manifest")
    registry_models = registry.get("models")
    if not isinstance(registry_models, list):
        raise ValueError("canonical pretrained registry models must be a list")
    registry_by_id: dict[str, Mapping[str, Any]] = {}
    for item in registry_models:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("canonical pretrained registry contains an invalid model")
        model_id = str(item["id"])
        if model_id in registry_by_id:
            raise ValueError(f"canonical pretrained registry duplicate id: {model_id}")
        registry_by_id[model_id] = item
    if set(registry_by_id) != set(_FORMAL_REGISTRY_IDS):
        raise ValueError("canonical pretrained registry IDs differ from the formal contract")
    if inventory.get("all_pretrain_assets_ready") is not True:
        raise ValueError("canonical pretrained inventory is not fully ready")
    if inventory.get("pretrain_root") != registry.get("remote_root"):
        raise ValueError("canonical pretrained registry/inventory roots differ")
    inventory_models = inventory.get("models")
    if not isinstance(inventory_models, list):
        raise ValueError("canonical pretrained inventory models must be a list")
    inventory_by_id = {
        str(item.get("id")): item
        for item in inventory_models
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    target = inventory_by_id.get(FORMAL_MODEL_REGISTRY_ID)
    if not isinstance(target, Mapping) or target.get("pretrain_asset_ready") is not True:
        raise ValueError("canonical Motion-R1 pretrained inventory entry is not ready")
    artifacts = target.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("canonical Motion-R1 pretrained artifacts must be a list")
    by_role = {
        str(item.get("role")): item
        for item in artifacts
        if isinstance(item, Mapping) and isinstance(item.get("role"), str)
    }
    base_record = by_role.get("base_and_processor")
    vq_record = by_role.get("motion_vqvae")
    if not isinstance(base_record, Mapping) or not isinstance(vq_record, Mapping):
        raise ValueError("canonical Motion-R1 base/VQ inventory records are missing")
    if base_record.get("status") != "valid" or vq_record.get("status") != "valid":
        raise ValueError("canonical Motion-R1 base/VQ inventory status is invalid")
    base_digest = _require_lower_sha256(
        base_record.get("tree_sha256"), location="canonical Motion-R1 base tree_sha256"
    )
    vq_digest = _require_lower_sha256(
        vq_record.get("sha256"), location="canonical Motion-R1 VQ-VAE sha256"
    )
    base_receipt = hash_path(
        Path(arguments.base_artifact_path).resolve(strict=True), symlink_policy="follow"
    )
    vq_receipt = hash_path(
        Path(arguments.motion_vqvae_asset_path).resolve(strict=True),
        symlink_policy="follow",
    )
    if base_receipt.kind != "directory" or base_receipt.digest != base_digest:
        raise ValueError("formal GRPO base bytes differ from the canonical Motion-R1 base")
    expected_files = base_record.get("selected_file_count")
    if (
        isinstance(expected_files, bool)
        or not isinstance(expected_files, int)
        or base_receipt.file_count != expected_files
    ):
        raise ValueError("formal GRPO base file count differs from canonical inventory")
    if vq_receipt.kind != "file" or vq_receipt.digest != vq_digest:
        raise ValueError("formal GRPO VQ-VAE bytes differ from the canonical Motion-R1 asset")


def _bind_formal_code_identity(config: Mapping[str, Any], arguments: SimpleNamespace) -> None:
    """Require formal execution, plugins, register and contract to share one code root."""

    code_raw = arguments.code_path
    if not isinstance(code_raw, str) or not code_raw.strip():
        raise ValueError("formal GRPO provenance.code_path is required")
    code_path = Path(code_raw)
    _reject_symlink_components(code_path)
    if code_path.resolve(strict=True) != REPO_ROOT.resolve(strict=True):
        raise ValueError(
            "formal GRPO provenance.code_path must be the actual frozen code root "
            "executing this runner"
        )
    runner_code_raw = getattr(arguments, "runner_code_path", None)
    if not isinstance(runner_code_raw, str) or not runner_code_raw:
        raise ValueError("formal GRPO provenance.runner_code_path is required")
    runner_code_path = Path(runner_code_raw)
    _reject_symlink_components(runner_code_path)
    if runner_code_path.resolve(strict=True) != Path(__file__).resolve(strict=True):
        raise ValueError(
            "formal GRPO provenance.runner_code_path must be the exact executing runner"
        )
    run_cfg = _get_section(config, "run")
    contract_raw = run_cfg.get("runtime_contract_path")
    if Path(str(contract_raw)).resolve(strict=True) != FORMAL_RUNTIME_CONTRACT_PATH.resolve(
        strict=True
    ):
        raise ValueError("formal GRPO runtime contract must come from the executing code root")
    plugin_cfg = _get_section(config, "plugins")
    expected = {
        "external_plugins": FORMAL_EXTERNAL_PLUGIN_PATH.resolve(strict=True),
        "custom_register_path": FORMAL_CUSTOM_REGISTER_PATH.resolve(strict=True),
    }
    for key, expected_path in expected.items():
        raw = plugin_cfg.get(key)
        values = [raw] if isinstance(raw, (str, Path)) else raw
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"formal GRPO plugins.{key} must contain exactly one frozen path")
        candidate = Path(str(values[0]))
        _reject_symlink_components(candidate)
        if candidate.resolve(strict=True) != expected_path:
            raise ValueError(
                f"formal GRPO plugins.{key} must be the approved file in the executing code root"
            )


def _positive_integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _validate_formal_training_contract(config: Mapping[str, Any]) -> int:
    """Freeze a production optimizer-step target before Swift may be launched."""

    run_cfg = _get_section(config, "run")
    train_cfg = _get_section(config, "training")
    if run_cfg.get("purpose") != FORMAL_PURPOSE:
        raise ValueError(f"formal GRPO run.purpose must be {FORMAL_PURPOSE!r}")
    if run_cfg.get("unsafe_legacy_no_manifest") not in (None, False):
        raise ValueError("formal GRPO forbids unsafe_legacy_no_manifest")
    expected_steps = _positive_integer(
        run_cfg.get("expected_optimizer_steps"),
        location="run.expected_optimizer_steps",
    )
    max_steps = _positive_integer(train_cfg.get("max_steps"), location="training.max_steps")
    if max_steps != expected_steps:
        raise ValueError("training.max_steps must equal run.expected_optimizer_steps")
    _positive_integer(train_cfg.get("lora_rank"), location="training.lora_rank")
    lora_alpha = train_cfg.get("lora_alpha")
    if (
        isinstance(lora_alpha, bool)
        or not isinstance(lora_alpha, (int, float))
        or not math.isfinite(float(lora_alpha))
        or float(lora_alpha) <= 0
    ):
        raise ValueError("training.lora_alpha must be finite and positive")
    lora_dropout = train_cfg.get("lora_dropout")
    if (
        isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (int, float))
        or not math.isfinite(float(lora_dropout))
        or not 0 <= float(lora_dropout) <= 1
    ):
        raise ValueError("training.lora_dropout must be in [0, 1]")
    if train_cfg.get("target_modules") != "all-linear":
        raise ValueError("formal GRPO requires training.target_modules=all-linear")
    required_defaults = {
        "tuner_type": "lora",
        "tuner_backend": "peft",
        "use_swift_lora": False,
        "target_regex": None,
        "target_parameters": None,
        "lora_dtype": None,
        "lorap_lr_ratio": None,
        "lorap_emb_lr": 1.0e-6,
        "freeze_llm": False,
        "freeze_vit": True,
        "freeze_aligner": True,
    }
    differing_defaults = sorted(
        key
        for key, expected_value in required_defaults.items()
        if key not in train_cfg or train_cfg.get(key) != expected_value
    )
    if differing_defaults:
        raise ValueError(
            "formal GRPO training LoRA/backend defaults differ: "
            f"fields={differing_defaults}"
        )
    _expected_modules_to_save(config)
    if train_cfg.get("lora_bias") != "none":
        raise ValueError("formal GRPO requires training.lora_bias=none")
    if train_cfg.get("use_rslora") is not False:
        raise ValueError("formal GRPO requires training.use_rslora=false")
    if train_cfg.get("use_dora") is not False:
        raise ValueError("formal GRPO requires training.use_dora=false")
    run_name = str(run_cfg.get("run_name", "")).strip()
    if not run_name or any(marker in run_name.casefold() for marker in ("smoke", "debug")):
        raise ValueError("formal finetune run_name must be non-empty and must not be smoke/debug")
    learning_rate = train_cfg.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("formal GRPO learning_rate must be finite and positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("formal GRPO learning_rate must be finite and positive")
    if train_cfg.get("save_only_model") is not False:
        raise ValueError("formal GRPO requires training.save_only_model=false")
    if train_cfg.get("save_safetensors") is not True:
        raise ValueError("formal GRPO requires training.save_safetensors=true")
    if train_cfg.get("callbacks") != [FORMAL_TRAINING_CALLBACK]:
        raise ValueError(
            "formal GRPO training.callbacks must contain exactly the frozen training-receipt callback"
        )
    if train_cfg.get("deepspeed") not in (None, False, ""):
        raise ValueError(
            "formal GRPO training receipt currently requires replicated parameters; deepspeed is unsupported"
        )
    logging_steps = _positive_integer(
        train_cfg.get("logging_steps"), location="training.logging_steps"
    )
    if logging_steps != 1:
        raise ValueError(
            "formal GRPO requires training.logging_steps=1 so every optimizer step has finite-loss evidence"
        )
    if str(train_cfg.get("save_strategy", "")).casefold() != "steps":
        raise ValueError("formal GRPO requires training.save_strategy=steps")
    save_steps = _positive_integer(train_cfg.get("save_steps"), location="training.save_steps")
    if save_steps > expected_steps or expected_steps % save_steps:
        raise ValueError("training.save_steps must divide expected_optimizer_steps")
    artifact_name = Path(str(run_cfg.get("artifact_path", ""))).name
    if artifact_name != f"checkpoint-{expected_steps}":
        raise ValueError(
            "formal GRPO artifact_path must name the exact expected optimizer-step checkpoint"
        )
    return expected_steps


def _validate_formal_leakage_audit(
    config: Mapping[str, Any], arguments: SimpleNamespace
) -> None:
    """Bind a strict leakage audit to the exact current train/val/benchmark bytes."""

    paths = {
        "train": Path(arguments.train_data_path).resolve(strict=True),
        "validation": Path(arguments.validation_data_path).resolve(strict=True),
        "benchmark": Path(arguments.benchmark_path).resolve(strict=True),
    }
    for name, path in paths.items():
        _reject_symlink_components(path)
        if path.suffix.casefold() not in {".json", ".jsonl"}:
            raise ValueError(f"formal GRPO {name} input must be strict JSON/JSONL")
    benchmark_rows = _local_records(paths["benchmark"])
    for line_no, row in benchmark_rows:
        if not row:
            raise ValueError(
                f"formal GRPO benchmark record must not be empty: {paths['benchmark']}:{line_no}"
            )
        if not any(key in row for key in ("sample_id", "group_id", "id")):
            raise ValueError("formal GRPO benchmark records require a stable identity")
        if not any(key in row for key in ("question", "prompt", "messages")):
            raise ValueError("formal GRPO benchmark records require question/prompt/messages")
    audit_path = Path(arguments.leakage_audit_path).resolve(strict=True)
    _reject_symlink_components(audit_path)
    payload = _strict_json(audit_path.read_text(encoding="utf-8"), location=str(audit_path))
    if not isinstance(payload, Mapping):
        raise ValueError("formal GRPO leakage audit must be one JSON object")
    required_keys = {
        "schema",
        "status",
        "batch_id",
        "normalization_algorithm",
        "train_sha256",
        "validation_sha256",
        "benchmark_sha256",
        "train_rows",
        "validation_rows",
        "benchmark_rows",
        "overlap_counts",
    }
    if set(payload) != required_keys:
        raise ValueError("formal GRPO leakage audit has missing or unknown fields")
    if payload.get("schema") != FORMAL_LEAKAGE_SCHEMA or payload.get("status") != "passed":
        raise ValueError("formal GRPO leakage audit schema/status is not publishable")
    if payload.get("batch_id") != _get_section(config, "run").get("batch_id"):
        raise ValueError("formal GRPO leakage audit batch_id differs from the run")
    if payload.get("normalization_algorithm") != FORMAL_NORMALIZATION_ALGORITHM:
        raise ValueError("formal GRPO leakage audit normalization algorithm differs")
    for name, path in paths.items():
        expected = hash_path(path, symlink_policy="reject").digest
        if payload.get(f"{name}_sha256") != expected:
            raise ValueError(f"formal GRPO leakage audit {name} hash differs")
        expected_rows = len(_local_records(path))
        if payload.get(f"{name}_rows") != expected_rows:
            raise ValueError(f"formal GRPO leakage audit {name} row count differs")
    overlap_counts = payload.get("overlap_counts")
    expected_overlap_keys = {
        "sample_id",
        "group_id",
        "media_sha256",
        "normalized_prompt",
        "normalized_solution",
        "near_duplicate",
    }
    if not isinstance(overlap_counts, Mapping) or set(overlap_counts) != expected_overlap_keys:
        raise ValueError("formal GRPO leakage audit overlap_counts schema differs")
    if any(isinstance(value, bool) or value != 0 for value in overlap_counts.values()):
        raise ValueError("formal GRPO leakage audit reports non-zero overlap")


def _local_records(path: Path) -> List[Tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Tuple[int, Mapping[str, Any]]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = _strict_json(raw, location=f"{path}:{line_no}")
                if not isinstance(value, Mapping):
                    raise ValueError(f"Expected object record in {path}:{line_no}")
                rows.append((line_no, value))
        if not rows:
            raise ValueError(f"Local GRPO JSONL dataset must contain at least one record: {path}")
        return rows
    if suffix == ".json":
        value = _strict_json(path.read_text(encoding="utf-8"), location=str(path))
        if not isinstance(value, list):
            raise ValueError(f"Top-level JSON dataset must be a list: {path}")
        rows = []
        for index, record in enumerate(value, start=1):
            if not isinstance(record, Mapping):
                raise ValueError(f"Expected object record in {path}[{index - 1}]")
            rows.append((index, record))
        if not rows:
            raise ValueError(f"Local GRPO JSON dataset must contain at least one record: {path}")
        return rows
    raise ValueError(f"Local GRPO dataset must be .json or .jsonl: {path}")


def _configured_dataset_values(data_cfg: Mapping[str, Any], key: str) -> list[Any]:
    raw = data_cfg.get(key, [])
    values = [raw] if isinstance(raw, (str, Path)) else raw
    if not isinstance(values, list):
        raise ValueError(f"`data.{key}` must be list/str/path, got {type(values)}")
    return [value for value in values if value is not None]


def _resolve_local_dataset_path(value: Any) -> Path | None:
    dataset_path = Path(str(value))
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        if dataset_path.suffix.lower() in {".json", ".jsonl"}:
            raise FileNotFoundError(f"Local GRPO dataset not found: {dataset_path}")
        return None
    if dataset_path.is_dir():
        raise ValueError(f"Local GRPO dataset directories are unsupported: {dataset_path}")
    return dataset_path


def _require_lower_sha256(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{location} must be one lowercase SHA-256 digest")
    return value


def _reject_symlink_components(path: Path) -> None:
    """Reject a formal input/output path with a symlink component."""

    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"formal GRPO paths must not traverse symlinks: {path}")
        if component == Path(candidate.anchor):
            break


def _verified_media_file(
    raw_path: Any,
    expected_digest: Any,
    *,
    dataset_path: Path,
    location: str,
    media_cache: MutableMapping[Path, str],
) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path.strip() or "://" in raw_path:
        raise ValueError(f"{location} must be one non-empty local filesystem path")
    digest = _require_lower_sha256(expected_digest, location=f"{location}_sha256")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = dataset_path.parent / candidate
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{location} does not resolve to an existing local file") from exc
    if not resolved.is_file():
        raise ValueError(f"{location} must resolve to a regular file")
    cached_digest = media_cache.get(resolved)
    if cached_digest is not None:
        if cached_digest != digest:
            raise ValueError(
                f"{location} conflicts with another SHA-256 binding for {resolved}"
            )
        if resolved.stat().st_size <= 0:
            raise ValueError(f"{location} must resolve to a non-empty regular file")
        return resolved, digest
    receipt = hash_path(candidate, symlink_policy="reject")
    if receipt.kind != "file" or receipt.total_bytes <= 0:
        raise ValueError(f"{location} must resolve to a non-empty regular file")
    if receipt.digest != digest:
        raise ValueError(
            f"{location} SHA-256 mismatch: expected {digest}, observed {receipt.digest}"
        )
    media_cache[resolved] = receipt.digest
    return resolved, digest


def _formal_message_bindings(
    record: Mapping[str, Any], *, location: str
) -> tuple[list[str], str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError(f"formal GRPO {location} messages must contain exactly one user message")
    videos: list[str] = []
    prompt_parts: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"formal GRPO {location} messages[{message_index}] must be an object"
            )
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"formal GRPO {location} messages[{message_index}] has unknown/missing keys"
            )
        role = message.get("role")
        content = message.get("content")
        if role != "user":
            raise ValueError(
                f"formal GRPO {location} messages[{message_index}].role must be 'user'"
            )
        if not isinstance(content, list) or not content:
            raise ValueError(
                f"formal GRPO {location} messages[{message_index}].content must be a "
                "non-empty canonical multimodal list"
            )
        for item_index, item in enumerate(content):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"formal GRPO {location} messages[{message_index}].content"
                    f"[{item_index}] must be an object"
                )
            item_type = item.get("type")
            if item_type == "video":
                if set(item) != {"type", "video"}:
                    raise ValueError(
                        f"formal GRPO {location} video content has unknown/missing keys"
                    )
                video = item.get("video")
                if not isinstance(video, str) or not video.strip():
                    raise ValueError(
                        f"formal GRPO {location} video content requires one path"
                    )
                videos.append(video)
            elif item_type == "text":
                if set(item) != {"type", "text"}:
                    raise ValueError(
                        f"formal GRPO {location} text content has unknown/missing keys"
                    )
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    prompt_parts.append(text_value)
                else:
                    raise ValueError(f"formal GRPO {location} text content must be non-empty")
            else:
                raise ValueError(
                    f"formal GRPO {location} forbids content type {item_type!r}; "
                    "only text/video are allowed"
                )
    if len(videos) != 1:
        raise ValueError(f"formal GRPO {location} must bind exactly one message video")
    if not prompt_parts:
        raise ValueError(f"formal GRPO {location} requires non-empty user prompt text")
    return videos, "\n".join(prompt_parts)


def _message_video_paths(record: Mapping[str, Any], *, location: str) -> list[str]:
    return _formal_message_bindings(record, location=location)[0]


def _normalized_text_identity(value: str, *, strip_prompt_markers: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if strip_prompt_markers:
        normalized = re.sub(r"<motion_start>\s*<motion>\s*<motion_end>", " ", normalized)
        normalized = re.sub(r"\[(?:qid|branch)=[^\]]+\]", " ", normalized)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _formal_media_binding(
    record: Mapping[str, Any],
    metadata: RewardMetadata,
    *,
    dataset_path: Path,
    line_no: int,
    media_cache: MutableMapping[Path, str],
) -> Mapping[str, Any]:
    location = f"{dataset_path}:{line_no}"
    allowed_keys = set(_FORMAL_RECORD_COMMON_KEYS)
    if "request_id" in record:
        allowed_keys.add("request_id")
    if metadata.branch is RewardBranch.VM:
        allowed_keys.update({"motion", "motion_sha256"})
    if set(record) != allowed_keys:
        unknown = sorted(set(record) - allowed_keys)
        missing = sorted(allowed_keys - set(record))
        raise ValueError(
            f"formal GRPO {location} record schema differs; unknown={unknown}, missing={missing}"
        )
    videos, prompt_text = _formal_message_bindings(record, location=location)
    video_path, video_digest = _verified_media_file(
        videos[0],
        record.get("video_sha256"),
        dataset_path=dataset_path,
        location=f"{location} video",
        media_cache=media_cache,
    )
    result: dict[str, Any] = {
        "video_path": str(video_path),
        "video_sha256": video_digest,
        "normalized_prompt_sha256": _normalized_text_identity(
            prompt_text, strip_prompt_markers=True
        ),
        "normalized_solution_sha256": _normalized_text_identity(str(record["solution"])),
    }
    motion = record.get("motion")
    if metadata.branch is RewardBranch.VM:
        motion_path, motion_digest = _verified_media_file(
            motion,
            record.get("motion_sha256"),
            dataset_path=dataset_path,
            location=f"{location} motion",
            media_cache=media_cache,
        )
        result.update(
            {"motion_path": str(motion_path), "motion_sha256": motion_digest}
        )
    return result


def _enforce_formal_preflight_policy(
    config: Mapping[str, Any], *, formal_artifact: bool
) -> None:
    if not formal_artifact:
        return
    run_cfg = _get_section(config, "run")
    data_cfg = _get_section(config, "data")
    if run_cfg.get("dataset_precheck", True) is not True:
        raise ValueError("formal GRPO forbids disabling dataset_precheck")
    forbidden_run_keys = {
        "dataset_precheck_max_samples",
        "dataset_precheck_limit",
        "skip_dataset_precheck",
        "disable_dataset_precheck",
    }
    forbidden_data_keys = {
        "max_samples",
        "max_train_samples",
        "dataset_sample",
        "train_dataset_sample",
        "dataset_precheck_max_samples",
    }
    active = set(forbidden_run_keys.intersection(run_cfg)).union(
        forbidden_data_keys.intersection(data_cfg)
    )
    for key in run_cfg:
        normalized = str(key).casefold()
        if normalized != "dataset_precheck" and "precheck" in normalized:
            active.add(str(key))
        if ("skip" in normalized or "disable" in normalized) and "dataset" in normalized:
            active.add(str(key))
    for key in data_cfg:
        normalized = str(key).casefold()
        if "sample" in normalized and any(
            token in normalized for token in ("max", "limit", "count", "size")
        ):
            active.add(str(key))
        if normalized.startswith("max_") or normalized.endswith("_limit"):
            active.add(str(key))
    if data_cfg.get("load_from_cache_file") is not False:
        active.add("load_from_cache_file")
    if data_cfg.get("custom_dataset_info") not in (None, "", [], {}):
        active.add("custom_dataset_info")
    if active:
        raise ValueError(
            "formal GRPO requires direct full-file loading and a full dataset scan; "
            f"bypass/cache/truncation keys are forbidden: {sorted(active)}"
        )


def _precheck_dataset_records(
    config: Mapping[str, Any],
    *,
    formal_artifact: bool = False,
    binding_sink: MutableMapping[Path, str] | None = None,
) -> List[Tuple[Path, int, Mapping[str, Any]]]:
    run_cfg = _get_section(config, "run")
    data_cfg = _get_section(config, "data")
    enabled = bool(run_cfg.get("dataset_precheck", True))
    if not enabled:
        if formal_artifact:
            raise ValueError("formal GRPO requires dataset_precheck")
        return []

    roles = ["dataset"]
    if formal_artifact:
        roles.append("val_dataset")

    train_checked_rows: List[Tuple[Path, int, Mapping[str, Any]]] = []
    role_identities: dict[str, dict[str, set[str]]] = {}
    role_counts: dict[str, int] = {}
    media_cache: dict[Path, str] = {}

    for role in roles:
        dataset_values = _configured_dataset_values(data_cfg, role)
        if formal_artifact and len(dataset_values) != 1:
            raise ValueError(f"formal GRPO data.{role} must contain exactly one local file")

        checked_rows: List[Tuple[Path, int, Mapping[str, Any]]] = []
        seen_rollouts: Dict[Tuple[str, int], Tuple[str, str, str]] = {}
        seen_sample_ids: set[str] = set()
        group_rows: Dict[str, List[tuple[RewardMetadata, Mapping[str, Any] | None]]] = {}
        identities = {
            "sample_id": set(),
            "group_id": set(),
            "video_sha256": set(),
            "motion_sha256": set(),
            "request_id": set(),
            "normalized_prompt_sha256": set(),
            "normalized_solution_sha256": set(),
        }

        for value in dataset_values:
            dataset_path = _resolve_local_dataset_path(value)
            if dataset_path is None:
                if formal_artifact:
                    raise ValueError(f"formal GRPO data.{role} must be a local JSON/JSONL file")
                # Preserve legacy hub-dataset dry-run support only outside formal mode.
                continue

            dataset_receipt_before = None
            if formal_artifact:
                _reject_symlink_components(dataset_path)
                dataset_receipt_before = hash_path(dataset_path, symlink_policy="reject")
            records = _local_records(dataset_path)
            for line_no, record in records:
                motion = record.get("motion")
                has_motion = motion not in (None, "", [], {})
                sample_id = record.get("sample_id")
                group_id = record.get("group_id")
                request_id = record.get("request_id")

                try:
                    metadata = RewardMetadata.from_mapping(record)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid reward metadata at {dataset_path}:{line_no} "
                        f"(sample_id={sample_id!r}, group_id={group_id!r}, "
                        f"request_id={request_id!r})"
                    ) from exc
                if _semantic_reward_enabled(config):
                    try:
                        validate_semantic_reference(
                            record.get("solution"), metadata.gold_answer
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid semantic solution at {dataset_path}:{line_no} "
                            f"(sample_id={sample_id!r}, group_id={group_id!r})"
                        ) from exc
                if metadata.branch not in {RewardBranch.VM, RewardBranch.V}:
                    raise ValueError(
                        f"GRPO training supports only v/vm branches at "
                        f"{dataset_path}:{line_no}"
                    )
                if metadata.branch is RewardBranch.VM and not has_motion:
                    raise ValueError(
                        f"VM sample missing motion at {dataset_path}:{line_no} "
                        f"(sample_id={sample_id!r}, group_id={group_id!r}, "
                        f"request_id={request_id!r})"
                    )
                if metadata.branch is RewardBranch.V and has_motion:
                    raise ValueError(
                        f"V sample must not include motion at {dataset_path}:{line_no} "
                        f"(sample_id={sample_id!r}, group_id={group_id!r}, "
                        f"request_id={request_id!r})"
                    )

                identity = (metadata.sample_id, metadata.rollout_id)
                signature = (
                    metadata.group_id,
                    metadata.branch.value,
                    metadata.gold_answer,
                )
                previous = seen_rollouts.get(identity)
                if previous is not None:
                    detail = "conflicting" if previous != signature else "duplicate"
                    raise ValueError(
                        f"{detail} rollout identity at {dataset_path}:{line_no}: {identity}"
                    )
                seen_rollouts[identity] = signature
                if formal_artifact and metadata.sample_id in seen_sample_ids:
                    raise ValueError(
                        f"formal GRPO data.{role} has duplicate sample_id "
                        f"{metadata.sample_id!r}"
                    )
                seen_sample_ids.add(metadata.sample_id)

                media: Mapping[str, Any] | None = None
                if formal_artifact:
                    media = _formal_media_binding(
                        record,
                        metadata,
                        dataset_path=dataset_path,
                        line_no=line_no,
                        media_cache=media_cache,
                    )
                    identities["video_sha256"].add(str(media["video_sha256"]))
                    motion_digest = media.get("motion_sha256")
                    if motion_digest is not None:
                        identities["motion_sha256"].add(str(motion_digest))
                    identities["normalized_prompt_sha256"].add(
                        str(media["normalized_prompt_sha256"])
                    )
                    identities["normalized_solution_sha256"].add(
                        str(media["normalized_solution_sha256"])
                    )
                identities["sample_id"].add(metadata.sample_id)
                identities["group_id"].add(metadata.group_id)
                if metadata.request_id is not None:
                    identities["request_id"].add(metadata.request_id)
                group_rows.setdefault(metadata.group_id, []).append((metadata, media))
                checked_rows.append((dataset_path, line_no, record))
            if formal_artifact:
                dataset_receipt_after = hash_path(dataset_path, symlink_policy="reject")
                if dataset_receipt_after != dataset_receipt_before:
                    raise ValueError(
                        f"formal GRPO data.{role} changed during preflight: {dataset_path}"
                    )
                if binding_sink is not None:
                    binding_sink[dataset_path.resolve(strict=True)] = dataset_receipt_after.digest

        for group_id, grouped in sorted(group_rows.items()):
            rows = [metadata for metadata, _ in grouped]
            branches = {row.branch for row in rows}
            if branches != {RewardBranch.VM, RewardBranch.V}:
                raise ValueError(
                    f"GRPO {role} group {group_id!r} must co-locate exactly vm and v branches"
                )
            vm_count = sum(row.branch is RewardBranch.VM for row in rows)
            v_count = sum(row.branch is RewardBranch.V for row in rows)
            if vm_count != 1 or v_count != 1:
                raise ValueError(
                    f"GRPO canonical group {group_id!r} must contain exactly one vm and one v "
                    f"prompt before runtime generation expansion: {vm_count}/{v_count}"
                )
            if len({row.gold_answer for row in rows}) != 1:
                raise ValueError(f"GRPO group {group_id!r} has conflicting gold answers")
            if formal_artifact:
                media_rows = [media for _, media in grouped]
                if any(media is None for media in media_rows):  # pragma: no cover
                    raise RuntimeError("formal GRPO media validation produced no binding")
                video_paths = {str(media["video_path"]) for media in media_rows if media}
                video_hashes = {
                    str(media["video_sha256"]) for media in media_rows if media
                }
                if len(video_paths) != 1 or len(video_hashes) != 1:
                    raise ValueError(
                        f"formal GRPO group {group_id!r} VM/V rows must bind the same video"
                    )

        if formal_artifact and not checked_rows:
            raise ValueError(f"formal GRPO preflight did not scan data.{role} records")
        if role == "dataset":
            train_checked_rows = checked_rows
        role_identities[role] = identities
        role_counts[role] = len(checked_rows)

    if formal_artifact:
        for media_path, expected_digest in sorted(
            media_cache.items(), key=lambda item: str(item[0])
        ):
            try:
                final_receipt = hash_path(media_path, symlink_policy="reject")
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"formal GRPO media changed or disappeared during preflight: {media_path}"
                ) from exc
            if (
                final_receipt.kind != "file"
                or final_receipt.total_bytes <= 0
                or final_receipt.digest != expected_digest
            ):
                raise ValueError(
                    f"formal GRPO media changed during preflight: {media_path}"
                )
        train_ids = role_identities["dataset"]
        validation_ids = role_identities["val_dataset"]
        for identity_name in (
            "sample_id",
            "group_id",
            "video_sha256",
            "motion_sha256",
            "request_id",
            "normalized_prompt_sha256",
            "normalized_solution_sha256",
        ):
            overlap = sorted(train_ids[identity_name].intersection(validation_ids[identity_name]))
            if overlap:
                preview = overlap[:3]
                raise ValueError(
                    "formal GRPO train/validation identity leakage for "
                    f"{identity_name}: {preview}"
                )
        if binding_sink is not None:
            binding_sink.update(media_cache)

    total_checked = sum(role_counts.values())
    if total_checked > 0:
        rendered = ", ".join(f"{role}={count}" for role, count in role_counts.items())
        print(f"Dataset precheck passed: {total_checked} records validated ({rendered}).")
    elif formal_artifact:
        raise ValueError("formal GRPO preflight did not scan any local dataset records")
    return train_checked_rows


def _ensure_paths(config: Mapping[str, Any]) -> None:
    run_cfg = _get_section(config, "run")
    output_dir = run_cfg.get("output_dir")
    if output_dir:
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)


def _artifact_arguments(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> SimpleNamespace:
    run_cfg = _get_section(config, "run")
    model_cfg = _get_section(config, "model")
    provenance_cfg = _get_section(config, "provenance")

    def configured(name: str, fallback: Any = None) -> Any:
        return provenance_cfg.get(name, run_cfg.get(name, fallback))

    return SimpleNamespace(
        batch_id=run_cfg.get("batch_id"),
        model_registry_id=run_cfg.get("model_registry_id"),
        base_artifact_path=configured("base_artifact_path", model_cfg.get("model")),
        train_data_path=configured("train_data_path"),
        validation_data_path=configured("validation_data_path"),
        benchmark_path=configured("benchmark_path"),
        leakage_audit_path=configured("leakage_audit_path"),
        config_path=configured("config_path", str(config_path)),
        code_path=configured("code_path"),
        runner_code_path=configured("runner_code_path", str(Path(__file__).resolve())),
        environment_path=configured("environment_path"),
        motion_vqvae_asset_path=configured("motion_vqvae_asset_path"),
        artifact_root=run_cfg.get("artifact_root"),
        artifact_manifest_path=run_cfg.get("artifact_manifest_path"),
        reload_receipt_path=run_cfg.get("reload_receipt_path"),
        resume_manifest=run_cfg.get("resume_manifest"),
        unsafe_legacy_no_manifest=bool(run_cfg.get("unsafe_legacy_no_manifest", False)),
    )


def _one_local_dataset_path(value: Any, *, name: str) -> Path:
    values = [value] if isinstance(value, (str, Path)) else value
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"formal GRPO {name} must contain exactly one local dataset path")
    raw = values[0]
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError(f"formal GRPO {name} must be a non-empty local path")
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        raise ValueError(f"formal GRPO {name} must be an absolute path")
    _reject_symlink_components(candidate)
    try:
        candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"formal GRPO {name} is not an existing local file") from exc
    if not candidate.is_file() or candidate.suffix.casefold() not in {".json", ".jsonl"}:
        raise ValueError(f"formal GRPO {name} must be one .json/.jsonl file")
    return candidate


def _bind_formal_loader_inputs(
    config: Mapping[str, Any],
    artifact_arguments: SimpleNamespace,
    *,
    config_path: Path | None = None,
    binding_sink: MutableMapping[Path, str] | None = None,
) -> Path:
    """Bind actual Swift loader/model assets to the paths hashed in provenance."""

    data_cfg = _get_section(config, "data")
    model_cfg = _get_section(config, "model")
    train_path = _one_local_dataset_path(data_cfg.get("dataset"), name="data.dataset")
    expected_train = Path(artifact_arguments.train_data_path).resolve(strict=True)
    if train_path != expected_train:
        raise ValueError("GRPO data.dataset must exactly equal provenance.train_data_path")
    validation_path = _one_local_dataset_path(
        data_cfg.get("val_dataset"), name="data.val_dataset"
    )
    expected_validation = Path(artifact_arguments.validation_data_path).resolve(strict=True)
    if validation_path != expected_validation:
        raise ValueError(
            "GRPO data.val_dataset must exactly equal provenance.validation_data_path"
        )
    if train_path == validation_path:
        raise ValueError("formal GRPO train and validation datasets must be distinct")
    _local_records(train_path)
    _local_records(validation_path)

    actual_base_raw = model_cfg.get("model")
    if not isinstance(actual_base_raw, str) or not actual_base_raw.strip():
        raise ValueError("formal GRPO model.model must be one non-empty local path")
    actual_base = Path(actual_base_raw).resolve(strict=True)
    expected_base = Path(artifact_arguments.base_artifact_path).resolve(strict=True)
    if actual_base != expected_base:
        raise ValueError("GRPO model.model must exactly equal provenance.base_artifact_path")
    actual_base_hash = hash_path(actual_base, symlink_policy="follow")
    expected_base_hash = hash_path(expected_base, symlink_policy="follow")
    if actual_base_hash != expected_base_hash or actual_base_hash.total_bytes <= 0:
        raise ValueError("formal GRPO base model identity/hash differs from provenance")
    if binding_sink is not None:
        binding_sink[train_path] = hash_path(train_path, symlink_policy="reject").digest
        binding_sink[validation_path] = hash_path(
            validation_path, symlink_policy="reject"
        ).digest
        binding_sink[actual_base] = actual_base_hash.digest
    if config_path is not None:
        expected_config = Path(artifact_arguments.config_path).resolve(strict=True)
        if config_path.resolve(strict=True) != expected_config:
            raise ValueError("executed GRPO config must exactly equal provenance.config_path")

    model_kwargs = model_cfg.get("model_kwargs", {})
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, Mapping):
        raise ValueError("model.model_kwargs must be a mapping")
    actual_vqvae = model_kwargs.get("vqvae_path")
    provenance_vqvae = artifact_arguments.motion_vqvae_asset_path
    if actual_vqvae in (None, "") or provenance_vqvae in (None, ""):
        raise ValueError(
            "formal motion GRPO requires an explicit config-bound and provenance-bound VQ-VAE"
        )
    actual_vqvae_path = Path(str(actual_vqvae)).resolve(strict=True)
    if not actual_vqvae_path.is_file() or actual_vqvae_path.stat().st_size <= 0:
        raise ValueError("GRPO model motion VQ-VAE must be a non-empty regular file")
    expected_vqvae = Path(str(provenance_vqvae)).resolve(strict=True)
    if actual_vqvae_path != expected_vqvae:
        raise ValueError(
            "GRPO model.model_kwargs.vqvae_path must equal provenance.motion_vqvae_asset_path"
        )
    if hash_path(actual_vqvae_path, symlink_policy="follow") != hash_path(
        expected_vqvae, symlink_policy="follow"
    ):
        raise ValueError("formal GRPO VQ-VAE identity/hash differs from provenance")
    if binding_sink is not None:
        binding_sink[actual_vqvae_path] = hash_path(
            actual_vqvae_path, symlink_policy="follow"
        ).digest
    artifact_arguments.motion_vqvae_asset_path = str(actual_vqvae_path)
    if isinstance(config, MutableMapping):
        data_section = config.get("data")
        model_section = config.get("model")
        if isinstance(data_section, MutableMapping):
            data_section["dataset"] = [str(train_path)]
            data_section["val_dataset"] = [str(validation_path)]
        if isinstance(model_section, MutableMapping):
            model_section["model"] = str(actual_base)
            mutable_kwargs = model_section.get("model_kwargs")
            if isinstance(mutable_kwargs, MutableMapping):
                mutable_kwargs["vqvae_path"] = str(actual_vqvae_path)
    return train_path


def _capture_formal_input_snapshot(
    arguments: SimpleNamespace,
    *,
    media_bindings: Mapping[Path, str],
) -> Mapping[str, SimpleNamespace]:
    """Hash every formal source actually consumed so publication can recheck it."""

    for path, expected_digest in media_bindings.items():
        _reject_symlink_components(path)
        observed = hash_path(path, symlink_policy="reject")
        if observed.digest != expected_digest:
            raise ValueError(
                f"formal GRPO input changed after preflight and before launch: {path}"
            )

    specifications: list[tuple[str, Path, str]] = [
        ("base_artifact", Path(arguments.base_artifact_path), "follow"),
        ("train_data", Path(arguments.train_data_path), "reject"),
        ("validation_data", Path(arguments.validation_data_path), "reject"),
        ("benchmark", Path(arguments.benchmark_path), "reject"),
        ("leakage_audit", Path(arguments.leakage_audit_path), "reject"),
        ("config", Path(arguments.config_path), "reject"),
        ("runner_code", Path(arguments.runner_code_path), "reject"),
        ("environment", Path(arguments.environment_path), "reject"),
        ("motion_vqvae", Path(arguments.motion_vqvae_asset_path), "follow"),
    ]
    for index, path in enumerate(_FORMAL_CODE_PATHS):
        specifications.append((f"code_{index}", path, "reject"))
    for index, (path, expected_digest) in enumerate(
        sorted(media_bindings.items(), key=lambda item: str(item[0]))
    ):
        specifications.append((f"media_{index}_{expected_digest[:12]}", path, "reject"))

    snapshot: dict[str, SimpleNamespace] = {}
    seen: dict[tuple[Path, str], str] = {}
    for label, raw_path, policy in specifications:
        if policy == "reject":
            _reject_symlink_components(raw_path)
        resolved = raw_path.resolve(strict=True)
        key = (resolved, policy)
        if key in seen:
            continue
        receipt = hash_path(raw_path, symlink_policy=policy)
        if receipt.total_bytes <= 0:
            raise ValueError(f"formal GRPO frozen input is empty: {label}")
        snapshot[label] = SimpleNamespace(
            path=raw_path.absolute(),
            resolved=resolved,
            policy=policy,
            receipt=receipt,
        )
        seen[key] = label
    return snapshot


def _verify_formal_input_snapshot(snapshot: Mapping[str, SimpleNamespace]) -> None:
    for label, frozen in snapshot.items():
        try:
            if Path(frozen.path).resolve(strict=True) != frozen.resolved:
                raise ValueError("resolved target changed")
            observed = hash_path(Path(frozen.path), symlink_policy=frozen.policy)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"formal GRPO frozen input changed or disappeared during Swift: {label}"
            ) from exc
        if observed != frozen.receipt:
            raise ValueError(f"formal GRPO frozen input changed during Swift: {label}")


def _validate_group_aware_batching(
    config: Mapping[str, Any],
    checked_rows: List[Tuple[Path, int, Mapping[str, Any]]],
) -> tuple[tuple[str, ...], ...]:
    """Validate the exact ordered prompt batches Swift is configured to generate."""

    if "motion_vm_v_bonus" not in _reward_names(config):
        raise ValueError("formal GRPO requires motion_vm_v_bonus runtime co-location proof")
    data_cfg = _get_section(config, "data")
    training_cfg = _get_section(config, "training")
    grpo_cfg = _get_section(config, "grpo")
    if data_cfg.get("dataset_shuffle") is not False:
        raise ValueError("formal group-aware GRPO requires data.dataset_shuffle=false")
    train_batch = training_cfg.get("per_device_train_batch_size")
    if isinstance(train_batch, bool) or not isinstance(train_batch, int) or train_batch <= 0:
        raise ValueError("training.per_device_train_batch_size must be a positive integer")
    num_generations = grpo_cfg.get("num_generations")
    generation_batch_size = grpo_cfg.get("generation_batch_size")
    if (
        isinstance(num_generations, bool)
        or not isinstance(num_generations, int)
        or num_generations <= 0
    ):
        raise ValueError("formal GRPO requires positive grpo.num_generations")
    if (
        isinstance(generation_batch_size, bool)
        or not isinstance(generation_batch_size, int)
        or generation_batch_size <= 0
        or generation_batch_size % num_generations != 0
    ):
        raise ValueError(
            "formal GRPO requires generation_batch_size divisible by num_generations"
        )
    prompt_batch_size = generation_batch_size // num_generations
    if prompt_batch_size < 2:
        raise ValueError(
            "generation_batch_size must co-locate at least two prompts; "
            "per_device_train_batch_size=1 alone cannot pair VM/V"
        )
    if prompt_batch_size % 2 != 0:
        raise ValueError(
            "generation_batch_size / num_generations must be an even prompt count "
            "so canonical VM/V pairs cannot cross a generation-batch boundary"
        )

    ordered_groups: list[tuple[str, int]] = []
    closed: set[str] = set()
    current: str | None = None
    count = 0
    for _, _, record in checked_rows:
        group_id = RewardMetadata.from_mapping(record).group_id
        if group_id != current:
            if current is not None:
                ordered_groups.append((current, count))
                closed.add(current)
            if group_id in closed:
                raise ValueError(f"GRPO group {group_id!r} is not contiguous in loader order")
            current, count = group_id, 1
        else:
            count += 1
    if current is not None:
        ordered_groups.append((current, count))
    if not ordered_groups:
        raise ValueError("group-aware batching has no local records")

    batches: list[tuple[str, ...]] = []
    pending: list[str] = []
    pending_count = 0
    for group_id, group_size in ordered_groups:
        if group_size != 2:
            raise ValueError(
                f"canonical group {group_id!r} must be one VM/V prompt pair, got {group_size}"
            )
        if group_size > prompt_batch_size:
            raise ValueError(
                f"group {group_id!r} has {group_size} prompts and cannot fit one reward call"
            )
        if pending_count and pending_count + group_size > prompt_batch_size:
            batches.append(tuple(pending))
            pending, pending_count = [], 0
        pending.append(group_id)
        pending_count += group_size
        if pending_count == prompt_batch_size:
            batches.append(tuple(pending))
            pending, pending_count = [], 0
    if pending:
        batches.append(tuple(pending))
    return tuple(batches)


def _prepare_colocation_launch(
    config: Mapping[str, Any],
    *,
    artifact_arguments: SimpleNamespace,
    config_path: Path,
    train_path: Path,
    checked_rows: List[Tuple[Path, int, Mapping[str, Any]]],
    launch_env: Dict[str, str],
) -> SimpleNamespace:
    batches = _validate_group_aware_batching(config, checked_rows)
    run_cfg = _get_section(config, "run")
    expected_calls = run_cfg.get("colocation_expected_calls")
    if (
        isinstance(expected_calls, bool)
        or not isinstance(expected_calls, int)
        or expected_calls <= 0
    ):
        raise ValueError(
            "formal GRPO requires positive run.colocation_expected_calls so every "
            "reward invocation is frozen before launch"
        )
    receipt_raw = run_cfg.get("colocation_receipt_path")
    if receipt_raw in (None, ""):
        raise ValueError("formal GRPO requires run.colocation_receipt_path")
    root = Path(artifact_arguments.artifact_root).resolve(strict=True)
    receipt_path = resolve_within_root(
        receipt_raw, root, must_exist=False, allow_root=False
    )
    if receipt_path.exists():
        raise ValueError("runtime co-location receipt path must be fresh before launch")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    configured_env = _get_section(config, "run").get("env", {}) or {}
    if not isinstance(configured_env, Mapping):
        raise ValueError("run.env must be a mapping")
    overridden = sorted(COLOCATION_ENV_KEYS.intersection(str(key) for key in configured_env))
    if overridden:
        raise ValueError(f"run.env cannot override reserved co-location keys: {overridden}")
    num_generations = _get_section(config, "grpo").get("num_generations")
    if isinstance(num_generations, bool) or not isinstance(num_generations, int):
        raise ValueError("formal GRPO requires positive grpo.num_generations")
    records_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for _, _, record in checked_rows:
        group_id = RewardMetadata.from_mapping(record).group_id
        records_by_group.setdefault(group_id, []).append(record)
    call_templates: list[tuple[RewardMetadata, ...]] = []
    for batch in batches:
        records = [record for group_id in batch for record in records_by_group[group_id]]
        expanded: list[RewardMetadata] = []
        for record in records:
            for _ in range(num_generations):
                expanded.append(
                    RewardMetadata.from_mapping(record, generation_id=len(expanded))
                )
        call_templates.append(tuple(expanded))
    if not call_templates:  # pragma: no cover - guarded by batching validation
        raise RuntimeError("formal GRPO co-location plan has no call templates")
    planned_calls = tuple(
        call_templates[index % len(call_templates)] for index in range(expected_calls)
    )
    binding = SimpleNamespace(
        path=receipt_path,
        nonce=secrets.token_hex(16),
        dataset_digest=hash_path(train_path, symlink_policy="reject").digest,
        config_digest=hash_path(config_path, symlink_policy="reject").digest,
        batches=batches,
    )
    payload = initialize_runtime_colocation_plan(
        binding.path,
        nonce=binding.nonce,
        dataset_digest=binding.dataset_digest,
        config_digest=binding.config_digest,
        planned_calls=planned_calls,
    )
    binding.plan_digest = payload["plan"]["plan_sha256"]
    launch_env[COLOCATION_PATH_ENV] = str(binding.path)
    launch_env[COLOCATION_NONCE_ENV] = binding.nonce
    launch_env[COLOCATION_DATASET_ENV] = binding.dataset_digest
    launch_env[COLOCATION_CONFIG_ENV] = binding.config_digest
    launch_env[COLOCATION_PLAN_ENV] = binding.plan_digest
    return binding


def _resolve_grpo_artifact_path(
    config: Mapping[str, Any], artifact_arguments: SimpleNamespace
) -> Path:
    run_cfg = _get_section(config, "run")
    artifact_raw = run_cfg.get("artifact_path")
    if artifact_raw in (None, ""):
        raise ValueError("formal GRPO requires run.artifact_path for exact reload/publication")
    root = Path(artifact_arguments.artifact_root).resolve(strict=True)
    artifact_candidate = Path(str(artifact_raw))
    if not artifact_candidate.is_absolute():
        artifact_candidate = root / artifact_candidate
    _reject_symlink_components(artifact_candidate)
    artifact = resolve_within_root(
        artifact_raw, root, must_exist=True, allow_root=False
    )
    if not artifact.is_dir():
        raise ValueError("formal GRPO artifact_path must be a saved model/adapter directory")
    output = resolve_within_root(
        run_cfg.get("output_dir"), root, must_exist=True, allow_root=False
    )
    if artifact != output and output not in artifact.parents:
        raise ValueError(
            "formal GRPO artifact_path must be run.output_dir or one explicit descendant"
        )
    artifact_receipt = hash_path(artifact, symlink_policy="reject")
    if artifact_receipt.kind != "directory" or artifact_receipt.total_bytes <= 0:
        raise ValueError("formal GRPO artifact leaf must be one non-empty directory")
    adapter_config = artifact / "adapter_config.json"
    if not adapter_config.is_file() or adapter_config.stat().st_size <= 0:
        raise ValueError(
            "formal GRPO artifact leaf is missing non-empty adapter_config.json"
        )
    safe_weights = artifact / ADAPTER_SAFE_WEIGHTS_NAME
    unsafe_weights = artifact / "adapter_model.bin"
    if unsafe_weights.exists() or unsafe_weights.is_symlink():
        raise ValueError(
            "formal GRPO artifact leaf forbids adapter_model.bin; safetensors is mandatory"
        )
    if (
        not safe_weights.is_file()
        or safe_weights.is_symlink()
        or safe_weights.stat().st_size <= 0
    ):
        raise ValueError(
            "formal GRPO artifact leaf must contain one non-empty regular "
            "adapter_model.safetensors"
        )
    try:
        capture_adapter_checkpoint(artifact, require_swift_extension=True)
    except CheckpointBindingError as exc:
        raise ValueError(f"formal GRPO adapter checkpoint is invalid: {exc}") from exc
    return artifact


def _validate_grpo_training_evidence(config: Mapping[str, Any], artifact: Path) -> None:
    """Require exact-step trainer and optimizer evidence from the fresh Swift leaf."""

    expected_steps = _positive_integer(
        _get_section(config, "run").get("expected_optimizer_steps"),
        location="run.expected_optimizer_steps",
    )
    required_files = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    )
    for name in required_files:
        path = artifact / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"formal GRPO artifact lacks non-empty training evidence {name}")
    rng_files = tuple(artifact.glob("rng_state*.pth"))
    if not rng_files or any(not path.is_file() or path.stat().st_size <= 0 for path in rng_files):
        raise ValueError("formal GRPO artifact lacks non-empty rng_state*.pth evidence")
    trainer_state_path = artifact / "trainer_state.json"
    state = _strict_json(
        trainer_state_path.read_text(encoding="utf-8"), location=str(trainer_state_path)
    )
    if not isinstance(state, Mapping):
        raise ValueError("formal GRPO trainer_state.json must be one JSON object")
    if state.get("global_step") != expected_steps:
        raise ValueError(
            "formal GRPO trainer_state global_step differs from expected optimizer steps"
        )
    history = state.get("log_history")
    if not isinstance(history, list) or not history:
        raise ValueError("formal GRPO trainer_state must contain non-empty log_history")
    finite_loss = False
    for item in history:
        if not isinstance(item, Mapping):
            raise ValueError("formal GRPO trainer_state log_history entries must be objects")
        step = item.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            continue
        if not (0 < float(step) <= expected_steps):
            continue
        for key in ("loss", "train_loss"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise ValueError("formal GRPO trainer_state contains non-finite loss")
            finite_loss = True
    if not finite_loss:
        raise ValueError("formal GRPO trainer_state proves no finite positive-step loss")


def _validate_grpo_training_update_receipt(
    config: Mapping[str, Any],
    arguments: SimpleNamespace,
    artifact: Path,
    *,
    nonce: str,
) -> AdapterCheckpointBinding:
    """Require child-produced, nonce-bound before/after optimizer delta evidence."""

    path = artifact / "grpo_training_receipt.json"
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("formal GRPO artifact lacks grpo_training_receipt.json")
    payload = _strict_json(path.read_text(encoding="utf-8"), location=str(path))
    required = {
        "schema",
        "status",
        "nonce",
        "batch_id",
        "model_registry_id",
        "expected_optimizer_steps",
        "observed_global_step",
        "optimizer_step_count",
        "finite_loss_count",
        "gradient_observation_count",
        "trainable_tensor_count",
        "changed_tensor_count",
        "initial_trainable_state_sha256",
        "final_trainable_state_sha256",
        "adapter_filename",
        "adapter_payload_sha256",
        "adapter_payload_size_bytes",
        "adapter_tensor_count",
        "frozen_embedding_tensor_count",
        "final_saveable_adapter_state_sha256",
        "adapter_config_filename",
        "adapter_config_payload_sha256",
        "adapter_config_payload_size_bytes",
        "adapter_config_semantic_sha256",
        "adapter_config_critical",
        "adapter_extension_filename",
        "adapter_extension_payload_sha256",
        "adapter_extension_payload_size_bytes",
        "adapter_extension_semantic_sha256",
        "adapter_extension_semantics",
        "max_abs_gradient",
        "max_abs_delta",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("formal GRPO training update receipt schema differs")
    if (
        payload.get("schema") != FORMAL_TRAINING_RECEIPT_SCHEMA
        or payload.get("status") != "optimizer_and_checkpoint_verified"
        or not secrets.compare_digest(str(payload.get("nonce", "")), nonce)
        or payload.get("batch_id") != arguments.batch_id
        or payload.get("model_registry_id") != FORMAL_MODEL_REGISTRY_ID
    ):
        raise ValueError("formal GRPO training update receipt binding differs")
    expected = _positive_integer(
        _get_section(config, "run").get("expected_optimizer_steps"),
        location="run.expected_optimizer_steps",
    )
    for key in (
        "expected_optimizer_steps",
        "observed_global_step",
        "optimizer_step_count",
        "finite_loss_count",
        "gradient_observation_count",
    ):
        observed = _positive_integer(
            payload.get(key), location=f"training receipt {key}"
        )
        if observed != expected:
            raise ValueError(
                "formal GRPO training update receipt step evidence differs"
            )
    for key in ("trainable_tensor_count", "changed_tensor_count"):
        _positive_integer(payload.get(key), location=f"training receipt {key}")
    if payload["changed_tensor_count"] > payload["trainable_tensor_count"]:
        raise ValueError("formal GRPO training receipt changed tensor count is impossible")
    initial_hash = _require_lower_sha256(
        payload.get("initial_trainable_state_sha256"), location="initial trainable state"
    )
    final_hash = _require_lower_sha256(
        payload.get("final_trainable_state_sha256"), location="final trainable state"
    )
    if initial_hash == final_hash:
        raise ValueError("formal GRPO training receipt proves no parameter-state change")
    for key in ("max_abs_gradient", "max_abs_delta"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"formal GRPO training receipt {key} must be finite positive")
    if payload.get("adapter_filename") != ADAPTER_SAFE_WEIGHTS_NAME:
        raise ValueError("formal GRPO training receipt adapter filename differs")
    adapter_payload_hash = _require_lower_sha256(
        payload.get("adapter_payload_sha256"), location="adapter payload"
    )
    saveable_state_hash = _require_lower_sha256(
        payload.get("final_saveable_adapter_state_sha256"),
        location="final saveable adapter state",
    )
    adapter_payload_size = _positive_integer(
        payload.get("adapter_payload_size_bytes"),
        location="training receipt adapter_payload_size_bytes",
    )
    adapter_tensor_count = _positive_integer(
        payload.get("adapter_tensor_count"),
        location="training receipt adapter_tensor_count",
    )
    frozen_embedding_tensor_count = payload.get("frozen_embedding_tensor_count")
    if (
        isinstance(frozen_embedding_tensor_count, bool)
        or not isinstance(frozen_embedding_tensor_count, int)
        or frozen_embedding_tensor_count < 0
    ):
        raise ValueError(
            "training receipt frozen_embedding_tensor_count must be a non-negative integer"
        )
    if adapter_tensor_count != (
        payload["trainable_tensor_count"] + frozen_embedding_tensor_count
    ):
        raise ValueError(
            "formal GRPO training receipt saveable/trainable tensor counts differ"
        )
    if payload.get("adapter_config_filename") != ADAPTER_CONFIG_NAME:
        raise ValueError("formal GRPO training receipt adapter config filename differs")
    adapter_config_payload_hash = _require_lower_sha256(
        payload.get("adapter_config_payload_sha256"), location="adapter config payload"
    )
    adapter_config_semantic_hash = _require_lower_sha256(
        payload.get("adapter_config_semantic_sha256"),
        location="adapter config semantics",
    )
    adapter_config_payload_size = _positive_integer(
        payload.get("adapter_config_payload_size_bytes"),
        location="training receipt adapter_config_payload_size_bytes",
    )
    adapter_config_critical = payload.get("adapter_config_critical")
    if not isinstance(adapter_config_critical, Mapping):
        raise ValueError("formal GRPO training receipt adapter config critical fields differ")
    if payload.get("adapter_extension_filename") != ADDITIONAL_CONFIG_NAME:
        raise ValueError("formal GRPO training receipt adapter extension filename differs")
    adapter_extension_payload_hash = _require_lower_sha256(
        payload.get("adapter_extension_payload_sha256"),
        location="adapter extension payload",
    )
    adapter_extension_semantic_hash = _require_lower_sha256(
        payload.get("adapter_extension_semantic_sha256"),
        location="adapter extension semantics",
    )
    adapter_extension_payload_size = _positive_integer(
        payload.get("adapter_extension_payload_size_bytes"),
        location="training receipt adapter_extension_payload_size_bytes",
    )
    adapter_extension_semantics = payload.get("adapter_extension_semantics")
    if not isinstance(adapter_extension_semantics, Mapping):
        raise ValueError(
            "formal GRPO training receipt adapter extension semantics differ"
        )
    try:
        disk_binding = capture_adapter_checkpoint(
            artifact, require_swift_extension=True
        )
    except CheckpointBindingError as exc:
        raise ValueError(
            f"formal GRPO training receipt cannot recapture exact adapter payload: {exc}"
        ) from exc
    extension_binding = disk_binding.extension_config
    if extension_binding is None:  # pragma: no cover - capture invariant
        raise ValueError("formal GRPO checkpoint lacks its Swift extension binding")
    if (
        disk_binding.filename != payload["adapter_filename"]
        or disk_binding.payload_sha256 != adapter_payload_hash
        or disk_binding.payload_size_bytes != adapter_payload_size
        or disk_binding.tensor_count != adapter_tensor_count
        or disk_binding.state_sha256 != saveable_state_hash
        or disk_binding.config.filename != payload["adapter_config_filename"]
        or disk_binding.config.payload_sha256 != adapter_config_payload_hash
        or disk_binding.config.payload_size_bytes != adapter_config_payload_size
        or disk_binding.config.semantic_sha256 != adapter_config_semantic_hash
        or adapter_config_critical_fields(disk_binding.config.semantics)
        != adapter_config_critical
        or extension_binding.filename != payload["adapter_extension_filename"]
        or extension_binding.payload_sha256 != adapter_extension_payload_hash
        or extension_binding.payload_size_bytes != adapter_extension_payload_size
        or extension_binding.semantic_sha256 != adapter_extension_semantic_hash
        or extension_binding.semantics != adapter_extension_semantics
    ):
        raise ValueError(
            "formal GRPO training receipt does not bind the current exact checkpoint payload"
        )
    _validate_formal_adapter_config_semantics(
        config, disk_binding.config.semantics
    )
    _validate_formal_swift_extension_semantics(
        config, extension_binding.semantics
    )
    return disk_binding


def _validate_formal_output_destinations(
    config: Mapping[str, Any], artifact_arguments: SimpleNamespace
) -> None:
    run_cfg = _get_section(config, "run")
    output_raw = run_cfg.get("output_dir")
    artifact_raw = run_cfg.get("artifact_path")
    if output_raw in (None, "") or artifact_raw in (None, ""):
        raise ValueError("formal GRPO requires run.output_dir and run.artifact_path")
    root = Path(artifact_arguments.artifact_root).resolve(strict=True)
    output_candidate = Path(str(output_raw))
    artifact_candidate = Path(str(artifact_raw))
    if not output_candidate.is_absolute():
        output_candidate = root / output_candidate
    if not artifact_candidate.is_absolute():
        artifact_candidate = root / artifact_candidate
    _reject_symlink_components(output_candidate)
    _reject_symlink_components(artifact_candidate)
    output = resolve_within_root(output_raw, root, must_exist=False, allow_root=False)
    artifact = resolve_within_root(artifact_raw, root, must_exist=False, allow_root=False)
    if run_cfg.get("add_version") is not False:
        raise ValueError(
            "formal GRPO requires run.add_version=false; implicit version/timestamp leaves "
            "cannot be proven or auto-discovered"
        )
    if output != artifact and output not in artifact.parents:
        raise ValueError(
            "formal GRPO artifact_path must be run.output_dir or one explicit descendant"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("formal GRPO output directory must be fresh and empty")
    if artifact != output and artifact.exists():
        raise ValueError("formal GRPO artifact leaf must not exist before launch")
    for name in (
        "base_artifact_path",
        "train_data_path",
        "validation_data_path",
        "benchmark_path",
        "leakage_audit_path",
        "config_path",
        "code_path",
        "runner_code_path",
        "environment_path",
        "motion_vqvae_asset_path",
        "resume_manifest",
    ):
        raw = getattr(artifact_arguments, name, None)
        if raw in (None, ""):
            continue
        source = Path(raw).resolve(strict=True)
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                f"formal GRPO output/artifact overlaps immutable {name} source"
            )
    for name, raw in (
        ("reload_receipt_path", artifact_arguments.reload_receipt_path),
        ("colocation_receipt_path", run_cfg.get("colocation_receipt_path")),
        ("artifact_manifest_path", artifact_arguments.artifact_manifest_path),
    ):
        if raw in (None, ""):
            continue
        destination = resolve_within_root(raw, root, must_exist=False, allow_root=False)
        if (
            destination == output
            or output in destination.parents
            or destination == artifact
            or artifact in destination.parents
        ):
            raise ValueError(f"{name} must be outside the artifact directory being hashed")


def _model_reload_inputs(config: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    model_cfg = _get_section(config, "model")
    family = model_cfg.get("model_family")
    model_path = model_cfg.get("model")
    if not isinstance(family, str) or not family:
        raise ValueError("formal GRPO reload requires explicit model.model_family")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("formal GRPO reload requires model.model")
    kwargs = model_cfg.get("model_kwargs", {}) or {}
    if not isinstance(kwargs, Mapping):
        raise ValueError("model.model_kwargs must be a mapping")
    return family, model_path, dict(kwargs)


def _motion_token_ids(bundle: Any) -> tuple[int, int]:
    tokenizer = getattr(bundle.processor, "tokenizer", None)
    if tokenizer is None:
        raise ArtifactValidationError("reloaded processor must expose tokenizer")
    if bundle.spec.supports_motion:
        receipt = setup_motion_tokens(tokenizer, bundle.model)
        return receipt.motion_start_token_id, receipt.motion_end_token_id
    fallback = getattr(tokenizer, "eos_token_id", 0)
    if isinstance(fallback, bool) or not isinstance(fallback, int) or fallback < 0:
        fallback = 0
    return fallback, fallback


def _expected_modules_to_save(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _get_section(config, "training").get("modules_to_save")
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw]
    else:
        values = []
    modules = tuple(item for item in values if item)
    if not modules or len(set(modules)) != len(modules):
        raise ValueError(
            "formal GRPO LoRA requires non-empty unique training.modules_to_save"
        )
    return modules


def _validate_formal_adapter_config_semantics(
    config: Mapping[str, Any], semantics: Mapping[str, Any]
) -> None:
    """Bind pinned PEFT disk semantics to the exact formal Swift arguments."""

    train_cfg = _get_section(config, "training")
    model_cfg = _get_section(config, "model")
    expected_direct = {
        "peft_type": "LORA",
        "peft_version": "0.18.0",
        "task_type": "CAUSAL_LM",
        "r": train_cfg.get("lora_rank"),
        "lora_alpha": train_cfg.get("lora_alpha"),
        "lora_dropout": train_cfg.get("lora_dropout"),
        "modules_to_save": sorted(_expected_modules_to_save(config)),
        "bias": train_cfg.get("lora_bias"),
        "use_rslora": train_cfg.get("use_rslora"),
        "use_dora": train_cfg.get("use_dora"),
        "lora_bias": False,
    }
    differing = sorted(
        key for key, expected in expected_direct.items() if semantics.get(key) != expected
    )
    if differing:
        raise ValueError(
            "formal GRPO adapter config differs from frozen YAML/runtime: "
            f"fields={differing}"
        )
    expected_defaults = {
        "inference_mode": True,
        "auto_mapping": None,
        "revision": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "trainable_token_indices": None,
        "loftq_config": {},
        "eva_config": None,
        "corda_config": None,
        "use_qalora": False,
        "qalora_group_size": 16,
        "layer_replication": None,
        "target_parameters": None,
        "alora_invocation_tokens": None,
        "arrow_config": None,
        "ensure_weight_tying": False,
    }
    nondefault = sorted(
        key for key, expected in expected_defaults.items() if semantics.get(key) != expected
    )
    if nondefault:
        raise ValueError(
            "formal GRPO adapter config enables an unapproved PEFT feature: "
            f"fields={nondefault}"
        )
    target_modules = semantics.get("target_modules")
    if target_modules in (None, [], "", "all-linear"):
        raise ValueError(
            "formal GRPO adapter config must contain the non-empty target set expanded from all-linear"
        )
    base_raw = semantics.get("base_model_name_or_path")
    model_raw = model_cfg.get("model")
    if not isinstance(base_raw, str) or not isinstance(model_raw, str):
        raise ValueError("formal GRPO adapter config lacks its canonical local base path")
    try:
        adapter_base = Path(base_raw).resolve(strict=True)
        configured_base = Path(model_raw).resolve(strict=True)
    except OSError as exc:
        raise ValueError("formal GRPO adapter config base path cannot be resolved") from exc
    if adapter_base != configured_base:
        raise ValueError("formal GRPO adapter config base path differs from model.model")


def _validate_formal_swift_extension_semantics(
    config: Mapping[str, Any], semantics: Mapping[str, Any]
) -> None:
    """Bind ms-swift's sidecar-only LoRA fields to the frozen YAML."""

    train_cfg = _get_section(config, "training")
    expected = {
        "lora_dtype": train_cfg.get("lora_dtype"),
        "lorap_lr_ratio": train_cfg.get("lorap_lr_ratio"),
        "lorap_emb_lr": train_cfg.get("lorap_emb_lr"),
    }
    if set(semantics) != set(expected) or dict(semantics) != expected:
        differing = sorted(
            key
            for key in set(expected) | set(semantics)
            if expected.get(key) != semantics.get(key)
            or (key in expected) != (key in semantics)
        )
        raise ValueError(
            "formal GRPO Swift LoRA extension differs from frozen YAML: "
            f"fields={differing}"
        )


def _require_modules_in_adapter_state(
    state: Mapping[str, Any], expected_modules: tuple[str, ...]
) -> None:
    names = tuple(str(name).casefold() for name in state)
    missing = []
    for module in expected_modules:
        marker = module.casefold()
        if not any(
            name == marker
            or name.startswith(marker + ".")
            or name.endswith("." + marker)
            or f".{marker}." in name
            for name in names
        ):
            missing.append(module)
    if missing:
        raise ArtifactValidationError(
            f"GRPO adapter state is missing expected modules_to_save: {missing}"
        )


def _load_grpo_lora_states(
    config: Mapping[str, Any], artifact_path: Path
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[int, int],
    tuple[str, str, str],
]:
    try:
        from peft import PeftModel
        from peft.utils.save_and_load import load_peft_weights
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("formal GRPO LoRA reload requires peft") from exc
    family, model_path, model_kwargs = _model_reload_inputs(config)
    captured_disk = capture_adapter_checkpoint(
        artifact_path, require_swift_extension=True
    )
    extension_binding = captured_disk.extension_config
    if extension_binding is None:  # pragma: no cover - capture invariant
        raise ArtifactValidationError(
            "saved GRPO adapter lacks its Swift LoRA extension binding"
        )
    _validate_formal_swift_extension_semantics(
        config, extension_binding.semantics
    )
    disk_state = load_peft_weights(str(artifact_path), device="cpu")
    if not isinstance(disk_state, Mapping) or not disk_state:
        raise ArtifactValidationError("saved GRPO adapter state is empty")
    try:
        loaded_disk_hash = require_identical_adapter_states(
            captured_disk.state, disk_state
        )
    except CheckpointBindingError as exc:
        raise ArtifactValidationError(
            f"PEFT load_peft_weights differs from exact safetensors payload: {exc}"
        ) from exc
    if loaded_disk_hash != captured_disk.state_sha256:
        raise ArtifactValidationError(
            "PEFT load_peft_weights canonical state hash differs from exact payload"
        )
    _validate_grpo_adapter_update_state(disk_state)
    bundle = default_model_factory.load_bundle(
        family=family,
        model_name_or_path=model_path,
        model_kwargs=model_kwargs,
    )
    token_ids = _motion_token_ids(bundle)
    reloaded = PeftModel.from_pretrained(
        bundle.model,
        str(artifact_path),
        is_trainable=False,
    )
    try:
        reloaded_state = extract_peft_adapter_state(reloaded)
        reloaded_config = live_peft_adapter_config_semantics(
            reloaded, expect_trainable=False
        )
        reloaded_config_hash = require_identical_adapter_configs(
            reloaded_config, captured_disk.config.semantics
        )
    except CheckpointBindingError as exc:
        raise ArtifactValidationError(
            f"freshly reloaded GRPO adapter/config differs from exact disk: {exc}"
        ) from exc
    if reloaded_config_hash != captured_disk.config.semantic_sha256:
        raise ArtifactValidationError(
            "freshly reloaded GRPO adapter config semantic hash differs"
        )
    _validate_formal_adapter_config_semantics(config, reloaded_config)
    if not isinstance(reloaded_state, Mapping) or not reloaded_state:
        raise ArtifactValidationError("freshly reloaded GRPO adapter state is empty")
    validate_lora_adapter_pairs(disk_state)
    validate_lora_adapter_pairs(reloaded_state)
    expected_modules = _expected_modules_to_save(config)
    _require_modules_in_adapter_state(disk_state, expected_modules)
    _require_modules_in_adapter_state(reloaded_state, expected_modules)
    peft_configs = getattr(reloaded, "peft_config", None)
    if not isinstance(peft_configs, Mapping) or not peft_configs:
        raise ArtifactValidationError("fresh GRPO PEFT reload exposes no adapter config")
    for adapter_name, peft_config in peft_configs.items():
        configured = tuple(sorted(getattr(peft_config, "modules_to_save", ()) or ()))
        if configured != tuple(sorted(expected_modules)):
            raise ArtifactValidationError(
                f"GRPO adapter {adapter_name!r} modules_to_save differ from frozen config"
            )
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("formal GRPO processor reload requires transformers") from exc
    reloaded_processor = transformers.AutoProcessor.from_pretrained(str(artifact_path))
    reloaded_tokenizer = getattr(reloaded_processor, "tokenizer", None)
    if reloaded_tokenizer is None:
        raise ArtifactValidationError("saved GRPO processor must expose tokenizer")
    if bundle.spec.supports_motion and verify_motion_tokens(
        reloaded_tokenizer, bundle.model
    ) != token_ids:
        raise ArtifactValidationError("saved GRPO processor motion token ids changed")
    processor_before, processor_after, processor_assets_hash = verify_processor_save_reload(
        bundle.processor,
        reloaded_processor,
        artifact_path=artifact_path,
    )
    return (
        disk_state,
        reloaded_state,
        token_ids,
        (processor_before, processor_after, processor_assets_hash),
    )


def _validate_grpo_adapter_update_state(state: Mapping[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("formal GRPO adapter update proof requires torch") from exc
    observed_lora_b = False
    changed_lora_b = False
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise ArtifactValidationError(f"saved GRPO adapter tensor is invalid: {name}")
        tensor = value.detach().cpu()
        if not bool(torch.isfinite(tensor).all().item()):
            raise ArtifactValidationError(f"saved GRPO adapter contains non-finite tensor: {name}")
        if "lora_b" in str(name).casefold():
            observed_lora_b = True
            if int(torch.count_nonzero(tensor).item()) > 0:
                changed_lora_b = True
    if not observed_lora_b or not changed_lora_b:
        raise ArtifactValidationError(
            "saved GRPO adapter has no non-zero LoRA-B update evidence"
        )


def _load_grpo_full_states(
    config: Mapping[str, Any], artifact_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, int]]:
    del config, artifact_path
    raise ArtifactValidationError(
        "formal GRPO full publication is disabled until Swift emits an independent "
        "raw-disk producer state receipt; two fresh reloads cannot prove saved state"
    )


def _generate_grpo_reload_receipt(
    config: Mapping[str, Any],
    artifact_arguments: SimpleNamespace,
    artifact_path: Path,
) -> Path:
    receipt_path = resolve_within_root(
        artifact_arguments.reload_receipt_path,
        artifact_arguments.artifact_root,
        must_exist=False,
        allow_root=False,
    )
    if receipt_path.exists():
        raise ValueError("GRPO reload receipt path must be fresh before verification")
    tuner_type = str(_get_section(config, "training").get("tuner_type", "")).casefold()
    if tuner_type in {"lora", "peft"}:
        before, after, token_ids, processor_evidence = _load_grpo_lora_states(
            config, artifact_path
        )
        expected_modules = _expected_modules_to_save(config)
    elif tuner_type == "full":
        _load_grpo_full_states(config, artifact_path)
        raise AssertionError("formal full GRPO must fail before receipt generation")
    else:
        raise ValueError("formal GRPO training.tuner_type must be lora/peft or full")
    state_hash = verify_state_mapping_reload(before, after)
    artifact_digest = hash_path(artifact_path, symlink_policy="reject").digest
    receipt = ReloadVerificationReceipt(
        batch_id=artifact_arguments.batch_id,
        model_id=artifact_arguments.model_registry_id,
        artifact_hash=artifact_digest,
        expected_modules=expected_modules,
        reloaded_modules=expected_modules,
        motion_start_token_id=token_ids[0],
        motion_end_token_id=token_ids[1],
        state_hash_before=state_hash,
        state_hash_after=state_hash,
        processor_state_hash_before=processor_evidence[0],
        processor_state_hash_after=processor_evidence[1],
        processor_assets_hash=processor_evidence[2],
    )
    return write_reload_verification_receipt(
        receipt_path,
        receipt,
        allowed_root=artifact_arguments.artifact_root,
        overwrite=False,
    )


def _run_swift_safely(command: List[str], *, env: Dict[str, str]) -> None:
    safe_command = shlex.join(redact_command_for_log(command))
    try:
        subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"swift GRPO failed with exit code {exc.returncode}; command={safe_command}"
        ) from None
    except OSError as exc:
        raise RuntimeError(
            f"swift GRPO could not start ({type(exc).__name__}); command={safe_command}"
        ) from None
    except Exception as exc:
        raise RuntimeError(
            f"swift GRPO execution failed ({type(exc).__name__}); command={safe_command}"
        ) from None


def _isolated_formal_swift_command(
    command: List[str], *, python_path: Path, swift_path: Path
) -> List[str]:
    """Run the verified Swift entry point through this interpreter with -I/-B."""

    if not command or command[0] != "swift":
        raise RuntimeError("formal GRPO expected the canonical Swift command prefix")
    actual_python = Path(sys.executable).resolve(strict=True)
    bound_python = Path(python_path).resolve(strict=True)
    if bound_python != actual_python:
        raise RuntimeError("formal GRPO isolated Swift interpreter binding differs")
    verified_swift = Path(swift_path).resolve(strict=True)
    if not verified_swift.is_file() or verified_swift.stat().st_size <= 0:
        raise RuntimeError("formal GRPO isolated Swift entry point is invalid")
    return [str(bound_python), "-I", "-B", str(verified_swift), *command[1:]]


def _build_launch_env(
    config: Mapping[str, Any], *, formal_artifact: bool = False
) -> Tuple[Dict[str, str], Dict[str, str]]:
    run_cfg = _get_section(config, "run")
    env_cfg = run_cfg.get("env", {})
    if env_cfg is None:
        env_cfg = {}
    if not isinstance(env_cfg, Mapping):
        raise ValueError("`run.env` must be a mapping of environment variable names to values.")

    env = dict(os.environ)
    for key, value in env_cfg.items():
        env_key = str(key)
        if value is None:
            env.pop(env_key, None)
            continue
        env_value = str(value)
        env[env_key] = env_value
    if formal_artifact:
        # The verified Swift entry point is run with ``python -I -B`` below.
        # Also sanitize the inherited environment so any worker Python
        # processes spawned by Swift/Accelerate inherit the same user-site,
        # unsafe-path and bytecode protections.  CUDA library discovery is
        # intentionally preserved (for example LD_LIBRARY_PATH).
        for key in tuple(env):
            if key.upper().startswith("PYTHON"):
                env.pop(key, None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"
    return env, describe_environment_overrides(env_cfg)


def _normalized_distribution_name(value: str) -> str:
    return value.strip().casefold().replace("_", "-")


def _public_package_version(name: str, value: str) -> str:
    # CUDA PyTorch wheels may expose a local suffix (for example +cu128).  The
    # CUDA runtime is checked independently and fail-closed below.
    return value.split("+", 1)[0] if _normalized_distribution_name(name) == "torch" else value


def _read_pinned_environment(path: Path) -> Mapping[str, str]:
    packages: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            # A formal snapshot may include index annotations, but critical API
            # packages must appear below as exact name==version records.
            continue
        name, version = line.split("==", 1)
        normalized = _normalized_distribution_name(name)
        if not normalized or not version.strip():
            raise ValueError(f"invalid environment pin at {path}:{line_no}")
        if normalized in packages:
            raise ValueError(f"duplicate environment pin for {normalized!r} at {path}:{line_no}")
        packages[normalized] = version.strip()
    return packages


def _load_formal_runtime_contract(
    config: Mapping[str, Any], artifact_arguments: SimpleNamespace
) -> Mapping[str, Any]:
    run_cfg = _get_section(config, "run")
    raw_path = run_cfg.get("runtime_contract_path")
    if raw_path in (None, ""):
        raise ValueError("formal GRPO requires run.runtime_contract_path")
    contract_path = resolve_within_root(
        raw_path, REPO_ROOT, must_exist=True, allow_root=False
    )
    payload = _strict_json(
        contract_path.read_text(encoding="utf-8"), location=str(contract_path)
    )
    if not isinstance(payload, Mapping):
        raise ValueError("formal GRPO runtime contract must be a JSON object")
    if payload.get("schema") != FORMAL_RUNTIME_SCHEMA:
        raise ValueError(
            f"formal GRPO runtime contract schema must be {FORMAL_RUNTIME_SCHEMA!r}"
        )
    python_major_minor = payload.get("python_major_minor")
    if not isinstance(python_major_minor, str) or not python_major_minor:
        raise ValueError("formal GRPO runtime contract requires python_major_minor")
    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or not packages:
        raise ValueError("formal GRPO runtime contract requires exact package pins")
    normalized_packages: dict[str, str] = {}
    for name, version in packages.items():
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise ValueError("formal GRPO runtime package pins must be name/version strings")
        normalized = _normalized_distribution_name(name)
        if normalized in normalized_packages:
            raise ValueError(f"duplicate runtime package pin: {normalized}")
        normalized_packages[normalized] = version
    required_packages = {
        "accelerate",
        "ms-swift",
        "transformers",
        "peft",
        "safetensors",
        "torch",
    }
    missing_packages = sorted(required_packages - set(normalized_packages))
    if missing_packages:
        raise ValueError(
            f"formal GRPO runtime contract misses critical packages: {missing_packages}"
        )
    if payload.get("swift_cli") != "swift":
        raise ValueError("formal GRPO runtime contract must freeze swift_cli='swift'")
    cuda_prefix = payload.get("cuda_runtime_prefix")
    if not isinstance(cuda_prefix, str) or not cuda_prefix:
        raise ValueError("formal GRPO runtime contract requires cuda_runtime_prefix")
    api_checks = payload.get("api_checks")
    if not isinstance(api_checks, list) or not api_checks:
        raise ValueError("formal GRPO runtime contract requires explicit api_checks")
    for index, check in enumerate(api_checks):
        if not isinstance(check, Mapping):
            raise ValueError(f"runtime api_checks[{index}] must be an object")
        module_name = check.get("module")
        attributes = check.get("attributes")
        signatures = check.get("signatures", {})
        if (
            not isinstance(module_name, str)
            or not module_name
            or not isinstance(attributes, list)
            or not attributes
            or not all(isinstance(name, str) and name for name in attributes)
            or not isinstance(signatures, Mapping)
        ):
            raise ValueError(
                f"runtime api_checks[{index}] requires module and non-empty attributes"
            )
        for attribute_name, parameter_names in signatures.items():
            if (
                not isinstance(attribute_name, str)
                or attribute_name not in attributes
                or not isinstance(parameter_names, list)
                or not parameter_names
                or not all(
                    isinstance(parameter_name, str) and parameter_name
                    for parameter_name in parameter_names
                )
                or len(set(parameter_names)) != len(parameter_names)
            ):
                raise ValueError(
                    f"runtime api_checks[{index}] has an invalid frozen signature"
                )

    environment_raw = artifact_arguments.environment_path
    if not isinstance(environment_raw, str) or not environment_raw:
        raise ValueError("formal GRPO provenance.environment_path is required")
    environment_path = Path(environment_raw).resolve(strict=True)
    if not environment_path.is_file() or environment_path.stat().st_size <= 0:
        raise ValueError("formal GRPO environment snapshot must be one non-empty file")
    environment_packages = _read_pinned_environment(environment_path)
    for name, expected in normalized_packages.items():
        observed = environment_packages.get(name)
        if observed is None:
            raise ValueError(
                f"formal GRPO environment snapshot is missing exact pin for {name}"
            )
        if _public_package_version(name, observed) != expected:
            raise ValueError(
                f"formal GRPO environment snapshot pin differs for {name}: "
                f"expected {expected}, observed {observed}"
            )
    return payload


def _require_bound_interpreter(env: Mapping[str, str]) -> Path:
    configured = env.get(FORMAL_PYTHON_ENV)
    if not isinstance(configured, str) or not configured.strip():
        raise RuntimeError(
            f"GRPO runtime preflight requires absolute {FORMAL_PYTHON_ENV}; "
            "launch through scripts/train_grpo_ms_swift.sh"
        )
    candidate = Path(configured)
    if not candidate.is_absolute():
        raise RuntimeError(f"{FORMAL_PYTHON_ENV} must be an absolute interpreter path")
    try:
        configured_python = candidate.resolve(strict=True)
        actual_python = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("GRPO bound Python interpreter does not exist") from exc
    if configured_python != actual_python:
        raise RuntimeError(
            f"GRPO runner interpreter differs from {FORMAL_PYTHON_ENV}; "
            "refusing a mixed environment"
        )
    return actual_python


def _require_same_environment_swift(env: Mapping[str, str], python_path: Path) -> Path:
    swift_raw = shutil.which("swift", path=env.get("PATH"))
    if not swift_raw:
        raise RuntimeError(
            "GRPO runtime preflight failed before launch: swift CLI is missing from PATH"
        )
    swift_path = Path(swift_raw).resolve(strict=True)
    prefix = Path(sys.prefix).resolve(strict=True)
    allowed_parents = {
        python_path.parent,
        prefix,
        prefix / "bin",
        prefix / "Scripts",
    }
    if swift_path.parent not in allowed_parents:
        raise RuntimeError(
            "GRPO runtime preflight found swift outside the bound Python environment"
        )
    if not swift_path.is_file() or swift_path.stat().st_size <= 0:
        raise RuntimeError("GRPO runtime preflight found an invalid Swift launcher")
    if os.name != "nt":
        try:
            with swift_path.open("rb") as handle:
                first_line = handle.readline(4096).decode("utf-8").strip()
            words = shlex.split(first_line[2:]) if first_line.startswith("#!") else []
            launcher_python = Path(words[0]).resolve(strict=True) if words else None
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError("GRPO Swift launcher shebang is invalid") from exc
        if launcher_python != python_path.resolve(strict=True):
            raise RuntimeError(
                "GRPO Swift launcher shebang does not bind the runner interpreter"
            )
    return swift_path


def _validate_formal_swift_launcher_hash(
    config: Mapping[str, Any], swift_path: Path
) -> Any:
    expected = _require_lower_sha256(
        _get_section(config, "run").get("swift_launcher_sha256"),
        location="run.swift_launcher_sha256",
    )
    receipt = hash_path(swift_path, symlink_policy="reject")
    if receipt.kind != "file" or receipt.total_bytes <= 0 or receipt.digest != expected:
        raise RuntimeError(
            "formal GRPO Swift launcher bytes differ from run.swift_launcher_sha256"
        )
    return receipt


def _import_runtime_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(
            f"GRPO runtime preflight failed before launch: cannot import {name} "
            f"({type(exc).__name__})"
        ) from None


def _runtime_api_attribute(module: Any, qualified_name: str) -> Any:
    value = module
    for component in qualified_name.split("."):
        if not hasattr(value, component):
            raise AttributeError(qualified_name)
        value = getattr(value, component)
    return value


def _validate_runtime_environment(
    config: Mapping[str, Any],
    artifact_arguments: SimpleNamespace,
    *,
    formal_artifact: bool,
    env: Mapping[str, str],
) -> Path:
    python_path = _require_bound_interpreter(env)
    swift_path = _require_same_environment_swift(env, python_path)
    swift_module = _import_runtime_module("swift")
    peft_module = _import_runtime_module("peft")
    torch_module = _import_runtime_module("torch")
    del swift_module, peft_module

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise RuntimeError("GRPO runtime preflight requires a CUDA-capable torch API")
    if not cuda.is_available():
        raise RuntimeError(
            "GRPO runtime preflight failed before launch: torch.cuda.is_available() is false"
        )

    if formal_artifact:
        contract = _load_formal_runtime_contract(config, artifact_arguments)
        _validate_formal_swift_launcher_hash(config, swift_path)
        try:
            swift_distribution = importlib.metadata.distribution("ms-swift")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("formal GRPO ms-swift distribution metadata is missing") from None
        swift_entry_points = [
            entry
            for entry in swift_distribution.entry_points
            if entry.group == "console_scripts" and entry.name == "swift"
        ]
        if len(swift_entry_points) != 1:
            raise RuntimeError(
                "formal GRPO requires exactly one ms-swift console_scripts entry named swift"
            )
        expected_python = str(contract["python_major_minor"])
        actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual_python != expected_python:
            raise RuntimeError(
                f"formal GRPO Python API version mismatch: expected {expected_python}, "
                f"observed {actual_python}"
            )
        for name, expected_value in dict(contract["packages"]).items():
            normalized = _normalized_distribution_name(str(name))
            try:
                observed = importlib.metadata.version(str(name))
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(
                    f"formal GRPO runtime package is missing: {name}=={expected_value}"
                ) from None
            if _public_package_version(normalized, observed) != str(expected_value):
                raise RuntimeError(
                    f"formal GRPO runtime version mismatch for {name}: "
                    f"expected {expected_value}, observed {observed}"
                )
        observed_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
        expected_cuda = str(contract["cuda_runtime_prefix"])
        if not isinstance(observed_cuda, str) or not observed_cuda.startswith(expected_cuda):
            raise RuntimeError(
                f"formal GRPO CUDA runtime mismatch: expected {expected_cuda}.x, "
                f"observed {observed_cuda!r}"
            )
        for check in contract["api_checks"]:
            module = _import_runtime_module(str(check["module"]))
            resolved: dict[str, Any] = {}
            missing: list[str] = []
            for name in check["attributes"]:
                try:
                    resolved[str(name)] = _runtime_api_attribute(module, str(name))
                except AttributeError:
                    missing.append(str(name))
            if missing:
                raise RuntimeError(
                    f"formal GRPO runtime API mismatch in {check['module']}: "
                    f"missing {missing}"
                )
            for name, expected_parameters in dict(
                check.get("signatures", {})
            ).items():
                try:
                    observed_parameters = list(
                        inspect.signature(resolved[str(name)]).parameters
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"formal GRPO runtime API signature is unavailable for "
                        f"{check['module']}.{name}"
                    ) from exc
                if observed_parameters != list(expected_parameters):
                    raise RuntimeError(
                        f"formal GRPO runtime API signature mismatch for "
                        f"{check['module']}.{name}: expected {list(expected_parameters)}, "
                        f"observed {observed_parameters}"
                    )
    print(
        "GRPO runtime preflight passed: bound interpreter, same-environment Swift/PEFT, "
        "and CUDA are available."
    )
    return swift_path


def _report_to_values(run_cfg: Mapping[str, Any]) -> List[str]:
    report_to = run_cfg.get("report_to")
    if report_to is None:
        return []
    if isinstance(report_to, str):
        values = [report_to]
    elif isinstance(report_to, (list, tuple, set)):
        values = [str(v) for v in report_to]
    else:
        values = [str(report_to)]

    normalized: List[str] = []
    for value in values:
        token = value.strip().lower()
        if token:
            normalized.append(token)
    return normalized


def _wandb_requested(config: Mapping[str, Any]) -> bool:
    run_cfg = _get_section(config, "run")
    return "wandb" in _report_to_values(run_cfg)


def _has_wandb_netrc_auth() -> bool:
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        auth = netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (NetrcParseError, OSError):
        return False
    return bool(auth and auth[2])


def _prepare_wandb_env(config: Mapping[str, Any], env: Dict[str, str]) -> Dict[str, str]:
    if not _wandb_requested(config):
        return {}

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "WandB is enabled via run.report_to but package wandb is not installed. "
            "Install with: pip install wandb"
        ) from exc

    if not env.get("WANDB_API_KEY") and not _has_wandb_netrc_auth():
        raise RuntimeError(
            "WandB is enabled via run.report_to but no auth was found. "
            "Run wandb login or set WANDB_API_KEY before launch."
        )

    run_cfg = _get_section(config, "run")
    applied: Dict[str, str] = {}

    if not env.get("WANDB_PROJECT"):
        env["WANDB_PROJECT"] = DEFAULT_WANDB_PROJECT
        applied["WANDB_PROJECT"] = DEFAULT_WANDB_PROJECT

    run_name = run_cfg.get("run_name")
    if run_name and not env.get("WANDB_NAME"):
        env["WANDB_NAME"] = str(run_name)
        applied["WANDB_NAME"] = str(run_name)

    if not env.get("WANDB_MODE"):
        env["WANDB_MODE"] = "online"
        applied["WANDB_MODE"] = "online"

    return describe_environment_overrides(applied)


def _enforce_context_expansion_guard(
    config: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    formal_artifact: bool = False,
) -> None:
    run_cfg = _get_section(config, "run")
    enabled = bool(run_cfg.get("context_expansion_guard", False))
    if not enabled:
        return

    if formal_artifact and "forbidden_context_limit_env_keys" in run_cfg:
        raise ValueError(
            "formal GRPO forbids overriding the frozen context-limit environment key set"
        )
    keys = run_cfg.get(
        "forbidden_context_limit_env_keys",
        list(DEFAULT_FORBIDDEN_CONTEXT_LIMIT_ENV_KEYS),
    )
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ValueError(
            "`run.forbidden_context_limit_env_keys` must be a string or a list of strings."
        )

    effective_env = os.environ if env is None else env
    active = sorted(key for key in keys if effective_env.get(key) not in (None, ""))
    if active:
        raise ValueError(
            "Context expansion guard found active environment limits. "
            f"Unset these env vars before launch: {active}"
        )
    print(f"Context expansion guard passed: no forbidden env limits in {keys}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Motion-r1 GRPO on ms-swift from YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Absolute path to a copied formal LoRA config derived from the formal template.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run static config/data/provenance checks and print the command; "
            "skip runtime imports."
        ),
    )
    mode.add_argument(
        "--preflight_only",
        action="store_true",
        help=(
            "Also verify the real Python/Swift/PEFT/CUDA runtime, but do not "
            "create outputs."
        ),
    )
    # Let argparse's help action run before the stricter duplicate checks.  In
    # particular, ``--help`` must remain a standard, config-free exit-0 path.
    parsed = parser.parse_args()
    raw_args = sys.argv[1:]
    config_count = sum(
        token == "--config" or token.startswith("--config=") for token in raw_args
    )
    if config_count != 1:
        parser.error("--config must be supplied exactly once")
    mode_count = sum(token in {"--dry_run", "--preflight_only"} for token in raw_args)
    if mode_count > 1:
        parser.error("at most one launch mode may be supplied")
    return parsed


def main() -> None:
    args = _parse_args()
    if not args.config.is_absolute():
        raise ValueError("formal GRPO --config must be an absolute path")
    _reject_symlink_components(args.config)
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    input_bindings: dict[Path, str] = {
        config_path: hash_path(config_path, symlink_policy="reject").digest
    }
    config = _read_yaml(config_path)
    _validate_required(config)
    _validate_grpo_training_mode(config, config_path=config_path)
    artifact_arguments = _artifact_arguments(config, config_path=config_path)
    formal_artifact = validate_artifact_policy(artifact_arguments, training_mode="grpo")
    if os.environ.get(FORMAL_LAUNCHER_ENV) == "1" and not formal_artifact:
        raise ValueError("the formal GRPO launcher refuses unsafe legacy no-manifest mode")
    if not formal_artifact:
        print(
            "WARNING: UNSAFE LEGACY NO-MANIFEST MODE; this output is ineligible "
            "for formal evaluation."
        )
    _enforce_formal_preflight_policy(config, formal_artifact=formal_artifact)
    train_path: Path | None = None
    expected_optimizer_steps: int | None = None
    if formal_artifact:
        _reject_formal_config_secrets(config)
        _validate_formal_model_identity(config)
        _bind_formal_code_identity(config, artifact_arguments)
        expected_optimizer_steps = _validate_formal_training_contract(config)
        _validate_formal_reward_policy(config)
        _enforce_formal_ambient_environment(config)
        train_path = _bind_formal_loader_inputs(
            config,
            artifact_arguments,
            config_path=config_path,
            binding_sink=input_bindings,
        )
        _validate_canonical_pretrained_assets(artifact_arguments)
        _validate_formal_leakage_audit(config, artifact_arguments)
        _load_formal_runtime_contract(config, artifact_arguments)
        _validate_formal_output_destinations(config, artifact_arguments)
        reload_target = resolve_within_root(
            artifact_arguments.reload_receipt_path,
            artifact_arguments.artifact_root,
            must_exist=False,
            allow_root=False,
        )
        if reload_target.exists():
            raise ValueError("GRPO reload receipt path must be fresh before launch")
    launch_env, env_overrides = _build_launch_env(
        config, formal_artifact=formal_artifact
    )
    _enforce_context_expansion_guard(
        config, env=launch_env, formal_artifact=formal_artifact
    )
    checked_rows = _precheck_dataset_records(
        config,
        formal_artifact=formal_artifact,
        binding_sink=input_bindings if formal_artifact else None,
    )
    if formal_artifact:
        launch_env[FORMAL_BOUND_INPUTS_ENV] = "1"
    command = _build_command(config)
    print("Resolved command:")
    print(shlex.join(redact_command_for_log(command)))
    resolved_env_overrides = dict(env_overrides)
    if resolved_env_overrides:
        print("Resolved env overrides:")
        print(json.dumps(resolved_env_overrides, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run:
        return

    frozen_inputs: Mapping[str, SimpleNamespace] | None = None
    if formal_artifact:
        frozen_inputs = _capture_formal_input_snapshot(
            artifact_arguments, media_bindings=input_bindings
        )

    swift_path = _validate_runtime_environment(
        config,
        artifact_arguments,
        formal_artifact=formal_artifact,
        env=launch_env,
    )
    if formal_artifact:
        # Do not exec the console script through its shebang: invoke the exact
        # already-running, runtime-validated interpreter in isolated mode.
        # This blocks user-site/PYTHON* path injection before Swift imports.
        bound_python = _require_bound_interpreter(launch_env)
        command = _isolated_formal_swift_command(
            command, python_path=bound_python, swift_path=swift_path
        )
    else:
        command[0] = str(swift_path)
    print("Validated launch command:")
    print(shlex.join(redact_command_for_log(command)))
    swift_launcher_receipt = (
        hash_path(swift_path, symlink_policy="reject") if formal_artifact else None
    )
    if frozen_inputs is not None:
        _verify_formal_input_snapshot(frozen_inputs)
    if args.preflight_only:
        return

    wandb_overrides = _prepare_wandb_env(config, launch_env)
    if wandb_overrides:
        print("Resolved WandB env overrides:")
        print(json.dumps(wandb_overrides, ensure_ascii=False, indent=2, sort_keys=True))
    _ensure_paths(config)

    colocation = None
    training_nonce: str | None = None
    if formal_artifact:
        if train_path is None:  # pragma: no cover - defensive formal binding guard
            raise RuntimeError("formal GRPO loader binding returned no path")
        colocation = _prepare_colocation_launch(
            config,
            artifact_arguments=artifact_arguments,
            config_path=config_path,
            train_path=train_path,
            checked_rows=checked_rows,
            launch_env=launch_env,
        )
        if frozen_inputs is None:  # pragma: no cover - defensive source binding guard
            raise RuntimeError("formal GRPO has no frozen input snapshot")
        _verify_formal_input_snapshot(frozen_inputs)
        training_nonce = secrets.token_hex(32)
        launch_env[FORMAL_TRAINING_NONCE_ENV] = training_nonce
        launch_env[FORMAL_TRAINING_BATCH_ENV] = str(artifact_arguments.batch_id)
        launch_env[FORMAL_TRAINING_STEPS_ENV] = str(expected_optimizer_steps)
        launch_env[FORMAL_TRAINING_ARTIFACT_ENV] = str(
            resolve_within_root(
                _get_section(config, "run").get("artifact_path"),
                artifact_arguments.artifact_root,
                must_exist=False,
                allow_root=False,
            )
        )

    _run_swift_safely(command, env=launch_env)
    if swift_launcher_receipt is not None and hash_path(
        swift_path, symlink_policy="reject"
    ) != swift_launcher_receipt:
        raise ValueError("formal GRPO Swift launcher changed during execution")
    run_cfg = _get_section(config, "run")
    if formal_artifact:
        if colocation is None:  # pragma: no cover - defensive formal binding guard
            raise RuntimeError("formal GRPO has no runtime co-location binding")
        if frozen_inputs is None:  # pragma: no cover - defensive source binding guard
            raise RuntimeError("formal GRPO has no frozen input snapshot")
        _verify_formal_input_snapshot(frozen_inputs)
        validate_runtime_colocation_receipt(
            colocation.path,
            nonce=colocation.nonce,
            dataset_digest=colocation.dataset_digest,
            config_digest=colocation.config_digest,
            plan_digest=colocation.plan_digest,
        )
        artifact_path = _resolve_grpo_artifact_path(config, artifact_arguments)
        if expected_optimizer_steps is None:  # pragma: no cover
            raise RuntimeError("formal GRPO has no expected optimizer-step binding")
        _validate_grpo_training_evidence(config, artifact_path)
        if training_nonce is None:  # pragma: no cover - formal launch invariant
            raise RuntimeError("formal GRPO has no training update nonce")
        _validate_grpo_training_update_receipt(
            config,
            artifact_arguments,
            artifact_path,
            nonce=training_nonce,
        )
        _generate_grpo_reload_receipt(config, artifact_arguments, artifact_path)
        # Re-open and re-hash the exact checkpoint after the independent PEFT
        # reload.  A payload replaced during reload cannot reach publication.
        _validate_grpo_training_update_receipt(
            config,
            artifact_arguments,
            artifact_path,
            nonce=training_nonce,
        )
        _verify_formal_input_snapshot(frozen_inputs)
        if hash_path(swift_path, symlink_policy="reject") != swift_launcher_receipt:
            raise ValueError("formal GRPO Swift launcher changed before publication")
    else:
        output_dir = run_cfg.get("output_dir")
        if output_dir in (None, ""):
            raise ValueError("run.output_dir is required for GRPO artifact publication")
        artifact_path = Path(str(output_dir))
    publication_receipt = write_artifact_from_arguments(
        artifact_arguments,
        training_mode="grpo",
        artifact_path=artifact_path,
    )
    if formal_artifact:
        if publication_receipt is None or training_nonce is None:
            raise RuntimeError("formal GRPO manifest publication returned no receipt binding")
        _validate_grpo_training_update_receipt(
            config,
            artifact_arguments,
            artifact_path,
            nonce=training_nonce,
        )
        manifested_artifact = hash_path(artifact_path, symlink_policy="reject")
        if manifested_artifact.digest != publication_receipt.artifact_digest:
            raise ValueError(
                "formal GRPO adapter/config changed while the manifest was published"
            )
        if frozen_inputs is None:  # pragma: no cover - formal invariant
            raise RuntimeError("formal GRPO manifest has no frozen input snapshot")
        _verify_formal_input_snapshot(frozen_inputs)


if __name__ == "__main__":
    main()
