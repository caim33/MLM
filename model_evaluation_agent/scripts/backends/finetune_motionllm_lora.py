#!/usr/bin/env python3
"""Project-owned MotionLLM LoRA/projector finetuner.

The upstream repository exposes the model and released weights but no training
entrypoint. This runner follows the upstream inference embedding path, freezes
the video tower and Vicuna base, and optimizes only LoRA plus the projector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from motion_eval.core import atomic_write_json, hash_path, sha256_json
from motion_eval.data import load_json_strict
from motion_eval.training_receipt import make_training_receipt, write_training_receipt


BACKEND_ID = "motionllm_official:lora_projector_v1"
MODEL_FAMILY = "motionllm_litgpt_video"
TRAINING_MODE = "lora_sft"

_STATE_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "language_trainable_tensor_count",
        "projector_trainable_tensor_count",
        "projector_state_tensor_count",
        "combined_trainable_tensor_count",
        "combined_trainable_parameter_count",
        "language_live_sha256",
        "language_disk_sha256",
        "language_reload_sha256",
        "projector_live_sha256",
        "projector_disk_sha256",
        "projector_reload_sha256",
        "combined_live_sha256",
        "combined_reload_sha256",
        "state_binding_sha256",
    }
)

# pytorchvideo 0.1.5 imports the pre-0.17 public torchvision module name.
# torchvision 0.19 retains the implementation under a private module, so
# expose the legacy alias before importing the upstream MotionLLM package.
try:
    import torchvision.transforms._functional_tensor as _functional_tensor

    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor",
        _functional_tensor,
    )
except ImportError:
    pass


MODEL_ROOT: Path
SOURCE: Path
Config: Any
GPT: Any
lora_filter: Any
mark_only_lora_as_trainable: Any
Tokenizer: Any
lazy_load: Any
lora: Any
EmptyInitOnDevice: Any
LanguageBindVideoTower: Any
build_vision_projector: Any
_CONFIGURED_ROOT: Path | None = None


def _configure(pretrained_root: Path) -> None:
    """Load the pinned MotionLLM source/runtime from the frozen asset root."""

    global MODEL_ROOT, SOURCE, Config, GPT, lora_filter
    global mark_only_lora_as_trainable, Tokenizer, lazy_load, lora
    global EmptyInitOnDevice, LanguageBindVideoTower, build_vision_projector
    global _CONFIGURED_ROOT
    root = pretrained_root.resolve(strict=True)
    if _CONFIGURED_ROOT is not None:
        if _CONFIGURED_ROOT != root:
            raise RuntimeError("MotionLLM backend cannot change pretrained roots")
        return
    MODEL_ROOT = root / "by_model" / "motionllm_official"
    SOURCE = MODEL_ROOT / "source"
    runtime_deps = root / "runtime_deps" / "motionllm"
    required = (
        SOURCE,
        runtime_deps,
        MODEL_ROOT / "official_lora.pth",
        MODEL_ROOT / "official_projector.pth",
        MODEL_ROOT / "vicuna_lit" / "lit_model.pth",
        MODEL_ROOT / "video_tower",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MotionLLM production assets/source are incomplete: " + ", ".join(missing)
        )
    sys.path[:0] = [str(runtime_deps), str(SOURCE)]
    os.chdir(SOURCE)
    from lit_gpt.lora import (
        Config as _Config,
        GPT as _GPT,
        lora_filter as _lora_filter,
        mark_only_lora_as_trainable as _mark_only_lora_as_trainable,
    )
    from lit_gpt.tokenizer import Tokenizer as _Tokenizer
    from lit_gpt.utils import lazy_load as _lazy_load
    from lit_llama.lora import lora as _lora
    from lit_llama.utils import EmptyInitOnDevice as _EmptyInitOnDevice
    from models.multimodal_encoder.languagebind import (
        LanguageBindVideoTower as _LanguageBindVideoTower,
    )
    from models.multimodal_projector.builder import (
        build_vision_projector as _build_vision_projector,
    )

    Config = _Config
    GPT = _GPT
    lora_filter = _lora_filter
    mark_only_lora_as_trainable = _mark_only_lora_as_trainable
    Tokenizer = _Tokenizer
    lazy_load = _lazy_load
    lora = _lora
    EmptyInitOnDevice = _EmptyInitOnDevice
    LanguageBindVideoTower = _LanguageBindVideoTower
    build_vision_projector = _build_vision_projector
    _CONFIGURED_ROOT = root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(
    state: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, str]]:
    """Hash an exact named tensor state without dtype-changing conversions."""

    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"{label} tensor state is missing or empty")
    aggregate = hashlib.sha256(b"motion-eval-saved-tensor-state-v1\0")
    entries: dict[str, str] = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{label} tensor state has an invalid key")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{label} tensor state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise RuntimeError(f"{label} tensor state is non-finite: {name}")
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        metadata = (
            f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8")
        )
        entry_sha256 = hashlib.sha256(metadata + raw).hexdigest()
        entries[name] = entry_sha256
        aggregate.update(len(metadata).to_bytes(8, "big"))
        aggregate.update(metadata)
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
    return aggregate.hexdigest(), entries


def require_tensor_state_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Fail unless two serialized tensor states are exactly equivalent."""

    expected_sha256, expected_entries = tensor_state_sha256(expected, label=label)
    actual_sha256, actual_entries = tensor_state_sha256(actual, label=label)
    missing = sorted(set(expected_entries) - set(actual_entries))
    unexpected = sorted(set(actual_entries) - set(expected_entries))
    mismatched = sorted(
        name
        for name in set(expected_entries) & set(actual_entries)
        if expected_entries[name] != actual_entries[name]
    )
    if missing or unexpected or mismatched or expected_sha256 != actual_sha256:
        raise RuntimeError(
            f"{label} saved tensor state does not match final in-memory state "
            f"(missing={missing}, unexpected={unexpected}, mismatched={mismatched})"
        )
    return actual_sha256


def verify_saved_tensor_state(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Reload one weight file and bind it to the state supplied to ``torch.save``."""

    if not path.is_file():
        raise FileNotFoundError(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise RuntimeError(f"{label} saved weight file is not a tensor mapping")
    return require_tensor_state_match(expected, saved, label=label)


def _cpu_tensor_state(state: Mapping[str, Any], *, label: str) -> dict[str, torch.Tensor]:
    """Materialize an immutable CPU copy and validate exact tensor metadata."""

    tensor_state_sha256(state, label=label)
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in state.items()
    }


def tensor_state_parameter_count(state: Mapping[str, Any], *, label: str) -> int:
    """Count elements only after validating the exact named tensor mapping."""

    tensor_state_sha256(state, label=label)
    count = sum(int(value.numel()) for value in state.values())
    if count <= 0:
        raise RuntimeError(f"{label} tensor state has no parameters")
    return count


def language_trainable_serialized_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Prove a one-to-one LM trainable-name to serialized-key mapping."""

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("MotionLLM language model has no trainable parameters")
    state = model.state_dict()
    selected = {
        name: value
        for name, value in state.items()
        if lora_filter(name, value)
    }
    missing = sorted(set(trainable) - set(selected))
    unexpected = sorted(set(selected) - set(trainable))
    if missing or unexpected:
        raise RuntimeError(
            "MotionLLM language trainables do not have one-to-one serialized "
            f"LoRA keys (missing={missing}, unexpected={unexpected})"
        )
    # This also rejects a state_dict hook that rewrites dtype, shape, or bytes
    # while leaving the same apparent key names.
    require_tensor_state_match(
        trainable,
        selected,
        label="MotionLLM live language trainables",
    )
    return _cpu_tensor_state(selected, label="MotionLLM live language trainables")


def projector_serialized_states(
    projector: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return exact full projector state and its strictly covered trainables."""

    trainable = {
        name: parameter
        for name, parameter in projector.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("MotionLLM projector has no trainable parameters")
    full_state = projector.state_dict()
    missing = sorted(set(trainable) - set(full_state))
    if missing:
        raise RuntimeError(
            f"MotionLLM projector trainables are absent from state_dict: {missing}"
        )
    serialized_trainable = {name: full_state[name] for name in trainable}
    require_tensor_state_match(
        trainable,
        serialized_trainable,
        label="MotionLLM live projector trainables",
    )
    return (
        _cpu_tensor_state(full_state, label="MotionLLM live projector"),
        _cpu_tensor_state(
            serialized_trainable,
            label="MotionLLM live projector trainables",
        ),
    )


def combined_trainable_state(
    language_state: Mapping[str, Any],
    projector_trainable_state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    combined = {
        **{f"language_model.{name}": value for name, value in language_state.items()},
        **{
            f"projector.{name}": value
            for name, value in projector_trainable_state.items()
        },
    }
    if len(combined) != len(language_state) + len(projector_trainable_state):
        raise RuntimeError("MotionLLM combined trainable names are not one-to-one")
    return _cpu_tensor_state(combined, label="MotionLLM combined trainables")


def validate_state_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STATE_BINDING_KEYS:
        raise RuntimeError("MotionLLM state binding schema is invalid")
    binding = dict(value)
    body = {
        key: item for key, item in binding.items() if key != "state_binding_sha256"
    }
    if (
        binding.get("schema_version") != "1.0"
        or binding.get("model_id") != "motionllm_official"
        or binding.get("state_binding_sha256") != sha256_json(body)
    ):
        raise RuntimeError("MotionLLM state binding identity/hash is invalid")
    for name in (
        "lora_r",
        "lora_alpha",
        "language_trainable_tensor_count",
        "projector_trainable_tensor_count",
        "projector_state_tensor_count",
        "combined_trainable_tensor_count",
        "combined_trainable_parameter_count",
    ):
        if type(binding.get(name)) is not int or binding[name] <= 0:
            raise RuntimeError(f"MotionLLM state binding {name} is invalid")
    dropout = binding.get("lora_dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise RuntimeError("MotionLLM state binding lora_dropout is invalid")
    for name in (
        "language_live_sha256",
        "language_disk_sha256",
        "language_reload_sha256",
        "projector_live_sha256",
        "projector_disk_sha256",
        "projector_reload_sha256",
        "combined_live_sha256",
        "combined_reload_sha256",
        "state_binding_sha256",
    ):
        digest = binding.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"MotionLLM state binding {name} is invalid")
    if not (
        binding["language_live_sha256"]
        == binding["language_disk_sha256"]
        == binding["language_reload_sha256"]
    ):
        raise RuntimeError("MotionLLM language live/disk/reload states differ")
    if not (
        binding["projector_live_sha256"]
        == binding["projector_disk_sha256"]
        == binding["projector_reload_sha256"]
    ):
        raise RuntimeError("MotionLLM projector live/disk/reload states differ")
    if binding["combined_live_sha256"] != binding["combined_reload_sha256"]:
        raise RuntimeError("MotionLLM combined live/reload trainable states differ")
    if binding["combined_trainable_tensor_count"] != (
        binding["language_trainable_tensor_count"]
        + binding["projector_trainable_tensor_count"]
    ):
        raise RuntimeError("MotionLLM state binding trainable counts disagree")
    if (
        binding["projector_state_tensor_count"]
        < binding["projector_trainable_tensor_count"]
    ):
        raise RuntimeError("MotionLLM projector state omits trainable tensors")
    return binding


def load_language_model(
    device: torch.device,
    rank: int,
    alpha: int,
    dropout: float,
    initialize_from_official_lora: bool,
    artifact_lora: Path | None = None,
) -> GPT:
    with EmptyInitOnDevice(
        device=device,
        dtype=torch.bfloat16,
        quantization_mode=None,
    ), lora(r=rank, alpha=alpha, dropout=dropout, enabled=True):
        config = Config.from_name(
            name="vicuna-7b-v1.5",
            r=rank,
            alpha=alpha,
            dropout=dropout,
            to_query=True,
            to_key=False,
            to_value=True,
            to_projection=False,
            to_mlp=False,
            to_head=False,
        )
        model = GPT(config).bfloat16()

    base = lazy_load(MODEL_ROOT / "vicuna_lit" / "lit_model.pth")
    state = dict(base)
    if artifact_lora is not None:
        state.update(lazy_load(artifact_lora))
    elif initialize_from_official_lora:
        state.update(lazy_load(MODEL_ROOT / "official_lora.pth"))
    else:
        # A strict load still needs the freshly initialized LoRA tensors.
        state.update({k: v for k, v in model.state_dict().items() if lora_filter(k, v)})
    model.load_state_dict(state, strict=True)
    mark_only_lora_as_trainable(model)
    return model


def load_video_path(
    device: torch.device,
    *,
    projector_checkpoint: Path | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, object]:
    args = SimpleNamespace(
        video_tower="LanguageBind/LanguageBind_Video_merge",
        mm_video_tower="LanguageBind/LanguageBind_Video_merge",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1024,
        hidden_size=4096,
    )
    # The upstream builder hard-codes ``./cache_dir``. Pass the canonical
    # snapshot itself so loading never depends on Hugging Face cache refs or
    # network access.
    video_snapshot = (MODEL_ROOT / "video_tower").resolve()
    tower = LanguageBindVideoTower(
        str(video_snapshot),
        args,
        delay_load=True,
        cache_dir=str(video_snapshot.parents[2]),
    )
    if not tower.is_loaded:
        tower.load_model()
    tower.requires_grad_(False)
    tower.to(device=device, dtype=torch.float16)

    projector = build_vision_projector(args)
    projector.load_state_dict(
        torch.load(
            projector_checkpoint or (MODEL_ROOT / "official_projector.pth"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    projector.to(device=device, dtype=torch.float32)
    return tower, projector, tower.video_processor


def pool_video_tokens(features: torch.Tensor, output_tokens: int) -> torch.Tensor:
    if output_tokens < 1:
        raise ValueError("--max-video-tokens must be positive")
    if features.shape[1] <= output_tokens:
        return features
    # Preserve coverage of the whole video rather than keeping only its prefix.
    pooled = F.adaptive_avg_pool1d(
        features.transpose(1, 2),
        output_tokens,
    )
    return pooled.transpose(1, 2)


def trainable_snapshot(
    model: torch.nn.Module, projector: torch.nn.Module
) -> tuple[torch.Tensor, str, dict[str, str]]:
    language_state = language_trainable_serialized_state(model)
    _projector_state, projector_trainables = projector_serialized_states(projector)
    combined = combined_trainable_state(language_state, projector_trainables)
    digest, entries = tensor_state_sha256(
        combined,
        label="MotionLLM combined trainables",
    )
    vector = torch.cat(
        [value.detach().float().reshape(-1) for _, value in sorted(combined.items())]
    )
    return vector, digest, entries


def conversation_pair(row: dict) -> tuple[str, str]:
    if "question" in row and ("answer" in row or "gold" in row):
        question = str(row["question"]).strip()
        options = row.get("options")
        if isinstance(options, dict) and set(options) == set("ABCD"):
            question += "\n" + "\n".join(
                f"{key}. {options[key]}" for key in "ABCD"
            )
        answer = str(row.get("answer", row.get("gold", ""))).strip()
        if answer in "ABCD" and len(answer) == 1:
            answer = f"<answer>{answer}</answer>"
        if not question or not answer:
            raise ValueError("MotionLLM row lacks question/answer supervision")
        return question, answer
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Each row needs question/answer or conversations")
    question = None
    answer = None
    for message in conversations:
        role = str(message.get("from", message.get("role", ""))).lower()
        value = str(message.get("value", message.get("content", ""))).strip()
        if question is None and role in {"human", "user"}:
            question = value.replace("<video>", "").strip()
        elif question is not None and role in {"gpt", "assistant"}:
            answer = value
            break
    if not question or not answer:
        raise ValueError("Could not find a user/assistant pair in conversations")
    return question, answer


def load_training_samples(args: argparse.Namespace) -> tuple[list[dict], Path | None]:
    if args.data_path:
        data_path = Path(args.data_path).resolve()
        text = data_path.read_text(encoding="utf-8")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [json.loads(line) for line in text.splitlines() if line.strip()]
        rows = value if isinstance(value, list) else [value]
        video_root = (
            Path(args.video_root).resolve()
            if args.video_root
            else data_path.parent
        )
        samples = []
        for row in rows:
            question, answer = conversation_pair(row)
            raw_video = Path(str(row["video"]))
            video = raw_video if raw_video.is_absolute() else video_root / raw_video
            samples.append(
                {
                    "video": video.resolve(),
                    "question": question,
                    "answer": answer,
                }
            )
            if args.limit is not None and len(samples) >= args.limit:
                break
    else:
        data_path = None
        if not (args.video and args.question and args.answer):
            raise ValueError(
                "Provide --data-path, or all of --video/--question/--answer"
            )
        samples = [
            {
                "video": Path(args.video).resolve(),
                "question": args.question,
                "answer": args.answer,
            }
        ]
    if not samples:
        raise ValueError("No MotionLLM training samples were selected")
    for sample in samples:
        if not sample["video"].is_file():
            raise FileNotFoundError(sample["video"])
    return samples, data_path


def build_training_example(
    sample: dict,
    device: torch.device,
    tower: torch.nn.Module,
    projector: torch.nn.Module,
    processor: object,
    tokenizer: Tokenizer,
    max_video_tokens: int,
) -> tuple[tuple, torch.Tensor, int]:
    pixel_values = processor(
        str(sample["video"]),
        return_tensors="pt",
    )["pixel_values"]
    if isinstance(pixel_values, list):
        pixel_values = [
            value.to(device=device, dtype=torch.float16)
            for value in pixel_values
        ]
    else:
        pixel_values = pixel_values.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        frozen_video_features = tower(pixel_values)
    if isinstance(frozen_video_features, list):
        frozen_video_features = torch.stack(frozen_video_features)
    video_features = projector(frozen_video_features.float())
    video_features = pool_video_tokens(video_features, max_video_tokens)
    token_count = video_features.shape[1]

    prefix_text = (
        "A chat between a curious user and an artificial intelligence assistant. "
        f"USER: {sample['question']} INPUT_VIDEO: "
    )
    suffix_text = ".\nASSISTANT: "
    prefix = tokenizer.encode(prefix_text, bos=True, eos=False, device=device).long()
    suffix = tokenizer.encode(suffix_text, bos=False, eos=False, device=device).long()
    answer = tokenizer.encode(
        sample["answer"],
        bos=False,
        eos=True,
        device=device,
    ).long()
    full_ids = torch.cat(
        [
            prefix,
            torch.zeros(token_count, dtype=torch.long, device=device),
            suffix,
            answer,
        ]
    ).unsqueeze(0)
    labels = full_ids.clone()
    answer_start = prefix.numel() + token_count + suffix.numel()
    labels[:, :answer_start] = -1
    multimodal_input = (
        full_ids,
        video_features.to(dtype=torch.bfloat16),
        [prefix.numel()],
    )
    return multimodal_input, labels, token_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--video-root")
    parser.add_argument("--video")
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-root", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-video-tokens", type=int, default=32)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-projector", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--initialize-from-official-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    for name in ("learning_rate_lora", "learning_rate_projector"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive integers")
    if not math.isfinite(args.lora_dropout) or not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be finite and in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("MotionLLM finetuning requires CUDA")
    _configure(Path(args.pretrained_root))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, data_path = load_training_samples(args)

    started_at = time.time()
    model = load_language_model(
        device,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.initialize_from_official_lora,
    )
    tower, projector, processor = load_video_path(device)
    tokenizer = Tokenizer(MODEL_ROOT / "vicuna_lit")

    before, initial_trainable_sha256, before_entries = trainable_snapshot(
        model, projector
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.learning_rate_lora,
            },
            {
                "params": [p for p in projector.parameters() if p.requires_grad],
                "lr": args.learning_rate_projector,
            },
        ]
    )

    model.train()
    projector.train()
    losses: list[float] = []
    video_token_counts: list[int] = []
    max_grad = 0.0
    nonzero_finite_gradient_steps = 0
    target_steps = args.max_steps or (args.epochs * len(samples))
    order: list[int] = []
    for step in range(target_steps):
        if step % len(samples) == 0:
            order = list(range(len(samples)))
            if args.shuffle:
                random.Random(args.seed + step // len(samples)).shuffle(order)
        sample = samples[order[step % len(samples)]]
        optimizer.zero_grad(set_to_none=True)
        multimodal_input, labels, token_count = build_training_example(
            sample,
            device,
            tower,
            projector,
            processor,
            tokenizer,
            args.max_video_tokens,
        )
        full_ids = multimodal_input[0]
        logits = model(multimodal_input, maxlen=full_ids.shape[1])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-1,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        step_max_grad = 0.0
        for parameter in list(model.parameters()) + list(projector.parameters()):
            if parameter.requires_grad and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite MotionLLM gradient")
                step_max_grad = max(
                    step_max_grad,
                    float(parameter.grad.detach().abs().max().cpu()),
                )
        if not math.isfinite(step_max_grad) or step_max_grad <= 0.0:
            raise RuntimeError("All MotionLLM gradients are zero")
        nonzero_finite_gradient_steps += 1
        max_grad = max(max_grad, step_max_grad)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        video_token_counts.append(token_count)

    after, final_trainable_sha256, after_entries = trainable_snapshot(model, projector)
    if set(before_entries) != set(after_entries):
        raise RuntimeError("MotionLLM trainable parameter names changed during training")
    changed_trainable_tensor_count = sum(
        before_entries[name] != after_entries[name] for name in before_entries
    )
    max_update = float((after - before).abs().max())
    if (
        not math.isfinite(max_update)
        or max_update <= 0.0
        or changed_trainable_tensor_count <= 0
        or initial_trainable_sha256 == final_trainable_sha256
    ):
        raise RuntimeError("Optimizer step did not change any trainable parameter")

    # Derive the artifact from the exact final live parameters, not merely
    # another lora_filter pass whose coverage could silently diverge.
    lora_state = language_trainable_serialized_state(model)
    projector_state, projector_trainable_state = projector_serialized_states(
        projector
    )
    combined_live_state = combined_trainable_state(
        lora_state,
        projector_trainable_state,
    )
    combined_live_sha256, combined_live_entries = tensor_state_sha256(
        combined_live_state,
        label="MotionLLM combined trainables",
    )
    if (
        combined_live_sha256 != final_trainable_sha256
        or combined_live_entries != after_entries
    ):
        raise RuntimeError(
            "MotionLLM final trainable receipt differs from live serializable state"
        )
    language_live_sha256, _ = tensor_state_sha256(
        lora_state, label="MotionLLM LoRA"
    )
    projector_live_sha256, _ = tensor_state_sha256(
        projector_state, label="MotionLLM projector"
    )
    lora_path = output_dir / "lora.pth"
    projector_path = output_dir / "linear.pth"
    torch.save(lora_state, lora_path)
    torch.save(projector_state, projector_path)
    language_disk_sha256 = verify_saved_tensor_state(
        lora_path,
        lora_state,
        label="MotionLLM LoRA",
    )
    projector_disk_sha256 = verify_saved_tensor_state(
        projector_path,
        projector_state,
        label="MotionLLM projector",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in projector.parameters() if p.requires_grad)
    serialized_trainable = tensor_state_parameter_count(
        combined_live_state,
        label="MotionLLM combined trainables",
    )
    if trainable != serialized_trainable:
        raise RuntimeError(
            "MotionLLM live trainable parameter count differs from serialized state"
        )
    # Free the training graph/model before constructing a genuinely fresh
    # official reload. CPU copies above retain the exact final live bytes.
    del optimizer, model, tower, projector, parameter, processor, tokenizer
    del loss, logits, multimodal_input, labels, full_ids
    torch.cuda.empty_cache()
    reloaded_model = load_language_model(
        device,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        initialize_from_official_lora=False,
        artifact_lora=lora_path,
    )
    reloaded_tower, reloaded_projector, _reloaded_processor = load_video_path(
        device,
        projector_checkpoint=projector_path,
    )
    reloaded_language_state = language_trainable_serialized_state(reloaded_model)
    reloaded_projector_state, reloaded_projector_trainables = (
        projector_serialized_states(reloaded_projector)
    )
    language_reload_sha256 = require_tensor_state_match(
        lora_state,
        reloaded_language_state,
        label="MotionLLM reloaded LoRA",
    )
    projector_reload_sha256 = require_tensor_state_match(
        projector_state,
        reloaded_projector_state,
        label="MotionLLM reloaded projector",
    )
    combined_reload_state = combined_trainable_state(
        reloaded_language_state,
        reloaded_projector_trainables,
    )
    combined_reload_sha256 = require_tensor_state_match(
        combined_live_state,
        combined_reload_state,
        label="MotionLLM reloaded combined trainables",
    )
    del reloaded_model, reloaded_tower, reloaded_projector
    torch.cuda.empty_cache()

    state_binding_body = {
        "schema_version": "1.0",
        "model_id": "motionllm_official",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "language_trainable_tensor_count": len(lora_state),
        "projector_trainable_tensor_count": len(projector_trainable_state),
        "projector_state_tensor_count": len(projector_state),
        "combined_trainable_tensor_count": len(combined_live_state),
        "combined_trainable_parameter_count": trainable,
        "language_live_sha256": language_live_sha256,
        "language_disk_sha256": language_disk_sha256,
        "language_reload_sha256": language_reload_sha256,
        "projector_live_sha256": projector_live_sha256,
        "projector_disk_sha256": projector_disk_sha256,
        "projector_reload_sha256": projector_reload_sha256,
        "combined_live_sha256": combined_live_sha256,
        "combined_reload_sha256": combined_reload_sha256,
    }
    state_binding = validate_state_binding(
        {
            **state_binding_body,
            "state_binding_sha256": sha256_json(state_binding_body),
        }
    )
    state_binding_path = atomic_write_json(
        output_dir / "state_binding.json",
        state_binding,
        root=output_dir,
        overwrite=False,
    )
    result = {
        "status": "passed",
        "model_id": "motionllm_official",
        "kind": "lora_plus_projector",
        "base_path": str(MODEL_ROOT),
        "data_path": str(data_path) if data_path else None,
        "data_sha256": sha256(data_path) if data_path else None,
        "sample_count": len(samples),
        "steps": target_steps,
        "losses": losses,
        "video_token_counts": video_token_counts,
        "trainable_parameters": trainable,
        "max_gradient": max_grad,
        "nonzero_finite_gradient_steps": nonzero_finite_gradient_steps,
        "max_parameter_update": max_update,
        "trainable_tensor_count": len(before_entries),
        "changed_trainable_tensor_count": changed_trainable_tensor_count,
        "initial_trainable_sha256": initial_trainable_sha256,
        "final_trainable_sha256": final_trainable_sha256,
        "saved_state_matches_final_memory": True,
        "lora_state_sha256": language_disk_sha256,
        "projector_state_sha256": projector_disk_sha256,
        "state_binding_path": str(state_binding_path),
        "state_binding_sha256": state_binding["state_binding_sha256"],
        "lora_path": str(lora_path),
        "projector_path": str(projector_path),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started_at,
        "initialized_from_official_lora": args.initialize_from_official_lora,
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


def run_finetune(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    pretrained_root: Path,
    output_dir: Path,
    work_dir: Path,
    training_steps: int,
    limit: int | None,
    seed: int,
    training_receipt_path: Path,
    evidence_bindings: Mapping[str, Any],
) -> int:
    del work_dir
    if not validation_manifest.is_file():
        raise FileNotFoundError(validation_manifest)
    argv = [
        "--data-path", str(train_manifest),
        "--pretrained-root", str(pretrained_root),
        "--output-dir", str(output_dir),
        "--max-steps", str(training_steps),
        "--seed", str(seed),
        "--no-shuffle",
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    result = main(argv)
    if result != 0:
        return result
    smoke = load_json_strict(output_dir / "smoke_result.json")
    if not isinstance(smoke, Mapping) or smoke.get("status") != "passed":
        raise RuntimeError("MotionLLM backend did not publish internal training proof")
    state_binding_path = (output_dir / "state_binding.json").resolve(strict=True)
    state_binding = validate_state_binding(load_json_strict(state_binding_path))
    if (
        smoke.get("saved_state_matches_final_memory") is not True
        or smoke.get("state_binding_path") != str(state_binding_path)
        or smoke.get("state_binding_sha256")
        != state_binding["state_binding_sha256"]
        or smoke.get("lora_state_sha256")
        != state_binding["language_disk_sha256"]
        or smoke.get("projector_state_sha256")
        != state_binding["projector_disk_sha256"]
        or smoke.get("final_trainable_sha256")
        != state_binding["combined_live_sha256"]
        or smoke.get("trainable_tensor_count")
        != state_binding["combined_trainable_tensor_count"]
        or smoke.get("trainable_parameters")
        != state_binding["combined_trainable_parameter_count"]
    ):
        raise RuntimeError(
            "MotionLLM smoke/training evidence is not bound to exact saved state"
        )
    artifact_sha256 = hash_path(
        output_dir, symlink_policy="reject", allowed_root=output_dir.parent
    ).digest
    receipt = make_training_receipt(
        **dict(evidence_bindings),
        planned_global_steps=training_steps,
        actual_global_steps=smoke.get("steps"),
        planned_optimizer_steps=training_steps,
        actual_optimizer_steps=smoke.get("steps"),
        finite_losses=smoke.get("losses", []),
        nonzero_finite_gradient_steps=smoke.get(
            "nonzero_finite_gradient_steps"
        ),
        max_gradient=smoke.get("max_gradient"),
        trainable_tensor_count=smoke.get("trainable_tensor_count"),
        trainable_parameter_count=smoke.get("trainable_parameters"),
        changed_trainable_tensor_count=smoke.get(
            "changed_trainable_tensor_count"
        ),
        initial_trainable_sha256=smoke.get("initial_trainable_sha256"),
        final_trainable_sha256=smoke.get("final_trainable_sha256"),
        max_parameter_update=smoke.get("max_parameter_update"),
        artifact_sha256=artifact_sha256,
    )
    write_training_receipt(
        training_receipt_path,
        receipt,
        root=output_dir.parent,
        overwrite=False,
    )
    return 0


def verify_reload(*, artifact: Path, pretrained_root: Path) -> bool:
    """Strictly rebuild base/LoRA/projector modules from the fresh files."""

    _configure(pretrained_root)
    lora_path = artifact / "lora.pth"
    projector_path = artifact / "linear.pth"
    if not lora_path.is_file() or not projector_path.is_file():
        raise FileNotFoundError("MotionLLM artifact lacks lora.pth or linear.pth")
    evidence = load_json_strict(artifact / "smoke_result.json")
    state_binding_path = (artifact / "state_binding.json").resolve(strict=True)
    state_binding = validate_state_binding(load_json_strict(state_binding_path))
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("saved_state_matches_final_memory") is not True
        or evidence.get("state_binding_path") != str(state_binding_path)
        or evidence.get("state_binding_sha256")
        != state_binding["state_binding_sha256"]
        or evidence.get("final_trainable_sha256")
        != state_binding["combined_live_sha256"]
        or evidence.get("trainable_tensor_count")
        != state_binding["combined_trainable_tensor_count"]
        or evidence.get("trainable_parameters")
        != state_binding["combined_trainable_parameter_count"]
    ):
        raise RuntimeError("MotionLLM artifact lacks disk-state binding evidence")
    saved_language = torch.load(lora_path, map_location="cpu", weights_only=True)
    saved_projector = torch.load(
        projector_path, map_location="cpu", weights_only=True
    )
    if not isinstance(saved_language, Mapping) or not isinstance(
        saved_projector, Mapping
    ):
        raise RuntimeError("MotionLLM saved weights are not tensor mappings")
    language_disk_sha256, _ = tensor_state_sha256(
        saved_language, label="MotionLLM LoRA"
    )
    projector_disk_sha256, _ = tensor_state_sha256(
        saved_projector, label="MotionLLM projector"
    )
    if (
        language_disk_sha256 != state_binding["language_disk_sha256"]
        or projector_disk_sha256 != state_binding["projector_disk_sha256"]
        or evidence.get("lora_state_sha256") != language_disk_sha256
        or evidence.get("projector_state_sha256") != projector_disk_sha256
    ):
        raise RuntimeError("MotionLLM saved state no longer matches training evidence")
    device = torch.device("cuda:0")
    model = load_language_model(
        device,
        rank=state_binding["lora_r"],
        alpha=state_binding["lora_alpha"],
        dropout=float(state_binding["lora_dropout"]),
        initialize_from_official_lora=False,
        artifact_lora=lora_path,
    )
    tower, projector, _processor = load_video_path(
        device,
        projector_checkpoint=projector_path,
    )
    reloaded_language = language_trainable_serialized_state(model)
    reloaded_projector, reloaded_projector_trainables = projector_serialized_states(
        projector
    )
    language_reload_sha256 = require_tensor_state_match(
        saved_language,
        reloaded_language,
        label="MotionLLM verifier reloaded LoRA",
    )
    projector_reload_sha256 = require_tensor_state_match(
        saved_projector,
        reloaded_projector,
        label="MotionLLM verifier reloaded projector",
    )
    reloaded_combined = combined_trainable_state(
        reloaded_language,
        reloaded_projector_trainables,
    )
    combined_reload_sha256, _ = tensor_state_sha256(
        reloaded_combined,
        label="MotionLLM verifier combined trainables",
    )
    if (
        language_reload_sha256 != state_binding["language_reload_sha256"]
        or projector_reload_sha256 != state_binding["projector_reload_sha256"]
        or combined_reload_sha256 != state_binding["combined_reload_sha256"]
        or len(reloaded_language)
        != state_binding["language_trainable_tensor_count"]
        or len(reloaded_projector_trainables)
        != state_binding["projector_trainable_tensor_count"]
        or len(reloaded_projector) != state_binding["projector_state_tensor_count"]
        or len(reloaded_combined)
        != state_binding["combined_trainable_tensor_count"]
        or tensor_state_parameter_count(
            reloaded_combined,
            label="MotionLLM verifier combined trainables",
        )
        != state_binding["combined_trainable_parameter_count"]
    ):
        raise RuntimeError("MotionLLM strict reload differs from final live state")
    del model, tower, projector
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    raise SystemExit(main())
