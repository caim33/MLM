#!/usr/bin/env python3
"""Batch-ready VideoLLaMA LoRA finetuner with a one-step smoke mode.

The base VideoLLaMA checkpoint is immutable. Only LoRA weights injected into
the language model are optimized and saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.utils.save_and_load import load_peft_weights
from torch.utils.data import DataLoader

from motion_eval.core import hash_path
from motion_eval.data import load_json_strict
from motion_eval.training_receipt import make_training_receipt, write_training_receipt


BACKEND_ID = "videollama_lora:peft_v1"
MODEL_FAMILY = "videollama_llama2_video"
TRAINING_MODE = "lora_sft"


PRETRAIN: Path
SOURCE: Path
BASE_CONFIG: Path
Config: Any
registry: Any
Video_Instruct_Dataset: Any
AlproVideoTrainProcessor: Any
_CONFIGURED_ROOT: Path | None = None


def _configure(pretrained_root: Path) -> None:
    """Bind the historical verified code to this batch's frozen asset root."""

    global PRETRAIN, SOURCE, BASE_CONFIG, Config, registry
    global Video_Instruct_Dataset, AlproVideoTrainProcessor, _CONFIGURED_ROOT
    root = pretrained_root.resolve(strict=True)
    if _CONFIGURED_ROOT is not None:
        if _CONFIGURED_ROOT != root:
            raise RuntimeError("VideoLLaMA backend cannot change pretrained roots")
        return
    PRETRAIN = root / "by_model" / "videollama_lora" / "base"
    torch_home = root / "downloads" / "VideoLLaMA-runtime" / "torch"
    # The audited remote layout keeps the pinned upstream checkout beside the
    # unified controller and the evaluation YAML under the project root.
    SOURCE = root.parents[2] / "video_model_sources" / "video-llama"
    BASE_CONFIG = (
        root.parents[3]
        / "MVBench_Eval"
        / "scripts"
        / "video_llama_motionx_eval_only_vl.yaml"
    )
    required = (
        PRETRAIN / "llama-2-7b-chat-hf",
        PRETRAIN / "VL_LLaMA_2_7B_Finetuned.pth",
        torch_home / "hub" / "checkpoints" / "eva_vit_g.pth",
        torch_home / "hub" / "checkpoints" / "blip2_pretrained_flant5xxl.pth",
        SOURCE / "video_llama",
        BASE_CONFIG,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "VideoLLaMA production assets/source are incomplete: " + ", ".join(missing)
        )
    os.environ["TORCH_HOME"] = str(torch_home)
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)
    from video_llama.common.config import Config as _Config
    from video_llama.common.registry import registry as _registry
    from video_llama.datasets.datasets.video_instruct_dataset import (
        Video_Instruct_Dataset as _Video_Instruct_Dataset,
    )
    # Imports are registration side effects required by the official factory.
    import video_llama.models  # noqa: F401
    import video_llama.processors  # noqa: F401
    from video_llama.processors import (
        AlproVideoTrainProcessor as _AlproVideoTrainProcessor,
    )

    Config = _Config
    registry = _registry
    Video_Instruct_Dataset = _Video_Instruct_Dataset
    AlproVideoTrainProcessor = _AlproVideoTrainProcessor
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
    """Fail unless PEFT saved exactly the final adapter/module tensor state."""

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


def verify_saved_peft_state(
    adapter_dir: Path,
    expected: Mapping[str, Any],
) -> str:
    """Reload the PEFT file and bind LoRA/modules_to_save to live memory."""

    saved = load_peft_weights(str(adapter_dir), device="cpu")
    if not isinstance(saved, Mapping):
        raise RuntimeError("VideoLLaMA PEFT weight file is not a tensor mapping")
    return require_tensor_state_match(
        expected,
        saved,
        label="VideoLLaMA PEFT adapter",
    )


def clean_question(text: str) -> str:
    return text.replace("<video>", "").strip()


def convert_sft(data_path: Path, output_path: Path, limit: int | None) -> Path:
    text = data_path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = value if isinstance(value, list) else [value]
    converted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("VideoLLaMA training rows must be JSON objects")
        conversations = row.get("conversations")
        if isinstance(conversations, list) and len(conversations) >= 2:
            question = clean_question(str(conversations[0].get("value", "")))
            answer = str(conversations[1].get("value", "")).strip()
        else:
            question = str(row.get("question", "")).strip()
            options = row.get("options")
            if isinstance(options, dict) and set(options) == set("ABCD"):
                question += "\n" + "\n".join(
                    f"{key}. {options[key]}" for key in "ABCD"
                )
            target = row.get("answer", row.get("gold"))
            answer = str(target).strip() if target is not None else ""
            if answer in "ABCD" and len(answer) == 1:
                answer = f"<answer>{answer}</answer>"
        if not question or not answer:
            raise ValueError("VideoLLaMA training row lacks question/answer supervision")
        video = row.get("video")
        if not isinstance(video, str) or not video.strip():
            raise ValueError("VideoLLaMA training row lacks a video path")
        converted.append(
            {
                "video": video,
                "QA": [
                    {
                        "q": question,
                        "a": answer,
                    }
                ],
            }
        )
        if limit is not None and len(converted) >= limit:
            break
    if not converted:
        raise ValueError("No training rows were selected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_runtime_config(path: Path, max_txt_len: int) -> Path:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    model = raw["model"]
    model["llama_model"] = str(PRETRAIN / "llama-2-7b-chat-hf")
    model["ckpt"] = str(PRETRAIN / "VL_LLaMA_2_7B_Finetuned.pth")
    model["imagebind_ckpt_path"] = str(PRETRAIN)
    model["max_txt_len"] = max_txt_len
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def load_model(
    device: torch.device,
    config_path: Path,
    rank: int,
    alpha: int,
    dropout: float,
    adapter_path: Path | None = None,
) -> tuple[torch.nn.Module, Config]:
    config_args = SimpleNamespace(cfg_path=str(config_path), gpu_id=0, options=None)
    cfg = Config(config_args)
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    for parameter in model.parameters():
        parameter.requires_grad = False
    if adapter_path is None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
    else:
        model.llama_model = PeftModel.from_pretrained(
            model.llama_model,
            str(adapter_path),
            is_trainable=False,
        )
    # VideoLLaMA's multimodal splice reaches embeddings through
    # ``llama_model.model.embed_tokens``. PEFT adds one wrapper level, so keep
    # that upstream access path valid without modifying the pinned source tree.
    model.llama_model.model.embed_tokens = model.llama_model.get_input_embeddings()
    model.llama_model.config.use_cache = False
    model = model.to(device)
    return model, cfg


def make_dataset(annotation: Path, cfg: Config, num_frames: int) -> Video_Instruct_Dataset:
    dataset = Video_Instruct_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/",
        ann_root=str(annotation),
        num_video_query_token=int(getattr(cfg.model_cfg, "num_video_query_token", 32)),
        tokenizer_name=str(cfg.model_cfg.llama_model),
        data_type="video",
        model_type="llama_v2",
    )
    dataset.num_frm = num_frames
    dataset.transform = AlproVideoTrainProcessor(
        image_size=dataset.resize_size,
        n_frms=num_frames,
    ).transform
    return dataset


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def trainable_fingerprint(
    model: torch.nn.Module,
) -> tuple[torch.Tensor, str, dict[str, str]]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named:
        raise RuntimeError("No PEFT parameters are trainable")
    unexpected = sorted(
        name
        for name, _parameter in named
        if "lora_" not in name and "modules_to_save." not in name
    )
    if unexpected:
        raise RuntimeError(
            "VideoLLaMA has trainable parameters that PEFT may not serialize: "
            f"{unexpected}"
        )
    if len({name for name, _ in named}) != len(named):
        raise RuntimeError("VideoLLaMA trainable parameter names are not unique")
    aggregate = hashlib.sha256(b"videollama-peft-trainable-state-v2\0")
    chunks: list[torch.Tensor] = []
    entry_hashes: dict[str, str] = {}
    for name, parameter in sorted(named):
        value = parameter.detach().float().cpu().contiguous()
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Non-finite VideoLLaMA PEFT state: {name}")
        raw = value.numpy().tobytes(order="C")
        metadata = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8")
        entry_hashes[name] = hashlib.sha256(metadata + raw).hexdigest()
        aggregate.update(len(metadata).to_bytes(8, "big"))
        aggregate.update(metadata)
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
        chunks.append(value.reshape(-1))
    return torch.cat(chunks), aggregate.hexdigest(), entry_hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--pretrained-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-txt-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0 or (
            name == "learning_rate" and value == 0.0
        ):
            raise ValueError(f"--{name.replace('_', '-')} is outside its valid range")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive integers")
    if not math.isfinite(args.lora_dropout) or not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be finite and in [0, 1)")
    if args.num_frames < 1 or args.max_txt_len < 1:
        raise ValueError("frame and text limits must be positive integers")
    if not torch.cuda.is_available():
        raise RuntimeError("VideoLLaMA LoRA finetuning requires CUDA")
    _configure(Path(args.pretrained_root))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_runtime_config(work_dir / "videollama_lora.yaml", args.max_txt_len)
    annotation = convert_sft(
        Path(args.data_path).resolve(),
        work_dir / "videollama_sft.json",
        args.limit,
    )

    device = torch.device("cuda:0")
    started_at = time.time()
    model, cfg = load_model(
        device,
        config_path,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    dataset = make_dataset(annotation, cfg, args.num_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collater,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    before, initial_trainable_sha256, before_entries = trainable_fingerprint(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model.train()
    iterator = iter(dataloader)
    losses: list[float] = []
    max_grad = 0.0
    nonzero_finite_gradient_steps = 0
    for _ in range(args.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        step_max_grad = 0.0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite LoRA gradient")
                step_max_grad = max(
                    step_max_grad,
                    float(parameter.grad.detach().abs().max().cpu()),
                )
        if not math.isfinite(step_max_grad) or step_max_grad <= 0.0:
            raise RuntimeError("All LoRA gradients are zero")
        nonzero_finite_gradient_steps += 1
        max_grad = max(max_grad, step_max_grad)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    after, final_trainable_sha256, after_entries = trainable_fingerprint(model)
    if set(before_entries) != set(after_entries):
        raise RuntimeError("VideoLLaMA trainable parameter names changed during training")
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
        raise RuntimeError("Optimizer step did not change any PEFT parameter")

    adapter_dir = output_dir / "adapter"
    final_peft_state = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(model.llama_model).items()
    }
    model.llama_model.save_pretrained(adapter_dir)
    adapter_state_sha256 = verify_saved_peft_state(adapter_dir, final_peft_state)
    result = {
        "status": "passed",
        "model_id": "videollama_lora",
        "kind": "true_lora",
        "base_path": str(PRETRAIN),
        "data_path": str(Path(args.data_path).resolve()),
        "data_sha256": sha256(Path(args.data_path).resolve()),
        "steps": args.max_steps,
        "losses": losses,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "max_gradient": max_grad,
        "nonzero_finite_gradient_steps": nonzero_finite_gradient_steps,
        "max_parameter_update": max_update,
        "trainable_tensor_count": len(before_entries),
        "changed_trainable_tensor_count": changed_trainable_tensor_count,
        "initial_trainable_sha256": initial_trainable_sha256,
        "final_trainable_sha256": final_trainable_sha256,
        "saved_state_matches_final_memory": True,
        "adapter_state_sha256": adapter_state_sha256,
        "adapter_path": str(adapter_dir),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started_at,
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
    if not validation_manifest.is_file():
        raise FileNotFoundError(validation_manifest)
    argv = [
        "--data-path", str(train_manifest),
        "--pretrained-root", str(pretrained_root),
        "--output-dir", str(output_dir),
        "--work-dir", str(work_dir),
        "--max-steps", str(training_steps),
        "--seed", str(seed),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    result = main(argv)
    if result != 0:
        return result
    smoke = load_json_strict(output_dir / "smoke_result.json")
    if not isinstance(smoke, Mapping) or smoke.get("status") != "passed":
        raise RuntimeError("VideoLLaMA backend did not publish internal training proof")
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
    """Construct a fresh official base model and load the emitted PEFT adapter."""

    _configure(pretrained_root)
    adapter = artifact / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError("VideoLLaMA artifact lacks adapter_config.json")
    evidence = load_json_strict(artifact / "smoke_result.json")
    if not isinstance(evidence, Mapping) or evidence.get(
        "saved_state_matches_final_memory"
    ) is not True:
        raise RuntimeError("VideoLLaMA artifact lacks disk-state binding evidence")
    saved_state = load_peft_weights(str(adapter), device="cpu")
    if not isinstance(saved_state, Mapping):
        raise RuntimeError("VideoLLaMA PEFT weight file is not a tensor mapping")
    saved_sha256, _entries = tensor_state_sha256(
        saved_state,
        label="VideoLLaMA PEFT adapter",
    )
    if evidence.get("adapter_state_sha256") != saved_sha256:
        raise RuntimeError("VideoLLaMA adapter no longer matches training evidence")
    with tempfile.TemporaryDirectory(prefix="videollama-reload-") as temporary:
        config_path = write_runtime_config(Path(temporary) / "reload.yaml", 256)
        model, _cfg = load_model(
            torch.device("cuda:0"),
            config_path,
            rank=16,
            alpha=32,
            dropout=0.0,
            adapter_path=adapter,
        )
    tensors = [
        parameter.detach()
        for name, parameter in model.llama_model.named_parameters()
        if "lora_" in name
    ]
    if not tensors or any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("VideoLLaMA PEFT reload produced no finite LoRA tensors")
    del model
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    raise SystemExit(main())
