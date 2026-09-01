#!/usr/bin/env python3
"""Evaluate Motion-R1/Qwen3-VL-Motion PEFT adapters on MotionX MCQ by log-prob scoring."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor
from safetensors.torch import load_file

QWEN_ROOT = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune")
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional at runtime
    PeftModel = None  # type: ignore

CHOICES = ["A", "B", "C", "D"]


def strip_peft_prefix(key: str) -> str:
    prefix = "base_model.model."
    return key[len(prefix) :] if key.startswith(prefix) else key


def adapter_target_candidates(key: str) -> List[str]:
    stripped = strip_peft_prefix(key)
    candidates = [stripped]
    prefixes = (
        "base_model.model.model.",
        "base_model.model.",
        "model.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            candidates.append(key[len(prefix) :])
        if stripped.startswith(prefix):
            candidates.append(stripped[len(prefix) :])
    if stripped.startswith("model."):
        candidates.append(stripped[len("model.") :])

    seen = set()
    uniq: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    return uniq


def load_adapter_manually(model: Any, adapter_dir: str) -> Any:
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.safetensors"
    if not config_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"missing adapter files under {adapter_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = load_file(str(weight_path), device="cpu")
    base_state = model.state_dict()
    module_updates: Dict[str, torch.Tensor] = {}
    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key.endswith(".lora_A.weight"):
            lora_a[key[: -len(".lora_A.weight")]] = tensor
        elif key.endswith(".lora_B.weight"):
            lora_b[key[: -len(".lora_B.weight")]] = tensor
        elif ".lora_" not in key:
            for target_key in adapter_target_candidates(key):
                if target_key in base_state:
                    module_updates[target_key] = tensor.to(dtype=base_state[target_key].dtype)
                    break

    if module_updates:
        model.load_state_dict(module_updates, strict=False)

    scale = float(config.get("lora_alpha", 1)) / float(config.get("r", 1))
    missing_pairs = sorted(set(lora_a) ^ set(lora_b))
    if missing_pairs:
        raise RuntimeError(f"LoRA A/B key mismatch, examples={missing_pairs[:5]}")

    with torch.no_grad():
        for prefix in sorted(lora_a):
            module_name = strip_peft_prefix(prefix)
            module = model.get_submodule(module_name)
            if not hasattr(module, "weight"):
                raise RuntimeError(f"LoRA target has no weight: {module_name}")
            weight = module.weight
            delta = (lora_b[prefix].to(device=weight.device, dtype=weight.dtype) @ lora_a[prefix].to(device=weight.device, dtype=weight.dtype)) * scale
            if delta.shape != weight.shape:
                raise RuntimeError(f"LoRA delta shape mismatch for {module_name}: {tuple(delta.shape)} vs {tuple(weight.shape)}")
            weight.add_(delta)

    print(
        f"[manual_adapter] loaded {len(lora_a)} LoRA layers and {len(module_updates)} saved module tensors from {adapter_path}",
        file=sys.stderr,
    )
    return model


def load_jsonl(path: Path, limit: Optional[int], branch: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if branch != "all" and str(obj.get("branch", "")).lower() != branch:
                continue
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_option_letter(text: Any) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text, flags=re.I)
    return m.group(1).upper() if m else None


def extract_original_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(p for p in parts if p)


def build_messages(rec: Dict[str, Any], prompt_mode: str) -> List[Dict[str, Any]]:
    original = copy.deepcopy(rec.get("messages") or [])
    text = extract_original_text(original)
    if prompt_mode == "short":
        qidx = text.find("Question:")
        qa = text[qidx:].strip() if qidx >= 0 else text.replace("<motion_start><motion><motion_end>", "").strip()
        text = (
            "<motion_start><motion><motion_end>\n"
            "You are given video evidence and motion-based human pose evidence for a human action "
            "multiple-choice question.\n"
            "Answer with exactly one final option. Do not explain.\n"
            "Return it only in the form <answer>A</answer>, <answer>B</answer>, "
            "<answer>C</answer>, or <answer>D</answer>.\n\n"
            f"{qa}"
        )

    content: List[Dict[str, Any]] = []
    if rec.get("video"):
        content.append({"type": "video", "video": str(rec["video"])})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_inputs(processor: Any, messages: List[Dict[str, Any]], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs: Dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=args.image_patch_size,
    )
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs
    if video_kwargs:
        cleaned_video_kwargs: Dict[str, Any] = {}
        for key, value in video_kwargs.items():
            if isinstance(value, list) and not value:
                continue
            if key == "fps" and isinstance(value, list) and len(value) == 1:
                value = value[0]
            cleaned_video_kwargs[key] = value
        processor_kwargs.update(cleaned_video_kwargs)
    inputs = processor(**processor_kwargs)
    return move_to_device(dict(inputs), device)


def strip_position_keys(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask", "position_ids", "cache_position"}}


def repeat_tensor_for_candidates(key: str, value: torch.Tensor, n: int) -> torch.Tensor:
    if value.dim() == 0:
        return value
    if key in {"pixel_values", "pixel_values_videos"}:
        return value.repeat((n,) + (1,) * (value.dim() - 1))
    if key in {"image_grid_thw", "video_grid_thw"}:
        return value.repeat((n,) + (1,) * (value.dim() - 1))
    if key in {"second_per_grid_ts"}:
        return value.repeat(n) if value.dim() == 1 else value.repeat((n,) + (1,) * (value.dim() - 1))
    if value.shape[0] == 1:
        return value.repeat((n,) + (1,) * (value.dim() - 1))
    return value


def repeat_value_for_candidates(key: str, value: Any, n: int) -> Any:
    if isinstance(value, torch.Tensor):
        return repeat_tensor_for_candidates(key, value, n)
    if isinstance(value, list) and len(value) == 1:
        return value * n
    return value


def score_choices_batched(
    model: Any,
    tokenizer: Any,
    base_inputs: Dict[str, Any],
    candidate_texts: List[str],
    rec: Dict[str, Any],
) -> Dict[str, float]:
    device = base_inputs["input_ids"].device
    n = len(candidate_texts)
    base_ids = base_inputs["input_ids"]
    base_mask = base_inputs.get("attention_mask", torch.ones_like(base_ids))
    cand_rows = [tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device) for text in candidate_texts]
    max_len = max(int(row.numel()) for row in cand_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cand_batch = torch.full((n, max_len), int(pad_id), dtype=torch.long, device=device)
    cand_mask = torch.zeros((n, max_len), dtype=base_mask.dtype, device=device)
    lengths = []
    for i, row in enumerate(cand_rows):
        length = int(row.numel())
        lengths.append(length)
        cand_batch[i, :length] = row
        cand_mask[i, :length] = 1

    input_ids = torch.cat([base_ids.repeat(n, 1), cand_batch], dim=1)
    attention_mask = torch.cat([base_mask.repeat(n, 1), cand_mask], dim=1)
    full_inputs = {k: repeat_value_for_candidates(k, v, n) for k, v in strip_position_keys(base_inputs).items()}
    full_inputs["input_ids"] = input_ids
    full_inputs["attention_mask"] = attention_mask
    branch = str(rec.get("branch", "vm")).lower() or "vm"
    branch_for_model = "vm" if branch == "m" else branch
    outputs = model(
        **full_inputs,
        motion=[rec.get("motion")] * n,
        branch=[branch_for_model] * n,
        sample_id=[rec.get("sample_id")] * n,
        group_id=[rec.get("group_id")] * n,
        use_cache=False,
    )
    logits = outputs.logits.float()
    start = base_ids.shape[1]
    scores: Dict[str, float] = {}
    for i, (choice, cand_ids, length) in enumerate(zip(CHOICES, cand_rows, lengths)):
        pos = torch.arange(start - 1, start + length - 1, device=device)
        log_probs = F.log_softmax(logits[i, pos, :], dim=-1)
        scores[choice] = float(log_probs[torch.arange(length, device=device), cand_ids].sum().item())
    return scores


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    processor_source = args.processor or args.adapter or args.model
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def new_base_model() -> Any:
        base = Qwen3VlMotionForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map={"": args.device},
        )
        if not hasattr(base, "language_model") and hasattr(base, "model") and hasattr(base.model, "language_model"):
            base.language_model = base.model.language_model
        return base

    model = new_base_model()
    if args.adapter:
        adapter_path = Path(args.adapter)
        if (adapter_path / "adapter_config.json").exists() and (adapter_path / "adapter_model.safetensors").exists():
            model = load_adapter_manually(model, args.adapter)
        else:
            if PeftModel is None:
                raise RuntimeError("peft is unavailable, cannot load --adapter")
            model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return processor, model


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(dataset_path, args.limit, args.branch)
    if not records:
        raise RuntimeError(f"No records loaded from {dataset_path} branch={args.branch}")

    processor, model = load_model(args)
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    candidate_texts = [args.candidate_template.format(choice) for choice in CHOICES]

    counters: Dict[str, Counter] = defaultdict(Counter)
    failures = 0
    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for idx, rec in enumerate(tqdm(records, desc="Motion-R1 scoring", unit="sample")):
            try:
                messages = build_messages(rec, args.prompt_mode)
                base_inputs = build_inputs(processor, messages, args, device)
                scores = score_choices_batched(model, tokenizer, base_inputs, candidate_texts, rec)
                pred = max(scores.items(), key=lambda kv: kv[1])[0]
                error = None
            except Exception as exc:
                failures += 1
                scores = {}
                pred = None
                error = f"{type(exc).__name__}: {exc}"
            gt = extract_option_letter(rec.get("answer") or rec.get("solution"))
            correct = bool(gt and pred == gt)
            branch = str(rec.get("branch", args.branch)).lower() or "unknown"
            for key in ("overall", branch):
                counters[key]["total"] += 1
                counters[key]["correct"] += int(correct)
                counters[key][f"gt_{gt or 'missing'}"] += 1
                counters[key][f"pred_{pred or 'error'}"] += 1
            out_f.write(
                json.dumps(
                    {
                        "index": idx,
                        "benchmark_id": rec.get("benchmark_id"),
                        "group_id": rec.get("group_id"),
                        "sample_id": rec.get("sample_id"),
                        "branch": branch,
                        "gt": gt,
                        "pred": pred,
                        "correct": correct,
                        "scores": scores,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    metrics: Dict[str, Any] = {}
    for key in sorted(counters):
        c = counters[key]
        total = int(c["total"])
        correct = int(c["correct"])
        metrics[key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "failures": failures if key == "overall" else None,
            "parse_ok": total - (failures if key == "overall" else 0),
            "parse_rate": (total - (failures if key == "overall" else 0)) / total if total else 0.0,
            "gt_counts": {k[3:]: int(v) for k, v in c.items() if k.startswith("gt_")},
            "pred_counts": {k[5:]: int(v) for k, v in c.items() if k.startswith("pred_")},
        }
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "processor": args.processor,
        "dataset": str(dataset_path),
        "output": str(output_path),
        "prompt_mode": args.prompt_mode,
        "candidate_template": args.candidate_template,
        "num_records": len(records),
        "image_patch_size": args.image_patch_size,
        "metrics": metrics,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    if failures == len(records):
        raise RuntimeError(f"All {failures} records failed; see {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--processor", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch", choices=["all", "vm", "v", "m"], default="vm")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--prompt_mode", choices=["short", "original"], default="short")
    p.add_argument("--candidate_template", default="<answer>{}</answer>")
    p.add_argument("--image_patch_size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
