#!/usr/bin/env python3
"""Evaluate open VLM/text models on MotionX MCQ by candidate log-prob scoring."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover - depends on transformers build
    Qwen3VLForConditionalGeneration = None  # type: ignore

from qwen_vl_utils import process_vision_info

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional at runtime
    PeftModel = None  # type: ignore

CHOICES = ["A", "B", "C", "D"]


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


def make_short_prompt(text: str, input_mode: str) -> str:
    text = text.replace("<motion_start><motion><motion_end>", "").strip()
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text
    if input_mode == "text":
        evidence = "You are given a human action multiple-choice question."
    elif input_mode == "frames":
        evidence = "You are given sampled video frames for a human action question."
    else:
        evidence = "You are given video evidence for a human action question."
    return (
        f"{evidence}\n"
        "Answer with exactly one final option.\n"
        "Do not explain. The final answer must be one of A, B, C, or D.\n"
        "Return it only in the form <answer>A</answer>, <answer>B</answer>, <answer>C</answer>, or <answer>D</answer>.\n\n"
        f"{qa}"
    )


def sample_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("empty video")
        indices = np.linspace(0, total - 1, num=max(1, num_frames), dtype=int).tolist()
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Could not read frames from {video_path}")
        indices = np.linspace(0, total - 1, num=max(1, num_frames), dtype=int).tolist()
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode sampled frames from {video_path}")
        return frames


def add_pixel_limits(part: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    part = dict(part)
    for key in ("min_pixels", "max_pixels", "total_pixels"):
        value = getattr(args, key)
        if value is not None:
            part[key] = value
    return part


def build_messages(rec: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    original = copy.deepcopy(rec.get("messages") or [])
    text = extract_original_text(original)
    prompt = make_short_prompt(text, args.input_mode) if args.prompt_mode == "short" else text

    content: List[Dict[str, Any]] = []
    if args.input_mode == "video":
        video_path = rec.get("video")
        if not video_path:
            raise ValueError(f"record has no video: {rec.get('benchmark_id') or rec.get('sample_id')}")
        video_part: Dict[str, Any] = {"type": "video", "video": str(video_path)}
        if args.fps is not None:
            video_part["fps"] = args.fps
        if args.nframes is not None:
            video_part["nframes"] = args.nframes
        content.append(add_pixel_limits(video_part, args))
    elif args.input_mode == "frames":
        video_path = rec.get("video")
        if not video_path:
            raise ValueError(f"record has no video: {rec.get('benchmark_id') or rec.get('sample_id')}")
        for frame in sample_video_frames(str(video_path), args.num_frames):
            content.append(add_pixel_limits({"type": "image", "image": frame}, args))
    elif args.input_mode != "text":
        raise ValueError(f"Unsupported input_mode={args.input_mode}")
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def apply_chat_template(processor: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_inputs(processor: Any, messages: List[Dict[str, Any]], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    text = apply_chat_template(processor, messages, add_generation_prompt=True)
    processor_kwargs: Dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
    if args.input_mode in {"video", "frames"}:
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
            fps_value = video_kwargs.get("fps")
            if isinstance(fps_value, list):
                if len(fps_value) == 1:
                    video_kwargs["fps"] = fps_value[0]
                elif len(fps_value) == 0:
                    video_kwargs.pop("fps", None)
            processor_kwargs.update(video_kwargs)
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
    outputs = model(**full_inputs, use_cache=False)
    logits = outputs.logits.float()
    start = base_ids.shape[1]
    scores: Dict[str, float] = {}
    for i, (choice, cand_ids, length) in enumerate(zip(CHOICES, cand_rows, lengths)):
        pos = torch.arange(start - 1, start + length - 1, device=device)
        log_probs = F.log_softmax(logits[i, pos, :], dim=-1)
        scores[choice] = float(log_probs[torch.arange(length, device=device), cand_ids].sum().item())
    return scores


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    base_load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "single":
        base_load_kwargs["device_map"] = {"": args.device}
    elif args.device_map:
        base_load_kwargs["device_map"] = args.device_map

    processor_source = args.processor or args.adapter or args.model
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    errors: List[str] = []
    class_order: List[str]
    if args.model_class == "auto":
        class_order = ["qwen3vl", "image_text", "causal_lm"]
    else:
        class_order = [args.model_class]
    attn_order = [args.attn_implementation] if args.attn_implementation else [None]
    if args.attn_implementation and args.attn_implementation != "eager":
        attn_order.append("eager")
    if None not in attn_order:
        attn_order.append(None)
    for cls in class_order:
        for attn_impl in attn_order:
            load_kwargs = dict(base_load_kwargs)
            if attn_impl:
                load_kwargs["attn_implementation"] = attn_impl
            try:
                if cls == "qwen3vl":
                    if Qwen3VLForConditionalGeneration is None:
                        raise RuntimeError("Qwen3VLForConditionalGeneration is unavailable")
                    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
                elif cls == "image_text":
                    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
                elif cls == "causal_lm":
                    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
                else:
                    raise ValueError(f"unknown model_class={cls}")
                if args.adapter:
                    if PeftModel is None:
                        raise RuntimeError("peft is unavailable, cannot load --adapter")
                    model = PeftModel.from_pretrained(model, args.adapter)
                model.eval()
                return processor, model
            except Exception as exc:
                errors.append(f"{cls}/attn={attn_impl or 'default'}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load model. Attempts:\n" + "\n".join(errors))


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
        for idx, rec in enumerate(tqdm(records, desc=f"Scoring {args.input_mode}", unit="sample")):
            try:
                messages = build_messages(rec, args)
                base_inputs = build_inputs(processor, messages, args, device)
                scores = score_choices_batched(model, tokenizer, base_inputs, candidate_texts)
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
            row = {
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
                "input_mode": args.input_mode,
                "candidate_template": args.candidate_template,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        "input_mode": args.input_mode,
        "prompt_mode": args.prompt_mode,
        "candidate_template": args.candidate_template,
        "num_records": len(records),
        "num_frames": args.num_frames if args.input_mode == "frames" else None,
        "fps": args.fps if args.input_mode == "video" else None,
        "nframes": args.nframes if args.input_mode == "video" else None,
        "image_patch_size": args.image_patch_size,
        "max_pixels": args.max_pixels,
        "total_pixels": args.total_pixels,
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
    p.add_argument("--adapter", default=None, help="Optional PEFT adapter directory")
    p.add_argument("--processor", default=None, help="Optional processor/tokenizer directory")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch", choices=["all", "vm", "v"], default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input_mode", choices=["video", "frames", "text"], default="video")
    p.add_argument("--model_class", choices=["auto", "qwen3vl", "image_text", "causal_lm"], default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_map", default="single", help="'single', 'auto', or empty string")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--prompt_mode", choices=["short", "original"], default="short")
    p.add_argument("--candidate_template", default="<answer>{}</answer>")
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--nframes", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=200704, help="Per image/video frame max pixels")
    p.add_argument("--total_pixels", type=int, default=1605632, help="Video total pixel budget")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
