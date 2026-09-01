#!/usr/bin/env python3
"""Evaluate Video-LLaVA or Video-LLaVA LoRA on MotionX QA_500 by generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Video-LLaVA"
LEGACY = ROOT / "codex_envs" / "legacy_torch211_cu128"
LEGACY_TF = ROOT / "codex_envs" / "legacy_tf431"
VIDEO_EXTRA = ROOT / "codex_envs" / "video_extra"

for path in reversed([SRC, LEGACY, LEGACY_TF, VIDEO_EXTRA]):
    sys.path.insert(0, str(path))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch  # noqa: E402
import huggingface_hub.utils as hub_utils  # noqa: E402

hub_utils.insecure_hashlib = hashlib

from videollava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from videollava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from videollava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token  # noqa: E402
from videollava.model.builder import load_pretrained_model  # noqa: E402


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
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text, flags=re.I)
    return match.group(1).upper() if match else None


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


def make_short_prompt(text: str) -> str:
    text = text.replace("<motion_start><motion><motion_end>", "").strip()
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text
    return (
        "You are given video evidence for a human action multiple-choice question.\n"
        "Analyze the video carefully and answer with exactly one final option.\n"
        "Do not explain. The final answer must be one of A, B, C, or D.\n"
        "Return it only in the form <answer>A</answer>, <answer>B</answer>, "
        "<answer>C</answer>, or <answer>D</answer>.\n\n"
        f"{qa}"
    )


def build_prompt(rec: Dict[str, Any], prompt_mode: str) -> str:
    text = extract_original_text(rec.get("messages") or [])
    return make_short_prompt(text) if prompt_mode == "short" else text.replace("<motion_start><motion><motion_end>", "")


def load_state(model: Path, adapter: Optional[Path], device: str, device_map: str):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_path = str(adapter or model)
    model_base = str(model) if adapter else None
    model_name = "videollava-lora" if adapter else get_model_name_from_path(str(model))
    tokenizer, model_obj, processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device=device,
        device_map=device_map,
    )
    model_obj.eval()
    return tokenizer, model_obj, processor, context_len


def infer_one(state, video_path: str, question: str, max_new_tokens: int) -> str:
    tokenizer, model, processor, _context_len = state
    video_processor = processor["video"]
    device = next(model.parameters()).device
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + "".join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + "\n" + question
    else:
        qs = "".join([DEFAULT_IMAGE_TOKEN] * 8) + "\n" + question
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    video_tensor = video_processor.preprocess(video_path, return_tensors="pt")["pixel_values"][0].half().to(device)
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    input_token_len = input_ids.shape[1]
    output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)]
    return output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(args.model, args.adapter, args.device, args.device_map)

    correct = 0
    parsed = 0
    errors = 0
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(rows):
            gold = extract_option_letter(rec.get("answer") or rec.get("solution"))
            pred = None
            raw = ""
            err = None
            try:
                raw = infer_one(state, str(rec.get("video")), build_prompt(rec, args.prompt_mode), args.max_new_tokens)
                pred = extract_option_letter(raw)
            except Exception as exc:
                errors += 1
                err = repr(exc)
            if pred:
                parsed += 1
            ok = pred == gold if gold else False
            correct += int(ok)
            f.write(json.dumps({
                "index": idx,
                "sample_id": rec.get("sample_id"),
                "group_id": rec.get("group_id"),
                "branch": rec.get("branch"),
                "gold": gold,
                "prediction": pred,
                "correct": ok,
                "raw_output": raw,
                "error": err,
            }, ensure_ascii=False) + "\n")
            f.flush()

    total = len(rows)
    summary = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "output": str(args.output),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
        "errors": errors,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
