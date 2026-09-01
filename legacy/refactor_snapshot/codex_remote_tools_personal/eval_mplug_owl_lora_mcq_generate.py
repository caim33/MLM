#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoTokenizer

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "mPLUG-Owl" / "mPLUG-Owl"
sys.path.insert(0, str(SRC))
os.chdir(str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from mplug_owl_video.configuration_mplug_owl import MplugOwlConfig
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

MplugOwlConfig.__repr__ = lambda self: f"{self.__class__.__name__}(model_type={getattr(self, 'model_type', None)})"

PROMPT_PREFIX = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: {question}\n"
    "AI: "
)


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


def load_state(model_path: Path, adapter: Optional[Path], device: torch.device):
    model = MplugOwlForConditionalGeneration.from_pretrained(str(model_path), torch_dtype=torch.bfloat16).to(device)
    if adapter:
        model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter)).to(device)
    model.eval()
    image_processor = MplugOwlImageProcessor.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    processor = MplugOwlProcessor(image_processor, tokenizer)
    return model, tokenizer, processor


def clean_generation(text: str) -> str:
    if "AI:" in text:
        text = text.split("AI:")[-1]
    return text.strip()


def infer_one(state, video_path: str, question: str, num_frames: int, max_new_tokens: int) -> str:
    model, tokenizer, processor = state
    prompt = PROMPT_PREFIX.format(question=question)
    inputs = processor(text=[prompt], videos=[video_path], num_frames=num_frames, return_tensors="pt")
    casted = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            value = value.to(model.device, dtype=torch.bfloat16)
        else:
            value = value.to(model.device)
        casted[key] = value
    max_length = int(casted["input_ids"].shape[1]) + max_new_tokens
    with torch.inference_mode():
        output_ids = model.generate(
            **casted,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
        )
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    return clean_generation(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(args.model, args.adapter, torch.device(args.device))

    correct = 0
    parsed = 0
    errors = 0
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(rows):
            gold = extract_option_letter(rec.get("answer") or rec.get("solution"))
            raw = ""
            pred = None
            err = None
            try:
                raw = infer_one(state, str(rec.get("video")), build_prompt(rec, args.prompt_mode), args.num_frames, args.max_new_tokens)
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
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
