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

import cv2
import torch
import transformers
from PIL import Image
from peft import PeftModel

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Otter"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "src"))
os.chdir(str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from otter_ai import OtterForConditionalGeneration


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


def extract_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = [round(i * (total - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        pass

    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        video.release()
        raise RuntimeError(f"Could not read frames from {video_path}")
    indices = [round(i * (total - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
    frames: List[Image.Image] = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, min(int(idx), total - 1))
        ok, frame = video.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
    video.release()
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return frames


def formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def load_state(model_path: Path, adapter: Optional[Path], device: torch.device):
    model = OtterForConditionalGeneration.from_pretrained(
        str(model_path),
        device_map=None,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if adapter:
        model.lang_encoder = PeftModel.from_pretrained(model.lang_encoder, str(adapter)).to(device)
    model.text_tokenizer.padding_side = "left"
    if model.text_tokenizer.pad_token_id is None:
        model.text_tokenizer.pad_token = model.text_tokenizer.eos_token
    model.eval()
    image_processor = transformers.CLIPImageProcessor()
    return model, image_processor


def infer_one(state, video_path: str, question: str, num_frames: int, max_new_tokens: int) -> str:
    model, image_processor = state
    tokenizer = model.text_tokenizer
    frames = extract_frames(video_path, num_frames=num_frames)
    vision_x = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    lang_x = tokenizer([formatted_prompt(question)], return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.inference_mode():
        generated = model.generate(
            vision_x=vision_x.to(device, dtype=model_dtype),
            lang_x=lang_x["input_ids"].to(device),
            attention_mask=lang_x["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            no_repeat_ngram_size=3,
        )
    text = tokenizer.decode(generated[0])
    return text.split("<answer>")[-1].split("<|endofchunk|>")[0].strip().strip('"')


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
    parser.add_argument("--num_frames", type=int, default=8)
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
