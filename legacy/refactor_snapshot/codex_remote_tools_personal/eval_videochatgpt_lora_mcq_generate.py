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

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Video-ChatGPT"
sys.path.insert(0, str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import (
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    get_spatio_temporal_features_torch,
)
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.video_conversation import SeparatorStyle, conv_templates


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


def load_state(model_path: Path, projection_path: Path, adapter: Optional[Path]):
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(str(model_path), str(projection_path))
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter)).cuda()
    model.eval()
    return model, vision_tower, tokenizer, image_processor, video_token_len


def infer_one(state, video_path: str, question: str) -> str:
    model, vision_tower, tokenizer, image_processor, video_token_len = state
    frames = load_video(video_path)
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + "\n" + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + "\n" + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    conv = conv_templates["video-chatgpt_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    image_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    with torch.inference_mode():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:]
        video_features = get_spatio_temporal_features_torch(frame_features)
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        output_ids = model.generate(
            input_ids=input_ids,
            video_spatio_temporal_features=video_features.unsqueeze(0),
            do_sample=False,
            max_new_tokens=32,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return outputs.strip().rstrip(stop_str).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(args.model, args.projection, args.adapter)

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
                raw = infer_one(state, str(rec.get("video")), build_prompt(rec, args.prompt_mode))
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
        "projection": str(args.projection),
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
