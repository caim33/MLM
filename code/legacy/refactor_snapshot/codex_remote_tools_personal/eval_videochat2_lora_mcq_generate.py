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

import decord
import torch

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Ask-Anything" / "video_chat2"
MODEL_DIR = ROOT / "MVBench_Eval" / "models" / "VideoChat2"
VICUNA_DIR = ROOT / "MVBench_Eval" / "models" / "vicuna-7b-v1.5"
sys.path.insert(0, str(SRC))
os.chdir(str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from conversation import Chat
from models import VideoChat2_it_vicuna as VideoChat2_it
from utils.config import Config
from utils.easydict import EasyDict


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


def load_state(adapter: Optional[Path], device: str, num_frames: int, model_max_length: int):
    cfg = Config.from_file(str(SRC / "configs" / "config.json"))
    cfg.model.vit_blip_model_path = str(MODEL_DIR / "umt_l16_qformer.pth")
    cfg.model.llama_model_path = str(VICUNA_DIR)
    cfg.model.videochat2_model_path = str(MODEL_DIR / "videochat2_7b_stage3.pth")
    cfg.model.vision_encoder.num_frames = num_frames
    cfg.model.vision_encoder.pretrained = ""
    cfg.model.low_resource = False
    cfg.model.freeze_vit = True
    cfg.model.freeze_qformer = True
    cfg.model.max_txt_len = model_max_length
    cfg.model.use_lora = True
    cfg.model.lora_r = 16
    cfg.model.lora_alpha = 32
    cfg.model.lora_dropout = 0.0

    model = VideoChat2_it(config=cfg.model).to(torch.device(device))
    if adapter:
        state_path = adapter / "videochat2_lora_trainables.pth"
        payload = torch.load(state_path, map_location="cpu")
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        msg = model.load_state_dict(state, strict=False)
        print(msg, flush=True)
    model.eval()
    return Chat(model, device=device)


def infer_one(chat: Chat, video_path: str, question: str, num_frames: int, max_new_tokens: int) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    decord.bridge.set_bridge("native")
    conv = EasyDict({"system": "", "roles": ("Human", "Assistant"), "messages": [], "sep": "###"})
    img_list = []
    _msg, img_list, conv = chat.upload_video(video_path, conv, img_list, num_segments=num_frames)
    conv = chat.ask(question, conv)
    output, _tokens, _conv = chat.answer(
        conv=conv,
        img_list=img_list,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=0.2,
    )
    return output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat = load_state(args.adapter, args.device, args.num_frames, args.model_max_length)

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
                raw = infer_one(chat, str(rec.get("video")), build_prompt(rec, args.prompt_mode), args.num_frames, args.max_new_tokens)
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
