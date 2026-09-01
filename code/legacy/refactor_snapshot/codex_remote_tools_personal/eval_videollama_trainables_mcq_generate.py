#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "video-llama"
CONFIG_PATH = ROOT / "MVBench_Eval" / "scripts" / "video_llama_motionx_eval_only_vl.yaml"
sys.path.insert(0, str(SRC))
os.chdir(str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, conv_llava_llama_2
from video_llama.datasets.builders import *  # noqa: F401,F403
from video_llama.models import *  # noqa: F401,F403
from video_llama.processors import *  # noqa: F401,F403
from video_llama.runners import *  # noqa: F401,F403
from video_llama.tasks import *  # noqa: F401,F403


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
    text = text.replace("<motion_start><motion><motion_end>", "").replace("<video>", "").strip()
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
    if prompt_mode == "short":
        return make_short_prompt(text)
    return text.replace("<motion_start><motion><motion_end>", "").replace("<video>", "")


def load_chat(adapter_dir: Path, num_frames: int) -> Chat:
    args = SimpleNamespace(cfg_path=str(CONFIG_PATH), gpu_id=0, options=None)
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cls = registry.get_model_class(model_config.arch)
    print("Loading VideoLLaMA", flush=True)
    model = model_cls.from_config(model_config)

    trainables_path = adapter_dir / "videollama_trainables.pth"
    if trainables_path.exists():
        payload = torch.load(trainables_path, map_location="cpu")
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded trainables from {trainables_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        raise FileNotFoundError(trainables_path)

    model = model.to(device).eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor_cfg.n_frms = num_frames
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print("VideoLLaMA loaded", flush=True)
    return Chat(model, vis_processor, device=device)


def infer_one(chat: Chat, video_path: str, question: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    conv = conv_llava_llama_2.copy()
    conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully."
    img_list: List[Any] = []
    chat.upload_video_without_audio(video_path, conv, img_list)
    chat.ask(question, conv)
    output, _tokens = chat.answer(
        conv=conv,
        img_list=img_list,
        max_new_tokens=64,
        num_beams=1,
        temperature=0.2,
        max_length=2000,
    )
    return output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt_mode", choices=["short", "full"], default="short")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat = load_chat(args.adapter, args.num_frames)

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
                raw = infer_one(chat, str(rec.get("video")), build_prompt(rec, args.prompt_mode))
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
        "adapter": str(args.adapter),
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
