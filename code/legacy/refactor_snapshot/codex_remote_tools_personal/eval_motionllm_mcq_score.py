#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SCRIPT_DIR = ROOT / "MVBench_Eval" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_motionllm_motionx import load_motionllm


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


def build_prompt(rec: Dict[str, Any]) -> str:
    text = extract_original_text(rec.get("messages") or [])
    text = text.replace("<motion_start><motion><motion_end>", "").replace("<video>", "").strip()
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text
    return f"{qa}\nAnswer with only one letter: A, B, C, or D."


def option_token_ids(tokenizer, device) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in "ABCD":
        ids = set()
        for text in (letter, " " + letter, letter.lower(), " " + letter.lower()):
            tok = tokenizer.encode(text, bos=False, eos=False, device=device).view(-1).tolist()
            if tok:
                ids.add(int(tok[-1]))
        out[letter] = sorted(ids)
    return out


def score_one(state: Dict[str, Any], video_path: str, prompt: str, opt_ids: Dict[str, List[int]]) -> tuple[str, Dict[str, float]]:
    device = state["device"]
    model = state["model"]
    tokenizer = state["tokenizer"]
    video_processor = state["video_processor"]
    mm_backbone = state["mm_backbone"]

    video_tensor = video_processor(video_path, return_tensors="pt")["pixel_values"]
    if isinstance(video_tensor, list):
        tensor = [v.to(device, dtype=torch.float16) for v in video_tensor]
    else:
        tensor = video_tensor.to(device, dtype=torch.float16)

    with torch.no_grad():
        video_feature = mm_backbone.get_multimodal_embeddings([tensor, ["video"]])

    prefix = (
        "A chat between a curious user and an artificial intelligence assistant, paired with an input "
        "that provides further context. The assistant gives helpful, detailed, and polite answers to "
        f"the user's questions. USER: {prompt} INPUT_VIDEO: {video_path}. \nASSISTANT: "
    )
    pre = torch.cat(
        (
            tokenizer.encode(prefix.split("INPUT_VIDEO: ")[0] + "\n", bos=True, eos=False, device=model.device).view(1, -1),
            tokenizer.encode("INPUT_VIDEO: ", bos=False, eos=False, device=model.device).view(1, -1),
        ),
        dim=1,
    )
    post = tokenizer.encode(". ASSISTANT: ", bos=False, eos=False, device=model.device).view(1, -1)
    before_len = pre.shape[-1]
    full = torch.cat((pre, torch.zeros((1, len(video_feature[0])), device=model.device, dtype=pre.dtype), post), dim=1).long()
    encoded = (full, video_feature[0].unsqueeze(0), [before_len])
    with torch.no_grad():
        logits = model(encoded, maxlen=full.size(1))[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)

    scores: Dict[str, float] = {}
    for letter, ids in opt_ids.items():
        vals = [float(log_probs[i].item()) for i in ids if i < log_probs.numel()]
        scores[letter] = max(vals) if vals else -math.inf
    pred = max(scores, key=scores.get)
    return pred, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    rows = load_jsonl(args.dataset, args.limit, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = load_motionllm()
    opt_ids = option_token_ids(state["tokenizer"], state["model"].device)

    correct = 0
    errors = 0
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(rows):
            gold = extract_option_letter(rec.get("answer") or rec.get("solution"))
            pred = None
            scores: Dict[str, float] = {}
            err = None
            try:
                pred, scores = score_one(state, str(rec.get("video")), build_prompt(rec), opt_ids)
            except Exception as exc:
                errors += 1
                err = repr(exc)
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
                "scores": scores,
                "error": err,
            }, ensure_ascii=False) + "\n")
            f.flush()

    total = len(rows)
    parsed = total - errors
    summary = {
        "model": "MotionLLM",
        "method": "option_logprob",
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
