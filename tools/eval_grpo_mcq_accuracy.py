#!/usr/bin/env python3
"""Evaluate Motion-R1 GRPO MCQ accuracy on ms-swift style JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration
from qwen_vl_utils import process_vision_info

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
OPTION_PATTERN = re.compile(r"^\s*([ABCD])\s*[\).\-\:]*\s*$", flags=re.IGNORECASE)


def extract_answer_text(text: str) -> str:
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def extract_option_letter(text: str) -> str:
    answer_text = extract_answer_text(text)
    if not answer_text:
        return ""
    match = OPTION_PATTERN.match(answer_text)
    if match:
        return match.group(1).upper()
    token_match = re.search(r"\b([ABCD])\b", answer_text, flags=re.IGNORECASE)
    return token_match.group(1).upper() if token_match else ""


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.shape[0] != 1:
            return False
        row = input_ids[0].tolist()
        for seq in self.stop_sequences:
            if len(row) >= len(seq) and row[-len(seq):] == seq:
                return True
        return False


def load_jsonl(path: Path, limit: Optional[int] = None, branch: str = "all") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item_branch = str(item.get("branch", "")).lower()
            if branch != "all" and item_branch != branch:
                continue
            records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def tensor_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tensor_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_to_device(v, device) for v in value]
    return value


def build_inputs(processor: AutoProcessor, messages: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
    # qwen-vl-utils may return single-sample video kwargs as one-element lists
    # (e.g. fps=[2.0]), while Qwen3VLProcessor validates scalar fields.
    normalized_video_kwargs = {}
    for key, value in video_kwargs.items():
        if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], (list, tuple, dict)):
            normalized_video_kwargs[key] = value[0]
        else:
            normalized_video_kwargs[key] = value

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **normalized_video_kwargs,
    )
    return tensor_to_device(dict(inputs), device)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset)
    checkpoint = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(dataset_path, limit=args.limit, branch=args.branch)
    if not records:
        raise RuntimeError(f"No records loaded from {dataset_path} with branch={args.branch}")

    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = Qwen3VlMotionForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
    )
    if not hasattr(model, "language_model") and hasattr(model, "model") and hasattr(model.model, "language_model"):
        # The training stack exposes this alias before calling the custom forward.
        # Direct standalone loading needs the same compatibility alias.
        model.language_model = model.model.language_model
    model.eval()
    device = next(model.parameters()).device

    tokenizer = processor.tokenizer
    stop_texts = ["</answer>", "<|im_end|>"]
    stop_sequences = [tokenizer.encode(x, add_special_tokens=False) for x in stop_texts]
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])

    counters: Dict[str, Counter] = defaultdict(Counter)

    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for idx, rec in enumerate(tqdm(records, desc="Evaluating", unit="sample")):
            branch = str(rec.get("branch", "unknown")).lower() or "unknown"
            messages = rec["messages"]
            inputs = build_inputs(processor, messages, device=device)

            motion_arg = rec.get("motion") if branch == "vm" else None
            generated = model.generate(
                **inputs,
                motion=motion_arg,
                branch=branch,
                sample_id=rec.get("sample_id"),
                group_id=rec.get("group_id"),
                rollout_id=rec.get("rollout_id"),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = generated[:, prompt_len:]
            pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

            gt_text = rec.get("answer") or rec.get("solution") or ""
            pred = extract_option_letter(pred_text)
            gt = extract_option_letter(gt_text)
            correct = bool(pred and gt and pred == gt)
            parse_ok = bool(pred)

            for key in ("overall", branch):
                counters[key]["total"] += 1
                counters[key]["correct"] += int(correct)
                counters[key]["parse_ok"] += int(parse_ok)
                counters[key][f"gt_{gt or 'missing'}"] += 1
                counters[key][f"pred_{pred or 'missing'}"] += 1

            row = {
                "index": idx,
                "group_id": rec.get("group_id"),
                "sample_id": rec.get("sample_id"),
                "branch": branch,
                "gt": gt,
                "pred": pred,
                "correct": correct,
                "parse_ok": parse_ok,
                "answer": gt_text,
                "prediction": pred_text,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "output": str(output_path),
        "max_new_tokens": args.max_new_tokens,
        "num_records": len(records),
        "metrics": {},
    }
    for key in sorted(counters.keys()):
        c = counters[key]
        total = int(c["total"])
        correct = int(c["correct"])
        parse_ok = int(c["parse_ok"])
        summary["metrics"][key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "parse_ok": parse_ok,
            "parse_rate": parse_ok / total if total else 0.0,
            "gt_counts": {k[3:]: int(v) for k, v in c.items() if k.startswith("gt_")},
            "pred_counts": {k[5:]: int(v) for k, v in c.items() if k.startswith("pred_")},
        }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=["all", "vm", "v"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
