#!/usr/bin/env python3
"""Evaluate Motion-R1/Qwen3-VL-Motion adapters by scoring option text candidates."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from eval_motionr1_lora_mcq_score import (  # noqa: E402
    build_inputs,
    build_messages,
    load_model,
    repeat_value_for_candidates,
    strip_position_keys,
)


def load_records(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def score_text_candidates(
    model: Any,
    tokenizer: Any,
    base_inputs: Dict[str, Any],
    candidate_texts: List[str],
    rec: Dict[str, Any],
    length_norm: str,
) -> Dict[int, float]:
    device = base_inputs["input_ids"].device
    n = len(candidate_texts)
    base_ids = base_inputs["input_ids"]
    base_mask = base_inputs.get("attention_mask", torch.ones_like(base_ids))
    cand_rows = [tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device) for text in candidate_texts]
    max_len = max(int(row.numel()) for row in cand_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cand_batch = torch.full((n, max_len), int(pad_id), dtype=torch.long, device=device)
    cand_mask = torch.zeros((n, max_len), dtype=base_mask.dtype, device=device)
    lengths: List[int] = []
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
    scores: Dict[int, float] = {}
    for i, (cand_ids, length) in enumerate(zip(cand_rows, lengths)):
        pos = torch.arange(start - 1, start + length - 1, device=device)
        log_probs = F.log_softmax(logits[i, pos, :], dim=-1)
        token_scores = log_probs[torch.arange(length, device=device), cand_ids]
        score = token_scores.mean() if length_norm == "mean" else token_scores.sum()
        scores[i] = float(score.item())
    return scores


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path, args.limit)
    if not records:
        raise RuntimeError(f"No records loaded from {dataset_path}")

    processor, model = load_model(args)
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)

    counters: Dict[str, Counter] = defaultdict(Counter)
    failures = 0
    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for idx, rec in enumerate(tqdm(records, desc="Option-text scoring", unit="sample")):
            try:
                option_texts = [str(x) for x in rec["option_texts"]]
                gt_idx = int(rec["answer_index"])
                candidate_texts = [args.candidate_template.format(text) for text in option_texts]
                messages = build_messages(rec, args.prompt_mode)
                base_inputs = build_inputs(processor, messages, args, device)
                scores = score_text_candidates(model, tokenizer, base_inputs, candidate_texts, rec, args.length_norm)
                pred_idx = max(scores.items(), key=lambda kv: kv[1])[0]
                error = None
            except Exception as exc:
                failures += 1
                option_texts = [str(x) for x in rec.get("option_texts", [])]
                gt_idx = int(rec.get("answer_index", -1))
                scores = {}
                pred_idx = None
                error = f"{type(exc).__name__}: {exc}"

            correct = pred_idx == gt_idx
            branch = str(rec.get("branch", args.branch)).lower() or "unknown"
            for key in ("overall", branch):
                counters[key]["total"] += 1
                counters[key]["correct"] += int(correct)
                counters[key][f"gt_{gt_idx}"] += 1
                counters[key][f"pred_{pred_idx if pred_idx is not None else 'error'}"] += 1
            out_f.write(
                json.dumps(
                    {
                        "index": idx,
                        "benchmark_id": rec.get("benchmark_id"),
                        "group_id": rec.get("group_id"),
                        "sample_id": rec.get("sample_id"),
                        "branch": branch,
                        "gt_index": gt_idx,
                        "pred_index": pred_idx,
                        "gt_text": option_texts[gt_idx] if 0 <= gt_idx < len(option_texts) else None,
                        "pred_text": option_texts[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(option_texts) else None,
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
        "length_norm": args.length_norm,
        "num_records": len(records),
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
    p.add_argument("--branch", choices=["all", "vm", "v", "m"], default="v")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--prompt_mode", choices=["short", "original"], default="original")
    p.add_argument("--candidate_template", default="<answer>{}</answer>")
    p.add_argument("--length_norm", choices=["sum", "mean"], default="mean")
    p.add_argument("--image_patch_size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
