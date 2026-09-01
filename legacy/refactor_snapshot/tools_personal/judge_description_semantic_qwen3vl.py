#!/usr/bin/env python3
"""Judge description semantic similarity with a local Qwen3-VL model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.shape[0] != 1:
            return False
        row = input_ids[0].tolist()
        return any(len(row) >= len(seq) and row[-len(seq):] == seq for seq in self.stop_sequences)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_sample_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        sid = str(row.get("sample_id") or row.get("id") or row.get("benchmark_id") or row.get("index"))
        out[sid] = row
    return out


def extract_json_text(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    json_text = extract_json_text(text)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def candidate_text(row: Dict[str, Any]) -> str:
    for key in ("final_answer", "extracted_final_answer", "answer_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    extracted = row.get("extracted_json_text")
    if isinstance(extracted, str) and extracted.strip():
        parsed = parse_json_object(extracted)
        if parsed:
            value = parsed.get("final_answer") or parsed.get("answer")
            if isinstance(value, str) and value.strip():
                return value.strip()

    pred = row.get("prediction")
    if isinstance(pred, str) and pred.strip():
        parsed = parse_json_object(pred)
        if parsed:
            value = parsed.get("final_answer") or parsed.get("answer")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return pred.strip()
    return ""


def clamp_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    return score


def build_prompt(gt: str, candidates: Dict[str, str]) -> str:
    return f"""You are evaluating generated human-motion descriptions against a gold reference.

Score each candidate from 0 to 100 for semantic similarity to GOLD. Focus on whether it describes the same action, body pose, temporal progression, limb movement, orientation, and motion details. Penalize hallucinated actions or missed key movement details. Do not reward verbosity by itself.

Also provide dimension scores from 0 to 100:
- action_alignment: same high-level action and temporal sequence
- motion_detail: correct limb, joint, pose, orientation, and dynamic details
- hallucination_control: avoids unsupported or contradictory details

Return ONLY valid JSON with this schema:
{{
  "scores": {{
    "pregrpo": {{"semantic_similarity": 0, "action_alignment": 0, "motion_detail": 0, "hallucination_control": 0}},
    "old600": {{"semantic_similarity": 0, "action_alignment": 0, "motion_detail": 0, "hallucination_control": 0}},
    "new600": {{"semantic_similarity": 0, "action_alignment": 0, "motion_detail": 0, "hallucination_control": 0}}
  }},
  "winner": "pregrpo|old600|new600|tie",
  "reason": "brief comparison"
}}

GOLD:
{gt}

PREGRPO:
{candidates.get("pregrpo", "")}

OLD600:
{candidates.get("old600", "")}

NEW600:
{candidates.get("new600", "")}
"""


def generate_judgment(model: Any, processor: Any, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.closed_empty_think:
        text += "<think>\n\n</think>\n\n"
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    stop_sequences = [tokenizer.encode("<|im_end|>", add_special_tokens=False)]
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|im_end|>", "").strip()


def normalize_judgment(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"scores": {}, "winner": "tie", "reason": ""}
    if not parsed:
        return result
    raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else parsed
    for name in ("pregrpo", "old600", "new600"):
        raw = raw_scores.get(name, {}) if isinstance(raw_scores, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        result["scores"][name] = {
            "semantic_similarity": clamp_score(raw.get("semantic_similarity")),
            "action_alignment": clamp_score(raw.get("action_alignment")),
            "motion_detail": clamp_score(raw.get("motion_detail")),
            "hallucination_control": clamp_score(raw.get("hallucination_control")),
        }
    winner = str(parsed.get("winner", "tie")).lower()
    result["winner"] = winner if winner in {"pregrpo", "old600", "new600", "tie"} else "tie"
    reason = parsed.get("reason")
    result["reason"] = reason.strip() if isinstance(reason, str) else ""
    return result


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("judge_error")]
    score_names = ("semantic_similarity", "action_alignment", "motion_detail", "hallucination_control")
    summary: Dict[str, Any] = {
        "n": len(rows),
        "valid": len(valid_rows),
        "errors": len(rows) - len(valid_rows),
        "model": args.model,
        "output": args.output,
        "winner_counts": dict(Counter(row.get("winner", "tie") for row in valid_rows)),
        "means": {},
    }
    for model_name in ("pregrpo", "old600", "new600"):
        summary["means"][model_name] = {}
        for score_name in score_names:
            vals = [
                float(row["scores"][model_name][score_name])
                for row in valid_rows
                if row.get("scores", {}).get(model_name, {}).get(score_name) is not None
            ]
            summary["means"][model_name][score_name] = round(mean(vals), 4)
    sem = summary["means"]
    summary["deltas"] = {
        "new600_minus_old600_semantic": round(sem["new600"]["semantic_similarity"] - sem["old600"]["semantic_similarity"], 4),
        "new600_minus_pregrpo_semantic": round(sem["new600"]["semantic_similarity"] - sem["pregrpo"]["semantic_similarity"], 4),
        "old600_minus_pregrpo_semantic": round(sem["old600"]["semantic_similarity"] - sem["pregrpo"]["semantic_similarity"], 4),
    }
    return summary


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bench_rows = load_jsonl(Path(args.benchmark))
    pre_rows = by_sample_id(load_jsonl(Path(args.pregrpo)))
    old_rows = by_sample_id(load_jsonl(Path(args.old600)))
    new_rows = by_sample_id(load_jsonl(Path(args.new600)))

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = processor.tokenizer

    rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for idx, bench in enumerate(tqdm(bench_rows[: args.limit], desc="Judging", unit="sample")):
            sid = str(bench.get("sample_id") or bench.get("benchmark_id") or idx)
            gt = bench.get("reference") or bench.get("answer") or bench.get("solution") or ""
            candidates = {
                "pregrpo": candidate_text(pre_rows.get(sid, {})),
                "old600": candidate_text(old_rows.get(sid, {})),
                "new600": candidate_text(new_rows.get(sid, {})),
            }
            row: Dict[str, Any] = {
                "index": idx,
                "sample_id": sid,
                "gt_answer": gt,
                "candidate_chars": {k: len(v) for k, v in candidates.items()},
                "judge_raw": "",
                "judge_error": None,
                "scores": {},
                "winner": "tie",
                "reason": "",
            }
            started = time.time()
            try:
                prompt = build_prompt(gt, candidates)
                raw = generate_judgment(model, processor, tokenizer, prompt, args)
                parsed = parse_json_object(raw)
                normalized = normalize_judgment(parsed)
                row.update(normalized)
                row["judge_raw"] = raw
            except Exception as exc:
                row["judge_error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = round(time.time() - started, 3)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            rows.append(row)

    summary = summarize(rows, args)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pregrpo", required=True)
    parser.add_argument("--old600", required=True)
    parser.add_argument("--new600", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/wangbenyou-sulongjie/Qwen3_vl_motion/model/Qwen3-VL-4B-Thinking")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--closed_empty_think", action="store_true", default=True)
    parser.add_argument("--no_closed_empty_think", dest="closed_empty_think", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
