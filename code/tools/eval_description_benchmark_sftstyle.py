#!/usr/bin/env python3
"""Generate Motion-R1 description benchmark outputs with SFT-style motion placeholders."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration
from qwen_vl_utils import process_vision_info

MOTION_PAD_TOKEN_ID = 160001


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.shape[0] != 1:
            return False
        row = input_ids[0].tolist()
        return any(len(row) >= len(seq) and row[-len(seq):] == seq for seq in self.stop_sequences)


def load_jsonl(path: Path, limit: Optional[int], branch: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row_branch = str(row.get("branch", "")).lower()
            if branch != "all" and row_branch != branch:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def tensor_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tensor_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_to_device(v, device) for v in value]
    return value


def exact_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer does not contain exact token {token!r}")
    recovered = tokenizer.convert_ids_to_tokens(token_id)
    if recovered != token:
        raise ValueError(f"Tokenizer token {token!r} resolved to id={token_id}, recovered={recovered!r}")
    return int(token_id)


def configure_motion_token_metadata(model: Any, tokenizer: Any) -> None:
    motion_start_id = exact_token_id(tokenizer, "<motion_start>")
    motion_end_id = exact_token_id(tokenizer, "<motion_end>")
    setattr(model.config, "motion_start_token_id", motion_start_id)
    setattr(model.config, "motion_end_token_id", motion_end_id)
    patterns = []
    for text in ("<motion>", "<motion_start><motion><motion_end>", "<motion_start> <motion> <motion_end>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            patterns.append([int(x) for x in ids])
    setattr(model.config, "motion_text_token_patterns", patterns)
    if hasattr(model, "_build_motion_text_token_patterns"):
        model._motion_text_token_patterns = model._build_motion_text_token_patterns()


def resolve_motion_path(path_like: Any) -> Path:
    if not path_like:
        raise ValueError("VM sample is missing `motion` path.")
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")
    return path


def motion_frame_count(path_like: Any) -> int:
    path = resolve_motion_path(path_like)
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if not loaded.files:
                raise ValueError(f"Motion npz has no arrays: {path}")
            shape = loaded[loaded.files[0]].shape
        else:
            shape = loaded.shape
        if len(shape) < 1:
            raise ValueError(f"Motion array must have a time dimension, got shape={shape} from {path}")
        return int(shape[0])
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()


def apply_sft_motion_placeholders(
    inputs: Dict[str, Any],
    rec: Dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if not args.sft_motion_placeholders:
        return inputs
    if str(rec.get("branch", "")).lower() != "vm":
        return inputs

    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected single-sample input_ids, got {None if input_ids is None else tuple(input_ids.shape)}")

    motion_start_id = exact_token_id(tokenizer, "<motion_start>")
    motion_end_id = exact_token_id(tokenizer, "<motion_end>")
    ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]

    span = None
    for start, token_id in enumerate(ids):
        if token_id != motion_start_id:
            continue
        for end in range(start + 1, len(ids)):
            if ids[end] == motion_end_id:
                span = (start, end)
                break
        if span is not None:
            break
    if span is None:
        raise ValueError(f"Missing textual motion boundary span for sample_id={rec.get('sample_id')!r}")

    raw_len = motion_frame_count(rec.get("motion"))
    divisor = max(1, int(args.motion_length_divisor))
    padded_len = int(math.ceil(raw_len / divisor) * divisor)
    num_placeholders = max(1, padded_len // divisor)
    replacement = [motion_start_id] + [MOTION_PAD_TOKEN_ID] * num_placeholders + [motion_end_id]
    start, end = span
    converted = torch.tensor([ids[:start] + replacement + ids[end + 1 :]], dtype=input_ids.dtype, device=device)
    inputs["input_ids"] = converted
    inputs["attention_mask"] = torch.ones_like(converted, device=device)
    return inputs


def build_inputs(processor: AutoProcessor, messages: List[Dict[str, Any]], device: torch.device, close_think: bool) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if close_think:
        text += "<think>\n\n</think>\n\n"
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

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


def strip_chat_tail(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    return text.strip()


def extract_json_text(text: str) -> tuple[Optional[str], str]:
    start = text.find("{")
    if start < 0:
        return None, "no_json_start"
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
                return text[start : pos + 1], "json_balanced"
    return text[start:], "json_unclosed"


def parse_description_json(text: str) -> Dict[str, Any]:
    json_text, status = extract_json_text(text)
    out: Dict[str, Any] = {
        "json_status": status,
        "json_parse_ok": False,
        "extracted_json_text": json_text or "",
        "extracted_keys": [],
        "final_answer": "",
    }
    if not json_text:
        return out
    try:
        parsed = json.loads(json_text)
    except Exception:
        return out
    out["json_parse_ok"] = True
    if isinstance(parsed, dict):
        out["extracted_keys"] = sorted(str(k) for k in parsed.keys())
        final_answer = parsed.get("final_answer") or parsed.get("answer") or ""
        if isinstance(final_answer, str):
            out["final_answer"] = final_answer.strip()
    return out


def summarize(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    statuses = Counter(row.get("json_status") for row in rows)
    lengths = [int(row.get("prediction_chars") or 0) for row in rows]
    final_lengths = [int(row.get("final_answer_len") or 0) for row in rows]
    n = len(rows)
    parse_ok = sum(1 for row in rows if row.get("json_parse_ok"))
    return {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "output": args.output,
        "branch": args.branch,
        "n": n,
        "json_parse_ok": parse_ok,
        "json_parse_ok_rate": parse_ok / n if n else 0.0,
        "statuses": dict(statuses),
        "avg_chars": sum(lengths) / n if n else 0.0,
        "max_chars": max(lengths) if lengths else 0,
        "avg_final_answer_chars": sum(final_lengths) / n if n else 0.0,
        "max_final_answer_chars": max(final_lengths) if final_lengths else 0,
        "max_new_tokens": args.max_new_tokens,
        "closed_empty_think": args.closed_empty_think,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(dataset_path, limit=args.limit, branch=args.branch)
    if not records:
        raise RuntimeError(f"No records loaded from {dataset_path} with branch={args.branch}")

    checkpoint = Path(args.checkpoint)
    processor = AutoProcessor.from_pretrained(str(args.processor or checkpoint), trust_remote_code=True)
    model = Qwen3VlMotionForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
    )
    if not hasattr(model, "language_model") and hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.language_model = model.model.language_model
    model.eval()
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    configure_motion_token_metadata(model, tokenizer)

    stop_sequences = [tokenizer.encode("<|im_end|>", add_special_tokens=False)]
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])
    written: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for idx, rec in enumerate(tqdm(records, desc="Description", unit="sample")):
            row: Dict[str, Any] = {
                "index": idx,
                "sample_id": rec.get("sample_id"),
                "benchmark_id": rec.get("benchmark_id"),
                "branch": str(rec.get("branch", "")).lower(),
                "video": rec.get("video"),
                "motion": rec.get("motion"),
                "reference": rec.get("reference") or rec.get("answer") or rec.get("solution") or "",
                "prediction": "",
                "error": None,
            }
            try:
                inputs = build_inputs(processor, rec["messages"], device=device, close_think=args.closed_empty_think)
                inputs = apply_sft_motion_placeholders(inputs, rec, tokenizer, args, device=device)
                branch = row["branch"] or "unknown"
                generated = model.generate(
                    **inputs,
                    motion=rec.get("motion") if branch == "vm" else None,
                    branch=branch,
                    sample_id=rec.get("sample_id"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                prompt_len = inputs["input_ids"].shape[1]
                pred = tokenizer.batch_decode(
                    generated[:, prompt_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )[0]
                pred = strip_chat_tail(pred)
                row["prediction"] = pred
                row["prediction_chars"] = len(pred)
                row.update(parse_description_json(pred))
                row["final_answer_len"] = len(row.get("final_answer") or "")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["prediction_chars"] = 0
                row["json_parse_ok"] = False
                row["json_status"] = "error"
                row["extracted_json_text"] = ""
                row["extracted_keys"] = []
                row["final_answer"] = ""
                row["final_answer_len"] = 0
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            written.append(row)

    summary = summarize(written, args)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=["all", "vm", "v"], default="vm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--sft_motion_placeholders", action="store_true")
    parser.add_argument("--motion_length_divisor", type=int, default=4)
    parser.add_argument("--closed_empty_think", action="store_true", default=True)
    parser.add_argument("--no_closed_empty_think", dest="closed_empty_think", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
