#!/usr/bin/env python3
"""Evaluate API multimodal models on MotionX MCQ with sampled video frames."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]


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
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text, flags=re.I)
    return m.group(1).upper() if m else None


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


def make_prompt(text: str) -> str:
    text = text.replace("<motion_start><motion><motion_end>", "").strip()
    qidx = text.find("Question:")
    qa = text[qidx:].strip() if qidx >= 0 else text
    return (
        "You are given sampled frames from a human action video.\n"
        "Answer the multiple-choice question using only the visual evidence.\n"
        "Return exactly one option letter inside <answer>...</answer>. Do not explain.\n\n"
        f"{qa}"
    )


def sample_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("empty video")
        indices = np.linspace(0, total - 1, num=max(1, num_frames), dtype=int).tolist()
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Could not read frames from {video_path}")
        indices = np.linspace(0, total - 1, num=max(1, num_frames), dtype=int).tolist()
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode sampled frames from {video_path}")
        return frames


def image_to_data_url(image: Image.Image, max_side: int, quality: int) -> str:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_openai_responses(args: argparse.Namespace, prompt: str, images: List[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get(args.api_key_env))
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for url in images:
        item: Dict[str, Any] = {"type": "input_image", "image_url": url}
        if args.image_detail:
            item["detail"] = args.image_detail
        content.append(item)
    payload: Dict[str, Any] = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": args.max_output_tokens,
    }
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    response = client.responses.create(**payload)
    return getattr(response, "output_text", "") or str(response)


def call_openai_compatible_chat(args: argparse.Namespace, prompt: str, images: List[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get(args.api_key_env), base_url=args.base_url)
    content: List[Dict[str, Any]] = []
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=args.max_output_tokens,
    )
    return response.choices[0].message.content or ""


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(dataset_path, args.limit, args.branch)
    if not records:
        raise RuntimeError(f"No records loaded from {dataset_path} branch={args.branch}")

    counters: Dict[str, Counter] = defaultdict(Counter)
    failures = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(tqdm(records, desc=f"API {args.provider}", unit="sample")):
            try:
                text = extract_original_text(rec.get("messages") or [])
                prompt = make_prompt(text)
                frames = sample_video_frames(str(rec["video"]), args.num_frames)
                images = [image_to_data_url(frame, args.max_side, args.jpeg_quality) for frame in frames]
                last_error = None
                answer_text = ""
                for attempt in range(args.retries + 1):
                    try:
                        if args.provider == "openai_responses":
                            answer_text = call_openai_responses(args, prompt, images)
                        else:
                            answer_text = call_openai_compatible_chat(args, prompt, images)
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt >= args.retries:
                            raise
                        time.sleep(args.retry_sleep * (attempt + 1))
                pred = extract_option_letter(answer_text)
                error = last_error if pred is None and last_error else None
            except Exception as exc:
                failures += 1
                answer_text = ""
                pred = None
                error = f"{type(exc).__name__}: {exc}"
            gt = extract_option_letter(rec.get("answer") or rec.get("solution"))
            correct = bool(gt and pred == gt)
            branch = str(rec.get("branch", args.branch)).lower() or "unknown"
            for key in ("overall", branch):
                counters[key]["total"] += 1
                counters[key]["correct"] += int(correct)
                counters[key][f"gt_{gt or 'missing'}"] += 1
                counters[key][f"pred_{pred or 'error'}"] += 1
            out_f.write(
                json.dumps(
                    {
                        "index": idx,
                        "benchmark_id": rec.get("benchmark_id"),
                        "group_id": rec.get("group_id"),
                        "sample_id": rec.get("sample_id"),
                        "branch": branch,
                        "gt": gt,
                        "pred": pred,
                        "correct": correct,
                        "answer_text": answer_text,
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
            "parse_rate": (total - (failures if key == "overall" else 0)) / total if total else 0.0,
            "gt_counts": {k[3:]: int(v) for k, v in c.items() if k.startswith("gt_")},
            "pred_counts": {k[5:]: int(v) for k, v in c.items() if k.startswith("pred_")},
        }
    summary = {
        "provider": args.provider,
        "model": args.model,
        "dataset": str(dataset_path),
        "output": str(output_path),
        "num_records": len(records),
        "num_frames": args.num_frames,
        "max_side": args.max_side,
        "image_detail": args.image_detail,
        "metrics": metrics,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai_responses", "openai_compatible_chat"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch", choices=["all", "vm", "v"], default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--base_url", default=None)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--max_side", type=int, default=768)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--image_detail", default="low")
    p.add_argument("--max_output_tokens", type=int, default=64)
    p.add_argument("--reasoning_effort", default="low")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_sleep", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
