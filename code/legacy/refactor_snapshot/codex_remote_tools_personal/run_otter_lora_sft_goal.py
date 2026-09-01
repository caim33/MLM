#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import get_cosine_schedule_with_warmup

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Otter"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "src"))
os.chdir(str(SRC))

from otter_ai import OtterForConditionalGeneration


def clean_question(text: str) -> str:
    return text.replace("<video>", "").strip()


def formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def extract_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError("empty video")
        if num_frames <= 1:
            indices = [total_frames // 2]
        else:
            indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        pass

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        raise RuntimeError(f"Could not read frames from {video_path}")
    if num_frames <= 1:
        indices = [total_frames // 2]
    else:
        indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames: List[Image.Image] = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, min(int(idx), total_frames - 1))
        ok, frame = video.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
    video.release()
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return frames


class MotionXSftDataset(Dataset):
    def __init__(self, path: str, branch: str = "all", limit: int | None = None):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if branch != "all":
            rows = [r for r in rows if str(r.get("branch", "")).lower() == branch]
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        conv = row["conversations"]
        return {
            "id": str(row.get("id") or row.get("sample_id") or idx),
            "video": str(row["video"]),
            "question": clean_question(str(conv[0]["value"])),
            "answer": str(conv[1]["value"]).strip(),
        }


def collate_identity(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return batch


def prepare_one(model: OtterForConditionalGeneration, image_processor, sample: Dict[str, str], num_frames: int, device: torch.device):
    tokenizer = model.text_tokenizer
    prompt = formatted_prompt(sample["question"])
    eos = "<|endofchunk|>"
    full_text = prompt + sample["answer"] + eos
    frames = extract_frames(sample["video"], num_frames=num_frames)
    vision_x = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    full = tokenizer([full_text], return_tensors="pt", padding=True)
    prompt_ids = tokenizer([prompt], return_tensors="pt", padding=False)["input_ids"]

    lang_x = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = lang_x.clone()
    labels[:, : int(prompt_ids.shape[1])] = -100
    if tokenizer.pad_token_id is not None:
        labels[lang_x == tokenizer.pad_token_id] = -100

    model_dtype = next(model.parameters()).dtype
    return {
        "vision_x": vision_x.to(device=device, dtype=model_dtype),
        "lang_x": lang_x.to(device=device),
        "attention_mask": attention_mask.to(device=device),
        "labels": labels.to(device=device),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=str(ROOT / "MVBench_Eval" / "models" / "OTTER-Video-LLaMA7B-DenseCaption"))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Otter-Video from {args.model_name_or_path}", flush=True)
    model = OtterForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.text_tokenizer.padding_side = "right"
    if model.text_tokenizer.pad_token_id is None:
        model.text_tokenizer.pad_token = model.text_tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.lang_encoder = get_peft_model(model.lang_encoder, peft_config)
    model.lang_encoder.print_trainable_parameters()
    model.train()
    model.vision_encoder.eval()
    model.perceiver.eval()

    image_processor = transformers.CLIPImageProcessor()
    dataset = MotionXSftDataset(args.data_path, branch=args.branch, limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_identity)
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.num_train_epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"Training samples={len(dataset)} total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps)
    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for micro_step, samples in enumerate(dataloader, start=1):
            losses = []
            for sample in samples:
                batch = prepare_one(model, image_processor, sample, args.num_frames, device)
                outputs = model(**batch)
                losses.append(outputs.loss if hasattr(outputs, "loss") else outputs[0])
            loss = torch.stack(losses).mean() / args.gradient_accumulation_steps
            loss.backward()
            accum_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if global_step % args.logging_steps == 0:
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "loss": accum_loss / max(1, args.logging_steps),
                                "lr": scheduler.get_last_lr()[0],
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    accum_loss = 0.0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.lang_encoder.save_pretrained(str(ckpt))
                    model.text_tokenizer.save_pretrained(str(ckpt))
                if global_step >= total_steps:
                    break
        if args.max_steps <= 0 and epoch >= math.ceil(args.num_train_epochs):
            break
    progress.close()

    model.lang_encoder.save_pretrained(str(output_dir))
    model.text_tokenizer.save_pretrained(str(output_dir))
    (output_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved adapter to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
