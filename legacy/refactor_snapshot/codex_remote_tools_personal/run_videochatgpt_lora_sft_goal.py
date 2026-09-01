#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import get_cosine_schedule_with_warmup

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Video-ChatGPT"
sys.path.insert(0, str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from video_chatgpt import video_conversation as conversation_lib
from video_chatgpt.constants import DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import get_spatio_temporal_features_torch


def clean_question(text: str) -> str:
    return text.replace("<video>", "").strip()


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


def build_text(tokenizer: Any, video_token_len: int, question: str, answer: str):
    qs = question + "\n" + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    conv = conversation_lib.conv_templates["video-chatgpt_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer)
    full_text = conv.get_prompt()

    conv_prompt = conversation_lib.conv_templates["video-chatgpt_v1"].copy()
    conv_prompt.append_message(conv_prompt.roles[0], qs)
    conv_prompt.append_message(conv_prompt.roles[1], None)
    prompt_text = conv_prompt.get_prompt()
    input_ids = tokenizer([full_text], return_tensors="pt").input_ids
    prompt_len = tokenizer([prompt_text], return_tensors="pt").input_ids.shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return input_ids, torch.ones_like(input_ids), labels


def extract_features(vision_tower, image_processor, video_path: str, device: torch.device):
    frames = load_video(video_path)
    image_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(device)
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:]
        features = get_spatio_temporal_features_torch(frame_features)
    return features


def prepare_one(state, sample: Dict[str, str], device: torch.device):
    model, vision_tower, tokenizer, image_processor, video_token_len = state
    features = extract_features(vision_tower, image_processor, sample["video"], device)
    input_ids, attention_mask, labels = build_text(tokenizer, video_token_len, sample["question"], sample["answer"])
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "video_spatio_temporal_features": features.unsqueeze(0).to(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=str(ROOT / "MVBench_Eval" / "models" / "LLaVA-7B-Lightening-v1-1"))
    parser.add_argument("--projection_path", default=str(ROOT / "MVBench_Eval" / "models" / "Video-ChatGPT" / "video_chatgpt-7B.bin"))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
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

    print("Loading VideoChatGPT", flush=True)
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name_or_path, args.projection_path)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()
    vision_tower.eval()
    state = (model, vision_tower, tokenizer, image_processor, video_token_len)

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
                batch = prepare_one(state, sample, device)
                outputs = model(**batch)
                losses.append(outputs.loss)
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
                    print(json.dumps({
                        "step": global_step,
                        "loss": accum_loss / max(1, args.logging_steps),
                        "lr": scheduler.get_last_lr()[0],
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }), flush=True)
                    accum_loss = 0.0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                if global_step >= total_steps:
                    break
        if args.max_steps <= 0 and epoch >= math.ceil(args.num_train_epochs):
            break
    progress.close()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved adapter to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
