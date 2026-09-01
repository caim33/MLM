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

import decord
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "Ask-Anything" / "video_chat2"
MODEL_DIR = ROOT / "MVBench_Eval" / "models" / "VideoChat2"
VICUNA_DIR = ROOT / "MVBench_Eval" / "models" / "vicuna-7b-v1.5"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys

sys.path.insert(0, str(SRC))
os.chdir(str(SRC))
decord.bridge.set_bridge("native")

from conversation import Chat
from models import VideoChat2_it_vicuna as VideoChat2_it
from transformers import get_cosine_schedule_with_warmup
from utils.config import Config


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


def make_conversation(question: str, answer: str, frame_msg: str) -> str:
    begin = "###"
    end_signal = " "
    msg = f" {frame_msg} " if frame_msg else ""
    return (
        f"{begin}Human: <Video></Video>{msg.rstrip()}{end_signal}"
        f"{begin}Human: {question}{end_signal}"
        f"{begin}Assistant: {answer}{end_signal}"
        f"{begin}"
    )


def load_model(args: argparse.Namespace, device: torch.device):
    cfg = Config.from_file(str(SRC / "configs" / "config.json"))
    cfg.model.vit_blip_model_path = str(MODEL_DIR / "umt_l16_qformer.pth")
    cfg.model.llama_model_path = str(VICUNA_DIR)
    cfg.model.videochat2_model_path = str(MODEL_DIR / "videochat2_7b_stage3.pth")
    cfg.model.vision_encoder.num_frames = args.num_frames
    cfg.model.vision_encoder.pretrained = ""
    cfg.model.low_resource = False
    cfg.model.freeze_vit = True
    cfg.model.freeze_qformer = True
    cfg.model.max_txt_len = args.model_max_length
    cfg.model.use_lora = True
    cfg.model.lora_r = args.lora_r
    cfg.model.lora_alpha = args.lora_alpha
    cfg.model.lora_dropout = args.lora_dropout
    print("Loading VideoChat2 stage3 with LoRA enabled", flush=True)
    model = VideoChat2_it(config=cfg.model).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}", flush=True)
    return model


def prepare_one(chat: Chat, model, sample: Dict[str, str], num_frames: int, device: torch.device):
    decord.bridge.set_bridge("native")
    vid, frame_msg = chat.load_video(sample["video"], num_segments=num_frames, return_msg=True)
    tc, h, w = vid.shape
    video = vid.reshape(1, tc // 3, 3, h, w).to(device)
    new_pos_emb = chat.get_sinusoid_encoding_table(n_position=(224 // 16) ** 2 * num_frames, cur_frame=num_frames)
    model.vision_encoder.encoder.pos_embed = new_pos_emb.to(device)
    text = make_conversation(sample["question"], sample["answer"], frame_msg)
    instruction = sample["question"][:512]
    return video, [text], [instruction]


def save_trainable(model, output_dir: Path, global_step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"model": trainable_state, "global_step": global_step}, output_dir / "videochat2_lora_trainables.pth")
    if hasattr(model.llama_model, "save_pretrained"):
        model.llama_model.save_pretrained(str(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=1024)
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
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    chat = Chat(model, device=str(device))
    model.train()

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
                video, text, instruction = prepare_one(chat, model, sample, args.num_frames, device)
                with torch.cuda.amp.autocast(enabled=True):
                    loss_dict = model(video, text, instruction)
                    losses.append(loss_dict["loss"])
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
                    save_trainable(model, output_dir / f"checkpoint-{global_step}", global_step)
                if global_step >= total_steps:
                    break
        if args.max_steps <= 0 and epoch >= math.ceil(args.num_train_epochs):
            break
    progress.close()

    save_trainable(model, output_dir, global_step)
    (output_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved VideoChat2 adapter to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
