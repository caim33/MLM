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
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "video-llama"
CONFIG_PATH = ROOT / "MVBench_Eval" / "scripts" / "video_llama_motionx_eval_only_vl.yaml"
sys.path.insert(0, str(SRC))
os.chdir(str(SRC))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset
from video_llama.models import *  # noqa: F401,F403
from video_llama.processors import *  # noqa: F401,F403


def clean_question(text: str) -> str:
    return text.replace("<video>", "").strip()


def convert_sft_to_videollama(data_path: Path, out_path: Path, branch: str, limit: int | None) -> Path:
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if branch != "all" and str(row.get("branch", "")).lower() != branch:
            continue
        conv = row["conversations"]
        out.append({
            "video": str(row["video"]),
            "QA": [{"q": clean_question(str(conv[0]["value"])), "a": str(conv[1]["value"]).strip()}],
        })
        if limit is not None and len(out) >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_model(device: torch.device):
    args = SimpleNamespace(cfg_path=str(CONFIG_PATH), gpu_id=0, options=None)
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in ("llama_proj", "video_Qformer", "video_query_tokens"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}", flush=True)
    return model, cfg


def make_dataset(annotation: Path, cfg: Config, num_frames: int):
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor_cfg.n_frms = num_frames
    tokenizer_name = str(cfg.model_cfg.llama_model)
    return Video_Instruct_Dataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/",
        ann_root=str(annotation),
        num_video_query_token=int(getattr(cfg.model_cfg, "num_video_query_token", 32)),
        tokenizer_name=tokenizer_name,
        data_type="video",
        model_type="llama_v2",
    )


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_trainable(model, output_dir: Path, global_step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if any(t in k for t in ("llama_proj", "video_Qformer", "video_query_tokens"))}
    torch.save({"model": state, "global_step": global_step}, output_dir / "videollama_trainables.pth")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", default=str(ROOT / "codex_runs" / "finetune_goal_20260717" / "video_lora" / "_videollama_work"))
    parser.add_argument("--branch", default="all")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    annotation = convert_sft_to_videollama(Path(args.data_path), work_dir / "videollama_motionx_sft.json", args.branch, args.limit)

    print("Loading VideoLLaMA", flush=True)
    model, cfg = load_model(device)
    dataset = make_dataset(annotation, cfg, args.num_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=dataset.collater,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.num_train_epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    print(f"Training samples={len(dataset)} total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    model.train()
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps)
    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for micro_step, batch in enumerate(dataloader, start=1):
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=True):
                loss = model(batch)["loss"] / args.gradient_accumulation_steps
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
    print(f"Saved trainables to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
