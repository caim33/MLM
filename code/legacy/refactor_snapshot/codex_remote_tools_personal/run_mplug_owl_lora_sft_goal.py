#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path("/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM")
SRC = ROOT / "codex_runs" / "video_model_sources" / "mPLUG-Owl" / "mPLUG-Owl"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys

sys.path.insert(0, str(SRC))
os.chdir(str(SRC))

from mplug_owl_video.configuration_mplug_owl import MplugOwlConfig
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration, get_media_indices, get_media_types
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

MplugOwlConfig.__repr__ = lambda self: f"{self.__class__.__name__}(model_type={getattr(self, 'model_type', None)})"

PROMPT_PREFIX = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: {question}\n"
    "AI: "
)


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


def prepare_one(processor: MplugOwlProcessor, tokenizer: Any, sample: Dict[str, str], num_frames: int, device: torch.device):
    prompt = PROMPT_PREFIX.format(question=sample["question"])
    answer = sample["answer"]
    eos = tokenizer.eos_token or "</s>"
    full_text = prompt + answer + eos

    inputs = processor(text=[full_text], videos=[sample["video"]], num_frames=num_frames, return_tensors="pt")
    prompt_ids = processor(text=[prompt], return_tensors="pt")["input_ids"]
    prompt_len = int(prompt_ids.shape[1])

    batch: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            batch[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            batch[key] = value.to(device=device)

    labels = batch["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[batch["input_ids"] < 0] = -100
    batch["labels"] = labels
    return batch


def build_video_language_inputs(model: MplugOwlForConditionalGeneration, batch: Dict[str, torch.Tensor]):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]
    video_pixel_values = batch["video_pixel_values"]
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    batch_size = input_ids.size(0)
    media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)]
    media_token_types = [get_media_types(input_ids[i], media_token_indices[i]) for i in range(batch_size)]
    num_videos_per_sample = [len([y for y in x if y < -1]) for x in media_token_types]

    text_ids = input_ids.clone()
    text_ids[text_ids < 0] = 0
    inputs_embeds = model.get_input_embeddings()(text_ids)
    if hasattr(model.language_model, "transformer") and hasattr(model.language_model.transformer, "word_embeddings_layernorm"):
        inputs_embeds = model.language_model.transformer.word_embeddings_layernorm(inputs_embeds)

    with torch.no_grad():
        video_embeds = model.vision_model(video_pixel_values, return_dict=True).last_hidden_state
        video_attention_mask = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=video_embeds.device)
        import einops

        video_attention_mask = einops.rearrange(video_attention_mask, "b t n -> b (t n)")
        query_tokens = model.query_tokens.expand(video_embeds.shape[0], -1, -1)
        temporal_query_tokens = model.temporal_query_tokens.expand(video_embeds.shape[0], -1, -1)
        query_outputs = model.abstractor(
            query_embeds=query_tokens,
            temporal_query_embeds=temporal_query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_attention_mask,
            return_dict=True,
        )
        video_embeds = query_outputs["last_hidden_state"]
    vid_seq_length = int(video_embeds.shape[1])

    text_chunk_embeds = []
    text_chunk_attns = []
    text_chunk_labels = []
    vid_idx = 0
    for b in range(batch_size):
        start = 0
        sample_embeds = []
        sample_attns = []
        sample_labels = []
        curr_video_idx = 0
        for i, pos in enumerate(media_token_indices[b]):
            if pos > start:
                sample_embeds.append(inputs_embeds[b, start:pos])
                sample_attns.append(attention_mask[b, start:pos])
                sample_labels.append(labels[b, start:pos])
            if media_token_types[b][i] >= -1:
                raise ValueError("mPLUG SFT script only expects video media tokens")
            visual = video_embeds[vid_idx + curr_video_idx]
            sample_embeds.append(visual)
            sample_attns.append(torch.ones(visual.shape[0], device=inputs_embeds.device, dtype=attention_mask.dtype))
            sample_labels.append(torch.full((visual.shape[0],), -100, device=inputs_embeds.device, dtype=labels.dtype))
            start = pos + vid_seq_length
            curr_video_idx += 1
        if start < inputs_embeds.shape[1]:
            sample_embeds.append(inputs_embeds[b, start:])
            sample_attns.append(attention_mask[b, start:])
            sample_labels.append(labels[b, start:])
        vid_idx += num_videos_per_sample[b]
        text_chunk_embeds.append(torch.cat(sample_embeds, dim=0))
        text_chunk_attns.append(torch.cat(sample_attns, dim=0))
        text_chunk_labels.append(torch.cat(sample_labels, dim=0))

    return {
        "inputs_embeds": torch.stack(text_chunk_embeds, dim=0),
        "attention_mask": torch.stack(text_chunk_attns, dim=0),
        "labels": torch.stack(text_chunk_labels, dim=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=str(ROOT / "MVBench_Eval" / "models" / "mplug-owl-llama-7b-video"))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--branch", default="all")
    parser.add_argument("--num_frames", type=int, default=4)
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

    print(f"Loading mPLUG-Owl-Video from {args.model_name_or_path}", flush=True)
    model = MplugOwlForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
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
    model.language_model = get_peft_model(model.language_model, peft_config)
    model.language_model.print_trainable_parameters()
    model.train()
    model.vision_model.eval()
    model.abstractor.eval()

    image_processor = MplugOwlImageProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = MplugOwlProcessor(image_processor, tokenizer)

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
                batch = prepare_one(processor, tokenizer, sample, args.num_frames, device)
                lm_inputs = build_video_language_inputs(model, batch)
                outputs = model.language_model(**lm_inputs)
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
                    model.language_model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                if global_step >= total_steps:
                    break
        if args.max_steps <= 0 and epoch >= math.ceil(args.num_train_epochs):
            break
    progress.close()

    model.language_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved adapter to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
