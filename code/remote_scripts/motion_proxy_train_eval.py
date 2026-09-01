#!/usr/bin/env python3
"""Train and evaluate lightweight MotionX QA motion proxy models.

The script consumes the Motion-R1 jsonl format used in qwen-vl-finetune:
- train/val raw files contain absolute motion paths for branch=vm rows.
- benchmark QA_500.jsonl contains paths relative to the qwen-vl-finetune root.

It intentionally writes self-contained checkpoints and summaries so it can be
run independently for several proxy model variants in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.I)
OPTION_RE = re.compile(r"^\s*([A-D])\.\s*(.+?)\s*$", re.M)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
LABEL_TO_ID = {c: i for i, c in enumerate("ABCD")}
ID_TO_LABEL = {i: c for c, i in LABEL_TO_ID.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def answer_to_label(text: str) -> int:
    m = ANSWER_RE.search(text or "")
    if not m:
        raise ValueError(f"cannot parse answer: {text!r}")
    return LABEL_TO_ID[m.group(1).upper()]


def extract_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return row["prompt"]
    parts: List[str] = []
    for msg in row.get("messages", []):
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
    return "\n".join(parts)


def strip_motion_tokens(text: str) -> str:
    return (
        text.replace("<motion_start><motion><motion_end>\n", "")
        .replace("<motion_start><motion><motion_end>", "")
    )


def extract_options(prompt: str) -> Dict[str, str]:
    return {m.group(1).upper(): m.group(2).strip() for m in OPTION_RE.finditer(prompt)}


def question_without_options(prompt: str) -> str:
    prompt = strip_motion_tokens(prompt)
    if "Question:" in prompt:
        prompt = prompt.split("Question:", 1)[1]
    if "Choose exactly one option:" in prompt:
        prompt = prompt.split("Choose exactly one option:", 1)[0]
    return prompt.strip()


def hash_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def resolve_motion_path(row: Dict[str, Any], qwen_root: str) -> Optional[str]:
    path = row.get("motion") or row.get("motion_path") or row.get("source_motion")
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(qwen_root, path)


def iter_records(path: str) -> Iterable[Dict[str, Any]]:
    """Read either JSONL or a JSON list/dict container.

    The goal benchmark is named QA_500.json on the remote side, while older
    local mirrors use QA_500.jsonl. Supporting both keeps proxy runs aligned
    with the user's requested test file instead of forcing a rename.
    """
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "samples", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


@dataclass
class Sample:
    sample_id: str
    group_id: str
    motion_path: str
    prompt: str
    question: str
    options: Dict[str, str]
    label: int


def load_samples(path: str, qwen_root: str, branch: str = "vm") -> List[Sample]:
    rows: List[Sample] = []
    for idx, row in enumerate(iter_records(path)):
        if branch != "all" and row.get("branch") != branch:
            continue
        motion_path = resolve_motion_path(row, qwen_root)
        if not motion_path or not os.path.exists(motion_path):
            continue
        prompt = extract_prompt(row)
        options = extract_options(prompt)
        if len(options) != 4:
            continue
        label = answer_to_label(row.get("answer") or row.get("solution") or "")
        rows.append(
            Sample(
                sample_id=row.get("sample_id") or row.get("benchmark_id") or str(idx),
                group_id=row.get("group_id") or "",
                motion_path=motion_path,
                prompt=prompt,
                question=question_without_options(prompt),
                options=options,
                label=label,
            )
        )
    return rows


class MotionCache:
    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self.cache: Dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray:
        if path not in self.cache:
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                arr = arr.reshape(arr.shape[0], -1).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.shape[0] > self.max_frames:
                idx = np.linspace(0, arr.shape[0] - 1, self.max_frames).astype(np.int64)
                arr = arr[idx]
            elif arr.shape[0] < self.max_frames:
                pad = np.zeros((self.max_frames - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            self.cache[path] = arr
        return self.cache[path]


def pooled_motion_features(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    amin = arr.min(axis=0)
    amax = arr.max(axis=0)
    delta = arr[-1] - arr[0]
    return np.concatenate([mean, std, amin, amax, delta]).astype(np.float32)


class MotionQADataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        cache: MotionCache,
        *,
        model_kind: str,
        text_dim: int,
    ) -> None:
        self.samples = samples
        self.cache = cache
        self.model_kind = model_kind
        self.text_dim = text_dim
        self.records: List[Dict[str, Any]] = []
        for sample in samples:
            motion = cache.get(sample.motion_path)
            self.records.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "motion": torch.from_numpy(motion.copy()),
                    "pooled": torch.from_numpy(pooled_motion_features(motion)),
                    "prompt_vec": torch.from_numpy(hash_text(strip_motion_tokens(sample.prompt), text_dim)),
                    "question_vec": torch.from_numpy(hash_text(sample.question, text_dim)),
                    "option_vecs": torch.from_numpy(
                        np.stack([hash_text(sample.options[c], text_dim) for c in "ABCD"], axis=0).astype(np.float32)
                    ),
                    "label": torch.tensor(sample.label, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "group_id": [x["group_id"] for x in batch],
        "motion": torch.stack([x["motion"] for x in batch]),
        "pooled": torch.stack([x["pooled"] for x in batch]),
        "prompt_vec": torch.stack([x["prompt_vec"] for x in batch]),
        "question_vec": torch.stack([x["question_vec"] for x in batch]),
        "option_vecs": torch.stack([x["option_vecs"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
    }


class AgcnProxyMLP(nn.Module):
    def __init__(self, pooled_dim: int, text_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(pooled_dim + text_dim),
            nn.Linear(pooled_dim + text_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["pooled"], batch["prompt_vec"]], dim=-1)
        return self.net(x)


class AgcnProxyRNN(nn.Module):
    def __init__(self, motion_dim: int, text_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(motion_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + text_dim),
            nn.Linear(hidden + text_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.in_proj(batch["motion"])
        _, h = self.rnn(x)
        feat = h[-1]
        return self.head(torch.cat([feat, batch["prompt_vec"]], dim=-1))


class OptionScoringMLP(nn.Module):
    def __init__(self, pooled_dim: int, text_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.motion = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.question = nn.Linear(text_dim, hidden)
        self.option = nn.Linear(text_dim, hidden)
        self.bias = nn.Linear(hidden, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        motion = self.motion(batch["pooled"]).unsqueeze(1)
        question = self.question(batch["question_vec"]).unsqueeze(1)
        options = self.option(batch["option_vecs"])
        fused = torch.tanh(motion + question + options)
        return self.bias(fused).squeeze(-1)


class OptionScoringRNN(nn.Module):
    def __init__(self, motion_dim: int, text_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(motion_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.question = nn.Linear(text_dim, hidden)
        self.option = nn.Linear(text_dim, hidden)
        self.bias = nn.Linear(hidden, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.in_proj(batch["motion"])
        _, h = self.rnn(x)
        motion = h[-1].unsqueeze(1)
        question = self.question(batch["question_vec"]).unsqueeze(1)
        options = self.option(batch["option_vecs"])
        fused = torch.tanh(motion + question + options)
        return self.bias(fused).squeeze(-1)


def make_model(name: str, motion_dim: int, pooled_dim: int, text_dim: int, hidden: int, dropout: float) -> nn.Module:
    if name == "agcn_mlp":
        return AgcnProxyMLP(pooled_dim, text_dim, hidden, dropout)
    if name == "agcn_rnn":
        return AgcnProxyRNN(motion_dim, text_dim, hidden, dropout)
    if name == "motionclip_mlp":
        return OptionScoringMLP(pooled_dim, text_dim, hidden, dropout)
    if name == "motionclip_rnn":
        return OptionScoringRNN(motion_dim, text_dim, hidden, dropout)
    raise ValueError(f"unknown model: {name}")


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ["motion", "pooled", "prompt_vec", "question_vec", "option_vecs", "label"]:
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def make_class_weights(samples: List[Sample], device: torch.device, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.zeros(4, dtype=np.float32)
    for sample in samples:
        counts[sample.label] += 1
    if np.any(counts == 0):
        return None
    if mode == "balanced":
        weights = counts.sum() / (4.0 * counts)
    elif mode == "sqrt_balanced":
        weights = np.sqrt(counts.sum() / (4.0 * counts))
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def materialize_dataset(ds: MotionQADataset, device: torch.device) -> Dict[str, Any]:
    records = ds.records
    tensor_keys = ["motion", "pooled", "prompt_vec", "question_vec", "option_vecs", "label"]
    data: Dict[str, Any] = {
        "sample_id": [r["sample_id"] for r in records],
        "group_id": [r["group_id"] for r in records],
    }
    for key in tensor_keys:
        data[key] = torch.stack([r[key] for r in records]).to(device)
    return data


def tensor_batch(data: Dict[str, Any], idx: torch.Tensor | slice) -> Dict[str, Any]:
    return {
        "sample_id": data["sample_id"] if isinstance(idx, slice) else None,
        "group_id": data["group_id"] if isinstance(idx, slice) else None,
        "motion": data["motion"][idx],
        "pooled": data["pooled"][idx],
        "prompt_vec": data["prompt_vec"][idx],
        "question_vec": data["question_vec"][idx],
        "option_vecs": data["option_vecs"][idx],
        "label": data["label"][idx],
    }


def iter_tensor_batches(
    data: Dict[str, Any],
    batch_size: int,
    *,
    shuffle: bool,
) -> Iterable[Tuple[Dict[str, Any], Optional[torch.Tensor], int, int]]:
    n = int(data["label"].shape[0])
    if shuffle:
        order = torch.randperm(n, device=data["label"].device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield tensor_batch(data, idx), idx, start, min(start + batch_size, n)
    else:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield tensor_batch(data, slice(start, end)), None, start, end


@torch.no_grad()
def evaluate_tensor(
    model: nn.Module,
    data: Dict[str, Any],
    batch_size: int,
    *,
    details_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    pred_counts = {c: 0 for c in "ABCD"}
    gt_counts = {c: 0 for c in "ABCD"}
    details: List[Dict[str, Any]] = []
    for batch, _, start, end in iter_tensor_batches(data, batch_size, shuffle=False):
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["label"], reduction="sum")
        preds = logits.argmax(dim=-1)
        total += int(batch["label"].numel())
        correct += int((preds == batch["label"]).sum().item())
        loss_sum += float(loss.item())
        probs = logits.softmax(dim=-1).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().tolist()
        pred_ids = preds.detach().cpu().tolist()
        for offset, (gt, pred) in enumerate(zip(labels, pred_ids)):
            gt_letter = ID_TO_LABEL[int(gt)]
            pred_letter = ID_TO_LABEL[int(pred)]
            gt_counts[gt_letter] += 1
            pred_counts[pred_letter] += 1
            if details_path is not None:
                i = start + offset
                details.append(
                    {
                        "sample_id": data["sample_id"][i],
                        "group_id": data["group_id"][i],
                        "answer": gt_letter,
                        "pred_answer": pred_letter,
                        "correct": pred_letter == gt_letter,
                        "probs": {ID_TO_LABEL[j]: float(probs[offset, j]) for j in range(4)},
                    }
                )
    if details_path is not None:
        with open(details_path, "w", encoding="utf-8") as f:
            for item in details:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else math.nan,
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    details_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    pred_counts = {c: 0 for c in "ABCD"}
    gt_counts = {c: 0 for c in "ABCD"}
    details: List[Dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["label"], reduction="sum")
        preds = logits.argmax(dim=-1)
        total += int(batch["label"].numel())
        correct += int((preds == batch["label"]).sum().item())
        loss_sum += float(loss.item())
        probs = logits.softmax(dim=-1).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().tolist()
        pred_ids = preds.detach().cpu().tolist()
        for i, (gt, pred) in enumerate(zip(labels, pred_ids)):
            gt_letter = ID_TO_LABEL[int(gt)]
            pred_letter = ID_TO_LABEL[int(pred)]
            gt_counts[gt_letter] += 1
            pred_counts[pred_letter] += 1
            if details_path is not None:
                details.append(
                    {
                        "sample_id": raw_batch["sample_id"][i],
                        "group_id": raw_batch["group_id"][i],
                        "answer": gt_letter,
                        "pred_answer": pred_letter,
                        "correct": pred_letter == gt_letter,
                        "probs": {ID_TO_LABEL[j]: float(probs[i, j]) for j in range(4)},
                    }
                )
    if details_path is not None:
        with open(details_path, "w", encoding="utf-8") as f:
            for item in details:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else math.nan,
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["agcn_mlp", "agcn_rnn", "motionclip_mlp", "motionclip_rnn"])
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--qwen-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--class-weights", default="sqrt_balanced", choices=["none", "balanced", "sqrt_balanced"])
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(args.train, args.qwen_root, branch="vm")
    val_samples = load_samples(args.val, args.qwen_root, branch="vm")
    test_samples = load_samples(args.test, args.qwen_root, branch="vm")
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(
            f"empty split: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
        )

    cache = MotionCache(args.max_frames)
    # Prime one sample to infer dimensions.
    first = cache.get(train_samples[0].motion_path)
    motion_dim = first.shape[1]
    pooled_dim = pooled_motion_features(first).shape[0]

    train_ds = MotionQADataset(train_samples, cache, model_kind=args.model, text_dim=args.text_dim)
    val_ds = MotionQADataset(val_samples, cache, model_kind=args.model, text_dim=args.text_dim)
    test_ds = MotionQADataset(test_samples, cache, model_kind=args.model, text_dim=args.text_dim)
    train_data = materialize_dataset(train_ds, device)
    val_data = materialize_dataset(val_ds, device)
    test_data = materialize_dataset(test_ds, device)

    model = make_model(args.model, motion_dim, pooled_dim, args.text_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    class_weights = make_class_weights(train_samples, device, args.class_weights)

    best_val = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    ckpt_path = out_dir / "best_model.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0
        train_correct = 0
        train_loss = 0.0
        for batch, _, _, _ in iter_tensor_batches(train_data, args.batch_size, shuffle=True):
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["label"], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * int(batch["label"].numel())
            train_total += int(batch["label"].numel())
            train_correct += int((logits.argmax(dim=-1) == batch["label"]).sum().item())
        scheduler.step()
        val_metrics = evaluate_tensor(model, val_data, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "motion_dim": motion_dim,
                    "pooled_dim": pooled_dim,
                    "text_dim": args.text_dim,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val,
                },
                ckpt_path,
            )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_final = evaluate_tensor(model, val_data, args.batch_size, details_path=out_dir / "val_details.jsonl")
    test_final = evaluate_tensor(model, test_data, args.batch_size, details_path=out_dir / "test_details.jsonl")

    summary = {
        "model": args.model,
        "train_path": args.train,
        "val_path": args.val,
        "test_path": args.test,
        "qwen_root": args.qwen_root,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "val": val_final,
        "test": test_final,
        "checkpoint": str(ckpt_path),
        "elapsed_seconds": time.time() - start,
        "history": history,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
