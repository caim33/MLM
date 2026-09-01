#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def load_hf_state(hf_dir: Path) -> dict[str, torch.Tensor]:
    index_path = hf_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = ["pytorch_model.bin"]

    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = hf_dir / shard_name
        print(f"loading {shard_path}", flush=True)
        shard = torch.load(shard_path, map_location="cpu")
        state.update(shard)
        del shard
    return state


def permute_qk(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    out_features, in_features = weight.shape
    head_dim = out_features // n_heads
    return weight.view(n_heads, 2, head_dim // 2, in_features).transpose(1, 2).reshape(out_features, in_features)


def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int) -> torch.Tensor:
    head_dim = q.shape[0] // n_heads
    in_features = q.shape[1]
    q = q.view(n_heads, head_dim, in_features)
    k = k.view(n_heads, head_dim, in_features)
    v = v.view(n_heads, head_dim, in_features)
    return torch.stack((q, k, v), dim=1).reshape(n_heads * 3 * head_dim, in_features)


def convert_state(hf: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    n_layers = int(cfg.get("num_hidden_layers", 32))
    n_heads = int(cfg.get("num_attention_heads", 32))
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    out: dict[str, torch.Tensor] = {}
    out["transformer.wte.weight"] = hf["model.embed_tokens.weight"]
    out["lm_head.linear.weight"] = hf["lm_head.weight"]
    out["transformer.ln_f.weight"] = hf["model.norm.weight"]

    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        target = f"transformer.h.{i}"
        out[f"{target}.norm_1.weight"] = hf[f"{prefix}.input_layernorm.weight"]
        out[f"{target}.norm_2.weight"] = hf[f"{prefix}.post_attention_layernorm.weight"]
        if n_heads != n_kv_heads:
            raise ValueError("This converter currently expects MHA with n_heads == n_kv_heads")
        out[f"{target}.attn.attn.linear.weight"] = fuse_qkv(
            permute_qk(hf[f"{prefix}.self_attn.q_proj.weight"], n_heads),
            permute_qk(hf[f"{prefix}.self_attn.k_proj.weight"], n_kv_heads),
            hf[f"{prefix}.self_attn.v_proj.weight"],
            n_heads,
        )
        out[f"{target}.attn.proj.linear.weight"] = hf[f"{prefix}.self_attn.o_proj.weight"]
        out[f"{target}.mlp.fc_1.linear.weight"] = hf[f"{prefix}.mlp.gate_proj.weight"]
        out[f"{target}.mlp.fc_2.linear.weight"] = hf[f"{prefix}.mlp.up_proj.weight"]
        out[f"{target}.mlp.proj.linear.weight"] = hf[f"{prefix}.mlp.down_proj.weight"]
    return out


def write_lit_config(hf_dir: Path, out_dir: Path) -> None:
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    lit_cfg = {
        "name": "vicuna-7b-v1.5",
        "hf_config": cfg,
    }
    (out_dir / "lit_config.json").write_text(json.dumps(lit_cfg, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hf_config = json.loads((args.hf_dir / "config.json").read_text(encoding="utf-8"))
    hf = load_hf_state(args.hf_dir)
    lit = convert_state(hf, hf_config)
    del hf
    out_model = args.out_dir / "lit_model.pth"
    print(f"saving {out_model}", flush=True)
    torch.save(lit, out_model)

    for name in [
        "config.json",
        "generation_config.json",
        "pytorch_model.bin.index.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = args.hf_dir / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    for shard in sorted(args.hf_dir.glob("pytorch_model-*.bin")):
        dst = args.out_dir / shard.name
        if not dst.exists():
            try:
                dst.symlink_to(shard)
            except OSError:
                shutil.copy2(shard, dst)
    write_lit_config(args.hf_dir, args.out_dir)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
