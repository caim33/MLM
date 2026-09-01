#!/usr/bin/env python3
"""Small launcher for qwen-vl-finetune LoRA SFT with Codex-created datasets."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune")
SFT_DIR = Path(os.environ.get("CODEX_SFT_DIR", "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/sft_data_20260716"))

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "qwenvl" / "train"))


def _register_datasets() -> None:
    import qwenvl.data as data_registry

    entries = {
        "codex_motionx_qa_train_v": SFT_DIR / "motionx_qa_train_v.json",
        "codex_motionx_qa_train_vm": SFT_DIR / "motionx_qa_train_vm.json",
        "codex_motionx_qa_train_v_smoke32": SFT_DIR / "motionx_qa_train_v_smoke32.json",
        "codex_motionx_qa_train_vm_smoke32": SFT_DIR / "motionx_qa_train_vm_smoke32.json",
        "codex_motionx_qa_val_v": SFT_DIR / "motionx_qa_val_v.json",
        "codex_motionx_qa_val_vm": SFT_DIR / "motionx_qa_val_vm.json",
    }
    for name, path in entries.items():
        data_registry.data_dict[name] = {"annotation_path": str(path), "data_path": ""}


def _add_motion_tokens(tokenizer: Any) -> Any:
    if tokenizer is None:
        return tokenizer
    tokens = ["<motion_start>", "<motion_end>"]
    missing = [tok for tok in tokens if tokenizer.convert_tokens_to_ids(tok) is None]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return tokenizer


def _patch_auto_loaders() -> None:
    import transformers

    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration") and hasattr(
        transformers, "Qwen3VLForConditionalGeneration"
    ):
        transformers.Qwen3_5ForConditionalGeneration = transformers.Qwen3VLForConditionalGeneration
    if not hasattr(transformers, "Qwen3_5MoeForConditionalGeneration") and hasattr(
        transformers, "Qwen3VLMoeForConditionalGeneration"
    ):
        transformers.Qwen3_5MoeForConditionalGeneration = transformers.Qwen3VLMoeForConditionalGeneration

    orig_processor_from_pretrained = transformers.AutoProcessor.from_pretrained
    orig_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def processor_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        processor = orig_processor_from_pretrained(*args, **kwargs)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = _add_motion_tokens(processor.tokenizer)
        return processor

    def tokenizer_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        tokenizer = orig_tokenizer_from_pretrained(*args, **kwargs)
        return _add_motion_tokens(tokenizer)

    transformers.AutoProcessor.from_pretrained = processor_from_pretrained
    transformers.AutoTokenizer.from_pretrained = tokenizer_from_pretrained


def _module(model: Any, *paths: str) -> Optional[Any]:
    for path in paths:
        cur = model
        ok = True
        for name in path.split("."):
            cur = getattr(cur, name, None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    return None


def _set_requires_grad(module: Any, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled


def _patch_lora_sft_module(lora_sft: Any) -> None:
    def set_model(model_args: Any, model: Any) -> None:
        visual = _module(model, "visual", "model.visual")
        merger = _module(model, "visual.merger", "model.visual.merger")
        language_model = _module(model, "language_model", "model.language_model")
        lm_head = _module(model, "lm_head", "model.lm_head")
        motion_encoder = _module(model, "motion_encoder", "model.motion_encoder")

        _set_requires_grad(visual, bool(model_args.tune_mm_vision))
        _set_requires_grad(merger, bool(model_args.tune_mm_mlp))
        _set_requires_grad(language_model, bool(model_args.tune_mm_llm))
        _set_requires_grad(lm_head, bool(model_args.tune_mm_llm))
        _set_requires_grad(motion_encoder, bool(getattr(model_args, "tune_mm_motion", False)))

    lora_sft.set_model = set_model


def _load_lora_sft_module() -> Any:
    module_name = "qwenvl.train.lora_sft"
    source_path = PROJECT_ROOT / "qwenvl" / "train" / "lora_sft.py"
    source = source_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    source = source.replace("    Qwen3_5ForConditionalGeneration,\n", "")
    source = source.replace("    Qwen3_5MoeForConditionalGeneration,\n", "")
    alias_block = (
        "Qwen3_5ForConditionalGeneration = Qwen3VLForConditionalGeneration\n"
        "Qwen3_5MoeForConditionalGeneration = Qwen3VLMoeForConditionalGeneration\n\n"
    )
    source = source.replace("project_root = Path(__file__).parent.parent.parent", alias_block + "project_root = Path(__file__).parent.parent.parent", 1)

    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "qwenvl.train"
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _register_datasets()
    _patch_auto_loaders()
    lora_sft = _load_lora_sft_module()

    _patch_lora_sft_module(lora_sft)
    lora_sft.train(attn_implementation=os.environ.get("CODEX_ATTN_IMPLEMENTATION", "sdpa"))


if __name__ == "__main__":
    main()
