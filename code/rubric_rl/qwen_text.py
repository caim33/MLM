"""Small text-only generation wrapper for local Qwen/Qwen-VL checkpoints."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional


def extract_json_text(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
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
                return text[start : pos + 1]
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    json_text = extract_json_text(text)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_list(text: str) -> Optional[list[Any]]:
    start = text.find("[")
    if start < 0:
        return None
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : pos + 1])
                except Exception:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


class QwenTextGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        max_memory: Optional[Dict[Any, str]] = None,
        model_class: str = "image_text",
        allow_attention_fallback: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        dtype_obj = getattr(torch, dtype) if dtype != "auto" else "auto"
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if dtype_obj != "auto":
            load_kwargs["dtype"] = dtype_obj
        else:
            load_kwargs["torch_dtype"] = "auto"
        if max_memory:
            load_kwargs["max_memory"] = max_memory

        if model_class == "causal_lm":
            processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if revision is not None:
                processor_kwargs["revision"] = revision
            self.processor = AutoTokenizer.from_pretrained(model_path, **processor_kwargs)
            model_cls = AutoModelForCausalLM
        elif model_class == "image_text":
            processor_kwargs = {"trust_remote_code": True}
            if revision is not None:
                processor_kwargs["revision"] = revision
            self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
            model_cls = AutoModelForImageTextToText
        else:
            raise ValueError(f"Unsupported model_class={model_class}")

        try:
            self.model = model_cls.from_pretrained(model_path, **load_kwargs)
        except Exception as exc:
            if (
                not allow_attention_fallback
                or attn_implementation in {"sdpa", "eager", ""}
            ):
                raise
            print(
                f"attention_backend_fallback from {attn_implementation} to sdpa: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            load_kwargs["attn_implementation"] = "sdpa"
            self.model = model_cls.from_pretrained(model_path, **load_kwargs)
            self.effective_attn_implementation = "sdpa"
        else:
            self.effective_attn_implementation = attn_implementation
        self.model.eval()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 1600,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prefill: str = "",
        stop_after_json_list: bool = False,
    ) -> str:
        return self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            prefill=prefill,
            stop_after_json_list=stop_after_json_list,
        )[0]

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        max_new_tokens: int = 1600,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prefill: str = "",
        stop_after_json_list: bool = False,
    ) -> list[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        texts = []
        for messages in batch_messages:
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            texts.append(text + prefill)
        inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        class JsonListStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer: Any, prompt_length: int, prefix: str) -> None:
                self.tokenizer = tokenizer
                self.prompt_length = prompt_length
                self.prefix = prefix
                self.calls = 0

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                self.calls += 1
                if self.calls % 8 != 0:
                    return False
                generated_ids = input_ids[:, self.prompt_length :]
                texts = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return all(extract_json_list((self.prefix + text).strip()) is not None for text in texts)

        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            }
            if min_new_tokens > 0:
                generate_kwargs["min_new_tokens"] = min_new_tokens
            if do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            if stop_after_json_list:
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [JsonListStoppingCriteria(self.tokenizer, prompt_len, prefill)]
                )
            generated = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        decoded = self.tokenizer.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if prefill:
            return [(prefill + item).strip() for item in decoded]
        return [item.strip() for item in decoded]
