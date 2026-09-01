#!/usr/bin/env python3
"""
使用 Qwen3-VL-4B-Thinking 模型，对 video-only JSONL 预测文件
`4B_thinking_stage1_motionX_v1_0_uncode_2560_prenorm_preds_video_only.json`
中的视频进行再次推理，生成新的文本描述。

输入 JSONL 每行示例（与 stage1 输出一致）：
{
  "index": 1,
  "video": "Videos/Motion-X/batch_0002/11192_27793.mp4",
  "motion": null,
  "tgt": "... 人工标注文本 ...",
  "prediction": "... stage1 模型输出（含 <think> ...） ..."
}

本脚本：
- 逐行读取上述 JSONL；
- 根据 `video` 字段加载视频；
- 使用 Qwen3-VL-4B-Thinking 进行视频理解；
- 生成 2~3 句 overall action overview 文本；
- 将结果以 JSONL 写出，每行包含原始字段以及新的 `prediction_v2`。
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable

from motionllm.data import read_jsonl, resolve_media_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-4B-Thinking inference on video-only JSONL data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Qwen3-VL-4B-Thinking 本地模型目录。",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="stage1 推理得到的 JSONL 文件路径。",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="视频数据根目录，`video` 字段为相对此目录的路径。",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 JSONL 文件路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备；auto 在运行阶段根据 CUDA 可用性选择。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成的最大新 token 数。",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="仅对前 N 条样本进行推理（默认全部）。",
    )
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read strict JSONL without silently dropping malformed rows."""

    yield from read_jsonl(path)


def build_chat_for_video(video_tensor, video_path: str) -> Dict[str, Any]:
    """
    构造 Qwen3-VL 风格的多模态对话消息。
    `video_tensor` 仅用于占位，真正的张量由 processor 处理。
    """
    user_text = (
        "You are given a short human action video. "
        "Please write an overall action overview in 2-3 sentences, "
        "describing only the observable body movements and interactions, "
        "without mentioning intentions, emotions or unobservable goals."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_tensor,  # 真实内容由 processor(videos=...) 传入
                    "path": video_path,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return messages


def main():
    args = parse_args()

    # Heavy dependencies are loaded only after argument parsing so ``--help``
    # remains available in a CPU-only base environment.
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    data_root = Path(args.data_root)

    # 预先计算总行数用于进度条（可能较大，按需可去掉）
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = None

    processed = 0

    with torch.no_grad():
        for record in tqdm(
            iter_jsonl(str(input_path)),
            total=total_lines,
            desc="Inference (Qwen3-VL-4B-Thinking)",
        ):
            if args.num_samples is not None and processed >= args.num_samples:
                break

            video_rel = record.get("video")
            if not isinstance(video_rel, str) or not video_rel:
                raise ValueError(
                    f"input row {processed + 1} must contain a non-empty video path"
                )

            video_path = resolve_media_path(data_root, video_rel)

            # Qwen3-VL 的 processor 支持直接传入视频路径列表
            # 这里 messages 主要用于生成 text prompt
            dummy_video = None
            messages = build_chat_for_video(dummy_video, str(video_path))
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[chat_text],
                videos=[[str(video_path)]],
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = generated_ids[:, input_len:]
            pred_texts = processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            pred = pred_texts[0] if pred_texts else ""

            out_record = dict(record)
            out_record["prediction_v2"] = pred
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1

    out_f.close()
    print(f"Inference finished. Saved to: {out_path}")


if __name__ == "__main__":
    main()
