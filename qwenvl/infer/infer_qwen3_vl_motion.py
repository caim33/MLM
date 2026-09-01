#!/usr/bin/env python3
"""Run strict Qwen3-VL Motion + Video inference from explicit local assets."""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
# 将项目根目录（qwen-vl-finetune，包含 qwenvl 包的目录）加入 sys.path
repo_root = project_root.parent.parent
sys.path.insert(0, str(repo_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL-Motion inference on test set")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Local base Qwen3-VL model or checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Training checkpoint directory (contains adapter weights). If None, use model_name_or_path.",
    )
    parser.add_argument(
        "--vqvae_path",
        type=str,
        default=None,
        help="VQ-VAE .pth checkpoint path (required if loading full motion model from base).",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Test JSON path (must match format expected by data_processor).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base data path for resolving relative video/motion paths in the test JSON.",
    )
    parser.add_argument(
        "--motion_mean_path",
        type=str,
        required=True,
        help="Explicit motion normalization Mean.npy file.",
    )
    parser.add_argument(
        "--motion_std_path",
        type=str,
        required=True,
        help="Explicit motion normalization Std.npy file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path for predictions. If None, only print to stdout.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size; the strict flattened path currently requires 1.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of test samples (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device; auto resolves CUDA availability at runtime.",
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("--batch_size must be 1 for identity-preserving flattened inference")
    return args


def build_data_args(
    test_data_path: str,
    data_path: str,
    motion_mean_path: str,
    motion_std_path: str,
    expected_motion_dim: int,
    motion_placeholder_token_id: int,
):
    from qwenvl.data import register_dataset
    from qwenvl.train.argument import DataArguments

    dataset_name = "infer_runtime"
    register_dataset(
        dataset_name,
        annotation_path=Path(test_data_path).resolve(strict=True),
        data_path=Path(data_path).resolve(strict=True),
        split="eval",
        replace=True,
        motion_mean_path=Path(motion_mean_path).resolve(strict=True),
        motion_std_path=Path(motion_std_path).resolve(strict=True),
        expected_motion_dim=expected_motion_dim,
    )
    data_args = DataArguments()
    data_args.dataset_use = dataset_name
    data_args.eval_dataset_use = dataset_name
    # 在推理阶段关闭内部的数据 packing，直接用单条样本，
    # 后面手动使用 FlattenedDataCollatorForSupervisedDataset 进行打包，
    # 保证每条样本都包含 motion 字段，避免 KeyError('motion')。
    data_args.data_flatten = False
    data_args.data_packing = False
    data_args.model_type = "qwen3vl"
    data_args.model_max_length = 32768
    data_args.motion_placeholder_token_id = motion_placeholder_token_id
    return data_args


def validate_motion_placeholder_binding(
    placeholder_id: object,
    *,
    tokenizer_size: int,
    embedding_size: int,
    boundary_token_ids: tuple[object, object] = (None, None),
) -> str:
    """Validate a persisted placeholder as a vocabulary token or external sentinel.

    Historical MotionLLM checkpoints deliberately use ``160001`` outside the
    Qwen vocabulary.  The data adapter inserts that sentinel directly and the
    model removes/replaces it before text embedding lookup.  A value is safe
    when it is either addressable by both tokenizer and embeddings, or outside
    both ranges; a half-in/half-out value indicates a broken artifact binding.
    """

    if isinstance(placeholder_id, bool) or not isinstance(placeholder_id, int):
        raise ValueError("motion_placeholder_token_id must be a non-negative integer")
    if placeholder_id < 0:
        raise ValueError("motion_placeholder_token_id must be a non-negative integer")
    if tokenizer_size <= 0 or embedding_size <= 0:
        raise ValueError("tokenizer and embedding sizes must be positive")
    if placeholder_id in boundary_token_ids:
        raise ValueError("motion placeholder must differ from motion boundary token ids")

    shared_size = min(tokenizer_size, embedding_size)
    outer_size = max(tokenizer_size, embedding_size)
    if placeholder_id < shared_size:
        return "vocabulary_token"
    if placeholder_id >= outer_size:
        return "external_sentinel"
    raise ValueError(
        "motion_placeholder_token_id is addressable by only one of tokenizer/model; "
        "the checkpoint binding is inconsistent"
    )


def main():
    args = parse_args()

    # Keep ``--help`` available without importing Torch/Transformers or model
    # code. Heavy dependencies are loaded only for an actual inference run.
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoConfig, AutoProcessor

    from models.qwen3_vl_motion import Qwen3VlMotionForConditionalGeneration
    from qwenvl.data.data_processor import (
        FlattenedDataCollatorForSupervisedDataset,
        make_supervised_data_module,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.checkpoint_path or args.model_name_or_path
    load_from_checkpoint = Path(checkpoint)

    if not load_from_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint or model path not found: {load_from_checkpoint}")

    model_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    if getattr(model_config, "motion_placeholder_token_id", None) is None:
        raise ValueError(
            "checkpoint config must explicitly persist motion_placeholder_token_id; "
            "legacy guessed token IDs are not accepted by clean inference"
        )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = Qwen3VlMotionForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        vqvae_path=args.vqvae_path,
        config=model_config,
        motion_config_overrides={
            "motion_normalization_mean_path": str(
                Path(args.motion_mean_path).resolve(strict=True)
            ),
            "motion_normalization_std_path": str(
                Path(args.motion_std_path).resolve(strict=True)
            ),
        },
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    placeholder_id = getattr(model.config, "motion_placeholder_token_id", None)
    if placeholder_id is None:
        raise ValueError("model config did not bind motion_placeholder_token_id")
    tokenizer_size = len(processor.tokenizer)
    embedding_size = int(model.get_input_embeddings().num_embeddings)
    validate_motion_placeholder_binding(
        placeholder_id,
        tokenizer_size=tokenizer_size,
        embedding_size=embedding_size,
        boundary_token_ids=(
            getattr(model.config, "motion_start_token_id", None),
            getattr(model.config, "motion_end_token_id", None),
        ),
    )
    expected_motion_dim = int(model.motion_spec.input_dim)

    data_args = build_data_args(
        test_data_path=args.test_data_path,
        data_path=args.data_path,
        motion_mean_path=args.motion_mean_path,
        motion_std_path=args.motion_std_path,
        expected_motion_dim=expected_motion_dim,
        motion_placeholder_token_id=int(placeholder_id),
    )
    from motionllm.training import bind_motion_length_divisor

    bind_motion_length_divisor(data_args, model)
    data_module = make_supervised_data_module(processor, data_args)
    eval_dataset = data_module["eval_dataset"]
    if eval_dataset is None:
        raise RuntimeError("eval_dataset is None. Ensure eval_dataset_use is set and data_list contains the test config.")

    # 使用支持 motion 字段的 FlattenedDataCollator，避免 KeyError('motion')
    data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    if args.num_samples is not None:
        total = min(args.num_samples, len(eval_dataset))
    else:
        total = len(eval_dataset)

    # 若指定输出路径，则以 JSONL 形式流式写入，每行一条样本，包含视频路径、目标文本和模型预测
    out_f = None
    out_file = None
    if args.output_path:
        out_file = Path(args.output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_file, "w", encoding="utf-8")

    results = []
    sample_idx = 0  # 全局样本索引，与 eval_dataset.list_data_dict 对齐

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_loader, total=total, desc="Inference")):
            if args.num_samples is not None and idx >= args.num_samples:
                break
            # 起始输入（包含文本 + <video><motion> 占位等）；position_ids 由模型内部根据 input_ids/vision 计算
            # 训练数据中 input_ids 已经包含了参考答案的 token；推理时只保留「提示部分」
            input_ids = batch["input_ids"].to(model.device)  # [1, seq]
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(model.device)
                # labels 中 != IGNORE_INDEX(-100) 的位置对应答案区域，取第一个作为生成起点
                ignore_index = -100
                first_answer_pos = (labels[0] != ignore_index).nonzero(as_tuple=False)
                if first_answer_pos.numel() > 0:
                    cut = first_answer_pos[0].item()
                    if cut > 0 and cut < input_ids.shape[1]:
                        input_ids = input_ids[:, :cut]

            motion = batch["motion"].to(model.device)
            motion_lengths = batch["motion_lengths"]
            pixel_values_videos = batch.get("pixel_values_videos")
            video_grid_thw = batch.get("video_grid_thw")
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(model.device)
            if video_grid_thw is not None:
                video_grid_thw = video_grid_thw.to(model.device)

            # 使用 model.generate()，motion / video 仅在首步传入（由 prepare_inputs_for_generation 控制），后续步用 KV cache
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)
            pad_token_id = (
                processor.tokenizer.pad_token_id
                if processor.tokenizer.pad_token_id is not None
                else processor.tokenizer.eos_token_id
            )

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                motion=motion,
                motion_lengths=motion_lengths,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

            input_len = input_ids.shape[1]
            gen_ids = generated[:, input_len:]
            pred_texts = processor.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # 对当前 batch 内的每条样本，逐条写入：视频路径、目标文本（tgt）和预测
            for text in pred_texts:
                meta = eval_dataset.list_data_dict[sample_idx]
                video_path = meta.get("video", None)
                motion_path = meta.get("motion", meta.get("motion_path", None))
                # 取 conversations 中最后一条 from == "gpt" 作为目标回答
                tgt = ""
                for conv in meta.get("conversations", []):
                    if conv.get("from") == "gpt":
                        tgt = conv.get("value", "")

                record = {
                    "index": sample_idx,
                    "video": video_path,
                    "motion": motion_path,
                    "tgt": tgt,
                    "prediction": text,
                }

                if out_f is not None:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                else:
                    results.append(record)

                sample_idx += 1

    if out_f is not None:
        out_f.close()
        print(f"Predictions saved (JSONL) to {out_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
