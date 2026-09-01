#!/usr/bin/env bash
# Qwen3-VL-Motion 推理脚本：使用训练权重在 motionx_overall_action_conversations_v0_1_test 上测试
# 用法: 在 qwen-vl-finetune 目录下执行: bash run_infer_qwen3_vl_motion.sh

set -e

# 训练好的权重目录（例如训练脚本的 output_dir，内含 config.json、pytorch_model.bin 等）
CHECKPOINT_PATH="/wangbenyou-dengyizhe/Qwen3_vl_motion/qwen-vl-finetune/4B_thinking_stage1_motionX_v1_0_uncode_2560_prenorm"

# 若从基座+VQ-VAE 加载（未用训练 checkpoint），则设置 VQVAE_PATH；用训练 checkpoint 时留空
VQVAE_PATH=""

# 测试集路径（默认使用 data_list 中的 motionX_v0_1_test）
TEST_JSON="/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_0_test_sample100_video_only.json"
DATA_PATH="/wangbenyou-dengyizhe/Data"

# 输出预测结果 JSON
OUTPUT_PATH="/wangbenyou-dengyizhe/Qwen3_vl_motion/qwen-vl-finetune/qwenvl/infer/4B_thinking_stage1_motionX_v1_0_uncode_2560_prenorm_preds_video_only.json"

# 可选：限制测试样本数（默认跑全量）
NUM_SAMPLES=100
MAX_NEW_TOKENS=2048

cd "$(dirname "$0")"

VQVAE_ARG=""
if [[ -n "$VQVAE_PATH" ]]; then
  VQVAE_ARG="--vqvae_path $VQVAE_PATH"
fi

NUM_SAMPLES_ARG=""
if [[ -n "$NUM_SAMPLES" ]]; then
  NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

python infer_qwen3_vl_motion.py \
  --model_name_or_path "$CHECKPOINT_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  $VQVAE_ARG \
  --test_data_path "$TEST_JSON" \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  $NUM_SAMPLES_ARG \
  --batch_size 1

echo "Done. Predictions saved to: $OUTPUT_PATH"
