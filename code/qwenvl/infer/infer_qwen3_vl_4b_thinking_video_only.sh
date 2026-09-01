python /wangbenyou-dengyizhe/Qwen3_vl_motion/qwen-vl-finetune/qwenvl/infer/infer_qwen3_vl_4b_thinking_video_only.py \
  --model_path /wangbenyou-dengyizhe/Qwen3_vl_motion/model/Qwen3-VL-4B-Thinking \
  --input_jsonl /wangbenyou-dengyizhe/Qwen3_vl_motion/qwen-vl-finetune/inference_output/4B_thinking_stage1_motionX_v1_0_uncode_2560_prenorm_preds_video_only.json \
  --data_root /wangbenyou-dengyizhe/Data \
  --output_path /wangbenyou-dengyizhe/Qwen3_vl_motion/qwen-vl-finetune/inference_output/4B_thinking_stage2_motionX_v1_0_video_only.jsonl \
  --max_new_tokens 2560 \
  --num_samples 100