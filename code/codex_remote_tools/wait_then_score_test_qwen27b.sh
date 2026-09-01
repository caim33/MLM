#!/usr/bin/env bash
set -euo pipefail

ROOT="/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM"
PY="$ROOT/codex_envs/mllm/bin/python"
MODEL="$ROOT/codex_models/Qwen__Qwen3.6-27B"
DATA="$ROOT/data/rubric_rl"
CRITERIA="$DATA/sample_summary_qwen36_27b_v2_fast_merged_criteria.jsonl"
CANDIDATES="$DATA/sample_summary_candidates_v2_score_test.jsonl"
OUTPUT="$DATA/sample_summary_qwen36_27b_v2_score_test_rewarded_after_training.jsonl"

cd "$ROOT"
echo "watch_started_at=$(date -Is)"
echo "criteria=$CRITERIA"
echo "candidates=$CANDIDATES"
echo "output=$OUTPUT"

stable_free_count=0
selected_gpus=""
while true; do
  selected_gpus="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 < 5000) print $1}' \
      | head -n 2 \
      | paste -sd, -
  )"
  if [[ "$selected_gpus" == *,* ]]; then
    stable_free_count=$((stable_free_count + 1))
    echo "free_check $(date -Is) selected_gpus=$selected_gpus stable=$stable_free_count"
  else
    stable_free_count=0
    echo "waiting $(date -Is) selected_gpus=${selected_gpus:-none}"
  fi

  if [[ "$stable_free_count" -ge 3 ]]; then
    break
  fi
  sleep 60
done

echo "judge_started_at=$(date -Is) cuda_visible_devices=$selected_gpus"
"$PY" -m rubric_rl.judge_motion_caption_v2 \
  --model "$MODEL" \
  --criteria "$CRITERIA" \
  --candidates "$CANDIDATES" \
  --output "$OUTPUT" \
  --candidate-key candidate \
  --limit 4 \
  --cuda-visible-devices "$selected_gpus" \
  --max-memory "0:38GiB,1:38GiB,cpu:120GiB" \
  --max-new-tokens 2600 \
  --keep-raw
echo "judge_finished_at=$(date -Is)"
