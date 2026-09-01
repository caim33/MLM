#!/usr/bin/env bash
set -u

ROOT=/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM
RUNS="$ROOT/codex_runs"
MODEL_ROOT="$ROOT/MVBench_Eval/models"

echo "## HOST"
hostname
date -Is
df -h "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "## REQUIRED_PATHS"
for path in \
  "$ROOT/codex_models/Qwen__Qwen3.6-27B" \
  "$ROOT/codex_models/Qwen__Qwen3.5-4B" \
  "$ROOT/codex_models/Qwen__Qwen3-VL-8B-Instruct" \
  "$ROOT/codex_models/Qwen__Qwen3-VL-4B-Instruct" \
  "$ROOT/codex_models/qwen3_vl_motion_checkpoint_0426" \
  /wangbenyou-sulongjie/Motion-r1/model/pretrained/VQVAE/net_best_fid.pth \
  "$MODEL_ROOT/LLaVA-7B-Lightening-v1-1" \
  "$MODEL_ROOT/Video-ChatGPT/video_chatgpt-7B.bin" \
  "$MODEL_ROOT/VideoChat2/umt_l16_qformer.pth" \
  "$MODEL_ROOT/VideoChat2/videochat2_7b_stage3.pth" \
  "$MODEL_ROOT/vicuna-7b-v1.5" \
  "$MODEL_ROOT/mplug-owl-llama-7b-video" \
  "$MODEL_ROOT/OTTER-Video-LLaMA7B-DenseCaption" \
  "$ROOT/MVBench_Eval/scripts/video_llama_motionx_eval_only_vl.yaml"
do
  if [ -e "$path" ] || [ -L "$path" ]; then
    target=$(readlink -f "$path" 2>/dev/null || true)
    bytes=$(timeout 15s du -sb "$path" 2>/dev/null | awk '{print $1}')
    bytes=${bytes:-TIMEOUT}
    printf 'PRESENT\t%s\t%s\t%s\n' "$path" "$bytes" "$target"
  else
    printf 'MISSING\t%s\n' "$path"
  fi
done

echo "## SOURCE_REVISIONS"
for repo in \
  "$RUNS/video_model_sources/Video-LLaVA" \
  "$RUNS/video_model_sources/Video-ChatGPT" \
  "$RUNS/video_model_sources/Ask-Anything" \
  "$RUNS/video_model_sources/video-llama" \
  "$RUNS/video_model_sources/mPLUG-Owl" \
  "$RUNS/video_model_sources/Otter" \
  "$ROOT/2s-AGCN" \
  "$ROOT/MotionCLIP"
do
  if [ -d "$repo/.git" ]; then
    rev=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)
    remote=$(git -C "$repo" remote get-url origin 2>/dev/null || true)
    dirty=$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
    files=$(find "$repo" -type f -not -path '*/.git/*' 2>/dev/null | wc -l)
    printf 'GIT\t%s\t%s\t%s\tdirty=%s\tfiles=%s\n' "$repo" "$rev" "$remote" "$dirty" "$files"
  elif [ -d "$repo" ]; then
    files=$(find "$repo" -type f 2>/dev/null | wc -l)
    printf 'DIR\t%s\tfiles=%s\n' "$repo" "$files"
  else
    printf 'MISSING\t%s\n' "$repo"
  fi
done

echo "## ENVIRONMENTS"
for py in \
  "$ROOT/codex_envs/mllm/bin/python" \
  "$ROOT/codex_envs/legacy_torch211_cu128/bin/python" \
  "$ROOT/codex_envs/legacy_tf431/bin/python" \
  /wangbenyou-sulongjie/anaconda3/envs/qwen3_vl/bin/python3.10 \
  python3
do
  if command -v "$py" >/dev/null 2>&1 || [ -x "$py" ]; then
    printf 'PYTHON\t%s\t' "$py"
    timeout 10s "$py" -c 'import sys; print(sys.version.split()[0])' 2>&1 || true
    timeout 20s "$py" - <<'PY' 2>/dev/null || true
mods = ("torch", "transformers", "peft", "accelerate", "decord")
for name in mods:
    try:
        mod = __import__(name)
        print(f"PKG\t{name}\t{getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"PKG_MISSING\t{name}\t{type(exc).__name__}")
PY
  else
    printf 'MISSING_PYTHON\t%s\n' "$py"
  fi
done

echo "## VIDEO_LLAMA_CONFIG"
cfg="$ROOT/MVBench_Eval/scripts/video_llama_motionx_eval_only_vl.yaml"
if [ -f "$cfg" ]; then
  grep -nEi 'llama_model|ckpt|checkpoint|pretrain|qformer|vit|bert|imagebind|model_path' "$cfg" || true
fi

echo "## MODEL_FILE_SUMMARIES"
for path in \
  "$MODEL_ROOT/LLaVA-7B-Lightening-v1-1" \
  "$MODEL_ROOT/VideoChat2" \
  "$MODEL_ROOT/vicuna-7b-v1.5" \
  "$MODEL_ROOT/mplug-owl-llama-7b-video" \
  "$MODEL_ROOT/OTTER-Video-LLaMA7B-DenseCaption"
do
  if [ -d "$path" ]; then
    echo "FILES $path"
    find "$path" -maxdepth 2 -type f -printf '%P\t%s\n' 2>/dev/null | sort | head -80
  fi
done
