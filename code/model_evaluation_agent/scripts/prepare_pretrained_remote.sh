#!/usr/bin/env bash
set -euo pipefail

ROOT=/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM
ASSETS="$ROOT/codex_runs/unified_model_eval/shared_assets/pretrained"
SOURCES="$ASSETS/sources"
BY_MODEL="$ASSETS/by_model"
DOWNLOADS="$ASSETS/downloads"
TOOLS="$ASSETS/tools"

mkdir -p "$SOURCES" "$BY_MODEL" "$DOWNLOADS" "$TOOLS"

clone_once() {
  url=$1
  dst=$2
  if [ -d "$dst/.git" ]; then
    echo "SOURCE_PRESENT $dst $(git -C "$dst" rev-parse HEAD)"
    return
  fi
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    echo "ERROR target exists but is not a git repository: $dst" >&2
    exit 3
  fi
  git clone "$url" "$dst"
  echo "SOURCE_CLONED $dst $(git -C "$dst" rev-parse HEAD)"
}

link_once() {
  src=$1
  dst=$2
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then
    echo "ERROR source missing: $src" >&2
    exit 4
  fi
  mkdir -p "$(dirname "$dst")"
  if [ -L "$dst" ]; then
    current=$(readlink -f "$dst" || true)
    wanted=$(readlink -f "$src" || true)
    if [ "$current" = "$wanted" ]; then
      echo "LINK_PRESENT $dst -> $current"
      return
    fi
    echo "ERROR conflicting link: $dst -> $current; wanted $wanted" >&2
    exit 5
  fi
  if [ -e "$dst" ]; then
    echo "ERROR target exists and is not a link: $dst" >&2
    exit 6
  fi
  ln -s "$src" "$dst"
  echo "LINK_CREATED $dst -> $(readlink -f "$dst")"
}

clone_once https://github.com/lshiwjx/2s-AGCN.git "$SOURCES/2s-AGCN"
clone_once https://github.com/GuyTevet/MotionCLIP.git "$SOURCES/MotionCLIP"
clone_once https://github.com/IDEA-Research/MotionLLM.git "$SOURCES/MotionLLM"

link_once "$ROOT/codex_models/Qwen__Qwen3.6-27B" "$BY_MODEL/qwen36_27b_lora/base"
link_once "$ROOT/codex_models/qwen3_vl_motion_checkpoint_0426" "$BY_MODEL/motionr1_vm_lora/base"
link_once /wangbenyou-sulongjie/Motion-r1/model/pretrained/VQVAE/net_best_fid.pth "$BY_MODEL/motionr1_vm_lora/motion_vqvae.pth"
link_once "$ROOT/codex_models/Qwen__Qwen3-VL-8B-Instruct" "$BY_MODEL/qwen3vl_8b_lora/base"
link_once "$ROOT/codex_models/Qwen__Qwen3-VL-4B-Instruct" "$BY_MODEL/qwen3vl_4b_lora/base"
link_once "$ROOT/codex_models/Qwen__Qwen3.5-4B" "$BY_MODEL/qwen35_4b_lora/base"

link_once "$ROOT/MVBench_Eval/models/Video-LLaVA-7B" "$BY_MODEL/videollava_7b_lora/base"
link_once "$ROOT/MVBench_Eval/cache_dir/models--LanguageBind--LanguageBind_Video_merge/snapshots/efc40ec6ba6b2081276c11e7e19b24f08a099e79" "$BY_MODEL/videollava_7b_lora/video_tower"

link_once "$ROOT/MVBench_Eval/models/LLaVA-7B-Lightening-v1-1" "$BY_MODEL/videochatgpt_lora/base"
link_once "$ROOT/MVBench_Eval/models/Video-ChatGPT/video_chatgpt-7B.bin" "$BY_MODEL/videochatgpt_lora/projector.bin"

link_once "$ROOT/MVBench_Eval/models/VideoChat2/umt_l16_qformer.pth" "$BY_MODEL/videochat2_lora/umt_l16_qformer.pth"
link_once "$ROOT/MVBench_Eval/models/VideoChat2/videochat2_7b_stage3.pth" "$BY_MODEL/videochat2_lora/stage3.pth"
link_once "$ROOT/MVBench_Eval/models/vicuna-7b-v1.5" "$BY_MODEL/videochat2_lora/vicuna"

link_once "$ROOT/MVBench_Eval/models/Video-LLaMA-2-7B-Finetuned" "$BY_MODEL/videollama_trainables/base"
link_once "$ROOT/MVBench_Eval/models/Video-LLaMA-2-7B-Finetuned" "$BY_MODEL/videollama_lora/base"

link_once "$ROOT/MVBench_Eval/models/mplug-owl-llama-7b-video" "$BY_MODEL/mplug_owl_video_lora/base"
link_once "$ROOT/MVBench_Eval/models/OTTER-Video-LLaMA7B-DenseCaption" "$BY_MODEL/otter_video_lora/base"

link_once "$SOURCES/2s-AGCN" "$BY_MODEL/agcn_official/source"
link_once "$SOURCES/MotionCLIP" "$BY_MODEL/motionclip_official/source"
link_once "$SOURCES/MotionLLM" "$BY_MODEL/motionllm_official/source"
link_once "$ROOT/MVBench_Eval/cache_dir/models--LanguageBind--LanguageBind_Video_merge/snapshots/efc40ec6ba6b2081276c11e7e19b24f08a099e79" "$BY_MODEL/motionllm_official/video_tower"
link_once "$ROOT/MVBench_Eval/models/vicuna-7b-v1.5" "$BY_MODEL/motionllm_official/vicuna_hf"
if [ -f "$DOWNLOADS/MotionLLM-vicuna-7b-v1.5-lit/lit_model.pth" ]; then
  link_once "$DOWNLOADS/MotionLLM-vicuna-7b-v1.5-lit" "$BY_MODEL/motionllm_official/vicuna_lit"
fi

echo "PREPARED $ASSETS"
