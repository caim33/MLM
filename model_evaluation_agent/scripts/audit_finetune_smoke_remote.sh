#!/usr/bin/env bash
set -euo pipefail

MLLM_ROOT=/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM
CONTROLLER="$MLLM_ROOT/codex_runs/unified_model_eval"
PRETRAIN="$CONTROLLER/shared_assets/pretrained"
VIDEOLLAMA_SOURCE="$MLLM_ROOT/codex_runs/video_model_sources/video-llama"
VIDEOLLAMA_BASE="$MLLM_ROOT/MVBench_Eval/models/Video-LLaMA-2-7B-Finetuned"
MOTIONLLM_SOURCE="$PRETRAIN/sources/MotionLLM"

printf '%s\n' 'GPU_MEMORY'
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader

printf '%s\n' 'VIDEOLLAMA_BASE_TREE'
find "$VIDEOLLAMA_BASE" -maxdepth 3 -type f \
  -printf '%s %P\n' | sort -k2
find "$VIDEOLLAMA_BASE" -maxdepth 3 -type l \
  -printf '%p -> %l\n' | sort

printf '%s\n' 'SOURCE_REVISIONS'
git -C "$VIDEOLLAMA_SOURCE" rev-parse HEAD
git -C "$MOTIONLLM_SOURCE" rev-parse HEAD

printf '%s\n' 'PYTHON_ENVIRONMENTS'
for python_bin in "$MLLM_ROOT"/codex_envs/*/bin/python; do
  if [[ ! -x "$python_bin" ]]; then
    continue
  fi
  echo "===$python_bin"
  "$python_bin" - <<'PY' 2>&1 || true
import sys
print("python", sys.version.split()[0])
for package in ("torch", "transformers", "peft", "bitsandbytes", "lightning"):
    try:
        module = __import__(package)
        print(package, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(package, "MISSING", type(exc).__name__, str(exc))
PY
done

printf '%s\n' 'EXISTING_RELEVANT_RUNNERS'
find "$MLLM_ROOT/codex_runs" -maxdepth 6 -type f \
  \( -iname '*videollama*' -o -iname '*motionllm*' \) \
  -printf '%p\n' | sort | sed -n '1,400p'
