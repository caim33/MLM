#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/full_sft.sh [--dataset-use NAME] [--validation-dataset-use NAME]

Formal Motion-R1 VM SFT binds exactly one unsampled train dataset alias and
one distinct unsampled validation alias. Each alias must resolve in
qwenvl/data/__init__.py to the annotation file supplied through
TRAIN_DATA_PATH or VALIDATION_DATA_PATH respectively. Comma-separated aliases
and % sampling are not accepted by the formal provenance contract.

The launcher requires an absolute PYTHON_EXECUTABLE and invokes that exact
interpreter with ``-I -m torch.distributed.run``. Distributed settings are
MASTER_ADDR (default 127.0.0.1), MASTER_PORT (random local port), NNODES
(default 1), NODE_RANK (default 0), and NPROC_PER_NODE (default 1).
ENVIRONMENT_PATH must be that interpreter's isolated environment root, and
RUNNER_CODE_PATH must be the checkout's qwenvl tree. Formal provenance hashes
actual installed files plus a strict SFT source allowlist before and after training.
Formal SFT uses the pinned scripts/zero2.json. ZeRO-3 is rejected because the
current save/reload proof does not implement an all-rank ZeRO-3 parameter gather.
Formal publication is currently fail-closed: the controller has not yet wired
an external-HMAC-bound pre-spawn snapshot and verified in-memory worker bundle.
Use the Python entrypoint's --unsafe_legacy_no_manifest mode only for ineligible
non-release smoke runs.
EOF
}

# Parse the small public CLI before inspecting the formal-run environment so
# that --help remains useful on a clean shell. The two aliases are single-use:
# accepting a later override would make the logged command ambiguous.
train_dataset_override=""
validation_dataset_override=""
train_dataset_seen=0
validation_dataset_seen=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-use|--datasets)
      if (( train_dataset_seen != 0 )); then
        echo "Train dataset override may be supplied only once" >&2
        exit 2
      fi
      if (( $# < 2 )); then
        echo "$1 requires a value" >&2
        exit 2
      fi
      [[ "$2" != -* ]] || { echo "$1 requires a value" >&2; exit 2; }
      train_dataset_override="$2"
      train_dataset_seen=1
      shift 2
      ;;
    --validation-dataset-use)
      if (( validation_dataset_seen != 0 )); then
        echo "Validation dataset override may be supplied only once" >&2
        exit 2
      fi
      if (( $# < 2 )); then
        echo "$1 requires a value" >&2
        exit 2
      fi
      [[ "$2" != -* ]] || { echo "$1 requires a value" >&2; exit 2; }
      validation_dataset_override="$2"
      validation_dataset_seen=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# This shell file cannot prove its own already-executed bytes (or BASH_ENV /
# loader startup) and therefore must never start a formal worker directly.
# Keep --help available, then fail before the first interpreter probe.  The
# Python worker contains the same block as defense in depth.
echo "Formal Qwen SFT is blocked: use a future verified controller that supplies an external-HMAC-bound pre-spawn snapshot and in-memory worker source bundle" >&2
exit 78

required_env=(
  BATCH_ID MODEL_REGISTRY_ID BASE_ARTIFACT_PATH TRAIN_DATA_PATH
  VALIDATION_DATA_PATH BENCHMARK_PATH LEAKAGE_AUDIT_PATH CONFIG_PATH
  CODE_PATH RUNNER_CODE_PATH ENVIRONMENT_PATH ARTIFACT_ROOT ARTIFACT_MANIFEST_PATH
  TRAINING_RECEIPT_PATH RELOAD_RECEIPT_PATH BATCH_RECEIPT_SHA256
  ATTEMPT_SHA256 PYTHON_EXECUTABLE TRAIN_DATASET_USE VALIDATION_DATASET_USE
  MOTION_VQVAE_ASSET_PATH SEED
)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

# torchrun launches each worker as ``PYTHON_EXEC -u <script>`` rather than
# propagating the launcher's ``-I`` flag. Reject ambient Python code-injection
# knobs, then publish an explicit clean worker environment and interpreter.
unsafe_python_env=(
  PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHON_EXEC
)
for name in "${unsafe_python_env[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    echo "Formal SFT rejects ambient Python environment variable: ${name}" >&2
    exit 2
  fi
done
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHON_EXEC
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
for name in \
  BASE_ARTIFACT_PATH TRAIN_DATA_PATH VALIDATION_DATA_PATH BENCHMARK_PATH \
  LEAKAGE_AUDIT_PATH CONFIG_PATH CODE_PATH RUNNER_CODE_PATH ENVIRONMENT_PATH ARTIFACT_ROOT \
  MOTION_VQVAE_ASSET_PATH; do
  if [[ ! -e "${!name}" ]]; then
    echo "Required provenance path does not exist: ${name}" >&2
    exit 2
  fi
done
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be an explicit non-negative integer" >&2
  exit 2
fi

for digest_name in BATCH_RECEIPT_SHA256 ATTEMPT_SHA256; do
  if [[ ! "${!digest_name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${digest_name} must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done

canonical_model_registry_id=motionr1_vm_lora
canonical_model_family=qwen3_vl_motion
if [[ "$MODEL_REGISTRY_ID" != "$canonical_model_registry_id" ]]; then
  echo "This launcher is bound to MODEL_REGISTRY_ID=${canonical_model_registry_id}" >&2
  exit 2
fi
if [[ -n "${MODEL_FAMILY:-}" && "$MODEL_FAMILY" != "$canonical_model_family" ]]; then
  echo "This launcher is bound to MODEL_FAMILY=${canonical_model_family}" >&2
  exit 2
fi
model_family=$canonical_model_family

if [[ "$PYTHON_EXECUTABLE" != /* || ! -f "$PYTHON_EXECUTABLE" || ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "PYTHON_EXECUTABLE must be an absolute regular executable file" >&2
  exit 2
fi
interpreter_identity=$("$PYTHON_EXECUTABLE" -I -c \
  'import os, sys; print(os.path.realpath(sys.executable))') || {
  echo "PYTHON_EXECUTABLE could not report its interpreter identity" >&2
  exit 2
}
if [[ -z "$interpreter_identity" || "$interpreter_identity" == *$'\n'* ]]; then
  echo "PYTHON_EXECUTABLE returned an invalid interpreter identity" >&2
  exit 2
fi
python_real=$(realpath -e -- "$PYTHON_EXECUTABLE")
if [[ "$python_real" != "$interpreter_identity" ]]; then
  echo "PYTHON_EXECUTABLE identity does not match the executable on disk" >&2
  exit 2
fi
environment_identity=$("$PYTHON_EXECUTABLE" -I -c \
  'import os, sys; print(os.path.realpath(sys.prefix))') || {
  echo "PYTHON_EXECUTABLE could not report its environment root" >&2
  exit 2
}
base_environment_identity=$("$PYTHON_EXECUTABLE" -I -c \
  'import os, sys; print(os.path.realpath(sys.base_prefix))') || {
  echo "PYTHON_EXECUTABLE could not report its base environment root" >&2
  exit 2
}
if [[ "$environment_identity" == "$base_environment_identity" ]]; then
  echo "Formal SFT requires an isolated Python environment" >&2
  exit 2
fi
environment_real=$(realpath -e -- "$ENVIRONMENT_PATH")
if [[ ! -d "$ENVIRONMENT_PATH" || "$environment_real" != "$environment_identity" ]]; then
  echo "ENVIRONMENT_PATH must identify the current isolated environment root" >&2
  exit 2
fi
export PYTHON_EXEC="$PYTHON_EXECUTABLE"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "${script_dir}/.." && pwd -P)
code_path_real=$(realpath -e -- "$CODE_PATH")
if [[ ! -d "$CODE_PATH" || "$code_path_real" != "$project_root" ]]; then
  echo "CODE_PATH must be the project root containing this launcher" >&2
  exit 2
fi
runner_code_real=$(realpath -e -- "$RUNNER_CODE_PATH")
if [[ ! -d "$RUNNER_CODE_PATH" || "$runner_code_real" != "${project_root}/qwenvl" ]]; then
  echo "RUNNER_CODE_PATH must be the actual Qwen runner tree at CODE_PATH/qwenvl" >&2
  exit 2
fi

train_dataset=${train_dataset_override:-$TRAIN_DATASET_USE}
validation_dataset=${validation_dataset_override:-$VALIDATION_DATASET_USE}

for item in "$train_dataset" "$validation_dataset"; do
  if [[ -z "$item" || "$item" == *","* || "$item" == *"%"* ]]; then
    echo "Formal dataset aliases must be non-empty, singular, and unsampled" >&2
    exit 2
  fi
done
if [[ "$train_dataset" == "$validation_dataset" ]]; then
  echo "Train and validation dataset aliases must be distinct" >&2
  exit 2
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$("$PYTHON_EXECUTABLE" -I -c 'import secrets; print(20001 + secrets.randbelow(9999))')}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
for setting in MASTER_PORT NNODES NODE_RANK NPROC_PER_NODE; do
  if [[ ! "${!setting}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "${setting} must be a canonical non-negative integer" >&2
    exit 2
  fi
done
if ! "$PYTHON_EXECUTABLE" -I -c \
  'import sys; port, nodes, rank, procs=map(int, sys.argv[1:]); raise SystemExit(not (1 <= port <= 65535 and nodes >= 1 and procs >= 1 and 0 <= rank < nodes))' \
  "$MASTER_PORT" "$NNODES" "$NODE_RANK" "$NPROC_PER_NODE"; then
  echo "Invalid torchrun topology or port" >&2
  exit 2
fi

if [[ -n "${DEEPSPEED_CONFIG+x}" ]]; then
  echo "Formal SFT rejects DEEPSPEED_CONFIG overrides; scripts/zero2.json is mandatory" >&2
  exit 2
fi
deepspeed=${project_root}/scripts/zero2.json
if [[ ! -f "$deepspeed" ]]; then
  echo "DeepSpeed configuration does not exist: $deepspeed" >&2
  exit 2
fi

motion_dataname=${MOTION_DATANAME:-t2m}
motion_quantizer=${MOTION_QUANTIZER:-ema}
vqvae_nb_code=${VQVAE_NB_CODE:-512}
vqvae_code_dim=${VQVAE_CODE_DIM:-512}
vqvae_output_emb_width=${VQVAE_OUTPUT_EMB_WIDTH:-512}
vqvae_down_t=${VQVAE_DOWN_T:-2}
vqvae_stride_t=${VQVAE_STRIDE_T:-2}
vqvae_width=${VQVAE_WIDTH:-512}
vqvae_depth=${VQVAE_DEPTH:-3}
vqvae_dilation_growth_rate=${VQVAE_DILATION_GROWTH_RATE:-3}
vqvae_activation=${VQVAE_ACTIVATION:-relu}
vqvae_norm=${VQVAE_NORM:-none}
for setting in \
  vqvae_nb_code vqvae_code_dim vqvae_output_emb_width vqvae_stride_t \
  vqvae_width vqvae_depth vqvae_dilation_growth_rate; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${setting} must be a positive integer" >&2
    exit 2
  fi
done
if [[ ! "$vqvae_down_t" =~ ^(0|[1-9][0-9]*)$ ]] || (( vqvae_down_t > 30 )); then
  echo "vqvae_down_t must be an integer between 0 and 30" >&2
  exit 2
fi

learning_rate=${LEARNING_RATE:-2e-4}
batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}
grad_accum_steps=${GRADIENT_ACCUMULATION_STEPS:-32}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
if ! "$PYTHON_EXECUTABLE" -I -c \
  'import math, sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
  "$learning_rate"; then
  echo "LEARNING_RATE must be finite and greater than zero" >&2
  exit 2
fi
for setting in batch_size grad_accum_steps num_train_epochs; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${setting} must be a positive integer" >&2
    exit 2
  fi
done
report_to=${REPORT_TO:-none}
if [[ "$report_to" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "REPORT_TO=wandb requires WANDB_API_KEY in the process environment" >&2
  exit 2
fi

group_num_mv=${GROUP_NUM_MV:-1}
group_num_motion=${GROUP_NUM_MOTION:-0}
group_num_video=${GROUP_NUM_VIDEO:-0}
group_num_text=${GROUP_NUM_TEXT:-0}
for setting in group_num_mv group_num_motion group_num_video group_num_text; do
  if [[ ! "${!setting}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "${setting} must be a canonical non-negative integer" >&2
    exit 2
  fi
done
if ! "$PYTHON_EXECUTABLE" -I -c \
  'import sys; mv, motion, video, text=map(int, sys.argv[1:]); raise SystemExit(not (mv >= 1 and motion == video == text == 0))' \
  "$group_num_mv" "$group_num_motion" "$group_num_video" "$group_num_text"; then
  echo "Motion-R1 VM formal training requires GROUP_NUM_MV>0 and all other GROUP_NUM_* values equal to 0" >&2
  exit 2
fi

entry_file=${project_root}/qwenvl/train/full_sft.py
if [[ ! -f "$entry_file" ]]; then
  echo "Training entry point does not exist inside CODE_PATH" >&2
  exit 2
fi
output_dir=${OUTPUT_DIR:-"${ARTIFACT_ROOT}/checkpoints/${BATCH_ID}/${MODEL_REGISTRY_ID}/full_sft"}
artifact_root_real=$(realpath "${ARTIFACT_ROOT}")
output_dir_real=$(realpath -m "${output_dir}")
manifest_real=$(realpath -m "${ARTIFACT_MANIFEST_PATH}")
training_receipt_real=$(realpath -m "${TRAINING_RECEIPT_PATH}")
reload_real=$(realpath -m "${RELOAD_RECEIPT_PATH}")
case "${output_dir_real}/" in "${artifact_root_real}/"*) ;; *) echo "OUTPUT_DIR must be inside ARTIFACT_ROOT" >&2; exit 2;; esac
case "${manifest_real}/" in "${output_dir_real}/"*) echo "ARTIFACT_MANIFEST_PATH must be outside OUTPUT_DIR" >&2; exit 2;; esac
case "${training_receipt_real}/" in "${output_dir_real}/"*) echo "TRAINING_RECEIPT_PATH must be outside OUTPUT_DIR" >&2; exit 2;; esac
case "${reload_real}/" in "${output_dir_real}/"*) echo "RELOAD_RECEIPT_PATH must be outside OUTPUT_DIR" >&2; exit 2;; esac
if [[ "$manifest_real" == "$training_receipt_real" || "$manifest_real" == "$reload_real" || "$training_receipt_real" == "$reload_real" ]]; then
  echo "Manifest, training receipt, and reload receipt paths must be distinct" >&2
  exit 2
fi
if [[ -e "$output_dir" || -e "$ARTIFACT_MANIFEST_PATH" || -e "$TRAINING_RECEIPT_PATH" || -e "$RELOAD_RECEIPT_PATH" ]]; then
  echo "Formal output, manifest, training receipt, and reload receipt paths must be fresh" >&2
  exit 2
fi

log_grad_norm_every=${LOG_GRAD_NORM_EVERY:-0}
if [[ ! "$log_grad_norm_every" =~ ^[0-9]+$ ]]; then
  echo "LOG_GRAD_NORM_EVERY must be a non-negative integer" >&2
  exit 2
fi
log_dir=${LOG_DIR:-${project_root}/logs}
mkdir -p "$log_dir"
log_file="${log_dir}/$(basename "$output_dir").log"

args=(
  --deepspeed "$deepspeed"
  --model_family "$model_family"
  --model_name_or_path "$BASE_ARTIFACT_PATH"
  --batch_id "$BATCH_ID"
  --model_registry_id "$MODEL_REGISTRY_ID"
  --base_artifact_path "$BASE_ARTIFACT_PATH"
  --train_data_path "$TRAIN_DATA_PATH"
  --validation_data_path "$VALIDATION_DATA_PATH"
  --benchmark_path "$BENCHMARK_PATH"
  --leakage_audit_path "$LEAKAGE_AUDIT_PATH"
  --config_path "$CONFIG_PATH"
  --code_path "$CODE_PATH"
  --runner_code_path "$RUNNER_CODE_PATH"
  --environment_path "$ENVIRONMENT_PATH"
  --artifact_root "$ARTIFACT_ROOT"
  --artifact_manifest_path "$ARTIFACT_MANIFEST_PATH"
  --training_receipt_path "$TRAINING_RECEIPT_PATH"
  --reload_receipt_path "$RELOAD_RECEIPT_PATH"
  --batch_receipt_sha256 "$BATCH_RECEIPT_SHA256"
  --attempt_sha256 "$ATTEMPT_SHA256"
  --motion_vqvae_asset_path "$MOTION_VQVAE_ASSET_PATH"
  --motion_vqvae_path "$MOTION_VQVAE_ASSET_PATH"
  --motion_dataname "$motion_dataname"
  --motion_quantizer "$motion_quantizer"
  --vqvae_nb_code "$vqvae_nb_code"
  --vqvae_code_dim "$vqvae_code_dim"
  --vqvae_output_emb_width "$vqvae_output_emb_width"
  --vqvae_down_t "$vqvae_down_t"
  --vqvae_stride_t "$vqvae_stride_t"
  --vqvae_width "$vqvae_width"
  --vqvae_depth "$vqvae_depth"
  --vqvae_dilation_growth_rate "$vqvae_dilation_growth_rate"
  --vqvae_activation "$vqvae_activation"
  --vqvae_norm "$vqvae_norm"
  --motion_length_divisor "$((2**vqvae_down_t))"
  --motion_timestamps_sync_with_video True
  --dataset_use "$train_dataset"
  --eval_dataset_use "$validation_dataset"
  --seed "$SEED"
  --data_flatten True
  --tune_mm_vision False
  --tune_mm_mlp False
  --tune_mm_llm False
  --bf16
  --output_dir "$output_dir"
  --num_train_epochs "$num_train_epochs"
  --per_device_train_batch_size "$batch_size"
  --gradient_accumulation_steps "$grad_accum_steps"
  --max_pixels 50176
  --min_pixels 784
  --save_strategy epoch
  --save_total_limit 1
  --learning_rate "$learning_rate"
  --weight_decay 0
  --warmup_ratio 0.03
  --max_grad_norm 1
  --lr_scheduler_type cosine
  --logging_steps 1
  --log_grad_norm_every "$log_grad_norm_every"
  --model_max_length 8192
  --gradient_checkpointing True
  --dataloader_num_workers 4
  --run_name "${RUN_NAME:-qwen3vl-full-sft}"
  --report_to "$report_to"
)

export GROUP_NUM_MV=$group_num_mv
export GROUP_NUM_MOTION=$group_num_motion
export GROUP_NUM_VIDEO=$group_num_video
export GROUP_NUM_TEXT=$group_num_text

if (( NNODES != 1 || NODE_RANK != 0 )); then
  echo "Formal Qwen SFT supports one node only; cross-node provenance is not implemented" >&2
  exit 3
fi
echo "Formal Qwen SFT is blocked: missing controller external-HMAC-bound pre-spawn snapshot and verified in-memory worker source bootstrap" >&2
exit 3
