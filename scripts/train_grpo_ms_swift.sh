#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${MOTION_GRPO_PYTHON:-}" ]]; then
  echo "ERROR: set MOTION_GRPO_PYTHON to the absolute Python interpreter of the frozen GRPO environment." >&2
  exit 64
fi
if [[ "${MOTION_GRPO_PYTHON}" != /* || ! -x "${MOTION_GRPO_PYTHON}" ]]; then
  echo "ERROR: MOTION_GRPO_PYTHON must be an existing executable absolute path." >&2
  exit 64
fi
if [[ "${1:-}" != "--config" || -z "${2:-}" ]]; then
  echo "Usage: MOTION_GRPO_PYTHON=/abs/env/bin/python $0 --config /abs/formal-grpo.yaml [--dry_run|--preflight_only]" >&2
  exit 64
fi

CONFIG_PATH="$2"
shift 2
if [[ "${CONFIG_PATH}" != /* || ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: --config must name an existing absolute formal GRPO YAML file." >&2
  exit 64
fi
if [[ "$#" -gt 1 ]]; then
  echo "ERROR: expected at most one mode: --dry_run or --preflight_only." >&2
  exit 64
fi
if [[ "$#" -eq 1 && "$1" != "--dry_run" && "$1" != "--preflight_only" ]]; then
  echo "ERROR: unknown formal GRPO argument: $1" >&2
  exit 64
fi

cd "${ROOT_DIR}"
export MOTION_GRPO_FORMAL_LAUNCHER=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
exec "${MOTION_GRPO_PYTHON}" -I -B qwenvl/grpo_ms_swift/runner/train_grpo_ms_swift.py \
  --config "${CONFIG_PATH}" \
  "$@"
