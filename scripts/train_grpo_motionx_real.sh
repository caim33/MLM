#!/bin/bash
set -euo pipefail

echo "ERROR: configs/grpo/motionx_real.yaml is a quarantined historical full-GRPO config and is not publishable." >&2
echo "Copy configs/grpo/formal/motionr1_vm_lora.template.yaml, freeze every path, then use scripts/train_grpo_ms_swift.sh." >&2
exit 64
