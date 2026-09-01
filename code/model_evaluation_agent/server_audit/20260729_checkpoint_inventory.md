# Historical checkpoint inventory

Updated: 2026-07-29 (Asia/Shanghai)

Remote inventory:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval/shared_assets/checkpoint_inventory.json`

Inventory SHA-256:

`3eb46f95d1271ea80db6153e5440df781563c9cc0e8e1f9f1902c174f0f15fd9`

Summary:

- Canonical registry entries: 15
- Historical finetune artifacts indexed: 11
- Missing historical/official artifacts: 4
- Selected primary weight bytes hashed: 8,110,604,432
- Large files copied: 0
- Historical recovery symlinks created: 11
- GPU processes started: 0

Missing entries:

- `videollama_lora`
- `agcn_official`
- `motionclip_official`
- `motionllm_official`

Five Qwen/Motion-R1 base paths were verified present: Qwen3.6-27B,
Qwen3.5-4B, Qwen3-VL-8B, Qwen3-VL-4B, and the Motion-R1 base checkpoint.
Legacy video-model base dependencies still require a per-batch asset audit.

Every indexed artifact is labeled `historical_recovery_only` and
`usable_as_current_batch_finetune=false`. It cannot satisfy the user's
fresh-finetune requirement for a new data batch.
