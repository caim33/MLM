# Finetune smoke and checkpoint cleanup complete

Updated: 2026-07-30 (Asia/Shanghai)

## Outcome

- VideoLLaMA true PEFT LoRA: one real optimizer step passed.
- MotionLLM project-owned LoRA + projector runner: one real optimizer step
  passed.
- All 15 canonical model entries still pass the pretrained-asset gate after
  cleanup.
- Historical finetune, proxy, and smoke-generated weight outputs removed:
  68,903,427,584 allocated bytes (64.17 GiB).
- Canonical pretrained tree physical usage: approximately 16 GiB.
- Canonical logical referenced bytes: 245,177,919,010 bytes. This is not
  additional physical usage because the large existing model stores are linked
  into `by_model/`.

## Smoke evidence

Machine-readable evidence:

`server_audit/20260730_finetune_smoke.json`

VideoLLaMA:

- Loss: `0.3471278250217438`
- Trainable parameters: `8,388,608`
- Maximum gradient: `0.01922607421875`
- Maximum parameter update: `1.9999988580821082e-05`
- Peak CUDA allocation: `29,794,477,056` bytes

MotionLLM:

- Batch JSON runner smoke: passed through `--data-path`
- Loss: `2.0877151489257812`
- Trainable parameters: `54,534,144`
- Maximum gradient: `0.17034912109375`
- Maximum parameter update: `3.0517578125e-05`
- Peak CUDA allocation: `15,046,987,776` bytes

Generated smoke weights were hashed into the evidence file and then deleted.
They are not pretrained assets and must not be reused by a future batch.

## Newly fixed pretrain/runtime dependencies

VideoLLaMA now uses canonical offline torch.hub checkpoints:

- `downloads/VideoLLaMA-runtime/torch/hub/checkpoints/eva_vit_g.pth`
- `downloads/VideoLLaMA-runtime/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth`

MotionLLM now has an isolated, pinned legacy runtime under
`runtime_deps/motionllm/`:

- `transformers==4.28.1`
- `tokenizers==0.13.3`
- `peft==0.8.2`
- `huggingface_hub==0.22.2`

The shared server Python environment was not downgraded.

## Cleanup scope

Removed only weight-output subtrees/files from historical LoRA, GRPO, SFT,
proxy, and smoke runs. Source code, training data, configs, logs, manifests,
and evaluation evidence were retained. No canonical pretrained link resolved
into any deleted target, no active process referenced a deleted target, and
the unrelated QA-generation processes were not changed.

The deletion is not recoverable from the controller directory. A future
evaluation batch must create fresh finetune artifacts from the canonical
pretrained tree.

## Final inventory

- Models ready: `15 / 15`
- Unique hashed files: `2,836`
- Unique logical referenced bytes: `245,177,919,010`
- Inventory:
  `server_audit/20260730_pretrained_inventory.json`
- Inventory SHA-256:
  `e94415c6447eda8c89b55fb51abf0202b2c57a2f9dbd1d2f6528b60946aa0543`
