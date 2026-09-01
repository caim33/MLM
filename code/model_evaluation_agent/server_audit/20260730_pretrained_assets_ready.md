# Pretrained assets ready audit

Updated: 2026-07-30 (Asia/Shanghai)

Remote root:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval/shared_assets/pretrained`

Result:

- Canonical registry entries: 15
- Pretrain asset ready: 15
- Unique selected files content-hashed: 147
- Unique selected bytes content-hashed: 242,649,463,491
- Offline component smoke: passed
- AGCN official CUDA initialization/forward: passed
- MotionCLIP official encoder strict load: passed, 0 missing/unexpected keys
- MotionLLM base + official LoRA strict load: passed, 0
  missing/unexpected keys
- MotionLLM projector strict load: passed, 0 missing/unexpected keys
- Qwen/Motion-R1 processor offline loads: 5/5 passed
- Preparation/hash/smoke processes left running: 0
- At final handoff, unrelated Qwen QA-generation jobs were using the GPUs; they were not started or modified by this work.

Evidence:

- `20260729_pretrained_inventory.json`
  - SHA-256:
    `ee6c546b23f59cb21de46242620cdb809217b3b01c04475711b57f7dbcd93f32`
- `20260729_pretrained_component_smoke.json`
  - SHA-256:
    `a2e13527af82140ca34bb72a66d0bd9b70fd9aa684a813e78f0132f01d53372c`
- `20260729_pretrained_download_manifest.json`
  - SHA-256:
    `f2a3dcb0d4195cd9d4a11a8e61569dcf49ea56809ea1357fa75ef7bc89ab1e8d`

Newly restored:

- Official 2s-AGCN source at
  `953c14fc10883cd869646328f5d522e9e9282063`. Its official training
  recipe does not require a pretrain checkpoint.
- Official MotionCLIP source at
  `8eae36d59465711d52bcc14853d1e081022f5056` and paper checkpoint
  SHA-256
  `acafc4bc0d3300ff92a01fc8b75e1a9d129fa253899a2c4648c2260298358e21`.
- Official MotionLLM source at
  `6695061aea0e1b8a48004cfea48c72e21ce67b3a`, released weights at
  `feda751a8c906a0c5d18e241b3140bc6badd5913`, and converted
  Vicuna-7B-v1.5 Lit-GPT base.

Interpretation:

This audit closes the missing-pretrain/base-weight gate. It does not count any
staged upstream weight or historical adapter as the next batch's fresh
finetune artifact. MotionLLM's public upstream repository still lacks a
finetune script, and VideoLLaMA LoRA still requires a new-batch training smoke;
those are training-runner gates rather than missing-pretrain gates.
