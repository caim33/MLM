# Pretrained asset handoff

> **Current override (2026-08-21):** The inventory and smoke below are
> historical 2026-07-30 evidence. Re-audit them on the authenticated current
> server; they never count as current-batch fresh finetune completion.

Updated: 2026-07-30 (Asia/Shanghai)

## Outcome

The remote pretrained-asset gate covers all 15 canonical registry entries.

- Registry models: 15
- Pretrain asset ready: 15
- Content-hashed unique files: 2,836
- Unique logical referenced bytes: 245,177,919,010
- Canonical tree physical usage: approximately 16 GiB
- Offline component smoke: passed
- VideoLLaMA true-LoRA optimizer-step smoke: passed
- MotionLLM batch-JSON LoRA-plus-projector optimizer-step smoke: passed
- Preparation, hashing, and smoke processes left running: 0

The logical byte count is not a new 245 GB copy. Existing large model stores
are linked into `by_model/`; the controller's canonical pretrained directory
itself occupies approximately 16 GiB.

`AGCN` is the intentional exception to the phrase “pretrain checkpoint”: the
official 2s-AGCN recipe initializes the official architecture and trains it
from scratch. Its official source is pinned and its CUDA forward smoke passes;
no proxy or invented checkpoint is substituted.

This status means the base/pretrain inputs and verified runner prerequisites
are staged. It does **not** waive the rule that every new data batch must
create a fresh finetune artifact for every model before evaluation begins.

## Remote layout

Controller:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval`

Canonical pretrained root:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval/shared_assets/pretrained`

Important paths:

```text
shared_assets/pretrained/
  by_model/
    <registry_model_id>/
      base, processor, tower, projector, or official checkpoint links
  sources/
    2s-AGCN/
    MotionCLIP/
    MotionLLM/
  downloads/
    MotionCLIP/
    EvanTHU--MotionLLM-7B/
    MotionLLM-vicuna-7b-v1.5-lit/
    VideoLLaMA-runtime/
      torch/hub/checkpoints/
        eva_vit_g.pth
        blip2_pretrained_flant5xxl.pth
  runtime_deps/
    motionllm/
  download_manifest.json
  pretrained_inventory.json
  component_smoke.json
```

New public downloads are retained under `downloads/`. Do not copy linked bases
or canonical downloads into a batch directory.

## Verified special cases

### VideoLLaMA

The upstream code loads two base components through torch.hub. They are now
canonical offline assets:

- EVA-ViT SHA-256:
  `99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f`
- BLIP-2 Q-Former SHA-256:
  `4b3839ea6c617f315ead9bf4036bbb0f0cf6bf62695ecfc14968ea626af03a29`
- True PEFT-LoRA runner:
  `scripts/finetune_videollama_lora.py`
- One-step smoke: finite loss, non-zero gradient, and non-zero parameter update

### AGCN

- Official repo: `https://github.com/lshiwjx/2s-AGCN.git`
- Revision: `953c14fc10883cd869646328f5d522e9e9282063`
- License: CC-BY-NC-4.0
- Pretrain checkpoint: not required by the official training recipe
- Smoke: official model initialization and CUDA forward passed

### MotionCLIP

- Official repo: `https://github.com/GuyTevet/MotionCLIP.git`
- Revision: `8eae36d59465711d52bcc14853d1e081022f5056`
- Official paper checkpoint:
  `by_model/motionclip_official/pretrained.pth.tar`
- Checkpoint SHA-256:
  `acafc4bc0d3300ff92a01fc8b75e1a9d129fa253899a2c4648c2260298358e21`
- Smoke: all 101 encoder keys loaded strictly; dummy motion output was finite

### MotionLLM

- Official repo: `https://github.com/IDEA-Research/MotionLLM.git`
- Revision: `6695061aea0e1b8a48004cfea48c72e21ce67b3a`
- Official public weights revision:
  `feda751a8c906a0c5d18e241b3140bc6badd5913`
- Official LoRA SHA-256:
  `c28beab8172cab510f6f414bea8bcec1ac27dca8f54d0f0189a7bda0c53e0977`
- Official projector SHA-256:
  `682606481c2724dfc9340257102b746657d2f0ec520695244e55691884608753`
- Converted Vicuna Lit-GPT base:
  `by_model/motionllm_official/vicuna_lit/lit_model.pth`
- Pinned isolated runtime:
  `runtime_deps/motionllm/`
- Strict-load smoke: base 227 keys + official LoRA 64 keys, zero
  missing/unexpected keys; projector and tokenizer checks passed
- Batch runner smoke: finite loss, non-zero gradient, and non-zero parameter
  update using `scripts/finetune_motionllm_lora.py --data-path ...`

The public MotionLLM repository does not publish a finetune entrypoint. The
controller therefore uses an explicitly project-owned runner that follows the
official multimodal embedding path and trains LoRA plus the projector. This
ownership distinction must remain visible in every batch manifest.

## Cleanup state

Historical LoRA, GRPO, SFT, proxy, and infrastructure-smoke weight outputs
were deleted after reference checks and evidence capture:

- Freed allocated bytes: 68,903,427,584
- Freed: 64.17 GiB
- Retained: canonical pretrain, source, runners, data, configs, logs,
  manifests, and evaluation evidence
- Recovery: deleted weight outputs are not recoverable from the controller

Future batches must create fresh weights under
`batches/<batch_id>/02_finetune/`.

## Evidence

- Canonical specification: `pretrained_registry.json`
- Content inventory:
  `server_audit/20260730_pretrained_inventory.json`
- Inventory SHA-256:
  `e94415c6447eda8c89b55fb51abf0202b2c57a2f9dbd1d2f6528b60946aa0543`
- Component smoke:
  `server_audit/20260729_pretrained_component_smoke.json`
- Component smoke SHA-256:
  `a2e13527af82140ca34bb72a66d0bd9b70fd9aa684a813e78f0132f01d53372c`
- Finetune-runner smoke:
  `server_audit/20260730_finetune_smoke.json`
- Cleanup handoff:
  `server_audit/20260730_smoke_cleanup_complete.md`

## Mandatory next-batch sequence

1. Run the pretrained gate before freezing training commands.
2. Freeze train/validation/benchmark and leakage manifests.
3. Point every model only to its canonical pretrain inputs.
4. Produce a new batch-local finetune checkpoint/adapter for every unblocked
   registry model.
5. Observe the global finetune barrier.
6. Evaluate only current-batch finetune outputs.

Pretrain assets, official released LoRA, and historical evidence are never
accepted as the new batch's finetune artifact.
