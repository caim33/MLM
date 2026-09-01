# Persistent evaluation memory

> **Current override (2026-08-21):** Read `CURRENT_REFACTOR_STATUS.md` first.
> All catalog production execution is fail-closed pending the verified
> multi-root bootstrap; older remote readiness statements are historical.

Updated: 2026-07-30 (Asia/Shanghai)

## Role

I am the single agent responsible for executing and evaluating every model in
the canonical registry. I own asset audit, fresh finetuning, leakage audit,
smoke tests, full evaluation, manifests, per-sample outputs, error accounting,
and the final unified release.

Primary source:
`History/对话文档/T9_T10_T11_全模型统一评估说明_20260729.md`.

## Latest user override

For every newly supplied data batch:

1. Register and freeze the batch's train, validation, benchmark, media, and
   derivation manifests.
2. Finetune **all 15 registry models** on the approved train/validation data.
3. Each model must produce a checkpoint or adapter created inside the current
   batch. An old checkpoint cannot satisfy this gate.
4. Use a global phase barrier. Do not begin the evaluation phase until every
   registry entry is either `finetune_complete` or has an evidence-backed
   `blocked` status.
5. Evaluate only models whose current-batch finetune status is complete.
   A blocked model remains blocked; it is never replaced by a baseline or
   proxy.

This replaces the older allowance to directly re-evaluate an existing
checkpoint.

## Pretrain readiness clarification

The user requires base/pretrained checkpoints to be staged before the next
data batch so that the batch can enter finetuning without first hunting for
weights. The canonical remote entry is:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval/shared_assets/pretrained/by_model/<model_id>`

Use `pretrained_registry.json` as the specification and
`shared_assets/pretrained/pretrained_inventory.json` as the byte-hash
evidence. As of 2026-07-30, all 15 registry entries pass the pretrain asset
gate and the component smoke. AGCN intentionally has no pretrain checkpoint
because its official recipe trains the official architecture from random
initialization.

The canonical tree occupies approximately 16 GiB physically. The inventory's
245,177,919,010 logical referenced bytes are not a second 245 GB copy: most
large models are symlinked from existing server model stores. Never duplicate
those linked bases into a batch directory.

VideoLLaMA additionally requires the canonical offline EVA-ViT and BLIP-2
torch.hub checkpoints under `downloads/VideoLLaMA-runtime/`. MotionLLM requires
the isolated packages under `runtime_deps/motionllm/`; do not downgrade the
shared Python environment.

Pretrain readiness never counts as current-batch finetune completion.
`scripts/finetune_videollama_lora.py` passed a true PEFT-LoRA optimizer-step
smoke. The official MotionLLM repository has no public finetune entrypoint, so
the explicitly project-owned `scripts/finetune_motionllm_lora.py` is used; its
batch-JSON LoRA-plus-projector path also passed a real optimizer-step smoke.
The evidence is `server_audit/20260730_finetune_smoke.json`.

Historical finetune weight subtrees and generated smoke weights were deleted
after hashing on 2026-07-30. Source, configs, logs, data, manifests, and eval
evidence remain. Never look for a reusable historical adapter: every new batch
must write fresh weights under its own `02_finetune/<model_id>/`.

## Canonical coverage

The only main-table coverage is the 15 entries in `model_registry.json`:

- Qwen3.6-27B LoRA (V)
- Motion-R1 VM LoRA (VM)
- Qwen3-VL-8B LoRA (V)
- Qwen3-VL-4B LoRA (V)
- Qwen3.5-4B LoRA (V)
- Video-LLaVA-7B LoRA (V)
- VideoChatGPT LoRA (V)
- VideoChat2 LoRA (V)
- VideoLLaMA trainables (V)
- VideoLLaMA LoRA (V)
- mPLUG-Owl Video LoRA (V)
- Otter-Video LoRA (V)
- AGCN official (M)
- MotionCLIP official (M)
- MotionLLM official restored video path (V)

Motion-R1 V-only/M-only results are diagnostic or ablation results, never a
replacement for its VM main result.

## Non-negotiable data gates

Formal full evaluation is forbidden until:

- the benchmark has a unique frozen version and SHA-256;
- V, M, VM, and T views derive from one canonical QA file;
- option permutation, seed, media hash, text hash, and derivation-code hash
  are recorded;
- sample ID, group ID, media-content hash, normalized question/options, and
  near-duplicate leakage checks pass between train/validation and benchmark;
- every fresh checkpoint has training provenance and a leakage audit;
- evaluator prompt, preprocessing, parser, errors, timeouts, and result schema
  are frozen.

The currently inspected server manifest describes the older `QA_500` dataset.
It does not by itself prove that `QA500-v2` and the new unified protocol are
frozen. Until a new batch passes the gates, only 1/8/32-sample smoke runs are
allowed and no formal 500-row result may be released.

## Finetune contract

Every finetune manifest must record:

- batch ID and model registry ID;
- official repo/source, license, revision, and dirty-state/code hash;
- base model identity/revision/hash;
- train/validation manifest paths, row counts, and SHA-256;
- excluded benchmark identity and leakage-audit artifact;
- modality actually consumed;
- preprocessing and sampling;
- optimizer, schedule, epochs/steps, precision, distributed config, seed;
- Python/CUDA/PyTorch/framework environment;
- produced checkpoint/adapter path and SHA-256;
- start/end time and terminal status.

No `eval/` output is valid unless its manifest references the SHA-256 of that
same batch's completed finetune artifact.

## Evaluation contract

For generative models:

- deterministic generation only: `do_sample=false`, `temperature=0`, fixed
  seed;
- the only valid prediction syntax is a complete
  `<answer>[A-D]</answer>` tag;
- an isolated A/B/C/D in explanations is invalid;
- option log-probability/forced-choice scoring is diagnostic only and cannot
  enter the generation main table.

For discriminative models:

- use the canonical option permutation;
- save four raw A/B/C/D scores, final prediction, gold, error state, and
  preprocessing metadata;
- never reorder options per model.

For every model:

- use identical canonical question/options;
- V cannot read motion, M cannot read video, VM must read both;
- invalid, media error, OOM, runtime error, and timeout remain in the fixed
  denominator;
- run asset audit, provenance audit, smoke-1, smoke-8, smoke-32, full
  benchmark, and manifest validation in order.

## Required outputs

Each model must deliver:

- `predictions.jsonl`
- `summary.json`
- `run_manifest.json`
- `status.md`

The batch release must deliver:

- `all_models_results.csv`
- `all_models_results.md`
- `blocked_models.md`
- `evaluation_release_manifest.json`

Old accuracy, parse rate, ranking, predictions, logs, and checkpoints may be
used only to recover assets, commands, and provenance. They are not accepted
as new results or sanity targets.

## Known incompatibilities in historical code

The historical `eval_*_mcq_*.py` runners commonly fall back to parsing any
isolated A/B/C/D when no answer tag is found. That parser is forbidden for new
main-table runs.

Historical Motion-R1/open-VLM evaluators include option-score paths. These may
be retained as diagnostics only.

Historical `motion_proxy_train_eval.py`, `BABEL_AGCN`,
`BABEL_MotionCLIP`, and `MotionX_AGCN` outputs are proxies. Renaming them does
not make them official AGCN or MotionCLIP results.

## Security

`dev_env_connection.txt` is the current connection source and contains a
plaintext secret. Never print or copy it. Load its password into a
process-scoped environment variable only, clear it after use, and keep all
memory/audit/manifests credential-free.
