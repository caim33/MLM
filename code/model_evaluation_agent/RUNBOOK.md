# All-model finetune/eval runbook

> **Current override (2026-08-21):** Read `CURRENT_REFACTOR_STATUS.md` and
> `docs/COMMON_COMMANDS.md` before this historical runbook. Direct legacy
> commands cannot bypass the current production blocker or fresh-finetune gate.

## Batch identity

Create one immutable `batch_id`, for example
`qa500v2_20260729_<datahash8>`. Never reuse a batch directory after its input
manifest changes.

Recommended remote root:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval/batches/<batch_id>`

Required layout:

```text
<batch_id>/
  00_inputs/
    batch_manifest.json
    benchmark_manifest.json
    leakage_audit.json
  01_asset_audit/
    <model_id>.json
  02_finetune/
    <model_id>/
      run_manifest.json
      status.md
      checkpoint-or-adapter
  03_eval/
    <model_id>/
      smoke_1/
      smoke_8/
      smoke_32/
      full/
        predictions.jsonl
        summary.json
        run_manifest.json
        status.md
  04_release/
    all_models_results.csv
    all_models_results.md
    blocked_models.md
    evaluation_release_manifest.json
```

## Phase 0: input freeze

Create the batch skeleton from the remote controller root:

```bash
python3 scripts/new_batch.py <batch_id> --description "<short description>"
```

1. Inventory train, validation, benchmark, video, motion, and canonical QA.
2. Compute row counts and SHA-256.
3. Freeze option permutation and derivation code.
4. Run sample/group/content/question/option and near-duplicate leakage checks.
5. Stop if the benchmark is not uniquely versioned or leakage is unresolved.

## Phase 1: asset and provenance audit

For all 15 registry entries, verify official source, license, revision, base
weights, processor, dependencies, modality path, and training entrypoint.
Historical adapters are evidence only.

Before creating any training command, run the canonical pretrain gate from the
remote controller root:

```bash
python3 scripts/validate_pretrained_ready.py
```

All training commands must read base/pretrained inputs from:

`shared_assets/pretrained/by_model/<model_id>/`

The detailed mapping and hashes are in `pretrained_registry.json`,
`shared_assets/pretrained/pretrained_inventory.json`, and
`shared_assets/pretrained/component_smoke.json`. If the gate fails, repair or
formally block the affected model; do not silently fall back to a historical
adapter, baseline, or proxy.

AGCN has no required pretrain checkpoint in its official recipe. Its valid
starting state is the pinned official implementation with freshly initialized
parameters. MotionCLIP uses the pinned official paper checkpoint. MotionLLM
uses the converted Vicuna Lit-GPT base, LanguageBind video tower, and official
published LoRA/projector as its staged upstream inputs. Its runner must prepend
`shared_assets/pretrained/runtime_deps/motionllm` to `PYTHONPATH`. Those
published weights still do not count as the current batch's fresh finetune
output.

An entry may be marked `blocked` only with concrete missing-path, missing-code,
missing-weight, incompatible-environment, or unrecoverable-provenance
evidence. Proxy availability is not evidence of official readiness.

## Phase 2: fresh finetune for every model

Create a new current-batch checkpoint/adapter for each unblocked model.
Validate that the training config points only to the frozen train/validation
manifests and never to benchmark rows or media.

The finetune stage is globally closed until all 15 entries are one of:

- `finetune_complete`
- `blocked`

`failed`, `pending`, or a historical checkpoint does not open the eval gate.

### Verified VideoLLaMA and MotionLLM runners

Both runners consume a JSON list whose rows contain an absolute `video` path
and a `conversations` user/assistant pair. Resolve and hash that JSON before
training. Use the batch's own output directories:

```bash
ROOT=/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval
PY=/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_envs/mllm/bin/python
BATCH="$ROOT/batches/<batch_id>"

# VideoLLaMA true PEFT LoRA
"$PY" "$ROOT/scripts/finetune_videollama_lora.py" \
  --data-path "$BATCH/00_inputs/<videollama_train.json>" \
  --work-dir "$BATCH/02_finetune/videollama_lora/work" \
  --output-dir "$BATCH/02_finetune/videollama_lora/artifact" \
  --max-steps <frozen_step_count>

# MotionLLM project-owned LoRA + projector runner
PRE="$ROOT/shared_assets/pretrained"
SRC="$PRE/by_model/motionllm_official/source"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PRE/runtime_deps/motionllm:$SRC${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$ROOT/scripts/finetune_motionllm_lora.py" \
  --data-path "$BATCH/00_inputs/<motionllm_train.json>" \
  --video-root "<optional_root_for_relative_video_paths>" \
  --output-dir "$BATCH/02_finetune/motionllm_official/artifact" \
  --epochs <frozen_epoch_count> \
  --max-video-tokens <frozen_token_count>
```

For an infrastructure preflight, add `--limit 1 --max-steps 1`. Never reuse
the preflight artifact as the batch artifact. The verified smoke evidence is
`server_audit/20260730_finetune_smoke.json`; the generated smoke weights were
deleted after hashing.

MotionLLM's `--max-video-tokens` changes the training preprocessing and must be
declared in the batch manifest. The memory-safe infrastructure smoke used 8
tokens; that value is not automatically the production choice.

When all 15 finetune manifests are terminal, run:

```bash
python3 scripts/validate_batch.py --stage finetune batches/<batch_id>
python3 scripts/open_eval_stage.py batches/<batch_id>
```

`open_eval_stage.py` refuses to create `03_eval/` while the global finetune
barrier is closed.

## Phase 3: evaluation

Only models with `finetune_complete` may enter evaluation.

For each such model:

1. `smoke_1`: load, modality, and output schema.
2. `smoke_8`: answer positions and question types.
3. `smoke_32`: parser, error rate, memory, speed, timeout.
4. `full`: only after all three smoke gates pass.
5. Validate exact denominator, one row per canonical sample, hashes, modality,
   and error taxonomy.

After setting every smoke `status.json` to `passed`, open full evaluation:

```bash
python3 scripts/open_full_eval.py batches/<batch_id> --all
```

The script refuses to create any `full/` directory until the model's 1/8/32
smokes are all marked passed.

Do not silently retry only failed samples under changed settings. A retry must
create a new run attempt with the same frozen protocol or a new batch/protocol
version.

## Phase 4: release

Run:

```bash
python3 scripts/validate_batch.py --stage release batches/<batch_id>
```

Release only when the validator succeeds and every registry model has an
evidence-backed terminal status. Main tables include only new current-batch
results from the frozen benchmark.

## Handoff at the start of each user-supplied batch

Report:

- detected input files and assumptions;
- batch ID and hashes;
- blockers before training;
- estimated GPU allocation and sequencing;
- the explicit global rule: all model finetunes are accounted for before eval
  begins.
