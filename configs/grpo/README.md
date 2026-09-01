# GRPO configuration boundary

Only copied, fully resolved LoRA configs derived from
`formal/motionr1_vm_lora.template.yaml` are eligible for a new formal batch.
The formal runner accepts `training.tuner_type: lora` (or the PEFT alias),
requires a fresh exact adapter leaf, and publishes only after independent
reload verification.

The 22 historical full-GRPO YAML files were moved to
`legacy/refactor_snapshot/configs/grpo_personal/`. They are enumerated in
`legacy_full_inventory.json`; they are not templates, formal smoke configs,
current-batch evidence, or publishable artifacts. The runner fails closed on
every `tuner_type: full` config.

Formal full-GRPO publication remains disabled. Do not relabel a legacy full
config as LoRA. Start from the formal template and create a fresh batch-bound
adapter instead.

After copying and resolving the template, use one frozen interpreter for all
three gates:

```bash
export MOTION_GRPO_PYTHON="/absolute/grpo-env/bin/python"
bash scripts/train_grpo_ms_swift.sh --config /absolute/batch/configs/motionr1_vm_lora_grpo.yaml --dry_run
bash scripts/train_grpo_ms_swift.sh --config /absolute/batch/configs/motionr1_vm_lora_grpo.yaml --preflight_only
bash scripts/train_grpo_ms_swift.sh --config /absolute/batch/configs/motionr1_vm_lora_grpo.yaml
```

`--dry_run` performs complete static config, provenance, split, schema, media
path/hash, and output-destination checks without importing the GPU stack.
`--preflight_only` additionally imports the frozen Swift/PEFT APIs, checks exact
critical versions and CUDA, and exits before creating an output or receipt.
Only the final command trains, independently reloads the exact adapter leaf,
and then publishes its manifest.

Before any gate, replace the template's step/call placeholders with positive
integers. `run.expected_optimizer_steps` and `training.max_steps` must match;
`save_steps` must divide that value and `artifact_path` must end in the exact
`checkpoint-N`. Production run names containing `smoke` or `debug` are
rejected. Copy the whole repository as the immutable code bundle, invoke the
launcher from that copy, and keep all outputs outside it; `provenance.code_path`
must equal the code root that is actually executing the runner.
Set `run.swift_launcher_sha256` to the lowercase SHA-256 of the exact `swift`
console script resolved inside `MOTION_GRPO_PYTHON`'s environment. The runner
checks its bytes, shebang, bound interpreter and ms-swift distribution entry
point, then executes that already-resolved absolute path.

The leakage audit is strict JSON with schema
`motionllm.grpo.leakage_audit.v1`. It binds the current batch ID, exact
train/validation/benchmark SHA-256 values and row counts, normalization ID
`nfkc_casefold_whitespace_v1`, and zero counts for `sample_id`, `group_id`,
`media_sha256`, `normalized_prompt`, `normalized_solution`, and
`near_duplicate`. An empty or generic audit receipt is rejected.

After Swift exits, publication additionally requires a fresh exact checkpoint
with the sole safe adapter weight format `adapter_model.safetensors`,
`trainer_state.json`, `optimizer.pt`,
`scheduler.pt`, `training_args.bin`, and `rng_state*.pth`. Global step must
equal the frozen expected optimizer steps, a finite positive-step loss must be
present, and the independently loaded adapter must contain finite, non-zero
LoRA-B update evidence. `adapter_model.bin`, symlinked files, empty tensors and
implicit serialization fallback are forbidden; the formal config therefore
requires `training.save_safetensors: true`.

A frozen child callback must also write
`grpo_training_receipt.json` into the exact checkpoint leaf, bound to the
runner-generated nonce, with non-zero gradient and before/after trainable-state
delta evidence. Every optimizer step requires a finite non-zero gradient.
`logging_steps` is fixed to `1`, and the receipt must contain exactly one
gradient observation and one finite-loss observation per expected optimizer
step. Missing or all-zero gradients, a missing
`accelerator.optimizer_step_was_skipped` API, or a skipped AMP optimizer step
stop the run before the step is counted. At the final save event, pinned PEFT
extracts the complete live default-adapter saveable state, including LoRA A/B
and every `modules_to_save` tensor. Every `requires_grad` tensor must map one to
one to that state. Pinned safetensors then reads a stable private 0600 copy of
the exact checkpoint payload, and keys, dtype, shape and storage bytes must all
match. The v2 receipt binds both the canonical tensor-state SHA-256 and the raw
payload SHA-256/size; the parent runner recaptures these bytes before and after
independent reload.

The same checkpoint must contain strict, non-symlinked `adapter_config.json`
and ms-swift `additional_config.json` files. The former is checked against the
exact 36-key PEFT 0.18.0 LoRA schema and the live default adapter; the latter
must contain exactly `lora_dtype`, `lorap_lr_ratio`, and `lorap_emb_lr` and
must match the explicit formal YAML defaults. Raw file SHA-256/size and
normalized semantic SHA-256 values for both files are included in the v2
receipt. Missing, duplicate, extra, non-finite, type-invalid, mutated, or
replaced config data stops publication. A frozen independent reload is checked
with the pure saveable-state extractor; trainable coverage is required only
for the live training model.

The formal template enables the frozen
`motion_training_receipt` callback; if the pinned ms-swift runtime does not
invoke every required Trainer event, production stops without a manifest.
Dataset, media, model, VQ-VAE, environment, config and critical code hashes are
checked again before publication. The code snapshot recursively includes the
complete `src/motionllm` and `src/motion_eval` package trees, so transitive
imports cannot escape a hand-maintained allowlist; any mutation aborts
publication. The launcher and verified Swift entry point both use the exact
bound Python in isolated, no-bytecode mode (`-I -B`). Child workers inherit
`PYTHONNOUSERSITE=1`, `PYTHONSAFEPATH=1`, and
`PYTHONDONTWRITEBYTECODE=1`; CUDA's `LD_LIBRARY_PATH` remains available.

The JSONL files under `tests/fixtures/grpo` exercise the complete schema/path/
hash preflight. Their byte fixtures are intentionally not decodable GPU media;
see that directory's README before interpreting a smoke result.
