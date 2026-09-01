# MotionLLM refactor instructions

## Mandatory reading

Before modifying code, read:

- `docs/REFACTOR_HANDOFF.md`
- `docs/ARCHITECTURE_TARGET.md`
- `docs/COMMON_COMMANDS.md`
- `model_evaluation_agent/00_从这里开始.md`
- `model_evaluation_agent/MEMORY.md`
- `model_evaluation_agent/RUNBOOK.md`
- `model_evaluation_agent/model_registry.json`
- `model_evaluation_agent/PRETRAINED_ASSETS.md`
- `model_evaluation_agent/pretrained_registry.json`
- the newest file in `model_evaluation_agent/server_audit/`

## Source of truth

- `D:\MotionLLM\History` is immutable evidence. Never edit it.
- `D:\MotionLLM\motionllm_refactor` is the only active refactor tree.
- Do not reintroduce `.bak` files, archived outputs, credentials, private keys, connection files, or model weights.
- Preserve legacy public entrypoints through thin facades until their compatibility tests pass.

## Required multi-agent workflow

The coordinating AI must use multiple subagents when the work spans independent modules.

1. Assign non-overlapping file ownership before edits.
2. Use implementation agents for independent modules.
3. Use a separate review agent that did not author the reviewed files.
4. Use a separate adversarial-test agent for malformed data, failures, concurrency, and security.
5. Do not let two agents edit the same file concurrently.
6. Integrate centrally, run the full test suite, and resolve review findings before handoff.

Recommended waves:

- Wave A: contracts/data; motion/fusion; evaluation/controller.
- Wave B: model facade; training/GRPO; model adapters.
- Wave C: independent review; failure injection; GPU smoke.

## Compatibility invariants

Keep these names and semantics until an explicit migration release:

- `Qwen3VlMotionForConditionalGeneration`
- `from_pretrained`, `forward`, `prepare_inputs_for_generation`
- legacy state-dict prefixes `motion_encoder`, `motion_prenorm`, `motion_proj`, `motion_postnorm`, `motion_boundary_embed`
- sample metadata `motion`, `motion_lengths`, `branch`, `sample_id`, `group_id`, `rollout_id`, `request_id`
- `<motion_start><motion><motion_end>` placeholder protocol
- generation encodes motion during prefill only
- 15 canonical model IDs and the global fresh-finetune barrier

Bad legacy behavior is not a compatibility invariant. In particular, do not preserve silent sample substitution, loose answer parsing, path-name model guessing, unchecked `strict=False`, or secrets in source.

## Evaluation and finetune rules

- Every new batch requires a fresh finetune artifact for every unblocked registry model.
- Do not start evaluation until all 15 models are `finetune_complete` or evidence-backed `blocked`.
- Main-table generation accepts only a complete `<answer>[A-D]</answer>`.
- Invalid output, missing media, timeout, OOM, and runtime errors remain in the fixed denominator.
- Historical accuracies, historical predictions, baseline runs, option-score diagnostics, and proxy AGCN/MotionCLIP runs never enter the new main table.

## Security

- Never print or copy credentials from `D:\MotionLLM\dev_env_connection.txt`.
- Secrets may exist only in process-scoped environment variables.
- Never use `AutoAddPolicy` in new SSH code; pin and verify the host key.
- Redact password/token/key values in command previews, logs, errors, manifests, and reviews.
- GPU keepalive must be project-owned, PID/UUID tracked, idle-only, and fail closed when GPU status cannot be determined.

## Definition of done for a change

- The requested behavior is covered by tests.
- CPU tests, compile checks, secret scan, and affected integration tests pass.
- Compatibility impact is documented.
- A non-authoring reviewer has checked the change.
- Extreme/failure cases relevant to the module have been exercised.
- Documentation commands match real CLI behavior.

