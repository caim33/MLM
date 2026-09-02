# Production runner/backend status

> Current global blocker (2026-08-21): production finetune, evaluation,
> reload verification, and completion are disabled until a controller-verified
> `-I -S -B` multi-root source/environment bootstrap is implemented. Only
> finetune preflight may execute, and its artifact cannot be promoted. Backend
> presence below is therefore necessary but not sufficient.

Catalog runner files are strict, model-bound CLI facades. A facade being
present is not evidence that its model integration exists: each batch receipt
also freezes the implementation backend path, presence state, and file hash.

## Reviewed implementations

| Model | Finetune backend | Evaluation backend | Reload backend |
|---|---|---|---|
| `videollama_lora` | real PEFT LoRA optimizer path | missing | fresh base + PEFT adapter reload |
| `motionllm_official` | real project-owned LoRA + projector optimizer path | missing | fresh base + strict LoRA/projector reload |
| Other 13 registry models | missing | missing | missing |

No missing integration emits an artifact, prediction, or success receipt. The
facade exits non-zero before creating output. Historical scripts and weights
are not substituted.

Because production attempts must be evaluable end to end, the controller
requires finetune, evaluation, and reload backends before creating a production
finetune attempt. Until reviewed evaluation implementations are added, use a
formal component blocker during the finetune phase, for example:

```text
motion-eval finetune block BATCH_ID --model-id videollama_lora \
  --reason missing_code --component backend:evaluation \
  --detail "reviewed production evaluation backend is not installed"
```

The corresponding expected missing path and frozen runtime-contract hash are
recorded in blocker evidence. Preflight finetune attempts require only the real
finetune backend and remain available for the two implemented models.

## Runtime preflight

Install the declared Linux/CUDA dependencies from `requirements/sft.txt` (or
the `sft` project extra), then run:

```text
python model_evaluation_agent/scripts/preflight_runner_dependencies.py \
  --model-id videollama_lora --pretrained-root "$PRETRAINED_ROOT" --require-cuda
python model_evaluation_agent/scripts/preflight_runner_dependencies.py \
  --model-id motionllm_official --pretrained-root "$PRETRAINED_ROOT" --require-cuda
```

The production facades repeat dependency, Linux, CUDA, GPU-binding, frozen
input, and output-freshness checks immediately before importing model code.
The inventory reports finetune readiness separately, but exits successfully
only when all three production backend roles are present and ready. At the
current reviewed state both models therefore remain formally blocked for
production even if their real finetune dependency checks pass.
