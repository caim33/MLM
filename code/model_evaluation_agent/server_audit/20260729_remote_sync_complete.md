# Remote controller sync completed

Completed: 2026-07-29 (Asia/Shanghai)

Remote environment:

- Environment name: `auto4gpu_20260729204650`
- Remote hostname: `3v8pjmt48lnm8-0`
- GPU: 4 x NVIDIA H20-3e
- Credentials: intentionally omitted

Controller root created and populated:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval`

The remote controller contains the persistent memory, runbook, 15-model
registry, templates, batch creator, global finetune gate, smoke/full-eval
gates, release validator, and synchronization helper.

Remote `scripts/selftest_workflow.py` completed successfully without loading
models or using GPU compute. The synthetic test exercised:

1. creation of one 15-model batch;
2. frozen input and leakage gates;
3. 12 synthetic current-batch finetune completions;
4. 3 evidence-backed synthetic blockers;
5. global finetune validation;
6. eval-stage opening;
7. smoke-1, smoke-8, and smoke-32 gates;
8. full-eval opening for eligible models;
9. fixed-denominator predictions and summaries;
10. final release validation.

Observed terminal output included:

- `VALID stage=finetune`
- `EVALUABLE 12`
- `BLOCKED 3`
- `VALID stage=release`
- `SELFTEST PASSED`

No real batch was created under `batches/`, no training or evaluation process
was started, and no historical metric was promoted.
