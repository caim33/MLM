# Current refactor status

Updated: 2026-08-21 (Asia/Shanghai)

The active repository is `/wangbenyou-sulongjie/caimeng/qwen-codebase`. Read, in order:

1. `docs/status/README.md`
2. `docs/architecture/ARCHITECTURE.md`
3. `docs/guides/USAGE_GUIDE.md`
4. `docs/architecture/formal/SFT_FORMAL_PROVENANCE.md`
5. `docs/architecture/formal/SFT_FORMAL_BOOTSTRAP_PLAN.md`
6. `model_evaluation_agent/RUNNER_BACKENDS.md`

The 2026-07-29/30 asset and CUDA smoke records in older handoffs are historical
evidence, not current server validation or fresh-batch results.

All catalog production finetune/evaluation/verifier and completion paths are
currently fail-closed with `blocker=verified-multi-root-bootstrap`. Only
finetune preflight/debug may execute, and those artifacts cannot be promoted.
Formal Qwen SFT is independently fail-closed as well. Do not remove these gates;
implement and audit the controller-verified `-I -S -B` multi-root bootstrap first.

No new accuracy, QA500-v2 predictions, formal checkpoint, or release was
generated. The global fresh-finetune barrier and `1 -> 8 -> 32 -> 500`
evaluation order remain mandatory for all 15 registry models.
