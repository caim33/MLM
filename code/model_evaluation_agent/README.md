# Unified model evaluation agent

This is the persistent working directory for MotionLLM all-model finetuning
and evaluation.

Chinese entry point: `00_从这里开始.md`.

Start every new data batch by reading `00_从这里开始.md` and `MEMORY.md`,
then follow `RUNBOOK.md`.
`model_registry.json` is the canonical 15-model coverage list. The latest
server inspection is under `server_audit/`.

## Directory roles

- `MEMORY.md`: durable user requirements, protocol, and non-negotiable rules.
- `RUNBOOK.md`: phase gates and the per-batch filesystem contract.
- `model_registry.json`: machine-readable model coverage and current asset
  readiness.
- `server_audit/`: dated, credential-free snapshots of remote state.
- `templates/`: required output templates.
- `scripts/`: batch creation and phase-gate utilities.
- `batches/`: per-batch manifests and reports. Large weights and media
  stay on the server and are referenced by path plus SHA-256.

Historical files under `History/` are evidence for recovery and auditing only.
They are not current official evaluation results.
