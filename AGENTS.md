# Engineering Rules

This repository is the clean, maintainable MotionLLM/Qwen codebase. The old
QwenVL checkout and the previous refactor requirements are evidence, not active
implementation policy. Their preserved copies under `legacy/` are read-only.

## Read before changing code

1. `docs/status/README.md`
2. `docs/architecture/ARCHITECTURE.md`
3. `docs/guides/DEVELOPMENT.md`
4. `docs/guides/USAGE_GUIDE.md`
5. `docs/architecture/MIGRATION.md` when touching compatibility code

## Source-of-truth boundaries

- `src/motionllm/contracts`: dependency-light domain contracts.
- `src/motionllm/data`: strict data reading, paths, messages, datasets, and
  collation contracts.
- `src/motionllm/motion`: motion arrays, normalization, and temporal handling.
- `src/motionllm/fusion`, `models`, `training`, `grpo`: model-facing core.
- `src/motion_eval`: evaluation data, controller, runtime, backends, and reports.
- `qwenvl/`, top-level `models/`, and `rubric_rl/`: compatibility entrypoints.
  New core logic must not be added there when it belongs under `src/`.
- `legacy/`: provenance-only material. Active modules must never import it.

Dependencies point inward: compatibility entrypoints may import `src/` modules;
core modules must not import compatibility or legacy modules.

## Required behavior

- No hard-coded dataset, model, checkpoint, output, user-home, or server path in
  active Python modules. Paths come from CLI arguments or explicit config.
- Importing a module or asking for `--help` must not initialize CUDA, load model
  weights, open datasets, or contact a remote service.
- Invalid rows fail at their original identity. Never substitute the next row,
  silently skip malformed examples, or remove errors from an evaluation
  denominator.
- Errors crossing module boundaries are typed and include the source path and
  row/sample identity without echoing payloads or secrets.
- Production actions are disabled by default. A development or preflight result
  cannot be relabeled as production evidence.
- Secrets are injected through process environment or a secret manager and must
  not appear in code, config, manifests, logs, commands, or documentation.

## Change protocol

1. State the module boundary and public API being changed.
2. Add or update the smallest relevant unit/contract test.
3. Keep compatibility wrappers thin; document any intentionally supported old
   argument or import path.
4. Run the affected test directory. Cross-module changes also run integration
   tests and `scripts/run_checks.py` when the checkout has complete dependencies.
5. Update `docs/status/README.md` when capability or blocker state changes.

Changes spanning data, training, evaluation, or compatibility modules require an
independent review and an adversarial test pass before handoff.

## Definition of done

- The package imports on CPU with the base dependencies.
- `python -m motion_eval --help` succeeds without Torch/CUDA.
- Public commands have `--help`, examples, side effects, and output locations
  documented.
- Unit and contract tests for the changed modules pass; skipped tests are listed
  explicitly and are never counted as passed.
- No active code imports `legacy/`, and no generated cache, credential, model
  weight, dataset, prediction, or run output is committed.
