# Qwen data compatibility layer

This directory is a compatibility facade for historical `qwenvl.data` imports.
The maintained implementation lives in `src/motionllm/qwen` and translates
Qwen training/inference interfaces into the strict contracts under
`src/motionllm/data`. Neither layer imports `legacy/`.

## Module map

| Module | Responsibility |
|---|---|
| `src/motionllm/qwen/registry.py` | Resolve file-backed dataset aliases or validated process-scoped aliases |
| `src/motionllm/qwen/processor.py` | Build Qwen messages, obtain assistant masks, and expand motion placeholders |
| `src/motionllm/qwen/dataset_adapter.py` | Read configured annotations and prepare identity-preserving Torch items |
| `src/motionllm/qwen/collators.py` | Convert `motionllm.data.CollationPlan` into standard or flattened Torch batches |
| `src/motionllm/qwen/rope2d.py` | Qwen3-VL position IDs with explicit tokenizer IDs |
| `data_processor.py` | Thin re-export facade for historical imports |

## Dataset aliases

Committed aliases use `configs/datasets/<alias>.dataset.json`. Machine-specific
config folders are selected with `MOTIONLLM_DATASET_CONFIG_DIR`. Inference may
register explicit CLI paths without mutating a global dictionary:

```python
from qwenvl.data import register_dataset

register_dataset(
    "infer_runtime",
    annotation_path="/absolute/input.jsonl",
    data_path="/absolute/media-root",
    split="eval",
)
```

Motion rows also require explicit `motion_mean_path`, `motion_std_path`, and an
optional `expected_motion_dim`. These assets are never discovered beside the
Python package.

## Compatibility invariants

- `sample_id`, `group_id`, branch, logical ownership, and motion lengths survive
  both standard and flattened collation.
- A failed row is retried only at the same index and is never replaced by the
  next sample.
- Video tensors and grids must agree with the declared physical branch.
- Sequences that exceed the tokenizer limit fail rather than truncating a
  multimodal span.
- Chat templates must return an assistant token mask; historical hard-coded
  assistant/end token IDs are not used.

Qwen2/Qwen2.5 visual RoPE and legacy image-only rows remain fail-closed until
they have independent contract tests. They must not be represented as verified
by the Qwen3 adapter.

## Focused validation

```bash
python -m pytest tests/unit/test_collator_legacy_interface.py -q
```

Torch and Transformers are optional heavy dependencies. If they are absent,
pytest skips this file; a skip is not a pass.
