# GRPO formal schema fixtures

`formal_vm_v_train.jsonl` and `formal_vm_v_validation.jsonl` are minimal,
valid inputs for the runner's complete formal data preflight. Each split has
one canonical VM/V pair, strict semantic references, one shared video binding,
and row-level media SHA-256 values. The two splits use disjoint sample, group,
video, and motion identities.

The `.fixture` media files are deliberately tiny non-empty byte fixtures for
schema, path, hash, split-leakage, and co-location tests. They are **not**
decodable videos or motion tensors and therefore must never be presented as a
CUDA/ms-swift end-to-end smoke result. A real launch must replace them with
frozen decodable media while preserving the same row contract.
