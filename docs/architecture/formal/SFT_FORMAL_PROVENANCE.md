# Qwen SFT formal-provenance status

## Current eligibility

Formal Qwen full-SFT and LoRA-SFT publication is **fail-closed**. The two shell
launchers exit before their first Python probe, and both Python workers reject a
formal manifest request as defense in depth. `--unsafe_legacy_no_manifest`
remains available only for smoke/debug runs; its outputs are not eligible for a
batch result, resume, release, or evaluation.

The catalog controller also fails closed for every production finetune,
evaluation, and independent-reload verifier. The refusal occurs before the GPU
lease, `ATTEMPT_STARTED`, and `run_verified_python`; production finetune and
evaluation completion refuse existing outputs as well. Finetune preflight is
still available because it cannot publish a formal artifact. The stable
diagnostic is `blocker=verified-multi-root-bootstrap`. A batch model can record
the same terminal condition with reason `unrecoverable_provenance`, component
`verified-multi-root-bootstrap`; that evidence binds the frozen controller-code,
catalog-runner, and runtime-contract digests and is revalidated at every batch
audit.

The blocker is deliberate. A snapshot created inside `full_sft.py` or
`lora_sft.py` happens after Python startup, site processing, and imports of
Torch, Transformers, Qwen, and project modules. Pre/post equality at that point
does not prove which bytes already executed. The in-process snapshot schema is
therefore labelled as a secondary diagnostic, not a pre-spawn execution
attestation.

Formal execution may be re-enabled only after the external, HMAC-bound batch
controller:

1. captures source, data, base-model, environment, interpreter, and runtime
   evidence before importing any worker/project module;
2. starts each worker from verified in-memory source bytes with a positive
   environment allowlist and no mutable checkout fallback;
3. supplies the same controller-bound snapshot identity to every worker; and
4. performs the matching post-training content audit before receipt and
   manifest publication.

The current `run_verified_python` single-root bootstrap is not that contract.
Its `-I` option does not suppress system site initialization, so a system
`sitecustomize` can run before the stdin bootstrap. The catalog facades now
stop production before importing `motion_eval` as defense in depth, but that
script-level check is also necessarily later than interpreter startup. No
current catalog production execution is described as verified. A dedicated
follow-up must implement a controller-verified `-I -S -B` bootstrap whose
in-memory bundle covers every required root and whose environment policy does
not fall back to mutable installed or checkout code.

Until that bootstrap exists, formal Qwen training is restricted to one node and
then rejected. Multi-node provenance is not implemented. A future single-node
implementation must also bind `WORLD_SIZE == LOCAL_WORLD_SIZE`, loopback rendezvous,
zero restarts, and consistent evidence across local ranks.

## Source and environment contracts

The secondary snapshot uses explicit source roots rather than hashing the whole
repository. It covers the complete `src`, `models`, and `qwenvl` trees plus the
pinned project/config/launcher files. This deliberately covers every project
tree admitted as a Python import root; the strict verifier independently
rebuilds the fixed inventory and rejects omitted or extra files.
Repository metadata, logs, checkpoints, receipts, and generated artifacts are
outside this allowlist. Source-backed bytecode cache files inside an admitted
tree are included and hashed. Lexical paths and every selected ancestor are checked before
resolution, so a symlinked fixed file cannot be hidden by `resolve()`.
Sourceless `.pyc`/`.pyo` files are rejected. `.git` and `.cache` directories
inside an admitted import tree are rejected rather than silently pruned, so
they cannot become undeclared import roots.

The environment diagnostic binds the actual worker `sys.prefix`, interpreter
entry and resolved target, installed distribution `METADATA`/`RECORD` entries
and actual installed file hashes. It rejects editable `direct_url.json`, legacy
`.egg-link`, executable or external `.pth`,
`include-system-site-packages != false`, Python startup injection, and native
loader injection. It also records actual `sys.path`, `sys.meta_path`,
`sys.base_prefix` standard-library contents, normalized `LD_LIBRARY_PATH`, and
loaded native-runtime targets (including lexical-link and target-content
digests). `sys.path` and `sys.meta_path` origins are confined to the fully
inventoried `src`, `models`, and `qwenvl` trees, reconstructed distribution site
roots, or the inventoried standard library. A broad checkout, environment, or
base-prefix directory is not an admissible import root: each `sys.path` entry
must equal a declared root, and each file-backed meta finder must name an exact
inventoried file. Every directory symlink, junction, or other reparse point
inside the isolated environment is rejected; recording the link identity while
omitting its reachable target files is not accepted. These checks remain
defense-in-depth; they do not remove the pre-spawn bootstrap blocker.

The source/environment/model/data evidence is recomputed after training and must
match the persisted pre-model diagnostic exactly. Training-receipt schema 2
binds both digests and the snapshot-file digest. Independent artifact and resume
validation reparses the complete snapshot, checks each detailed manifest's
schema and self-hash, and cross-checks every provenance role against the training
receipt and artifact manifest.

The catalog receipt currently names the controller package tree `code` and its
facade scripts `runner_code`, while the Qwen training snapshot uses those names
for the training checkout and `qwenvl` tree. Those are four distinct roles, not
two interchangeable hashes. Reload validation follows and strictly parses a
schema-2 snapshot, then explicitly blocks formal Qwen publication until the
receipt schema separately represents `controller_code`, `catalog_runner_code`,
`training_code`, and `training_runner_code`. It never aliases the incompatible
same-named roles.

## Distributed state and pretrained assets

Formal SFT is pinned to `scripts/zero2.json`. ZeRO-3 is rejected because the
current save/reload proof does not gather and independently verify every sharded
parameter across all ranks.

Large pretrained assets receive a full content hash at batch freeze and at
explicit phase audits (`validate_batch`, opening evaluation, opening full
evaluation, build/verify release). Normal state transitions consume the
immutable batch-receipt index bound to the external event-HMAC state. They check
the index self-hash, receipt/state binding, canonical logical path, and symlink
target, but **do not rehash the underlying asset and do not claim that path,
mtime, or size proves content**. The immutable index has no TTL; phase audits,
not an unrecoverable expiry, are the revalidation boundary.

Gradient evidence aggregates finite/nonzero/max state on-device and performs a
single three-scalar host synchronization per step (plus one distributed MAX
reduction when distributed), preserving the existing proof semantics.

The implementation sequence and adversarial test matrix for removing the
blocker are in `SFT_FORMAL_BOOTSTRAP_PLAN.md`.
