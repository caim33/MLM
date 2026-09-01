# Initial server/code audit

Audit time: 2026-07-29, Asia/Shanghai  
Mode: read-only  
Credentials: intentionally omitted

## Connection and capacity

- Connection succeeded using the current workspace connection file.
- Remote hostname: `1tfj6bl4serni-0`
- Remote user: `root`
- GPU: 4 x NVIDIA H20-3e, about 143771 MiB each.
- At inspection time all four GPUs showed zero allocated compute memory and
  zero utilization.
- Root filesystem: about 1.0 TiB total, 668 GiB available.

## Important remote roots

- Main Qwen/Motion-R1 repo:
  `/wangbenyou-sulongjie/qwen-vl-finetune`
- Historical all-model workspace:
  `/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM`
- Historical all-model run:
  `/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/finetune_goal_20260717`
- Historical runner directory:
  `/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_tools`
- Model cache:
  `/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_models`

## Data observed

Active clean-training manifest:
`/wangbenyou-sulongjie/qwen-vl-finetune/data/grpo_training/manifest.json`

Observed active text files:

| File | Rows | SHA-256 |
|---|---:|---|
| `text/train/train_1928qa_paired.jsonl` | 3856 | `a5a8cc6bd3a704f49c57234e4e0f3a86d70cebe0cc9f74796ca1c07e1bbaff6b` |
| `text/val/val_86qa_vm_only.jsonl` | 86 | `2cac8c062418b174fe7056b5a51e685a4e175f59c7a10acb37092871cf0e5bff` |
| historical `QA_500.jsonl` | 500 | `a7140146ed89968e293ebad83d6b87cbcdf8d91281539c91a30d2f0aa4ef1da9` |

The clean manifest states that 684 train rows and 14 validation rows were
removed by sample/group overlap with the historical QA500. This is useful
provenance evidence, but the inspected benchmark manifest is the older
`QA_500` manifest created on 2026-07-02. It does not establish the required
QA500-v2 freeze, option-permutation manifest, derivation hashes, or full
content/near-duplicate leakage audit.

## Code observed

Historical code hashes:

| File | SHA-256 |
|---|---|
| `orchestrate_goal_finetune.py` | `50770b296373e45a3eeba83beacd137b6a5fb1aec8545ed4b4af30eaa8e4c72d` |
| `prepare_goal_finetune_data.py` | `0df06f4d1b2c07fccfcba1649896f8da83c66ef5bc73e2f6d3ca87375b75fe1a` |
| `eval_videollava_lora_mcq_generate.py` | `5c252f159991508d2b31e206caecb5ece9715f28d0d885d033a01a9f0e3ad6f6` |

The old orchestrator launches proxy MLP/RNN jobs for AGCN and MotionCLIP and
records blockers for several official models. It is not the new unified
pipeline.

The historical generation evaluators accept a complete answer tag first but
then fall back to any isolated A/B/C/D. This violates the current strict-parser
rule. Historical score evaluators also contain forced option-scoring paths,
which are diagnostic-only under the new protocol.

## Model asset state

Qwen base caches were observed for Qwen3.6-27B, Qwen3.5-4B,
Qwen3-VL-8B-Instruct, and Qwen3-VL-4B-Instruct. The historical run also
contains adapters/trainables and logs for the Qwen/Motion-R1 and legacy video
models. These are recovery evidence only; none satisfies a new batch's fresh
finetune requirement.

Paths named `2s-AGCN` and `MotionCLIP` exist under the all-model workspace, but
the inspection found zero regular source files across those two roots and only
small directory/Git skeletons. No usable source revision or license could be
verified.

`BABEL_AGCN`, `BABEL_MotionCLIP`, and `MotionX_AGCN` contain proxy artifacts
and `.pyc` files, including files explicitly named `*_proxy.py`. They cannot
be used as official AGCN or MotionCLIP.

The historical run marks MotionLLM blocked for missing/restorable official
repo, runner, weights, and current finetune artifact. The initial inspection
did not establish a usable official MotionLLM finetune path.

## Mandatory next audit before a real batch

1. Re-read the latest connection file without printing it.
2. Recheck GPU processes, disk, mounts, and relevant paths because the
   environment is ephemeral.
3. Freeze the new data batch and QA500-v2 before training.
4. Restore and verify official AGCN, MotionCLIP, and MotionLLM sources.
5. Replace or patch the historical parsers to enforce exact answer tags.
6. Build new current-batch finetune manifests for every unblocked model.
