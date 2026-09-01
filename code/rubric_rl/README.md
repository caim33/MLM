# Rubric RL pipelines

There are three deliberately isolated contracts in this directory:

- QA multiple-choice: mode `qa_mc`, ORM name `qa_mc_rubric`.
- Motion Rubric V2: mode `source_aware_reasoning_motion_rubric_v2`, ORM
  name `motion_rubric_v2`.
- Stage 1 temporal caption: mode `temporal_caption`; retained as historical
  compatibility code and **not** registered as a V2 ORM.

Criteria and judgments are never interchangeable across these modes. Both
active rewards reject Stage 1 artifacts, and the Stage 1 reward rejects V2.

## Active strict commands

```text
python -m rubric_rl.extract_qa_mc_criteria --help
python -m rubric_rl.judge_qa_mc --help
python -m rubric_rl.prepare_cot_gt_v2 --help
python -m rubric_rl.extract_motion_criteria_v2 --help
python -m rubric_rl.judge_motion_caption_v2 --help
```

Every active producer publishes the final JSONL atomically and writes
`<output>.inventory.json` containing its SHA-256, row count, unique-ID count,
rubric version, source-file hashes, and a canonical run contract covering the
model/revision, generation settings, prompt/code hashes, field mappings,
fallbacks, shard selection, and total row limit. Interrupted work remains in
`<output>.partial`; pass `--resume` to validate and continue it. Duplicate IDs,
invalid complete JSONL lines, schema mismatches, or any changed run-contract
field fail closed. `--limit` is the total target prefix, not additional rows on
each resume. Model-backed active commands require `--model-revision`.

The former segmented/fast-segmented V2 generators are disabled because their
fragment prompts emitted fields forbidden by the frozen V2 schema. Use the
single strict V2 extractor above.

The ORM accepts either a strictly validated precomputed judgment (useful for
tests) or an online judge configured with a process-scoped endpoint/token.
QA columns are `qa_rubric_criteria` and optional `qa_rubric_judgment`. Motion
V2 columns are `motion_rubric_v2_criteria`, `motion_rubric_v2_id`, and optional
`motion_rubric_v2_judgment`. Without a precomputed judgment, configure:

```text
MOTION_GRPO_QA_RUBRIC_JUDGE_URL
MOTION_GRPO_QA_RUBRIC_JUDGE_TOKEN
MOTION_GRPO_MOTION_RUBRIC_V2_JUDGE_URL
MOTION_GRPO_MOTION_RUBRIC_V2_JUDGE_TOKEN
```

All judge endpoints must use HTTPS and resolve only to public global addresses.
Redirects are forbidden, DNS is resolved once and pinned for the TLS connection,
and one wall-clock deadline covers resolution, connection, request, and body.
Tokens stay process-scoped and are excluded from representations and error
messages. Each judgment is bound to the sample ID, criteria hash, candidate
hash, and nonce; precomputed judgments must be supplied per completion and are
never scalar-broadcast across a batch.

## Historical Stage 1 temporal-caption line

This folder contains a small Stage 1 Rubric-RL pipeline for long, time-segmented motion descriptions.

The intended flow is:

1. Offline: dense GT motion description -> reusable criteria with stable IDs.
2. Online / GRPO reward: candidate caption -> Qwen judge IDs -> deterministic reward post-processing.

The design keeps Qwen away from arithmetic. Qwen only decides which criterion IDs are satisfied, present, aligned, missed, or hallucinated. Python computes the reward.

## Files

- `prompts.py`: offline criteria extraction prompt and online judge prompt.
- `prompt_templates/offline_criteria_prompt.txt`: standalone offline prompt copy.
- `prompt_templates/online_judge_prompt.txt`: standalone online judge prompt copy.
- `reward.py`: criteria ID normalization and reward computation.
- `qwen_text.py`: text-only generation wrapper for local Qwen or Qwen-VL checkpoints.
- `extract_motion_criteria.py`: batch offline criteria extraction CLI.
- `judge_motion_caption.py`: batch online-style judge CLI for candidate captions.

## Offline Criteria Extraction

Example on the remote machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_envs/mllm/bin/python \
  -m rubric_rl.extract_motion_criteria \
  --model /wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_models/Qwen__Qwen3.6-27B \
  --input /path/to/gt_descriptions.jsonl \
  --output /path/to/criteria.jsonl \
  --gt-key gt_description \
  --max-memory "0:38GiB,1:38GiB,cpu:120GiB" \
  --keep-raw
```

Input rows should contain a sample ID plus a GT description field, for example:

```json
{"sample_id":"motion_001","gt_description":"Overall Action Overview: ..."}
```

The output contains:

```json
{"sample_id":"motion_001","criteria":{"mode":"temporal_caption","global_facts":[{"id":"g1","criterion":"..."}],"segments":[...]}}
```

## Online Judge / Reward Test

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_envs/mllm/bin/python \
  -m rubric_rl.judge_motion_caption \
  --model /wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_models/Qwen__Qwen3.6-27B \
  --criteria /path/to/criteria.jsonl \
  --candidates /path/to/candidates.jsonl \
  --output /path/to/rewarded_candidates.jsonl \
  --candidate-key candidate \
  --max-memory "0:38GiB,1:38GiB,cpu:120GiB" \
  --keep-raw
```

Candidate rows:

```json
{"sample_id":"motion_001","candidate":"[00.00-00.80] The person starts still..."}
```

## Reward Logic

The online judge returns IDs:

```json
{
  "global_action": {
    "satisfied_ids": ["g1"],
    "missed_ids": []
  },
  "segment_level_caption_facts": {
    "present_ids": ["s1_f1"],
    "aligned_ids": ["s1_f1"],
    "missed_ids": [],
    "misplaced_ids": []
  },
  "temporal_order_score": 15,
  "language_conciseness_score": 10,
  "hallucination_penalty": 0,
  "hallucinations": []
}
```

Then `reward.py` computes:

```text
global_score = 20 * len(satisfied_ids) / total_global_facts

segment_score = 55 * (
  0.3 * len(present_ids) / total_segment_facts
  + 0.7 * len(aligned_ids) / total_segment_facts
)

raw_score =
  global_score
+ segment_score
+ temporal_order_score
+ language_conciseness_score
+ hallucination_penalty

reward = clamp(raw_score, 0, 100) / 100
```

This separates "the motion fact is mentioned" from "the motion fact is in the correct time segment." A fully reversed caption can still receive some segment-present credit, but it loses aligned and temporal-order credit.
