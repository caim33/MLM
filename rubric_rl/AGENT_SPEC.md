# Agent Spec: Temporal Motion Rubric RL

This document tells another AI agent how to generate the same kind of Rubric-RL assets for a different motion-description dataset.

## Goal

Build a two-step Rubric-RL reward pipeline for long, time-segmented motion descriptions:

1. Offline criteria extraction:
   Dense GT motion description -> reusable criteria JSON.
2. Online judging:
   Candidate caption + criteria JSON -> satisfied/missed criterion IDs -> deterministic reward.

The model judge must not invent the scoring scheme. The scoring scheme is fixed.

## Stage 1 Target

This spec is for Stage 1 temporal caption mode:

- Output should be time-segmented captions.
- Captions should be semantically correct but not dense anatomical reports.
- Exact angles, distances, frame counts, and numeric thresholds are not required.
- Left/right may be unreliable, so rewrite side-specific facts into side-neutral phrasing when possible.
- The GT is the only reference and may not be exhaustive.

## Offline Criteria Extraction

Given a dense GT description, extract:

```json
{
  "mode": "temporal_caption",
  "global_facts": [
    "..."
  ],
  "segments": [
    {
      "time": "...",
      "facts": [
        "..."
      ]
    }
  ],
  "motion_dynamics": [
    "..."
  ],
  "temporal_phases": [
    "..."
  ],
  "negative_criteria": [
    {
      "criterion": "...",
      "type": "unsupported_detail | contradiction | unrelated_motion"
    }
  ]
}
```

Rules:

- Keep global facts focused on motion/action only.
- Do not include clothing, identity, mask, room, deck, background, surface, or environment facts unless they directly change the motion.
- If the surface affects motion, write the effect, such as "the feet slide or glide," not the scene.
- Preserve explicit GT time ranges. If the GT has 6 time segments, output 6 segment entries in the same order.
- Each segment should contain 3-5 key caption-level facts.
- Use 3 facts for simple/static segments and 4-5 for complex segments.
- Each fact must be atomic: one action, posture, or motion property.
- Convert dense anatomical details into natural caption-level facts.
- Avoid exact values. Example: "hip flexion reaches 90 degrees" -> "one knee lifts high toward the waist."
- If left/right may be unreliable, rewrite "right foot" / "left knee" into "one foot" / "one knee" when this does not change the action.
- Extract 3-5 coarse temporal phases only. These are for broad order, not detailed scoring.
- Negative criteria should catch serious hallucinations: extra actors, unsupported props, unrelated activities, fabricated body motions, or temporal contradictions.
- Negative criteria should not punish missing clothing, scene, harmless paraphrases, or plausible caption wording.

## Add Stable IDs

Do not rely on the judge model to create IDs. Add IDs in code after offline extraction.

Use:

```text
global facts: g1, g2, g3...
segment facts: s1_f1, s1_f2, s2_f1, s2_f2...
negative criteria: n1, n2, n3...
```

Example:

```json
{
  "global_facts": [
    {
      "id": "g1",
      "criterion": "The subject performs a seated hip mobility exercise."
    }
  ],
  "segments": [
    {
      "time": "00.00-01.50",
      "facts": [
        {
          "id": "s1_f1",
          "criterion": "The person sits upright."
        }
      ]
    }
  ]
}
```

Why IDs matter:

- LLM judges often miscount.
- Returning IDs is more stable than returning counts.
- Python should compute all final scores.

## Online Judge Output

The online judge receives:

- criteria JSON with IDs
- candidate motion caption

It must return only valid JSON:

```json
{
  "global_action": {
    "satisfied_ids": [],
    "missed_ids": []
  },
  "segment_level_caption_facts": {
    "present_ids": [],
    "aligned_ids": [],
    "missed_ids": [],
    "misplaced_ids": []
  },
  "temporal_order_score": 0,
  "language_conciseness_score": 0,
  "hallucination_penalty": 0,
  "hallucinations": []
}
```

Definitions:

- `satisfied_ids`: global facts expressed by the candidate.
- `present_ids`: segment facts whose action content appears anywhere in the candidate, even if in the wrong segment.
- `aligned_ids`: segment facts that appear in the correct segment, nearby segment, or reasonable merged segment.
- `missed_ids`: facts that do not appear anywhere.
- `misplaced_ids`: facts that appear but in the wrong temporal phase.

Consistency rules:

- `aligned_ids` must be a subset of `present_ids`.
- `missed_ids` and `present_ids` should not overlap.
- `misplaced_ids` is usually `present_ids - aligned_ids` when the only issue is temporal placement.
- The judge must return only IDs that exist in the criteria.
- The judge must not return counts or compute reward.

## Fixed Scoring

Weights:

```text
global_action: 20
segment_level_caption_facts: 55
temporal_order: 15
language_conciseness: 0-10
hallucination_penalty: 0 to -20
```

Segment score separates "mentioned" from "correctly placed":

```text
segment_score =
55 * (
  0.3 * len(present_ids) / total_segment_facts
  + 0.7 * len(aligned_ids) / total_segment_facts
)
```

This prevents a fully reversed but content-complete caption from receiving zero segment credit.

Temporal order score must be one of:

```text
15: major phases are in correct order
10: mostly correct, one phase missing/merged/slightly misplaced
5: some actions correct, temporal progression weak/confusing
0: order absent, reversed, or mostly wrong
```

Language score:

```text
0-10 integer
```

Judge clarity, grammar, caption style, temporal organization, and repetition.

Hallucination penalty must be one of:

```text
0, -5, -10, -15, -20
```

Use hallucination penalty for unsupported additions or contradictions, not for omissions. Omission is handled by missed IDs.

## Post-Processing

After the online judge returns JSON:

1. Remove invalid IDs.
2. Force `aligned_ids = aligned_ids ∩ present_ids`.
3. Compute totals from the criteria, not from the judge.
4. Snap temporal score to one of `0, 5, 10, 15`.
5. Clamp language score to `0..10`.
6. Snap hallucination penalty to one of `0, -5, -10, -15, -20`.
7. Compute:

```text
global_score = 20 * len(satisfied_ids) / total_global_facts

segment_score =
55 * (
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

## Expected Behavior

- Good concise caption:
  - high global
  - high present
  - high aligned
  - high temporal
  - no hallucination

- Vague caption:
  - some global
  - low present/aligned
  - no hallucination if it does not fabricate

- Fully reversed but content-complete caption:
  - high global
  - high present
  - low/zero aligned
  - temporal order 0
  - no hallucination unless it adds unsupported content

- Severe hallucination:
  - low global
  - low present/aligned
  - hallucination penalty -15 or -20

## Existing Files

In this repo, the implementation lives in:

```text
rubric_rl/prompts.py
rubric_rl/prompt_templates/offline_criteria_prompt.txt
rubric_rl/prompt_templates/online_judge_prompt.txt
rubric_rl/reward.py
rubric_rl/extract_motion_criteria.py
rubric_rl/judge_motion_caption.py
```

