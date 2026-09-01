# Rubric V2 Current Prompts And Postprocessing

This document records the current Rubric-RL V2 setup after the Qwen3.6-27B one-shot test.

## Current Default

The current default offline path is a one-shot whole-sample rubric extraction:

```text
python -m rubric_rl.extract_motion_criteria_v2 \
  --max-new-tokens 65535 \
  --keep-raw
```

`extract_motion_criteria_v2.py` calls `build_offline_messages()`, which uses `OFFLINE_SYSTEM_PROMPT` and `OFFLINE_USER_TEMPLATE` from `rubric_rl/prompts_v2.py`.

The older global/segment/failed-segment templates are still present as legacy fallback tools, but they are not the current default.

## Rubric Weights

These weights are deterministic postprocessing configuration. They are intentionally not included in the current offline prompt, because offline extraction should produce criteria only and should not behave like a judge.

```text
final_motion_answer: 70
  global_activity: 5
  basic_action_facts / description: 25
  temporal_structure: 10
  body_configuration: 10
  motion-grounded details: 20
    numeric_kinematics: 10
    laterality: 5
    camera_relative_orientation: 5

reasoning_process: 20
  source_separation: 4
  conflict_detection: 4
  trust_hierarchy_application: 6
  numeric_evidence_use: 4
  reasoning_answer_consistency: 2

language_format: 10
hallucination_or_source_contradiction_penalty: 0 to -25
```

## Trust Hierarchy

```text
1. Motion is authoritative for numeric kinematics, laterality, camera/screen orientation, and numerically supported body configuration.
2. Video/final summary is preferred for high-level activity semantics, scene/context, objects, broad temporal/action semantics, and person-object interactions motion cannot measure.
3. Use motion numbers to adjudicate posture, grounded/sitting/kneeling/contact-like claims, and conflicts with video.
4. Do not trust motion narrative verbs unless supported by numeric evidence or video semantics.
```

## Offline Prompt

System:

```text
You are a source-aware rubric extractor for video+motion motion reports.

Extract reusable criteria for evaluating both a model's final motion answer and its visible reasoning process.
Follow the source trust hierarchy exactly.
Do not judge a candidate answer.
Do not create a reward formula.
Return only valid JSON.
```

User template:

```text
Input report:
{gt_description}

Task:
Extract one complete source-aware rubric for the entire report in a single JSON object.
Do not judge a candidate answer and do not compute reward.

Extraction fields to cover:
- final-answer criteria:
  - global_activity
  - basic_action_facts / description
  - temporal_structure
  - body_configuration
  - motion-grounded details:
    - numeric_kinematics
    - laterality
    - camera_relative_orientation
- visible reasoning criteria:
  - source_separation
  - conflict_detection
  - trust_hierarchy_application
  - numeric_evidence_use
  - reasoning_answer_consistency
- negative_criteria for hallucinations, unsupported claims, source contradictions, wrong laterality, wrong orientation, and wrong numeric values.

Offline extraction must only produce reusable criteria. Do not assign points, scores, rewards, or penalties.

Source trust hierarchy:
1. Motion is authoritative for numeric kinematics, laterality, camera/screen orientation, and numerically supported body configuration.
2. Video/final summary is preferred for high-level activity semantics, scene/context, objects, broad temporal/action semantics, and person-object interactions motion cannot measure.
3. Use motion numbers to adjudicate posture, grounded/sitting/kneeling/contact-like claims, and conflicts with video.
4. Do not trust motion narrative verbs unless supported by numeric evidence or video semantics.

Extraction rules:
1. Preserve every explicit segment time range and order.
2. Use each segment answer for final-answer criteria.
3. Use each segment think for reasoning criteria and source-trust decisions.
4. Keep important explicit numeric values, but keep the rubric compact.
5. Do not invent numeric values.
6. Numeric criteria must be measurement-neutral. If narrative wording conflicts with the numbers, trust the numbers and rewrite the criterion neutrally.
   - Prefer wording like "right knee angle is 54-62 degrees" or "head height is +0.15m to +0.57m".
   - Do not use qualitative labels such as straight, extended, deep, upright, high, planted, or braced unless the numeric range directly supports that label.
   - Distinguish limb location from joint straightness: "leg behind" is not the same as "knee straight".
7. Numeric criteria contain only `target_range` and `unit`. The extractor must
   not emit tolerance, score, point, reward, margin, or penalty fields.
8. If a value is a range, use [low, high]. If it is a single value, use [value, value].
9. Per segment, extract exactly:
   - 2 basic_action_facts
   - 1 body_configuration
   - 3 numeric_kinematics, choosing the most important explicit numbers
   - 1 laterality criterion if left/right appears, otherwise []
   - 1 camera_relative_orientation criterion if camera/screen orientation appears, otherwise []
   - 2 reasoning_criteria from the think text
   - rejected_claims: []
10. Extract exactly:
   - 1 global_activity
   - 5 to 7 temporal_phases
   - 6 negative_criteria
11. Keep all criterion strings short. Do not output evidence fields. Do not output markdown. Return only JSON.

Return only valid JSON with this schema:

{
  "mode": "source_aware_reasoning_motion_rubric_v2",
  "global_activity": [
    {"criterion": "high-level activity criterion", "source": "video+motion"}
  ],
  "segments": [
    {
      "time": "...",
      "basic_action_facts": [
        {"criterion": "...", "source": "motion | video+motion"}
      ],
      "body_configuration": [
        {"criterion": "...", "source": "motion"}
      ],
      "numeric_kinematics": [
        {
          "criterion": "...",
          "quantity": "...",
          "body_part": "...",
          "target_range": [0, 0],
          "unit": "degrees | m | s",
          "source": "motion"
        }
      ],
      "laterality": [
        {"criterion": "...", "source": "motion"}
      ],
      "camera_relative_orientation": [
        {"criterion": "...", "source": "motion"}
      ],
      "reasoning_criteria": [
        {
          "criterion": "...",
          "type": "source_separation | conflict_detection | trust_hierarchy_application | numeric_evidence_use | reasoning_answer_consistency",
          "source": "think"
        }
      ],
      "rejected_claims": []
    }
  ],
  "temporal_phases": ["..."],
  "negative_criteria": [
    {
      "criterion": "...",
      "type": "unsupported_detail | contradiction | wrong_laterality | wrong_orientation | numeric_contradiction | unrelated_motion",
      "source_of_truth": "motion | video | video+motion"
    }
  ]
}
```

## Online Prompt

System:

```text
You are a strict source-aware judge for video+motion motion reports.

Judge final answer correctness, numeric closeness, visible reasoning quality, source-trust compliance, and hallucinations.
Do not compute the final reward.
Return only valid JSON.
```

User template:

```text
Criteria:
{criteria_json}

Candidate response:
{candidate_response}

Task:
Evaluate the candidate against the criteria.

Rules:
1. Evaluate visible final answer content separately from visible reasoning or think content.
2. Reasoning exists only inside exactly one non-empty, lowercase `<think>...</think>` block. If it is absent or malformed, all reasoning criteria are missed regardless of judge output.
3. For final answer facts, accept semantic paraphrases.
4. For basic_action_facts and body_configuration:
   - present_ids means the fact appears anywhere.
   - aligned_ids means the fact appears in the correct segment, nearby segment, or a reasonable merged segment.
   - Generic routine-level phrases such as "sometimes upright", "does lunges", or "arms reach" are not enough for aligned_ids unless they are tied to the correct time range or ordered phase.
   - Do not mark every segment correct from one broad summary sentence.
5. For numeric_kinematics, exact equality is not required:
   - strict_value_match_ids: candidate value is within target_range plus strict_tolerance.
   - loose_value_match_ids: candidate value is outside strict tolerance but within loose_tolerance.
   - semantic_present_ids: the same segment/phase mentions the correct body part and measured quantity or directly implied numeric state, even if no close numeric value is given.
   - wrong_value_ids: candidate gives a value outside loose_tolerance or reverses the numeric meaning.
    - Every claimed numeric match/wrong ID has one `observed_values` row whose
      `candidate_text` is an exact candidate substring containing the value,
      unit, and body part or quantity. Python verifies this independently.
    - The five numeric buckets are pairwise disjoint and exhaustive. Quoted,
      negated, hypothetical, uncertain, meta-level, or immediately corrected
      evidence receives no strict, loose, or semantic credit.
6. Do not assign semantic_present_ids from generic statements like "the torso is sometimes upright/folded" or "the legs alternate".
7. Missing numeric values are omissions, not hallucinations.
8. Wrong numeric values, wrong laterality, and wrong camera-relative orientation are contradictions.
9. Distinguish limb location from joint straightness: "leg extends behind" can describe location and should not be treated as "knee is straight" unless the candidate explicitly claims a straight knee.
10. For laterality, judge left/right body parts using the motion-grounded criteria.
11. For camera_relative_orientation, judge orientation relative to the camera or screen, not anatomical left/right.
12. If criterion wording and target_range conflict, trust target_range.
13. Use only IDs that exist in the criteria. Never invent IDs.
14. Evaluate `negative_criteria` and return only IDs actually triggered by the candidate.
15. Do not return counts or compute reward.

Temporal structure score must be one of 0, 3, 6, 10:
- 10: major phases and segment order are correct.
- 6: mostly correct with one missing, merged, or slightly misplaced phase.
- 3: some actions are correct but ordering is weak or confusing.
- 0: order is absent, reversed, or mostly wrong.

Language format score is an integer from 0 to 10.
Hallucination/source contradiction penalty must be one of 0, -5, -10, -15, -20, -25.

Return only valid JSON:

{
  "final_motion_answer": {
    "global_activity": {"satisfied_ids": [], "missed_ids": []},
    "basic_action_facts": {
      "present_ids": [],
      "aligned_ids": [],
      "missed_ids": [],
      "misplaced_ids": []
    },
    "body_configuration": {
      "present_ids": [],
      "aligned_ids": [],
      "missed_ids": [],
      "misplaced_ids": []
    },
    "numeric_kinematics": {
      "semantic_present_ids": [],
      "strict_value_match_ids": [],
      "loose_value_match_ids": [],
      "wrong_value_ids": [],
      "missed_ids": [],
      "observed_values": [
        {"id": "...", "candidate_value": 0, "unit": "...", "candidate_text": "..."}
      ]
    },
    "laterality": {"correct_ids": [], "wrong_ids": [], "missed_ids": []},
    "camera_relative_orientation": {"correct_ids": [], "wrong_ids": [], "missed_ids": []},
    "temporal_structure_score": 0
  },
  "reasoning_process": {
    "satisfied_ids": [],
    "missed_ids": [],
    "contradicted_ids": []
  },
  "negative_criteria": {"triggered_ids": []},
  "language_format_score": 0,
  "hallucination_or_source_contradiction_penalty": 0,
  "hallucinations": []
}
```

## Deterministic Postprocessing

Qwen returns IDs and coarse scalar fields only. Python computes reward in `rubric_rl/reward_v2.py`.

Main normalization:

```text
ensure_criteria_ids(criteria)
- assigns stable IDs:
  - global_activity: g{index}
  - basic_action_facts: s{segment}_a{index}
  - body_configuration: s{segment}_b{index}
  - numeric_kinematics: s{segment}_n{index}
  - laterality: s{segment}_l{index}
  - camera_relative_orientation: s{segment}_o{index}
  - reasoning_criteria: s{segment}_r{index}
  - negative_criteria: neg{index}
- rejects empty required categories, invalid counts/fields, duplicate IDs, and
  IDs placed in another category's namespace.
- optional laterality/orientation categories may be empty, but empty categories
  score zero rather than receiving free credit.
- rejects unknown judge IDs at the online boundary; deterministic scoring logs
  and removes unknown IDs if called directly.
```

Numeric matching:

```text
strict numeric match: target_range +/- strict_tolerance
loose numeric match: target_range +/- loose_tolerance
semantic-only numeric weight: 0.25

degrees tolerance: strict +/-10, loose +/-20
m tolerance: strict +/-0.05, loose +/-0.10
s tolerance: strict +/-0.20, loose +/-0.50
```

`observed_values` is not trusted by itself. Python requires `candidate_text` to
be an exact substring of the raw candidate, independently parses the reported
number, verifies the unit and body-part/quantity association, and then applies
the fixed tolerance. Missing candidate text, fabricated values, mismatched
units, and unverifiable mappings receive no numeric credit. A reported wrong
value never receives semantic-only credit.

Scoring formula:

```text
global_activity_score = 5 * satisfied / total

basic_action_score =
  25 * (0.3 * present / total + 0.7 * aligned / total)

temporal_structure_score =
  nearest of {0, 3, 6, 10}

body_configuration_score =
  10 * (0.3 * present / total + 0.7 * aligned / total)

numeric_kinematics_score =
  10 * sum(per_numeric_item_credit) / total

per_numeric_item_credit:
  strict = 1.0
  loose = 0.7
  semantic-only = 0.25
  missed/wrong = 0.0

laterality_score = 5 * correct / total
camera_orientation_score = 5 * correct / total

reasoning_score:
  source_separation: 4
  conflict_detection: 4
  trust_hierarchy_application: 6
  numeric_evidence_use: 4
  reasoning_answer_consistency: 2

language_score = clamp 0..10
penalty = the more negative of:
  - the strictly allowed judge scalar
  - deterministic penalties derived from triggered negative criteria,
    wrong numeric/laterality/orientation IDs, reasoning contradictions, and
    correct/wrong or satisfied/contradicted conflicts

total_score = clamp(
  global + basic + temporal + body + numeric + laterality
  + camera_orientation + reasoning + language + penalty,
  0,
  100
)

reward = total_score / 100
```

Reasoning is normalized only over reasoning criterion types that actually exist.
All categorical alternatives are validated as pairwise-disjoint, exhaustive
partitions. A conflicting or incomplete judgment fails closed rather than being
silently resolved. Every accepted judgment is also bound to its sample ID,
canonical criteria SHA-256, candidate SHA-256, and nonce.

## Historical Qwen3.6-27B smoke test (non-production)

The following files and scores are retained only as implementation history.
They predate the frozen strict schema and the mandatory fresh-finetune global
phase barrier. They must not be copied into a current evaluation table, used as
model accuracy evidence, or treated as validation of the current reward code.

Files:

```text
criteria:
data/rubric_rl/sample_summary_qwen36_27b_v2_oneshot_65535_strict_criteria.jsonl

scores:
data/rubric_rl/sample_summary_qwen36_27b_v2_oneshot_65535_strict_score_test_rewarded.jsonl
```

Scores:

```text
good_with_reasoning: 97.7778
accurate_no_reasoning: 80.0000
vague_high_level: 19.0000
wrong_motion_details: 0.0000
```

At the time, this smoke run was used to exercise the then-current prompt. A new
schema-valid smoke run and fresh batch artifacts are required before making any
claim about the current implementation.
