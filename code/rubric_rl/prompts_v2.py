"""Prompt templates for source-aware, reasoning-aware motion rubric RL."""

from __future__ import annotations

import json
from typing import Any, Dict


OFFLINE_SYSTEM_PROMPT = """You are a source-aware rubric extractor for video+motion motion reports.

Extract reusable criteria for evaluating both a model's final motion answer and its visible reasoning process.
Follow the source trust hierarchy exactly.
Do not judge a candidate answer.
Do not create a reward formula.
Return only valid JSON."""


OFFLINE_USER_TEMPLATE = """Input report:
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

Offline extraction must only produce reusable motion and reasoning criteria.
Do not assign points, scores, rewards, penalties, or numeric tolerance metadata.
Numeric tolerance is not part of the ground-truth rubric. It will be applied later by deterministic reward post-processing.

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
   - Prefer wording like "right knee angle is 54 to 62 degrees" or "head height is +0.15m to +0.57m".
   - Do not use qualitative labels such as straight, extended, deep, upright, high, planted, or braced inside numeric_kinematics unless the numeric range directly supports that label.
   - Distinguish limb location from joint straightness: "leg behind" is not the same as "knee straight".
7. Numeric criteria must include target_range and unit only. Do not output strict_tolerance, loose_tolerance, tolerance, margin, scoring, or closeness fields.
8. If a value is a range, use [low, high]. If it is a single value, use [value, value].
9. body_configuration may use qualitative posture terms only when they are directly supported by motion numbers or unambiguous motion facts.
10. laterality means anatomical left/right body-side identity. Use motion as the source of truth.
11. camera_relative_orientation means facing direction relative to the camera/screen, such as "faces the camera", "angled toward the camera's right", or "turns away from the camera". Use motion as the source of truth when available.
12. basic_action_facts should capture core visible/motion action facts and description, but should not duplicate every numeric detail.
13. reasoning_criteria should describe what a candidate's visible reasoning should do, not repeat final-answer facts.
14. Per segment, extract exactly:
   - 2 basic_action_facts
   - 1 body_configuration
   - 3 numeric_kinematics, choosing the most important explicit numbers
   - 1 laterality criterion if left/right appears, otherwise []
   - 1 camera_relative_orientation criterion if camera/screen orientation appears, otherwise []
   - 2 reasoning_criteria from the think text
   - rejected_claims: []
15. Extract exactly:
   - 1 global_activity
   - 5 to 7 temporal_phases
   - 6 negative_criteria
16. Keep all criterion strings short. Do not output evidence fields. Do not output markdown. Return only JSON.
17. Do not output id fields. Stable IDs will be added by post-processing.

Return only valid JSON with this schema:

{{
  "mode": "source_aware_reasoning_motion_rubric_v2",
  "global_activity": [
    {{"criterion": "high-level activity criterion", "source": "video | motion | video+motion"}}
  ],
  "segments": [
    {{
      "time": "...",
      "basic_action_facts": [
        {{"criterion": "...", "source": "video | motion | video+motion"}}
      ],
      "body_configuration": [
        {{"criterion": "...", "source": "motion"}}
      ],
      "numeric_kinematics": [
        {{
          "criterion": "...",
          "quantity": "...",
          "body_part": "...",
          "target_range": [0, 0],
          "unit": "degrees | m | s",
          "source": "motion"
        }}
      ],
      "laterality": [
        {{"criterion": "...", "source": "motion"}}
      ],
      "camera_relative_orientation": [
        {{"criterion": "...", "source": "motion | video+motion"}}
      ],
      "reasoning_criteria": [
        {{
          "criterion": "...",
          "type": "source_separation | conflict_detection | trust_hierarchy_application | numeric_evidence_use | reasoning_answer_consistency",
          "source": "think"
        }}
      ],
      "rejected_claims": []
    }}
  ],
  "temporal_phases": ["..."],
  "negative_criteria": [
    {{
      "criterion": "...",
      "type": "unsupported_detail | contradiction | wrong_laterality | wrong_orientation | numeric_contradiction | unrelated_motion",
      "source_of_truth": "motion | video | video+motion"
    }}
  ]
}}"""


ONLINE_SYSTEM_PROMPT = """You are a strict source-aware judge for video+motion motion reports.

Judge final answer correctness, numeric closeness, visible reasoning quality, source-trust compliance, and hallucinations.
Do not compute the final reward.
Return only valid JSON."""


ONLINE_USER_TEMPLATE = """Criteria:
{criteria_json}

Candidate response:
{candidate_response}

Task:
Evaluate the candidate against the criteria.

Rules:
1. Evaluate visible final answer content separately from visible reasoning or think content.
2. Reasoning exists only inside exactly one non-empty, lowercase <think>...</think> block. If that structure is absent or malformed, all reasoning criteria are missed.
3. For final answer facts, accept semantic paraphrases.
4. For basic_action_facts and body_configuration:
   - present_ids means the fact appears anywhere.
   - aligned_ids means the fact appears in the correct segment, nearby segment, or a reasonable merged segment.
   - Generic routine-level phrases such as "sometimes upright", "does lunges", or "arms reach" are not enough for aligned_ids unless they are tied to the correct time range or ordered phase.
   - Do not mark every segment correct from one broad summary sentence.
5. For numeric_kinematics, exact equality is not required:
   - Use these deterministic tolerance rules for closeness:
     degrees: strict +/-10, loose +/-20.
     m: strict +/-0.05, loose +/-0.10.
     s: strict +/-0.20, loose +/-0.50.
   - strict_value_match_ids: candidate value is within target_range plus the strict tolerance for its unit.
   - loose_value_match_ids: candidate value is outside strict tolerance but within the loose tolerance for its unit.
   - semantic_present_ids: the same segment/phase mentions the correct body part and measured quantity or directly implied numeric state, even if no close numeric value is given.
   - wrong_value_ids: candidate gives a value outside the loose tolerance for its unit or reverses the numeric meaning.
   - These five numeric ID lists are pairwise disjoint and together contain every numeric criterion ID. semantic_present_ids is only for a measurement mentioned without a numeric value.
   - Quoted, negated, hypothetical, uncertain, meta-level, or immediately corrected values are not strict or loose evidence.
   - Every numeric ID placed in a value-match or wrong-value list must have exactly one observed_values row.
   - candidate_text must be an exact verbatim substring of the candidate and must contain the reported number, unit, body part or measured quantity. Python independently verifies it; unverifiable rows receive no numeric credit.
6. Do not assign semantic_present_ids from generic statements like "the torso is sometimes upright/folded" or "the legs alternate".
7. Missing numeric values are omissions, not hallucinations.
8. Wrong numeric values, wrong laterality, and wrong camera-relative orientation are contradictions.
9. Distinguish limb location from joint straightness: "leg extends behind" can describe location and should not be treated as "knee is straight" unless the candidate explicitly claims a straight knee.
10. For laterality, judge left/right body parts using the motion-grounded criteria.
11. For camera_relative_orientation, judge orientation relative to the camera or screen, not anatomical left/right.
12. If criterion wording and target_range conflict, trust target_range.
13. Use only IDs that exist in the criteria. Never invent IDs.
14. Evaluate negative_criteria and return only IDs that the candidate actually triggers.
15. Do not return counts or compute reward.
16. Every categorical set is an exact partition: global satisfied/missed; basic and body present/missed plus aligned/misplaced over present; laterality correct/wrong/missed; orientation correct/wrong/missed; reasoning satisfied/missed/contradicted. Never repeat an ID across alternatives and never omit an eligible ID.
17. Do not output a binding field. The trusted host adds the sample/criteria/candidate/nonce binding after local generation.

Temporal structure score must be one of 0, 3, 6, 10:
- 10: major phases and segment order are correct.
- 6: mostly correct with one missing, merged, or slightly misplaced phase.
- 3: some actions are correct but ordering is weak or confusing.
- 0: order is absent, reversed, or mostly wrong.

Language format score is an integer from 0 to 10.
Hallucination/source contradiction penalty must be one of 0, -5, -10, -15, -20, -25.

Return only valid JSON:

{{
  "final_motion_answer": {{
    "global_activity": {{"satisfied_ids": [], "missed_ids": []}},
    "basic_action_facts": {{
      "present_ids": [],
      "aligned_ids": [],
      "missed_ids": [],
      "misplaced_ids": []
    }},
    "body_configuration": {{
      "present_ids": [],
      "aligned_ids": [],
      "missed_ids": [],
      "misplaced_ids": []
    }},
    "numeric_kinematics": {{
      "semantic_present_ids": [],
      "strict_value_match_ids": [],
      "loose_value_match_ids": [],
      "wrong_value_ids": [],
      "missed_ids": [],
      "observed_values": [
        {{"id": "...", "candidate_value": 0, "unit": "...", "candidate_text": "..."}}
      ]
    }},
    "laterality": {{"correct_ids": [], "wrong_ids": [], "missed_ids": []}},
    "camera_relative_orientation": {{"correct_ids": [], "wrong_ids": [], "missed_ids": []}},
    "temporal_structure_score": 0
  }},
  "reasoning_process": {{
    "satisfied_ids": [],
    "missed_ids": [],
    "contradicted_ids": []
  }},
  "negative_criteria": {{"triggered_ids": []}},
  "language_format_score": 0,
  "hallucination_or_source_contradiction_penalty": 0,
  "hallucinations": []
}}"""


def build_offline_messages(gt_description: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": OFFLINE_SYSTEM_PROMPT},
        {"role": "user", "content": OFFLINE_USER_TEMPLATE.format(gt_description=gt_description)},
    ]


OFFLINE_GLOBAL_USER_TEMPLATE = """Input report summary:
{summary}

Final report answer:
{final_answer}

Segment timeline:
{segment_timeline}

Task:
Extract only the global criteria. Do not extract per-segment body or numeric criteria here.
Do not assign points, scores, rewards, or penalties.

Rubric fields to extract:
- global_activity: 1 to 2 criteria.
- temporal_phases: concise ordered phase labels.
- negative_criteria: 6 to 10 criteria for contradictions, wrong laterality, wrong orientation, wrong numeric values, unsupported props/actions, extra actors, or unrelated activities.

Source trust hierarchy:
1. Always trust motion for laterality, camera-relative orientation, joint angles, distances, heights, and motion-supported body configuration.
2. Prefer video for high-level activity semantics and scene/context.
3. Do not directly trust motion-generated narrative verbs unless supported by numeric evidence or video semantics.

Keep every criterion and evidence string concise.
Return only valid JSON:

{{
  "mode": "source_aware_reasoning_motion_rubric_v2",
  "global_activity": [
    {{"criterion": "...", "source": "video | final_answer | video+motion", "evidence": "..."}}
  ],
  "temporal_phases": ["..."],
  "negative_criteria": [
    {{
      "criterion": "...",
      "type": "unsupported_detail | contradiction | wrong_laterality | wrong_orientation | numeric_contradiction | unrelated_motion",
      "source_of_truth": "motion | video | video+motion",
      "evidence": "..."
    }}
  ]
}}"""


OFFLINE_SEGMENT_USER_TEMPLATE = """Input report summary:
{summary}

Segment:
time: {time_range}
think: {think}
answer: {answer}

Task:
Extract criteria only for this segment. Do not assign points, scores, rewards, or penalties.

Segment fields to cover:
- basic_action_facts / description
- body_configuration
- motion-grounded details:
  - numeric_kinematics
  - laterality
  - camera_relative_orientation
- reasoning_process

Source trust hierarchy:
1. Always trust motion for laterality, camera-relative orientation, joint angles, distances, heights, and body configuration when numerically supported.
2. Prefer video for high-level activity semantics, scene/context, objects, broad action semantics, temporal segmentation, and person-object interactions that motion cannot measure.
3. Use motion numbers to adjudicate body configuration, grounded/sitting/kneeling/contact-like posture, and conflicts with video.
4. Do not directly trust motion-generated narrative verbs such as swinging, stepping, marching, rotating, bracing, pushing, or lunging unless supported by numeric evidence or video semantics.

Extraction rules:
1. Use the answer for final-answer criteria.
2. Use the think field for reasoning criteria, source conflicts, rejected claims, and source-trust decisions.
3. Keep motion numeric values. Do not invent numeric values.
4. Numeric criteria must include target_range and unit only. Do not output strict_tolerance, loose_tolerance, tolerance, margin, scoring, or closeness fields.
5. If a value is a range, use that range. If it is a single value, use [value, value].
6. basic_action_facts: 2 to 4 concise criteria.
7. body_configuration: 1 to 3 concise motion-grounded criteria.
8. numeric_kinematics: 2 to 5 criteria, prioritizing torso tilt, major hip/knee angles, wrist/head heights, and arm angles.
9. laterality: 1 to 2 criteria if left/right appears.
10. camera_relative_orientation: exactly 1 criterion if camera/screen orientation appears.
11. reasoning_criteria: 1 to 3 criteria if think explains conflict/source-trust/numeric use.
12. rejected_claims: 0 to 2 concise rejected claims.
13. Keep every criterion and evidence string short.

Return only valid JSON:

{{
  "time": "{time_range}",
  "basic_action_facts": [
    {{"criterion": "...", "source": "video | motion | video+motion", "evidence": "..."}}
  ],
  "body_configuration": [
    {{"criterion": "...", "source": "motion", "evidence": "..."}}
  ],
  "numeric_kinematics": [
    {{
      "criterion": "...",
      "quantity": "...",
      "body_part": "...",
      "target_range": [0, 0],
      "unit": "degrees | m | s",
      "source": "motion",
      "evidence": "..."
    }}
  ],
  "laterality": [
    {{"criterion": "...", "source": "motion", "evidence": "..."}}
  ],
  "camera_relative_orientation": [
    {{"criterion": "...", "source": "motion", "evidence": "..."}}
  ],
  "reasoning_criteria": [
    {{
      "criterion": "...",
      "type": "source_separation | conflict_detection | trust_hierarchy_application | numeric_evidence_use | reasoning_answer_consistency",
      "source": "think",
      "evidence": "..."
    }}
  ],
  "rejected_claims": [
    {{"claim": "...", "rejected_because": "...", "trusted_source": "motion | video"}}
  ]
}}"""


def build_offline_global_messages(
    *,
    summary: str,
    final_answer: str,
    segment_timeline: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": OFFLINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": OFFLINE_GLOBAL_USER_TEMPLATE.format(
                summary=summary,
                final_answer=final_answer,
                segment_timeline=segment_timeline,
            ),
        },
    ]


def build_offline_segment_messages(
    *,
    summary: str,
    time_range: str,
    think: str,
    answer: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": OFFLINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": OFFLINE_SEGMENT_USER_TEMPLATE.format(
                summary=summary,
                time_range=time_range,
                think=think,
                answer=answer,
            ),
        },
    ]


def build_online_messages(criteria: Dict[str, Any], candidate_response: str) -> list[dict[str, Any]]:
    criteria_json = json.dumps(criteria, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": ONLINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ONLINE_USER_TEMPLATE.format(
                criteria_json=criteria_json,
                candidate_response=candidate_response,
            ),
        },
    ]
