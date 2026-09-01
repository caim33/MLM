# QA01 Multiple-Choice Rubric RL Spec

This document defines a pilot Rubric-RL setup for one QA benchmark case.

The goal is to reward not only the final multiple-choice answer, but also the visible reasoning process inside `<think>...</think>`.

The key design choice is:

- Python deterministically parses answer correctness and format.
- The online judge evaluates reasoning criteria and contradictions only.
- The judge never computes final reward.
- A wrong final answer is capped at 50 points.

## Case

Source file:

```text
data/benchmark/text/QA/QA_500.jsonl
```

Case:

```text
benchmark_id: QA_000001
case_name: QA01
task: QA
question_type: orientation_transition
```

Question:

```text
How does the person's orientation relative to the camera change from the initial segment (00.00-01.30s) to the lunge segment (02.80-04.20s)?
```

Options:

```text
A. From facing directly at the camera to facing slightly left of the camera
B. From facing directly at the camera to facing slightly right of the camera
C. From facing slightly left of the camera to facing directly at the camera
D. Remains facing directly at the camera throughout both segments
```

Gold answer:

```text
<answer>A</answer>
```

## Fixed Scoring

```text
answer_correctness: 30
reasoning_quality: 55
format_compliance: 10
language_conciseness: 5
contradiction_penalty: 0 to -20
```

Important caps:

```text
wrong final answer: max 50
no valid final answer: max 20
no visible reasoning: max 45
```

The wrong-answer cap is intentionally set to 50 so a model cannot receive a high reward from fluent but answer-inconsistent reasoning.

## Offline Prompt

Standalone copy:

```text
rubric_rl/prompt_templates/qa_mc_offline_prompt.txt
```

The offline extractor receives one QA record and returns reusable criteria. It uses only:

- the QA text
- the four options
- the gold answer

It must not inspect video or motion files, and it must not invent hidden motion evidence.

### Offline System Prompt

```text
You are a rubric extractor for multiple-choice human-motion QA.

Your job is to convert one QA record into reusable evaluation criteria for rubric RL.
Use only the QA text, the four options, and the provided gold answer.
Do not inspect video or motion assets.
Do not invent hidden visual or motion facts beyond what is implied by the question, options, and gold answer.
Do not judge a candidate response.
Do not compute reward.
Return only valid JSON.
```

### Offline User Prompt Template

```text
QA record:
{QA_JSON}

Task:
Extract criteria for evaluating a model response to this multiple-choice QA item.

The candidate response is expected to contain:
1. exactly one visible reasoning block inside <think>...</think>
2. exactly one final answer block inside <answer>...</answer>
3. exactly one uppercase option letter A, B, C, or D inside <answer>

Fixed scoring is handled by deterministic post-processing, not by the extractor:
- answer_correctness: 30
- reasoning_quality: 55
- format_compliance: 10
- language_conciseness: 5
- contradiction_penalty: 0 to -20

Extraction rules:
1. Extract the question, options, correct option, and correct option text exactly.
2. Infer a compact question_type such as time_range, orientation_transition, posture_state, limb_motion, laterality, numeric_comparison, action_identification, temporal_comparison, or other_motion_qa.
3. Extract 4 to 7 reasoning_criteria.
4. Reasoning criteria must describe what a good visible <think> process should mention.
5. Reasoning criteria must be checkable from the candidate text.
6. Reasoning criteria should be derived from the question and correct option, not from external evidence.
7. For comparison questions, include criteria about comparing both referenced phases or time ranges.
8. For orientation questions, include criteria about camera-relative direction.
9. Add standard format_criteria for the <think>/<answer> protocol.
10. Add negative_criteria for serious contradictions, wrong option support, generic reasoning, unrelated motion, or claims incompatible with the correct option.
11. Do not output point values, counts, or reward calculations.
12. Do not output markdown.

Return only valid JSON with the qa_mc criteria schema.
```

## QA01 Evaluation Criteria

Standalone copy:

```text
rubric_rl/prompt_templates/qa01_eval_criteria.json
```

```json
{
  "mode": "qa_mc",
  "benchmark_id": "QA_000001",
  "case_name": "QA01",
  "task": "QA",
  "question_type": "orientation_transition",
  "question_focus": "camera-relative orientation change from the initial segment to the lunge segment",
  "question": "How does the person's orientation relative to the camera change from the initial segment (00.00-01.30s) to the lunge segment (02.80-04.20s)?",
  "options": {
    "A": "From facing directly at the camera to facing slightly left of the camera",
    "B": "From facing directly at the camera to facing slightly right of the camera",
    "C": "From facing slightly left of the camera to facing directly at the camera",
    "D": "Remains facing directly at the camera throughout both segments"
  },
  "correct_option": "A",
  "correct_option_text": "From facing directly at the camera to facing slightly left of the camera",
  "reasoning_criteria": [
    {
      "id": "r1",
      "criterion": "The reasoning identifies that the question asks about orientation relative to the camera.",
      "type": "question_focus"
    },
    {
      "id": "r2",
      "criterion": "The reasoning compares the initial segment 00.00-01.30s with the lunge segment 02.80-04.20s.",
      "type": "segment_reference"
    },
    {
      "id": "r3",
      "criterion": "The reasoning states that the person initially faces directly at the camera.",
      "type": "correct_option_fact"
    },
    {
      "id": "r4",
      "criterion": "The reasoning states that during the lunge segment the person faces slightly left of the camera.",
      "type": "correct_option_fact"
    },
    {
      "id": "r5",
      "criterion": "The reasoning describes the change as going from direct-facing to slightly-left-facing.",
      "type": "comparison"
    },
    {
      "id": "r6",
      "criterion": "The reasoning explicitly connects that orientation change to option A.",
      "type": "option_mapping"
    }
  ],
  "format_criteria": [
    {
      "id": "f1",
      "criterion": "Exactly one <think>...</think> block is present.",
      "type": "format"
    },
    {
      "id": "f2",
      "criterion": "Exactly one <answer>...</answer> block is present.",
      "type": "format"
    },
    {
      "id": "f3",
      "criterion": "The <answer> block contains only one uppercase option letter A, B, C, or D.",
      "type": "format"
    },
    {
      "id": "f4",
      "criterion": "The reasoning appears before the final answer, with no extra final conclusion outside <answer>.",
      "type": "format"
    }
  ],
  "negative_criteria": [
    {
      "id": "n1",
      "criterion": "The reasoning says the orientation remains unchanged across the two segments.",
      "type": "contradiction"
    },
    {
      "id": "n2",
      "criterion": "The reasoning says the lunge segment faces slightly right of the camera.",
      "type": "contradiction"
    },
    {
      "id": "n3",
      "criterion": "The reasoning reverses the transition by saying it goes from slightly left-facing to direct-facing.",
      "type": "contradiction"
    },
    {
      "id": "n4",
      "criterion": "The reasoning supports option B, C, or D instead of option A.",
      "type": "wrong_option_support"
    },
    {
      "id": "n5",
      "criterion": "The reasoning is generic and does not address camera-relative orientation.",
      "type": "irrelevant_reasoning"
    },
    {
      "id": "n6",
      "criterion": "The reasoning focuses on unrelated limb, posture, or action details without answering the orientation comparison.",
      "type": "irrelevant_reasoning"
    }
  ]
}
```

## Online Judge Prompt

Standalone copy:

```text
rubric_rl/prompt_templates/qa_mc_online_judge_prompt.txt
```

The online judge sees the criteria JSON and candidate response only. It judges visible reasoning quality, contradiction severity, and a small language score. Final answer correctness and format are handled later in Python.

### Online Prompt

```text
SYSTEM:
You are a strict judge for multiple-choice human-motion QA reasoning.

Use only the provided criteria JSON and candidate response.
The criteria JSON contains the question, options, correct option, reasoning criteria, and negative criteria.

Your job:
- judge whether the visible <think> reasoning satisfies the provided reasoning criteria
- identify reasoning contradictions and triggered negative criteria
- give a 0-5 language conciseness score
- choose a contradiction penalty from the allowed set

Your job is NOT:
- do not inspect video or motion assets
- do not infer new ground-truth evidence beyond the criteria
- do not compute final answer correctness
- do not compute format score
- do not compute final reward
- do not create new criteria, IDs, counts, or scoring rules

Return only valid JSON.

USER:
Criteria:
{CRITERIA_JSON}

Candidate response:
{CANDIDATE_RESPONSE}

Task:
Evaluate the candidate response for QA multiple-choice rubric RL.

The deterministic training code will separately parse final answer correctness and format compliance.
You may read the final <answer> only to understand whether the reasoning is consistent with what the candidate concludes.
Do not assign points for answer correctness.

Core principle:
Reward answer-relevant reasoning, not length. A good <think> block should identify the question focus, mention the relevant time ranges or phases when needed, state the key facts implied by the correct option, and connect those facts to the correct option.

Judging rules:
1. Evaluate reasoning_criteria using only the visible content inside <think>...</think>.
2. If there is no visible <think> content, return no satisfied reasoning IDs.
3. A reasoning criterion is satisfied when the <think> block expresses the same meaning by paraphrase.
4. Do not satisfy a reasoning criterion from the bare option letter inside <answer>.
5. Do not satisfy a criterion from generic text such as "I compare the evidence" unless it states the specific required relation.
6. For a segment_reference criterion, the reasoning must mention or clearly distinguish the referenced phases, segments, or time ranges.
7. For a correct_option_fact criterion, the reasoning must state the key fact from the correct option, not just say the option is correct.
8. For a comparison criterion, the reasoning must describe the before/after or between-segment relation.
9. For an option_mapping criterion, the reasoning must explicitly connect the described facts to the correct option letter or option text.
10. Do not require exact punctuation, exact timestamp formatting, or word-for-word option text.
11. Do require the core semantic relation. For QA01, "directly at the camera -> slightly left of the camera" is different from "directly at the camera -> slightly right of the camera" and different from "no change."
12. Put a reasoning ID in contradicted_ids when the <think> block states the opposite or an incompatible version of that criterion.
13. Do not treat omission as contradiction. Omitted criteria belong in missed_ids, not contradicted_ids.
14. Trigger negative_criteria IDs only when the candidate actually contains the described problem.
15. If the final <answer> is wrong but the <think> correctly supports the gold option, do not add contradiction penalty solely for the wrong final answer. Python handles answer correctness and the wrong-answer cap.
16. If the <think> supports a wrong option, reverses the correct relation, or contradicts the correct option text, trigger the relevant negative criteria and apply an appropriate contradiction penalty.
17. Use only IDs that exist in the criteria.
18. Do not return counts.

ID consistency:
- satisfied_ids must contain only reasoning_criteria IDs.
- missed_ids must contain only reasoning_criteria IDs.
- contradicted_ids must contain only reasoning_criteria IDs.
- triggered_ids must contain only negative_criteria IDs.
- A reasoning ID should not appear in both satisfied_ids and contradicted_ids.
- missed_ids should include reasoning criteria that are neither satisfied nor contradicted.

Language conciseness score:
Return an integer from 0 to 5 for the whole candidate response:
- 5: clear, concise, grammatical, and focused on the question.
- 4: mostly clear with minor verbosity or awkward wording.
- 3: understandable but generic, repetitive, or partly unfocused.
- 2: hard to follow, too verbose, or weakly organized.
- 1: barely understandable.
- 0: empty, malformed, or not meaningful.

Contradiction penalty:
Choose one of 0, -5, -10, -15, -20.
- 0: no meaningful contradiction.
- -5: minor irrelevant or unsupported statement that does not change the main option relation.
- -10: one clear contradiction of the requested attribute, time relation, comparison, or correct option fact.
- -15: multiple contradictions, or reasoning that substantially supports a wrong option.
- -20: reasoning is mostly unrelated, fabricated, or incompatible with the correct option.

Return only valid JSON with this exact schema:

{
  "reasoning_quality": {
    "satisfied_ids": [],
    "missed_ids": [],
    "contradicted_ids": []
  },
  "negative_criteria": {
    "triggered_ids": []
  },
  "language_conciseness_score": 0,
  "contradiction_penalty": 0,
  "notes": []
}
```

## Post-Processing Rules

Standalone copy:

```text
rubric_rl/prompt_templates/qa_mc_postprocessing_rules.txt
```

Post-processing stays deterministic and compact. Use one code block only:

```text
inputs:
  criteria: qa_mc criteria JSON
  candidate: raw candidate response text
  judge: online judge JSON

fixed weights:
  answer_correctness = 30
  reasoning_quality = 55
  format_compliance = 10
  language_conciseness = 5
  contradiction_penalty in {0, -5, -10, -15, -20}

parse:
  think_blocks = all exact case-sensitive <think>...</think> spans
  answer_blocks = all exact case-sensitive <answer>...</answer> spans
  has_one_think = len(think_blocks) == 1
  has_one_answer = len(answer_blocks) == 1
  parsed_answer =
    stripped answer_blocks[0] if has_one_answer and stripped content full-matches [A-D]
    else null

answer_score:
  if parsed_answer == criteria.correct_option:
    answer_score = 30
  else:
    answer_score = 0

format_score:
  format_score = 0
  if has_one_think: format_score += 2
  if has_one_answer: format_score += 2
  if parsed_answer is not null: format_score += 3
  if has_one_think and has_one_answer and think block appears before answer block
     and there is no extra final answer statement outside <answer>:
    format_score += 3
  format_score = clamp(format_score, 0, 10)

sanitize judge output:
  valid_reasoning_ids = ids(criteria.reasoning_criteria)
  valid_negative_ids = ids(criteria.negative_criteria)
  satisfied_ids = set(judge.reasoning_quality.satisfied_ids) intersect valid_reasoning_ids
  contradicted_ids = set(judge.reasoning_quality.contradicted_ids) intersect valid_reasoning_ids
  triggered_negative_ids = set(judge.negative_criteria.triggered_ids) intersect valid_negative_ids
  invalid_ids_removed = all judge-returned ids not in their valid id sets
  satisfied_ids = satisfied_ids - contradicted_ids
  if not has_one_think:
    satisfied_ids = empty set
    contradicted_ids = empty set
  missed_ids = valid_reasoning_ids - satisfied_ids

reasoning_score:
  total_reasoning = max(len(valid_reasoning_ids), 1)
  reasoning_score = 55 * len(satisfied_ids) / total_reasoning

language_score:
  language_score = clamp(round(judge.language_conciseness_score), 0, 5)

penalty:
  penalty = nearest_allowed(judge.contradiction_penalty, {0, -5, -10, -15, -20})
  if triggered_negative_ids is not empty and penalty == 0:
    penalty = -5
  if any triggered negative criterion has type contradiction or wrong_option_support
     and penalty > -10:
    penalty = -10

raw_score:
  raw_score = answer_score + reasoning_score + format_score + language_score + penalty
  total_score = clamp(raw_score, 0, 100)

caps:
  applied_caps = []
  if parsed_answer != criteria.correct_option:
    total_score = min(total_score, 50)
    applied_caps.append("wrong_answer_max_50")
  if parsed_answer is null:
    total_score = min(total_score, 20)
    applied_caps.append("no_valid_answer_max_20")
  if not has_one_think:
    total_score = min(total_score, 45)
    applied_caps.append("no_visible_reasoning_max_45")

reward:
  reward = total_score / 100

debug:
  return parsed_answer, correct_option, answer_score, format_score,
  reasoning_score, language_score, penalty, raw_score, total_score,
  reward, satisfied_ids, missed_ids, contradicted_ids,
  triggered_negative_ids, invalid_ids_removed, applied_caps
```

## Example Candidate

High-scoring candidate:

```text
<think>The question asks about orientation relative to the camera across two time segments. In the initial 00.00-01.30s segment, the person faces directly at the camera. In the 02.80-04.20s lunge segment, the person turns to face slightly left of the camera. This matches option A.</think>
<answer>A</answer>
```

Expected scoring behavior:

```text
answer_correctness: full credit
reasoning_quality: full or near-full credit
format_compliance: full credit
language_conciseness: high credit
contradiction_penalty: 0
```

Wrong-answer but reasonable-looking candidate:

```text
<think>The person starts facing directly at the camera and then turns slightly left during the lunge, so the orientation changes as described.</think>
<answer>B</answer>
```

Expected scoring behavior:

```text
answer_correctness: 0
reasoning_quality: partial to high
format_compliance: full credit
language_conciseness: high credit
final cap: max 50 because the answer is wrong
```

Generic reasoning candidate:

```text
<think>I compare the motion carefully and select the option that best matches the evidence.</think>
<answer>A</answer>
```

Expected scoring behavior:

```text
answer_correctness: full credit
reasoning_quality: low credit
format_compliance: full credit
language_conciseness: medium to high
contradiction_penalty: usually 0 or -5
```
