#!/usr/bin/env python3
"""Build Motion-X GRPO JSONL from deepseek-chat QA jsons (hard/easy selectable)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MCQ_LETTERS = ("A", "B", "C", "D")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_response_text_to_qa_list(response_text: str) -> List[Dict[str, Any]]:
    response_text = response_text.strip()
    if not response_text:
        return []
    try:
        obj = json.loads(response_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
    return out


def _resolve_video_path(videos_dir: Path, item_id: str) -> Optional[Path]:
    direct = videos_dir / f"{item_id}.mp4"
    if direct.exists():
        return direct

    transformed: Optional[Path] = None
    if len(item_id) >= 10:
        transformed_name = f"{item_id[:6]}_{item_id[6:8]}_{item_id[8]}_{item_id[9:]}.mp4"
        transformed = videos_dir / transformed_name
        if transformed.exists():
            return transformed

    return None


def _resolve_motion_path(motion_dir: Path, item_id: str) -> Optional[Path]:
    p = motion_dir / f"{item_id}.npy"
    return p if p.exists() else None


def _format_mcq_question(question: str, options: Dict[str, str]) -> str:
    has_full_options = all(k in options for k in MCQ_LETTERS)
    if not has_full_options:
        return question.strip()
    return (
        f"{question.strip()}\n\n"
        "Choose exactly one option:\n"
        f"A. {str(options['A']).strip()}\n"
        f"B. {str(options['B']).strip()}\n"
        f"C. {str(options['C']).strip()}\n"
        f"D. {str(options['D']).strip()}\n"
    )


def _build_user_text(question_id: str, branch: str, question: str) -> str:
    return (
        f"[QID={question_id}] [BRANCH={branch}] "
        "You are given multimodal evidence. Analyze carefully before answering.\n\n"
        "Follow these rules strictly:\n"
        "1. First provide concise reasoning inside <think>...</think>.\n"
        "2. Then provide the final answer inside <answer>...</answer>.\n"
        "3. The final conclusion MUST appear only inside <answer>...</answer>.\n"
        "4. Do NOT place any final answer statement outside <answer> tags.\n"
        "5. Output exactly one pair of <think> and <answer> tags.\n\n"
        "If the question provides options (A/B/C/D), output only one uppercase option letter inside <answer>.\n\n"
        "Reasoning checklist (use only what is supported by the modalities):\n"
        "- main action\n"
        "- key body movement and temporal order\n"
        "- relevant spatial relations / object interactions\n\n"
        f"Question: {question}"
    )


def _build_vm_record(
    question_id: str,
    question: str,
    answer_letter: str,
    video_path: Path,
    motion_path: Path,
) -> Dict[str, Any]:
    tagged_answer = f"<answer>{answer_letter}</answer>"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {
                        "type": "text",
                        "text": "<motion_start><motion><motion_end>\n"
                        + _build_user_text(question_id=question_id, branch="vm", question=question),
                    },
                ],
            }
        ],
        "motion": str(motion_path),
        "solution": tagged_answer,
        "answer": tagged_answer,
        "group_id": question_id,
        "branch": "vm",
        "sample_id": f"{question_id}_vm",
        "rollout_id": 0,
    }


def _build_v_record(
    question_id: str,
    question: str,
    answer_letter: str,
    video_path: Path,
) -> Dict[str, Any]:
    tagged_answer = f"<answer>{answer_letter}</answer>"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {
                        "type": "text",
                        "text": _build_user_text(question_id=question_id, branch="v", question=question),
                    },
                ],
            }
        ],
        "solution": tagged_answer,
        "answer": tagged_answer,
        "group_id": question_id,
        "branch": "v",
        "sample_id": f"{question_id}_v",
        "rollout_id": 1,
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Motion-X GRPO JSONL from deepseek-chat QA files.")
    parser.add_argument("--qa_dir", type=Path, required=True)
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--motion_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta_output", type=Path, default=None)
    parser.add_argument("--difficulty", choices=["hard", "easy", "all"], default="hard")
    parser.add_argument("--strict_media", action="store_true")
    parser.add_argument("--expect_questions_per_item", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    qa_files = sorted(args.qa_dir.glob("*.json"))
    video_lookup: Dict[str, Path] = {vp.stem.replace("_", ""): vp for vp in args.videos_dir.glob("*.mp4")}

    records: List[Dict[str, Any]] = []
    skipped_missing_media: List[Dict[str, str]] = []
    invalid_files: List[str] = []
    question_count_by_item: Dict[str, int] = {}
    item_id_mismatches: List[Dict[str, str]] = []

    num_items_total = 0
    num_items_with_kept_questions = 0
    num_questions_total = 0
    num_kept_questions = 0

    for qa_file in qa_files:
        num_items_total += 1
        payload = _read_json(qa_file)
        item_id = qa_file.stem
        payload_item_id = str(payload.get("item_id", "")).strip()
        if payload_item_id and payload_item_id != item_id:
            item_id_mismatches.append({"qa_file": str(qa_file), "file_stem": item_id, "payload_item_id": payload_item_id})

        raw_qas = _parse_response_text_to_qa_list(str(payload.get("response_text", "")))
        if not raw_qas:
            invalid_files.append(str(qa_file))
            continue

        video_path = video_lookup.get(item_id)
        motion_path = _resolve_motion_path(args.motion_dir, item_id)
        if video_path is None or motion_path is None:
            skipped_missing_media.append(
                {
                    "item_id": item_id,
                    "video_found": str(video_path is not None),
                    "motion_found": str(motion_path is not None),
                }
            )
            if args.strict_media:
                raise FileNotFoundError(f"Missing media for item_id={item_id}")
            continue

        kept_for_item = 0
        hard_index = 0
        for qa in raw_qas:
            num_questions_total += 1

            question = str(qa.get("question", "")).strip()
            answer_key = str(qa.get("answer_key", "")).strip().upper()
            difficulty = str(qa.get("difficulty", "")).strip().lower()
            options = qa.get("options", {})
            if not isinstance(options, dict):
                options = {}

            if not question or answer_key not in MCQ_LETTERS:
                continue

            if args.difficulty != "all" and difficulty != args.difficulty:
                continue

            if difficulty == "hard":
                hard_index += 1
                qid_suffix = hard_index
            else:
                qid_suffix = kept_for_item + 1

            question_id = f"{item_id}_q{qid_suffix}"
            question_text = _format_mcq_question(question=question, options=options)

            records.append(
                _build_vm_record(
                    question_id=question_id,
                    question=question_text,
                    answer_letter=answer_key,
                    video_path=video_path,
                    motion_path=motion_path,
                )
            )
            records.append(
                _build_v_record(
                    question_id=question_id,
                    question=question_text,
                    answer_letter=answer_key,
                    video_path=video_path,
                )
            )
            kept_for_item += 1
            num_kept_questions += 1

        question_count_by_item[item_id] = kept_for_item
        if kept_for_item > 0:
            num_items_with_kept_questions += 1

    _write_jsonl(args.output, records)

    expected_mismatch_items: List[Dict[str, int]] = []
    if args.expect_questions_per_item > 0:
        for item_id, n in question_count_by_item.items():
            if n != args.expect_questions_per_item:
                expected_mismatch_items.append({"item_id": item_id, "kept_questions": n})

    meta = {
        "qa_dir": str(args.qa_dir),
        "videos_dir": str(args.videos_dir),
        "motion_dir": str(args.motion_dir),
        "output": str(args.output),
        "difficulty": args.difficulty,
        "num_items_total": num_items_total,
        "num_items_with_kept_questions": num_items_with_kept_questions,
        "num_questions_total_in_qa": num_questions_total,
        "num_kept_questions": num_kept_questions,
        "num_records": len(records),
        "num_groups": len(records) // 2,
        "num_skipped_missing_media": len(skipped_missing_media),
        "skipped_missing_media": skipped_missing_media,
        "num_invalid_files": len(invalid_files),
        "invalid_files": invalid_files,
        "num_item_id_mismatches": len(item_id_mismatches),
        "item_id_mismatches": item_id_mismatches,
        "expect_questions_per_item": args.expect_questions_per_item,
        "num_expect_mismatch_items": len(expected_mismatch_items),
        "expect_mismatch_items": expected_mismatch_items,
    }

    meta_output = args.meta_output or args.output.with_suffix(".meta.json")
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
