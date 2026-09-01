#!/usr/bin/env python3
"""Build GRPO JSONL from Motion-X-example media and report_20260415 QA markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ReportSample:
    sample_id: str
    status: str
    qas: List[Dict[str, str]]


SECTION_HEADER = re.compile(r"^#### 文本: `([^`]+)`\s*$", flags=re.MULTILINE)
MCQ_LETTERS = ("A", "B", "C", "D")


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _iter_sample_sections(report_text: str) -> Iterable[Tuple[str, str]]:
    matches = list(SECTION_HEADER.finditer(report_text))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report_text)
        yield match.group(1), report_text[start:end]


def _extract_first_json_list(text: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(obj, list):
            return obj
    raise ValueError("No JSON list found in section output.")


def _parse_qas(output_block: str) -> List[Dict[str, str]]:
    parsed = _extract_first_json_list(output_block)
    results: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip().lower()
        if not question or not answer:
            continue
        if difficulty not in {"easy", "hard"}:
            difficulty = "hard"
        results.append(
            {
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
            }
        )
    return results


def _swap_terms(text: str, pairs: Sequence[Tuple[str, str]]) -> str:
    out = text
    for idx, (left, right) in enumerate(pairs):
        token_left = f"__SWAP_LEFT_{idx}__"
        token_right = f"__SWAP_RIGHT_{idx}__"
        out = out.replace(left, token_left)
        out = out.replace(right, token_right)
        out = out.replace(token_left, right)
        out = out.replace(token_right, left)
    return out


def _swap_left_right(text: str) -> str:
    return _swap_terms(
        text,
        [
            ("left", "right"),
            ("Left", "Right"),
            ("LEFT", "RIGHT"),
            ("screen-left", "screen-right"),
            ("Screen-left", "Screen-right"),
            ("screen left", "screen right"),
            ("Screen left", "Screen right"),
        ],
    )


def _invert_relation_or_temporal(text: str) -> str:
    return _swap_terms(
        text,
        [
            ("before", "after"),
            ("Before", "After"),
            ("start", "end"),
            ("Start", "End"),
            ("begins", "ends"),
            ("Begins", "Ends"),
            ("increases", "decreases"),
            ("Increases", "Decreases"),
            ("forward", "backward"),
            ("Forward", "Backward"),
            ("bent", "straight"),
            ("Bent", "Straight"),
        ],
    )


def _pick_unique_texts(candidates: Sequence[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = re.sub(r"\s+", " ", item.strip())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _build_hard_mcq(
    *,
    question_id: str,
    question: str,
    answer: str,
    seed: int,
) -> Tuple[str, str, Dict[str, str]]:
    correct = re.sub(r"\s+", " ", answer.strip())
    candidates = [
        _swap_left_right(correct),
        _invert_relation_or_temporal(correct),
        _swap_left_right(_invert_relation_or_temporal(correct)),
        "The opposite side performs the key action, with mirrored body orientation and support relation.",
        "Both sides contribute equally during the key phase, and no dominant side is observed.",
        "The temporal order is reversed relative to the described movement, including orientation transition.",
    ]
    distractors = [x for x in _pick_unique_texts(candidates, limit=6) if x.lower() != correct.lower()]
    if len(distractors) < 3:
        raise ValueError(f"Not enough distinct distractors for {question_id}")
    option_texts = [correct, distractors[0], distractors[1], distractors[2]]

    digest = hashlib.md5(f"{question_id}:{seed}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = random.Random(local_seed)
    rng.shuffle(option_texts)

    option_map: Dict[str, str] = {}
    correct_letter = "A"
    for letter, text in zip(MCQ_LETTERS, option_texts):
        option_map[letter] = text
        if text.lower() == correct.lower():
            correct_letter = letter

    question_text = (
        f"{question.strip()}\n\n"
        "Choose exactly one option:\n"
        f"A. {option_map['A']}\n"
        f"B. {option_map['B']}\n"
        f"C. {option_map['C']}\n"
        f"D. {option_map['D']}\n"
    )
    return question_text, correct_letter, option_map


def parse_report(report_path: Path) -> List[ReportSample]:
    text = _read_text(report_path)
    samples: List[ReportSample] = []
    for header, body in _iter_sample_sections(text):
        source_id = Path(header).parts[0]
        status_match = re.search(r"\*\*状态\*\*:\s*([^\n\r]+)", body)
        status = status_match.group(1).strip() if status_match else "unknown"
        output_idx = body.find("**输出文本:**")
        if output_idx < 0:
            continue
        output_block = body[output_idx:]
        qas = _parse_qas(output_block)
        if not qas:
            continue
        samples.append(ReportSample(sample_id=source_id, status=status, qas=qas))
    return samples


def _resolve_video_path(media_dir: Path, sample_id: str) -> Optional[Path]:
    candidates = [
        media_dir / f"{sample_id}.mp4",
    ]
    if len(sample_id) >= 10:
        transformed = f"{sample_id[:6]}_{sample_id[6:8]}_{sample_id[8]}_{sample_id[9:]}.mp4"
        candidates.append(media_dir / transformed)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_motion_path(media_dir: Path, sample_id: str, prefer: str) -> Optional[Path]:
    preferred = media_dir / f"{sample_id}.{prefer}"
    if preferred.exists():
        return preferred
    fallback = media_dir / f"{sample_id}.npz" if prefer == "npy" else media_dir / f"{sample_id}.npy"
    if fallback.exists():
        return fallback
    return None


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
    sample_id: str,
    question: str,
    answer: str,
    difficulty: str,
    video_path: Path,
    motion_path: Path,
    mcq_options: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    tagged_answer = f"<answer>{answer}</answer>"
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
    sample_id: str,
    question: str,
    answer: str,
    difficulty: str,
    video_path: Path,
    mcq_options: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    tagged_answer = f"<answer>{answer}</answer>"
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


def build_records(
    samples: Sequence[ReportSample],
    media_dir: Path,
    motion_ext: str,
    strict_media: bool,
    difficulty_filter: str = "all",
    hard_to_mcq: bool = False,
    mcq_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    easy_q = 0
    hard_q = 0
    mcq_q = 0

    for sample in samples:
        video_path = _resolve_video_path(media_dir=media_dir, sample_id=sample.sample_id)
        motion_path = _resolve_motion_path(
            media_dir=media_dir, sample_id=sample.sample_id, prefer=motion_ext
        )
        if video_path is None or motion_path is None:
            reason = {
                "sample_id": sample.sample_id,
                "video_found": str(video_path is not None),
                "motion_found": str(motion_path is not None),
            }
            skipped.append(reason)
            if strict_media:
                raise FileNotFoundError(f"Missing media for sample: {reason}")
            continue

        for idx, qa in enumerate(sample.qas, start=1):
            question_id = f"{sample.sample_id}_q{idx}"
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            difficulty = qa["difficulty"].strip().lower()
            if difficulty_filter != "all" and difficulty != difficulty_filter:
                continue
            if difficulty == "easy":
                easy_q += 1
            elif difficulty == "hard":
                hard_q += 1

            mcq_options: Dict[str, str] = {}
            if hard_to_mcq and difficulty == "hard":
                question, answer, mcq_options = _build_hard_mcq(
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    seed=mcq_seed,
                )
                mcq_q += 1

            records.append(
                _build_vm_record(
                    question_id=question_id,
                    sample_id=sample.sample_id,
                    question=question,
                    answer=answer,
                    difficulty=difficulty,
                    video_path=video_path,
                    motion_path=motion_path,
                    mcq_options=mcq_options,
                )
            )
            records.append(
                _build_v_record(
                    question_id=question_id,
                    sample_id=sample.sample_id,
                    question=question,
                    answer=answer,
                    difficulty=difficulty,
                    video_path=video_path,
                    mcq_options=mcq_options,
                )
            )

    meta = {
        "num_report_samples": len(samples),
        "num_records": len(records),
        "num_questions": len(records) // 2,
        "num_skipped_samples": len(skipped),
        "skipped_samples": skipped,
        "num_easy_questions": easy_q,
        "num_hard_questions": hard_q,
        "num_mcq_questions": mcq_q,
        "difficulty_filter": difficulty_filter,
        "hard_to_mcq": hard_to_mcq,
    }
    return records, meta


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Motion-X report-based GRPO dataset.")
    parser.add_argument("--report", type=Path, required=True, help="Path to report_*.md")
    parser.add_argument(
        "--media_dir",
        type=Path,
        required=True,
        help="Directory containing Motion-X media files",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--meta_output",
        type=Path,
        default=None,
        help="Optional meta summary JSON path",
    )
    parser.add_argument(
        "--motion_ext",
        choices=["npy", "npz"],
        default="npy",
        help="Preferred motion file extension",
    )
    parser.add_argument(
        "--strict_media",
        action="store_true",
        help="Fail if any sample is missing matching video/motion media files",
    )
    parser.add_argument(
        "--difficulty_filter",
        choices=["all", "hard", "easy"],
        default="all",
        help="Keep only selected difficulty level.",
    )
    parser.add_argument(
        "--hard_to_mcq",
        action="store_true",
        help="Convert hard QA into 4-choice MCQ and store answer as option letter.",
    )
    parser.add_argument(
        "--mcq_seed",
        type=int,
        default=42,
        help="Random seed for deterministic MCQ option order.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples = parse_report(args.report)
    records, meta = build_records(
        samples=samples,
        media_dir=args.media_dir,
        motion_ext=args.motion_ext,
        strict_media=args.strict_media,
        difficulty_filter=args.difficulty_filter,
        hard_to_mcq=args.hard_to_mcq,
        mcq_seed=args.mcq_seed,
    )
    _write_jsonl(args.output, records)
    meta_output = args.meta_output or args.output.with_suffix(".meta.json")
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "meta_output": str(meta_output),
                **meta,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
