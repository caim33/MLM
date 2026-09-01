#!/usr/bin/env python3
"""Export first-step V/VM rollout completions and rewards for a target Motion-X sample ID."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

QUESTION_DIFFICULTY_RE = re.compile(
    r'"question"\s*:\s*"([^"]+)"[\s\S]{0,500}?"difficulty"\s*:\s*"(easy|hard)"',
    re.IGNORECASE,
)
QID_RE = re.compile(r"\[QID=([^\]]+)\]")
SAMPLE_QID_RE = re.compile(r"^(?P<base>.+)_q(?P<idx>\d+)_(?:vm|v)$")
ANSWER_TAG_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)

DEFAULT_WEIGHT_OPTION = 1.0
DEFAULT_WEIGHT_FORMAT = 0.8
DEFAULT_WEIGHT_BONUS = 1.0


@dataclass
class RolloutRecord:
    step: int
    qid: str
    branch: str
    completion: str
    semantic: float
    fmt: float
    bonus: float
    total: float
    line_no: int
    item_idx: int


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _parse_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
    return None


def extract_qid_difficulty_map(report_path: Path, target_id: str) -> Dict[str, Dict[str, str]]:
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    marker = f"{target_id}/step2_generation.json"
    start: Optional[int] = None
    end: Optional[int] = None

    for i, line in enumerate(lines):
        if line.startswith("#### 文本:") and marker in line:
            start = i
            continue
        if start is not None and line.startswith("#### 文本:") and marker not in line:
            end = i
            break

    if start is None:
        raise ValueError(f"Cannot find section for {marker} in report: {report_path}")

    section = "\n".join(lines[start:end]) if end is not None else "\n".join(lines[start:])
    pairs = QUESTION_DIFFICULTY_RE.findall(section)
    if len(pairs) < 3:
        raise ValueError(
            f"Expected at least 3 question/difficulty pairs for {target_id}, found {len(pairs)}"
        )

    mapping: Dict[str, Dict[str, str]] = {}
    for idx, (question, difficulty) in enumerate(pairs[:3], start=1):
        qid = f"{target_id}_q{idx}"
        mapping[qid] = {
            "difficulty": difficulty.lower(),
            "question": question,
        }
    return mapping


def parse_rollouts(
    completions_path: Path,
    target_id: str,
    weight_option: float,
    weight_format: float,
    weight_bonus: float,
) -> List[RolloutRecord]:
    records: List[RolloutRecord] = []
    target_prefix = f"{target_id}_"

    with completions_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)

            steps = payload.get("step", [])
            prompts = payload.get("prompt", [])
            completions = payload.get("completion", [])
            sem = payload.get("MotionSemanticORM", [])
            fmt = payload.get("MotionFormatORM", [])
            bonus = payload.get("MotionVMVGroupBonusORM", [])

            if not isinstance(prompts, list):
                continue

            for item_idx, prompt in enumerate(prompts):
                if not isinstance(prompt, str):
                    continue
                match = QID_RE.search(prompt)
                if not match:
                    continue
                qid = match.group(1)
                if not qid.startswith(target_prefix):
                    continue

                branch = "vm" if "[BRANCH=vm]" in prompt else ("v" if "[BRANCH=v]" in prompt else "unk")
                if branch not in {"vm", "v"}:
                    continue

                step_raw = None
                if isinstance(steps, list):
                    if item_idx < len(steps):
                        step_raw = steps[item_idx]
                    elif steps:
                        step_raw = steps[0]
                step = _parse_int(step_raw)
                if step is None:
                    continue

                completion = ""
                if isinstance(completions, list) and item_idx < len(completions):
                    value = completions[item_idx]
                    completion = value if isinstance(value, str) else str(value)

                semantic = _safe_float(sem[item_idx] if isinstance(sem, list) and item_idx < len(sem) else 0.0)
                fmt_score = _safe_float(fmt[item_idx] if isinstance(fmt, list) and item_idx < len(fmt) else 0.0)
                bonus_score = _safe_float(
                    bonus[item_idx] if isinstance(bonus, list) and item_idx < len(bonus) else 0.0
                )
                total = (
                    semantic * float(weight_option)
                    + fmt_score * float(weight_format)
                    + bonus_score * float(weight_bonus)
                )

                records.append(
                    RolloutRecord(
                        step=step,
                        qid=qid,
                        branch=branch,
                        completion=completion,
                        semantic=semantic,
                        fmt=fmt_score,
                        bonus=bonus_score,
                        total=total,
                        line_no=line_no,
                        item_idx=item_idx,
                    )
                )

    return records


def extract_qid_gt_answers(dataset_path: Path, target_id: str) -> Dict[str, str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    qid_to_answer: Dict[str, str] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str):
                continue
            matched = SAMPLE_QID_RE.match(sample_id)
            if not matched or matched.group("base") != target_id:
                continue
            qid = f"{target_id}_q{matched.group('idx')}"

            answer = payload.get("answer")
            if not isinstance(answer, str):
                continue
            answer = answer.strip()
            if not answer:
                continue

            answer_match = ANSWER_TAG_RE.search(answer)
            clean_answer = answer_match.group(1).strip() if answer_match else answer
            if qid not in qid_to_answer or not qid_to_answer[qid]:
                qid_to_answer[qid] = clean_answer

    return qid_to_answer


def first_step_filter(records: List[RolloutRecord]) -> Tuple[Dict[str, int], Dict[str, Dict[str, List[RolloutRecord]]]]:
    qid_first_step: Dict[str, int] = {}
    for rec in records:
        cur = qid_first_step.get(rec.qid)
        if cur is None or rec.step < cur:
            qid_first_step[rec.qid] = rec.step

    grouped: Dict[str, Dict[str, List[RolloutRecord]]] = {}
    for rec in records:
        if qid_first_step.get(rec.qid) != rec.step:
            continue
        grouped.setdefault(rec.qid, {"vm": [], "v": []})[rec.branch].append(rec)

    for qid in grouped:
        grouped[qid]["vm"].sort(key=lambda r: (r.line_no, r.item_idx))
        grouped[qid]["v"].sort(key=lambda r: (r.line_no, r.item_idx))

    return qid_first_step, grouped


def render_markdown(
    target_id: str,
    report_path: Path,
    completions_path: Path,
    qid_meta: Dict[str, Dict[str, str]],
    qid_gt: Dict[str, str],
    first_steps: Dict[str, int],
    grouped: Dict[str, Dict[str, List[RolloutRecord]]],
    weight_option: float,
    weight_format: float,
    weight_bonus: float,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# Rollout Export for {target_id}")
    lines.append("")
    lines.append(f"- Generated at: `{timestamp}`")
    lines.append(f"- Report source: `{report_path}`")
    lines.append(f"- Completions source: `{completions_path}`")
    lines.append(
        f"- Reward weights: option={weight_option}, format={weight_format}, bonus={weight_bonus}"
    )
    lines.append("- Rule: keep only the earliest step per QID; export full completion text.")
    lines.append("")

    qid_order = [f"{target_id}_q1", f"{target_id}_q2", f"{target_id}_q3"]

    for qid in qid_order:
        meta = qid_meta.get(qid, {"difficulty": "unknown", "question": ""})
        difficulty = meta.get("difficulty", "unknown")
        question = meta.get("question", "")
        gt_answer = qid_gt.get(qid, "")
        step = first_steps.get(qid)
        branch_data = grouped.get(qid, {"vm": [], "v": []})

        lines.append(f"## {qid} ({difficulty})")
        lines.append("")
        lines.append(f"- First step: `{step}`")
        lines.append(f"- Question: {question}")
        lines.append(f"- GT answer: {gt_answer if gt_answer else 'N/A'}")
        lines.append(f"- VM rollout count: `{len(branch_data['vm'])}`")
        lines.append(f"- V rollout count: `{len(branch_data['v'])}`")
        lines.append("")

        for branch in ["vm", "v"]:
            upper = branch.upper()
            lines.append(f"### {upper} group")
            lines.append("")
            if not branch_data[branch]:
                lines.append("- No rollout records found.")
                lines.append("")
                continue

            for idx, rec in enumerate(branch_data[branch], start=1):
                lines.append(f"#### {upper} rollout {idx}")
                lines.append("")
                lines.append(f"- step: `{rec.step}`")
                lines.append(f"- semantic: `{rec.semantic:.6f}`")
                lines.append(f"- format: `{rec.fmt:.6f}`")
                lines.append(f"- vm_v_bonus: `{rec.bonus:.6f}`")
                lines.append(f"- total_reward: `{rec.total:.6f}`")
                lines.append("")
                lines.append("````text")
                lines.append(rec.completion)
                lines.append("````")
                lines.append("")

    lines.append("## Validation")
    lines.append("")
    for qid in qid_order:
        step = first_steps.get(qid)
        vm_n = len(grouped.get(qid, {}).get("vm", []))
        v_n = len(grouped.get(qid, {}).get("v", []))
        lines.append(f"- {qid}: first_step={step}, vm_count={vm_n}, v_count={v_n}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export first-step V/VM rollout records for one sample ID.")
    parser.add_argument("--target-id", default="11191119210078")
    parser.add_argument(
        "--report-path",
        default="/wangbenyou-sulongjie/Motion-r1/data/report_20260415_120712.md",
    )
    parser.add_argument(
        "--completions-path",
        default=(
            "/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/grpo/"
            "motionx_real_r4_180/v1-20260418-044343/completions.jsonl"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="/wangbenyou-sulongjie/Motion-r1/data/grpo/motionx_report_grpo_promptv2_strict.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default=(
            "/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/analysis/"
            "11191119210078_first_step_rollouts.md"
        ),
    )
    parser.add_argument("--weight-option", type=float, default=DEFAULT_WEIGHT_OPTION)
    parser.add_argument("--weight-format", type=float, default=DEFAULT_WEIGHT_FORMAT)
    parser.add_argument("--weight-bonus", type=float, default=DEFAULT_WEIGHT_BONUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_path = Path(args.report_path)
    completions_path = Path(args.completions_path)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")
    if not completions_path.exists():
        raise FileNotFoundError(f"completions not found: {completions_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    qid_meta = extract_qid_difficulty_map(report_path, args.target_id)
    qid_gt = extract_qid_gt_answers(dataset_path, args.target_id)
    records = parse_rollouts(
        completions_path=completions_path,
        target_id=args.target_id,
        weight_option=args.weight_option,
        weight_format=args.weight_format,
        weight_bonus=args.weight_bonus,
    )
    first_steps, grouped = first_step_filter(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(
        target_id=args.target_id,
        report_path=report_path,
        completions_path=completions_path,
        qid_meta=qid_meta,
        qid_gt=qid_gt,
        first_steps=first_steps,
        grouped=grouped,
        weight_option=args.weight_option,
        weight_format=args.weight_format,
        weight_bonus=args.weight_bonus,
    )
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Exported markdown: {output_path}")
    for qid in [f"{args.target_id}_q1", f"{args.target_id}_q2", f"{args.target_id}_q3"]:
        vm_n = len(grouped.get(qid, {}).get("vm", []))
        v_n = len(grouped.get(qid, {}).get("v", []))
        print(f"{qid}: first_step={first_steps.get(qid)} vm={vm_n} v={v_n}")


if __name__ == "__main__":
    main()
