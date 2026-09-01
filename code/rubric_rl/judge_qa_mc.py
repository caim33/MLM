#!/usr/bin/env python3
"""Strict online-style judge for QA multiple-choice Rubric-RL."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from motionllm.grpo.qa_rubric import (
    QA_RUBRIC_VERSION,
    build_qa_judge_messages,
    compute_qa_rubric_reward,
    validate_qa_criteria,
    validate_qa_judgment,
)
from motionllm.grpo.rubric_common import (
    RubricValidationError,
    build_judgment_binding,
    strict_json_object,
)
import rubric_rl.artifacts as artifact_support
from rubric_rl.artifacts import (
    ArtifactError,
    AtomicJsonlArtifact,
    freeze_source_records,
    load_jsonl_strict,
)
from rubric_rl.qwen_text import QwenTextGenerator


def _split_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("SYSTEM:") or "\nUSER:" not in text:
        raise ValueError("QA judge prompt must contain leading SYSTEM: and USER: sections")
    system, user = text.split("\nUSER:", 1)
    system = system.removeprefix("SYSTEM:").strip()
    user = user.strip()
    if "{CRITERIA_JSON}" not in user or "{CANDIDATE_RESPONSE}" not in user:
        raise ValueError("QA judge prompt is missing required placeholders")
    return system, user


def _criteria_by_id(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_strict(path):
        sid = row["sample_id"]
        raw = row.get("criteria")
        if not isinstance(raw, Mapping):
            raise ArtifactError(f"criteria row {sid!r} has no criteria object")
        checked = validate_qa_criteria(raw)
        if checked["benchmark_id"] != sid:
            raise ArtifactError(
                f"criteria benchmark_id {checked['benchmark_id']!r} does not match row {sid!r}"
            )
        output[sid] = checked
    return output


def _candidate(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        for fallback in ("candidate", "prediction", "response", "answer_text"):
            value = row.get(fallback)
            if isinstance(value, str):
                break
    return value if isinstance(value, str) else ""


def _parse_max_memory(value: Optional[str]) -> Optional[dict[Any, str]]:
    if not value:
        return None
    parsed: dict[Any, str] = {}
    for chunk in value.split(","):
        key, memory = chunk.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = memory.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--candidate-key", default="candidate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--include-timing", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    written = 0
    skipped = 0
    seen_candidates: set[str] = set()
    code_paths = tuple(
        path.resolve()
        for path in (
            Path(__file__),
            Path(artifact_support.__file__),
            Path(__file__).with_name("qwen_text.py"),
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "qa_rubric.py",
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "rubric_common.py",
        )
    )
    source_paths = (
        args.criteria,
        args.candidates,
        args.prompt,
        *code_paths,
    )
    frozen_sources = freeze_source_records(source_paths)
    frozen_sha256 = {
        record["path"]: record["sha256"] for record in frozen_sources
    }
    prompt_path = args.prompt.resolve(strict=True)
    run_contract = {
        "operation": "judge_qa_mc",
        "model": args.model,
        "model_revision": args.model_revision,
        "generation": {
            "dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "effective_attn_implementation": args.attn_implementation,
            "attention_backend_fallback": "forbidden",
            "max_memory": args.max_memory,
            "max_new_tokens": args.max_new_tokens,
            "model_class": args.model_class,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "prompt": {
            "path": str(prompt_path),
            "sha256": frozen_sha256[str(prompt_path)],
        },
        "fields": {
            "id_key": "sample_id",
            "candidate_key": args.candidate_key,
            "candidate_fallback_keys": [
                "candidate",
                "prediction",
                "response",
                "answer_text",
            ],
        },
        "selection": {"limit": args.limit, "shard_index": 0, "num_shards": 1},
        "output": {
            "keep_raw": args.keep_raw,
            "include_timing": args.include_timing,
        },
        "code_sha256": {
            str(path): frozen_sha256[str(path)] for path in code_paths
        },
    }
    with AtomicJsonlArtifact(
        args.output,
        resume=args.resume,
        rubric_version=QA_RUBRIC_VERSION,
        source_paths=source_paths,
        expected_source_records=frozen_sources,
        run_contract=run_contract,
    ) as artifact:
        criteria = _criteria_by_id(args.criteria)
        system_prompt, user_template = _split_prompt(prompt_path)
        loaded_at = time.perf_counter()
        generator = QwenTextGenerator(
            args.model,
            revision=args.model_revision,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_memory=_parse_max_memory(args.max_memory),
            model_class=args.model_class,
            allow_attention_fallback=False,
        )
        load_seconds = time.perf_counter() - loaded_at
        if generator.effective_attn_implementation != args.attn_implementation:
            raise RuntimeError("formal rubric producer changed attention backend")
        done = artifact.done_ids
        for index, row in enumerate(load_jsonl_strict(args.candidates)):
            if args.limit is not None and index >= args.limit:
                break
            sid = row["sample_id"]
            if sid in seen_candidates:
                raise ArtifactError(f"duplicate candidate sample_id: {sid!r}")
            seen_candidates.add(sid)
            if sid in done:
                skipped += 1
                continue
            candidate = _candidate(row, args.candidate_key)
            criterion = criteria.get(sid)
            if criterion is None:
                payload: dict[str, Any] = {"sample_id": sid, "error": "missing_criteria"}
            elif not candidate.strip():
                payload = {"sample_id": sid, "error": "missing_candidate"}
            else:
                started = time.perf_counter()
                raw = generator.generate(
                    build_qa_judge_messages(
                        criterion,
                        candidate,
                        system_prompt=system_prompt,
                        user_template=user_template,
                    ),
                    max_new_tokens=args.max_new_tokens,
                )
                generation_seconds = time.perf_counter() - started
                try:
                    generated_judgment = strict_json_object(raw)
                    if "binding" in generated_judgment:
                        raise RubricValidationError(
                            "local judge output must not supply its own binding"
                        )
                    nonce = secrets.token_hex(32)
                    generated_judgment["binding"] = build_judgment_binding(
                        criterion,
                        candidate,
                        sample_id=sid,
                        nonce=nonce,
                    )
                    judgment = validate_qa_judgment(
                        generated_judgment,
                        criterion,
                        candidate_response=candidate,
                        expected_nonce=nonce,
                        reject_unknown_ids=True,
                    )
                except RubricValidationError as exc:
                    payload = {
                        "sample_id": sid,
                        "error": "invalid_judge_output",
                        "validation_error": str(exc),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
                else:
                    payload = {
                        "sample_id": sid,
                        "judgment": judgment,
                        "reward": compute_qa_rubric_reward(criterion, candidate, judgment),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
                if args.include_timing:
                    payload["timing"] = {
                        "load_seconds": load_seconds,
                        "generation_seconds": generation_seconds,
                    }
            artifact.append(payload)
            written += 1
        inventory = artifact.commit()

    print(
        json.dumps(
            {
                "event": "done",
                "written": written,
                "skipped": skipped,
                "artifact_sha256": inventory["artifact_sha256"],
                "inventory": str(args.output.with_name(args.output.name + ".inventory.json")),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
