#!/usr/bin/env python3
"""Online-style Qwen judge for V2 source-aware Rubric RL."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rubric_rl.prompts_v2 import build_online_messages
from rubric_rl.qwen_text import QwenTextGenerator
import rubric_rl.artifacts as artifact_support
from rubric_rl.artifacts import (
    ArtifactError,
    AtomicJsonlArtifact,
    freeze_source_records,
    iter_jsonl_objects,
)
from rubric_rl.reward_v2 import (
    MOTION_RUBRIC_V2_VERSION,
    RubricValidationError,
    compute_reward,
    ensure_criteria_ids,
    validate_motion_judgment_v2,
)
from motionllm.grpo.rubric_common import build_judgment_binding, strict_json_object


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    yield from iter_jsonl_objects(path)


def load_criteria(path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(load_jsonl(path)):
        sid = str(row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)
        if sid in by_id:
            raise ArtifactError(f"duplicate criteria sample_id: {sid!r}")
        if not isinstance(row.get("criteria"), dict):
            raise ArtifactError(f"criteria row {sid!r} has no criteria object")
        by_id[sid] = ensure_criteria_ids(row["criteria"])
    return by_id


def row_id(row: Dict[str, Any], id_key: str, index: int) -> str:
    return str(row.get(id_key) or row.get("sample_id") or row.get("id") or row.get("benchmark_id") or index)


def get_candidate(row: Dict[str, Any], candidate_key: str) -> str:
    for key in [candidate_key, "candidate", "prediction", "final_answer", "answer_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_max_memory(value: Optional[str]) -> Optional[Dict[Any, str]]:
    if not value:
        return None
    parsed: Dict[Any, str] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        key, mem = chunk.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = mem.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--candidate-key", default="candidate")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--model-class", choices=["image_text", "causal_lm"], default="image_text")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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
            Path(__file__).with_name("prompts_v2.py"),
            Path(__file__).with_name("reward_v2.py"),
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "motion_rubric_v2.py",
            Path(__file__).parents[1] / "src" / "motionllm" / "grpo" / "rubric_common.py",
        )
    )
    source_paths = (args.criteria, args.candidates, *code_paths)
    frozen_sources = freeze_source_records(source_paths)
    frozen_sha256 = {
        record["path"]: record["sha256"] for record in frozen_sources
    }
    prompt_builder_path = Path(__file__).with_name("prompts_v2.py").resolve(
        strict=True
    )
    run_contract = {
        "operation": "judge_motion_caption_v2",
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
            "builder": "rubric_rl.prompts_v2.build_online_messages",
            "sha256": frozen_sha256[str(prompt_builder_path)],
        },
        "fields": {
            "id_key": args.id_key,
            "candidate_key": args.candidate_key,
            "candidate_fallback_keys": [
                "candidate",
                "prediction",
                "final_answer",
                "answer_text",
            ],
        },
        "selection": {"limit": args.limit, "shard_index": 0, "num_shards": 1},
        "output": {"keep_raw": args.keep_raw, "include_timing": False},
        "code_sha256": {
            str(path): frozen_sha256[str(path)] for path in code_paths
        },
    }
    with AtomicJsonlArtifact(
        args.output,
        resume=args.resume,
        rubric_version=MOTION_RUBRIC_V2_VERSION,
        source_paths=source_paths,
        expected_source_records=frozen_sources,
        run_contract=run_contract,
    ) as artifact:
        criteria_by_id = load_criteria(args.criteria)
        generator = QwenTextGenerator(
            args.model,
            revision=args.model_revision,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_memory=parse_max_memory(args.max_memory),
            model_class=args.model_class,
            allow_attention_fallback=False,
        )
        if generator.effective_attn_implementation != args.attn_implementation:
            raise RuntimeError("formal rubric producer changed attention backend")
        done = artifact.done_ids
        for index, row in enumerate(load_jsonl(args.candidates)):
            if args.limit is not None and index >= args.limit:
                break
            sid = row_id(row, args.id_key, index)
            if sid in seen_candidates:
                raise ArtifactError(f"duplicate candidate sample_id: {sid!r}")
            seen_candidates.add(sid)
            if sid in done:
                skipped += 1
                continue
            metadata = {
                key: row[key]
                for key in ("candidate_id", "expected_quality", "expected_behavior")
                if key in row
            }
            criteria = criteria_by_id.get(sid)
            candidate = get_candidate(row, args.candidate_key)
            if not criteria:
                payload: Dict[str, Any] = {"sample_id": sid, **metadata, "error": "missing_criteria"}
            elif not candidate:
                payload = {"sample_id": sid, **metadata, "error": "missing_candidate"}
            else:
                raw = generator.generate(build_online_messages(criteria, candidate), max_new_tokens=args.max_new_tokens)
                try:
                    generated_judgment = strict_json_object(raw)
                    if "binding" in generated_judgment:
                        raise RubricValidationError(
                            "local judge output must not supply its own binding"
                        )
                    nonce = secrets.token_hex(32)
                    generated_judgment["binding"] = build_judgment_binding(
                        criteria,
                        candidate,
                        sample_id=sid,
                        nonce=nonce,
                    )
                    judgment = validate_motion_judgment_v2(
                        generated_judgment,
                        criteria,
                        candidate_response=candidate,
                        sample_id=sid,
                        expected_nonce=nonce,
                        reject_unknown_ids=True,
                    )
                except RubricValidationError as exc:
                    payload = {
                        "sample_id": sid,
                        **metadata,
                        "error": "invalid_judge_output",
                        "validation_error": str(exc),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
                else:
                    payload = {
                        "sample_id": sid,
                        **metadata,
                        "judgment": judgment,
                        "reward": compute_reward(
                            criteria, judgment, candidate, sample_id=sid
                        ),
                    }
                    if args.keep_raw:
                        payload["raw_response"] = raw
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
