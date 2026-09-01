from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rubric_rl import extract_qa_mc_criteria, prepare_cot_gt_v2
from rubric_rl.artifacts import (
    ArtifactError,
    AtomicJsonlArtifact,
    freeze_source_records,
    iter_jsonl_objects,
    load_jsonl_strict,
    sha256_file,
)


TEST_RUN_CONTRACT = {
    "operation": "artifact_unit_test",
    "model": {"id": "model-a", "revision": "revision-a"},
    "generation": {"max_new_tokens": 32},
}


def test_atomic_jsonl_commit_writes_hash_bound_inventory(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"sample_id":"input"}\n', encoding="utf-8")
    output = tmp_path / "criteria.jsonl"
    with AtomicJsonlArtifact(
        output,
        resume=False,
        rubric_version="test_v1",
        run_contract=TEST_RUN_CONTRACT,
        source_paths=(source,),
    ) as artifact:
        artifact.append({"sample_id": "s1", "value": 1})
        artifact.append({"sample_id": "s2", "value": 2})
        inventory = artifact.commit()
    assert inventory["rows"] == 2
    assert inventory["unique_ids"] == 2
    assert inventory["artifact_sha256"] == sha256_file(output)
    sidecar = json.loads(
        output.with_name(output.name + ".inventory.json").read_text(encoding="utf-8")
    )
    assert sidecar["sources"][0]["sha256"] == sha256_file(source)
    assert sidecar["run_contract"] == TEST_RUN_CONTRACT
    assert len(sidecar["run_contract_sha256"]) == 64
    assert [row["sample_id"] for row in load_jsonl_strict(output)] == ["s1", "s2"]


def test_atomic_session_preserves_last_committed_output_on_failure(tmp_path):
    output = tmp_path / "result.jsonl"
    output.write_text('{"sample_id":"old"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        with AtomicJsonlArtifact(
            output,
            resume=False,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ) as artifact:
            artifact.append({"sample_id": "new"})
            raise RuntimeError("injected failure")
    assert output.read_text(encoding="utf-8") == '{"sample_id":"old"}\n'
    assert output.with_name(output.name + ".partial").is_file()


def test_resume_recovers_only_complete_partial_rows_and_rejects_duplicates(tmp_path):
    output = tmp_path / "result.jsonl"
    partial = output.with_name(output.name + ".partial")
    with pytest.raises(RuntimeError):
        with AtomicJsonlArtifact(
            output,
            resume=False,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ) as artifact:
            artifact.append({"sample_id": "s1", "value": 1})
            raise RuntimeError("injected crash")
    with partial.open("ab") as handle:
        handle.write(b'{"sample_id":"crashed"')
    with AtomicJsonlArtifact(
        output,
        resume=True,
        rubric_version="test_v1",
        run_contract=TEST_RUN_CONTRACT,
    ) as artifact:
        assert artifact.done_ids == {"s1"}
        with pytest.raises(ArtifactError, match="duplicate"):
            artifact.append({"sample_id": "s1"})
        artifact.append({"sample_id": "s2"})
        inventory = artifact.commit()
    assert inventory["recovered_unterminated_tail"] is True
    assert [row["sample_id"] for row in load_jsonl_strict(output)] == ["s1", "s2"]


def test_resume_rejects_invalid_complete_line_and_duplicate_committed_id(tmp_path):
    output = tmp_path / "bad.jsonl"
    partial = output.with_name(output.name + ".partial")
    with pytest.raises(RuntimeError):
        with AtomicJsonlArtifact(
            output,
            resume=False,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ) as artifact:
            artifact.append({"sample_id": "seed"})
            raise RuntimeError("injected crash")
    partial.write_text('{"sample_id":"s1"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="invalid JSON"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ):
            pass
    partial.write_text('{"sample_id":"s1"}\n{"sample_id":"s1"}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ):
            pass


def test_resume_rejects_missing_or_mismatched_provenance(tmp_path):
    output = tmp_path / "result.jsonl"
    partial = output.with_name(output.name + ".partial")
    partial.write_text('{"sample_id":"s1"}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="without provenance"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
        ):
            pass

    partial.unlink()
    with AtomicJsonlArtifact(
        output,
        resume=False,
        rubric_version="test_v1",
        run_contract=TEST_RUN_CONTRACT,
    ) as artifact:
        artifact.append({"sample_id": "s1"})
        artifact.commit()
    with pytest.raises(ArtifactError, match="rubric_version"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v2",
            run_contract=TEST_RUN_CONTRACT,
        ):
            pass


def test_commit_rejects_source_changed_during_build(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"sample_id":"input"}\n', encoding="utf-8")
    output = tmp_path / "result.jsonl"
    with AtomicJsonlArtifact(
        output,
        resume=False,
        rubric_version="test_v1",
        run_contract=TEST_RUN_CONTRACT,
        source_paths=(source,),
    ) as artifact:
        artifact.append({"sample_id": "s1"})
        source.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
        with pytest.raises(ArtifactError, match="source changed"):
            artifact.commit()
    assert not output.exists()


def test_exact_source_snapshot_rejects_change_before_artifact_session(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"sample_id":"before"}\n', encoding="utf-8")
    frozen = freeze_source_records((source,))
    source.write_text('{"sample_id":"after"}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="exact-byte snapshot"):
        with AtomicJsonlArtifact(
            tmp_path / "output.jsonl",
            resume=False,
            rubric_version="test_v1",
            run_contract=TEST_RUN_CONTRACT,
            source_paths=(source,),
            expected_source_records=frozen,
        ):
            pass


def test_qa_producer_freezes_sources_before_model_load_and_forbids_fallback(
    tmp_path, monkeypatch
):
    source = tmp_path / "qa.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "QA_000001",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Question: Which option is correct?\n\n"
                            "Choose exactly one option:\n"
                            "A. alpha\nB. beta\nC. gamma\nD. delta"
                        ),
                    }
                ],
                "answer": "<answer>A</answer>",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("SYSTEM: strict\nUSER: {QA_JSON}\n", encoding="utf-8")
    output = tmp_path / "criteria.jsonl"

    class MutatingGenerator:
        def __init__(self, *args, **kwargs):
            del args
            assert kwargs["allow_attention_fallback"] is False
            self.effective_attn_implementation = kwargs["attn_implementation"]
            prompt.write_text(
                "SYSTEM: changed-after-snapshot\nUSER: {QA_JSON}\n",
                encoding="utf-8",
            )

        def generate(self, messages, *, max_new_tokens):
            del messages, max_new_tokens
            return "{}"

    monkeypatch.setattr(
        extract_qa_mc_criteria, "QwenTextGenerator", MutatingGenerator
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_qa_mc_criteria.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--model",
            "model-a",
            "--model-revision",
            "immutable-revision-a",
            "--prompt",
            str(prompt),
            "--attn-implementation",
            "flash_attention_2",
            "--fallback-deterministic",
        ],
    )
    with pytest.raises(ArtifactError, match="source changed"):
        extract_qa_mc_criteria.main()

    state = json.loads(
        output.with_name(output.name + ".partial.state.json").read_text(
            encoding="utf-8"
        )
    )
    source_paths = {record["path"] for record in state["sources"]}
    assert str(
        Path(extract_qa_mc_criteria.artifact_support.__file__).resolve()
    ) in source_paths
    generation = state["run_contract"]["generation"]
    assert generation["effective_attn_implementation"] == "flash_attention_2"
    assert generation["attention_backend_fallback"] == "forbidden"


def test_resume_rejects_changed_run_contract_for_partial_and_committed_artifacts(tmp_path):
    output = tmp_path / "result.jsonl"
    original = {
        "operation": "judge",
        "model": {"id": "model-a", "revision": "rev-a"},
        "generation": {"max_new_tokens": 64},
        "limit": 1,
    }
    changed = {
        **original,
        "model": {"id": "model-b", "revision": "rev-b"},
    }

    with pytest.raises(RuntimeError):
        with AtomicJsonlArtifact(
            output,
            resume=False,
            rubric_version="test_v1",
            run_contract=original,
        ) as artifact:
            artifact.append({"sample_id": "s1"})
            raise RuntimeError("injected crash")

    with pytest.raises(ArtifactError, match="provenance"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v1",
            run_contract=changed,
        ):
            pass

    with AtomicJsonlArtifact(
        output,
        resume=True,
        rubric_version="test_v1",
        run_contract=original,
    ) as artifact:
        artifact.commit()

    with pytest.raises(ArtifactError, match="run_contract"):
        with AtomicJsonlArtifact(
            output,
            resume=True,
            rubric_version="test_v1",
            run_contract=changed,
        ):
            pass


def test_run_contract_is_detached_and_rejects_non_json_values(tmp_path):
    output = tmp_path / "result.jsonl"
    contract = {"operation": "test", "nested": {"model": "model-a"}}
    artifact = AtomicJsonlArtifact(
        output,
        resume=False,
        rubric_version="test_v1",
        run_contract=contract,
    )
    contract["nested"]["model"] = "mutated-after-construction"
    with artifact:
        artifact.append({"sample_id": "s1"})
        inventory = artifact.commit()
    assert inventory["run_contract"]["nested"]["model"] == "model-a"

    with pytest.raises(ArtifactError, match="finite JSON"):
        AtomicJsonlArtifact(
            tmp_path / "invalid.jsonl",
            resume=False,
            rubric_version="test_v1",
            run_contract={"temperature": float("nan")},
        )


def test_prepare_cot_limit_is_total_target_rows_across_resume(tmp_path, monkeypatch):
    source = tmp_path / "cot.jsonl"
    rows = [
        {
            "sample_id": f"s{index}",
            "video_name": f"v{index}",
            "description_json": {
                "sample_summary": f"summary {index}",
                "per_segment": [
                    {
                        "time_range": "0-1",
                        "cot_type": "motion",
                        "think": f"reason {index}",
                        "answer": f"answer {index}",
                    }
                ],
                "final_answer": f"final {index}",
            },
        }
        for index in range(3)
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "gt.jsonl"

    base_argv = [
        "prepare_cot_gt_v2.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--limit",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    assert prepare_cot_gt_v2.main() == 0
    monkeypatch.setattr(sys, "argv", [*base_argv, "--resume"])
    assert prepare_cot_gt_v2.main() == 0

    assert [row["sample_id"] for row in load_jsonl_strict(output)] == ["s0"]
    inventory = json.loads(
        output.with_name(output.name + ".inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["rows"] == 1
    assert inventory["run_contract"]["limit"] == 1


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"sample_id":"s","sample_id":"t"}\n', "duplicate JSON key"),
        ('{"sample_id":"s","value":NaN}\n', "non-finite"),
        ('[]\n', "must be an object"),
        ('\n', "blank JSONL"),
    ],
)
def test_input_jsonl_reader_rejects_ambiguous_or_non_object_rows(tmp_path, line, message):
    path = tmp_path / "input.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ArtifactError, match=message):
        list(iter_jsonl_objects(path))
