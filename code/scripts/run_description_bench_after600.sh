#!/usr/bin/env bash
set -euo pipefail

cd /wangbenyou-sulongjie/qwen-vl-finetune

PY=/wangbenyou-sulongjie/anaconda3/envs/qwen3_vl/bin/python3.10
OUT=outputs/benchmark/description_sftstyle_after600/20260705_163500
LOG_DIR="$OUT/logs"
SHARD_DIR="$OUT/shards"
BENCH=data/benchmark/text/description/description_500.jsonl
NEW_CKPT=/wangbenyou-sulongjie/qwen-vl-finetune/outputs/grpo/motionx_374_hard_mcq_r8_b4_600_train2270_val100vm_len512_sftstyle/v6-20260702-212510/checkpoint-600
PROCESSOR=/wangbenyou-sulongjie/qwen-vl-finetune/checkpoints/sft_checkpoint_0426
PREGRPO=/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/description_eval/compare_sample500_pregrpo_old600_vm/pregrpo_vm_desc_sample500_predictions_final.jsonl
OLD600=/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/description_eval/compare_sample500_pregrpo_old600_vm/old600_vm_desc_sample500_predictions_final.jsonl
PREV_JUDGE=/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/description_eval/compare_sample500_pregrpo_old600_vm/judge_qwen3vl4b_semantic_summary_vm_sample500.json

mkdir -p "$OUT" "$LOG_DIR" "$SHARD_DIR"

echo "[desc-supervisor] start $(date -Is)"
echo "[desc-supervisor] out=$OUT"

"$PY" - <<'PY'
import json
from pathlib import Path

src = Path("data/benchmark/text/description/description_500.jsonl")
out = Path("outputs/benchmark/description_sftstyle_after600/20260705_163500")
shard_dir = out / "shards"
rows = []
for line in src.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    row["branch"] = "vm"
    rows.append(row)
(out / "description_500_vm_branch.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
for i in range(4):
    shard = rows[i * 125 : (i + 1) * 125]
    (shard_dir / f"description_500_vm_branch_shard{i}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard),
        encoding="utf-8",
    )
print({"total": len(rows), "shards": [len(rows[i * 125 : (i + 1) * 125]) for i in range(4)]})
PY

echo "[desc-supervisor] launch generation shards $(date -Is)"
pids=()
for i in 0 1 2 3; do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$i"
    export TRANSFORMERS_VERBOSITY=error
    "$PY" tools/eval_description_benchmark_sftstyle.py \
      --checkpoint "$NEW_CKPT" \
      --processor "$PROCESSOR" \
      --dataset "$SHARD_DIR/description_500_vm_branch_shard${i}.jsonl" \
      --output "$OUT/new600_vm_desc_shard${i}.jsonl" \
      --branch vm \
      --max_new_tokens 2048 \
      --device cuda:0 \
      --sft_motion_placeholders
  ) > "$LOG_DIR/gen_shard${i}.log" 2>&1 &
  pids+=("$!")
  echo "[desc-supervisor] shard${i}_pid=${pids[-1]}"
done

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    echo "[desc-supervisor] shard${idx} done $(date -Is)"
  else
    echo "[desc-supervisor] shard${idx} FAILED $(date -Is)"
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "[desc-supervisor] generation failed"
  exit 1
fi

echo "[desc-supervisor] merge generation $(date -Is)"
cat "$OUT"/new600_vm_desc_shard0.jsonl \
    "$OUT"/new600_vm_desc_shard1.jsonl \
    "$OUT"/new600_vm_desc_shard2.jsonl \
    "$OUT"/new600_vm_desc_shard3.jsonl \
    > "$OUT/new600_vm_desc_sample500_predictions.jsonl"

"$PY" - <<'PY'
import json
from collections import Counter
from pathlib import Path

out = Path("outputs/benchmark/description_sftstyle_after600/20260705_163500")
rows = [json.loads(line) for line in (out / "new600_vm_desc_sample500_predictions.jsonl").open(encoding="utf-8") if line.strip()]
for idx, row in enumerate(rows):
    row["sample500_order"] = idx
    row["index"] = idx
(out / "new600_vm_desc_sample500_predictions.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
n = len(rows)
summary = {
    "n": n,
    "errors": sum(1 for row in rows if row.get("error")),
    "json_parse_ok": sum(1 for row in rows if row.get("json_parse_ok")),
    "json_parse_ok_rate": sum(1 for row in rows if row.get("json_parse_ok")) / n if n else 0,
    "statuses": dict(Counter(row.get("json_status") for row in rows)),
    "avg_chars": sum(row.get("prediction_chars", 0) for row in rows) / n if n else 0,
    "avg_final_answer_chars": sum(row.get("final_answer_len", 0) for row in rows) / n if n else 0,
    "max_chars": max((row.get("prediction_chars", 0) for row in rows), default=0),
}
(out / "new600_vm_desc_sample500_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[desc-supervisor] launch judge $(date -Is)"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_VERBOSITY=error
"$PY" tools/judge_description_semantic_qwen3vl.py \
  --benchmark "$BENCH" \
  --pregrpo "$PREGRPO" \
  --old600 "$OLD600" \
  --new600 "$OUT/new600_vm_desc_sample500_predictions.jsonl" \
  --output "$OUT/judge_qwen3vl4b_semantic_all3_sample500.jsonl" \
  --device cuda:0 \
  > "$LOG_DIR/judge_all3.log" 2>&1

echo "[desc-supervisor] aggregate final table $(date -Is)"
"$PY" - <<'PY'
import json
from pathlib import Path

out = Path("outputs/benchmark/description_sftstyle_after600/20260705_163500")
new_gen = json.loads((out / "new600_vm_desc_sample500_summary.json").read_text(encoding="utf-8"))
judge = json.loads((out / "judge_qwen3vl4b_semantic_all3_sample500.summary.json").read_text(encoding="utf-8"))
prev_path = Path("/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune/outputs/description_eval/compare_sample500_pregrpo_old600_vm/judge_qwen3vl4b_semantic_summary_vm_sample500.json")
prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None
table = {
    "result_dir": str(out),
    "new_checkpoint": "/wangbenyou-sulongjie/qwen-vl-finetune/outputs/grpo/motionx_374_hard_mcq_r8_b4_600_train2270_val100vm_len512_sftstyle/v6-20260702-212510/checkpoint-600",
    "generation_summary": {"new600_vm": new_gen},
    "judge_all3": judge,
    "previous_pairwise_pregrpo_old600": prev,
}
(out / "description_compare_summary_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(table, ensure_ascii=False, indent=2))
PY

echo "[desc-supervisor] done $(date -Is)"
