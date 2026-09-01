# 远端固定执行流程

所有新任务都从以下 controller 执行：

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval`

旧实验目录只保留源码、配置、数据、日志和证据。历史 finetune 权重已经
清理，也不能满足新批次的 fresh-finetune 门禁。

## 0. 同步与只读自检

本地更新流程文件后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File D:\MotionLLM\model_evaluation_agent\scripts\sync_to_remote.ps1
```

同步脚本会运行纯元数据 self-test，不加载模型。

远端开始前检查：

```bash
cd /wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval
sed -n '1,240p' 00_从这里开始.md
sed -n '1,320p' MEMORY.md
sed -n '1,360p' RUNBOOK.md
cat model_registry.json
cat pretrained_registry.json
ls -1t server_audit/ | head
nvidia-smi
df -h .
```

不要停止、修改或复用不属于本批次的活动进程。

## 1. 创建并冻结批次

```bash
python3 scripts/new_batch.py <batch_id> --description "<description>"
```

把 train、validation、benchmark、video、motion、canonical QA 和派生代码
登记到 `batches/<batch_id>/00_inputs/`，记录绝对路径、行数、revision 和
SHA-256。

完成以下泄漏检查：

- sample ID、group ID
- 媒体内容 hash
- 归一化 question/options
- exact duplicate 和 near duplicate

未冻结唯一 benchmark 或泄漏未解决时，不得训练。

## 2. 校验 canonical pretrain

```bash
python3 scripts/validate_pretrained_ready.py
```

预期输出包括：

```text
PRETRAIN_READY models=15
```

所有模型只能从 `shared_assets/pretrained/` 中登记的 canonical 输入启动。
不要把 base 复制进批次目录。VideoLLaMA 的 torch.hub 权重和 MotionLLM 的
隔离 runtime 已经包含在 pretrain gate 内。

## 3. 固定 15 个 finetune 任务

为 `model_registry.json` 的每个模型创建：

`batches/<batch_id>/02_finetune/<model_id>/run_manifest.json`

manifest 至少冻结：

- train/validation 路径、行数和 SHA-256
- benchmark 排除和泄漏审计 SHA-256
- base/pretrain 路径、revision 和 SHA-256
- 模态输入与预处理
- optimizer、schedule、epochs/steps、precision、seed
- 代码 revision/dirty state
- 计划输出路径

AGCN 必须使用正式实现并按官方配方从随机初始化训练。MotionCLIP 必须使用
正式实现。MotionLLM 使用项目专用 runner，但 manifest 必须明确标记
`runner_ownership=project_owned`，不能伪称 upstream finetune script。

## 4. 先完成全模型 finetune

对所有未阻塞模型，用本批冻结的 train/validation 产生新 checkpoint 或
adapter。VideoLLaMA 和 MotionLLM 的已验证命令见 `RUNBOOK.md`。

每个成功模型记录：

- `status=finetune_complete`
- 当前批次 artifact 路径与 SHA-256
- 训练开始/结束时间
- 实际 step、loss、环境和显存信息

无法运行时只能写有具体证据的 `blocked`。`pending`、`failed`、历史权重或
proxy 都不能打开 eval。

## 5. 全局 finetune barrier

15 个模型全部成为 `finetune_complete` 或证据充分的 `blocked` 后：

```bash
python3 scripts/validate_batch.py --stage finetune batches/<batch_id>
python3 scripts/open_eval_stage.py batches/<batch_id>
```

`open_eval_stage.py` 会在 barrier 未关闭时拒绝创建 `03_eval/`。

## 6. 逐模型 eval smoke

只对 `finetune_complete` 模型运行：

1. `smoke_1`
2. `smoke_8`
3. `smoke_32`

每个目录写 `status.json`，至少包含 `status=passed` 和实际样本数。必须检查：

- 模态确实被读取
- 题目和选项顺序一致
- 生成 parser 只接受完整 `<answer>[A-D]</answer>`
- invalid/OOM/timeout/media/runtime error 保留在固定分母
- memory、速度和输出 schema

## 7. 打开 full eval

```bash
python3 scripts/open_full_eval.py batches/<batch_id> --all
```

任一模型的 1/8/32 smoke 未通过时，脚本不会为该模型创建 `full/`。

正式评估只读取本批次 artifact 的 SHA-256，不得切回 base、历史 adapter、
baseline 或 option-score 诊断。

## 8. Release

```bash
python3 scripts/validate_batch.py --stage release batches/<batch_id>
```

每个模型必须交付：

- `predictions.jsonl`
- `summary.json`
- `run_manifest.json`
- `status.md`

批次必须交付：

- `all_models_results.csv`
- `all_models_results.md`
- `blocked_models.md`
- `evaluation_release_manifest.json`

## 禁止事项

- 不得在全模型 finetune barrier 前开始 eval。
- 不得复用历史或 smoke 权重作为本批 finetune artifact。
- 不得把旧 accuracy、旧 prediction、baseline、proxy 或诊断分数放入主表。
- 不得针对不同模型改变 benchmark、选项顺序或固定分母。
- 不得删除 invalid、OOM、timeout、媒体错误或 runtime error 行。
- 不得把服务器凭据写入脚本、日志、manifest 或回复。
