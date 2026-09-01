# MotionLLM 常用命令

以下命令与当前 `python -m motion_eval --help` 对齐。示例中的路径是占位符，正式运行前必须替换并先执行 `--dry-run`。

> **当前 production 状态（2026-08-21）：**所有 catalog production
> finetune/evaluation/verifier 与 formal Qwen SFT 都会以
> `blocker=verified-multi-root-bootstrap` 主动拒绝。下面的 controller 命令用于
> 检查计划、preflight 和未来解除 blocker 后的标准流程；不得删除 blocker 或把
> preflight/debug artifact 手工提交为正式结果。

## 1. 本地环境

```powershell
$Repo = 'D:\MotionLLM\motionllm_refactor'
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$Git = 'C:\Users\caim33\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
Set-Location $Repo

& $Py --version
& $Py -m motion_eval --help
```

不要读取、打印或复制连接文件内容：

```powershell
# 禁止：Get-Content D:\MotionLLM\dev_env_connection.txt
```

## 2. 代码检查

完整门禁：

```powershell
& $Py scripts\run_checks.py
```

分项执行：

```powershell
& $Py -m compileall -q src models qwenvl model_evaluation_agent
& $Py -m pytest tests\unit -q
& $Py -m pytest tests\contract -q
& $Py -m pytest tests\integration -q
& $Py -m pytest tests\stress -q
& $Py scripts\secret_scan.py
& $Git status --short
& $Git diff --check
```

单个测试：

```powershell
& $Py -m pytest tests\unit\test_strict_parser.py -q
& $Py -m pytest tests\integration\test_controller_workflow.py -k smoke -vv
```

## 3. 统一 CLI 公共路径

正式批次建议显式设置所有根目录：

```powershell
$Batches = 'D:\runtime\motionllm\batches'
$Keepalive = 'D:\runtime\motionllm\keepalive'
$RunnerRoot = 'D:\MotionLLM\motionllm_refactor\model_evaluation_agent'
$PretrainedRoot = 'D:\runtime\motionllm\pretrained'
$Registry = 'model_evaluation_agent\model_registry.json'
$PretrainedRegistry = 'model_evaluation_agent\pretrained_registry.json'
$CodeRoot = 'src\motion_eval'
$Batch = 'qa500v2_YYYYMMDD_<hash>'
```

下文省略重复的 registry/root 参数以便阅读；正式运行可以附加：

```text
--workspace-root <batches>
--registry <model_registry.json>
--pretrained-registry <pretrained_registry.json>
--code-root <src/motion_eval>
--runner-root <model_evaluation_agent>
--pretrained-root <pretrained root>
--controller-interpreter <absolute python>
--keepalive-root <keepalive root>
--keepalive-owner motionllm
```

## 4. Registry 与 GPU 只读检查

```powershell
& $Py -m motion_eval registry validate --dry-run
& $Py -m motion_eval registry validate
& $Py -m motion_eval gpu status --dry-run
& $Py -m motion_eval gpu status
```

`gpu status` 失败时必须停止；不能把查询失败解释为 GPU 空闲。

## 5. 创建不可变批次

六个输入角色是：`train`、`validation`、`benchmark`、`media_manifest`、`derivation_code`、`leakage_audit`。

```powershell
$Config = 'D:\data\qa500v2\batch_config.json'
$Train = 'D:\data\qa500v2\train.jsonl'
$Validation = 'D:\data\qa500v2\validation.jsonl'
$Benchmark = 'D:\data\qa500v2\benchmark.jsonl'
$Media = 'D:\data\qa500v2\media_manifest.json'
$Derivation = 'D:\data\qa500v2\derive.py'
$Leakage = 'D:\data\qa500v2\leakage_audit.json'

& $Py -m motion_eval batch create $Batch `
  --workspace-root $Batches `
  --registry $Registry `
  --pretrained-registry $PretrainedRegistry `
  --code-root $CodeRoot `
  --runner-root $RunnerRoot `
  --pretrained-root $PretrainedRoot `
  --controller-interpreter $Py `
  --keepalive-root $Keepalive `
  --input "train=$Train" `
  --input "validation=$Validation" `
  --input "benchmark=$Benchmark" `
  --input "media_manifest=$Media" `
  --input "derivation_code=$Derivation" `
  --input "leakage_audit=$Leakage" `
  --config $Config `
  --description 'fresh finetune before QA500-v2 evaluation' `
  --dry-run
```

确认 dry-run 输出后，去掉最后的 `--dry-run` 创建批次。批次 ID 和批次目录是不可变的；失败重试请使用新 ID。

验证与查看计划：

```powershell
& $Py -m motion_eval batch validate $Batch --workspace-root $Batches
& $Py -m motion_eval plan $Batch --workspace-root $Batches --dry-run
```

## 6. Fresh finetune

当前只有 finetune `--purpose preflight` 可以实际启动，且只有
`videollama_lora` 与 `motionllm_official` 有 reviewed finetune backend。
`--purpose production` 会在 GPU lease、状态事件和 Python spawn 前拒绝；
`finetune complete` 也会拒绝已有产物。完整 backend 状态见
`model_evaluation_agent\RUNNER_BACKENDS.md`。

先生成模型 ID 列表：

```powershell
& $Py -m motion_eval registry validate
```

单模型生产 attempt：

```powershell
$Model = 'qwen36_27b_lora'
$Attempt = 'ft_001'
$Gpu = 'GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'

& $Py -m motion_eval finetune attempt $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --attempt-id $Attempt `
  --python-executable $Py `
  --gpu $Gpu `
  --purpose production `
  --dry-run
```

确认后去掉 `--dry-run`，再执行冻结命令：

```powershell
& $Py -m motion_eval finetune run-attempt $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --attempt-id $Attempt
```

worker 与独立 reload verifier 都成功后，提交 manifest：

```powershell
$Manifest = Join-Path $Batches "$Batch\02_finetune\$Model\attempts\$Attempt\run_manifest.json"
& $Py -m motion_eval finetune complete $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --attempt-id $Attempt `
  --manifest $Manifest
```

查看 15 模型全局屏障：

```powershell
& $Py -m motion_eval finetune barrier $Batch --workspace-root $Batches
```

`finetune block` 只允许 registry 中确实缺失的路径、代码或权重，并需要具体 component/evidence。运行失败、OOM、超时和环境问题属于可重试错误，不能伪造成永久 blocker。

## 7. 打开评估并按 1 → 8 → 32 → 500 执行

只有所有模型 finetune 达到 `complete` 或正式 `blocked` 后：

```powershell
& $Py -m motion_eval gate open-eval $Batch --workspace-root $Batches --dry-run
& $Py -m motion_eval gate open-eval $Batch --workspace-root $Batches
```

创建并运行 smoke attempt：

```powershell
$Stage = 'smoke_1'   # 然后依次 smoke_8、smoke_32
$Attempt = 'eval_001'

& $Py -m motion_eval eval attempt $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --stage $Stage `
  --attempt-id $Attempt `
  --python-executable $Py `
  --gpu $Gpu `
  --dry-run

& $Py -m motion_eval eval run-attempt $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --stage $Stage `
  --attempt-id $Attempt
```

验证 prediction：

```powershell
$Predictions = Join-Path $Batches "$Batch\03_eval\$Model\$Stage\attempts\$Attempt\predictions.jsonl"
& $Py -m motion_eval eval smoke $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --size 1 `
  --attempt-id $Attempt `
  --predictions $Predictions
```

所有可评估模型依次通过 1、8、32 后：

```powershell
& $Py -m motion_eval gate open-full $Batch --workspace-root $Batches --dry-run
& $Py -m motion_eval gate open-full $Batch --workspace-root $Batches
```

随后以 `--stage full` 创建/运行 attempt，并用以下命令验证固定 500 条：

```powershell
& $Py -m motion_eval eval full $Batch `
  --workspace-root $Batches `
  --model-id $Model `
  --attempt-id $Attempt `
  --predictions $Predictions
```

## 8. Release

```powershell
& $Py -m motion_eval release build $Batch --workspace-root $Batches --dry-run
& $Py -m motion_eval release build $Batch --workspace-root $Batches
& $Py -m motion_eval release verify $Batch --workspace-root $Batches
& $Py -m motion_eval batch validate $Batch --workspace-root $Batches
```

release 只能包含当前批次由 controller 验证的 fresh artifact 和 prediction。

## 9. GPU keepalive

推荐命令是 `gpu keepalive`；旧 `keepalive status` 仅为 deprecated alias。

只读计划：

```powershell
& $Py -m motion_eval gpu keepalive start `
  --root $Keepalive `
  --gpu-uuid $Gpu `
  --dry-run
```

在真实 `nvidia-smi` 证明该卡空闲后启动：

```powershell
& $Py -m motion_eval gpu keepalive start `
  --root $Keepalive `
  --gpu-uuid $Gpu

& $Py -m motion_eval gpu keepalive status --root $Keepalive
```

正式 finetune/eval 前停止同一 UUID：

```powershell
& $Py -m motion_eval gpu keepalive stop `
  --root $Keepalive `
  --gpu-uuid $Gpu `
  --dry-run

& $Py -m motion_eval gpu keepalive stop `
  --root $Keepalive `
  --gpu-uuid $Gpu
```

keepalive、finetune 和 eval 共用同一 GPU UUID role mutex；不得手工删除 lease 文件或按模糊进程名批量 kill。

## 10. 服务器只读审计

连接工具必须使用已固定 host key，并从安全连接来源把凭据读入进程级环境变量。不要把密码作为命令参数、环境回显或脚本内容。

登录后首先执行只读检查：

```bash
set -euo pipefail
date
hostname
pwd
df -h
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
pgrep -af 'motionllm|motion_eval|qwen|swift|finetune|keepalive' || true
```

再逐一验证远端候选根目录，不要直接执行带 `--delete` 的同步。同步前必须 dry-run、排除 `History`、连接文件、`.git`、cache、data、weights 和 outputs。

## 11. Qwen SFT 与 catalog production blocker

帮助与本地检查仍可查看：

```powershell
& $Py qwenvl\train\full_sft.py --help
& $Py qwenvl\train\lora_sft.py --help
& $Py model_evaluation_agent\scripts\finetune_videollama_lora.py --help
& $Py model_evaluation_agent\scripts\finetune_motionllm_lora.py --help
```

Linux shell 的 `--help` 也应可查看，但 formal 执行会在任何 Python probe 前
退出 78：

```bash
bash scripts/full_sft.sh --help
bash scripts/lora_sft.sh --help
```

`--unsafe_legacy_no_manifest` 仅供隔离 smoke/debug，不得写入 batch、resume、
release 或 evaluation。当前本机 `.venv` 是 editable 安装且为 CPU PyTorch，
不具备 formal 资格。

解除 blocker 前必须完成 `docs/SFT_FORMAL_BOOTSTRAP_PLAN.md` 中的
controller-verified `-I -S -B` 多根 in-memory bundle、外部 HMAC pre-spawn
snapshot、环境正向白名单与 post audit。任何 post-import `__file__` 检查都不能
替代这一步。

两个已实现 backend 的非发布 preflight 依赖检查：

```powershell
& $Py model_evaluation_agent\scripts\preflight_runner_dependencies.py `
  --model-id videollama_lora `
  --pretrained-root D:\runtime\motionllm\pretrained

& $Py model_evaluation_agent\scripts\preflight_runner_dependencies.py `
  --model-id motionllm_official `
  --pretrained-root D:\runtime\motionllm\pretrained
```

远端 Linux/CUDA 审计完成后才可增加 `--require-cuda`。preflight 成功仍不等于
production artifact。

## 12. Formal GRPO LoRA checkpoint 门禁

Formal Motion-R1 GRPO 只能从复制并完整解析后的 LoRA 模板启动：

```powershell
$FormalGrpo = 'D:\runtime\motionllm\batches\<batch_id>\configs\motionr1_vm_lora_grpo.yaml'
$env:MOTION_GRPO_PYTHON = $Py
& $Py qwenvl/grpo_ms_swift/runner/train_grpo_ms_swift.py --help
& bash scripts/train_grpo_ms_swift.sh --config $FormalGrpo --dry_run
& bash scripts/train_grpo_ms_swift.sh --config $FormalGrpo --preflight_only
& bash scripts/train_grpo_ms_swift.sh --config $FormalGrpo
```

配置必须显式使用 `training.tuner_type: lora`、
`training.save_safetensors: true` 和唯一的
`motion_training_receipt` callback。最终产物必须是精确的
`checkpoint-N/adapter_model.safetensors`；callback 会把最终 live PEFT 状态
（LoRA 与 `modules_to_save`）逐 tensor 绑定到磁盘 keys、dtype、shape 和原始
bytes，并在 v2 receipt 中记录 canonical state SHA-256 与 payload SHA-256。
同一 receipt 还会绑定严格的 `adapter_config.json` 与 ms-swift
`additional_config.json`：记录各自原始 SHA-256/大小和规范化语义 SHA-256，
并核对 live LoRA、正式 YAML 和独立冻结重载。`additional_config.json` 只允许
`lora_dtype`、`lorap_lr_ratio`、`lorap_emb_lr` 三个字段。
缺失/全零 gradient、AMP skipped step、额外未保存的 trainable、旧权重、
配置字段漂移、symlink、空权重或运行中替换都会中止发布。
正式 shell 启动器和已验证 Swift 入口都会通过同一个绑定解释器以
`-I -B` 隔离模式运行；不要绕过 `scripts/train_grpo_ms_swift.sh` 执行训练。

Formal runtime 的完整环境快照必须包含 lock 中全部精确版本，包括
`accelerate==1.13.0`、`peft==0.18.0` 与 `safetensors==0.8.0`。

## 13. Rubric RL 严格入口

先检查真实参数；旧 segmented V2 入口已禁用：

```powershell
& $Py -m rubric_rl.extract_qa_mc_criteria --help
& $Py -m rubric_rl.judge_qa_mc --help
& $Py -m rubric_rl.prepare_cot_gt_v2 --help
& $Py -m rubric_rl.extract_motion_criteria_v2 --help
& $Py -m rubric_rl.judge_motion_caption_v2 --help
```

QA criteria 示例：

```powershell
& $Py -m rubric_rl.extract_qa_mc_criteria `
  --input D:\data\qa_train.jsonl `
  --output D:\runtime\rubric\qa_criteria.jsonl `
  --model D:\models\judge `
  --model-revision REPLACE_IMMUTABLE_REVISION `
  --prompt rubric_rl\prompt_templates\qa_mc_offline_prompt.txt `
  --resume
```

Motion V2 criteria 示例：

```powershell
& $Py -m rubric_rl.extract_motion_criteria_v2 `
  --input D:\data\motion_gt.jsonl `
  --output D:\runtime\rubric\motion_v2_criteria.jsonl `
  --model D:\models\judge `
  --model-revision REPLACE_IMMUTABLE_REVISION `
  --resume
```

最终 JSONL 只在整批成功后原子发布；同时生成
`<output>.inventory.json`，记录数据 SHA-256、行数、唯一 ID 数和输入文件
hash。中断进度保存在 `<output>.partial`，只能通过 `--resume` 继续。

ms-swift 注册名为 `qa_mc_rubric` 和 `motion_rubric_v2`。若没有随数据提供
严格校验过的 judgment，必须通过进程环境配置在线 judge；非本机地址只接受
HTTPS。token 不得写入 YAML、命令文档或 manifest。
