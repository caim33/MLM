# MotionLLM 重构验证报告

更新时间：2026-08-21（Asia/Shanghai）

## 结论

当前代码已经完成本地安全收口，但答案不是“SFT、GRPO、Rubric RL 和全模型
评估都能正式正常执行”。准确状态如下：

| 路径 | 本地状态 | 正式运行状态 |
|---|---|---|
| GRPO | checkpoint/config/receipt/reload 逻辑通过 | 未做 Linux/CUDA 真实训练 |
| Rubric RL | QA 与 Motion V2 schema/reward/judge 逻辑通过 | 未做在线 judge、重型模型或 CUDA 实跑 |
| Qwen full/LoRA SFT | 训练证据、artifact 和攻击测试通过 | formal 路径主动 fail-closed |
| 15 模型 catalog | controller、屏障、receipt 和状态机通过 | 所有 production finetune/eval/verifier 主动 fail-closed |

统一 production blocker 是：

```text
blocker=verified-multi-root-bootstrap
```

当前安全结论来自“正式发布路径已在 Python 启动前关闭”，不是来自“多根可信
执行已经实现”。只有 finetune preflight/debug 可以运行，其产物不能提交
`complete_finetune`、不能打开 evaluation，也不能进入主结果。

## 全仓门禁

执行：

```powershell
Set-Location D:\MotionLLM\motionllm_refactor
.\.venv\Scripts\python.exe scripts\run_checks.py
```

最终结果：

- Python 编译：通过；
- 凭据扫描：通过；
- pytest：`922 passed, 14 skipped, 2 warnings in 438.27s`；
- 标准门禁退出码：0；
- `pip check`：无损坏依赖；
- `git diff --check`：通过，仅有既有 PowerShell 文件 LF/CRLF 提示。

14 个 skip 均为当前 Windows 主机缺少 Bash、不能创建 symlink/junction，或 venv
没有 symlink launcher。它们不是通过项，必须在 Linux 远端补跑。两个 warning
分别来自 CPU 环境的 `pin_memory` 和 PEFT 自动保存已 resize embedding。

主要定向结果（互相重叠，不能与全仓计数相加）：

- SFT/catalog：`186 passed, 2 skipped`；
- controller workflow：`18 passed, 1 skipped`；
- controller state：`16 passed`；
- snapshot security：`13 passed`；
- catalog contract：`19 passed`；
- GRPO + Rubric RL 联合套件：`298 passed, 6 skipped`；
- trusted pretrained index/controller：`20 passed`。

独立 reviewer 最终复跑 omit source、`.cache/.git`、内部/外部 `sys.path`、
repo-root 注入、`.pth`、native loader、meta path、Windows junction、role-split、
pre-spawn blocker 和 complete 绕过后，未发现当前发布边界内剩余 P0/P1。

## SFT

已修复并验证：

- formal source inventory 由 producer 和独立 parser 各自重建完整 `src`、
  `models`、`qwenvl` 清单并精确比对；
- source-backed bytecode、`__pycache__`、symlink/reparse、`.cache/.git`、遗漏或
  额外源码均 fail-closed；
- environment snapshot 绑定 interpreter、stdlib、distribution RECORD、实际
  installed files、`.pth`、`sys.path`、`meta_path`、native runtime 和 loader
  环境；
- 目录 symlink/junction、外部路径、editable environment、sourceless bytecode
  和自洽伪造快照均拒绝；
- training receipt、artifact manifest 和独立 reload verifier 重新解析并绑定
  file SHA、自哈希、pre/post、code/runner/environment/data/base/config；
- ZeRO 配置固定为 `scripts/zero2.json`，当前证明不支持 ZeRO-3；
- gradient evidence 在 device 侧聚合，每个 optimizer step 只做一次三标量 host
  sync（分布式时另有一次 MAX reduction）。

Formal Qwen shell 在任何 Python probe 前退出 78，Python worker 也作纵深拒绝。
所有 catalog production attempt 会在 GPU lease、`ATTEMPT_STARTED` 和任何
Python spawn 前拒绝；`complete_finetune`、`complete_evaluation` 与独立
verifier 也不能绕过。

原因是现有 `python -I -c` 仍可在 bootstrap 前运行 system `sitecustomize`，
且单根 runner bundle 没有同时冻结 `motion_eval`、训练代码、依赖环境和多 rank
contract。删除 blocker 或只做 post-import `__file__` 检查都不构成修复。

解除条件见 `docs/architecture/formal/SFT_FORMAL_BOOTSTRAP_PLAN.md`：先实现 controller-verified
`-I -S -B` 多根 in-memory bundle、外部 HMAC pre-spawn snapshot 和 post audit，
再先开放 single-node/NPROC=1，最后单独实现和验证 N-rank。

## GRPO

Formal Motion-R1 LoRA GRPO 的本地门禁已覆盖：

- live PEFT trainable 与 `adapter_model.safetensors` 的 keys、dtype、shape、原始
  bytes 精确一致；
- `adapter_config.json` 与 ms-swift `additional_config.json` 同时绑定原始文件
  SHA/大小和规范化语义 SHA；
- 独立冻结 `PeftModel.from_pretrained(..., is_trainable=False)` 重载与磁盘状态
  精确一致；
- missing/extra/duplicate key、配置漂移、symlink、TOCTOU、全零 gradient、
  AMP skipped step、少跑或多跑 optimizer step 均拒绝；
- exact lock 包含 Accelerate 1.13.0、PEFT 0.18.0 与 safetensors 0.8.0。

本机真实 PEFT 重载测试已运行，但 PyTorch 是 CPU build；没有声称完成 CUDA
forward/backward、真实 ms-swift GPU 训练或新 checkpoint 发布。

## Rubric RL

QA Rubric 与 Motion Rubric V2 已有严格 schema、artifact inventory、offline
criteria、online adapter、deterministic reward、deadline 和对抗性测试。五个主
CLI 的 `--help` 均成功，旧 segmented Motion V2 入口保持禁用。

本轮没有调用真实 HTTPS judge，没有运行重型 Qwen judge，也没有做 CUDA/远端
GRPO 集成，因此不能把本地 mock/ORM 测试写成正式 RL 训练完成。

## Controller 与模型 backend

- registry 固定 15 个模型；每个新批次仍要求所有模型 fresh finetune 或正式
  blocker，之后才能全局打开 eval；
- 每个可评估模型仍按 `1 -> 8 -> 32 -> 500`；
- AGCN 和 MotionCLIP 都必须使用正式实现并 finetune，proxy 禁止；
- 大型 pretrained asset 在 batch freeze 与显式 phase/release audit 做全 SHA；
  普通 transition 使用外部 HMAC state 绑定的 immutable index，不反复重哈希
  约 245 GB 逻辑内容，也不虚假声称路径/mtime 能证明内容；
- 当前仅审查实现 `videollama_lora` 与 `motionllm_official` 的 finetune/reload
  backend；其他 13 个 finetune backend、全部 15 个 evaluation backend、其他
  13 个 reload backend 仍缺失；
- 即使补齐这些模型 backend，也必须先解除 multi-root bootstrap blocker。

详细矩阵见 `model_evaluation_agent/RUNNER_BACKENDS.md`。

## 本地与远端环境

本机：

```text
Python 3.12.13
PyTorch 2.7.0+cpu
CUDA available: false
Transformers 4.57.3
Accelerate 1.13.0
PEFT 0.18.0
safetensors 0.8.0
project install: editable
```

该环境只适合本地逻辑测试，不具备 formal SFT 发布资格。

新 SSH 端点本轮较早时曾 TCP 可达并完成 ED25519 host-key 固定；最终复查连续
4 次不可达，直接 socket 返回 `ConnectionRefused`。唯一允许的认证来源
`D:\MotionLLM\dev_env_connection.txt` 仍记录旧端口 `31349`（最后修改
2026-07-29），不是用户本轮指定的 `31976`。聊天密码没有进入命令、环境持久化
文件、代码、文档、日志或 manifest。因此本轮没有已认证 SSH、`nvidia-smi`、
Linux skip 回归、CUDA smoke、远端同步、fresh finetune 或 QA500-v2 evaluation。

## 未产生的结果

本轮没有生成新的 accuracy、模型排名、QA500-v2 predictions、正式 checkpoint
或 release。历史 accuracy、prediction、baseline、option-score/log-prob 与
AGCN/MotionCLIP proxy 继续作废。`History` 未修改。

工作区包含本轮未提交修改；本报告不声称存在新的 git commit。
