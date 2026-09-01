# MotionLLM 重构交接

更新时间：2026-08-21（Asia/Shanghai）

## 先看结论

活动仓库是：

```text
D:\MotionLLM\motionllm_refactor
```

`D:\MotionLLM\History` 是只读历史证据，不要在其中继续开发。当前本地测试门禁
已通过，但所有 catalog production finetune/evaluation/verifier 与 formal Qwen
SFT 仍因 `verified-multi-root-bootstrap` 主动阻断。不要删除 blocker 来获得
“能跑”的假象。

下一位 AI 的阅读顺序：

1. `D:\MotionLLM\AGENTS.md`
2. `D:\MotionLLM\model_evaluation_agent\CURRENT_REFACTOR_STATUS.md`
3. 本文件
4. `docs/VERIFICATION_REPORT.md`
5. `docs/COMMON_COMMANDS.md`
6. `docs/SFT_FORMAL_PROVENANCE.md`
7. `docs/SFT_FORMAL_BOOTSTRAP_PLAN.md`
8. `model_evaluation_agent/RUNNER_BACKENDS.md`

## 代码位置

| 路径 | 职责 |
|---|---|
| `src/motionllm/contracts` | sample、modality、option 等稳定契约 |
| `src/motionllm/data` | strict reader、message、dataset、collator |
| `src/motionllm/motion` | motion IO、normalization、temporal 处理 |
| `src/motionllm/fusion` | motion token、projector、embedding 注入 |
| `src/motionllm/models` | Qwen/MotionLLM 模型与 generation |
| `src/motionllm/training` | SFT、LoRA、证据、artifact/save/reload |
| `src/motionllm/grpo` | QA/Motion rubric、reward、Swift adapter |
| `src/motion_eval/data` | batch receipt、QA500、泄漏与 pretrained index |
| `src/motion_eval/controller` | batch、attempt、事件链、全局阶段屏障 |
| `src/motion_eval/runtime` | 进程、GPU lease、keepalive、SSH target |
| `src/motion_eval/evaluation` | strict answer parser 与 prediction schema |
| `src/motion_eval/reporting` | release manifest 与结果表 |
| `model_evaluation_agent` | 15 模型 registry、CLI facade、backend inventory |
| `qwenvl/train` | Qwen SFT 兼容入口；核心逻辑应回到 `src/` |
| `qwenvl/grpo_ms_swift` | ms-swift GRPO 兼容入口与 callback |
| `rubric_rl` | QA/Motion criteria、judge、artifact 与 reward CLI |
| `tests` | unit、contract、integration、stress |

旧 facade 只负责参数适配和调用统一模块，不能扩展成第二套状态机或第二份模型
实现。

## 固定项目规则

1. 新批次先冻结 train、validation、canonical QA500-v2、媒体、派生代码、option
   permutation 和 leakage audit。
2. 15 个 registry 模型都必须产生当前批次 fresh finetune artifact，或给出可
   复核 blocker；这是全局阶段屏障。
3. 所有 finetune 达到终态前，任何模型都不能开始正式 evaluation。
4. 可评估模型严格按 `1 -> 8 -> 32 -> 500`；invalid/runtime error 留在固定分母。
5. AGCN 与 MotionCLIP 都要 finetune；proxy 禁止进入主表。
6. 历史 accuracy、prediction、checkpoint、baseline、option-score/log-prob 和
   排名全部不能进入新主结果。
7. 凭据只从 `D:\MotionLLM\dev_env_connection.txt` 读入进程级环境变量，不能
   复制到代码、Markdown、JSON、CSV、日志、manifest 或回复。

## 当前模块状态

### SFT

SFT state binding、gradient/step evidence、receipt、artifact、resume/reload 与
source/environment snapshot 的本地测试已通过。ZeRO-2 固定，ZeRO-3 未证明。

Formal Qwen shell/Python 保持 fail-closed。所有 catalog production attempt 也
会在 GPU lease、事件 append 和 Python spawn 前拒绝；complete API 无法绕过。
preflight/debug 产物永远不能发布。

真正解除需要 controller-verified `-I -S -B` 多根内存 bundle、外部 HMAC
pre-spawn snapshot、positive env allowlist 与 post audit。先做 NPROC=1，再做
N-rank；详细契约见 `docs/SFT_FORMAL_BOOTSTRAP_PLAN.md`。

### GRPO

LoRA live/disk/frozen-reload、safetensors、PEFT config、ms-swift additional config、
callback receipt、exact optimizer steps 与 adversarial path/config 测试已通过。
尚未在真实 CUDA/ms-swift 环境跑新训练。

### Rubric RL

QA 与 Motion Rubric V2 的严格本地路径可用，旧 Stage-1/segmented 体系不能与
V2 混用。在线 judge、重型 Qwen 和 GPU integration 尚未实跑。

### 全模型评估

一个评估 Agent 统一负责 15 个模型即可，不需要每个模型单开任务。当前 backend
矩阵仍不完整：只有两个 reviewed finetune/reload backend，零 evaluation backend。
生产阶段既受 backend 缺失阻断，也受 multi-root bootstrap 阻断。

## 下一阶段顺序

1. 用户先把新端点和认证更新到安全连接文件；只读登录并审计 hostname、磁盘、
   目录、revision、`nvidia-smi`、GPU UUID 与进程。
2. 在全新的远端 staging 目录同步；禁止覆盖历史目录和使用 `--delete`。
3. 在 Linux 补跑 14 个 Windows skip，并建立非 editable、精确冻结的 CUDA 环境。
4. 实现 `docs/SFT_FORMAL_BOOTSTRAP_PLAN.md` 的 controller NPROC=1 多根 bootstrap，
   做真实 1-step forward/backward/update/save/reload；独立复审后再开放。
5. 单独扩展 GPU acquire-many、rank group、NCCL 与 N-rank receipt，不能用 N=1
   测试冒充。
6. 补齐其他 13 个 finetune、15 个 evaluation、13 个 reload backend；AGCN 和
   MotionCLIP 使用正式实现。
7. 冻结唯一 QA500-v2 与 evaluator 后创建新 batch；先全模型 fresh finetune，
   再按 1/8/32/500 评估并发布。

## 建议 subagent 分工

总控保留文件所有权、接口决策和最终验收；最多并行三类：

- Bootstrap/SFT Agent：只负责 multi-root runtime、formal SFT 与对应测试；
- Model Backend Agent：逐个补齐并审查 15 模型 finetune/eval/reload backend；
- Independent Reviewer：只读攻击复现、Linux/CUDA 实测和最终 P0/P1 审计。

同一文件同一时间只能有一个作者。作者定向测试通过后必须由不同 reviewer
复核，最后再跑：

```powershell
Set-Location D:\MotionLLM\motionllm_refactor
.\.venv\Scripts\python.exe scripts\run_checks.py
```

当前基线是 `922 passed, 14 skipped`；skip 必须在 Linux 补跑，不能当成通过。
