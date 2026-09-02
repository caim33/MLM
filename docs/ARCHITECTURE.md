# Clean Codebase 架构

## 总体结构

```text
qwen-codebase/
├── src/
│   ├── motionllm/             # MotionLLM 权威核心
│   │   ├── contracts/         # 样本、模态、答案等纯合同
│   │   ├── data/              # 严格读取、路径、消息、dataset、collation
│   │   ├── qwen/              # Qwen processor、dataset、collator、RoPE
│   │   ├── motion/            # motion 数组与时序处理
│   │   ├── fusion/            # motion token 与 projector
│   │   ├── models/            # 模型无关的模型服务
│   │   ├── training/          # SFT/LoRA 与 artifact
│   │   └── grpo/              # reward、rubric 与 group logic
│   └── motion_eval/           # 统一评估系统
│       ├── data/              # 严格 JSON、benchmark、receipt、leakage
│       ├── contracts/         # registry/backend 公共合同
│       ├── controller/        # batch、attempt、状态和 gate
│       ├── runtime/           # 进程、GPU、lease、远端边界
│       ├── adapters/          # 模型 backend 适配
│       ├── evaluation/        # strict parser 与 prediction
│       └── reporting/         # release manifest 与结果表
├── qwenvl/                    # Qwen 训练/推理过渡入口；data 为薄 facade
├── models/                    # Qwen3-VL-Motion 过渡模型实现
├── rubric_rl/                 # Rubric 过渡实现与旧 CLI
├── configs/                   # 活动模板；不含凭据和个人绝对路径
├── scripts/                   # 很薄的命令启动器
├── tests/                     # unit / contract / integration / stress
├── docs/                      # 使用、架构、状态、迁移和开发文档
├── tools/                     # 数据处理、审计和远端辅助工具
│   └── remote/                # 当前远端工具；legacy/ 为旧兼容脚本
└── legacy/                    # 只读来源证据，不在 import path
```

## 依赖方向

```text
contracts
   ↓
data ──→ motion
   ↓        ↓
fusion / models
   ↓
training / grpo

motion_eval.data → controller → adapters/evaluation → reporting

qwenvl / top-level models / rubric_rl ──→ src 中的权威模块
legacy ──X──→ 任何活动模块
```

核心层不能反向导入兼容层。`motionllm.qwen` 允许依赖
`motionllm.data`、`motionllm.motion` 和 `motionllm.fusion`，但这些下层模块
不知道 Qwen 的具体模型类。顶层 `qwenvl.data` 只转发到
`motionllm.qwen`，不承载新业务逻辑。

## 数据流

```text
外部 JSONL / legacy conversation
        ↓ strict reader / adapter
canonical Sample + resolved media
        ↓ message builder / motion loader
model-ready item
        ↓ collation plan
Qwen collator / model forward
        ↓ strict answer parser
prediction + typed error
        ↓ controller fixed denominator
report / release verifier
```

## Legacy 策略

`legacy/qwen_vl_original/` 保存旧仓库中可追溯的 Qwen 数据代码。它包含硬编码服务器路径，也缺少新 refactor 的 sample identity、logical ownership 和 fail-closed contract，因此只用于迁移对照。活动代码不得通过 `sys.path`、相对 import 或复制粘贴方式直接执行它。

`qwenvl.data` 已经是转发到 `motionllm.qwen` 的薄 facade。Full/LoRA SFT、
Qwen3-VL-Motion 模型和 Rubric 部分仍是顶层过渡实现；新改动应逐步下沉到
`src/`，但迁移前必须保持测试和旧命令可用，不能用文档把未完成工作包装成
“薄兼容层”。
