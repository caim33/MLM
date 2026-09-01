# 文档入口

服务器活动目录：

```text
/wangbenyou-sulongjie/caimeng/qwen-codebase
```

## 新接手者阅读顺序

1. `STATUS.md`：当前真实状态和复跑结果。
2. `ARCHITECTURE.md`：目录边界与依赖方向。
3. `GPU_SMOKE_20260830.md`：真实 checkpoint、媒体与 A100 推理证据。
4. `USAGE_GUIDE.md`：推理、训练、GRPO、Rubric 和评估入口。
5. `DEVELOPMENT.md`：环境、修改与测试流程。
6. `MIGRATION.md`：旧 QwenVL 行为的迁移方法。
7. `ENGINEERING_REQUIREMENTS.md`：clean codebase 的验收合同。
8. `SOURCE_PROVENANCE.md`：旧 Git、refactor 包和新实现的来源边界。

## 活动文档

| 文档 | 用途 |
|---|---|
| `STATUS.md` | 当前能力、限制和测试基线 |
| `GPU_SMOKE_20260830.md` | 真实 GPU 单样本推理、overlay 与运行证据 |
| `ARCHITECTURE.md` | 权威模块、兼容层和 legacy 的边界 |
| `USAGE_GUIDE.md` | 命令、前置条件、副作用和排错 |
| `DEVELOPMENT.md` | 开发环境、修改顺序和分层验证 |
| `MIGRATION.md` | 从旧 QwenVL 到新适配层的映射 |
| `ENGINEERING_REQUIREMENTS.md` | 数据、模块、运行模式、测试和文档要求 |
| `SOURCE_PROVENANCE.md` | 文件来源、commit、archive 和 hash |

## 历史 refactor 文档

下列文档保留为设计与审计证据，不自动成为新版本要求：

| 文档 | 历史用途 |
|---|---|
| `REFACTOR_HANDOFF.md` | 上一轮重构交接 |
| `VERIFICATION_REPORT.md` | 旧 checkout 的历史验证结果 |
| `ARCHITECTURE_TARGET.md` | 上一轮目标架构 |
| `COMMON_COMMANDS.md` | 旧控制器命令索引 |
| `SFT_FORMAL_BOOTSTRAP_PLAN.md` | 旧 formal bootstrap 计划 |
| `SFT_FORMAL_PROVENANCE.md` | 旧 formal provenance 设计 |
| `REMOTE_GPU_STATUS.md` | 当时的远端 GPU 状态 |

历史报告中的 `922 passed, 14 skipped, 2 warnings` 不能作为 clean codebase 当前通过数。当前结果只写入 `STATUS.md`。
