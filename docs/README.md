# 文档入口

文档按用途分组。只有 `status/README.md`、`architecture/` 和 `guides/` 描述当前
代码；带日期的状态、报告和 `archive/` 都是运行证据，不能代替当前复跑结果。

## 建议阅读顺序

1. [`status/README.md`](status/README.md)：当前能力、限制和测试基线。
2. [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md)：模块边界和依赖方向。
3. [`guides/USAGE_GUIDE.md`](guides/USAGE_GUIDE.md)：推理、SFT、GRPO、Rubric 和评估。
4. [`guides/DEVELOPMENT.md`](guides/DEVELOPMENT.md)：环境、修改顺序和测试。
5. [`architecture/MIGRATION.md`](architecture/MIGRATION.md)：旧 QwenVL 的迁移边界。

## 目录

| 目录 | 内容 | 当前性 |
|---|---|---|
| `guides/` | 使用手册和开发说明 | 当前 |
| `architecture/` | 架构、工程要求、迁移和 formal SFT 设计 | 当前 |
| `status/` | 当前总状态及带日期的服务器/GPU 证据 | 总状态当前；日期文件为快照 |
| `operations/` | GitHub Pages 等发布运维记录 | 当前流程 + 历史证据 |
| `provenance/` | 来源、交接包、基线清单和 hash | 审计证据 |
| `reports/` | 数据统计与质量报告 | 带日期快照 |
| `archive/` | 已被当前文档取代但仍需追溯的报告 | 历史，不作为操作说明 |

## 已收口的旧文档

- `ARCHITECTURE_TARGET.md` 的有效合同和 GPU 并发边界已合并进正式架构。
- `COMMON_COMMANDS.md` 含旧个人路径和过时 blocker 说明，当前命令以使用手册和
  各 CLI 的 `--help` 为准。
- `REFACTOR_HANDOFF.md` 已由当前状态、架构和开发说明取代。
- `REMOTE_GPU_STATUS.md` 已由 `status/SERVER_20260901.md` 取代。
- 2026-08-21 的旧验证报告保存在 `archive/`，其中通过数不能当作当前基线。

服务器活动目录为 `/wangbenyou-sulongjie/caimeng/qwen-codebase`。
