# MotionLLM Refactor Workspace

这是 MotionLLM 的重构交接工作区。服务器整理副本位于 `/wangbenyou-sulongjie/caimeng/qwen-codebase`。历史快照只作为只读证据；新代码、测试、批次状态和交接文档均维护在本目录。

> 当前服务器副本不是完整可运行 checkout：缺少 `src/motion_eval/data/`、`src/motionllm/data/` 和 `qwenvl/data/`。先阅读 `docs/USAGE_GUIDE.md` 的完整性检查，不要用旧目录或同名第三方包拼接内部模块。

开始工作前依次阅读：

1. `AGENTS.md`
2. `docs/REFACTOR_HANDOFF.md`
3. `docs/COMMON_COMMANDS.md`
4. `docs/USAGE_GUIDE.md`
5. `docs/ARCHITECTURE_TARGET.md`
6. `model_evaluation_agent/00_从这里开始.md`
7. `model_evaluation_agent/MEMORY.md`
8. `model_evaluation_agent/RUNBOOK.md`
9. `model_evaluation_agent/model_registry.json`
10. `model_evaluation_agent/PRETRAINED_ASSETS.md`
11. `model_evaluation_agent/pretrained_registry.json`
12. `model_evaluation_agent/server_audit/` 下最新审计

关键位置：

- 服务器整理副本：`/wangbenyou-sulongjie/caimeng/qwen-codebase`
- 使用手册：`docs/USAGE_GUIDE.md`
- 文档索引：`docs/README.md`
- 模型核心：`src/motionllm/`
- 15 模型统一控制器：`src/motion_eval/`
- 兼容入口：`models/`、`qwenvl/`、`model_evaluation_agent/scripts/`
- 测试：`tests/unit/`、`tests/contract/`、`tests/integration/`、`tests/stress/`

最重要的规则：

- 每个新评估批次覆盖的每个模型都必须先产生该批次的新 finetune artifact，或形成可复核的正式阻塞证据。
- 所有模型的 finetune 阶段达到终态后，才允许全局打开评估阶段。
- 正式评估顺序固定为 `1 → 8 → 32 → 500`。
- AGCN 与 MotionCLIP 都是正式 finetune 模型，不得使用历史 MLP/RNN proxy。
- 历史 accuracy、prediction、baseline、option-score 和 proxy 结果不得进入新主结果表。
- 不把密码、token、私钥或连接文件内容写入代码、文档、命令、日志或 manifest。

快速检查：

```bash
cd /wangbenyou-sulongjie/caimeng/qwen-codebase
for path in src/motion_eval/data src/motionllm/data qwenvl/data; do
  test -d "$path" || echo "MISSING: $path"
done
```

较新历史验证报告记录：`922 passed, 14 skipped, 2 warnings`。这是 2026-08-21 的报告结果，不是当前不完整服务器副本的复跑；14 个 skip 仍需在 Linux 补跑。完整证据、独立 review 结论和远端阻塞见 `docs/VERIFICATION_REPORT.md`。

完整使用说明见 `docs/USAGE_GUIDE.md`；交接与命令见 `docs/REFACTOR_HANDOFF.md` 和 `docs/COMMON_COMMANDS.md`。Formal Qwen SFT 仍以 exit `78` fail-closed；catalog/controller production 路径由 `verified-multi-root-bootstrap` 主动阻断，禁止通过删除 gate 推进。
