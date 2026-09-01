# Legacy 来源证据

本目录仅用于保存迁移来源、旧要求和可复核 hash。活动代码不得 import、执行或把这里加入 `PYTHONPATH`。

## 内容

- `qwen_vl_original/`：旧 `/wangbenyou-sulongjie/qwen-vl-finetune` 中可精确追溯到 Git HEAD 的 Qwen 数据代码。
- `refactor_snapshot/`：clean codebase 建立前的 README/AGENTS、22 份含个人路径的
  GRPO 历史配置、个人运维脚本和旧 smoke 资产。它们被移动到这里是为了让活动
  配置可移植，不代表可重新执行。

需要迁移旧行为时，先在 `docs/MIGRATION.md` 中登记来源，再在 `src/` 中重新实现并用 contract test 验证。不要直接修改 legacy 文件。
