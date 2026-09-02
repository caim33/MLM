# Source provenance

本 clean codebase 不是旧仓库某一个 commit 的直接 checkout。它由两个可区分来源整理而来，任何迁移和验证都必须保留这一区分。

## 1. 旧 QwenVL Git 仓库

```text
path: /wangbenyou-sulongjie/qwen-vl-finetune
branch: main
HEAD: 7f2f6c1d5651e069f849128435081d98e367909c
HEAD time: 2026-07-06T20:07:57+00:00
```

审计结果：14 个可达提交，无 remote、tag、stash 或其他 branch；失联 amend 对象中也没有 `src/motionllm/data` 或 `src/motion_eval/data`。

服务器工作树并不等于 HEAD：存在 19 个 tracked 改动以及未跟踪实验配置/数据。clean codebase 不把这些状态混成一个 Git commit；只有明确登记来源的文件才迁移。

`qwenvl/data` 五个文件与 HEAD blob 一致，详见 `legacy/qwen_vl_original/SOURCE_MANIFEST.md`。

## 2. MotionLLM refactor 交接包

```text
archive: MotionLLM_refactor_complete_handoff_20260829.zip
sha256: d96d7d2b6ee48943c58c6adf14d736954de95eaa74b6bd66ccb036b9010fa460
```

交接包包含新的 contracts、motion、fusion、training、GRPO、controller、runtime、tests 和文档，但遗漏：

```text
src/motionllm/data/
src/motion_eval/data/
qwenvl/data/
```

旧 Git 只能恢复 `qwenvl/data` 的重构前基线；两个 `src/.../data` 模块从未存在于旧 Git，需要依据当前 contracts/tests 重新实现，并作为 clean codebase 的新源码，而不是声称“精确恢复”。

## 3. Clean codebase 原则

- 旧文件保持在 `legacy/`，不进入活动 import path。
- 新实现以公开 API、现有 tests 和
  `docs/architecture/ENGINEERING_REQUIREMENTS.md` 为依据。
- 找不到历史安全协议细节时，新建明确 schema/version，不伪装成未知旧 receipt 的兼容实现。
- 所有测试数字来自 clean codebase 当前复跑，并记录解释器、平台和 skipped 项。

## 4. 迁移证据归档

服务器保存：

```text
/wangbenyou-sulongjie/caimeng/handoff/qwen_vl_legacy_data_git_7f2f6c1_20260829.tar.gz
sha256: b99881e85d28d3228f36093d145db37ade233ff7fcb887f88a1dd843d8f13c37
```

该压缩包只含旧 `qwenvl/data`，不能独立代表完整旧仓库或 clean codebase。
