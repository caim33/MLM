# Qwen / MotionLLM Codebase

MotionLLM / Qwen 的活动代码、研究网页、数据审计工具和状态文档统一仓库。

服务器规范路径为 `/wangbenyou-sulongjie/caimeng/qwen-codebase`。

## 目录

- `code/`：可维护的 MotionLLM / Qwen codebase。新核心逻辑位于 `code/src/`。
- `site/`：MotionLLM Research Atlas、Qwen 使用手册与数据统计网页。
- `tools/data_audit/`：数据盘点、配对检查和兼容视图工具。
- `tools/site_qa/`：静态网页链接检查。
- `tools/monitor/`：AIStation 监控源码；依赖 Windows、Chrome 和已登录会话，不能在 Linux 服务器原样运行。
- `reports/`：当前数据统计报告。
- `docs/`：仓库级当前状态与交接边界。

## 从这里开始

1. 阅读 `docs/CURRENT_SERVER_STATUS_20260901.md`，了解当前服务器验证结果和未完成项。
2. 阅读 `code/docs/STATUS.md` 与 `code/docs/ARCHITECTURE.md`。
3. 开发和运行命令见 `code/docs/USAGE_GUIDE.md`。
4. 网页入口为 `site/index.html`，数据统计页为 `site/guide/dataset-statistics.html`。

## 不进入 Git 的内容

数据集、模型权重、checkpoint、运行环境、GPU lease、训练输出、实验预测和历史交接包不属于本仓库。服务器上的规范数据入口仍是：

```text
/wangbenyou-sulongjie/caimeng/dataset
```

服务器兼容入口会继续保留，现有脚本无需因为目录收口立即修改。

## 发布边界

仓库应先保持私有。网页包含论文 PDF、研究提取内容和内部数据统计；代码的 `legacy/` 与历史审计材料也可能含内部路径。公开发布或启用公开 GitHub Pages 前，必须另做版权、保密和凭据复核。
