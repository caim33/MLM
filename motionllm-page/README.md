# MotionLLM Research Atlas

这是 `qwen-codebase` 仓库内的独立静态网页目录。打开 `index.html` 即可离线浏览，无需构建步骤或外部前端依赖。Qwen/MotionLLM 代码使用手册位于 `guide/qwen-codebase.html`，数据统计与质量报告位于 `guide/dataset-statistics.html`。

公开地址：

- 主页：<https://caim33.github.io/MLM/>
- Qwen 使用手册：<https://caim33.github.io/MLM/guide/qwen-codebase.html>
- 数据统计与质量报告：<https://caim33.github.io/MLM/guide/dataset-statistics.html>

目录说明：

- `index.html`：研究网页入口。
- `papers.js`：论文精读数据与来源链接。
- `assets/figures/`：从论文 PDF 或官方项目页提取的主图。
- `source_papers/`：精读时使用的论文原文与文本提取。
- `agent_reports/`：代码审计、motion 论文、RL/评估论文的并行研究报告。
- `guide/qwen-codebase.html`：Qwen codebase 的目录、命令、状态、现行代码证据和排错手册。
- `guide/dataset-statistics.html`：MotionX、HumanML3D、SONIC 和 Qwen QA 的规模、质量、配对与 GPU 烟测报告。

本地预览：

```bash
cd /path/to/qwen-codebase/motionllm-page
python -m http.server 8767 --bind 127.0.0.1
```

然后访问 `http://127.0.0.1:8767/`、`http://127.0.0.1:8767/guide/qwen-codebase.html` 和 `http://127.0.0.1:8767/guide/dataset-statistics.html`。

服务器规范目录为 `/wangbenyou-sulongjie/caimeng/qwen-codebase/motionllm-page`。
原顶层 `caimeng/motionllm-page` 兼容链接已于 2026-09-02 移除；
`codex_work/web/paper_research_site` 仍作为工作区快捷入口。服务器预览：

```bash
cd /wangbenyou-sulongjie/caimeng/qwen-codebase/motionllm-page
python3 -m http.server 8767 --bind 127.0.0.1
```

服务器只监听本机回环地址；需要从个人电脑查看时使用 SSH 端口转发，不要把端口直接暴露到公网。

## GitHub Pages

公开仓库为 <https://github.com/caim33/MLM>。仓库内的 Pages 工作流只发布本目录，不需要 Node.js 构建：

1. 向 `main` 推送 `motionllm-page/**` 或工作流修改；
2. `Deploy MotionLLM page` 自动打包本目录；
3. 部署成功后访问 <https://caim33.github.io/MLM/>。

`.nojekyll` 用于让 GitHub Pages 原样发布静态文件。当前仓库与网页已经公开；后续添加页面、论文 PDF、提取文本或内部数据统计前，仍需逐次复核版权与保密边界。

研究边界：历史 accuracy、prediction、baseline、proxy 与 option-score 诊断不作为新批次主结果。任何后续正式评估仍需遵守 fresh finetune 与全模型 finetune barrier。
