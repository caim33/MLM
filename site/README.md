# MotionLLM Research Atlas

打开 `index.html` 即可离线浏览。网页无需构建步骤或外部前端依赖。Qwen/MotionLLM 代码使用手册位于 `guide/qwen-codebase.html`，数据统计与质量报告位于 `guide/dataset-statistics.html`。

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
cd E:\codex_work\MotionLLM\caimeng
python -m http.server 8767 --bind 127.0.0.1
```

然后访问 `http://127.0.0.1:8767/paper_research_site/`、`http://127.0.0.1:8767/paper_research_site/guide/qwen-codebase.html` 和 `http://127.0.0.1:8767/paper_research_site/guide/dataset-statistics.html`。

服务器同步副本放在 `/wangbenyou-sulongjie/caimeng/codex_work/web/paper_research_site`。服务器预览：

```bash
cd /wangbenyou-sulongjie/caimeng/codex_work/web
python3 -m http.server 8767 --bind 127.0.0.1
```

服务器只监听本机回环地址；需要从个人电脑查看时使用 SSH 端口转发，不要把端口直接暴露到公网。

研究边界：历史 accuracy、prediction、baseline、proxy 与 option-score 诊断不作为新批次主结果。任何后续正式评估仍需遵守 fresh finetune 与全模型 finetune barrier。
