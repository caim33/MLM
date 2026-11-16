# Caimeng Codebase Portal

这是仓库的公开主网站目录。首页提供数据统计、数据可视化、Codebase
使用说明、Paper Reading、Marin 训练图解和 DeepSeek 技术学习六个入口。

`deepseek/` 将本地 36 页 DeepSeek 学习报告整理为完整网页，包含 26 幅图的详细读图
说明、33 项参考资料、三项交互教学示例及原 PDF 下载。入口为
`https://caim33.github.io/MLM/deepseek/`。报告快照为 2026-10-26；2026-11-17 整理
网页并复核关键一手来源。网页版新增版本澄清、V3/V4.1 recipe 对照和指标释义，
原 PDF 保持原稿。FSDP＋EP 示例是教学拓扑，与 V3 的实际并行配置分开标注。

`marin-535b/` 是 Marin 535B-A23B 中文学习报告，包含全部 29 幅图的详细读图说明、
指标含义、3 段本地动画和离线交互数据。入口为
`https://caim33.github.io/MLM/marin-535b/`。资源使用相对路径，图下保留原始来源链接。
报告的研究快照为 2026-11-16；2026-11-17 接入本站，不代表训练事实已重新刷新。

`dataset-page/` 是从 AIStation `dataset/data_page/` 同步的 2026-09-02 静态统计
快照；公开页只包含汇总 JSON，不读取服务器的样本、视频或 SQLite 索引。
`guide/` contains the repository-wide Qwen Codebase guide. `motionllm-page/` retains Paper Reading and the older statistics guide.
GitHub Pages stages only approved static entries. It excludes `online_page/data_page/` service code, dependencies, indexes, and media.

数据可视化仍由受保护的在线工作台提供；GitHub Pages 只作为公开入口，不保存
账号、密码或运行时数据。


## Dynamic Data Atlas source (2026-09-27)

The complete data browser source now lives in online_page/data_page/. Its Python API reads the canonical dataset and a private SQLite index under dataset/data_page/data/. GitHub Pages uses the explicit static allowlist in tools/site/build_public_site.py, so this service code and its runtime data are not uploaded to Pages.
