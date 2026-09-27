# Caimeng Codebase Portal

这是仓库的公开主网站目录。首页提供四个入口：数据统计、数据可视化、Codebase
使用说明和 Paper Reading。

`dataset-page/` 是从 AIStation `dataset/data_page/` 同步的 2026-09-02 静态统计
快照；公开页只包含汇总 JSON，不读取服务器的样本、视频或 SQLite 索引。
`guide/` contains the repository-wide Qwen Codebase guide. `motionllm-page/` retains Paper Reading and the older statistics guide.
GitHub Pages stages only approved static entries. It excludes `online_page/data_page/` service code, dependencies, indexes, and media.

数据可视化仍由受保护的在线工作台提供；GitHub Pages 只作为公开入口，不保存
账号、密码或运行时数据。


## Dynamic Data Atlas source (2026-09-27)

The complete data browser source now lives in online_page/data_page/. Its Python API reads the canonical dataset and a private SQLite index under dataset/data_page/data/. GitHub Pages uses the explicit static allowlist in tools/site/build_public_site.py, so this service code and its runtime data are not uploaded to Pages.
