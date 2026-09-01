# Caimeng Codebase Portal

这是仓库的公开主网站目录。首页提供四个入口：数据统计、数据可视化、Codebase
使用说明和 Paper Reading。

`dataset-page/` 是从 AIStation `dataset/data_page/` 同步的 2026-09-02 静态统计
快照；公开页只包含汇总 JSON，不读取服务器的样本、视频或 SQLite 索引。
`motionllm-page/` 保留 Paper Reading、Codebase 指南和旧版统计证据。GitHub Pages
工作流只发布 `online_page/`，不会公开仓库里的训练代码、配置或原始数据文件。

数据可视化仍由受保护的在线工作台提供；GitHub Pages 只作为公开入口，不保存
账号、密码或运行时数据。
