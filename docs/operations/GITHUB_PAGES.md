# GitHub 与 Pages 发布记录

发布日期：2026-09-02（Asia/Shanghai）

## 公开入口

- 仓库：<https://github.com/caim33/MLM>
- Pages：<https://caim33.github.io/MLM/>
- Qwen 使用手册：<https://caim33.github.io/MLM/guide/qwen-codebase.html>
- 数据统计与质量报告：
  <https://caim33.github.io/MLM/motionllm-page/guide/dataset-statistics.html>
- Paper Reading：<https://caim33.github.io/MLM/motionllm-page/>

## 发布范围

Git 仓库保存 Qwen/MotionLLM 代码、测试、文档和 `online_page/`。Pages 工作流
只把 `online_page/` 发布为静态网站；Python 源码不会成为网站路由。原
`motionllm-page/` 已作为主网站的子页面保存在 `online_page/motionllm-page/`。

数据集、模型权重、checkpoint、服务器 `runtime/`、训练输出和 `history/` 没有
进入仓库。首次推送前已完成 secret scan、Git 完整性检查与大文件检查；仓库没有
超过 GitHub 单文件限制的文件。

## 首次发布证据

- 首次公开提交：`98abf8a58d300690a8afd84d5d85e2b765e0fb0f`
- 首次 Pages 工作流：`Deploy MotionLLM page`；主网站上线后为 `Deploy Codebase portal`
- 首次运行：<https://github.com/caim33/MLM/actions/runs/33536805826>
- 结果：成功，线上主页及两个 guide 页面均已直接打开验证。

工作流曾显示依赖 action 的 Node.js 20 弃用警告，但 GitHub runner 已强制使用
Node.js 24，首次部署成功。该警告不是本次发布失败；后续 action 版本升级时再消除。

## 后续更新

服务器活动仓库的远端为：

```text
origin  https://github.com/caim33/MLM.git
```

对 `main` 分支中 `online_page/**` 或 `.github/workflows/pages.yml` 的推送会自动
触发 Pages 部署。新增公开网页资产前，必须继续检查敏感信息、单文件大小、论文版权
和内部数据披露边界。
