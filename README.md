# MotionLLM / Qwen Clean Codebase

这是从原 `/wangbenyou-sulongjie/qwen-vl-finetune` 与后续 MotionLLM refactor 中重新整理的活动代码库。目标是模块边界清楚、容易修改、CPU 可检查、GPU 运行显式，并且不再依赖个人硬编码路径。

服务器目标目录：

```text
/wangbenyou-sulongjie/caimeng/qwen-codebase
```

公开入口：

- GitHub 仓库：<https://github.com/caim33/MLM>
- 研究网页：<https://caim33.github.io/MLM/>
- Qwen 使用手册：<https://caim33.github.io/MLM/guide/qwen-codebase.html>
- 数据统计与质量报告：<https://caim33.github.io/MLM/motionllm-page/guide/dataset-statistics.html>

## 先从这里开始

1. `docs/README.md`：文档地图以及当前/历史边界。
2. `docs/status/README.md`：现在能做什么、还缺什么。
3. `docs/architecture/ARCHITECTURE.md`：模块边界与依赖方向。
4. `docs/guides/USAGE_GUIDE.md`：推理、SFT、GRPO、Rubric 和评估命令。
5. `docs/guides/DEVELOPMENT.md`：环境、修改顺序和分层测试。

## 一眼看懂目录

```text
src/motionllm/        MotionLLM 权威核心（含 data、motion、qwen、training、grpo）
src/motion_eval/      统一评估、批次、GPU 与发布控制器
qwenvl/               Qwen 训练/推理过渡入口；data 子包是薄兼容层
models/               Qwen3-VL-Motion 过渡模型实现与旧 import
rubric_rl/            Rubric 过渡实现与旧 CLI
model_evaluation_agent/ registry、worker facade 与运维文档
configs/              数据、训练、GRPO 和评估配置
tests/                unit / contract / integration / stress
legacy/               只读来源证据，活动代码不得导入
docs/                 状态、架构、使用、迁移、审计和数据报告
requirements/         开发、SFT 和 GRPO 的冻结依赖文件
tools/                数据工具；remote/ 收纳远端辅助与旧兼容脚本
online_page/          Codebase 主网站、使用手册、数据统计与 Paper Reading
```

核心原则：新逻辑写在 `src/`；现有顶层 Qwen/model/Rubric 大文件属于待继续
下沉的过渡实现，不能把它们误称为已经完成迁移；`legacy/` 永远不进入 import path。

网页保持在独立的 `online_page/` 目录中，但与代码一起版本化。`.github/workflows/pages.yml` 只发布该目录，不会把 Python 源码作为网页内容上传。Paper Reading 页面位于 `online_page/motionllm-page/`。AIStation monitor 不属于当前代码库，历史副本保存在服务器 `history/archive/monitor-20260901/`。

## 快速建立 CPU 开发环境

```bash
cd /wangbenyou-sulongjie/caimeng/qwen-codebase
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

基础检查：

```bash
python -c "import motionllm, motion_eval"
python -m motion_eval --help
python -m pytest tests/unit tests/contract -q
```

只有需要 Qwen、Torch、CUDA 或训练时才安装：

```bash
python -m pip install -e '.[sft]'
```

交付服务器因缺 `python3-venv/ensurepip`，已使用不覆盖系统 CUDA/Torch 的独立
依赖目录；登录后的 `PYTHONPATH` 设置和验证命令见
`docs/guides/USAGE_GUIDE.md` 第 1 节。

## 数据与模型路径

活动代码不接受写死的个人服务器路径。数据集 alias 必须通过 `configs/datasets/` 中的显式配置解析；模型、checkpoint、VQ-VAE、媒体 root 和输出目录由 CLI 或配置传入。

旧 `qwenvl/data` 的 3 个源码文件与 2 个 normalization 资产保存在
`legacy/qwen_vl_original/`，仅用于迁移对照。它包含旧绝对路径和宽松数据行为，
不能直接作为活动实现运行。旧 GRPO 配置和个人运维脚本也已隔离到
`legacy/refactor_snapshot/`。

## 运行资格

- `inspect`：只读配置与 schema 检查。
- `dev`：小样本调试，产物不可发布。
- `preflight`：依赖、GPU、数据和 reload 检查，产物不可发布。
- `production`：必须显式启用并通过完整 verifier。

历史 checkpoint、prediction、accuracy、proxy 或 smoke 产物不会自动获得新 codebase 的 production 资格。

## 当前验证

请以 `docs/status/README.md` 的当前复跑结果为准。当前已完成一次真实 A100、历史
checkpoint、MotionX 视频与 motion 的端到端 smoke；它证明运行链路可用，不代表
模型精度或 production release。旧 refactor 报告中的
`922 passed, 14 skipped, 2 warnings` 只作为历史证据，不能代表本目录当前状态。

## 安全

- 不把密码、token、私钥、连接文件或个人绝对路径写入活动代码与活动配置。
  `legacy/` 和 dated `server_audit/` 只可作为不可执行来源证据保留旧路径。
- 不手工删除 GPU lease，不按模糊进程名批量终止任务。
- `legacy/`、历史数据和模型资产跨组织分发前必须单独做版权与保密审查。
