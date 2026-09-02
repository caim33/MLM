# 当前状态

更新日期：2026-09-02

## 已完成

- 从原 `/wangbenyou-sulongjie/qwen-vl-finetune` 的 live working tree 与可追溯
  Git 历史中核验来源；旧仓库保持不动。
- 恢复并重写 `motionllm.data`：严格 JSONL、路径 confinement、canonical
  sample、消息适配、identity-preserving dataset 与 logical/physical collation。
- 恢复 `motion_eval.data`：严格 JSON、固定 500 条 benchmark、冻结 receipt、
  leakage 与 pretrained index 校验。
- 建立 `motionllm.qwen` 正式适配层；顶层 `qwenvl.data` 只保留兼容 facade。
- 清除活动 Python 中的个人硬编码路径和 package-local `Mean.npy/Std.npy`
  回退；normalization 和 motion placeholder token 均显式绑定。
- Motion + Video 与 video-only 推理的 `--help` 可在无 CUDA 情况使用。
- 已用历史 merged checkpoint、真实 MotionX 视频与 motion、显式 Mean/Std 在
  A100 上完成 1 条端到端 Motion + Video 推理，进程返回 0；详细证据见
  `GPU_SMOKE_20260830.md`。
- README、架构、工程要求、开发、迁移、来源和使用手册已按新结构重写。
- wheel 已包含 `motionllm`、`motion_eval`、`qwenvl`、`models` 与 `rubric_rl`，
  并完成仓库外 import 检查。
- 已收口薄顶层目录：数据报告归入 `docs/reports/`，冻结依赖归入 `requirements/`，
  `codex_remote_tools/` 与 `remote_scripts/` 合并为 `tools/remote/`；运行缓存不进入 Git。
- 文档已按 `guides/architecture/status/operations/provenance/reports/archive` 分层；
  旧命令、旧交接和旧 GPU 状态不再与当前操作文档并列。
- GitHub Pages 已改为发布 `online_page/` Codebase 主网站；数据统计、动作可视化、
  使用说明和 Paper Reading 由统一首页进入，Python 源码不在发布范围。
- 数据统计入口已同步 AIStation `dataset/data_page/` 的 2026-09-02 汇总快照；公开页
  仅发布汇总 JSON，内部样本、视频、三维网格和 SQLite 索引仍留在服务器。
- 完整 Motion Viewer 已通过固定域名 `viewer.caimeng.online` 对外提供；本机自动占卡
  程序会在新的 AIStation 环境可连接后恢复 Viewer 与 Cloudflare Tunnel。

## 本机验证结果

环境：Windows、Python 3.12、Torch 2.13.0、Transformers 4.57.3、
PEFT 0.18.0，无 CUDA。

| 层级 | 结果 |
|---|---:|
| unit + contract | 729 passed，1 warning |
| integration | 147 passed，2 skipped，1 warning |
| stress | 78 passed |
| 合计 | **954 passed，2 skipped，2 warnings** |

另外：compileall、secret scan、wheel build/out-of-tree import、
`motion_eval --help`、两个 Qwen 推理帮助入口与 Full/LoRA 帮助入口均通过。
两个 skipped 是 Windows 主机没有 Bash 的 shell-facade 集成用例。

服务器已部署到 `/wangbenyou-sulongjie/caimeng/qwen-codebase`。服务器基础
Python 3.10 与 CUDA Torch 2.4 保持不动；由于系统缺 `python3-venv/ensurepip`，
Qwen 依赖改为安装在独立的
`/wangbenyou-sulongjie/caimeng/runtime/qwen-codebase-clean-py310`。该目录在被用户
清理后已于 2026-09-01 从 `history/` 中的完整离线 wheelhouse 重新建立，现有
Transformers 4.57.3、PEFT 0.18.0、Accelerate 1.14.0、PyAV 12.3.0，并使用
隔离的 protobuf 3.20.3 兼容服务器旧 ONNX。推理、Full SFT、LoRA SFT 的
`--help` 历史验证均返回 0；本次重建后 `motion_eval --help` 与 GPU keepalive
生命周期测试 **27 passed**。当前没有受管理的 keepalive：容器内 PID 与
`nvidia-smi` 报告的宿主机 PID 不同，严格所有权校验无法证明进程归属，因此启动
测试安全退出并清理了状态。系统 CUDA 和显存分配本身可用。

## 尚未验证

- 真实 GPU smoke 目前只有 1 条功能样本：目标为 D，预测为 A；它只证明代码链路
  可运行，不证明精度，也不等于正式 500 条评测。
- 尚未执行新的 CUDA Full/LoRA SFT 或 GRPO 训练。
- 当前重建的是基础运行依赖，不是完整 SFT/GRPO 环境；`qwen-vl-utils`、`decord`、
  `pytorchvideo`、`deepspeed`、`datasets`、`trl` 和 `ms-swift` 尚未安装。
- 被删除的历史 checkpoint overlay 尚未重建；真实 checkpoint smoke 需要先恢复并
  核验 overlay 来源。
- formal Qwen SFT 仍因缺少 external-HMAC 绑定的 pre-spawn snapshot 与
  verified in-memory worker bundle 而 fail-closed（exit 78）。
- 统一控制器的 production backend 仍受各自 verifier/gate 约束；本次 CPU
  回归不等于 15 模型 production release。
- Qwen2/Qwen2.5 visual RoPE 与 legacy image-only 数据暂时 fail-closed；当前
  维护并验证的是 Qwen3-VL Motion + Video 数据路径。
- Full/LoRA SFT、`models/qwen3_vl_motion.py` 与部分 Rubric 代码仍是顶层过渡
  实现，尚未全部下沉到 `src/`；模块归属已经标明，但结构迁移未宣称完成。

## 历史数字说明

旧 refactor 报告中的 `922 passed, 14 skipped, 2 warnings` 仅作为历史证据。
本页的 `954 passed, 2 skipped, 2 warnings` 才是 clean codebase 在上述本机
环境中的当前分层复跑结果；GPU 实跑证据另见 `GPU_SMOKE_20260830.md`。
