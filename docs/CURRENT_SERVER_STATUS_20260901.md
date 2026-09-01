# 当前服务器状态

核对日期：2026-09-01（Asia/Shanghai）

## 已确认完整

- 活动 Git 仓库位于 `/wangbenyou-sulongjie/caimeng/qwen-codebase`，Python 工程文件直接位于仓库根目录，网页位于 `motionllm-page/` 子目录。
- 规范数据位于 `/wangbenyou-sulongjie/caimeng/dataset`。
- Qwen 数据组织软链接共 10,524 个，断链 0。
- 最新网页包含 3 个 HTML 页面，检查 103 个本地目标，失败 0。
- 数据审计归档可以正常列出和解压。
- 原始 `/wangbenyou-sulongjie/qwen-vl-finetune` 未被覆盖。
- 当前代码模块、使用文档、架构文档和 GPU smoke 证据均已保留。
- monitor 已移出活动仓库并保存在 `history/archive/monitor-20260901/`。
- `/wangbenyou-sulongjie/caimeng/runtime` 已从 `history/` 中的完整 Python 3.10
  wheelhouse 重建，系统 Torch/CUDA 未被覆盖。

## 当前服务器测试

当前实例为 Ubuntu 22.04、Python 3.10、4 × Tesla V100S 32 GB。

一次服务器全量复跑结果：

```text
929 passed, 20 failed, 7 skipped, 5 warnings
```

失败分类：

- 6 个 GRPO preflight 失败来自启动测试时保留的环境变量 `PYTHONPATH`。在解释器启动后清除该环境变量，相关文件复跑为 `59 passed`。
- controller 单元/集成测试仍确认有 12 个失败，主要涉及新的 production manifest、冻结 worker bundle 与控制器测试 fixture 不一致。
- 原子文件哈希压力测试有 1 个失败，当前 Linux 临时文件系统没有检测到同大小的运行中改写。
- 并发状态转换测试在全量运行中失败过一次，在单独复跑中通过，暂记为环境相关或时序不稳定。

历史文档中的 `954 passed, 2 skipped` 是 Windows/Python 3.12 环境结果，不能替代上述服务器结果。

## GPU 状态

当前实例 4 张 V100 均空闲，没有运行中的项目 keepalive。旧 A100 状态已经随旧
runtime 清理，重建后的 `runtime/gpu-keepalive-managed/` 没有活动记录。当前 V100
上的受管理启动测试也没有遗留进程：容器内 Python PID 与 `nvidia-smi` 返回的宿主机
PID 不同，严格 PID + GPU UUID 所有权校验无法证明两者属于同一进程，因此 worker
安全退出并清理状态。Torch CUDA 可见性和指定 GPU 的显存分配已单独验证正常。

## 重建后的运行依赖

- 入口：`source /wangbenyou-sulongjie/caimeng/runtime/activate_qwen.sh`
- Python 3.10.12；系统 Torch `2.4.0a0+f70bd71a48.nv24.06`；Torch CUDA 12.5。
- 独立依赖：Transformers 4.57.3、PEFT 0.18.0、Accelerate 1.14.0、PyAV
  12.3.0、protobuf 3.20.3。
- 当前是基础运行环境，不是完整 SFT/GRPO 环境；训练栈和 checkpoint overlay 尚未
  恢复。完整清单见 `/wangbenyou-sulongjie/caimeng/runtime/README.md` 和
  `manifest.json`。

## 仍需处理

- 修复或更新 Linux controller 测试 fixture，并重新跑全量门禁。
- 修复原子文件哈希的跨文件系统检测策略。
- 把服务器测试启动方式改为不向 formal GRPO 测试泄漏 `PYTHONPATH`。
- 新的 CUDA smoke 应在当前 V100 环境单独记录，旧 A100 smoke 只保留为历史证据。
- 为容器场景设计并验证可证明的宿主机 PID 映射后，再启动受管理的 GPU keepalive；
  不应通过关闭所有权校验来绕过。
- 代码已于 2026-09-02 推送到公开仓库 <https://github.com/caim33/MLM>，网页由
  GitHub Actions 发布到 <https://caim33.github.io/MLM/>。后续新增论文 PDF、
  提取文本、`legacy/` 或审计材料时仍需逐次复核版权与保密边界。
