# 远端辅助工具

这里统一收纳不属于核心 Python 包的远端辅助脚本：

- `motion_proxy_train_eval.py`：旧 Motion proxy 训练/评估辅助入口。
- `prepare_qwen_benchmark_jsonl.py`：Qwen benchmark 数据转换工具。
- `legacy/`：从原 `codex_remote_tools/` 迁来的历史兼容脚本和依赖 stub。

新运行控制、GPU keepalive、SSH 边界和进程校验应使用
`src/motion_eval/runtime/`；不要继续向 `legacy/` 添加新逻辑。
