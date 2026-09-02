# 依赖文件

- `dev.lock`：CPU 开发和测试的精确版本。
- `sft.txt`：Qwen 推理、Full/LoRA SFT 与已审查 backend 的 CUDA/Linux 依赖范围。
- `grpo.lock`：Formal GRPO 的关键 API 精确版本；Torch/CUDA wheel 来源仍需按目标机器选择。

基础包优先通过 `python -m pip install -e .` 或 `.[dev]` 安装。不要把某台服务器的
完整 `pip freeze` 直接覆盖这些声明；正式运行环境应另行冻结并写入 provenance。
