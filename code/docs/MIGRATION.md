# 从旧 QwenVL 到 clean codebase

## 来源

旧仓库：

```text
/wangbenyou-sulongjie/qwen-vl-finetune
branch: main
HEAD: 7f2f6c1d5651e069f849128435081d98e367909c
commit time: 2026-07-06T20:07:57+00:00
```

旧 `qwenvl/data` 的五个文件在检查时与该 commit 的 blob 一致。原始副本保存在 `legacy/qwen_vl_original/qwenvl/data/`；活动代码不得导入它。

## 为什么不能直接复制

- 旧 registry 把数据路径写死到其他用户的服务器目录。
- 旧 processor 没有新的 `logical_samples`、motion ownership、branch identity 和坏样本原位失败合同。
- 旧 Qwen model、Full/LoRA SFT 和 shell launcher 与 refactor 版本均有大量差异。
- 旧仓库没有 `src/motionllm/data` 和 `src/motion_eval/data`。

因此迁移采用“行为提取 + contract test + 新适配层”，而不是目录覆盖。

## 迁移映射

| 旧位置 | 新位置 | 策略 |
|---|---|---|
| `qwenvl/data/__init__.py` | `configs/datasets/` + Qwen registry adapter | 删除硬编码路径，保留显式 alias 解析 |
| `qwenvl/data/data_processor.py` | `motionllm.data` + Qwen processor adapter | 核心身份/路径/collation 下沉，Qwen tensor 处理留在适配层 |
| `qwenvl/data/rope2d.py` | Qwen adapter | 保留算法，增加独立单元测试 |
| `qwenvl/data/Mean.npy`, `Std.npy` | 外部资产配置 | 不把训练资产作为隐式包资源 |
| 旧 `models/qwen3_vl_motion.py` | `src/motionllm` 权威实现 + wrapper | 只保留旧 import 兼容 |
| 旧 SFT shell/Python | `motionllm.training` + 顶层过渡 CLI | 路径与运行模式已显式化；CLI 仍待继续下沉 |

## 兼容承诺

短期保留常用旧命令和 import，但输出会提示其兼容层身份。任何旧 alias 若没有显式配置文件映射必须失败，不能回退到个人绝对路径。

## 尚未声称完成的内容

- 旧数据 processor 的全部 Qwen 图像/视频预处理行为尚未逐项迁移。
- GPU SFT、GRPO、模型 reload 和正式评估需在 clean codebase 上重新验证。
- 历史 checkpoint、prediction 和 accuracy 不自动继承为新版本结果。
