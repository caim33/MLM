# MotionLLM 当前架构

## 1. 依赖方向

```text
contracts <- pure domain logic <- framework adapters <- compatibility facade / CLI
```

- `contracts` 不导入 torch、transformers 或 ms-swift。
- `motion_eval.controller` 不加载大模型框架，只启动隔离 worker。
- GRPO 纯 reward 不依赖 ms-swift；Swift 只通过 adapter 调用。
- 新核心不反向依赖 legacy 入口。
- worker 不能改变 canonical benchmark 或 controller 状态。
- formal worker 必须是非交互式；`runtime.run_verified_python` 保留 stdin 作为已验证源码通道。

## 2. 目录

```text
src/
  motionllm/
    contracts/       sample、modality、option 契约
    data/            strict readers、paths、messages、datasets、collators
    motion/          IO、normalization、temporal、validation
    fusion/          placeholder、boundary、projector、embedding injection
    models/          config、state dict、generation、injection
    training/        factory、tokens、freeze、LoRA、SFT、artifact
    grpo/            schema、reward、VM/V pairing、Swift adapter、redaction

  motion_eval/
    contracts/       prediction 与错误码
    core/            hashing、atomic IO、safe paths
    data/            JSON/JSONL、benchmark、receipts、leakage
    adapters/        15 模型 typed CommandSpec
    controller/      registry、events、batch、attempt、barrier
    runtime/         process、verified source、GPU lease、keepalive、remote
    evaluation/      strict parser、rows、error accounting
    reporting/       release build/verify
```

## 3. 核心契约

### Sample

- 稳定的 `sample_id` 与 `group_id`；
- 显式模态 `V/M/VM/T`；
- canonical question/options/gold；
- 可验证的媒体引用；
- 不允许静默 fallback 或换样本。

### Artifact

- 当前 batch/model/attempt identity；
- 真实文件或目录 hash；
- base、data、code、config、environment provenance；
- fresh training steps；
- controller 启动的独立 save/reload 验证。
- video-only LoRA 的空 `modules_to_save` 是合法状态，完整 LoRA A/B 权重仍须逐项重载比对；motion LoRA 的 motion 模块和边界 token 必须同时通过验证。

### Prediction

- 每个 canonical sample 一行；
- 生成式模型保留 raw output，判别式模型保留固定 A/B/C/D score order；
- 只接受严格 `<answer>[A-D]</answer>`；
- invalid/runtime error 保留在固定分母；
- prediction 与当前 batch artifact、benchmark、media manifest 绑定。

### State

- append-only event；
- HMAC event anchor + batch 外单调 head；
- attempt nonce 与不可重用 reference；
- finetune 全局屏障；
- smoke `1 → 8 → 32` 屏障；
- release 可重算、可复核。

## 4. GPU 并发关系

```text
                 one GPU UUID role mutex
                /          |          \
        keepalive       finetune       eval
```

同一 UUID 只能有一个角色；不同 UUID 可以并行。worker、controller verifier 和 prepare/cleanup 全程持有同一 lease，避免检查与启动之间的窗口。Windows 与 Linux 都从操作系统重新读取进程 argv；失败启动只有在子进程已确认退出且 capability 完整匹配时才能回滚。

## 5. 兼容策略

`models/`、`qwenvl/`、`model_evaluation_agent/scripts/` 和 `codex_remote_tools/gpu_keepalive.py` 只保留薄 facade。兼容入口可以保留旧命令参数，但数据验证、状态转换、训练 artifact 和 GPU 所有权必须委托给 `src/` 中的唯一实现。
