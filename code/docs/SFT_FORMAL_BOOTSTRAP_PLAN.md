# Formal Qwen SFT bootstrap 实施计划

更新时间：2026-08-21（Asia/Shanghai）

## 结论

当前 Qwen full-SFT/LoRA-SFT 的 formal 发布路径保持 fail-closed。不能通过删除
`exit 78` 恢复，因为 worker 内生成的快照发生在 Python/site/Torch/Qwen 导入
之后，不能证明实际执行过的源码与环境。

当前所有 catalog production finetune/evaluation/verifier 与 `complete_*` 也
使用同一个 blocker。现有 `python -I -c` 仍可能在 bootstrap 前执行 system
`sitecustomize`，而单根 runner bundle 没有冻结 `motion_eval` 与依赖环境。
因此本计划既是 Qwen formal SFT 的解除条件，也是重新开放全模型 production
controller 的前置条件。

下一阶段先实现 **controller-only、single-node、NPROC=1** 的安全切片；直接
shell 继续拒绝 formal，NPROC 大于 1 也继续拒绝。N-rank 需要额外的 GPU
原子多租约、并发 rank 管理、NCCL 实测和事件/attempt schema 升级，不能与
NPROC=1 冒充为同一个完成状态。

## 目标调用链

```text
BatchController
  -> freeze formal SFT source/environment/data/base snapshot
  -> freeze one deterministic multi-root in-memory source bundle
  -> append FORMAL_BOOTSTRAP_FROZEN to external HMAC event state
  -> spawn absolute interpreter with -I -S -B and an allowlisted environment
  -> stdlib-only bootstrap verifies snapshot and bundle before project imports
  -> install protected-prefix in-memory importer
  -> execute Qwen SFT worker from frozen bytes
  -> controller performs post-training source/environment/data/base audit
  -> strict artifact + training receipt + independent reload verification
  -> only then publish ATTEMPT_EXECUTED / FINETUNE_COMPLETE
```

## 必须实现的契约

### 1. 多源码根 bundle

同一次稳定 capture 至少覆盖：

- `src/motion_eval` -> `motion_eval`
- `src/motionllm` -> `motionllm`
- `qwenvl` -> `qwenvl`
- `models` -> `models`
- `model_evaluation_agent/scripts` -> runner/backend 入口

每个根记录 lexical root、package prefix、每个文件相对路径/大小/SHA-256 和
root manifest SHA-256。拒绝 symlink/reparse、case-fold/module collision、
`__pycache__`、`.pyc`、`.pyo`。protected prefix 缺少模块时必须直接报错，
不能退回磁盘 `PathFinder`。

### 2. pre-spawn external snapshot

权威快照由 controller 在导入 worker/project 模块之前创建，不能由
`full_sft.py` 或 `lora_sft.py` 自己创建。快照至少绑定：

- batch/model/attempt/training-mode identity；
- batch receipt、attempt 与 external HMAC trust identity；
- interpreter、stdlib、site-package、native runtime 和严格环境策略；
- source bundle set；
- train/validation/benchmark/leakage/config；
- base model、processor、VQ/motion 资产；
- distributed/rank contract；
- snapshot 自哈希与文件哈希。

`FORMAL_BOOTSTRAP_FROZEN` 必须在 spawn 前进入 append-only HMAC 事件链；
`ATTEMPT_STARTED` 只能引用同一个 snapshot。自洽替换批次目录中的 JSON，
但没有对应 external HMAC head，必须被拒绝。

### 3. 启动环境

使用绝对解释器和精确 argv：`python -I -S -B -c <stdlib-only bootstrap>`。
formal 子进程环境从空映射按正向白名单构造，不能复制 ambient 环境。
明确拒绝所有未知变量以及 `PYTHON*`、`LD_*`、`DYLD_*`、`BASH_ENV`、`ENV`。
若确实需要 native library 路径，其目录和加载闭包必须进入 pre/post 快照。

不调用 `site.addsitedir`，不执行 `.pth`；第三方包路径以冻结清单显式加入。
首版禁止 JIT、动态 torch extension 和 Triton 动态代码，除非其工具链、缓存
和生成文件进入独立证明。

### 4. worker 与 post audit

worker 只接受 bootstrap 注入的只读 contract/nonce/snapshot identity；缺少或
不匹配时必须在 Torch/Transformers/PEFT 导入前退出。rank 0 才能写正式
artifact/receipt，其他 rank 不得竞争共享 manifest。

所有 worker 退出后，controller 重新全量核对 source、interpreter、环境、
native runtime、base、VQ、train/validation/benchmark/leakage/config，并重新
严格解析 snapshot、training receipt、artifact manifest 与 reload receipt。
任一差异都只能产生 failed execution，不能打开 eval。

pre/post 哈希不能防御运行中“临时改写后恢复”的同权限攻击；生产环境还需要
只读 mount、不同 OS principal 或容器 ACL。代码不得声称超过这个权限边界。

## 文件责任

- `src/motion_eval/runtime/process.py`：verified group runner、`-I -S -B`、
  并发日志排空和 fail-one/kill-all。
- 新 controller-side stdlib-only bootstrap 模块：多根 capture、bundle、快照和
  post audit。
- `src/motion_eval/controller/state.py`：`FORMAL_BOOTSTRAP_FROZEN` 状态与 reducer。
- `src/motion_eval/controller/batch.py`：freeze -> HMAC anchor -> spawn -> post audit。
- `src/motion_eval/training_receipt.py`：新的 external-snapshot receipt schema；
  不得把它伪装为现有 in-process schema 2。
- `src/motionllm/training/sft.py`：只接受 controller 注入 contract；本地快照降为
  诊断证据。
- `qwenvl/train/full_sft.py`、`qwenvl/train/lora_sft.py`：在重型导入前断言 contract。
- catalog Qwen backend：只能走 controller bundle，不得再 path-based spawn
  Python/torchrun。
- `model_evaluation_agent/scripts/runner_support.py`：独立 verifier 必须重新跟随
  external snapshot，验证三方绑定，不能只读 training receipt。

## 最小测试门禁

1. argv 必须精确包含 `-I -S -B -c`；恶意 `.pth`/`sitecustomize` marker 不执行。
2. ambient `PYTHONPATH`、`LD_PRELOAD`、`BASH_ENV` 不继承。
3. 多根正常 import；module collision、symlink、pyc 和磁盘 fallback 全部拒绝。
4. capture 后替换磁盘源码：child 仍执行旧 bundle，post audit 拒绝发布。
5. interpreter/site/native 在 spawn 前交换时，进程不得启动。
6. snapshot 自洽替换但 external HMAC 未更新时拒绝。
7. code/env/base/data/config 任一 post 变化都不能 `FINETUNE_COMPLETE`。
8. 独立 verifier 必须真的重新解析 snapshot/receipt/artifact 三方绑定。
9. direct shell 仍 exit 78；direct Python 无注入 contract 时在重型导入前失败。
10. 先通过 controller NPROC=1 mock e2e，再在 Linux/CUDA 跑真实 1-step；
    N-rank 只能在 GPU acquire-many 与 NCCL 1-step 都完成后另行开放。
