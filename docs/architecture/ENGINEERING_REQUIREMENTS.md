# 新工程要求

本文件替代旧重构阶段过度耦合的工作要求，作为 clean codebase 的验收合同。

## 1. 目标

1. 代码结构一眼可辨：核心、多模态数据、Qwen 适配、训练、GRPO、统一评估和 legacy 明确分层。
2. 基础 CPU 环境可以导入核心包、查看全部 CLI 帮助并运行数据/控制器测试。
3. GPU、权重、数据和网络只在显式执行对应命令后加载。
4. 路径、数据集、模型和输出全部配置化，不把任何人的服务器目录写进活动代码。
5. 保留旧命令和旧 import 的必要兼容层，但新业务逻辑只有一个权威实现。
6. 文档能够让新接手者判断“现在能做什么、缺什么、从哪里改”。

## 2. 模块要求

| 模块 | 职责 | 禁止事项 |
|---|---|---|
| `motionllm.contracts` | 纯领域类型与校验 | 导入 Torch、Transformers 或旧 Qwen 代码 |
| `motionllm.data` | JSONL、路径、message、dataset、collation | 硬编码数据路径、坏样本替换 |
| `motionllm.motion` | 数组、归一化、时序 | 训练状态和远端操作 |
| `motionllm.fusion/models` | motion token 与模型融合 | 数据集注册和批次控制 |
| `motionllm.training` | SFT/LoRA 配置、artifact、保存/重载 | 评估主表发布 |
| `motionllm.grpo` | reward、rubric、group logic | 自建另一套答案 parser |
| `motion_eval.data` | 严格 JSON、benchmark、receipt、leakage | 模型加载 |
| `motion_eval.controller` | 批次状态与阶段 gate | 隐式执行训练或评估 |
| `qwenvl/`、顶层 `models/` | 旧入口兼容 | 成为新的核心实现位置 |
| `legacy/` | 来源与迁移证据 | 被活动代码导入 |

## 3. 数据合同

- canonical sample 必须有稳定 `sample_id`、`group_id`、typed modality、严格选项和严格 gold answer。
- 相对媒体路径只能解析在声明 root 内；绝对路径也不得逃逸 root。
- 重复 JSON key、NaN/Infinity、非法 UTF-8、空 JSONL 行和非 object 行全部拒绝。
- 任一坏样本在其原行失败，不允许跳到下一行或缩小正式评估分母。
- 旧 Qwen conversation 格式只在适配层转换为 canonical contract。

## 4. 运行模式

| 模式 | 允许行为 | 产物资格 |
|---|---|---|
| `inspect` | 读取配置、显示计划、校验 schema | 无训练/评估资格 |
| `dev` | 小样本本地调试 | 仅调试 |
| `preflight` | 依赖、GPU、数据、模型 reload 检查 | 不可发布 |
| `production` | 显式启用、完整输入清单、固定输出目录 | 通过 verifier 后才可发布 |

生产模式默认关闭。旧 blocker 不作为新架构本身的设计目标，但不能把旧 smoke、debug、prediction 或 checkpoint 冒充新生产结果。

## 5. 测试要求

- `tests/unit`：单模块纯逻辑。
- `tests/contract`：JSON、sample、CLI 和兼容接口。
- `tests/integration`：跨数据、模型、训练或控制器流程。
- `tests/stress`：恶意 JSON、路径逃逸、并发、资源与状态异常。
- `tests/gpu`：单独运行并明确记录硬件、驱动、依赖和模型资产。

通过数必须对应当前 checkout 的真实复跑；历史报告只作为历史证据。

## 6. 文档要求

每个公共入口必须写清：用途、前置条件、命令、是否会加载模型/GPU、会写什么、
输出到哪里、失败如何处理。架构或状态改变时同步更新
`docs/architecture/ARCHITECTURE.md`、`docs/status/README.md` 和
`docs/guides/USAGE_GUIDE.md`。
