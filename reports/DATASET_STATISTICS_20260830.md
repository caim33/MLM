# caimeng 数据统计与质量实验报告

统计日期：2026-08-30（Asia/Shanghai）  
规范数据根：`/wangbenyou-sulongjie/caimeng/dataset`  
实验输出：`/wangbenyou-sulongjie/caimeng/dataset/experiments/20260830_sanity_seed20260830`

## 1. 结论

当前数据整体可用：规范目录中的真实文件可读，Qwen 媒体链接完整，抽样的 motion
数组没有 NaN/Inf、全零或维度错误，抽样视频全部可以解码；严格版 Qwen QA 没有
发现训练/验证/benchmark 泄漏，也没有缺媒体、非法答案或重复 sample ID。

但不能把它描述成“完全无问题”。本次确认了以下事项：

1. **当前 Qwen motion 数据有入口兼容问题。** 历史 VM/M 问题使用
   `<motion_start><motion><motion_end>`，重构后的处理器要求单独的 `<motion>`。
   未迁移时会在第一条样本报 `MotionPlaceholderError`。
2. **HumanML3D 少 2 个基础 motion。** `009707`、`011059` 及其 `M` 镜像 caption
   都存在，但对应 4 个 NPY 不存在。
3. **MotionX frame caption 有少量不对齐。** 3 个 caption 无 motion，30 个 motion
   无 frame caption。
4. **SONIC 有 4 条命名方向不一致。** caption 的精确文件名无 NPY，但都能找到
   相反 `_M` 状态的 twin NPY，像是标注键名问题而不是动作内容完全丢失。
5. **extended_qtext 不是视频级隔离协议。** 其 train 与固定 benchmark 共享 491 个
   base/video，但规范化问题文本没有重复；只适合明确允许“同视频不同问题”的实验。

推荐默认使用 `qwen_qa/views/recommended/` 的 strict 数据；训练或推理前生成当前
`<motion>` 格式的派生视图，不覆盖历史标注。

## 2. 统计口径

- 文件数、字节数、扩展名、软链接状态和文件名配对是**全量扫描**结果。
- 扫描不跟随软链接，因此兼容旧路径、Qwen `views` 和媒体链接不会被重复计入容量。
- 容量是 regular file 的 logical bytes；GB 使用十进制 `10^9`。
- NPY 数值/shape 检查使用固定种子 `20260830`，每个主数据集抽 32 个文件。
- 视频解码检查每个媒体池固定抽 12 个文件；没有全量解码 115,990 个视频。
- caption UTF-8/非空检查为固定抽样；SONIC JSONL 142,220 行完成了全量解析。
- 全量文件扫描时间为 2026-08-30 03:26–03:30（Asia/Shanghai）。

## 3. 总体规模

不含实验输出，当前规范数据根共有：

- **743,786 个真实文件**
- **189,655,680,000 bytes（189.656 GB / 176.631 GiB）**
- **10,524 个数据组织软链接**：Qwen source tree 10,510 个，views 14 个
- **0 个断开的 Qwen 软链接**

| 数据集 | 真实文件数 | 大小（GB） | 主要内容 |
|---|---:|---:|---|
| MotionX | 540,925 | 150.497 | motion、视频、三类 caption |
| HumanML3D | 58,460 | 4.339 | motion、caption |
| SONIC | 142,219 | 32.891 | motion、三种格式的 temporal metadata |
| Qwen QA | 2,182 | 1.929 | 两个媒体池、annotation/source tree |
| **合计** | **743,786** | **189.656** | 不含软链接目标重复容量和实验输出 |

## 4. 目录级全量统计

| 规范相对路径 | 文件数 | 大小（GB） | 格式 |
|---|---:|---:|---|
| `motionx/motion` | 64,246 | 20.133 | 64,246 NPY |
| `motionx/videos` | 115,990 | 109.238 | 115,990 MP4 |
| `motionx/captions/complex` | 296,465 | 7.958 | 296,464 JSON + 1 TXT |
| `motionx/captions/original` | 5 | 0.530 | 4 JSON + 1 JSONL |
| `motionx/captions/frame` | 64,219 | 12.637 | 64,219 TXT |
| `humanml3d/motion` | 29,228 | 4.333 | 29,228 NPY |
| `humanml3d/captions` | 29,232 | 0.006 | 29,232 TXT |
| `sonic/motion` | 142,216 | 32.681 | 142,216 NPY |
| `sonic/captions` | 3 | 0.209 | JSONL、CSV、Parquet 各 1 |
| `qwen_qa/media/motionx_374` | 1,122 | 0.456 | motion/video/QA 各 374 |
| `qwen_qa/media/generated_success_assets` | 952 | 1.169 | motion/video 各 473 + 6 个清单文件 |
| `qwen_qa/source_tree` | 108 | 0.304 | 99 个数据/元数据文件 + 9 个 README |
| `qwen_qa/views` | 0 | 0 | 14 个推荐入口软链接 |

`motionx/captions/original/.ipynb_checkpoints/overall_action_overview-checkpoint.json`
与正式 `overall_action_overview.json` SHA-256 完全相同，是 24,458,325 bytes 的
冗余 checkpoint；本次只记录，没有删除。

### 4.1 文本监督分类合同

历史目录里的 `caption` 不能继续当作单一文本类型。训练 manifest 应显式记录
`text_type=qa|caption|short_description|detailed_description`：

| 文本类型 | 结构与粒度 | 当前来源 | 使用约束 |
|---|---|---|---|
| QA | question + A/B/C/D options + answer | Qwen strict、extended_qtext、benchmark | 用于选择题 SFT/GRPO/评测；不能当自由 caption |
| Caption | 带 frame、时间戳或 event 边界的局部文本 | MotionX frame 64,219 TXT；SONIC 142,220 temporal rows | 用于时序定位和稠密 caption；不是整段动作摘要 |
| 一句话描述 | 序列级的一句摘要或短 paraphrase | MotionX `key_action_summary.json` 64,249 条；HumanML3D 29,232 TXT 文件 | 用于 motion-text 对齐、检索、短生成；文件可含多条改写 |
| 详细描述 | 多句或结构化长文本，含阶段、身体、物体与时序细节 | MotionX `overall_action_overview.json` 64,249 条、`descriptions.json` 64,249 条，以及 complex 流水线 | 需要独立 prompt 和长度策略；complex 的 296,465 个文件不是 296,465 条独立描述 |

`V/M/VM` 与上述文本类型是正交轴：前者只表示非文本输入组合，后者表示监督目标。
现有正式 Qwen V/M/VM 清单都是 QA，不能把 description 直接写进 question/options 字段。

### 4.2 按 Text 类型统计

| Text 类型 | 来源 | 统计单位 | 数量 | 可配 motion | 备注 |
|---|---|---|---:|---:|---|
| QA | Qwen strict train | group / branch row | 813 / 1,626 | VM 813 | 每组 V、VM 各一条 |
| QA | Qwen strict val | group / branch row | 86 / 172 | VM 86 | 每组 V、VM 各一条 |
| QA | Qwen benchmark | group / branch row | 500 / 1,500 | M 500；VM 500 | 每组 V、M、VM 各一条 |
| QA | Qwen extended_qtext | train / val group | 1,768 / 86 | 按 branch | 替代视图，不计入 strict 总数 |
| Caption | MotionX frame | sequence TXT | 64,219 | 64,216 | 3 个文本无 M；30 个 M 无 caption |
| Caption | SONIC temporal | row / event | 142,220 / 352,703 | 142,216 exact | 4 个 `_M` key 待确认 |
| 一句话描述 | MotionX key action summary | text row / unique video | 64,249 / 64,249 | 64,246 | 3 条文本没有 motion |
| 一句话描述 | HumanML3D | TXT file / non-empty sentence | 29,232 / 87,384 | 29,228 files | 4 个文本文件没有 motion |
| 详细描述 | MotionX overall action overview | text row / unique video | 64,249 / 64,249 | 64,246 | 序列级概览 |
| 详细描述 | MotionX descriptions | text row / unique video | 64,249 / 64,249 | 64,246 | 分阶段、身体和交互细节 |
| 详细描述 | MotionX complex pipeline | pipeline file | 296,465 | 不按文件计 | raw/intermediate/final，不是独立样本数 |

汇总时保留原始单位：strict QA 共 1,399 个不重复 group、3,298 个 branch row；
一句话描述共 151,633 条非空文本，其中 93,474 个 sequence 文件/记录可配 motion；
详细描述覆盖 64,249 个唯一 MotionX 视频，包含 128,498 条规范长文本记录。
Caption 因 MotionX 使用 sequence file、SONIC 使用 row/event，不能直接给出无歧义的单一总数。

## 5. Motion 数据统计

### 5.1 MotionX

- 全量：64,246 个 NPY，20.133 GB。
- 固定抽样 32 个：全部为 `float32`、shape `(299, 263)`，长度均为 299。
- 32/32 可加载，未发现 NaN/Inf、全零或 feature 维度错误。
- 全部 64,246 个 motion 都能按“视频文件名去下划线”规则找到视频。
- 115,990 个视频中有 51,744 个没有对应 motion；因此当前 motion 是视频池的子集，
  覆盖率为 55.39%。这不是损坏，但训练前不能假设每个视频都有 NPY。

### 5.2 HumanML3D

- 全量：29,228 个 NPY，等于 14,614 个基础 ID 的 original + `M` 镜像。
- 固定抽样 32 个：全部为 `float32`、第二维 263。
- 抽样长度范围 64–199，median 163.5。
- 32/32 可加载，未发现 NaN/Inf、全零或 feature 维度错误。

### 5.3 SONIC

- 全量：142,216 个 NPY，32.681 GB。
- 固定抽样 32 个：全部为 `float32`、第二维 263。
- 抽样长度范围 67–864，median 171。
- 32/32 可加载，未发现 NaN/Inf、全零或 feature 维度错误。

三个 motion 数据集都使用 263 维特征，但长度策略不同：MotionX 已统一到 299，
HumanML3D 和 SONIC 仍为变长序列。混合训练时必须在 collator 中显式 pad/truncate，
不能直接堆叠原始数组。

## 6. 视频统计与抽样解码

| 媒体池 | 全量 MP4 | 抽样解码 | codec | 抽样时长 | 抽样分辨率 |
|---|---:|---:|---|---|---|
| MotionX 主视频 | 115,990 | 12/12 成功 | H.264 | 6.733–10 s，median 10 s | 320×240 到 856×480 |
| Qwen `motionx_374` | 374 | 12/12 成功 | H.264 | 4.4–10 s，median 10 s | 576×432 到 856×480 |
| Qwen generated success | 473 | 12/12 成功 | H.264 | 全部 10 s | 720×480 到 1280×720 |

没有发现抽样视频无法打开或首帧无法解码。分辨率并不统一，视频处理器必须统一
resize/pad；不要在数据层假设固定宽高。

## 7. Caption 与文件名配对

### 7.1 HumanML3D

- caption：29,232 个 TXT；固定抽样均为 UTF-8、非空、每文件 3 条非空文本。
- motion：29,228 个 NPY。
- caption 有而 motion 无：`009707`、`011059`、`M009707`、`M011059`。
- motion 有而 caption 无：0。

建议训练清单中暂时排除这 4 个 caption，除非能从原数据源补回 NPY。

### 7.2 MotionX frame caption

- 64,219 个 TXT，stem 无重复。
- caption 有而 motion 无：
  `1119212259`、`24103020185`、`24103022586`。
- motion 有而 frame caption 无：30 个；完整 ID 清单在
  `dataset_full_alignment.json` 中。

如果任务必须同时使用 motion + frame caption，应以交集清单为准，不能只按目录
长度 zip。

### 7.3 SONIC temporal caption

- `seed_metadata_v002_temporal_labels.jsonl`：142,220 行，全量 JSON 解析成功，
  event schema 合法。
- `seed_metadata_v004.csv`：142,220 个数据行 + 1 个 header。
- `seed_metadata_v004.parquet`：142,220 行，1 个 row group。
- 4 条 caption key 没有精确同名 NPY：
  - `inj_torso_walk_180_R_max_001__A083`
  - `jog_ff_stop_180_R_005__A307_M`
  - `lift_crate_walk_ff_stop_225_R_001__A202`
  - `turn_walk_360_002__A049_M`
- 这 4 条都存在相反 `_M` 状态的 twin NPY。不要静默自动别名；应先确认 caption
  中左右语义是否确实对应 twin motion，再修正 key。

## 8. Qwen QA 数据统计

### 8.1 媒体池

| 媒体池 | motion | video | QA/清单 | 配对结果 |
|---|---:|---:|---:|---|
| `motionx_374` | 374 | 374 | 374 QA JSON | 374 组三者完全对应 |
| `generated_success_assets` | 473 | 473 | 6 个 manifest/归档文件 | 473 组 motion/video 完全对应 |

`source_tree` 有 10,510 个媒体软链接：6,640 个 MP4、3,870 个 NPY；全量检查
没有断链。`views` 的 14 个入口链接也全部有效。

### 8.2 推荐 strict 数据

| 用途 | branch | rows | group | A/B/C/D |
|---|---|---:|---:|---|
| SFT train | VM | 813 | 813 | 203 / 203 / 204 / 203 |
| SFT train | V | 813 | 813 | 203 / 203 / 204 / 203 |
| SFT train | V+VM | 1,626 | 813 | 406 / 406 / 408 / 406 |
| SFT val | VM | 86 | 86 | 21 / 21 / 22 / 22 |
| SFT val | V | 86 | 86 | 21 / 21 / 22 / 22 |
| SFT val | V+VM | 172 | 86 | 42 / 42 / 44 / 44 |
| benchmark | VM | 500 | 500 | 125 / 125 / 125 / 125 |
| benchmark | V | 500 | 500 | 125 / 125 / 125 / 125 |
| benchmark | M | 500 | 500 | 125 / 125 / 125 / 125 |

GRPO strict train/val VM 分别为 813/86，与对应 SFT group 数一致。

全量 schema/路径检查结果：

- 非法答案：0
- 空 user question：0
- 重复 sample ID：0
- 缺失 motion/video：0
- V+VM 配对失败：0
- strict train 与 val group 重叠：0
- strict train/val 与 benchmark base ID 重叠：0
- strict train/val 与 benchmark 规范化问题签名重叠：0

### 8.3 extended_qtext

- train VM：1,768 rows，A/B/C/D 各 442。
- val VM：86 rows，A/B/C/D 为 21/21/22/22。
- train 与 benchmark 共享 491 个 base/video。
- train 与 benchmark 的规范化 question + unordered options 签名重叠为 0。

它不是错误数据，而是更宽松的实验协议。做视频级泛化、论文主表或与 strict 结果
比较时，不应混用这两种协议。

## 9. 真实 checkpoint GPU 烟测

checkpoint：
`/wangbenyou-sulongjie/caimeng/runtime/checkpoint-overlays/qa374_sft_step3best_checkpoint-48_merged_full`

冻结样本：strict val VM 分层抽 8 条，A/B/C/D 各 2 条；源文件 SHA-256：
`ed685a71410d1e7243e26919312c10022e714ba2de629beef0c9737b3c3a7626`。

实验过程与结果：

1. 把 media root 错设为 `media/motionx_374` 时，包含
   `generated_success_assets` 的样本被安全路径校验拒绝。正确 root 应为共同父目录
   `/wangbenyou-sulongjie/caimeng/dataset/qwen_qa/media`。
2. 使用历史 motion anchor 时，第一条样本报 `MotionPlaceholderError`。
3. 生成不覆盖源文件的兼容视图，将每条唯一的
   `<motion_start><motion><motion_end>` 精确替换为 `<motion>`；8/8 行均只替换一次，
   receipt 已记录输入/输出 SHA-256。
4. 兼容视图在 GPU 1 上 **8/8 推理成功**；模型加载约 10 秒，8 条生成约 15 秒。
5. 8/8 输出都有合法 `<answer>[A-D]</answer>`；4/8 正确（50%）。预测分布为
   A=4、B=2、C=2、D=0，而目标分布为 A/B/C/D 各 2。

8 条样本只用于证明数据链路和 checkpoint 能完整运行，不能作为正式模型指标；但
预测没有 D 且偏 A，值得在完整 val/benchmark 上继续检查类别偏置。

运行日志还有三项环境警告：tokenizer 建议开启 `fix_mistral_regex=True`；视频当前
回退到即将弃用的 torchvision decoder（缺 `torchcodec`）；`top_p/top_k` 在当前
生成配置中被忽略。这些不是本次数据损坏，但应在代码运行文档中列为环境待办。

## 10. 建议的数据使用规则

1. 新实验默认选 `qwen_qa/views/recommended/`，并记录 annotation SHA-256。
2. VM/M 数据先生成 `<motion>` 兼容派生视图；历史 JSON/JSONL 保持只读。
3. Qwen `data_path` 使用两个媒体池共同父目录 `qwen_qa/media`。
4. HumanML3D 排除 4 个无 motion caption；MotionX motion+frame 任务按交集取样。
5. SONIC 的 4 个 `_M` 键在人工确认左右动作语义前，不自动修复。
6. 混合 MotionX/HumanML3D/SONIC 时明确序列 pad/truncate 和 loss mask。
7. 视频输入统一 resize/pad，并在环境中固定 decoder 依赖。
8. strict 与 extended_qtext 的结果分别命名、分别汇报，不能只写“Qwen QA”。

## 11. 可复现实验产物

实验目录包含：

- `dataset_full_inventory.json` / `.tsv`：全量文件、容量、扩展名和链接清单
- `dataset_full_alignment.json`：全量 HumanML3D、MotionX、SONIC、Qwen 媒体配对结果
- `data_sanity_report.json` / `.md`：固定种子数组、视频、caption、QA 质量检查
- `qwen_vm_val_stratified_8.json`：冻结的原始 8 条子集
- `qwen_vm_val_stratified_8_current_anchor.json`：仅用于当前处理器的兼容视图
- `qwen_vm_val_stratified_8_current_anchor_receipt.json`：替换数量和哈希凭据
- `qwen_vm_val_stratified_8_predictions_current_anchor.jsonl`：GPU 推理输出
- `qwen_vm_val_stratified_8_current_anchor.log`：成功运行日志
- `qwen_vm_val_stratified_8_raw_anchor.log`：历史 anchor 失败证据
- `data_sanity.py`、`dataset_inventory.py`、`full_alignment_audit.py`、
  `prepare_compatible_subset.py`：可重复运行的检查脚本

本报告中的“正常”只代表上述全量元数据检查和确定性抽样实验没有发现异常，不代表
未抽到的每个大型二进制文件都已逐个解码或数值扫描。
