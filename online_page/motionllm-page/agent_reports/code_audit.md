# MotionLLM 历史快照代码审计：从 motion token 到可信评测

> 审计对象：`D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801`  
> 审计方式：只读静态代码、配置与历史日志审计；未启动训练、未生成新的评测结果。  
> 当前协议边界：这个快照里的 checkpoint、accuracy、prediction、option-score、proxy 与 TensorBoard 数字只能作为历史诊断证据。任何新 batch 必须先让注册表中每个模型生成该 batch 专属的新 finetune artifact，并在全局 finetune barrier 全部结束或正式阻塞后，才允许开始 eval；历史结果不得进入新的主结果表。

## 1. 结论先行

这套项目不是“缺少一个更强模型”，而是有五个会直接改变结论可信度的系统性问题：

1. **同一 motion 输入在两套并存代码中会被编码成不同数量的 token。** 根目录/`remote-code` 版本与 `remote_edit` 版本不一致；旧 GRPO 数据写入文本 `<motion_start><motion><motion_end>`，根版本会把完整 motion embedding 插值压到这段文本所占的少数 token 槽，`remote_edit` 才补了 raw motion 对齐和 SFT-style `160001` pad 检查。若线上到底使用哪份源代码没有锁死，所谓 “VM 比 V 更好” 甚至不能证明模型真的收到等价、充分的 motion 证据。
2. **SFT 训练链可能没有从预期 stage-1 checkpoint 继续。** `full_sft.sh` 定义了 `init_model_or_path`，实际启动参数仍然传 `llm`；同时数据处理无条件截到 4096，尽管脚本宣称 8192。二者都属于静默语义改变。
3. **Rubric 的空标准会天然得到 80/100，明确错误的数值还可能获得 0.25 部分分。** 这不是 judge 偏好问题，而是确定性的奖励实现错误，会诱导模型或脏数据利用空 rubric/错误值漏洞。
4. **GRPO 的验证集没有测到它独有的优化目标。** 配置是 100% 参数全量更新、`beta=0.001`，历史日志显示后期 KL 已明显上升；而 val 是 VM-only，VM-vs-V group bonus 在验证时恒为 0，无法回答 motion 分支是否真的带来增益。
5. **历史 MCQ 主链不满足当前严格评测协议。** 生成式 parser 接受 `<answer>The answer is A</answer>` 之类的宽松格式；多个脚本使用强制 option log-prob，只能做诊断；benchmark manifest 没有内容 hash、泄漏报告或完整冻结证明，而且答案位置严重不平衡。

另有一个独立 P0：`D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\full_sft.sh:31` 存在明文实验跟踪服务凭据。本文不复述凭据；应立即撤销/轮换、从版本历史清除，并改为进程级 secret 注入。

## 2. 真实数据流

### 2.1 SFT

```text
JSON/JSONL
  -> qwenvl/data/data_processor.py
     - chat template + assistant label mask
     - image/video 预处理
     - .npy motion + Mean/Std 标准化 + 补帧
     - 每 L/divisor 个 motion 时刻插入一组 160001 placeholder
  -> FlattenedDataCollatorForSupervisedDataset
  -> models/qwen3_vl_motion.py
     - VQ-VAE motion encoder
     - LayerNorm(512) -> MLP 512→4096→2560
     - 用 motion embedding 替换 placeholder embedding
  -> Hugging Face Trainer
```

关键入口：

- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\full_sft.sh:146-192`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\train\full_sft.py:412-604`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\data\data_processor.py:880-914,990-1038,1237-1331`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\models\qwen3_vl_motion.py:480-584`

### 2.2 GRPO

```text
YAML
  -> scripts/train_grpo_ms_swift.sh
  -> qwenvl/grpo_ms_swift/runner/train_grpo_ms_swift.py
  -> swift rlhf --rlhf_type grpo
  -> 成对记录：VM(video+motion) / V(video-only)，同 group_id
  -> 每 prompt 采样 6 个 completion
  -> option accuracy + strict-ish format + VM/V group bonus
  -> 当前主配置 full-model update
```

数据生成在：

- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\build_motionx_deepseek_grpo_dataset.py:97-118`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\build_motionx_report_grpo_dataset.py:255-277`

奖励注册与 group bonus 在：

- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\grpo_ms_swift\plugins\swift_external_rewards.py:61-168`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\grpo_ms_swift\plugins\group_bonus_vm_v.py:48-100`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\grpo_ms_swift\plugins\rewards_semantic_format.py:25-38,113-126`

### 2.3 Rubric

```text
GT description
  -> Qwen text extractor -> 第一个可解析 JSON object
  -> ensure_criteria_ids（只做宽松归一化）
  -> criteria JSONL
candidate + criteria
  -> Qwen judge -> IDs + coarse fields
  -> reward_v2.py 确定性加权
```

关键入口：

- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\extract_motion_criteria_v2.py:73-96`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\judge_motion_caption_v2.py:84-119`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\qwen_text.py:10-46,212-231`
- `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:270-347`

### 2.4 评测

历史代码同时存在三种协议，必须在网页上明确分栏，不能合并：

| 协议 | 脚本例子 | 可以进入新主表吗 |
|---|---|---|
| 自由生成、严格完整标签 | 需要新实现为 `fullmatch(<answer>[A-D]</answer>)` | 可以，前提是 fresh finetune + global barrier + 固定分母 |
| 当前快照的宽松生成解析 | `tools/eval_grpo_mcq_accuracy.py` | 不可以，修复 parser 后重跑 |
| 强制候选 option log-prob | `codex_remote_tools/eval_motionr1_lora_mcq_score.py`、`eval_open_vlm_mcq_score.py` 等 | 仅诊断，永不进入新主表 |

## 3. 按优先级排列的代码事实与修复

### P0-A：锁死唯一 motion token 协议与部署 revision

**证据**

- 两个 dataset builder 都写入字面量 `"<motion_start><motion><motion_end>\n"`：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\build_motionx_deepseek_grpo_dataset.py:105-106`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\build_motionx_report_grpo_dataset.py:264-265`
- 根版本如果找不到时间步级 `160001` placeholder，会寻找文本 boundary span，并令 `target_len = end - start`，把完整 motion embedding 插值到这几个文本 token：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\models\qwen3_vl_motion.py:540-584`
- `remote_edit` 版本新增 motion length divisor、raw tensor 补齐、必须存在 `160001` SFT placeholder 和预期展开长度校验：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\remote_edit\qwen-vl-finetune\models\qwen3_vl_motion.py:108-111,487-516,543-558,830-853`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\remote_edit\qwen-vl-finetune\qwenvl\grpo_ms_swift\runner\train_grpo_ms_swift.py:314,371-400`
- 根目录、`remote-code`、`remote_snapshots` 与 `remote_edit` 有多份运行代码；`remote_edit` 文件内容/大小与根版本不同，却没有一个 batch manifest 证明实际部署的是哪一 revision。

**影响**

旧链路可能把数百帧动作证据压成少量文本槽；即使训练 loss 或 MCQ accuracy 提升，也不能归因于正确的 motion temporal token stream。多份源码又让复现者无法确定 server 上的真实行为。

**修复验收**

1. 合并为一个 canonical package，删除运行时 fallback；数据层只允许独立 `<motion>` marker，processor 必须展开为 `N = padded_length / divisor` 个 `160001`。
2. forward 前断言每样本 `placeholder_count == encoded_motion_steps`，不相等即 fail closed。
3. 每个 batch manifest 保存：git/source hash、config hash、processor/tokenizer revision、VQ-VAE hash、dataset hash、预训练基座 hash。
4. 增加反事实测试：同视频替换 motion、打乱 motion、全零 motion；VM logits/生成必须发生可测变化，且 V 分支保持不变。

### P0-B：修复 Rubric 的“空标准高分”和“错误值正奖励”

**证据**

- `_fraction_score` 和 `_present_aligned_score` 在 `total <= 0` 时返回满分：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:162-171`
- 没有 reasoning criteria 时直接给 20 分：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:312-323`
- 因而空 criteria + 空 judgment 会自动得到 global 5 + basic 25 + body 10 + numeric 10 + laterality 5 + orientation 5 + reasoning 20 = **80/100**，即 reward 0.8；temporal/language 默认 0 不改变这一事实：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:325-347`
- `wrong_value_ids` 被并入 `numeric_sem`，之后 semantic-only 分支仍给 0.25：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:286-303`
- `lat_wrong`、`ori_wrong`、`reasoning_contra` 被收集但没有直接扣分：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:305-311,335-347`
- extractor 只解析第一个 JSON object，criteria normalization 没有强制 prompt 声称的数量、字段、单位、区间与 ID 唯一性：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\qwen_text.py:10-46`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\rubric_rl\reward_v2.py:99-132`

**修复验收**

- rubric 缺失、空数组、ID 重复、类型数量不足、非法单位或 target range 均应标为 invalid sample，reward 不计算且进入可审计错误分母。
- `total == 0` 的可选维度应返回 0 并重新归一化到“实际存在的维度”，而不是满分；必需维度为空则整条 rubric fail closed。
- `wrong_value_ids` 必须从 semantic 集合剔除并显式为 0 或负分；laterality/orientation/reasoning contradiction 同理。
- 建立 100–200 条双人标注 calibration set，报告 judge-human 一致率、维度混淆、重复 judging 方差；盲化候选名并随机化顺序。

### P0-C：修复 SFT 续训路径、静默截断与错误样本替换

**证据**

- `full_sft.sh` 定义 `init_model_or_path`，但真正传给 Python 的仍是 `${llm}`：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\full_sft.sh:15-16,146-166`
- 脚本配置 8192，但 processor 设置 `MAX_DEBUG_SEQ_LEN = 4096` 并无条件 `min(..., 4096)`：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\full_sft.sh:173-192`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\data\data_processor.py:105,1037-1038`
- assistant mask 硬编码 token `77091` 与 EOT `151645`，没有 tokenizer revision 校验或“至少一个 supervised token”断言：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\data\data_processor.py:296-314`
- 取样失败后尝试 `i+1`，会静默丢掉坏样本并重复邻近样本，缺少错误 manifest：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\data\data_processor.py:544-599`
- `full_sft.py` 强制禁用 eval：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\train\full_sft.py:446-452`
- `lora_sft.py` 用 checkpoint basename 是否含任意字母 `a` 来判断 Qwen3 MoE；许多普通目录名会误命中：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\train\lora_sft.py:326-339`

**修复验收**

- 把 init/base checkpoint 变成一个必填参数，启动日志打印 resolved absolute path + hash，并在保存 artifact 中记录 parent artifact ID。
- 取消 debug 常量，长序列截断必须产生 `truncated=true`、原/后长度、被截断的 assistant label 数；若答案被截断直接判 invalid。
- 从 tokenizer/chat template 动态生成 assistant boundary；每 batch 断言 supervised tokens、motion placeholders 与各模态样本数。
- 数据错误不允许邻样本替换；保留原 sample_id、错误类型并计入固定分母，训练前先完成全集 preflight。
- MoE 由 `config.model_type`/`architectures` 判断，不从路径名猜。

### P1-A：证明 GRPO 学到的是 motion 增益，而不是全模型漂移

**证据**

- 主配置为 `tuner_type: full`、600 steps、`lr=1e-6`、6 generations、`beta=0.001`：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\configs\grpo\motionx_374_hard_mcq_r6_600_prompt_vm_motion_v_video_finalckpt.yaml:54-59,72-86`
- 历史 TensorBoard export 记录约 4.469B 参数、100% trainable，后期 KL 约 1.6；这是风险诊断，不是新结果：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\tensorboard_exports\grpo_ckpt600_logging.jsonl:1`
- 带 eval 的后续配置使用 `val100vm`，但 VM/V group bonus 只有 group 同时出现 VM 与 V 时才生效；VM-only validation 的 bonus 恒为 0：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\configs\grpo\motionx_374_hard_mcq_r6_600_train2270_val100vm_len768_lenreward.yaml:47-79,94-102`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\grpo_ms_swift\plugins\group_bonus_vm_v.py:48-100`
- runner 的 precheck 检查 branch/motion 是否存在，但没有证明每个 group 恰好一对 VM/V、题面/选项/视频一致、ID 唯一、答案位置平衡或无泄漏：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\grpo_ms_swift\runner\train_grpo_ms_swift.py:251-331`

**修复验收**

- 先做 projector/adapter-only、LLM LoRA、full 三组消融；保存 trainable parameter manifest，不能只相信 `tuner_type` 名称。
- paired validation 必须同时含同 group_id 的 VM/V，并报告 `Acc_VM`、`Acc_V`、paired delta、bootstrap CI、bonus gate hit-rate。
- 训练同时监控 KL、entropy、clip ratio、VM/V reward 分解；设 target-KL/early stop，而不是只看总 reward。
- 加 motion-shuffle、zero-motion、wrong-motion 三个反事实分支，区分“使用 motion”与“仅从视频/题面猜中”。

### P1-B：把 MCQ 主评测变成单一、严格、可冻结的协议

**证据**

- 当前生成 parser 在 `<answer>...</answer>` 内继续搜索第一个独立 A–D，所以格式不严格：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\tools\eval_grpo_mcq_accuracy.py:25-44`
- 单样本 generation 周围没有错误隔离；一个坏媒体/OOM 会中止整个 run，而不是产生固定分母 error row：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\tools\eval_grpo_mcq_accuracy.py:146-196`
- 多个历史脚本直接比较候选选项 log-prob，属于 option-score 诊断：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\codex_remote_tools\eval_motionr1_lora_mcq_score.py:241-286`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\codex_remote_tools\eval_motionr1_lora_option_text_score.py:41-90`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\codex_remote_tools\eval_open_vlm_mcq_score.py:235-269`
- benchmark manifest 只记录相对路径/数量/选择规则，没有文件 hash、派生链、去重和泄漏报告：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\data\benchmark\manifest.json:1-30`
- 静态统计 `data/benchmark/text/QA/QA_500.jsonl` 的 500 题答案位置为 A=128、B=213、C=134、D=25；多数类 B 已达 42.6%，D 仅 5%。这使文本偏置/字母偏置足以制造看似不错的分数。

**修复验收**

- 唯一有效 parser：去掉首尾空白后，完整匹配 `^<answer>[A-D]</answer>$`；任何多余 prose、多标签、缺标签都记 syntax error。
- 每样本 try/catch，固定 frozen sample_id 分母；error 进入主表而不是被过滤。
- 冻结 manifest 必须含逐文件 SHA-256、逐 sample provenance、split/去重/泄漏报告、答案位置分布、选项 permutation seed。
- 主表同时展示生成 accuracy、strict syntax rate、error rate、四类答案召回；option-score 单独放“诊断附录”，不可混入。

### P1-C：修复 judge 的身份/顺序偏置与变分母

**证据**

- semantic judge prompt 固定暴露 `PREGRPO`、`OLD600`、`NEW600`，候选顺序也固定：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\tools\judge_description_semantic_qwen3vl.py:121-152`
- 汇总时直接过滤 `judge_error`，导致不同模型/批次使用不同分母：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\tools\judge_description_semantic_qwen3vl.py:206-226`

**修复验收**

- 对每个样本随机匿名为 Candidate A/B/C，保存 permutation；做 swapped-order 复判。
- judge error 不删除：报告固定分母、coverage、error rate；主比较用 paired complete cases 但同时披露缺失原因。
- judge model、revision、prompt hash、temperature、seed 全部进入 manifest；在人工 calibration set 上报告一致率与置信区间。

### P2：模型/LoRA 代码的可维护性与隐性死模块

- motion projector 写死 `512→4096→2560`，限制了非 Qwen3-VL-4B 隐藏维度：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\models\qwen3_vl_motion.py:106-116`
- 定义了 `motion_postnorm`，实际 encode path 只调用 `motion_prenorm` 与 `motion_proj`，postnorm 是死模块：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\models\qwen3_vl_motion.py:111-112,480-525`
- VQ-VAE 用 `strict=False` 加载，只打印 missing/unexpected keys，不设失败阈值：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\models\qwen3_vl_motion.py:1212-1224`
- LoRA 脚本把整块 `visual`、`motion_encoder` 等列入 `modules_to_save`；这可能令名为 LoRA 的 run 保存/训练大模块，必须用实际 trainable-name/parameter-count 证明：  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\scripts\lora_sft.sh:28-35,83-110`  
  `D:\MotionLLM\History\Artifacts\project_snapshot_20260729_144801\qwenvl\train\lora_sft.py:290-304,421-425`

建议从 config 的 hidden size 构造 projector；要么实际应用 postnorm，要么删除并在训练开始时断言所有预期 trainable 参数都有梯度；VQ-VAE 核心 key 缺失时 fail closed；每个 artifact 写出完整 trainable parameter 清单与比例。

## 4. 新 batch 的最小可信流水线

以下不是可选“最佳实践”，而是让结果能进入主表的 gates：

1. **Freeze gate**：数据、processor、prompt、rubric、parser、模型 revision、VQ-VAE、所有配置都生成 hash manifest；完成 split 去重与 train/eval 泄漏审计。
2. **Pair integrity gate**：每个 `group_id` 恰好一条 VM + 一条 V；question/options/video/answer 完全一致；motion 仅 VM 可见；答案位置分布受控。
3. **Rubric schema gate**：空/缺/重复/非法 criteria 全部正式阻塞，不允许以默认高分继续。
4. **Per-model fresh finetune gate**：注册表 15 个模型各自产生该 batch 专属 artifact，保存 parent/hash/trainable manifest；历史 checkpoint 不能冒充。
5. **Global barrier**：15 个模型 finetune 全部 `complete` 或正式 `blocked` 之后，才开始任何模型的 eval。
6. **Eval gate**：仅自由生成 + 完整严格标签；固定 denominator；逐样本 error row；旧 option-score/历史 prediction/proxy 只能进诊断区。
7. **Modality proof gate**：VM/V paired delta + zero/shuffle/wrong motion 三个反事实；没有 modality sensitivity 就不能声称 motion reasoning。
8. **Uncertainty gate**：paired bootstrap CI，必要时 McNemar；judge 评测再加匿名顺序随机化与人类校准。

## 5. 最值得立刻做的三个实验

### 实验 1：motion evidence sanity matrix

固定同一视频/题目，依次输入 correct motion、shuffled motion、another-sample motion、zero motion、V-only。记录每个选项生成概率、严格生成答案、hidden-state 差异。若 correct 与扰动组几乎无差异，先修 motion token 链，不要继续扩大 GRPO。

### 实验 2：参数更新边界消融

同一 frozen train/eval batch 比较：projector-only、motion encoder + projector、LLM LoRA + projector、full GRPO。核心不是谁分最高，而是 `paired VM−V` 是否在更少参数、更低 KL 下稳定出现。这会直接回答当前 100% full update 是否必要。

### 实验 3：rubric adversarial unit suite

至少覆盖 empty criteria、缺失维度、重复 ID、wrong numeric、wrong left/right、wrong camera orientation、contradicted reasoning、parser partial JSON。先写确定性期望分，再跑 judge。当前 empty 与 wrong-value 用例应该立即失败，修复后才能用于 RL。

## 6. 与代码最紧密、建议补进论文网页的三篇原始文献

这里只补三篇“能直接改变你的实现/实验设计”的文献，避免与通用 motion 论文综述重复：

1. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** — GRPO 的原始来源。项目当前正使用 group-relative reward、多个 completion 和 KL 正则；应对照原文解释 group normalization、KL 项与采样设计，再决定当前 `beta=0.001` 和 VM/V group bonus 是否仍保持可识别的优化目标。  
   原文：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
2. **LoRA: Low-Rank Adaptation of Large Language Models** — 原始设计核心是冻结预训练权重，只注入低秩可训练矩阵。它与当前 `tuner_type: full`、以及 LoRA 脚本里大范围 `modules_to_save` 形成直接对照；网页可以把“名字叫 LoRA”与“实际 trainable 参数比例”明确区分。  
   原文：[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — 原文系统讨论 position、verbosity、self-enhancement bias。它直接解释为什么固定暴露 `PREGRPO/OLD600/NEW600` 名称和顺序不可信，也支持匿名候选、交换顺序、人类 agreement calibration。  
   原文：[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)

## 7. 网页呈现建议

为了让读者明确区分“事实、风险、论文启发、待验证假设”，建议每条卡片固定包含：

- **代码事实**：文件 + 行号 + 一句话行为描述。
- **为什么影响结论**：会影响输入语义、优化目标、分母还是归因。
- **论文对应点**：原论文提出什么；不要暗示代码已经遵循。
- **当前状态**：`confirmed bug` / `design risk` / `needs experiment`。
- **验收测试**：必须可执行、可判定 pass/fail。

最重要的视觉分区应是：

```text
历史诊断（不可进主表）
  ├─ historical accuracy/predictions
  ├─ forced option-score
  ├─ proxy AGCN/MotionCLIP
  └─ old TensorBoard/checkpoints

新 batch 主结果（只有过 gates 才出现）
  ├─ fresh finetune artifact × 15
  ├─ global finetune barrier
  ├─ strict free-generation eval
  ├─ fixed denominator + errors
  └─ paired VM/V + counterfactuals + CI
```

这一区分比再增加一张总 accuracy 表更重要：它能防止读者把“旧代码跑过的数”误认为在当前统一协议下可比较的证据。
