# MotionLLM 相关论文深读报告（Motion / LLM / VLM 主线）

> 研究范围：人类动作表征、动作—语言模型、视频/3D 动作理解与评测。  
> 原则：只使用论文原文、作者项目页或官方代码仓库；历史结果仅用于理解论文，不应写入 MotionLLM 新一批主结果表。  
> 图像：每篇均从原始 PDF 的主方法图裁出，180 dpi；`PDF 页`指 PDF 文件的物理页码，而非页脚印刷页码。  
> 对项目的判断基于当前快照中的 V（video+text）/ VM（video+motion+text）、Qwen-VL/Qwen3-VL、SFT/GRPO 与成对选择题设计。

## 一眼看懂：论文分层与优先级

| 层级 | 论文 | 与本项目关系 | 建议优先级 | 一句话结论 |
|---|---|---|---:|---|
| A 直接竞品 / 直接可用 | MotionLLM | 视频/动作双分支、动作问答最接近 | P0 | paired data 能提升总分，也会让动作分支向受限的视频分支妥协，必须测负迁移 |
| A | LLaMo / Human Motion Instruction Tuning | 原生动作输入、跨模态融合 | P0 | 保留时序动作特征，并让语言选择关键帧，比粗暴压成少量动作 token 更有希望 |
| A | MoChat | 骨架分组、空间与时间定位 | P0 | 按身体部位编码，加独立时间回归头，适合构造真正依赖 motion 的题 |
| A | LLMs are Good Action Recognizers | 离散动作 token + LLM 分类 | P1 | 离散化和层级 codebook 对分类有效，但不能据此宣称获得动作推理 |
| A | SMD | 无动作 encoder，骨架转结构化文字 | P0 | 最便宜、最可解释的强基线；Top-K 描述粒度应随任务改变 |
| A | HumanMoveVQA | 世界坐标轨迹/朝向 VQA | P0 | 动作理解不应只问动作名，必须覆盖距离、顺序、方向、轨迹可行性 |
| A | NextMotionQA | 动作 VLM 评测与 judge 可靠性 | P0 | LLM judge 在细粒度身体部位层面并不可靠，Rubric 必须有几何或人工校验 |
| A | Ego3DLM | SFT + GRPO、动作/文字跨模态奖励 | P0 | GRPO 应奖励“动作正确、文字正确、动作—文字一致”，而非只奖励 VM 赢 V |
| B 上游表征 / 生成 | MotionGPT | motion token 化 + instruction tuning | P1 | 统一任务很有价值，但 instruction tuning 也可能因配对文本少而损害语言能力 |
| B | MotionGPT-2 | body/hand 分离 codebook、SMPL-X | P1 | 身体和手的统计结构不同，不应强塞进同一个均匀 token 流 |
| B | MotionGPT3 | 连续 latent、双流 Transformer | P0 | 单流统一表征会产生模态干扰；连续 latent 对理解可能优于 VQ token |
| B | Motion-X / Motion-X++ | RGB—SMPL-X—文本—音频数据底座 | P0 | 可做同步成对数据，但必须保留来源、置信度并按主体/来源防泄漏 |
| B | MotionCLIP | 动作与 CLIP 语义空间对齐 | P1 | 语义对齐可作辅助损失，但 CLIP 不擅长左右、旋转方向和精细时序 |
| B | T2M-GPT | 稳定 VQ tokenizer 与自回归生成 | P1 | EMA/code reset/输入扰动值得借鉴；生成 FID 不能代替理解能力 |

## 先给 MotionLLM 项目的 10 条结论

1. **新增一个 SMD 结构化文本分支。** 直接把 NPY/SMPL 变成根轨迹 + 关键关节角度描述，和 raw-motion encoder 做 `raw / SMD / raw+SMD` 三路消融。SMD 在跨来源 QA 上的稳定性，是当前最便宜的验证方向。
2. **把“是否真的使用 motion”变成正式指标。** 对每题做 motion shuffle、motion zero-out、video shuffle、只给 motion、只给 video；报告正确答案概率下降量，而不只报告最终 accuracy。
3. **GRPO 奖励从分支胜负改成样本级证据奖励。** 建议拆为 `answer_correct + format + motion_text_consistency + counterfactual_sensitivity`，并单独限制 hallucination；Ego3DLM 的 cross-modal matching reward 是直接参考。
4. **保留全局世界坐标。** HumanMoveVQA 与 SMD 都显示根轨迹、距离、朝向、顺序是当前模型的关键盲点。仅使用局部关节坐标会把最有辨识度的信息抹掉。
5. **题目轴要显式分开。** 至少按 body part / direction / order / count / trajectory / temporal grounding / reasoning / hallucination 分类，避免平均分掩盖方向和顺序退化。
6. **hard negative 应是“语义相近、几何错误”。** 固定动作名，只改左右、关节、次数、顺序、位移方向或时间段；比随机换答案更能迫使模型读取 motion。
7. **不要让所有 motion token 共享一个无结构投影。** 比较身体分组（MoChat）、body/hand 双 codebook（MotionGPT-2）、连续双流（MotionGPT3）三种结构。
8. **LLM judge 只能做粗粒度辅助。** NextMotionQA 的系统级相关性在细粒度 judge 上甚至可为负；主评测应以 exact/geometry/human audit 为主。
9. **paired 数据不是自动有效。** MotionLLM 原文表明联合配对提高平均分，却可能降低 sequence 轴，原因是动作分支向只有 8 帧的视频瓶颈妥协。训练和验证都要有全局 phase barrier 以及分模态的负迁移审计。
10. **把生成论文当“表征候选库”，不要把 FID 当理解证据。** MotionGPT 系列、MotionCLIP、T2M-GPT 的价值主要是 tokenizer、对齐方式和训练阶段，而非其生成指标本身。

---

## A. 直接竞品 / 直接可用

### A1. MotionLLM: Understanding Human Behaviors from Human Motions and Videos（2024）

- **定位与区分**：与本项目最直接的竞品。它把视频和 3D motion 作为两种可替换的视觉提示，不是同时输入的 VM 融合模型；因此特别适合拿来解释“V 和 VM 为什么不一定相加”。
- **核心问题**：能否用一个 LLM 同时理解真实视频和 3D 人体动作，并在动作问答、行为描述和视频理解上共享知识。
- **方法机制**：视频用 LanguageBind encoder，motion 用 VQ-VAE motion encoder；各自经过 visual-language translator 接入 Vicuna-7B。第一阶段冻结 encoder 与 LLM，只训练 translator；第二阶段冻结 encoder，联合训练 translator 与 LLM。训练既有不配对的 motion/video instruction，也有来自同一动作的 paired instruction。
- **数据与评测**：MoVid 指令数据含约 272K H3DQA、24K Motion-X caption、200K Motion-XQA，并混入 Valley、Video-ChatGPT 等视频数据。MoVid-Bench 共 1,350 对问答（700 motion、650 video），按 body、sequence、direction、reasoning、hallucination 分轴；另测 BABEL-QA、MVBench 的人体动作子集、ActivityNet-QA。
- **关键发现**：MoVid-Bench motion 平均准确率 49.50%，video 49.00%；MotionGPT 为 36.86%，Video-LLaVA 为 42.53%。最重要的不是总分，而是消融：motion-only 38.48 → unpaired joint 48.07 → paired 49.50，但 sequence 从 46.20 降到 36.84；作者明确指出视频 encoder 仅处理 8 帧，paired training 可能迫使 motion 分支向视频瓶颈“妥协”。
- **作者自述限制**：视频采样帧少、压缩能力有限；动作与视频分支可能出现能力不对称。论文的 GPT-3.5 评分也只能视为辅助指标。
- **对本项目的 know-how**：
  1. paired 数据必须逐轴审计，不能只看总 accuracy；
  2. 对 V/VM 使用匹配的时间覆盖与容量，避免 motion 高分辨率信息被低帧率 video 拖累；
  3. 加 `shuffle motion / shuffle video / only-motion / only-video` 反事实测试；
  4. 将 sequence、direction、hallucination 单独作为 GRPO reward 或约束。
- **原始来源**：[论文](https://arxiv.org/abs/2405.20340) · [PDF](https://arxiv.org/pdf/2405.20340)
- **主图资产**：`assets/figures/motion_motionllm.png`；Figure 2，PDF 第 5 页。图意：MotionLLM 的视频/动作双 encoder 与 translator，以及两阶段联合指令训练。

### A2. Human Motion Instruction Tuning / LLaMo（CVPR 2025）

- **定位与区分**：MotionLLM 的直接后续竞品。与 MotionGPT 的“动作离散成语言 token”不同，LLaMo 明确保留 motion 的原生连续时序特征，再用语言引导融合。
- **核心问题**：动作转语言 token 会丢失细微姿态与时间信息；如何让 LLM 读取原生 motion，同时兼容视频输入。
- **方法机制**：冻结 video/motion encoder，训练 Feature Enhancer、Cross Talker 和 LLM。Cross Talker 先用文本引导的 top-K frame selection 找关键帧，再做局部+全局自适应注意力与 motion-text 双向注意力。视频可以先经 motion estimator 得到动作序列。
- **数据与评测**：使用 MoVid、HumanML3D、KIT-ML、Mo-RepCount，并自建 20K 棒球/高尔夫 Swing 视频、motion 和专家 QA。指标覆盖 MoVid-Bench、BABEL-QA、Mo-RepCount 的 OBO/MAE/OBZ/RMSE 与 Swing 的 accuracy/GPT-4 score。
- **关键发现**：MoVid motion 平均 55.32%、video 52.33%，分别高于 MotionLLM 的 49.50%/49.00%；BABEL-QA 0.458 vs 0.436。Swing overall 为 24.80% / 2.48，MotionLLM 为 16.53% / 1.57。Mo-RepCount OBO 0.389、MAE 0.324、OBZ 0.222、RMSE 6.15，说明统一模型仍未必胜过专门计数器。
- **作者自述限制**：需要进一步提升跨模态融合与计算效率以满足实时应用；冻结上游 encoder 也限制了针对动作细节的适配。
- **对本项目的 know-how**：
  1. raw motion 不必先离散为极少 token；保留高频时间结构；
  2. 可在 Qwen-VL 前加入“文本问题→关键帧/关键关节”选择器；
  3. 做 direct continuous motion 与 VQ-token 两路公平对照；
  4. 计数任务应单列，不能被总体问答分数掩盖。
- **原始来源**：[CVPR 论文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Human_Motion_Instruction_Tuning_CVPR_2025_paper.pdf) · [官方代码](https://github.com/ILGLJ/LLaMo)
- **主图资产**：`assets/figures/motion_llamo.png`；Figure 2，PDF 第 4 页。图意：多模态特征抽取、Cross Talker 与行为生成三模块。

### A3. MoChat: Joints-Grouped Spatio-Temporal Grounding LLM for Multi-Turn Motion Comprehension and Description（2024）

- **定位与区分**：直接面向 skeleton-to-LLM、多轮动作理解和时间定位。相比 MotionLLM 的统一 transformer motion encoder，MoChat 把身体拓扑先验显式写进网络。
- **核心问题**：普通动作编码器容易忽略身体部位关系，并且语言描述缺乏动作发生时间的定位。
- **方法机制**：Joints-Grouped Skeleton Encoder 按解剖部位分组，组合空间与时间注意力；projector 保留序列长度后接 Vicuna-13B。额外 regression head 预测开始/结束时间，用 DIoU 训练。数据包含基本描述、身体部位/左右空间对话，以及由原子动作与关节/轴极值构造的 temporal grounding 对话。
- **数据与评测**：HumanML3D 14,616 motions、44,970 captions，80/5/15 split；22-joint SMPL。空间 gap-fill test 2,574 条。描述用 BLEU/ROUGE/CIDEr/BERTScore/GPT4Score，空间题 exact accuracy，时间定位用 R@1@IoU 0.5/0.7。
- **关键发现**：caption BLEU-1 61.75、BLEU-4 21.60、ROUGE 47.59、CIDEr 51.57、BERT 45.59、GPT-4 5.99；空间准确率 85.70%，GPT-4V 68.02%；MoChat-R 时间定位为 21.89/12.02，TimeChat 2.10/0.40，plain MoChat 19.31/5.58。
- **作者自述限制**：13B 模型资源开销大；7B 或 LoRA 节省显存会导致作者认为不可接受的性能下降。若不混入约 3K Puffin 通用指令，会出现 catastrophic forgetting。自动生成的时间标注还可能继承 GLM-4 与启发式规则错误。
- **对本项目的 know-how**：
  1. motion projector 改为 body-part grouped，而不是一条同质 token 序列；
  2. 把时间段定位作为独立输出头/奖励，不要只让 LLM 生成自由文本；
  3. 保留少量通用指令 rehearsal，防止 SFT 后语言能力遗忘；
  4. 构造 joint × axis × left/right 的 hard negative。
- **原始来源**：[论文](https://arxiv.org/abs/2410.11404) · [PDF](https://arxiv.org/pdf/2410.11404)
- **主图资产**：`assets/figures/motion_mochat.png`；Figure 2，PDF 第 3 页。图意：JGSE、motion projector、LLM 与 temporal regression head 的完整链路。

### A4. LLMs are Good Action Recognizers（2024）

- **定位与区分**：动作识别方法，不是开放式动作问答模型。它证明“让 skeleton token 更像自然语言 token”可提升分类，但不能把该结果外推为 grounded reasoning。
- **核心问题**：如何把 skeleton action 映射为 LLM 可利用的离散 action sentence，并尽量继承预训练语言模型的结构先验。
- **方法机制**：action VQ-VAE 生成离散 token；对 token 分布加入 Zipf 先验、上下文相关性与和 LLaMA word embedding 的 MMD 对齐；以 hyperbolic codebook 表达骨架层级；LoRA 微调 LLaMA-13B。
- **数据与评测**：NTU RGB+D 60（约 56K、60 类）、NTU120（约 114K、120 类）、Toyota Smarthome（16,115、31 类）、UAV-Human（>20K、155 类），以分类准确率为主。
- **关键发现**：NTU60 XSub/XView 95.0/98.4；NTU120 88.7/91.5；Toyota 67.0/36.1/66.6；UAV 46.3。NTU120 XSet 上 no-bias 87.6，去 Zipf 89.9、去 context 89.8、去 MMD 90.3、完整 91.5；continuous 83.4 vs discrete 91.5；Euclidean codebook 89.7 vs hyperbolic 91.5；full-tune 79.6 vs LoRA 91.5。未见类别上 LLM-AR 62.4，full-tune 37.7。
- **限制说明**：论文没有单独的 limitation section。可由任务定义推断：它只验证固定类别分类，依赖人工设计的分布/层级约束，且没有验证 QA、时间证据或多模态 grounded reasoning；这些是“推断限制”，不是作者原话。
- **对本项目的 know-how**：检查 motion token 的频率是否 codebook collapse，比较欧氏/双曲层级表征；优先 LoRA 保护语言先验。但必须保留问答和反事实证据测试，避免“分类强 = 会推理”的错误结论。
- **原始来源**：[论文](https://arxiv.org/abs/2404.00532) · [PDF](https://arxiv.org/pdf/2404.00532)
- **主图资产**：`assets/figures/motion_llmar.png`；Figure 2，PDF 第 5 页。图意：欧氏动作特征投到双曲 codebook 量化，再解码重建。

### A5. Encoder-Free Human Motion Understanding via Structured Motion Descriptions（SMD，2026）

- **定位与区分**：最值得本项目立刻实现的强基线。它完全不用 motion encoder，不学 motion-language projector，而把几何动作确定性地写成结构化文本再喂给通用 LLM。
- **核心问题**：motion encoder 常对数据来源、骨架标准和任务迁移敏感；能否用可解释、无训练的结构化描述替代 encoder。
- **方法机制**：骨架序列转为 global trajectory description，加 26 个生物力学关节角度描述并按时间分段；文本直接输入 LLM，只做 LoRA。默认 Qwen2.5-7B，rank 16、alpha 32、dropout 0.05，约 40M trainable 参数。Top3 描述约 1K token，All26 约 4K token。
- **数据与评测**：BABEL-QA 1,109 motions/2,577 QA（1,800/384/393）；HuMMan-QA 925/3,123（2,066/524/533）；HumanML3D 14,616 motions/44,970 captions。两套 QA 标准化成固定 10 选项，以 exact match 计分；caption 用 R@k/MM-Dist、BLEU/ROUGE/CIDEr/BERT。
- **关键发现**：All26 在 BABEL/HuMMan 为 66.7/90.1，HumanML3D R@1 0.584、R@3 0.883、MM-Dist 2.35、CIDEr 53.16、BERT 45.58。Top3 反而更适合 QA，达 73.3/91.0。Top3 zero-shot 仍有 35.6/31.7（chance 10%）。同一结构化输入可跨 8 个 LLM、6 个家族；论文还显示同一 Qwen backbone 的 MotionGPT3-VAE 在 BABEL 为 50.1、HuMMan 仅 22.0，而 SMD 在 HuMMan 为 90.1，突出 encoder 的跨来源脆弱性。
- **作者自述限制**：All26 约 4K token，是 Top3 的约 4 倍、VAE latent 的约 15 倍；延迟 1.154s，对比同 backbone VAE 约 0.3s。固定 26 角/22-joint SMPL 不含手指；阈值难覆盖极慢或爆发动作；狭窄动作词汇导致 zero-shot NTU 失效；物体/手指主导活动存在先天歧义。
- **对本项目的 know-how**：
  1. 从现有 NPY 生成 SMD，作为第三种可审计输入；
  2. QA 默认 top-K 关节，而 caption 用全量描述，令粒度随任务自适应；
  3. 同时保留 absolute world trajectory 与局部关节角；
  4. 测 `raw encoder / SMD / both / shuffled SMD`；
  5. SMD 文本天然适合解释 rubric：能指出模型依赖了哪个关节、方向与时间段。
- **原始来源**：[论文](https://arxiv.org/abs/2604.21668) · [项目页](https://yaozhang182.github.io/motion-smd/) · [官方代码](https://github.com/yaozhang182/motion-smd)
- **主图资产**：`assets/figures/motion_smd.png`；Figure 2，PDF 第 3 页。图意：skeleton→SMD 的确定性转换，以及 SMD→LoRA LLM 的两阶段视图。

### A6. HumanMoveVQA: Can Video MLLMs Reason about Human Movement in Videos?（2026）

- **定位与区分**：邻近视频 VQA，但其问题直接瞄准“全局轨迹与朝向”，正好补足很多 skeleton QA 只识别局部动作类别的缺陷。
- **核心问题**：现代 video MLLM 能否在第一帧锚定的世界坐标中理解人体位移、旋转、顺序和轨迹可行性，而不只是说出动作名。
- **方法机制**：用 PromptHMR 得到 world-space SMPL-X；MotionScript 生成根位移和 roll/pitch/yaw 的空间 code；BLIP-2 提取外观标签；经人工 QC 后用确定性模板生成 MCQ，并构造“语义合理但几何不一致”的 near-miss distractor。
- **数据与评测**：来自 EMDB、RICH、EgoBody。测试共 97 videos、10,203 MCQ：EMDB 786、RICH 2,220、EgoBody 7,197。七类题为 existence、comparative、dominant、numerical、ordering、temporal、trajectory affordance。
- **关键发现**：EMDB normalized score：MotionLLM -9.53，base Qwen3-VL-8B 12.83，针对性 SFT 后 Qwen3-VL-8B 37.88；提升主要来自 numerical +40.9pp、trajectory +22.6、ordering +17.0。人工子集为 78.71，仍留下大间隙。跨数据集联合训练最好，EgoBody 数据的泛化最强。
- **作者自述限制**：tracking noise 会伤害细微旋转和短事件；暂不覆盖多人交互；Ordering 仍然困难。
- **对本项目的 know-how**：训练题中增加世界坐标距离/旋转/顺序/轨迹题；distractor 保持动词不变，只改几何和计数；把 camera motion robustness 单列；用同一问题比较 V、M、VM 三种证据来源。
- **原始来源**：[论文](https://arxiv.org/abs/2606.27999) · [PDF](https://arxiv.org/pdf/2606.27999)
- **主图资产**：`assets/figures/motion_humanmovevqa.png`；Figure 2，PDF 第 5 页。图意：世界坐标 SMPL-X、空间 code、外观标签、人工 QC 与 MCQ 生成流水线。

### A7. NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models（2026）

- **定位与区分**：不是新模型，而是评测设计与 VLM-as-judge 可靠性研究；对本项目的 Rubric/evaluation script 最直接。
- **核心问题**：现有动作问答是否足够细粒度，以及 VLM judge 是否真的能可靠评价身体部位、方向和复杂动作描述。
- **方法机制**：3 tasks × 3 semantics × 3 difficulty：T1 multi-select QA、T2 caption、T3 correction；语义轴是 body part、direction、action；难度 easy/medium/hard。构造时先由 Qwen3.6-Plus 基于 metadata 起草，再看视频修订，最后三位专家一致 accept/revise/reject。
- **数据与评测**：1,307 expert-verified instances、992 unique SMPL-H clips，来自 AMASS 的 16 个子集并用 BABEL/HumanML3D metadata。T1 511、T2 396、T3 400。T1 用 exact/Jaccard/precision；T2 用分难度 judge；T3 评 Identify/token recall/semantic correction。
- **关键发现**：整体 Gemini-3.1-Flash 58.44、Qwen3.6-Plus 54.85、Qwen3.5-27B 49.75；方向是普遍弱项，caption 是所有系统瓶颈。更关键的是 judge：Gemini judge 的 coarse V1 κ=0.701，但 body-part 细粒度 V2 κ=0.346、V3 κ=0.104；V3 的 system-level correlation 为 -0.146，说明“更细的自动 rubric”甚至可能把系统排反。
- **作者自述限制**：AMASS 分布缺少双人、物体和非刚体交互；single-view judge 看不到背面，未来应多视角；专家 QC 把规模限制在约 1K。
- **对本项目的 know-how**：
  1. 不把单个 LLM judge 当主真值；优先 exact、关节几何规则与抽样人工复核；
  2. 引入 multi-select 暴露“部分看懂”；
  3. 增加 T3 错误识别+修正，比只选 A/B/C/D 更能诊断；
  4. 报告 task × semantic × difficulty 三维矩阵，不只报告均值。
- **原始来源**：[论文](https://arxiv.org/abs/2606.04773) · [项目页](https://nextmotionqa.github.io/)
- **主图资产**：`assets/figures/motion_nextmotionqa.png`；Figure 2，PDF 第 4 页。图意：benchmark 三类任务与 VLM judge 评测流程。

### A8. Ego-Human Motion Prediction with 3D-Aware LLM / Ego3DLM（2026）

- **定位与区分**：任务是 egocentric 3D motion tracking/prediction，不是本项目的 MCQ；但它的 SFT→GRPO 与跨模态 reward 设计，是最可复用的后训练参考。
- **核心问题**：如何让 LLM 同时输出场景推理、过去/未来人体姿态与文字描述，并让动作和语言相互一致。
- **方法机制**：三阶段：（1）3D scene + ego video 上做 semantic/spatial scene QA 预训练；（2）SFT 输出结构化的 `[scene reasoning, past pose, future pose, past narration, future description]`；（3）GRPO 同时使用 motion JPE reward、text BLEU-4 reward、cross-modal matching reward 与 format reward。匹配 reward 惩罚 GT text↔pred motion、pred text↔GT motion、pred text↔pred motion 三类距离。G=6，`w_motion=1, w_text=.8, w_match=.02`，epsilon=.2，beta=.001；格式错误可从 0 罚到 -3。
- **数据与评测**：Nymeria，5 秒片段、10 fps、按 scene split；另构造 535K spatial QA + 115K semantic QA，覆盖 208 scenes。动作以 APE/JPE/ADE/FDE/FID，文字以 BLEU 等，跨模态以 feature distance。
- **关键发现**：相对 UniEgoMotion，prediction JPE 364.5，降低 14.1%；tracking APE 96.4，降低 36.7%。future/past BLEU-4 为 0.1039/0.1107，cross-modal distance 4.2571。GRPO 消融显示：motion reward 改善 JPE 和 FID，text reward 提高 BLEU，而 matching reward 可同时改善动作、文字与对齐。
- **作者自述限制**：需要预计算 3D 场景，这在未建图的真实场景中成本很高。
- **对本项目的 know-how**：把当前分支级 `VM > V bonus` 改成样本级证据 reward：`正确答案 + 格式 + motion-text/answer 一致性 + motion shuffle 后置信度下降`；分别奖励各模态正确和两模态一致，避免模型仅学到“VM 分支标签”。
- **原始来源**：[论文](https://arxiv.org/abs/2607.07001) · [项目页](https://jaewoo97.github.io/Ego3DLM/)
- **主图资产**：`assets/figures/motion_ego3dlm.png`；Figure 2，PDF 第 5 页。图意：3D-aware 预训练、结构化 SFT 与多奖励 GRPO 三阶段。

---

## B. 上游表征、生成与数据底座

### B1. MotionGPT: Human Motion as a Foreign Language（2023）

- **定位与区分**：开创性的“motion as language”统一生成/理解框架；对本项目主要提供 tokenizer 和任务混训思路，不是直接的视频+motion VQA 竞品。
- **核心问题**：能否把连续人体动作 token 化，使同一个 T5 在生成、描述、预测、补全等多种 motion-language 任务间迁移。
- **方法机制**：先训练 VQ-VAE motion tokenizer（512×512 codebook，时间下采样 4）；再以 supervised + masked unsupervised motion-language objective 预训练；最后以 15 个核心任务、1000+ prompt 做 instruction tuning。
- **数据与评测**：HumanML3D 14,616 motions/44,970 captions；KIT-ML 3,911/6,353。生成看 FID/Diversity/MultiModality/R-precision，描述看 BLEU/ROUGE/CIDEr/BERT，补全看 ADE/FDE。
- **关键发现**：HumanML3D 统一模型 text-to-motion Top-1 0.492、FID 0.232；caption R@3 0.827、BLEU-4 12.47、CIDEr 29.2；motion prediction FID 0.905（MDM 6.031），in-between FID 0.214（MDM 2.698）。
- **作者自述限制**：只覆盖 articulated body，不含脸、手、动物；缺少人—物、环境和多人互动；只有约 15K motions，模型变大并不总变好；有限配对文本下 instruction tuning 可能损害文本生成。
- **对本项目的 know-how**：沿用“先 tokenizer/alignment、再 instruction”的阶段隔离，并用统一 prompt taxonomy 管理任务；但新评测必须证明 token 真正保留方向、时间与细微关节信息。
- **原始来源**：[论文](https://arxiv.org/abs/2306.14795) · [官方代码](https://github.com/OpenMotionLab/MotionGPT)
- **主图资产**：`assets/figures/motion_motiongpt.png`；Figure 2，PDF 第 4 页。图意：motion tokenizer、motion vocabulary 与 T5 motion-language model。

### B2. MotionGPT-2: A General-Purpose Motion-Language Model for Motion Generation and Understanding（2024）

- **定位与区分**：MotionGPT 的 whole-body/SMPL-X 延伸。最大差异是 part-aware VQ-VAE，把 body 与 hand 分开建模。
- **核心问题**：单一 codebook 容易牺牲手部细节；如何用统一词表支持 text/pose 条件、whole-body 生成与理解。
- **方法机制**：冻结 LLaMA-3.1-8B 主体，用约 1% 额外 LoRA；Part-Aware VQ-VAE 有 body/hand 两个 encoder 与 codebook；再做 motion tokenization → motion-language alignment → instruction tuning 三阶段。支持生成、caption、prediction、in-between。
- **数据与评测**：HumanML3D、KIT-ML 与 Motion-X；后者标准化为 52-joint body+hand SMPL-X。沿用生成/检索与 caption 指标，并进行手部用户研究。
- **关键发现**：HumanML3D 上 LLaMA3.1-8B 版本 R@3 0.782、FID 0.191；相对论文复现的上一代 MotionGPT，R@3 提高约 10.4%，FID 降 0.254。Motion-X 上 PA-VQVAE vs vanilla：Top-1 0.349 vs 0.332，FID 0.619 vs 0.666。用户研究 81.5% 偏好 PA-VQVAE 手部动作，但 fine detail 均分仅 3.16。部分补全指标未全面超过旧模型，说明升级并非所有任务单调受益。
- **限制说明**：论文没有清晰独立的 limitation section。可合理推断但需标记为推断：VQ 仍有量化损失；缺少场景/多人/物体 grounding；生成和小规模用户偏好不能证明细粒度问答能力。
- **对本项目的 know-how**：若输入含 SMPL-X 52/55 joints，身体和手应分 projector/codebook；训练与评测必须分别报告 body 与 hand，而不是让大量身体关节掩盖手部失败。
- **原始来源**：[论文](https://arxiv.org/abs/2410.21747) · [PDF](https://arxiv.org/pdf/2410.21747)
- **主图资产**：`assets/figures/motion_motiongpt2.png`；Figure 2，PDF 第 4 页。图意：part-aware motion tokenizer、统一多模态 vocabulary、alignment 与 instruction tuning。

### B3. MotionGPT3: Human Motion as a Second Modality（2025）

- **定位与区分**：与前两代最本质的区别不是规模，而是放弃 VQ 离散化，采用连续 sequence-level VAE，并把文本/动作分为双流，减少模态干扰。
- **核心问题**：离散 VQ 会损伤动作保真度；把文本和动作硬塞入统一自回归 token 流又会造成 cross-modal interference。
- **方法机制**：连续 sequence-level VAE latent；text/motion 双流 Transformer 共享 attention；在自回归 backbone 中用 diffusion head 生成动作。三阶段为 text-to-motion pretrain 100K steps、cross-modal alignment 300K、joint finetune 50K。
- **数据与评测**：HumanML3D，统一评估 text-to-motion 与 motion caption；生成用 R-precision/FID/MM-Dist，描述用检索、BLEU/BERT 等。
- **关键发现**：统一 text-to-motion R@1 0.553、R@3 0.837、FID 0.208、MM-Dist 2.725；caption R@1 0.573、R@3 0.864、MM-Dist 2.426、BLEU-4 19.412、BERT 35.231。消融中 `Bimodal + VAE` 明显胜过 `Unified + VQ`，训练约快 2×，验证最高约快 4×。
- **作者自述限制**：左右/方向仍会失败；每段序列只有一个 VAE latent，不利于长动作的局部片段组合与对齐；OOD 文本泛化受数据限制。
- **对本项目的 know-how**：建立 continuous motion adapter baseline；比较 single-stream 与 dual-stream；先单独学 motion 表征，再 cross-modal align，最后 joint finetune。对长动作应使用 segment latent，而不是整段压成一个向量。
- **原始来源**：[论文](https://arxiv.org/abs/2506.24086) · [官方代码](https://github.com/OpenMotionLab/MotionGPT3)
- **主图资产**：`assets/figures/motion_motiongpt3.png`；Figure 2，PDF 第 5 页。图意：连续动作 latent、双流共享注意力、三阶段训练及生成/理解两种推理路径。

### B4. Motion-X（2023）→ Motion-X++（2025）数据谱系

- **定位与区分**：不是一个模型，而是本项目可用的 RGB—SMPL-X—文本上游数据底座。Motion-X++ 不是简单加量，而是补视频/音频、手脸精度、镜头切分与全球轨迹。
- **核心问题**：现有 motion-text 数据规模小、身体不完整、模态不齐，难以训练 whole-body 多模态模型。
- **数据与机制**：Motion-X 原版约 81.1K sequences、15.6M SMPL-X frames、144.2 hours，并含 81.1K semantic labels 与 frame-level pose text。Motion-X++ 扩展到 120.5K sequences、19.5M whole-body poses、80.8K RGB videos、45.3K audio、19.5M frame-level pose descriptions、120.5K sequence labels；用更强 shot detection、whole-body keypoint hierarchy、相机/全局轨迹优化和 GPT-4V caption。
- **评测与发现**：论文不仅比较 annotation pipeline，还在 text-to-motion、music-to-dance、mesh recovery、2D whole-body pose 上验证。其分析指出 Motion-X++ 的真实 motion diversity 13.174，高于 HumanML3D 的 9.837；加入 frame-level pose descriptions 可使相关 text-to-motion 设定的 FID 降约 38%。这些是上游数据效用，不是 VQA 准确率。
- **作者自述限制**：markerless pipeline 的动作质量低于 multi-view marker-based 系统；现有自动指标常与视觉结果不一致，需要改进。
- **对本项目的 know-how**：
  1. 构造 V/VM paired 样本必须来自同一同步 clip，不要“相似动作伪配对”；
  2. 每条保留 source、performer、capture type、tracking confidence、frame validity；
  3. 按 performer/source/action group split，防止视频外观与动作重复泄漏；
  4. 低置信手脸和遮挡片段不应生成精细手部题；
  5. 可把 frame-level pose description 再转 SMD，形成可解释 teacher signal。
- **原始来源**：[Motion-X 论文](https://arxiv.org/abs/2307.00818) · [Motion-X++ 论文](https://arxiv.org/abs/2501.05098)
- **主图资产**：`assets/figures/motion_motionxpp.png`；Figure 1，PDF 第 2 页。图意：Motion-X 与 Motion-X++ 在动作精度、模态、下游任务和文本描述上的升级对比。

### B5. MotionCLIP: Exposing Human Motion Generation to CLIP Space（2022）

- **定位与区分**：早期 motion-language 表征工作，不使用 LLM；价值在于把动作 latent 对齐到预训练 CLIP 的语义与图像空间。
- **核心问题**：稀少的 motion-text 数据能否借助 CLIP 的大规模语义空间获得开放词汇生成、编辑、插值与识别能力。
- **方法机制**：Transformer motion autoencoder 同时做动作重建，并把 motion latent 对齐到冻结 CLIP 的 text embedding；另把动作渲染为图像，与 CLIP image embedding 对齐。图像对齐为稀疏文本标签补充细粒度信号。
- **数据与评测**：BABEL/AMASS 用于动作与 60 类 action recognition；KIT motion-language 支持 text-to-motion；另以 HumanAct12、UESTC 等动作类别与用户研究验证生成/风格。它同时展示 OOD prompt、latent editing 与 interpolation。
- **关键发现**：BABEL-60 action recognition Top-1/Top-5 为 40.9%/57.71%，接近专用 2s-AGCN 的 41.14%/73.18%；去 image loss 降至 35.05%/50.26%，去 text loss 降至 4.54%/18.37%，说明文字语义是核心，图像对齐提供额外增益。
- **作者自述限制**：难理解 left/right/counter-clockwise 等方向；部分风格如 heavy/proud 表达失败；域外文化动作不稳定，例如特定球星庆祝或 Superman 标志姿势。
- **对本项目的 know-how**：可加 motion↔video/text 的 contrastive auxiliary loss，提高粗语义对齐；但方向、旋转、时间顺序必须由几何/时序任务单独监督，不能指望 CLIP 空间自动学会。
- **原始来源**：[论文](https://arxiv.org/abs/2203.08063) · [项目页](https://guytevet.github.io/motionclip-page/) · [官方代码](https://github.com/guytevet/motionclip)
- **主图资产**：`assets/figures/motion_motionclip.png`；Figure 2，PDF 第 2 页。图意：motion autoencoder 重建，同时对齐 CLIP text 与 rendered-image embedding。

### B6. T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations（CVPR 2023）

- **定位与区分**：纯 text-to-motion 生成基线，不做通用 LLM 问答；其贡献是稳定离散 tokenizer 与自回归训练细节。
- **核心问题**：如何让 VQ-VAE codebook 不坍缩，并缓解 teacher forcing 与推理时自回归分布不一致。
- **方法机制**：CNN VQ-VAE 把动作量化为 code indices，使用 EMA 更新和 code reset 维持 codebook 利用率；T2M-GPT 以 CLIP text 条件自回归生成 token，并在训练时以概率扰动/替换输入 token，减少 exposure bias，最后输出 End token。
- **数据与评测**：HumanML3D、KIT-ML；标准指标 R-Precision、FID、MM-Dist、Diversity、Multimodality，重复评测 20 次并给 95% CI。
- **关键发现**：HumanML3D 上 `tau=0.5` 的 R@1/2/3 为 0.491/0.680/0.775，FID 0.116、MM-Dist 3.118；MotionDiffuse 为 0.491/0.681/0.782、FID 0.630。也就是说它的文本一致性与强基线相近，但生成分布 FID 明显更好。量化消融显示 code reset 与 EMA 对重建和生成都重要。
- **作者自述限制**：过长文本会漏掉部分细节；部分动作的手脚有轻微抖动，作者认为与 VQ-VAE 有关，并建议更好的结构或后处理平滑。
- **对本项目的 know-how**：如果继续 VQ motion tokens，必须监控 code usage/perplexity、启用 EMA/code reset，并做输入扰动鲁棒训练；但不要把低 FID 当作问答理解提升，仍需 MCQ、时序定位与反事实 reliance 测试。
- **原始来源**：[CVPR 论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html) · [项目页](https://mael-zys.github.io/T2M-GPT/) · [官方代码](https://github.com/Mael-zys/T2M-GPT)
- **主图资产**：`assets/figures/motion_t2mgpt.png`；Figure 2，PDF 第 3 页。图意：Motion VQ-VAE 与由 CLIP text 条件驱动的 causal T2M-GPT。

---

## 可直接落地的实验矩阵

### 1. 输入表征消融

| 实验 | Video | Raw motion | SMD text | 目的 |
|---|---:|---:|---:|---|
| V | ✓ |  |  | 现有视频基线 |
| M-raw |  | ✓ |  | 测 motion encoder 本体 |
| M-SMD |  |  | ✓ | 测无 encoder、可解释基线 |
| VM-raw | ✓ | ✓ |  | 现有 VM |
| VM-SMD | ✓ |  | ✓ | 测结构化 motion 是否更稳 |
| VM-both | ✓ | ✓ | ✓ | 上限与互补性 |
| VM-shuffle-M | ✓ | 错配 |  | 测模型是否真用 raw motion |
| VM-shuffle-SMD | ✓ |  | 错配 | 测模型是否真用 SMD |

### 2. Reward 建议

建议每个样本记录以下独立 reward，不先合成一个不可解释总分：

- `r_answer`：选项正确，exact；
- `r_format`：严格答案格式；
- `r_motion_evidence`：正确 motion 相对 shuffled motion 的答案 log-prob margin；
- `r_video_evidence`：正确 video 相对 shuffled video 的 margin；
- `r_crossmodal`：motion-derived description 与最终 rationale/answer 的一致性；
- `r_hallucination`：题目不存在所述动作时的拒答/否定正确；
- `r_temporal`：预测时间段 IoU 或排序正确；
- `r_geometry`：左右、朝向、根位移、角度/计数的确定性规则分。

其中 `r_crossmodal` 可借鉴 Ego3DLM，但本项目最好使用 motion encoder/SMD 与 answer embedding 的匹配，加上 deterministic geometry check；不要完全依赖 LLM judge。

### 3. 评测切片

主结果至少同时给出：

- 模态：V / M / VM；
- 证据轴：body-part / direction / count / order / global trajectory / temporal grounding / reasoning / hallucination；
- 难度：easy / medium / hard；
- 数据来源：source dataset、capture type、camera motion、occlusion；
- 可靠性：exact、geometry verified、human-audited、LLM-judge-only；
- reliance：正确模态与 shuffled/zeroed 模态的概率差。

## 网页呈现建议

每篇论文卡片应明确显示：`类别`、`与项目距离`、`核心机制`、`硬结果`、`作者自述限制`、`我的推断`、`可执行 know-how`、`原文/代码`。不要把作者限制和本报告推断混写。默认首页可按下面三条阅读路线过滤：

1. **我要修 V/VM 融合**：MotionLLM → LLaMo → MotionGPT3 → Ego3DLM；
2. **我要改评测和 Rubric**：HumanMoveVQA → NextMotionQA → MoChat；
3. **我要改 motion 表征**：SMD → MotionGPT-2 → LLM-AR → MotionCLIP/T2M-GPT。

## 资产清单

| slug | 图片 | 图号 / PDF 页 | 用途 |
|---|---|---|---|
| motionllm | `assets/figures/motion_motionllm.png` | Fig. 2 / p5 | 双分支 + 两阶段训练 |
| llamo | `assets/figures/motion_llamo.png` | Fig. 2 / p4 | Cross Talker 框架 |
| mochat | `assets/figures/motion_mochat.png` | Fig. 2 / p3 | JGSE + temporal head |
| llmar | `assets/figures/motion_llmar.png` | Fig. 2 / p5 | 双曲 VQ-VAE |
| smd | `assets/figures/motion_smd.png` | Fig. 2 / p3 | skeleton→SMD→LLM |
| humanmovevqa | `assets/figures/motion_humanmovevqa.png` | Fig. 2 / p5 | world-space VQA 数据流水线 |
| nextmotionqa | `assets/figures/motion_nextmotionqa.png` | Fig. 2 / p4 | benchmark + judge |
| ego3dlm | `assets/figures/motion_ego3dlm.png` | Fig. 2 / p5 | SFT + GRPO |
| motiongpt | `assets/figures/motion_motiongpt.png` | Fig. 2 / p4 | motion tokenizer + T5 |
| motiongpt2 | `assets/figures/motion_motiongpt2.png` | Fig. 2 / p4 | part-aware tokenizer |
| motiongpt3 | `assets/figures/motion_motiongpt3.png` | Fig. 2 / p5 | continuous dual-stream |
| motionxpp | `assets/figures/motion_motionxpp.png` | Fig. 1 / p2 | X→X++ 数据谱系 |
| motionclip | `assets/figures/motion_motionclip.png` | Fig. 2 / p2 | CLIP 对齐 |
| t2mgpt | `assets/figures/motion_t2mgpt.png` | Fig. 2 / p3 | VQ-VAE + causal GPT |

## 证据边界

- 表中数字均为各论文自己的实验口径，**不能跨论文直接排名**；数据 split、输入模态、judge 和 metric 均不同。
- 2026 年论文（SMD、HumanMoveVQA、NextMotionQA、Ego3DLM）在当前日期可用于前沿调研，但应在网页中标为 preprint，避免和已正式发表工作混淆。
- MotionGPT 系列、MotionCLIP、T2M-GPT 的 FID/R-precision 是生成或检索指标，不等价于 MotionLLM 选择题准确率。
- 主图仅用于论文解读与内部研究展示；网页应保留论文标题、图号和原始链接，不应暗示图像由本项目原创。
