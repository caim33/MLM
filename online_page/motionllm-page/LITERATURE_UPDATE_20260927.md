# 2026-09-27 文献更新

本次核查只纳入 2026-09-27 及以前可由作者原文或官方项目页确认的工作。网页中的方法、实验和适用边界分别写在 `papers.js` 的卡片里；针对本项目的实验建议是迁移判断，尚非已验证收益。论文报告值不属于 MotionLLM 项目的新评测结果。

| 方向 | 论文 | 首发日期 | 本项目重点 |
| --- | --- | --- | --- |
| Motion | [Open-UniMo](https://arxiv.org/abs/2609.14615) | 2026-09-13 | 统一动作 token、motion-grounded CoT、GRPO 和往返一致性 |
| Motion | [FineMoLA](https://arxiv.org/abs/2608.01392) | 2026-08-02 | 无逐帧标注的帧—动作短语对齐 |
| Motion | [GMoT](https://arxiv.org/abs/2607.16322) | 2026-07-15 | 视频局部运动 token 与身体区域证据 |
| Motion | [MotionMERGE](https://arxiv.org/abs/2605.18956) | 2026-05-18 | 粗细粒度、身体部位和时间片段监督 |
| Motion | [LLaMo, continuous tokens](https://arxiv.org/abs/2602.12370) | 2026-02-12 | 连续动作表示与语言能力保留；不同于站内 CVPR 2025 同名论文 |
| Recipe | [LLaVA-OneVision-2](https://arxiv.org/abs/2605.25979) | 2026-05-25 | 短到长视频课程、密集时空监督 |
| Recipe | [OpenVLThinkerV2](https://arxiv.org/abs/2604.08539) | 2026-04-09 | 多任务奖励分布、回复长度与熵控制 |
| Recipe | [Video-R2](https://arxiv.org/abs/2511.23478) | 2025-11-28 | 时间戳 SFT、事件对齐奖励、推理答案一致性 |
| Recipe | [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) | 2025-11-26 | merger 预热、联合预训练、显式视频时间戳、后训练 |
| Recipe | [InternVL3.5](https://arxiv.org/abs/2508.18265) | 2025-08-25 | 广覆盖 SFT、离线 MPO 再在线 GSPO、按 rollout 筛难度 |

阅读顺序建议：先读 Qwen3-VL 和 LLaVA-OneVision-2 的感知课程，再读 FineMoLA / GMoT 的局部证据，之后读 Video-R2 / Open-UniMo 的奖励设计。OpenVLThinkerV2 与 InternVL3.5 用于检查多任务 RL 的稳定性和数据选择。所有实验建议均需同数据、同预算和当前批次 fresh artifact 才能解释为项目收益；正式评估还须遵守全 registry finetune 阶段屏障。

新条目只保存本站原创 SVG 机制图，图注均标为“非论文原图”。旧 PDF 资产与旧报告的出处仍以原卡片为准。
