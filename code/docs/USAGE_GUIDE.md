# MotionLLM / Qwen Clean Codebase 使用手册

适用目录：

```text
/wangbenyou-sulongjie/caimeng/qwen-codebase-clean
```

这份手册把“能检查”“能推理”“能训练”和“能发布”分开说明。命令中出现的
`/absolute/...` 必须替换成服务器上的真实绝对路径；活动代码不会猜测个人目录。

## 1. 环境

### CPU 开发与结构检查

```bash
cd /wangbenyou-sulongjie/caimeng/qwen-codebase-clean
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -c "import motionllm, motion_eval"
python -m motion_eval --help
python scripts/run_checks.py
```

`scripts/run_checks.py` 会生成 `__pycache__`。精简 Ubuntu 若缺少
`python3-venv/ensurepip`，应由管理员补系统组件或使用已有冻结环境，不要在项目
脚本中自动执行 apt。

### Qwen / CUDA 环境

先确认服务器的 CUDA、驱动和 PyTorch 版本，再安装重依赖：

```bash
python -m pip install -r requirements-sft.txt
```

不要把本地 `.venv` 复制到服务器；Python、CUDA 和动态库必须在目标机器重建。

当前交付服务器没有 `python3-venv/ensurepip`，因此已准备不覆盖系统 Torch 的
独立依赖目录。每次登录后先执行：

```bash
export MOTIONLLM_ROOT=/wangbenyou-sulongjie/caimeng/qwen-codebase-clean
export MOTIONLLM_RUNTIME=/wangbenyou-sulongjie/caimeng/runtime/qwen-codebase-clean-py310
export PYTHONPATH="$MOTIONLLM_RUNTIME:$MOTIONLLM_ROOT/src:$MOTIONLLM_ROOT"
cd "$MOTIONLLM_ROOT"

python3 -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
python3 -c "import av; print(av.__version__)"
python3 -m motion_eval --help
```

当前服务器运行目录已验证 Transformers 4.57.3、PEFT 0.18.0、Accelerate
1.14.0 和 PyAV 12.3.0。视频后端缺少 torchcodec 时会回退到 PyAV +
torchvision；PyAV 不应漏装。

离线 wheel 归档保存在
`/wangbenyou-sulongjie/caimeng/handoff/qwen_py310_wheelhouse_complete_20260830.tar.gz`。
它不包含 Torch；若换机器，必须先核对该机器自己的 CUDA/Torch，再决定是否复用。

## 2. 数据配置

交付服务器的统一数据入口为：

```text
/wangbenyou-sulongjie/caimeng/dataset
```

Motion-X、HumanML3D、SONIC 和历史 Qwen QA 的目录、兼容旧路径及迁移 receipt 见
`/wangbenyou-sulongjie/caimeng/dataset/README.md`。Qwen QA 默认选择：

```text
/wangbenyou-sulongjie/caimeng/dataset/qwen_qa/views/recommended
```

其中 strict keepbench 为 813 个 train group、86 个 validation group，并配套固定
QA500 benchmark。`views/extended_qtext` 为 1768/86，允许同一视频使用不同问题；
两者不能在未记录清理策略的情况下混用。详细说明见 dataset 根目录的
`QWEN_QA_GUIDE.md`。对应的 `motionx_374` 与生成/筛选成功媒体统一位于
`dataset/qwen_qa/media/`，原 Motion-r1/output_mcq 绝对路径继续作为兼容链接。

原 `/wangbenyou-sulongjie/qwen-vl-finetune/data` 现为兼容软链接。历史 JSONL 中
使用 `data/benchmark/...` 的相对路径仍可由旧代码解析；新实验必须在配置中记录
annotation 文件、数据策略、branch、媒体 root 和文件 SHA-256。

正式数据别名写在 `configs/datasets/<alias>.dataset.json`。机器专用配置不想提交时，
把同样格式的文件放到独立目录，并设置：

```bash
export MOTIONLLM_DATASET_CONFIG_DIR=/absolute/private-dataset-configs
```

Motion + Video 配置示例：

```json
{
  "schema_version": 1,
  "name": "motionx_train",
  "annotation_path": "/absolute/data/train.jsonl",
  "media_root": "/absolute/data",
  "split": "train",
  "motion_mean_path": "/absolute/stats/Mean.npy",
  "motion_std_path": "/absolute/stats/Std.npy",
  "expected_motion_dim": 263
}
```

约束：

- JSON/JSONL 必须是 UTF-8，重复键、NaN、空行和非法媒体路径会直接报错。
- `Mean.npy` 与 `Std.npy` 必须成对显式提供；代码不会从包目录自动寻找。
- 错误样本只会重试原 index，不会偷偷换成下一条样本。
- `sample_id`、`group_id`、branch 和 motion ownership 在拼批后仍保留。

## 3. 推理

### Motion + Video

先看参数，不加载 Torch 或模型：

```bash
python qwenvl/infer/infer_qwen3_vl_motion.py --help
```

单样本验证：

```bash
python qwenvl/infer/infer_qwen3_vl_motion.py \
  --model_name_or_path /absolute/model-or-checkpoint \
  --checkpoint_path /absolute/checkpoint \
  --vqvae_path /absolute/vqvae.pth \
  --test_data_path /absolute/test.jsonl \
  --data_path /absolute/media-root \
  --motion_mean_path /absolute/stats/Mean.npy \
  --motion_std_path /absolute/stats/Std.npy \
  --output_path /absolute/output/predictions.jsonl \
  --batch_size 1 \
  --num_samples 1 \
  --max_new_tokens 512 \
  --device cuda
```

脚本会从模型配置绑定真实 `motion_placeholder_token_id`，从模型合同取得 motion
维度，并用同一组显式 normalization 资产校验数据和模型。
`--output_path` 会创建父目录并覆盖同名输出文件，运行前先确认目标路径。

历史 qa374 checkpoint 使用 `motion_placeholder_token_id=160001`。这是模型合同中的
外部 sentinel，故意位于 tokenizer 与 embedding 范围之外；processor 会先用它标记
motion 位置，模型再在文本 embedding 前替换为 motion feature。不要把它强行加入词表。

### 已验证的历史 checkpoint 单样本 smoke

兼容 overlay 位于：

```text
/wangbenyou-sulongjie/caimeng/runtime/checkpoint-overlays/qa374_sft_step3best_checkpoint-48_merged_full
```

它只保存 Transformers 4.57.3 所需的配置/chat-template 兼容信息，权重与 tokenizer
仍链接原 checkpoint，原目录未修改。可复跑命令、输出 SHA 和已知警告见
`docs/GPU_SMOKE_20260830.md`。本次结果为：进程返回 0，目标 `D`、预测 `A`；这只是
功能 smoke，不是精度结论。

### Video-only Qwen3-VL

```bash
python qwenvl/infer/infer_qwen3_vl_4b_thinking_video_only.py \
  --model_path /absolute/local-model \
  --input_jsonl /absolute/input.jsonl \
  --data_root /absolute/video-root \
  --output_path /absolute/output.jsonl \
  --num_samples 1 \
  --max_new_tokens 256 \
  --device cuda
```

这个入口同样严格读取输入；缺文件或坏行不会被静默跳过。
`--output_path` 会写入/覆盖目标 JSONL。

## 4. Full / LoRA SFT

参数入口：

```bash
python qwenvl/train/full_sft.py --help
python qwenvl/train/lora_sft.py --help
bash scripts/full_sft.sh --help
bash scripts/lora_sft.sh --help
```

Motion 训练必须同时满足：

- 数据 alias 中配置 normalization mean/std；
- CLI 传入 `--motion_normalization_mean_path` 和
  `--motion_normalization_std_path`；
- 模型 token 初始化后，训练器会把
  `model.config.motion_placeholder_token_id` 显式绑定到 data adapter；
- VQ-VAE 路径、模型路径、训练与验证 split 均为明确资产。

`scripts/full_sft.sh` 和 `scripts/lora_sft.sh` 的 formal 发布启动目前会在训练前以
exit `78` 停止。这不是普通 SFT 代码故障，而是 formal provenance 尚缺
external-HMAC 绑定的 pre-spawn snapshot 与内存 worker bundle。不要删除 gate。
开发调试可以直接使用 Python 入口，但其产物不能冒充 formal release。

## 5. Motion-R1 GRPO

正式 GRPO 使用冻结的绝对 YAML 配置：

```bash
export MOTION_GRPO_PYTHON=/absolute/grpo-env/bin/python
FORMAL_CONFIG=/absolute/batch/motionr1_vm_lora_grpo.yaml

bash scripts/train_grpo_ms_swift.sh --config "$FORMAL_CONFIG" --dry_run
bash scripts/train_grpo_ms_swift.sh --config "$FORMAL_CONFIG" --preflight_only
```

- `--dry_run`：只检查配置、数据、hash、provenance 和输出目的地。
- `--preflight_only`：再检查依赖版本、Swift/PEFT API、CUDA 与绑定解释器。
- 不带模式参数才会训练并生成 reload/training receipt；执行前必须先通过前两层。

## 6. Rubric RL

先用帮助命令确认对应任务的输入格式：

```bash
python -m rubric_rl.extract_qa_mc_criteria --help
python -m rubric_rl.judge_qa_mc --help
python -m rubric_rl.prepare_cot_gt_v2 --help
python -m rubric_rl.extract_motion_criteria_v2 --help
python -m rubric_rl.judge_motion_caption_v2 --help
```

这些任务不是 dry-run：实际执行会加载 judge 模型、占用 GPU/CPU 并写结果。
中断产物使用 `.partial`，恢复时使用入口支持的 `--resume`，不要手工拼接 JSONL。

## 7. 统一评估控制器

```bash
python -m motion_eval --help
python -m motion_eval registry validate --dry-run
python -m motion_eval gpu status --dry-run
python -m motion_eval batch create BATCH_ID \
  --workspace-root /absolute/batches \
  --pretrained-root /absolute/pretrained-assets \
  --input train=/absolute/train.jsonl \
  --input validation=/absolute/validation.jsonl \
  --input benchmark=/absolute/benchmark.jsonl \
  --input media_manifest=/absolute/media-manifest.json \
  --input derivation_code=/absolute/derivation-code \
  --input leakage_audit=/absolute/leakage-audit.json \
  --config /absolute/controller-config.json \
  --dry-run
```

正式顺序是：冻结输入 → fresh finetune/明确 blocked 证据 → 1/8/32/500
评测 → release verify。invalid、OOM、timeout 和 runtime error 都留在固定分母里。
控制器 production backend 仍有显式 gate；`--help`、schema、dry-run 或 smoke 成功
不等于 15 模型 production 已验证。

每个阶段先查看精确子命令，再执行写操作：

```bash
python -m motion_eval batch create --help
python -m motion_eval batch validate --help
python -m motion_eval finetune --help
python -m motion_eval gate --help
python -m motion_eval eval --help
python -m motion_eval release --help
```

- `batch create` 不带 `--dry-run` 会创建不可变批次目录与 receipt。
- `finetune attempt/run-attempt/complete/block` 追加证据，不应手工改写状态文件。
- `gate open-eval/open-full` 只有前置 barrier 全部满足才写 gate。
- `eval smoke/full` 验证固定分母 prediction；不会替失败样本补成功结果。
- `release build` 写 manifest/table，`release verify` 重新计算并只读核验。
- `plan BATCH_ID` 只适用于已经创建并可验证的批次，不负责创建批次。

## 8. 测试与修改后的检查

```bash
python -m pytest tests/unit tests/contract -q
python -m pytest tests/integration -q
python -m pytest tests/stress -q
python scripts/run_checks.py
```

只改数据层时，至少运行：

```bash
python -m pytest \
  tests/unit/test_data_*.py \
  tests/unit/test_collator_legacy_interface.py \
  tests/contract/test_clean_architecture.py -q
```

跳过（skipped）不是通过。涉及 Torch/Qwen 的用例必须在安装重依赖的环境复跑；
涉及 CUDA、真实权重和真实媒体的流程必须另记 GPU 运行证据。

## 9. 常见问题

- `No module named motionllm`：没有激活环境，或未执行 `pip install -e .`。
- dataset alias 找不到：检查文件名、JSON 内 `name` 和
  `MOTIONLLM_DATASET_CONFIG_DIR` 是否一致。
- normalization 报错：mean/std 缺一、维度与模型不一致，或路径不是绝对现存文件。
- placeholder token 报错：先确认 checkpoint 配置。新 token 必须处于 tokenizer 与
  embedding 范围内；明确持久化的历史外部 sentinel 则必须同时位于两者范围之外，
  不能出现只落入一侧的半绑定状态。
- `exit 78`：formal SFT 安全 gate 生效；先做开发 smoke，不能绕过后宣称正式训练。
- CUDA OOM：先保持 batch size 1、限制帧数并跑单样本；OOM 结果不能从评测分母移除。

历史原始 Qwen 数据代码位于 `legacy/qwen_vl_original/`，仅供对照，禁止导入。
