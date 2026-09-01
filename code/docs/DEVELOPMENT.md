# 开发说明

## 建立环境

```bash
cd /wangbenyou-sulongjie/caimeng/qwen-codebase-clean
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

若精简 Ubuntu 缺少 `python3-venv/ensurepip`，不要在项目脚本里偷偷安装系统包；
由机器管理员补齐 venv 组件，或使用已经冻结的容器/环境。

只有需要 Qwen/训练时才安装重型依赖：

```bash
python -m pip install -e '.[sft]'
```

不要为了查看帮助或运行数据测试而安装 CUDA 包。

## 修改顺序

1. 在 `src/` 中找到权威模块。
2. 先写失败测试，明确输入、输出和错误类型。
3. 修改核心实现。
4. 必要时更新 `qwenvl/`、顶层 `models/` 或 `rubric_rl/` 的过渡实现；新增核心逻辑优先下沉到 `src/`。
5. 更新状态和使用文档。

## 分层检查

```bash
# 快速、CPU-only
python -m pytest tests/unit tests/contract -q

# 跨模块
python -m pytest tests/integration -q

# 对抗性
python -m pytest tests/stress -q

# 全部本地门禁
python scripts/run_checks.py
```

`scripts/run_checks.py` 会执行 `compileall` 并写入 `__pycache__`，不是严格只读命令。

GPU 测试必须单独记录 CUDA、驱动、Torch、Transformers、模型 revision、数据 manifest 和输出目录，不能混入 CPU 基线数字。

## Import 与 CLI 约束

- `python -c "import motionllm, motion_eval"` 必须在 base 依赖中成功。
- `python -m motion_eval --help` 不得导入 Torch 或探测 GPU。
- 各命令的 `--help` 不得打开数据、加载权重或创建输出目录。
- 配置错误应在启动重型运行前返回明确的非零退出码。

## Legacy 修改

不要修改 `legacy/`。若需要旧逻辑：

1. 在迁移文档中标明来源 commit、文件和 blob hash。
2. 把必要行为重写到新的权威模块。
3. 用 contract test 证明兼容，而不是让活动代码 import legacy。
