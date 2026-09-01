# 远端与 GPU 状态

更新时间：2026-08-21（Asia/Shanghai）

## 当前结论

- 用户指定的新端点 `10.26.6.88:31976` 本轮较早时曾通过 TCP 可达性检查，
  当时取得的 ED25519 host key 与本机为该端点固定的 key 一致；未来连接仍必须
  使用独立 `known_hosts` 和严格校验。
- 最终复查时连续 4 次短探测均不可达，直接 socket 结果为
  `ConnectionRefused`。因此当前不能声称该端点在线。
- 本机唯一允许的认证来源 `D:\MotionLLM\dev_env_connection.txt` 仍是旧内容：最后修改时间为 2026-07-29，记录端点为 `10.26.6.88:31349`，与本轮端点不一致。
- 公钥认证不可用；由于安全连接文件尚未更新，本轮没有建立已认证 SSH 会话。
- 聊天中的密码没有被写入命令、环境持久化文件、代码、Markdown、JSON、CSV、日志或 manifest，也没有被用作认证来源。
- 因此尚未同步活动仓库，尚未查询远端 `nvidia-smi`，尚未运行 Linux/CUDA smoke，尚未启动 GPU keepalive，也尚未开始正式 fresh finetune 或 QA500-v2 evaluation。

即使端口之后恢复，可达也只证明 SSH 服务有响应，不证明认证成功或 GPU 空闲。

## 恢复条件

请先把当前有效端点和认证更新到 `D:\MotionLLM\dev_env_connection.txt`。连接进程只能把密码读入进程级环境变量，禁止打印文件内容或把密码放进命令参数。

认证恢复后按以下顺序执行：

1. 使用固定 host key 建立会话。
2. 只读确认 hostname、时间、当前目录和磁盘空间。
3. 验证远端项目候选根目录及其 revision/hash。
4. 查询 `nvidia-smi`、GPU UUID、显存、利用率和计算进程。
5. 确认目标 UUID 真空闲，且不存在本项目 finetune/eval/keepalive lease。
6. 只同步到一个全新的、此前不存在的远端 staging 目录；禁止覆盖历史目录和使用 `--delete`。
7. 在 Linux 执行本地 Windows 因权限跳过的 symlink 测试。
8. 运行真实 CUDA keepalive start/status/stop 闭环。
9. 运行 SFT、GRPO、Rubric RL 所需的 CUDA/API smoke，包括 forward、backward、optimizer update、save/reload 和单条推理。
10. 正式 worker 启动前停止同卡 keepalive，再由 controller 获取 finetune/eval role lease。

## 远端候选目录

以下位置来自历史交接，只是候选，不构成当前存在性或正确性证明：

```text
/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM
/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval
/wangbenyou-sulongjie/qwen-vl-finetune
/wangbenyou-sulongjie/Motion-r1/qwen-vl-finetune
```

重新验证前，不得在这些路径执行覆盖、递归移动、删除或带 `--delete` 的同步。
