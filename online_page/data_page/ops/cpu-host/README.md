# 常驻服务器上的每日恢复器

当前部署：`192.168.0.5`，用户 `liubingqi`，每天北京时间 **08:00** 检查一次。
用户级 `motionllm-cpu-keeper.timer` 已启用，用户原有 `Linger=yes`。
Windows/Codex 任务已停用。网站与所有数据仍在 Aistation；这里仅运行恢复控制程序。

## 工作流程

定时器启动临时容器 → 连接学校 VPN → 查询 Aistation 与公网健康状态 → 必要时
恢复已登记的到期 CPU 环境 → 断开 VPN、移除容器。正常状态不重启网站。
VPN 使用容器自己的网络空间；不会修改宿主机默认路由或现有网站的连接。

服务限制 0.5 CPU、384 MiB、96 个进程、0 GPU，8 分钟超时；同名容器与
systemd oneshot 防止同一部署重复运行。网络故障在同次任务内最多重试一次，
原核心的提交前状态复查及 20 分钟冷却继续生效。没有持续轮询或 GPU 保活。

## 文件与命令

以下目录相对常驻服务器用户家目录：

| 路径 | 内容 |
| --- | --- |
| `.local/share/motionllm-cpu-keeper/app/` | 此目录、核心检查器和构建上下文 |
| `.config/motionllm-cpu-keeper/config.json` | 原核心配置，加 `vpnHost`、`vpnAddress`、`vpnGroup` |
| `.config/motionllm-cpu-keeper/credentials.json` | 专用凭据，0600；不纳入源码或镜像 |
| `.local/state/motionllm-cpu-keeper/` | 当前 ID、状态及受限 VPN 诊断日志 |
| `.config/systemd/user/motionllm-cpu-keeper.{service,timer}` | 服务与每日定时器 |

```bash
systemctl --user list-timers motionllm-cpu-keeper.timer
systemctl --user show motionllm-cpu-keeper.service -p Result -p ExecMainStatus
journalctl --user -u motionllm-cpu-keeper.service -n 20 --no-pager
systemctl --user start motionllm-cpu-keeper.service
```

暂停：`systemctl --user disable --now motionllm-cpu-keeper.timer`。
恢复：`systemctl --user enable --now motionllm-cpu-keeper.timer`。
`Persistent=true` 让服务器重新上线后补执行错过的每日任务。

`run.py --help` 不读凭据或联网。默认不执行环境变更，必须显式 `--execute`。
它从挂载的专用凭据读取 `vpn_username`、`vpn_password`、`aistation_username`、
`aistation_password`；VPN 密码经标准输入，平台凭据经子进程环境注入。
普通日志只含动作/状态/错误类别，VPN 原始诊断文件限制为 0600。
不要把凭据加入 Docker 构建上下文，也不要启用 `--exercise-restart`。

## 构建与验证

从父级 `ops/` 目录构建：

```bash
docker build -t motionllm-cpu-keeper:20261021 -f cpu-host/Dockerfile .
node --test aistation-cpu-keeper.test.mjs
python3 cpu-host/test_run.py
```

Docker Hub 在当前主机下载失败，因此本次使用已有 `caimeng-web-192:base`
作为本地 Ubuntu 基础层，构建参数为 `--build-arg BASE_IMAGE=caimeng-web-192:base`。
这创建独立镜像，不更新原网站容器。Node 从官方发行地址下载并核对 SHA256；
OpenConnect 来自 Ubuntu 软件源。实际执行强制 `runc`，不挂入 GPU。

7 项核心测试和 4 项包装器测试通过。在 Windows VPN 断开时已实测独立 VPN 登录、
Aistation 模板/状态查询与公网健康检查，输出 `HEALTHY / Running`，退出码 0。
宿主机路由未变，结束后容器清理成功。完整 72 小时到期尚未实测；每日一次可能
使到期后等待接近 24 小时，再加平台排队和启动时间。常驻服务器及网络仍须可用。

参考：[OpenConnect 参数](https://www.infradead.org/openconnect/manual.html)、
[systemd 定时器](https://www.freedesktop.org/software/systemd/man/255/systemd.timer.html)。
