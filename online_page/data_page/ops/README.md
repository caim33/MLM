# Viewer 自动启动与故障恢复

容器内使用 Linux/Python 标准库管理网站后端与 Cloudflare Tunnel；
外部 Windows/Node 检查器负责普通 CPU 环境到期后的重新启动。
不加载 CUDA、不启动 GPU 保活、不重建数据索引。原始数据、模型和索引由已有部署提供。

## 已部署入口

远端运行目录：

```text
/wangbenyou-sulongjie/caimeng/runtime/cloudflare-viewer
```

Aistation 创建或重新启动环境时，在“启动命令”中保留：

```bash
bash /wangbenyou-sulongjie/caimeng/runtime/cloudflare-viewer/viewer-bootstrap.sh
```

脚本启动后台 supervisor，并在容器 `.bashrc` 的交互判断之前安装登录补启动
入口。环境一旦启动即可恢复网站，无需打开 SSH；仅登录 Aistation 网页不会
自动启动已经被平台终止的环境。

2026-10-21 已实测：新容器启动后、第一次 SSH 登录前，公网 `/health` 返回
200。平台的 `/reconfig` 调整资源接口会忽略启动命令；应使用创建/启动环境
流程设置，并重新查询 `/rebuild` 确认 `command` 保存成功。

自动分配工具 `tools/aistation_auto_occupy_watch.mjs` 会继承模板的 `command`。
必须选择带上述启动命令的当前模板；旧模板的空命令不能自动启动网站。
不要为了网站反复申请四张 GPU。GPU 环境终止后的资源重新申请不由本服务处理。

## 配置与操作

源码在此目录；部署时复制到运行目录。`viewer-service.json` 是仅存在于服务器的
部署配置，包含 `runtime_root`、`interval_seconds`、数据路径环境变量 `env`，
以及两个服务的 `argv`、`cwd`、`health_url`、`startup_grace`、`failure_limit`。
不包含密码、隧道令牌或凭据内容。Cloudflare 原有配置与凭据继续留在运行目录。

当前健康检查：

- 后端：`http://127.0.0.1:8765/health`。
- 隧道连接就绪：`http://127.0.0.1:8766/ready`，指标端口仅监听回环地址。
- 公网：`https://viewer.caimeng.online/health`。

```bash
cd /wangbenyou-sulongjie/caimeng/runtime/cloudflare-viewer
bash start-public-viewer.sh
python3 viewer-supervisor.py --config viewer-service.json status
```

`start` 通过文件锁避免并发启动多份进程；`run` 前台运行；`status` 输出状态，
本地健康检查失败或状态超过 30 秒未更新时返回非零。`--help` 不连接服务。
启动脚本只负责启动，旧入口 `start-public-viewer.sh` 最多等待约一分钟就绪。

检查间隔 5 秒。进程退出后延迟 2–60 秒重试；后端持续三次检查失败后重启，
隧道持续十二次检查失败后重启，首次启动允许 60 秒准备时间。
只终止当前 supervisor 自己创建、尚未回收的子进程组。

状态：`supervisor.json`；进程：`viewer.pid`、`cloudflared.pid`；
日志：`logs/supervisor.log`、`logs/viewer.log`、`logs/cloudflared.log`。
supervisor 日志自动轮转；子进程日志在重启时超过 10 MB 则保留一份旧日志。
退出 SSH 不停止服务。整体 supervisor 异常退出后，由下次登录或环境启动补启动。

## 平台停止与全天在线的边界

2026-10-21 读取平台 `/api/iresource/v1/work-platform/config`：
自动终止开启，GPU 策略检查周期 240 分钟、MAX 利用率阈值 1%，最长运行
4320 分钟（72 小时）。前一环境运行 14600 秒后进入 Halt，与 GPU 空闲
回收规则吻合；当前账户无权读取操作日志，因此没有逐次终止事件的审计证据。

网站访问主要使用 CPU，不会维持 GPU 利用率。整个环境被终止时，
容器内恢复程序也会停止，无法承诺全天在线。长期部署需要管理员允许的常驻
CPU 服务并能访问同一数据目录。当前账号无常驻服务部署权限；已按用户选择，
先使用普通 CPU 环境及以下外部到期恢复机制。

## 普通 CPU 环境的到期恢复

2026-10-21 已把原网站环境转换为 **2 核 CPU、0 GPU**，继续读取原 GPFS 数据。
名称仍为 `auto4gpu_20261020141820`，属于历史名称；平台内存字段保留 `0`，
没有确认独立 4 GiB 限额。当前环境 ID 由状态文件和平台查询获取，不写死在源码。

`aistation-cpu-keeper.mjs` 是外部检查器；`invoke-cpu-keeper.ps1` 提供文件锁与
登录信息加载。生产动作默认关闭，必须指定 `-Execute`。检查器只恢复已登记的
同名 CPU 环境，拒绝 GPU、资源或启动命令变化，不修改数据、镜像或挂载。

已启用当前对话的 Codex 自动任务 `aistation-cpu`，每 5 分钟运行一次：

```powershell
& 'D:\MotionLLM\MLM_site_update\online_page\data_page\ops\invoke-cpu-keeper.ps1' `
  -Config 'D:\MotionLLM\.codex_tmp\viewer-cpu-keeper\config.json' -Execute
```

### 参数与配置

- 省略 `-Execute`：检查及预演；仍更新本地状态，不创建/重启环境。
- `-Refresh`：强制查询平台。正常在线时每小时刷新平台信息；到期前十分钟或
  公网异常时立即刷新。
- `-ExerciseRestart`：主动重启运行中的 CPU 环境，用于人工演练；自动任务禁止使用。
- `node aistation-cpu-keeper.mjs --help`：不访问网络；Node CLI 使用对应的
  `--config`、`--execute`、`--refresh`、`--exercise-restart`。

配置字段：`enabled`、`baseUrl`、`workspaceName`、初始 `wpId`、`cpu`、`memory`、
`startupCommand`、`healthUrl`、`stateFile`、`agentsFile`、`cryptoPackageFile`、
`nodeExecutable`。配置没有密码；Windows 包装器从本机指定的登录条目读取凭据，
注入子进程环境。Node 需要 `sm-crypto`；其包路径由 `cryptoPackageFile` 指定。

本机部署配置/状态在 `D:\MotionLLM\.codex_tmp\viewer-cpu-keeper\`，
SM2 依赖在 `D:\MotionLLM\.codex_tmp\viewer-test-deps\`。这些是自动任务依赖，
不要作为临时文件清理。停止自动恢复：暂停 Codex 任务，或设置 `enabled=false`。

### 行为与限制

只有 Halt 且运行时长达到平台期限（容差 120 秒）才请求启动；运行时长仅用于
判断到期，不能证实停机操作者。提前停止、暂停或未知时长留待核查。
请求前重新检查状态及同组 CPU 容量；同一环境不重复申请。
状态文件记录当前 ID、到期时间、健康检查和操作时间；提交前落盘，未知结果采用
20 分钟冷却，避免重复请求。标准输出仅为动作/错误类别，不含认证内容。

电脑须开机、Codex 应用运行且 Cisco VPN 可用；到期后会有检查周期加排队/启动
时间的中断，容量不足或登录验证会延迟恢复。容器启动后仍由原 bootstrap/supervisor
恢复网站，无需 SSH 登录。人工转换 CPU 配置并真实重启已验证成功，公网页面、
健康和动作接口返回 200，CPU 网格读取成功；尚未等待完整 72 小时到期实测。

## 验证

```bash
python3 test_supervisor.py
bash -n viewer-bootstrap.sh start-public-viewer.sh
node --test aistation-cpu-keeper.test.mjs
```

两项 Linux 集成测试使用独立临时目录和 HTTP 子进程，验证并发重复启动、
崩溃恢复和卡死恢复，不读真实数据。真实网站的后端和隧道恢复另做上线检查。
7 项 Node 测试覆盖到期/未知时长判断、重复环境、配置变化、挂载保留、预演、
容量不足、并发替换和请求结果不明，不执行真实资源申请。
