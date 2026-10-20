# Viewer 自动启动与故障恢复

只管理网站后端与 Cloudflare Tunnel。Linux，Python 标准库；不加载 CUDA、
不启动 GPU 保活、不重建数据索引。原始数据、模型和索引由已有部署提供。

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
CPU 服务并能访问同一数据目录，或为网站提供单独的数据副本。
本次仅看到 GPU 资源组，没有确认可长期运行的 CPU 服务配额。

## 验证

```bash
python3 test_supervisor.py
bash -n viewer-bootstrap.sh start-public-viewer.sh
```

两项 Linux 集成测试使用独立临时目录和 HTTP 子进程，验证并发重复启动、
崩溃恢复和卡死恢复，不读真实数据。真实网站的后端和隧道恢复另做上线检查。
