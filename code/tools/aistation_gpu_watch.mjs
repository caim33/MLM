#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    threshold: 4,
    groups: "A100_40G,A100_80G,H20",
    intervalSec: 300,
    stateFile: path.join(repoRoot, ".codex_tmp", "aistation_gpu_monitor_state.json"),
    logFile: path.join(repoRoot, ".codex_tmp", "aistation_gpu_watch.log"),
  };

  for (const arg of argv) {
    if (arg.startsWith("--threshold=")) args.threshold = Number(arg.slice(12));
    else if (arg.startsWith("--groups=")) args.groups = arg.slice(9);
    else if (arg.startsWith("--interval-sec=")) args.intervalSec = Number(arg.slice(15));
    else if (arg.startsWith("--state-file=")) args.stateFile = arg.slice(13);
    else if (arg.startsWith("--log-file=")) args.logFile = arg.slice(11);
  }

  if (!Number.isFinite(args.threshold) || args.threshold < 1) {
    throw new Error("threshold must be a positive number");
  }
  if (!Number.isFinite(args.intervalSec) || args.intervalSec < 30) {
    throw new Error("interval-sec must be at least 30");
  }
  return args;
}

function log(logFile, message) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  fs.appendFileSync(logFile, `${new Date().toISOString()} ${message}\n`, "utf8");
}

function runCheck(args) {
  const monitorScript = path.join(__dirname, "aistation_gpu_monitor.mjs");
  const child = spawn(
    process.execPath,
    [
      monitorScript,
      `--threshold=${args.threshold}`,
      `--groups=${args.groups}`,
      `--state-file=${args.stateFile}`,
    ],
    {
      cwd: repoRoot,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  return new Promise((resolve) => {
    child.on("close", () => {
      try {
        resolve(JSON.parse(stdout));
      } catch {
        resolve({
          checkedAt: new Date().toISOString(),
          status: "ERROR",
          notify: true,
          message: stderr.trim() || stdout.trim() || "GPU monitor command failed",
        });
      }
    });
  });
}

function formatNotification(result) {
  if (result.status === "OK") {
    const groups = (result.qualifyingGroups || [])
      .map((group) => `${group.groupName}: ${group.free}/${group.total} 空闲`)
      .join("; ");
    const nodes = (result.qualifyingNodes || [])
      .map(
        (node) =>
          `${node.groupName}/${node.nodeName}(${node.nodeIp}) ${node.cardType}: ${node.free}/${node.total}`,
      )
      .join("; ");
    return {
      title: "Aistation GPU 可用",
      body: [groups, nodes].filter(Boolean).join("\n"),
    };
  }

  if (result.status === "AUTH_REQUIRED") {
    return {
      title: "Aistation GPU 监控需要登录",
      body: "登录态过期。请重新登录监控用的 Chrome profile。",
    };
  }

  return {
    title: "Aistation GPU 监控异常",
    body: result.message || "查询失败，请检查日志。",
  };
}

function showWindowsNotification(title, body) {
  const script = `
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.Visible = $true
$n.BalloonTipTitle = @'
${title}
'@
$n.BalloonTipText = @'
${body}
'@
$n.ShowBalloonTip(15000)
Start-Sleep -Seconds 16
$n.Dispose()
`;
  const encoded = Buffer.from(script, "utf16le").toString("base64");
  const child = spawn(
    "powershell.exe",
    [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-WindowStyle",
      "Hidden",
      "-EncodedCommand",
      encoded,
    ],
    { detached: true, stdio: "ignore", windowsHide: true },
  );
  child.unref();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  log(args.logFile, `watcher started interval=${args.intervalSec}s groups=${args.groups}`);

  for (;;) {
    const result = await runCheck(args);
    log(
      args.logFile,
      `check status=${result.status} alert=${Boolean(result.alert)} notify=${Boolean(result.notify)}`,
    );

    if (result.status === "OK" && result.notify) {
      const notification = formatNotification(result);
      log(args.logFile, `notify title=${notification.title} body=${notification.body.replace(/\s+/g, " ")}`);
      showWindowsNotification(notification.title, notification.body);
    }

    await new Promise((resolve) => setTimeout(resolve, args.intervalSec * 1000));
  }
}

main().catch((error) => {
  const fallbackLog = path.join(repoRoot, ".codex_tmp", "aistation_gpu_watch.log");
  log(fallbackLog, `fatal ${error.stack || error.message || String(error)}`);
  process.exitCode = 1;
});
