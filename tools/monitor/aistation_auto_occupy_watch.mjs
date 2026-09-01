#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    intervalSec: 3600,
    threshold: 4,
    priority: ["H20", "A800", "A100_80G", "A100_40G"],
    templateWpId: "5d65c9e3-463c-4dac-97d5-dd0e67accbe4",
    execute: false,
    once: false,
    stateFile: path.join(repoRoot, ".codex_tmp", "aistation_auto_occupy_state.json"),
    logFile: path.join(repoRoot, ".codex_tmp", "aistation_auto_occupy.log"),
    connectionFile: path.join(repoRoot, "dev_env_connection.txt"),
    connectionRetrySec: 30,
    connectionMaxAttempts: 40,
    sshConfigFile: path.join(os.homedir(), ".ssh", "config"),
    sshHostAlias: "10.26.6.88",
  };
  for (const arg of argv) {
    if (arg === "--execute") args.execute = true;
    else if (arg === "--once") args.once = true;
    else if (arg.startsWith("--interval-sec=")) args.intervalSec = Number(arg.slice(15));
    else if (arg.startsWith("--threshold=")) args.threshold = Number(arg.slice(12));
    else if (arg.startsWith("--priority=")) args.priority = arg.slice(11).split(",").map((x) => x.trim()).filter(Boolean);
    else if (arg.startsWith("--template-wp-id=")) args.templateWpId = arg.slice(17);
    else if (arg.startsWith("--state-file=")) args.stateFile = arg.slice(13);
    else if (arg.startsWith("--log-file=")) args.logFile = arg.slice(11);
    else if (arg.startsWith("--connection-file=")) args.connectionFile = arg.slice(18);
    else if (arg.startsWith("--connection-retry-sec=")) args.connectionRetrySec = Number(arg.slice(23));
    else if (arg.startsWith("--connection-max-attempts=")) args.connectionMaxAttempts = Number(arg.slice(26));
    else if (arg.startsWith("--ssh-config-file=")) args.sshConfigFile = arg.slice("--ssh-config-file=".length);
    else if (arg.startsWith("--ssh-host-alias=")) args.sshHostAlias = arg.slice("--ssh-host-alias=".length);
  }
  return args;
}

function log(logFile, message) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  fs.appendFileSync(logFile, `${new Date().toISOString()} ${message}\n`, "utf8");
}

function runNode(script, args, timeoutMs = 120000) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [path.join(__dirname, script), ...args], {
      cwd: repoRoot,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      try {
        resolve({ code, data: JSON.parse(stdout), stderr });
      } catch {
        resolve({ code: code || 1, data: null, stderr: stderr || stdout });
      }
    });
  });
}

function readState(stateFile) {
  if (!fs.existsSync(stateFile)) return {};
  try {
    return JSON.parse(fs.readFileSync(stateFile, "utf8"));
  } catch {
    return {};
  }
}

function writeState(stateFile, state) {
  fs.mkdirSync(path.dirname(stateFile), { recursive: true });
  fs.writeFileSync(stateFile, JSON.stringify(state, null, 2));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function redactSecrets(value) {
  return String(value || "")
    .replace(/("(?:password|passwd|pwd)"\s*:\s*")[^"]*(")/gi, "$1<redacted>$2")
    .replace(/((?:password|passwd|pwd)\s*=\s*)\S+/gi, "$1<redacted>");
}

function readConnectionEndpoint(connectionFile) {
  if (!fs.existsSync(connectionFile)) return { host: "", port: 0 };
  const values = {};
  for (const line of fs.readFileSync(connectionFile, "utf8").split(/\r?\n/)) {
    const index = line.indexOf("=");
    if (index <= 0) continue;
    values[line.slice(0, index).trim()] = line.slice(index + 1).trim();
  }
  const commandMatch = String(values.ssh_command || "").match(
    /ssh\s+(?:[^@\s]+@)?([^\s]+)\s+-p\s+(\d+)/i,
  );
  return {
    host: values.host || commandMatch?.[1] || "",
    port: Number(values.port || commandMatch?.[2] || 0),
  };
}

function testTcp(host, port, timeoutMs = 5000) {
  return new Promise((resolve) => {
    if (!host || !Number.isInteger(port) || port <= 0) {
      resolve(false);
      return;
    }
    const socket = net.createConnection({ host, port });
    let settled = false;
    const finish = (reachable) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(reachable);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
  });
}

async function ensureConnectionInfo(args, wpId, wpName, force = false) {
  if (!wpId) return { action: "connection_skip_missing_wp_id" };

  const state = readState(args.stateFile);
  if (
    !force &&
    state.lastConnectionWpId === wpId &&
    state.lastConnectionVerified === true &&
    state.lastSshConfigUpdatedAt &&
    fs.existsSync(args.connectionFile)
  ) {
    return {
      action: "connection_current",
      wpId,
      outFile: args.connectionFile,
      reachable: true,
    };
  }

  let lastDetail = "";
  for (let attempt = 1; attempt <= args.connectionMaxAttempts; attempt += 1) {
    const written = await runNode(
      "aistation_write_ssh_info.mjs",
      [
        wpId,
        wpName || "",
        args.connectionFile,
        args.sshConfigFile,
        args.sshHostAlias,
      ],
      120000,
    );
    if (
      written.code === 0 &&
      written.data?.status === "OK" &&
      written.data?.hasPassword === true &&
      written.data?.sshConfig?.updated === true
    ) {
      const endpoint = readConnectionEndpoint(args.connectionFile);
      const reachable = await testTcp(endpoint.host, endpoint.port);
      if (reachable) {
        writeState(args.stateFile, {
          ...readState(args.stateFile),
          lastConnectionWpId: wpId,
          lastConnectionName: wpName || "",
          lastConnectionUpdatedAt: Date.now(),
          lastConnectionEndpoint: endpoint,
          lastConnectionVerified: true,
          lastSshConfigUpdatedAt: Date.now(),
          lastSshConfigPath: args.sshConfigFile,
          lastSshHostAlias: args.sshHostAlias,
        });
        return {
          action: "connection_updated",
          wpId,
          outFile: args.connectionFile,
          host: endpoint.host,
          port: endpoint.port,
          hasPassword: true,
          reachable: true,
          sshConfigFile: args.sshConfigFile,
          sshHostAlias: args.sshHostAlias,
          sshConfigUpdated: true,
          attempt,
        };
      }
      lastDetail = `SSH endpoint is not reachable yet (${endpoint.host || "unknown"}:${endpoint.port || "unknown"})`;
    } else {
      lastDetail = redactSecrets(written.stderr || JSON.stringify(written.data) || `exit=${written.code}`).slice(0, 500);
    }

    log(
      args.logFile,
      JSON.stringify({
        action: "connection_retry",
        wpId,
        attempt,
        maxAttempts: args.connectionMaxAttempts,
        detail: lastDetail,
      }),
    );
    if (attempt < args.connectionMaxAttempts) {
      await sleep(args.connectionRetrySec * 1000);
    }
  }

  writeState(args.stateFile, {
    ...readState(args.stateFile),
    lastConnectionWpId: wpId,
    lastConnectionName: wpName || "",
    lastConnectionUpdatedAt: Date.now(),
    lastConnectionVerified: false,
    lastConnectionError: lastDetail,
  });
  return {
    action: "connection_update_failed",
    wpId,
    outFile: args.connectionFile,
    reachable: false,
    detail: lastDetail,
  };
}

function timestampName(prefix) {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${prefix}_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function chooseCandidate(availability, priority, threshold) {
  for (const groupName of priority) {
    const nodes = (availability.nodeSummary || [])
      .filter(
        (node) =>
          node.groupName === groupName &&
          node.free >= threshold &&
          node.status === "ready" &&
          node.resourceStatus === "healthy",
      )
      .sort((a, b) => b.free - a.free || b.cpuFree - a.cpuFree);
    if (nodes.length) return nodes[0];
  }
  return null;
}

function buildPayload(template, candidate, threshold) {
  const payload = JSON.parse(JSON.stringify(template));
  payload.wpName = timestampName("auto4gpu");
  payload.groupId = candidate.groupId;
  payload.acceleratorCardKind = "GPU";
  payload.acceleratorCardType = candidate.cardType;
  payload.acceleratorCard = threshold;
  payload.switchType = candidate.switchType || payload.switchType;
  payload.nodeList = [];
  return payload;
}

function notify(title, body) {
  const ps = `
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
  const encoded = Buffer.from(ps, "utf16le").toString("base64");
  const child = spawn("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-EncodedCommand", encoded], {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  child.unref();
}

async function tick(args) {
  const usage = await runNode("aistation_user_gpu_usage.mjs", []);
  if (usage.code !== 0 || usage.data?.status !== "OK") {
    return { action: "skip_usage_error", detail: usage.stderr || JSON.stringify(usage.data) };
  }
  if (usage.data.activeGpuWorkspaceCount > 0) {
    return {
      action: "skip_user_active",
      activeGpuTotal: usage.data.activeGpuTotal,
      active: usage.data.activeGpuRows,
    };
  }

  const availability = await runNode("aistation_gpu_monitor.mjs", [
    `--threshold=${args.threshold}`,
    `--groups=${args.priority.join(",")}`,
    `--state-file=${path.join(repoRoot, ".codex_tmp", "aistation_auto_occupy_availability_state.json")}`,
  ]);
  if (availability.code !== 0 || availability.data?.status !== "OK") {
    return { action: "skip_availability_error", detail: availability.stderr || JSON.stringify(availability.data) };
  }

  const candidate = chooseCandidate(availability.data, args.priority, args.threshold);
  if (!candidate) {
    return { action: "skip_no_candidate" };
  }

  const state = readState(args.stateFile);
  const now = Date.now();
  if (state.lastCreateAttemptAt && now - state.lastCreateAttemptAt < 2 * 60 * 60 * 1000) {
    return { action: "skip_cooldown", candidate };
  }

  const rebuild = await runNode("aistation_api_get.mjs", [
    `/api/iresource/v1/work-platform/${args.templateWpId}/rebuild`,
  ]);
  if (rebuild.code !== 0 || rebuild.data?.flag === false) {
    return { action: "skip_template_error", detail: rebuild.stderr || JSON.stringify(rebuild.data) };
  }

  const payload = buildPayload(rebuild.data.resData, candidate, args.threshold);
  const payloadSummary = {
    wpName: payload.wpName,
    image: payload.image,
    groupId: payload.groupId,
    groupName: candidate.groupName,
    nodeName: candidate.nodeName,
    nodeIp: candidate.nodeIp,
    cardType: candidate.cardType,
    acceleratorCard: payload.acceleratorCard,
    cpu: payload.cpu,
    memory: payload.memory,
    shmSize: payload.shmSize,
    switchType: payload.switchType,
  };

  writeState(args.stateFile, {
    ...state,
    lastCreateAttemptAt: now,
    lastCandidate: payloadSummary,
  });

  if (!args.execute) return { action: "dry_run_create", payloadSummary };

  const payloadFile = path.join(repoRoot, ".codex_tmp", `aistation_auto_payload_${now}.json`);
  fs.mkdirSync(path.dirname(payloadFile), { recursive: true });
  fs.writeFileSync(payloadFile, JSON.stringify(payload), "utf8");
  try {
    const created = await runNode("aistation_api_post.mjs", [
      "/api/iresource/v1/work-platform/",
      payloadFile,
    ]);
    if (created.code !== 0 || created.data?.flag === false) {
      return { action: "create_failed", payloadSummary, detail: created.stderr || JSON.stringify(created.data) };
    }
    writeState(args.stateFile, {
      ...readState(args.stateFile),
      lastCreatedAt: Date.now(),
      lastCreated: payloadSummary,
      lastCreateResponse: created.data.resData || true,
    });
    return { action: "created", payloadSummary, response: created.data.resData || true };
  } finally {
    fs.rmSync(payloadFile, { force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  log(args.logFile, `auto watcher started execute=${args.execute} interval=${args.intervalSec}s priority=${args.priority.join(">")}`);
  if (args.once) {
    const result = await tick(args);
    log(args.logFile, JSON.stringify(result));
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  for (;;) {
    const result = await tick(args);
    let connection = null;
    if (result.action === "created") {
      connection = await ensureConnectionInfo(
        args,
        result.response?.wpId,
        result.payloadSummary?.wpName,
        true,
      );
    } else if (result.action === "skip_user_active") {
      const active =
        result.active?.find((row) => row.status === "Running") ||
        result.active?.[0];
      if (active?.wpId) {
        connection = await ensureConnectionInfo(args, active.wpId, active.name);
      }
    }
    const logRecord = connection ? { ...result, connection } : result;
    log(args.logFile, JSON.stringify(logRecord));
    if (result.action === "created") {
      notify(
        "Aistation 自动占卡成功",
        `${result.payloadSummary.groupName}/${result.payloadSummary.nodeName} ${result.payloadSummary.cardType} ${result.payloadSummary.acceleratorCard}卡\n环境: ${result.payloadSummary.wpName}`,
      );
    } else if (result.action === "create_failed") {
      notify("Aistation 自动占卡失败", String(result.detail || "创建请求失败").slice(0, 300));
    }
    const retrySec = result.action === "skip_usage_error"
      ? Math.min(args.intervalSec, 60)
      : args.intervalSec;
    await sleep(retrySec * 1000);
  }
}

main().catch((error) => {
  const logFile = path.join(repoRoot, ".codex_tmp", "aistation_auto_occupy.log");
  log(logFile, `fatal ${error.stack || error.message || String(error)}`);
  process.exitCode = 1;
});
