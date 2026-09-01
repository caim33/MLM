#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const BASE_URL = process.env.AISTATION_URL || "https://aistation.sribd.cn:32206";
const DEBUG_PORT = Number(process.env.AISTATION_CHROME_DEBUG_PORT || 9222);
const PROFILE_DIR =
  process.env.AISTATION_CHROME_PROFILE ||
  path.join(os.tmpdir(), "codex-aistation-chrome-profile");
const DEFAULT_GROUPS = ["A100_40G", "A100_80G", "H20"];

function parseArgs(argv) {
  const args = {
    threshold: Number(process.env.AISTATION_GPU_THRESHOLD || 4),
    groups: (process.env.AISTATION_GPU_GROUPS || DEFAULT_GROUPS.join(","))
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    launch: process.env.AISTATION_NO_CHROME_LAUNCH !== "1",
    stateFile: process.env.AISTATION_GPU_MONITOR_STATE || "",
  };

  for (const arg of argv) {
    if (arg.startsWith("--threshold=")) {
      args.threshold = Number(arg.slice("--threshold=".length));
    } else if (arg.startsWith("--groups=")) {
      args.groups = arg
        .slice("--groups=".length)
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    } else if (arg.startsWith("--state-file=")) {
      args.stateFile = arg.slice("--state-file=".length);
    } else if (arg === "--no-launch") {
      args.launch = false;
    }
  }

  if (!Number.isFinite(args.threshold) || args.threshold < 1) {
    throw new Error("threshold must be a positive number");
  }
  if (!args.groups.length) {
    throw new Error("at least one resource group must be configured");
  }
  return args;
}

function readState(stateFile) {
  if (!stateFile || !fs.existsSync(stateFile)) return {};
  try {
    return JSON.parse(fs.readFileSync(stateFile, "utf8"));
  } catch {
    return {};
  }
}

function writeState(stateFile, state) {
  if (!stateFile) return;
  fs.mkdirSync(path.dirname(stateFile), { recursive: true });
  fs.writeFileSync(stateFile, JSON.stringify(state, null, 2));
}

function alertFingerprint(result) {
  if (!result.alert) return "";
  const groups = (result.qualifyingGroups || [])
    .map((group) => `${group.groupName}:${group.free}`)
    .sort();
  const nodes = (result.qualifyingNodes || [])
    .map((node) => `${node.groupName}/${node.nodeName}:${node.free}`)
    .sort();
  return JSON.stringify({
    status: result.status,
    groups,
    nodes,
    message: result.message || "",
  });
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchJson(url, options = {}, timeoutMs = 10000) {
  const response = await fetchWithTimeout(url, options, timeoutMs);
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }
  return response.json();
}

function curlEscape(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function curlJson(url, headers) {
  return new Promise((resolve, reject) => {
    const child = spawn("curl.exe", ["--config", "-"], {
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `curl exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(
          new Error(
            `curl returned non-JSON response: ${stdout.slice(0, 200)} ${
              error.message
            }`,
          ),
        );
      }
    });

    const configLines = [
      "insecure",
      "silent",
      "show-error",
      "location",
      "max-time = 30",
      `url = "${curlEscape(url)}"`,
      ...Object.entries(headers).map(
        ([key, value]) => `header = "${curlEscape(`${key}: ${value}`)}"`,
      ),
    ];
    child.stdin.end(`${configLines.join("\n")}\n`);
  });
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    path.join(
      process.env.LOCALAPPDATA || "",
      "Google\\Chrome\\Application\\chrome.exe",
    ),
  ].filter(Boolean);

  return candidates.find((candidate) => fs.existsSync(candidate));
}

async function getTargets() {
  return fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`, {}, 5000);
}

async function waitForChrome() {
  const deadline = Date.now() + 25000;
  let lastError;
  while (Date.now() < deadline) {
    try {
      return await getTargets();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError || new Error("Chrome remote debugging did not start");
}

async function ensureChrome(allowLaunch) {
  try {
    return { targets: await getTargets(), launched: false };
  } catch {
    if (!allowLaunch) {
      throw new Error(
        `Chrome remote debugging is not available on port ${DEBUG_PORT}`,
      );
    }
  }

  const chrome = findChrome();
  if (!chrome) {
    throw new Error("Chrome executable was not found");
  }

  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  const args = [
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${PROFILE_DIR}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--headless=new",
    `${BASE_URL}/index.html#/startpage`,
  ];
  const child = spawn(chrome, args, {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  child.unref();

  return { targets: await waitForChrome(), launched: true };
}

async function createTarget() {
  const url = `${BASE_URL}/index.html#/startpage`;
  const endpoint = `http://127.0.0.1:${DEBUG_PORT}/json/new?${encodeURIComponent(url)}`;
  try {
    return await fetchJson(endpoint, { method: "PUT" }, 10000);
  } catch {
    return fetchJson(endpoint, {}, 10000);
  }
}

function withCdp(target, callback) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    let id = 0;
    const pending = new Map();
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error("CDP connection timed out"));
    }, 90000);

    ws.onopen = async () => {
      try {
        const send = (method, params = {}, timeoutMs = 60000) => {
          const requestId = ++id;
          ws.send(JSON.stringify({ id: requestId, method, params }));
          return new Promise((sendResolve, sendReject) => {
            const requestTimeout = setTimeout(() => {
              if (pending.has(requestId)) {
                pending.delete(requestId);
                sendReject(new Error(`${method} timed out`));
              }
            }, timeoutMs);
            pending.set(requestId, {
              resolve: (value) => {
                clearTimeout(requestTimeout);
                sendResolve(value);
              },
              reject: (error) => {
                clearTimeout(requestTimeout);
                sendReject(error);
              },
            });
          });
        };

        const result = await callback(send);
        clearTimeout(timeout);
        ws.close();
        resolve(result);
      } catch (error) {
        clearTimeout(timeout);
        ws.close();
        reject(error);
      }
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (!message.id || !pending.has(message.id)) return;
      const handler = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) {
        handler.reject(new Error(JSON.stringify(message.error)));
      } else {
        handler.resolve(message.result);
      }
    };

    ws.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("CDP WebSocket error"));
    };
  });
}

function summarize(groups, nodes, targetGroups, threshold) {
  const targetSet = new Set(targetGroups.map((group) => group.toLowerCase()));
  const isTarget = (groupName) => targetSet.has(String(groupName).toLowerCase());

  const groupSummary = groups
    .filter((group) => isTarget(group.groupName))
    .map((group) => {
      const total = Number(group.acceleratorCardCount || 0);
      const used = Number(group.usedAcceleratorCardCount || 0);
      return {
        groupId: group.groupId,
        groupName: group.groupName,
        cardTypes: (group.cardMap || []).map((card) => ({
          type: card.cardType,
          total: Number(card.cardCount || 0),
          memGB: Number(card.cardMem || 0),
        })),
        total,
        used,
        free: Math.max(0, total - used),
        network: (group.networkTypeList || []).join(","),
        taskNum: Number(group.taskNum || 0),
      };
    })
    .sort((a, b) => b.free - a.free || a.groupName.localeCompare(b.groupName));

  const nodeSummary = nodes
    .filter((node) => isTarget(node.groupName))
    .map((node) => {
      const total = Number(node.acceleratorCard || 0);
      const used = Number(node.acceleratorCardUsage || 0);
      return {
        groupId: node.groupId,
        groupName: node.groupName,
        nodeName: node.nodeName,
        nodeIp: node.nodeIp,
        cardType: node.nodeCardType || node.cardType,
        memGB: Number(node.acceleratorCardMemory || 0),
        total,
        used,
        free: Math.max(0, total - used),
        cpuFree: Number(node.cpu || 0) - Number(node.cpuUsage || 0),
        switchType: node.switchType,
        status: node.nodeStatus,
        resourceStatus: node.nodeResourceStatus,
      };
    })
    .sort(
      (a, b) =>
        b.free - a.free ||
        a.groupName.localeCompare(b.groupName) ||
        a.nodeName.localeCompare(b.nodeName),
    );

  return {
    groupSummary,
    nodeSummary,
    qualifyingGroups: groupSummary.filter((group) => group.free >= threshold),
    qualifyingNodes: nodeSummary.filter((node) => node.free >= threshold),
  };
}

async function queryAistation(targetGroups, threshold) {
  const { targets } = await ensureChrome(true);
  let target =
    targets.find(
      (item) =>
        item.type === "page" &&
        item.webSocketDebuggerUrl &&
        item.url.startsWith(BASE_URL),
    ) ||
    targets.find((item) => item.type === "page" && item.webSocketDebuggerUrl);

  if (!target) {
    target = await createTarget();
  }

  const token = await withCdp(target, async (send) => {
    await send("Runtime.enable");
    await send("Network.enable");

    const cookies = await send("Network.getAllCookies");
    const cookieToken =
      (cookies.cookies || []).find(
        (cookie) =>
          cookie.name === "Access_Token" &&
          String(cookie.domain || "").includes("aistation.sribd.cn"),
      )?.value || "";
    if (cookieToken) return cookieToken;

    const result = await send("Runtime.evaluate", {
      expression:
        'decodeURIComponent(document.cookie.match(/(?:^|;\\\\s*)Access_Token=([^;]+)/)?.[1] || "")',
      awaitPromise: true,
      returnByValue: true,
      timeout: 60000,
    });
    if (result.exceptionDetails) {
      throw new Error(JSON.stringify(result.exceptionDetails));
    }
    return result.result.value;
  });

  if (!token) {
    return {
      checkedAt: new Date().toISOString(),
      status: "AUTH_REQUIRED",
      alert: true,
      message: "Aistation login token was not found. Please log in again.",
    };
  }

  const headers = {
    "X-Auth-Token": token,
    "X-Accept-Language": "zh-cn",
    "X-Time-Zone": String(0 - new Date().getTimezoneOffset() / 60),
    Accept: "application/json, text/plain, */*",
  };
  async function requestJson(url) {
    return curlJson(`${BASE_URL}${url}`, headers);
  }
  const groupsRaw = await requestJson(
    "/api/iresource/v1/node-group?currentPage=1&pageSize=200",
  );
  const nodesRaw = await requestJson(
    "/api/iresource/v1/node?getUsage=1&nodeStatusFlag=2&currentPage=1&pageSize=200",
  );

  for (const [label, payload] of [
    ["node-group", groupsRaw],
    ["node", nodesRaw],
  ]) {
    if (!payload?.flag) {
      const errCode = payload?.errCode || "UNKNOWN_ERROR";
      return {
        checkedAt: new Date().toISOString(),
        status: errCode.includes("TOKEN") ? "AUTH_REQUIRED" : "ERROR",
        alert: errCode.includes("TOKEN"),
        message: `${label} query failed: ${errCode}`,
        error: {
          errCode,
          errMessage: payload?.errMessage || null,
          exceptionMsg: payload?.exceptionMsg || null,
        },
      };
    }
  }

  const groups = groupsRaw?.resData?.data || [];
  const nodes = nodesRaw?.resData?.data || [];
  const summary = summarize(groups, nodes, targetGroups, threshold);
  const alert =
    summary.qualifyingGroups.length > 0 || summary.qualifyingNodes.length > 0;

  return {
    checkedAt: new Date().toISOString(),
    status: "OK",
    threshold,
    targetGroups,
    alert,
    ...summary,
  };
}

let args;
try {
  args = parseArgs(process.argv.slice(2));
  const result = await queryAistation(args.groups, args.threshold);
  const fingerprint = alertFingerprint(result);
  const state = readState(args.stateFile);
  result.notify = Boolean(result.alert && fingerprint !== state.lastFingerprint);
  if (args.stateFile) {
    writeState(args.stateFile, {
      lastCheckedAt: result.checkedAt,
      lastStatus: result.status,
      lastFingerprint: fingerprint,
    });
  }
  console.log(JSON.stringify(result, null, 2));
  process.exitCode = 0;
} catch (error) {
  const result = {
    checkedAt: new Date().toISOString(),
    status: "ERROR",
    alert: true,
    message: error.message || String(error),
  };
  if (args?.stateFile) {
    const fingerprint = alertFingerprint(result);
    const state = readState(args.stateFile);
    result.notify = fingerprint !== state.lastFingerprint;
    writeState(args.stateFile, {
      lastCheckedAt: result.checkedAt,
      lastStatus: result.status,
      lastFingerprint: fingerprint,
    });
  } else {
    result.notify = true;
  }
  console.log(JSON.stringify(result, null, 2));
  process.exitCode = 1;
}
