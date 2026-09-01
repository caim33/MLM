#!/usr/bin/env node

import { spawn } from "node:child_process";

const BASE_URL = process.env.AISTATION_URL || "https://aistation.sribd.cn:32206";
const DEBUG_PORT = Number(process.env.AISTATION_CHROME_DEBUG_PORT || 9222);

async function fetchJson(url, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function withCdp(target, callback) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    let id = 0;
    const pending = new Map();
    const timer = setTimeout(() => {
      ws.close();
      reject(new Error("CDP connection timed out"));
    }, 60000);

    ws.onopen = async () => {
      try {
        const send = (method, params = {}, timeoutMs = 30000) => {
          const requestId = ++id;
          ws.send(JSON.stringify({ id: requestId, method, params }));
          return new Promise((sendResolve, sendReject) => {
            const requestTimer = setTimeout(() => {
              if (pending.has(requestId)) {
                pending.delete(requestId);
                sendReject(new Error(`${method} timed out`));
              }
            }, timeoutMs);
            pending.set(requestId, {
              resolve: (value) => {
                clearTimeout(requestTimer);
                sendResolve(value);
              },
              reject: (error) => {
                clearTimeout(requestTimer);
                sendReject(error);
              },
            });
          });
        };
        const result = await callback(send);
        clearTimeout(timer);
        ws.close();
        resolve(result);
      } catch (error) {
        clearTimeout(timer);
        ws.close();
        reject(error);
      }
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (!message.id || !pending.has(message.id)) return;
      const handler = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) handler.reject(new Error(JSON.stringify(message.error)));
      else handler.resolve(message.result);
    };
    ws.onerror = () => {
      clearTimeout(timer);
      reject(new Error("CDP WebSocket error"));
    };
  });
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
      } catch {
        reject(new Error(`non-JSON response: ${stdout.slice(0, 200)}`));
      }
    });

    const config = [
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
    child.stdin.end(`${config.join("\n")}\n`);
  });
}

function tableData(payload) {
  const resData = payload?.resData ?? payload;
  if (Array.isArray(resData)) return resData;
  if (Array.isArray(resData?.data)) return resData.data;
  return [];
}

function numberFrom(value) {
  if (value === null || value === undefined || value === "") return 0;
  const match = String(value).match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function getGpuCount(row) {
  const resourceCount = String(row.acceleratorCardTypeAndNum || "").match(
    /[:：]\s*(\d+)\s*$/,
  );
  return Math.max(
    numberFrom(row.acceleratorCard),
    numberFrom(row.accelerator),
    numberFrom(row.acceleratorCardNum),
    resourceCount ? Number(resourceCount[1]) : 0,
  );
}

function isOwn(row, user) {
  const userIds = new Set(
    [user.userId, user.id].filter(Boolean).map((value) => String(value)),
  );
  const names = new Set(
    [user.account, user.username, user.userName]
      .filter(Boolean)
      .map((value) => String(value)),
  );
  return [
    row.wpOwnerId,
    row.userId,
    row.ownerId,
    row.creatorId,
    row.createUserId,
  ].some((value) => userIds.has(String(value))) ||
    [row.userName, row.ownerName, row.account, row.creator, row.createUser].some(
      (value) => names.has(String(value)),
    );
}

function mergeRows(baseRows, statusRows, usageRows) {
  const byId = new Map();
  for (const rows of [baseRows, statusRows, usageRows]) {
    for (const row of rows) {
      const id = row.wpId || row.id;
      if (!id) continue;
      byId.set(id, { ...(byId.get(id) || {}), ...row });
    }
  }
  return [...byId.values()];
}

const targets = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
const target = targets.find(
  (item) =>
    item.type === "page" &&
    item.webSocketDebuggerUrl &&
    String(item.url).startsWith(BASE_URL),
);
if (!target) throw new Error("No Aistation Chrome target found");

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
    returnByValue: true,
  });
  return result.result?.value || "";
});
if (!token) throw new Error("No Aistation login token found");

const headers = {
  "X-Auth-Token": token,
  "X-Accept-Language": "zh-cn",
  "X-Time-Zone": "8",
  Accept: "application/json, text/plain, */*",
};

const request = (url) => curlJson(`${BASE_URL}${url}`, headers);
const [userRaw, listRaw, baseRaw, statusRaw, usageRaw] = await Promise.all([
  request("/api/idam/v1/token"),
  request("/api/iresource/v1/work-platform?currentPage=1&pageSize=200"),
  request("/api/iresource/v1/work-platform/base?currentPage=1&pageSize=200"),
  request("/api/iresource/v1/work-platform/status?currentPage=1&pageSize=200"),
  request("/api/iresource/v1/work-platform/usage?currentPage=1&pageSize=200"),
]);

const failed = [
  ["user", userRaw],
  ["list", listRaw],
  ["base", baseRaw],
  ["status", statusRaw],
  ["usage", usageRaw],
].filter(([, payload]) => payload && payload.flag === false);
if (failed.length) {
  console.log(
    JSON.stringify(
      {
        status: "ERROR",
        failed: failed.map(([name, payload]) => ({
          name,
          errCode: payload.errCode,
          errMessage: payload.errMessage,
        })),
      },
      null,
      2,
    ),
  );
  process.exit(1);
}

const user = userRaw.resData || {};
const merged = mergeRows(
  tableData(baseRaw).concat(tableData(listRaw)),
  tableData(statusRaw),
  tableData(usageRaw),
);
const ownRows = merged.filter((row) => isOwn(row, user));
const activeGpuRows = ownRows
  .filter((row) => ["Running", "Timeout", "Pending"].includes(row.wpStatus))
  .filter((row) => getGpuCount(row) > 0)
  .map((row) => ({
    wpId: row.wpId || row.id,
    name: row.wpName || row.name,
    status: row.wpStatus,
    gpuCount: getGpuCount(row),
    gpuType:
      row.acceleratorCardType ||
      row.cardType ||
      row.acceleratorCardTypeAndNum ||
      "-",
    resource: row.acceleratorCardTypeAndNum || "-",
    nodeIp: row.nodeIp || row.hostIp || "-",
    nodeList: row.nodeList || row.nodes || [],
    runTime: row.runTime ?? null,
    remainTime: row.remainTime ?? null,
  }));

console.log(
  JSON.stringify(
    {
      status: "OK",
      checkedAt: new Date().toISOString(),
      user: {
        userId: user.userId,
        account: user.account,
        username: user.username,
      },
      ownWorkspaceCount: ownRows.length,
      activeGpuWorkspaceCount: activeGpuRows.length,
      activeGpuTotal: activeGpuRows.reduce((sum, row) => sum + row.gpuCount, 0),
      activeGpuRows,
    },
    null,
    2,
  ),
);
