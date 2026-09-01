#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";

const BASE_URL = process.env.AISTATION_URL || "https://aistation.sribd.cn:32206";
const DEBUG_PORT = Number(process.env.AISTATION_CHROME_DEBUG_PORT || 9223);
const apiPath = process.argv[2];
const jsonFile = process.argv[3];
const method = process.env.AISTATION_API_METHOD || "POST";

if (!apiPath || !apiPath.startsWith("/") || !jsonFile) {
  console.error("Usage: node tools/aistation_api_post.mjs /api/path payload.json");
  process.exit(2);
}

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
    ws.onopen = async () => {
      try {
        const send = (cdpMethod, params = {}) => {
          const requestId = ++id;
          ws.send(JSON.stringify({ id: requestId, method: cdpMethod, params }));
          return new Promise((sendResolve, sendReject) => {
            pending.set(requestId, { resolve: sendResolve, reject: sendReject });
            setTimeout(() => {
              if (pending.has(requestId)) {
                pending.delete(requestId);
                sendReject(new Error(`${cdpMethod} timed out`));
              }
            }, 30000);
          });
        };
        const result = await callback(send);
        ws.close();
        resolve(result);
      } catch (error) {
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
    ws.onerror = () => reject(new Error("CDP WebSocket error"));
  });
}

function curlEscape(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function curlJson(url, headers, dataFile) {
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
    child.on("close", (code) => {
      if (code !== 0) return reject(new Error(stderr.trim() || `curl ${code}`));
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
      "max-time = 60",
      `request = "${curlEscape(method)}"`,
      `url = "${curlEscape(url)}"`,
      `data-binary = "@${curlEscape(dataFile)}"`,
      ...Object.entries(headers).map(
        ([key, value]) => `header = "${curlEscape(`${key}: ${value}`)}"`,
      ),
    ];
    child.stdin.end(`${config.join("\n")}\n`);
  });
}

if (!fs.existsSync(jsonFile)) throw new Error(`Payload file not found: ${jsonFile}`);

const targets = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
const target = targets.find(
  (item) =>
    item.type === "page" &&
    item.webSocketDebuggerUrl &&
    String(item.url).startsWith(BASE_URL),
);
if (!target) throw new Error("No Aistation page target found");

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

const data = await curlJson(
  `${BASE_URL}${apiPath}`,
  {
    "X-Auth-Token": token,
    "X-Accept-Language": "zh-cn",
    "X-Time-Zone": "8",
    Accept: "application/json, text/plain, */*",
    "Content-Type": "application/json",
  },
  jsonFile,
);
console.log(JSON.stringify(data, null, 2));
