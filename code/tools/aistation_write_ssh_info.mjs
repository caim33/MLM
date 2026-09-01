#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const BASE_URL = process.env.AISTATION_URL || "https://aistation.sribd.cn:32206";
const DEBUG_PORT = Number(process.env.AISTATION_CHROME_DEBUG_PORT || 9223);
const wpId = process.argv[2];
const wpName = process.argv[3] || "";
const outFile =
  process.argv[4] ||
  path.resolve(process.cwd(), "dev_env_connection.txt");

if (!wpId) {
  console.error("Usage: node tools/aistation_write_ssh_info.mjs <wpId> [wpName] [outFile]");
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
      if (message.error) handler.reject(new Error(JSON.stringify(message.error)));
      else handler.resolve(message.result);
    };

    ws.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("CDP WebSocket error"));
    };
  });
}

function pick(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "") || "";
}

const targets = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
const target = targets.find(
  (item) =>
    item.type === "page" &&
    item.webSocketDebuggerUrl &&
    String(item.url).startsWith(BASE_URL),
);
if (!target) throw new Error("No Aistation page target found");

const sshInfo = await withCdp(target, async (send) => {
  await send("Runtime.enable");
  const expression = `
(async () => {
  const app = document.querySelector("#app")?.__vue_app__;
  const gp = app?.config?.globalProperties;
  if (!gp?.$crypto || !gp?.$axios) throw new Error("Aistation frontend helpers are unavailable");
  const decryptor = await gp.$crypto.getLocalDecryptor();
  const info = await gp.$axios({
    url: "/api/iresource/v1/work-platform/${wpId}/pod/default/shell",
    method: "get",
    params: { k: decryptor.cipherKey }
  });
  const result = { ...info };
  if (result.password) result.password = decryptor.decrypt(result.password);
  return result;
})()
`;
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    timeout: 60000,
  });
  if (result.exceptionDetails) {
    throw new Error(JSON.stringify(result.exceptionDetails));
  }
  return result.result.value;
});

const host = pick(sshInfo.host, sshInfo.ip, sshInfo.address, sshInfo.nodeIp);
const port = pick(sshInfo.port, sshInfo.nodePort, sshInfo.sshPort);
const username = pick(sshInfo.username, sshInfo.user, sshInfo.account, "root");
const password = pick(sshInfo.password, sshInfo.passwd);
const command = pick(
  sshInfo.command,
  sshInfo.sshCommand,
  host && port ? `ssh ${username}@${host} -p ${port}` : "",
);

const lines = [
  "Aistation development environment connection",
  "",
  `environment_name=${wpName}`,
  `wp_id=${wpId}`,
  `ssh_command=${command}`,
  `host=${host}`,
  `port=${port}`,
  `username=${username}`,
  `password=${password}`,
  "",
  "raw_fields_without_password=",
  JSON.stringify(
    Object.fromEntries(
      Object.entries(sshInfo).filter(([key]) => !/password|passwd/i.test(key)),
    ),
    null,
    2,
  ),
  "",
];

fs.writeFileSync(outFile, lines.join("\r\n"), "utf8");
console.log(JSON.stringify({ status: "OK", outFile, hasPassword: Boolean(password) }, null, 2));
