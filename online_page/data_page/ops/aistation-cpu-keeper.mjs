#!/usr/bin/env node
// External recovery for one registered CPU website workspace. No GPU allocation.
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const ACTIVE = new Set(["Running", "Timeout", "Pending", "Creating", "ImagePulling"]);
const rows = value => Array.isArray(value) ? value : value?.data || [];

export function selectTarget(workspaces, config) {
  const matching = workspaces.filter(r => (r.wpName || r.name) === config.workspaceName);
  const active = matching.filter(r => ACTIVE.has(r.wpStatus));
  if (active.length > 1) throw new Error("DUPLICATE_WORKSPACE");
  if (active.length) return active[0];
  const pinned = matching.find(r => (r.wpId || r.id) === config.wpId);
  if (pinned) return pinned;
  throw new Error("REGISTERED_WORKSPACE_MISSING");
}

export function validateTemplate(template, config) {
  if (template.acceleratorCardKind !== "CPU" || Number(template.acceleratorCard) !== 0)
    throw new Error("REFUSE_GPU_WORKSPACE");
  if (Number(template.cpu) !== config.cpu || template.command !== config.startupCommand)
    throw new Error("WORKSPACE_CONFIGURATION_CHANGED");
  if (Number(template.memory) !== config.memory)
    throw new Error("WORKSPACE_MEMORY_CHANGED");
}

export function decision(row, state, now, lifespanSeconds, exercise = false) {
  if (ACTIVE.has(row.wpStatus)) {
    return exercise && row.wpStatus === "Running" ? "EXERCISE" : "WAIT";
  }
  if (row.wpStatus !== "Halt") return "NEEDS_ATTENTION";
  const runtime = Number(row.runTime);
  if (row.runTime == null || !Number.isFinite(runtime) || runtime < 0)
    return "RUNTIME_UNKNOWN_NEEDS_ATTENTION";
  if (runtime < lifespanSeconds - 120) return "EARLY_STOP_NEEDS_ATTENTION";
  if (state.lastAttemptAt && now - state.lastAttemptAt < 20 * 60_000) return "COOLDOWN";
  return "RESTART_EXPIRED";
}

export async function reconcile(api, config, state, options) {
  const { now, execute, exercise, persist } = options;
  let target = selectTarget(await api.ownedWorkspaces(), { ...config, wpId: state.wpId || config.wpId });
  const id = target.wpId || target.id;
  const template = await api.request("/api/iresource/v1/work-platform/" + id + "/rebuild");
  validateTemplate(template, config);
  const policy = await api.request("/api/iresource/v1/work-platform/config");
  const minutes = Number(policy.timeoutConfig?.timeoutValueMin);
  if (!Number.isFinite(minutes) || minutes <= 0) throw new Error("LIFETIME_POLICY_UNKNOWN");
  const action = decision(target, state, now, minutes * 60, exercise);
  state.wpId = id;
  state.lastPlatformCheckAt = now;
  state.remainingSeconds = Number(target.remainTime);
  state.expiresAt = Number.isFinite(state.remainingSeconds) ? now + state.remainingSeconds * 1000 : null;
  state.platformStatus = target.wpStatus;
  persist(state);
  if (!["RESTART_EXPIRED", "EXERCISE"].includes(action)) return { action, status: target.wpStatus };
  const nodes = rows(await api.request("/api/iresource/v1/node?getUsage=1&nodeStatusFlag=2&currentPage=1&pageSize=200"));
  const candidate = nodes.find(n => n.groupId === template.groupId &&
    n.nodeStatus === "ready" && n.nodeResourceStatus === "healthy" &&
    Number(n.cpu) - Number(n.cpuUsage) >= config.cpu);
  if (!candidate) return { action: "WAIT_CAPACITY" };
  // Re-read immediately before a mutation, including after an unknown prior outcome.
  const fresh = selectTarget(await api.ownedWorkspaces(), { ...config, wpId: id });
  if ((fresh.wpId || fresh.id) !== id || fresh.wpStatus !== target.wpStatus)
    return { action: "STATE_CHANGED_RECHECK" };
  if (!execute) return { action: "DRY_RUN", planned: action, cpu: config.cpu, gpu: 0 };
  state.lastAttemptAt = now;
  state.lastAction = action;
  persist(state); // A timeout must not cause a blind duplicate create.
  const suffix = action === "EXERCISE" ? "/reconfig" : "";
  await api.request("/api/iresource/v1/work-platform/" + id + suffix, {
    method: "POST", body: { ...template, nodeList: [] },
  });
  return { action: "RESTART_REQUESTED", reason: action, cpu: config.cpu, gpu: 0 };
}

function curlJson(url, { method = "GET", body, headers = {}, timeout = 30 } = {}) {
  const quote = s => String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\r/g, "\\r").replace(/\n/g, "\\n");
  const lines = ["silent", "show-error", "fail-with-body", "max-time = " + timeout,
    'url = "' + quote(url) + '"', 'request = "' + method + '"'];
  if (body !== undefined) {
    headers = { ...headers, "Content-Type": "application/json" };
    lines.push('data = "' + quote(JSON.stringify(body)) + '"');
  }
  for (const [key, value] of Object.entries(headers))
    lines.push('header = "' + quote(key + ": " + value) + '"');
  return new Promise((resolve, reject) => {
    const child = spawn(process.platform === "win32" ? "curl.exe" : "curl", ["--config", "-"],
      { windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    let output = "";
    child.stdout.on("data", chunk => { output += chunk; });
    child.stderr.resume();
    child.on("error", () => reject(new Error("HTTP_CLIENT_START_FAILED")));
    child.on("close", code => {
      if (code) return reject(new Error("HTTP_FAILED_" + code));
      try { resolve(JSON.parse(output)); } catch { reject(new Error("INVALID_HTTP_RESPONSE")); }
    });
    child.stdin.on("error", () => {});
    child.stdin.end(lines.join("\n") + "\n");
  });
}

class Aistation {
  constructor(config) { this.config = config; this.token = ""; }
  async request(endpoint, options = {}) {
    const value = await curlJson(this.config.baseUrl + endpoint, {
      ...options, headers: { "X-Accept-Language": "zh-cn", "X-Time-Zone": "8",
        ...(this.token ? { "X-Auth-Token": this.token } : {}) },
    });
    if (value.flag === false)
      throw new Error("API_" + String(value.errCode || "REJECTED").replace(/[^A-Za-z0-9_-]/g, "").slice(0,50));
    return value.resData === undefined ? value : value.resData;
  }
  async login() {
    const require = createRequire(this.config.cryptoPackageFile);
    const { sm2 } = require("sm-crypto");
    if (await this.request("/api/idam/v1/captcha")) throw new Error("INTERACTIVE_CAPTCHA_REQUIRED");
    const key = await this.request("/api/idam/v1/security/secret");
    const username = process.env.AISTATION_LOGIN_USERNAME, password = process.env.AISTATION_LOGIN_PASSWORD;
    if (!username || !password) throw new Error("PROCESS_CREDENTIALS_MISSING");
    const value = await this.request("/api/idam/v1/login", { method: "POST", body: {
      account: "04" + sm2.doEncrypt(username, key, 0),
      password: "04" + sm2.doEncrypt(password, key, 0),
    }});
    delete process.env.AISTATION_LOGIN_PASSWORD;
    delete process.env.AISTATION_LOGIN_USERNAME;
    if (!value.token) throw new Error("LOGIN_FAILED");
    this.token = value.token;
    this.user = await this.request("/api/idam/v1/token");
  }
  async ownedWorkspaces() {
    const payloads = await Promise.all(["", "/base", "/status", "/usage"].map(
      suffix => this.request("/api/iresource/v1/work-platform" + suffix + "?currentPage=1&pageSize=200")));
    const merged = new Map();
    for (const payload of payloads) for (const row of rows(payload)) {
      const id = row.wpId || row.id;
      if (id) merged.set(id, { ...merged.get(id), ...row });
    }
    const ids = [this.user.userId, this.user.id].filter(Boolean).map(String);
    const names = [this.user.account, this.user.username, this.user.userName].filter(Boolean);
    return [...merged.values()].filter(r =>
      [r.wpOwnerId, r.userId, r.ownerId, r.creatorId].some(v => ids.includes(String(v))) ||
      [r.userName, r.ownerName, r.account, r.creator].some(v => names.includes(v)));
  }
}

function save(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file + ".tmp", JSON.stringify(value, null, 2) + "\n");
  fs.renameSync(file + ".tmp", file);
}

export async function main(args) {
  if (args.includes("--help")) {
    console.log("aistation-cpu-keeper.mjs --config FILE [--execute] [--refresh] [--exercise-restart]\n" +
      "Use invoke-cpu-keeper.ps1 for credential loading and exclusive locking. Default: dry-run.");
    return;
  }
  const configIndex = args.indexOf("--config");
  if (configIndex < 0 || !args[configIndex + 1]) throw new Error("CONFIG_REQUIRED");
  const config = JSON.parse(fs.readFileSync(args[configIndex + 1], "utf8").replace(/^\uFEFF/, ""));
  if (!config.enabled) { console.log('{"action":"DISABLED"}'); return; }
  let state = {};
  if (fs.existsSync(config.stateFile)) state = JSON.parse(fs.readFileSync(config.stateFile, "utf8"));
  const now = Date.now();
  let healthy = false;
  try { healthy = (await curlJson(config.healthUrl, { timeout: 15 })).ok === true; } catch {}
  const exercise = args.includes("--exercise-restart");
  let result;
  if (healthy && !exercise && !args.includes("--refresh") &&
      state.platformStatus === "Running" &&
      (!state.lastAttemptAt || state.lastAttemptAt < state.lastPlatformCheckAt) &&
      state.lastPlatformCheckAt &&
      now - state.lastPlatformCheckAt < 3_600_000 && state.expiresAt > now + 600_000) {
    result = { action: "HEALTHY" };
  } else {
    const api = new Aistation(config);
    try {
      await api.login();
      result = await reconcile(api, config, state, { now,
        execute: args.includes("--execute"), exercise,
        persist: value => save(config.stateFile, value) });
      if (result.action === "WAIT" && healthy) result.action = "HEALTHY";
      else if (result.action === "WAIT" && result.status === "Running") result.action = "WAIT_WEBSITE";
    } finally { api.token = ""; }
  }
  state.lastCheckAt = now;
  state.publicHealthy = healthy;
  state.lastResult = result.action;
  save(config.stateFile, state);
  console.log(JSON.stringify(result));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).catch(error => {
    const safe = /^[A-Z][A-Z0-9_]+$/.test(error.message) ? error.message : "KEEPER_FAILED";
    console.error(JSON.stringify({ action: "NEEDS_ATTENTION", reason: safe }));
    process.exitCode = 1;
  }).finally(() => {
    delete process.env.AISTATION_LOGIN_USERNAME;
    delete process.env.AISTATION_LOGIN_PASSWORD;
  });
}
