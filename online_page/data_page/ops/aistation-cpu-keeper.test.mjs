import test from "node:test";
import assert from "node:assert/strict";
import { decision, selectTarget, validateTemplate, reconcile } from "./aistation-cpu-keeper.mjs";

const config = { workspaceName: "viewer", wpId: "old", cpu: 2, memory: 0, startupCommand: "boot-viewer" };
const template = { wpName: "viewer", cpu: 2, memory: 0, acceleratorCardKind: "CPU",
  acceleratorCard: 0, command: "boot-viewer", groupId: "group", volumes: [{ path: "/persistent" }],
  env: [{ key: "KEEP", value: "original" }], image: "same-image" };
const halted = { wpId: "old", wpName: "viewer", wpStatus: "Halt", runTime: 259300 };
const running = { ...halted, wpId: "new", wpStatus: "Running", runTime: 5 };
const now = 10_000_000;

test("only expiry restarts; manual early stop, pause and pending do not", () => {
  assert.equal(decision(halted, {}, now, 259200), "RESTART_EXPIRED");
  assert.equal(decision({ ...halted, runTime: 100 }, {}, now, 259200), "EARLY_STOP_NEEDS_ATTENTION");
  assert.equal(decision({ ...halted, wpStatus: "Pause" }, {}, now, 259200), "NEEDS_ATTENTION");
  assert.equal(decision({ ...halted, wpStatus: "Pending" }, {}, now, 259200), "WAIT");
  assert.equal(decision(halted, { lastAttemptAt: now - 1000 }, now, 259200), "COOLDOWN");
  for (const runTime of [undefined, null, "unknown", -1])
    assert.equal(decision({ ...halted, runTime }, {}, now, 259200), "RUNTIME_UNKNOWN_NEEDS_ATTENTION");
});

test("active replacement wins; wrong name and duplicate targets are rejected", () => {
  assert.equal(selectTarget([halted, running], config).wpId, "new");
  assert.throws(() => selectTarget([{ ...running, wpName: "training" }], config), /MISSING/);
  assert.throws(() => selectTarget([running, { ...running, wpId: "third" }], config), /DUPLICATE/);
});

test("refuses GPU, changed resource allocation and changed startup command", () => {
  validateTemplate(template, config);
  for (const changed of [{ acceleratorCardKind: "GPU" }, { acceleratorCard: 4 },
    { cpu: 60 }, { command: "other-job" }, { memory: 8 }]) {
    assert.throws(() => validateTemplate({ ...template, ...changed }, config));
  }
});

function mockApi(second = [halted], failPost = false, enoughCapacity = true) {
  let calls = 0;
  const writes = [];
  return { writes,
    ownedWorkspaces: async () => ++calls === 1 ? [halted] : second,
    request: async (url, options) => {
      if (options?.method === "POST") {
        writes.push({ url, body: options.body });
        if (failPost) throw new Error("RESPONSE_LOST");
        return {};
      }
      if (url.endsWith("/rebuild")) return structuredClone(template);
      if (url.endsWith("/config")) return { timeoutConfig: { timeoutValueMin: 4320 } };
      return { data: [{ groupId: "group", nodeStatus: "ready", nodeResourceStatus: "healthy",
        cpu: 10, cpuUsage: enoughCapacity ? 4 : 10 }] };
    } };
}
const opts = { now, execute: true, exercise: false, persist: () => {} };

test("expiry uses existing restart endpoint and preserves image, env and volumes", async () => {
  const api = mockApi();
  const result = await reconcile(api, config, {}, opts);
  assert.equal(result.action, "RESTART_REQUESTED");
  assert.equal(api.writes.length, 1);
  assert.equal(api.writes[0].url, "/api/iresource/v1/work-platform/old");
  assert.deepEqual(api.writes[0].body.volumes, template.volumes);
  assert.deepEqual(api.writes[0].body.env, template.env);
  assert.equal(api.writes[0].body.image, template.image);
  assert.equal(api.writes[0].body.acceleratorCard, 0);
});

test("dry run and capacity shortage never create a workspace", async () => {
  const api = mockApi();
  assert.equal((await reconcile(api, config, {}, { ...opts, execute: false })).action, "DRY_RUN");
  assert.equal(api.writes.length, 0);
  const full = mockApi([halted], false, false);
  assert.equal((await reconcile(full, config, {}, opts)).action, "WAIT_CAPACITY");
  assert.equal(full.writes.length, 0);
});

test("concurrent replacement is discovered before mutation", async () => {
  const api = mockApi([running]);
  assert.equal((await reconcile(api, config, {}, opts)).action, "STATE_CHANGED_RECHECK");
  assert.equal(api.writes.length, 0);
});

test("unknown create outcome records cooldown before request", async () => {
  const api = mockApi([halted], true);
  const state = {};
  let saved;
  await assert.rejects(reconcile(api, config, state, {
    ...opts, persist: value => { saved = structuredClone(value); },
  }), /RESPONSE_LOST/);
  assert.equal(saved.lastAttemptAt, now);
  assert.equal(decision(halted, saved, now + 1000, 259200), "COOLDOWN");
});
