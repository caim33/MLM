import test from "node:test";
import assert from "node:assert/strict";
import { requestData } from "../static/viewer-request.mjs";
const immediate = async () => {};

test("recovers from a transient network failure without reloading the page", async () => {
  let calls = 0;
  const result = await requestData("/fixture", {wait: immediate, fetchImpl: async () => {
    if (++calls === 1) throw new TypeError("Failed to fetch");
    return new Response('{"ok":true}');
  }});
  assert.equal(calls, 2); assert.equal(result.body.ok, true);
});
test("retries when the connection fails while reading a binary response body", async () => {
  let calls = 0;
  const result = await requestData("/fixture", {kind:"buffer",wait:immediate,fetchImpl:async () => {
    if (++calls === 1) return {ok:true,arrayBuffer:async () => {throw new TypeError("stream lost");}};
    return new Response(new Uint8Array([3,4]), {headers:{"X-Motion-Frames":"1"}});
  }});
  assert.equal(calls,2); assert.deepEqual([...new Uint8Array(result.body)],[3,4]);
  assert.equal(result.headers.get("X-Motion-Frames"),"1");
});
test("timeouts abort stalled reads and finish after the bounded retry budget", async () => {
  let calls=0;
  await assert.rejects(requestData("/fixture",{timeoutMs:5,retries:1,wait:immediate,fetchImpl:async (_, {signal}) => {
    calls++;
    return {ok:true,json:() => new Promise((_,reject) => signal.addEventListener("abort",() => reject(new Error("aborted"))))};
  }}),/请求超时.*2 次/);
  assert.equal(calls,2);
});
test("server 503 is retried but a missing sample 404 is not", async () => {
  for (const [status,expected] of [[503,3],[404,1]]) {
    let calls=0;
    await assert.rejects(requestData("/fixture",{wait:immediate,fetchImpl:async () => {
      calls++; return new Response("",{status});
    }}),new RegExp(`HTTP ${status}`));
    assert.equal(calls,expected);
  }
});
test("switching samples cancels the prior request without retrying it", async () => {
  const external=new AbortController(); let calls=0;
  const pending=requestData("/fixture",{signal:external.signal,wait:immediate,fetchImpl:(_, {signal}) => {
    calls++;
    return new Promise((_,reject) => signal.addEventListener("abort",() => reject(new Error("aborted"))));
  }});
  external.abort();
  await assert.rejects(pending,{name:"AbortError"});
  assert.equal(calls,1);
});
test("a truncated JSON response can recover on the next request", async () => {
  let calls=0;
  const result=await requestData("/fixture",{wait:immediate,fetchImpl:async () =>
    new Response(++calls===1 ? '{"items":' : '{"items":[]}')});
  assert.equal(calls,2); assert.deepEqual(result.body.items,[]);
});
