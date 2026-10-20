// Read the entire response inside the timeout: a connection can fail after headers.
const abortError = () => Object.assign(new Error("请求已取消"), { name: "AbortError" });

export async function requestData(url, {
  kind = "json", signal, timeoutMs = 15000, retries = 2,
  fetchImpl = fetch, wait = ms => new Promise(resolve => setTimeout(resolve, ms)),
} = {}) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (signal?.aborted) throw abortError();
    const controller = new AbortController();
    const cancel = () => controller.abort();
    signal?.addEventListener("abort", cancel, { once: true });
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
    let failure, retryable = true;
    try {
      const response = await fetchImpl(url, { signal: controller.signal, cache: "no-store" });
      if (!response.ok) {
        retryable = [408, 425, 429].includes(response.status) || response.status >= 500;
        throw new Error(`服务请求失败（HTTP ${response.status}）`);
      }
      const body = kind === "json" ? await response.json() : await response.arrayBuffer();
      if (signal?.aborted) throw abortError();
      return { body, headers: response.headers };
    } catch (error) {
      if (signal?.aborted) throw abortError();
      failure = timedOut ? new Error("数据请求超时")
        : error instanceof TypeError ? new Error("网络连接中断")
        : error instanceof SyntaxError ? new Error("服务返回的数据不完整")
        : error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
    }
    if (!retryable || attempt === retries) {
      if (attempt) failure.message += `，已尝试 ${attempt + 1} 次`;
      throw failure;
    }
    await wait(400 * (attempt + 1));
  }
}
