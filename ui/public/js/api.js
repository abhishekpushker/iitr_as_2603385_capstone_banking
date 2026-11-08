// Client for the API as the ui container proxies it under /api/ (D-058). The proxy adds the
// X-API-Key header; the browser never holds a key.

const BASE = "/api";

function defaults(options) {
  return { fetch: options.fetch ?? globalThis.fetch.bind(globalThis) };
}

async function errorFrom(response) {
  try {
    const body = await response.json();
    if (body && typeof body.error_code === "string") {
      return { code: body.error_code, message: body.message, trace_id: body.trace_id ?? null, status: response.status };
    }
  } catch {
    // not JSON: the proxy's own error page
  }
  return { code: `HTTP_${response.status}`, message: `The service answered ${response.status}.`, trace_id: null, status: response.status };
}

/** POST /v1/ask → `{ ok: true, data }` or `{ ok: false, error: { code, message, trace_id, status } }`. */
export async function ask(message, sessionId, options = {}) {
  const { fetch } = defaults(options);
  try {
    const response = await fetch(`${BASE}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal: options.signal,
    });
    if (!response.ok) return { ok: false, error: await errorFrom(response) };
    return { ok: true, data: await response.json() };
  } catch {
    return { ok: false, error: { code: "NETWORK_ERROR", message: "The agent could not be reached.", trace_id: null, status: 0 } };
  }
}

/** DELETE /v1/sessions/{id}: forget the server-side memory. Never throws. */
export async function deleteSession(sessionId, options = {}) {
  const { fetch } = defaults(options);
  try {
    const response = await fetch(`${BASE}/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    return response.ok;
  } catch {
    return false;
  }
}

/** GET /readyz → true when the agent (services, Chroma, both collections) is ready. */
export async function readiness(options = {}) {
  const { fetch } = defaults(options);
  try {
    const response = await fetch(`${BASE}/readyz`);
    if (!response.ok) return false;
    const body = await response.json();
    return body?.status === "ready";
  } catch {
    return false;
  }
}
