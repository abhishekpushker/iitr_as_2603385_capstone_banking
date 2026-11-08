// The UI's client for the proxied API (`/api/...` on the ui container). Run by `make test-ui`.
import assert from "node:assert/strict";
import { test } from "node:test";

import { ask, deleteSession, readiness } from "../public/js/api.js";

function fakeFetch(handler) {
  const calls = [];
  const fn = async (url, init = {}) => {
    calls.push({ url, init });
    return handler(url, init);
  };
  fn.calls = calls;
  return fn;
}

const json = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
});

test("ask posts the message and session id as JSON and returns the response", async () => {
  const fetch = fakeFetch(() => json(200, { answer: "a", trace_id: "t" }));
  const result = await ask("What is EMI?", "ui-abcdefghij", { fetch });
  assert.deepEqual(result, { ok: true, data: { answer: "a", trace_id: "t" } });
  const [{ url, init }] = fetch.calls;
  assert.equal(url, "/api/v1/ask");
  assert.equal(init.method, "POST");
  assert.equal(init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(init.body), { message: "What is EMI?", session_id: "ui-abcdefghij" });
  assert.equal(init.headers["X-API-Key"], undefined); // the proxy adds the key, never the browser
});

test("ask maps the API error envelope", async () => {
  const fetch = fakeFetch(() =>
    json(413, { error_code: "BUDGET_EXCEEDED", message: "the request exceeds the token budget", trace_id: "t9" }),
  );
  assert.deepEqual(await ask("x", "ui-abcdefghij", { fetch }), {
    ok: false,
    error: { code: "BUDGET_EXCEEDED", message: "the request exceeds the token budget", trace_id: "t9", status: 413 },
  });
});

test("ask maps a non-JSON error (e.g. the proxy's own 502)", async () => {
  const fetch = fakeFetch(() => ({
    ok: false,
    status: 502,
    json: async () => {
      throw new SyntaxError("not json");
    },
  }));
  const result = await ask("x", "ui-abcdefghij", { fetch });
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "HTTP_502");
  assert.equal(result.error.status, 502);
});

test("ask maps a network failure", async () => {
  const fetch = fakeFetch(() => {
    throw new TypeError("Failed to fetch");
  });
  const result = await ask("x", "ui-abcdefghij", { fetch });
  assert.deepEqual(result, {
    ok: false,
    error: { code: "NETWORK_ERROR", message: "The agent could not be reached.", trace_id: null, status: 0 },
  });
});

test("deleteSession sends DELETE for the encoded id", async () => {
  const fetch = fakeFetch(() => ({ ok: true, status: 204, json: async () => null }));
  assert.equal(await deleteSession("ui-abcdefghij", { fetch }), true);
  assert.equal(fetch.calls[0].url, "/api/v1/sessions/ui-abcdefghij");
  assert.equal(fetch.calls[0].init.method, "DELETE");
});

test("deleteSession never throws", async () => {
  const fetch = fakeFetch(() => {
    throw new TypeError("offline");
  });
  assert.equal(await deleteSession("ui-abcdefghij", { fetch }), false);
});

test("readiness reports ready only on 200 + status ready", async () => {
  assert.equal(await readiness({ fetch: fakeFetch(() => json(200, { status: "ready", checks: {} })) }), true);
  assert.equal(await readiness({ fetch: fakeFetch(() => json(503, { status: "not_ready", checks: {} })) }), false);
  assert.equal(
    await readiness({
      fetch: fakeFetch(() => {
        throw new TypeError("offline");
      }),
    }),
    false,
  );
});
