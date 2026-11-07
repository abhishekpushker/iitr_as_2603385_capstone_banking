"""Structured logging over the network (§ARCH 7.17): N mixed requests (HTTP and WS, error paths
included) → exactly N `event=request` lines, unique trace ids, schema-valid, no raw canary."""

from __future__ import annotations

import json
from collections import Counter

import httpx
from websockets.sync.client import connect

from credagent.observability.logging import LogEntry
from tests.e2e.conftest import CANARIES, CANARY_TEXT, USER, WS_URL, log_lines, trace


def test_n_mixed_requests_give_exactly_n_valid_request_lines(client: httpx.Client) -> None:
    sent: list[str] = []

    def http(method: str, path: str, **kw: object) -> httpx.Response:
        tid = trace()
        sent.append(tid)
        headers = {**kw.pop("headers", {}), "X-Trace-Id": tid}  # type: ignore[dict-item]
        r = client.request(method, path, headers=headers, **kw)  # type: ignore[arg-type]
        assert r.headers["X-Trace-Id"] == tid
        return r

    http("GET", "/healthz")
    http("GET", "/readyz")
    http("POST", "/v1/ask", headers=USER, json={"message": CANARY_TEXT})
    http("POST", "/v1/ask", headers={}, json={"message": "hi"})  # 401
    http("POST", "/v1/ask", headers=USER, json={"message": "", "x": CANARIES[0]})  # 422
    http("POST", "/v1/ask", headers=USER, content=b"x" * 70_000)  # 413 payload
    http("GET", "/v1/stats", headers=USER)
    http("DELETE", "/v1/sessions/session-log-1", headers=USER)
    http("GET", "/does-not-exist")  # 404
    with connect(WS_URL, additional_headers=USER) as ws:
        json.loads(ws.recv())  # the session frame is not a request
        for frame in (
            json.dumps({"type": "ask", "message": CANARY_TEXT}),
            "{bad json",
            '{"type":"reset"}',
        ):
            ws.send(frame)
            sent.append(json.loads(ws.recv())["trace_id"])

    mine = [x for x in log_lines() if x["event"] == "request" and x["trace_id"] in set(sent)]
    assert len(mine) == len(sent) == 12
    assert Counter(x["trace_id"] for x in mine) == Counter(sent)  # one line each, unique ids
    entries = [LogEntry.model_validate(x) for x in mine]
    assert {e.transport for e in entries} == {"http", "ws"}
    assert {e.outcome for e in entries} >= {"answered", "unauthorized", "validation_error"}
    assert all(e.timings_ms.get("total") is not None for e in entries)
    raw = "\n".join(json.dumps(x, ensure_ascii=False) for x in log_lines())
    assert all(c not in raw for c in CANARIES)
