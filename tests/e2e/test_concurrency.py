"""Edge (§ARCH 8): 20 concurrent `/v1/ask` across 10 sessions → no 5xx, no record id from one
session in another's answer or tool calls, and each session's history holds both turns."""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

from tests.e2e.conftest import USER, log_lines, trace

RECORD_IDS = [f"LA-{i:04d}" for i in range(1, 11)]
TEMPLATES = ("What is the status of application {}?", "Has application {} been approved yet?")


def test_twenty_concurrent_asks_over_ten_sessions(client: httpx.Client) -> None:
    run = uuid.uuid4().hex[:8]
    jobs = [(f"conc-{run}-{i}", rid, t.format(rid), trace())
            for i, rid in enumerate(RECORD_IDS) for t in TEMPLATES]  # fmt: skip

    def ask(job: tuple[str, str, str, str]) -> httpx.Response:
        session, _, message, tid = job
        return client.post("/v1/ask", json={"message": message, "session_id": session},
                           headers={**USER, "X-Trace-Id": tid})  # fmt: skip

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        responses = list(pool.map(ask, jobs))

    assert [r.status_code for r in responses] == [200] * 20, [r.text for r in responses]
    for (session, rid, _, _), r in zip(jobs, responses, strict=True):
        body = r.json()
        assert body["session_id"] == session
        assert body["application"]["record_id"] == rid
        seen = set(re.findall(r"LA-\d{4}", r.text))
        assert seen == {rid}, (session, seen)  # answer, tools, args: only this session's id
    lines = {x["trace_id"]: x for x in log_lines() if x["event"] == "request"}
    for i in range(10):
        session = f"conc-{run}-{i}"
        lengths = sorted(lines[tid]["history_length"] for s, _, _, tid in jobs if s == session)
        assert lengths == [2, 4], (session, lengths)
