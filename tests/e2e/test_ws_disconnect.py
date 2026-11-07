"""High-risk edge (§ARCH 7.16): client A asks and disconnects at once; client B, connected at the
same time, completes three turns with memory carried. The server stays healthy, A's pipeline
finishes and is logged as `ws_disconnect_mid_turn`, and B has exactly three request lines."""

from __future__ import annotations

import json

import httpx
from websockets.sync.client import connect

from tests.e2e.conftest import USER, WS_URL, log_lines, wait_for_log

A_ASK = "Is there a prepayment penalty on a floating-rate home loan?"
B_TURNS = (
    "What is the status of application LA-0007?",
    "What are the eligibility criteria for it?",
    "And does it carry a prepayment penalty?",
)


def test_disconnect_mid_turn_leaves_a_concurrent_client_and_the_server_intact(
    client: httpx.Client,
) -> None:
    with connect(WS_URL, additional_headers=USER) as b:
        b_session = json.loads(b.recv())["session_id"]
        with connect(WS_URL, additional_headers=USER) as a:
            a_session = json.loads(a.recv())["session_id"]
            a.send(json.dumps({"type": "ask", "message": A_ASK}))
        # A is gone while its pipeline runs; B now talks over the same server.
        answers = []
        for turn in B_TURNS:
            b.send(json.dumps({"type": "ask", "message": turn}))
            answers.append(json.loads(b.recv()))

    assert [x["type"] for x in answers] == ["answer"] * 3
    assert answers[0]["application"]["record_id"] == "LA-0007"
    assert all(x["intent"] != "clarify" for x in answers[1:])  # "it" resolved from memory
    assert client.get("/healthz").json() == {"status": "ok"}

    assert wait_for_log(
        lambda x: x["event"] == "ws_disconnect_mid_turn" and x.get("session_id") == a_session
    ), "A's mid-turn disconnect was never logged"
    lines = log_lines()
    b_requests = [x for x in lines if x["event"] == "request" and x["session_id"] == b_session]
    assert len(b_requests) == 3 and [x["history_length"] for x in b_requests] == [2, 4, 6]
    a_requests = [x for x in lines if x["event"] == "request" and x["session_id"] == a_session]
    assert len(a_requests) == 1 and a_requests[0]["outcome"] == "answered"  # it finished
