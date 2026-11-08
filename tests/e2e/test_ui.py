"""The chat UI container (D-058) against `ui-offline` → `api-offline` on the backend network.

The `ui` service is nginx: it serves the static page and proxies an allow-list of API routes
under `/api/`, adding the user `X-API-Key` itself so the key never reaches the browser.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import httpx
import pytest

from tests.e2e.conftest import CANARIES, CANARY_TEXT, S, log_lines

UI = "http://ui-offline:8080"
POLICY_Q = "Is there a prepayment penalty on a floating-rate home loan?"
SCRIPTS = ("/js/app.js", "/js/api.js", "/js/history.js", "/js/pii.js")


@pytest.fixture(scope="module")
def ui() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=UI, timeout=180) as c:
        yield c


def session_id() -> str:
    return "ui-e2e-" + uuid.uuid4().hex.translate(str.maketrans("0123456789", "ghijklmnop"))


# ------------------------------------------------------------------ static page
def test_the_page_is_served_with_a_strict_content_security_policy(ui: httpx.Client) -> None:
    r = ui.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Cred Loan Assistant</title>" in r.text
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "server" not in r.headers or "nginx/" not in r.headers["server"]


@pytest.mark.parametrize("path", SCRIPTS)
def test_scripts_are_served_as_javascript(ui: httpx.Client, path: str) -> None:
    r = ui.get(path)
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


@pytest.mark.parametrize("path", ("/", "/styles.css", *SCRIPTS))
def test_no_api_key_reaches_the_browser(ui: httpx.Client, path: str) -> None:
    body = ui.get(path).text
    assert S.api_key.get_secret_value() not in body
    assert S.admin_api_key.get_secret_value() not in body


def test_the_page_loads_nothing_from_another_origin(ui: httpx.Client) -> None:
    for path in ("/", "/styles.css", *SCRIPTS):
        body = ui.get(path).text
        assert "http://" not in body and "https://" not in body, path


def test_health(ui: httpx.Client) -> None:
    r = ui.get("/healthz")
    assert r.status_code == 200


# ------------------------------------------------------------------ the proxy
def test_ready_through_the_proxy(ui: httpx.Client) -> None:
    r = ui.get("/api/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_ask_through_the_proxy_without_a_key_from_the_browser(ui: httpx.Client) -> None:
    r = ui.post("/api/v1/ask", json={"message": POLICY_Q})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intent"] == "policy"
    assert body["sources"]


def test_a_key_sent_by_the_browser_is_replaced_by_the_proxy(ui: httpx.Client) -> None:
    r = ui.post(
        "/api/v1/ask",
        json={"message": POLICY_Q},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 200, r.text


def test_session_memory_carries_across_turns_through_the_proxy(ui: httpx.Client) -> None:
    sid = session_id()
    first = ui.post("/api/v1/ask", json={"message": "What is the status of application LA-0037?",
                                         "session_id": sid})  # fmt: skip
    assert first.status_code == 200, first.text
    assert first.json()["application"]["record_id"] == "LA-0037"
    second = ui.post("/api/v1/ask", json={"message": "What credit score does it need?",
                                          "session_id": sid})  # fmt: skip
    assert second.status_code == 200, second.text
    assert second.json()["session_id"] == sid
    assert "kb-01-loan-eligibility" in {s["doc_id"] for s in second.json()["sources"]}
    assert ui.delete(f"/api/v1/sessions/{sid}").status_code == 204


def test_the_pii_canaries_are_masked_on_the_proxied_path(ui: httpx.Client) -> None:
    r = ui.post("/api/v1/ask", json={"message": CANARY_TEXT, "session_id": session_id()})
    assert r.status_code == 200, r.text
    trace_id = r.json()["trace_id"]
    assert not any(c in r.text for c in CANARIES)
    lines = [x for x in log_lines() if x.get("trace_id") == trace_id]
    assert lines
    assert not any(c in str(line) for line in lines for c in CANARIES)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/documents"),
        ("GET", "/api/v1/stats"),
        ("GET", "/api/docs"),
        ("GET", "/api/openapi.json"),
        ("GET", "/api/v1/ws/chat"),
        ("GET", "/api/v1/ask"),
        ("POST", "/api/v1/sessions/ui-abcdefghij"),
    ],
)
def test_only_the_allow_listed_routes_are_proxied(ui: httpx.Client, method: str, path: str) -> None:
    r = ui.request(method, path, json={})
    assert r.status_code in {404, 405}, (method, path, r.status_code)


def test_an_oversized_body_is_refused_at_the_proxy(ui: httpx.Client) -> None:
    r = ui.post("/api/v1/ask", content=b"x" * (S.max_body_bytes + 1),
                headers={"Content-Type": "application/json"})  # fmt: skip
    assert r.status_code == 413
