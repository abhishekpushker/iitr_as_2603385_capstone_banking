"""The FastAPI app in-process (§ARCH 7.16, 7.17): routes, auth, error mapping, middleware and the
WS loop over the real `Orchestrator` with the fake crew of `test_orchestrator` (no CrewAI, no
Chroma server). The same behaviour is exercised over the network in tests/e2e."""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from credagent.api import app as app_module
from credagent.api.app import assert_single_worker, create_app, initialize
from credagent.api.middleware import accept_trace_id, new_trace_id
from credagent.api.routes_ws import send_frame
from credagent.cache.response_cache import ResponseCache
from credagent.governance.budget import PreflightResult
from credagent.guardrails.pii import PiiMasker
from credagent.kb.version import KbVersion
from credagent.observability.logging import LogEntry
from credagent.observability.metrics import Metrics
from credagent.pipeline.documents import DocumentService
from credagent.pipeline.factory import Services
from credagent.pipeline.orchestrator import RecordCollector
from credagent.rag.store import VectorStore
from credagent.settings import Settings, get_settings
from tests.unit.test_documents import TEXT, HashEncoder
from tests.unit.test_orchestrator import ATTACK, CANARIES, RAW_PII, T1, T2, FakeCrew, make

_n = itertools.count()
S = get_settings()
USER = {"X-API-Key": S.api_key.get_secret_value()}
ADMIN = {"X-API-Key": S.admin_api_key.get_secret_value()}
MASKER = PiiMasker()


class TooBig:
    def preflight(self, masked_text: str) -> PreflightResult:
        return PreflightResult(exceeds=True, projected_tokens=99_999, budget_tokens=1)


class Boom:
    def __call__(self, ctx: Any, query: str) -> Any:
        raise RuntimeError(f"crew exploded on {CANARIES[0]}")


@dataclass
class Api:
    client: TestClient
    services: Services
    log_path: Path

    def lines(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text(encoding="utf-8").splitlines()]

    def requests(self) -> list[dict[str, Any]]:
        return [x for x in self.lines() if x["event"] == "request"]


def build_services(crew: Any = None, *, metrics: Metrics | None = None, **parts: Any) -> Services:
    collector = RecordCollector()
    metrics = Metrics() if metrics is None else metrics
    orch, _, _, store = make(crew, sink=collector, metrics=metrics, **parts)
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vectors = VectorStore(client)
    n = next(_n)
    names = {"fixed": f"ut_api_fixed_{n}", "sentence": f"ut_api_sentence_{n}"}
    for name in names.values():
        vectors.collection(name)
    kb = KbVersion("v0")
    cache: ResponseCache[Any] = ResponseCache(8, 60, metrics=metrics)
    docs = DocumentService(vectors=vectors, encoder=HashEncoder(), collections=names, kb=kb,
                           cache=cache)  # fmt: skip
    return Services(orchestrator=orch, sessions=store, metrics=metrics, cache=cache,
                    vectors=vectors, collections=names, kb=kb, documents=docs,
                    collector=collector)  # fmt: skip


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return S.model_copy(update={"log_path": tmp_path / "requests.jsonl"})


def _api(settings: Settings, services: Services) -> Iterator[Api]:
    with TestClient(create_app(settings, services), raise_server_exceptions=False) as client:
        yield Api(client, services, settings.log_path)


@pytest.fixture
def api(tmp_settings: Settings) -> Iterator[Api]:
    yield from _api(tmp_settings, build_services())


# ------------------------------------------------------------------ health, ask, trace ids
def test_health_ready_and_the_openapi_models(api: Api) -> None:
    assert api.client.get("/healthz").json() == {"status": "ok"}
    ready = api.client.get("/readyz")
    assert ready.status_code == 200 and all(ready.json()["checks"].values())
    schemas = api.client.get("/openapi.json").json()["components"]["schemas"]
    assert {"AskRequest", "AskResponse", "AddDocumentRequest", "AddDocumentResponse",
            "StatsResponse", "ErrorResponse"} <= set(schemas)  # fmt: skip


def test_ask_creates_a_session_carries_memory_and_logs_one_masked_line(api: Api) -> None:
    trace = new_trace_id(MASKER)
    r1 = api.client.post(
        "/v1/ask", json={"message": RAW_PII}, headers={**USER, "X-Trace-Id": trace}
    )
    assert r1.status_code == 200 and r1.headers["X-Trace-Id"] == trace == r1.json()["trace_id"]
    sid = r1.json()["session_id"]
    r2 = api.client.post("/v1/ask", json={"message": T2, "session_id": sid}, headers=USER)
    assert r2.json()["application"] is None and r2.json()["intent"] != "clarify"
    lines = api.requests()
    assert [x["trace_id"] for x in lines] == [trace, r2.json()["trace_id"]]
    assert [x["history_length"] for x in lines] == [2, 4]
    for x in lines:
        LogEntry.model_validate(x)
        assert all(isinstance(n, int) for pair in x["llm_usage"] for n in pair)
    raw = api.log_path.read_text(encoding="utf-8")
    assert all(c not in raw for c in CANARIES)


def test_inbound_trace_ids_are_validated() -> None:
    assert accept_trace_id("0123abcd-ef", MASKER) == "0123abcd-ef"
    for bad in (None, "NOT-HEX!", "abc", "2345-6789-0123-aaaa"):  # the last is an Aadhaar shape
        assert accept_trace_id(bad, MASKER) is None
    fresh = new_trace_id(MASKER)
    assert accept_trace_id(fresh, MASKER) == fresh


def test_an_invalid_inbound_trace_id_is_replaced(api: Api) -> None:
    r = api.client.get("/healthz", headers={"X-Trace-Id": "<script>"})
    assert r.headers["X-Trace-Id"] != "<script>"
    assert accept_trace_id(r.headers["X-Trace-Id"], MASKER) == r.headers["X-Trace-Id"]


# ------------------------------------------------------------------ auth and errors
def test_auth_missing_wrong_and_non_admin(api: Api) -> None:
    for headers in ({}, {"X-API-Key": "wrong"}):
        r = api.client.post("/v1/ask", json={"message": T1}, headers=headers)
        assert (r.status_code, r.json()["error_code"]) == (401, "UNAUTHORIZED")
    r = api.client.post("/v1/documents", headers=USER, json={
        "doc_id": "kb-x-doc", "title": "Title", "topic": "loans", "text": TEXT})  # fmt: skip
    assert (r.status_code, r.json()["error_code"]) == (403, "FORBIDDEN")
    assert api.client.get("/v1/stats", headers=ADMIN).status_code == 200  # admin ⊇ user
    assert [x["outcome"] for x in api.requests()][:3] == ["unauthorized"] * 3


def test_validation_error_never_echoes_input(api: Api) -> None:
    body = {"message": "", "session_id": CANARIES[1], "extra": CANARIES[0]}
    r = api.client.post("/v1/ask", json=body, headers=USER)
    assert (r.status_code, r.json()["error_code"]) == (422, "VALIDATION_ERROR")
    assert all(c not in r.text for c in CANARIES)
    assert {tuple(e["loc"]) for e in r.json()["errors"]} >= {("body", "message")}
    pan_session = api.client.post("/v1/ask", json={"message": T1, "session_id": CANARIES[0]},
                                  headers=USER)  # fmt: skip
    assert pan_session.status_code == 422 and CANARIES[0] not in pan_session.text
    assert api.client.delete(f"/v1/sessions/{CANARIES[0]}", headers=USER).status_code == 422
    assert api.requests()[-1]["outcome"] == "validation_error"


def test_not_found_uses_the_error_envelope(api: Api) -> None:
    r = api.client.get("/nope")
    assert (r.status_code, r.json()["error_code"]) == (404, "NOT_FOUND")


def test_body_cap_by_header_and_by_stream(api: Api) -> None:
    big = b'{"message": "' + b"a" * (S.max_body_bytes + 10) + b'"}'
    r = api.client.post(
        "/v1/ask", content=big, headers={**USER, "Content-Type": "application/json"}
    )
    assert (r.status_code, r.json()["error_code"]) == (413, "PAYLOAD_TOO_LARGE")
    chunks = iter([big[: S.max_body_bytes // 2], big[S.max_body_bytes // 2 :]])
    r = api.client.post(
        "/v1/ask", content=chunks, headers={**USER, "Content-Type": "application/json"}
    )
    assert (r.status_code, r.json()["error_code"]) == (413, "PAYLOAD_TOO_LARGE")
    assert [x["outcome"] for x in api.requests()] == ["validation_error"] * 2


@pytest.mark.parametrize(
    ("parts", "status", "code", "outcome"),
    [
        ({"preflight": TooBig()}, 413, "BUDGET_EXCEEDED", "budget_rejected"),
        ({"crew": FakeCrew(raw="{}")}, 500, "SCHEMA_VIOLATION", "error"),
        ({"crew": Boom()}, 500, "INTERNAL_ERROR", "error"),
    ],
)
def test_pipeline_failures_map_to_codes_without_a_stack(
    tmp_settings: Settings, parts: dict[str, Any], status: int, code: str, outcome: str
) -> None:
    for api in _api(tmp_settings, build_services(**parts)):
        r = api.client.post("/v1/ask", json={"message": T1}, headers=USER)
        assert (r.status_code, r.json()["error_code"]) == (status, code)
        assert "Traceback" not in r.text and CANARIES[0] not in r.text
        (line,) = api.requests()
        assert (line["outcome"], line["error_code"], line["status_code"]) == (outcome, code, status)
        assert CANARIES[0] not in api.log_path.read_text(encoding="utf-8")


def test_retrieval_unavailable_is_503_on_http_and_an_error_frame_on_ws(
    tmp_settings: Settings,
) -> None:
    """BUG-007: Chroma down after startup → 503 `RETRIEVAL_UNAVAILABLE`, logged as `error`."""
    policy = "What is the prepayment penalty on a fixed-rate personal loan?"
    for api in _api(tmp_settings, build_services(FakeCrew(rag_down=True))):
        r = api.client.post("/v1/ask", json={"message": policy}, headers=USER)
        assert (r.status_code, r.json()["error_code"]) == (503, "RETRIEVAL_UNAVAILABLE")
        with api.client.websocket_connect("/v1/ws/chat", headers=dict(USER)) as ws:
            ws.receive_json()  # session frame
            ws.send_json({"type": "ask", "message": policy})
            frame = ws.receive_json()
            assert (frame["type"], frame["error_code"]) == ("error", "RETRIEVAL_UNAVAILABLE")
        lines = api.requests()
        assert [(x["outcome"], x["status_code"]) for x in lines] == [("error", 503)] * 2


def test_pipeline_timeout_is_504(tmp_settings: Settings) -> None:
    gate = threading.Event()
    for api in _api(tmp_settings, build_services(FakeCrew(block=gate), timeout_s=0.05)):
        r = api.client.post("/v1/ask", json={"message": T1}, headers=USER)
        gate.set()
        assert (r.status_code, r.json()["error_code"]) == (504, "PIPELINE_TIMEOUT")
        assert api.requests()[0]["outcome"] == "timeout"


# ------------------------------------------------------------------ documents, sessions, stats
def _doc(doc_id: str = "kb-lunar-rover", text: str = TEXT, **over: Any) -> dict[str, Any]:
    return {"doc_id": doc_id, "title": "Lunar rover loan", "topic": "loans", "text": text, **over}


def test_documents_add_duplicate_overwrite_injection_and_sentence_rules(api: Api) -> None:
    api.services.cache.put("stale", "x")
    r = api.client.post("/v1/documents", json=_doc(), headers=ADMIN)
    assert r.status_code == 201, r.text
    assert r.json()["chunks"]["sentence"] == 3 and r.json()["kb_version"] == api.services.kb.current
    assert len(api.services.cache) == 0
    dup = api.client.post("/v1/documents", json=_doc(), headers=ADMIN)
    assert (dup.status_code, dup.json()["error_code"]) == (409, "DUPLICATE_DOCUMENT")
    assert (
        api.client.post("/v1/documents", json=_doc(overwrite=True), headers=ADMIN).status_code
        == 201
    )
    evil = api.client.post(
        "/v1/documents", json=_doc("kb-evil-doc", f"{ATTACK} {TEXT}"), headers=ADMIN
    )
    assert (evil.status_code, evil.json()["error_code"]) == (422, "INJECTION_DETECTED")
    one = api.client.post("/v1/documents", json=_doc("kb-one-doc", "x" * 60), headers=ADMIN)
    assert (one.status_code, one.json()["error_code"]) == (422, "VALIDATION_ERROR")
    outcomes = ["answered", "validation_error", "answered", "blocked_injection", "validation_error"]
    assert [x["outcome"] for x in api.requests()] == outcomes


def test_delete_session_resets_memory_and_stats_report_counters(api: Api) -> None:
    sid = "session-delete-1"
    api.client.post("/v1/ask", json={"message": T1, "session_id": sid}, headers=USER)
    assert api.services.sessions.history_length(sid) == 2
    assert api.client.delete(f"/v1/sessions/{sid}", headers=USER).status_code == 204
    assert api.services.sessions.history_length(sid) == 0
    api.services.metrics.incr("cache_hits")
    stats = api.client.get("/v1/stats", headers=USER).json()
    assert stats["cache_hits"] == 1 and stats["kb_version"] == "v0"
    assert {"retrieval_calls", "generation_calls", "cache_misses", "llm_calls"} <= set(stats)


# ------------------------------------------------------------------ WebSocket
def test_ws_session_ask_bad_json_reset_and_errors(tmp_settings: Settings) -> None:
    for api in _api(tmp_settings, build_services()):
        with api.client.websocket_connect("/v1/ws/chat", headers=dict(USER)) as ws:
            sid = ws.receive_json()
            assert sid["type"] == "session"
            ws.send_text("{not json")
            bad = ws.receive_json()
            assert (bad["type"], bad["error_code"]) == ("error", "VALIDATION_ERROR")
            ws.send_json({"type": "ask", "message": T1})
            answer = ws.receive_json()
            assert answer["type"] == "answer" and answer["session_id"] == sid["session_id"]
            ws.send_json({"type": "reset"})
            assert ws.receive_json()["type"] == "reset_ok"
            assert api.services.sessions.history_length(sid["session_id"]) == 0
        lines = api.requests()
        assert [x["transport"] for x in lines] == ["ws"] * 3
        assert [x["outcome"] for x in lines] == ["validation_error", "answered", "answered"]


def test_ws_pipeline_errors_become_error_frames(tmp_settings: Settings) -> None:
    for api in _api(tmp_settings, build_services(preflight=TooBig())):
        with api.client.websocket_connect("/v1/ws/chat", headers=dict(USER)) as ws:
            ws.receive_json()
            ws.send_json({"type": "ask", "message": T1})
            assert ws.receive_json()["error_code"] == "BUDGET_EXCEEDED"
    for api in _api(tmp_settings, build_services(Boom())):
        with api.client.websocket_connect("/v1/ws/chat", headers=dict(USER)) as ws:
            ws.receive_json()
            ws.send_json({"type": "ask", "message": T1})
            frame = ws.receive_json()
            assert frame["error_code"] == "INTERNAL_ERROR" and CANARIES[0] not in json.dumps(frame)
            ws.send_json({"type": "ask", "message": T1})  # the connection survived
            assert ws.receive_json()["type"] == "error"


def test_ws_bad_auth_closes_1008_before_accept(api: Api) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        api.client.websocket_connect("/v1/ws/chat", headers={"X-API-Key": "wrong"}),
    ):
        pass
    assert exc.value.code == 1008


class GoneSocket:
    """A client that left: every send raises what the contract test pinned."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def send_json(self, data: Any) -> None:
        raise self.exc


@pytest.mark.anyio
@pytest.mark.parametrize("exc", [WebSocketDisconnect(1006), WebSocketDisconnected("closed")])
async def test_send_after_close_is_swallowed(exc: BaseException) -> None:
    assert await send_frame(GoneSocket(exc), {"type": "answer"}) is False  # type: ignore[arg-type]


# ------------------------------------------------------------------ startup
def test_not_ready_until_services_are_built(tmp_settings: Settings) -> None:
    def never() -> Services:
        raise ConnectionError("chroma down")

    app = create_app(tmp_settings, build=never)
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "not_ready"
        ask = client.post("/v1/ask", json={"message": T1}, headers=USER)
        assert (ask.status_code, ask.json()["error_code"]) == (503, "NOT_READY")
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/v1/ws/chat", headers=dict(USER)),
        ):
            pass
        assert exc.value.code == 1013


@pytest.mark.anyio
async def test_initialize_retries_until_built(tmp_settings: Settings) -> None:
    services, calls = build_services(), []

    def flaky() -> Services:
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("not yet")
        return services

    app: FastAPI = create_app(tmp_settings, build=flaky)
    await initialize(app, flaky, retry_s=0)
    assert app.state.services is services and len(calls) == 2
    events = [json.loads(x)["event"] for x in tmp_settings.log_path.read_text().splitlines()]
    assert events == ["startup_error", "ready"]


def test_single_worker_is_asserted(monkeypatch: pytest.MonkeyPatch) -> None:
    """P14: `--workers > 1` runs the app in a child process; refuse to start there."""
    assert_single_worker()
    monkeypatch.setattr(app_module.multiprocessing, "parent_process", SimpleNamespace)
    with pytest.raises(RuntimeError, match="single worker"):
        assert_single_worker()


def test_module_app_is_created_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_settings: Settings
) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: tmp_settings)
    assert isinstance(app_module.app, FastAPI)
    with pytest.raises(AttributeError):
        _ = app_module.nothing_here


def test_stats_count_llm_calls_per_persona(tmp_settings: Settings) -> None:
    metrics = Metrics()
    services = build_services(FakeCrew(charge=5), metrics=metrics)
    for api in _api(tmp_settings, services):
        api.client.post("/v1/ask", json={"message": T1}, headers=USER)
        calls = api.client.get("/v1/stats", headers=USER).json()["llm_calls"]
        assert calls == {"composer": 1, "editor": 1, "reviewer": 1}  # crew (fake) + review
