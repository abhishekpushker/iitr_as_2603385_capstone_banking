"""Request middleware (§ARCH 7.16, 7.17), pure ASGI, outermost first:

1. `TraceLogMiddleware`: accepts an inbound `X-Trace-Id` only if it matches
   `^[a-f0-9-]{8,64}$` (and the PII masker leaves it unchanged), else makes one; echoes it in
   the `X-Trace-Id` response header; turns any unhandled exception into a 500 `INTERNAL_ERROR`
   with no stack (the masked stack goes to the log); writes **exactly one** `event="request"`
   line per request. The line is written just before the final body chunk is sent, so a client
   that has its response can already read the line; a `finally` writes it if that never happens.
2. `BodyLimitMiddleware`: 413 `PAYLOAD_TOO_LARGE` above `MAX_BODY_BYTES`, by `Content-Length` or
   by counting a streamed body (buffered, then replayed to the app).
3. `TimingMiddleware`: the handler's latency, for the log line.

WebSocket scopes pass straight through; the WS loop logs its own frames with `request_fields`.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Final

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from credagent.api.errors import error_json
from credagent.guardrails.pii import PiiMasker, PiiType
from credagent.observability.logging import EventLog, LogOutcome, truncate
from credagent.pipeline.orchestrator import TurnRecord

TRACE_ID_PATTERN: Final = re.compile(r"^[a-f0-9-]{8,64}$")
OUTCOME_BY_CODE: Final[dict[str, LogOutcome]] = {
    "UNAUTHORIZED": "unauthorized", "FORBIDDEN": "unauthorized",
    "BUDGET_EXCEEDED": "budget_rejected", "PIPELINE_TIMEOUT": "timeout",
    "INJECTION_DETECTED": "blocked_injection", "INTERNAL_ERROR": "error",
    "SCHEMA_VIOLATION": "error", "NOT_READY": "error", "RETRIEVAL_UNAVAILABLE": "error",
}  # fmt: skip


def new_trace_id(masker: PiiMasker) -> str:
    """A UUID4 the PII masker leaves intact (a run of digits could read as an account number,
    and the scrub filter would then rewrite the id in the log)."""
    while True:
        candidate = str(uuid.uuid4())
        if masker.mask(candidate).text == candidate:
            return candidate


def accept_trace_id(value: str | None, masker: PiiMasker) -> str | None:
    if value is None or not TRACE_ID_PATTERN.fullmatch(value):
        return None
    return value if masker.mask(value).text == value else None


@dataclass
class RequestState:
    """What one request accumulates for its log line (`scope["state"]["log"]`)."""

    trace_id: str
    session_id: str | None = None
    record: TurnRecord | None = None
    error_code: str | None = None
    masked_text: str = ""
    pii_counts: dict[PiiType, int] = field(default_factory=dict)
    injection_score: float = 0.0
    latency_ms: float | None = None


def outcome_for(error_code: str | None, record: TurnRecord | None) -> LogOutcome:
    if error_code is not None:
        return OUTCOME_BY_CODE.get(error_code, "validation_error")
    return "answered" if record is None else record.outcome


def request_fields(
    state: RequestState,
    *,
    transport: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
) -> dict[str, Any]:
    """The `LogEntry` fields for one request (HTTP) or frame (WS), from the pipeline's record
    when there is one, else from the masked fields the route filled in."""
    r = state.record
    text, truncated = truncate(state.masked_text if r is None else r.masked_text)
    return {
        "trace_id": state.trace_id, "session_id": state.session_id, "transport": transport,
        "method": method, "path": path, "status_code": status_code,
        "outcome": outcome_for(state.error_code, r),
        "request_text_masked": text, "truncated": truncated,
        "pii_masked": dict(state.pii_counts if r is None else r.pii_counts),
        "injection_score": state.injection_score if r is None else r.injection_score,
        "intent": None if r is None else r.intent,
        "cache_hit": False if r is None else r.cache_hit,
        "tools_invoked": [] if r is None else [f"{t.agent}:{t.tool}" for t in r.tools_invoked],
        "review_approved": None if r is None else r.review_approved,
        "tokens_estimated": 0 if r is None else r.tokens_estimated,
        "cost_estimated_usd": 0.0 if r is None else r.cost_estimated_usd,
        "llm_usage": [] if r is None else [[u.prompt_tokens, u.completion_tokens]
                                          for u in r.llm_usage],
        "history_length": None if r is None else r.history_length,
        "latency_ms": latency_ms,
        "timings_ms": {"total": latency_ms} if r is None else dict(r.timings_ms),
        "error_code": state.error_code if r is None else (state.error_code or r.error_code),
    }  # fmt: skip


def _path(scope: Scope) -> str:
    """The route template when routing matched (no ids in the log), else the raw path."""
    route = scope.get("route")
    return str(getattr(route, "path", scope.get("path", "")))


class TraceLogMiddleware:
    def __init__(self, app: ASGIApp, *, log: EventLog, masker: PiiMasker) -> None:
        self.app = app
        self.log = log
        self.masker = masker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        trace_id = accept_trace_id(headers.get("x-trace-id"), self.masker)
        state = RequestState(trace_id=new_trace_id(self.masker) if trace_id is None else trace_id)
        scope.setdefault("state", {})["log"] = state
        status, started, logged = 500, False, False

        def emit() -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            latency = state.latency_ms
            latency = (time.perf_counter() - start) * 1000 if latency is None else latency
            self.log.request(request_fields(
                state, transport="http", method=scope["method"], path=_path(scope),
                status_code=status, latency_ms=latency,
            ))  # fmt: skip

        async def send_traced(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started, status = True, message["status"]
                MutableHeaders(scope=message)["X-Trace-Id"] = state.trace_id
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                emit()
            await send(message)

        try:
            await self.app(scope, receive, send_traced)
        except Exception:  # noqa: BLE001  (the generic 500 handler: never a stack to the client)
            self.log.event("error", level=logging.ERROR, trace_id=state.trace_id,
                           stack=traceback.format_exc())  # fmt: skip
            if not started:
                await error_json(state, "INTERNAL_ERROR")(scope, receive, send_traced)
            else:
                state.error_code = "INTERNAL_ERROR"
        finally:
            emit()


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope["state"]["log"]
        for key, value in scope["headers"]:
            if key.lower() == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await error_json(state, "PAYLOAD_TOO_LARGE")(scope, receive, send)
                return
        chunks: list[bytes] = []
        size, more = 0, True
        while more:
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await error_json(state, "PAYLOAD_TOO_LARGE")(scope, receive, send)
                return
            chunks.append(chunk)
            more = message.get("more_body", False)
        body, replayed = b"".join(chunks), False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class TimingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state: RequestState = scope["state"]["log"]
        start = time.perf_counter()

        async def send_timed(message: Message) -> None:
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                state.latency_ms = (time.perf_counter() - start) * 1000
            await send(message)

        await self.app(scope, receive, send_timed)
