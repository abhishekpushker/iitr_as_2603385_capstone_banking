"""`WS /v1/ws/chat` (§ARCH 7.16): one session per connection, one log line per client frame.

* Handshake: an unknown `X-API-Key` → close 1008 before accept; not ready yet → close 1013.
* On accept: `{"type": "session", "session_id": …}`.
* Each client frame (ask, reset, or invalid) gets its own trace id and exactly one
  `event="request"` line, written **before** the reply is sent. Invalid JSON or schema → an error
  frame; the connection stays open. An ask runs `Orchestrator.handle` shielded from
  cancellation, so the pipeline (and memory) finish even if the client has gone.
* The client leaving: `receive` → `ws_disconnect`; a failed send after a turn →
  `ws_disconnect_mid_turn`. The send-after-close types are the two the contract test pinned
  (`WebSocketDisconnect`, then `WebSocketDisconnected`). Nothing is re-raised.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

import anyio
from fastapi import APIRouter, WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from credagent.api.auth import role_for
from credagent.api.errors import STATUS, error_payload, safe_errors
from credagent.api.middleware import RequestState, new_trace_id, request_fields
from credagent.api.models import CLIENT_FRAME, AskFrame, FieldError
from credagent.crew.factory import SchemaViolation
from credagent.guardrails.pii import PiiMasker
from credagent.observability.logging import EventLog
from credagent.pipeline.factory import Services
from credagent.pipeline.orchestrator import PipelineError

PATH = "/v1/ws/chat"
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TRY_AGAIN_LATER = 1013
router = APIRouter()


async def send_frame(ws: WebSocket, frame: dict[str, Any]) -> bool:
    """Send; `False` if the client is gone."""
    try:
        await ws.send_json(frame)
    except (WebSocketDisconnect, WebSocketDisconnected):
        return False
    return True


class ChatLoop:
    def __init__(self, ws: WebSocket, services: Services, log: EventLog, masker: PiiMasker) -> None:
        self.ws, self.services, self.log, self.masker = ws, services, log, masker
        self.session_id = new_trace_id(masker)

    async def run(self) -> None:
        await self.ws.accept()
        if not await send_frame(self.ws, {"type": "session", "session_id": self.session_id}):
            self.log.event("ws_disconnect", session_id=self.session_id)
            return
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                self.log.event("ws_disconnect", session_id=self.session_id)
                return
            text = message.get("text")
            raw = (
                text
                if text is not None
                else (message.get("bytes") or b"").decode("utf-8", "replace")
            )
            state = RequestState(trace_id=new_trace_id(self.masker), session_id=self.session_id)
            start = time.perf_counter()
            reply = await self._reply(raw, state)
            status = 200 if state.error_code is None else STATUS[state.error_code]
            self.log.request(request_fields(
                state, transport="ws", method="WS", path=PATH, status_code=status,
                latency_ms=(time.perf_counter() - start) * 1000,
            ))  # fmt: skip
            if not await send_frame(self.ws, reply):
                self.log.event("ws_disconnect_mid_turn", session_id=self.session_id,
                               trace_id=state.trace_id)  # fmt: skip
                return

    def _error(self, state: RequestState, code: str, errors: list[FieldError] | None = None
               ) -> dict[str, Any]:  # fmt: skip
        state.error_code = code
        return {"type": "error", **error_payload(code, state.trace_id, errors)}

    async def _reply(self, raw: str, state: RequestState) -> dict[str, Any]:
        try:
            frame = CLIENT_FRAME.validate_json(raw)
        except ValidationError as exc:
            return self._error(state, "VALIDATION_ERROR", safe_errors(exc.errors()))
        if not isinstance(frame, AskFrame):
            self.services.sessions.reset(self.session_id)
            return {"type": "reset_ok", "trace_id": state.trace_id}
        try:
            with anyio.CancelScope(shield=True):  # the turn completes even if the client left
                response = await self.services.orchestrator.handle(
                    frame.message, self.session_id, "ws", trace_id=state.trace_id
                )
        except PipelineError as exc:
            return self._error(state, exc.error_code)
        except SchemaViolation:
            return self._error(state, "SCHEMA_VIOLATION")
        except Exception:  # noqa: BLE001  (never re-raised: the connection and server stay up)
            self.log.event("error", level=logging.ERROR, trace_id=state.trace_id,
                           stack=traceback.format_exc())  # fmt: skip
            return self._error(state, "INTERNAL_ERROR")
        finally:
            state.record = self.services.collector.take(state.trace_id)
        return {"type": "answer", **response.model_dump(mode="json")}


@router.websocket(PATH)
async def chat(ws: WebSocket) -> None:
    state = ws.app.state
    if role_for(ws.headers.get("x-api-key"), state.settings) is None:
        await ws.close(code=CLOSE_POLICY_VIOLATION)
        return
    services: Services | None = state.services
    if services is None:
        await ws.close(code=CLOSE_TRY_AGAIN_LATER)
        return
    try:
        await ChatLoop(ws, services, state.log, state.masker).run()
    except (WebSocketDisconnect, WebSocketDisconnected):
        state.log.event("ws_disconnect")
    except Exception:  # noqa: BLE001  (a broken connection never takes the server down)
        state.log.event("error", level=logging.ERROR, stack=traceback.format_exc())
