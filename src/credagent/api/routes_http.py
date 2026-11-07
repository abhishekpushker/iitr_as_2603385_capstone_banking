"""HTTP endpoints (§ARCH 7.16). Handlers are thin: the pipeline does the work, the error
handlers map its typed exceptions, the middleware writes the log line from `request.state.log`.
"""

from __future__ import annotations

from typing import Annotated, Any

import anyio.to_thread
from fastapi import APIRouter, Depends, Path, Request, Response
from fastapi.responses import JSONResponse

from credagent.api.auth import require_admin, require_user
from credagent.api.errors import ApiError
from credagent.api.middleware import RequestState, new_trace_id
from credagent.api.models import (
    SESSION_ID_PATTERN,
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
    ChunkCounts,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    StatsResponse,
    contains_pii,
)
from credagent.pipeline.factory import Services

ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 409, 413, 422, 500, 503, 504)
}
router = APIRouter()


def services_of(request: Request) -> Services:
    services: Services | None = request.app.state.services
    if services is None:
        raise ApiError("NOT_READY")
    return services


def state_of(request: Request) -> RequestState:
    state: RequestState = request.state.log
    return state


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
async def readyz(request: Request) -> JSONResponse:
    services: Services | None = request.app.state.services
    checks = {"services": services is not None, "chroma": False, "collections": False}
    if services is not None:
        checks |= await anyio.to_thread.run_sync(services.readiness)
    ready = all(checks.values())
    body = ReadyResponse(status="ready" if ready else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)


@router.post("/v1/ask", response_model=AskResponse, responses=ERRORS)
async def ask(
    body: AskRequest, request: Request, _: Annotated[str, Depends(require_user)]
) -> AskResponse:
    services, state = services_of(request), state_of(request)
    session_id = (
        new_trace_id(request.app.state.masker) if body.session_id is None else body.session_id
    )
    state.session_id = session_id
    try:
        return await services.orchestrator.handle(
            body.message, session_id, "http", trace_id=state.trace_id
        )
    finally:
        state.record = services.collector.take(state.trace_id)


@router.post("/v1/documents", status_code=201, response_model=AddDocumentResponse, responses=ERRORS)
async def add_document(
    body: AddDocumentRequest, request: Request, _: Annotated[str, Depends(require_admin)]
) -> AddDocumentResponse:
    services, state = services_of(request), state_of(request)
    prepared = services.documents.prepare(
        doc_id=body.doc_id, title=body.title, topic=body.topic, text=body.text
    )
    state.masked_text, state.pii_counts = prepared.body, prepared.pii_counts
    state.injection_score = prepared.injection_score
    added = await anyio.to_thread.run_sync(
        lambda: services.documents.add(prepared, overwrite=body.overwrite)
    )
    return AddDocumentResponse(
        doc_id=added.doc_id, chunks=ChunkCounts(**added.chunks), kb_version=added.kb_version
    )


@router.delete("/v1/sessions/{session_id}", status_code=204, responses=ERRORS)
async def delete_session(
    request: Request,
    session_id: Annotated[str, Path(pattern=SESSION_ID_PATTERN)],
    _: Annotated[str, Depends(require_user)],
) -> Response:
    if contains_pii(session_id):
        raise ApiError("VALIDATION_ERROR")
    services_of(request).sessions.reset(session_id)
    state_of(request).session_id = session_id
    return Response(status_code=204)


@router.get("/v1/stats", response_model=StatsResponse, responses=ERRORS)
async def stats(request: Request, _: Annotated[str, Depends(require_user)]) -> StatsResponse:
    services = services_of(request)
    return StatsResponse.model_validate({
        **services.metrics.snapshot(), "cache_entries": len(services.cache),
        "sessions": len(services.sessions), "kb_version": services.kb.current,
    })  # fmt: skip
