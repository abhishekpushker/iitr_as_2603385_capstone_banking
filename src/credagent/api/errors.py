"""Error envelope and handlers (§ARCH 7.16, P9).

Every error is `ErrorResponse{error_code, message, trace_id}` with a fixed message per code: no
input value, no exception text, no stack. Validation errors add `errors[{loc, msg, type}]`, built
by hand from pydantic's list so `input` and `ctx` are never returned (FastAPI's default handler
echoes the input). Strings in `loc`/`msg` are masked as well: an unknown field name is user text.
Each handler records the code on the request's log state, so the request line carries it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from credagent.api.models import ErrorResponse, FieldError
from credagent.crew.factory import SchemaViolation
from credagent.guardrails.pii import PiiMasker
from credagent.pipeline.orchestrator import PipelineError

STATUS: Final[dict[str, int]] = {
    "UNAUTHORIZED": 401, "FORBIDDEN": 403, "NOT_FOUND": 404, "METHOD_NOT_ALLOWED": 405,
    "DUPLICATE_DOCUMENT": 409, "BUDGET_EXCEEDED": 413, "PAYLOAD_TOO_LARGE": 413,
    "VALIDATION_ERROR": 422, "INJECTION_DETECTED": 422, "INTERNAL_ERROR": 500,
    "SCHEMA_VIOLATION": 500, "NOT_READY": 503, "PIPELINE_TIMEOUT": 504,
}  # fmt: skip
MESSAGES: Final[dict[str, str]] = {
    "UNAUTHORIZED": "a valid X-API-Key header is required",
    "FORBIDDEN": "this endpoint needs the admin key",
    "NOT_FOUND": "no such endpoint",
    "METHOD_NOT_ALLOWED": "method not allowed",
    "DUPLICATE_DOCUMENT": "a document with this doc_id exists; set overwrite to replace it",
    "BUDGET_EXCEEDED": "the request exceeds the token budget",
    "PAYLOAD_TOO_LARGE": "the request body is too large",
    "VALIDATION_ERROR": "the request is invalid",
    "INJECTION_DETECTED": "the document contains a prompt-injection pattern",
    "INTERNAL_ERROR": "internal error",
    "SCHEMA_VIOLATION": "the agent produced an invalid response",
    "NOT_READY": "the service is starting; try again shortly",
    "PIPELINE_TIMEOUT": "the request took too long",
}
_HTTP_CODES: Final = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
_MASKER = PiiMasker()


class ApiError(Exception):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def safe_errors(errors: Iterable[Any]) -> list[FieldError]:
    def scrub(value: str) -> str:
        return _MASKER.mask(value).text

    return [
        FieldError(
            loc=[scrub(p) if isinstance(p, str) else int(p) for p in e.get("loc", ())],
            msg=scrub(str(e.get("msg", ""))),
            type=str(e.get("type", "")),
        )  # fmt: skip
        for e in errors
    ]


def error_payload(
    error_code: str, trace_id: str, errors: list[FieldError] | None = None
) -> dict[str, Any]:
    body = ErrorResponse(error_code=error_code, message=MESSAGES[error_code], trace_id=trace_id,
                         errors=errors)  # fmt: skip
    return body.model_dump(mode="json", exclude_none=True)


def error_json(state: Any, error_code: str, errors: list[FieldError] | None = None) -> JSONResponse:
    """The envelope for `state` (the request's log state), whose `error_code` it sets."""
    state.error_code = error_code
    return JSONResponse(error_payload(error_code, state.trace_id, errors), STATUS[error_code])


def _on(code_of: Callable[[Any], str]) -> Callable[[Request, Exception], Any]:
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return error_json(request.state.log, code_of(exc))

    return handler


async def _validation(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return error_json(request.state.log, "VALIDATION_ERROR", safe_errors(errors))


def install_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation)
    app.add_exception_handler(ApiError, _on(lambda e: e.error_code))
    app.add_exception_handler(PipelineError, _on(lambda e: e.error_code))
    app.add_exception_handler(SchemaViolation, _on(lambda e: "SCHEMA_VIOLATION"))
    app.add_exception_handler(
        StarletteHTTPException, _on(lambda e: _HTTP_CODES.get(e.status_code, "VALIDATION_ERROR"))
    )
