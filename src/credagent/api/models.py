"""Request and response models of the API (§ARCH 7.16). `AskResponse` is the pipeline's
(`schemas`); everything else is the HTTP/WS surface. Every request model forbids extra fields.

A session id is a memory key and is logged, so one the PII masker would change (a PAN fits the
id pattern) is a validation error, never stored (§ARCH 6 stage 1).
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from credagent.guardrails.pii import PiiMasker
from credagent.schemas import AskResponse
from credagent.settings import get_settings

__all__ = [
    "AddDocumentRequest", "AddDocumentResponse", "AskFrame", "AskRequest", "AskResponse",
    "CLIENT_FRAME", "ChunkCounts", "ErrorResponse", "FieldError", "HealthResponse",
    "ReadyResponse", "ResetFrame", "StatsResponse", "contains_pii",
]  # fmt: skip

SESSION_ID_PATTERN: Final = r"^[A-Za-z0-9_-]{8,64}$"
DOC_ID_PATTERN: Final = r"^kb-[a-z0-9-]{3,60}$"
MAX_MESSAGE_CHARS: Final = get_settings().max_message_chars
_REQUEST = ConfigDict(extra="forbid")
_MASKER = PiiMasker()


def contains_pii(value: str) -> bool:
    return _MASKER.mask(value).text != value


class AskRequest(BaseModel):
    model_config = _REQUEST

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)

    @field_validator("session_id")
    @classmethod
    def _no_pii(cls, value: str | None) -> str | None:
        if value is not None and contains_pii(value):
            raise ValueError("session_id must not contain personal data")
        return value


class AddDocumentRequest(BaseModel):
    model_config = _REQUEST

    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    title: str = Field(min_length=3, max_length=120)
    topic: str = Field(min_length=3, max_length=60)
    text: str = Field(min_length=50, max_length=4000)
    overwrite: bool = False


class ChunkCounts(BaseModel):
    fixed: int
    sentence: int


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks: ChunkCounts
    kb_version: str


class StatsResponse(BaseModel):
    """The cache-demo counters (§ARCH 7.15) plus the cache size, live sessions and kb_version."""

    retrieval_calls: int
    embedding_calls: int
    generation_calls: int
    cache_hits: int
    cache_misses: int
    llm_calls: dict[str, int]
    cache_entries: int
    sessions: int
    kb_version: str


class FieldError(BaseModel):
    """Where and why validation failed; never the input value or the error context (P9)."""

    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str
    errors: list[FieldError] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


# ------------------------------------------------------------------ WebSocket client frames
class AskFrame(BaseModel):
    model_config = _REQUEST

    type: Literal["ask"]
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ResetFrame(BaseModel):
    model_config = _REQUEST

    type: Literal["reset"]


CLIENT_FRAME: Final[TypeAdapter[AskFrame | ResetFrame]] = TypeAdapter(
    Annotated[AskFrame | ResetFrame, Field(discriminator="type")]
)
