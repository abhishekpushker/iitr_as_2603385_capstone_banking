"""Domain Pydantic models (§ARCH 7.11).

`LoanCategory` and `LoanStatus` mirror `dataset.CATEGORY_WEIGHTS` / `STATUS_WEIGHTS`. They are
shared through a unit test that asserts equality, not an import (dataset.py is stdlib-only).
Phase 3 adds the grounded-generation result (`SourceRef`, `GeneratedAnswer`); Phase 5 the tool
outputs (`LoanStatusFound` / `LoanStatusNotFound`, `RetrievalToolOutput` / `RagToolError`).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from credagent.kb.chunking import Strategy

LoanCategory = Literal["Personal", "Home", "Auto", "Education", "Business"]
LoanStatus = Literal["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

# §ARCH 7.5 step 1: the answer when retrieval is below the calibrated threshold.
FALLBACK_ANSWER: Final = "I don't know based on the available policy documents."

_MODEL = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceRef(BaseModel):
    model_config = _MODEL

    doc_id: str
    chunk_id: str
    similarity: float = Field(ge=-1, le=1)


class GeneratedAnswer(BaseModel):
    model_config = _MODEL

    answer: str = Field(min_length=1)
    sources: tuple[SourceRef, ...]
    grounded: bool
    top1_similarity: float = Field(ge=-1, le=1)


# ------------------------------------------------------------------ tool outputs (Phase 5)
# Every tool returns `json.dumps` of one of these, so the agents only ever see validated shapes.
RECORD_ID_PATTERN: Final = r"^LA-[0-9]{4}$"  # ASCII digits only; `\d` would accept "LA-٠٠٠١"
RecordIdError = Literal["invalid_record_id", "not_found"]


class LoanStatusFound(BaseModel):
    """`check_loan_application_status` for a known id (§ARCH 7.7). The brief's minimum is
    `status`, `loan_amount_inr`, `escalation_score`; the rest serve the Composer and transcripts."""

    model_config = _MODEL

    found: Literal[True]
    record_id: str = Field(pattern=RECORD_ID_PATTERN)
    category: LoanCategory
    status: LoanStatus
    loan_amount_inr: int = Field(gt=0)
    days_since_created: int = Field(ge=0, le=30)
    flagged_for_fraud_review: bool
    escalation_score: float = Field(ge=0, le=1)
    escalation_recommended: bool
    escalation_threshold: float = Field(ge=0, le=1)


class LoanStatusNotFound(BaseModel):
    model_config = _MODEL

    found: Literal[False]
    error: RecordIdError


LOAN_STATUS_OUTPUT: Final[TypeAdapter[LoanStatusFound | LoanStatusNotFound]] = TypeAdapter(
    LoanStatusFound | LoanStatusNotFound
)


class RetrievalToolOutput(BaseModel):
    """`rag_lookup` result: the grounded (or fallback) answer and where it came from."""

    model_config = _MODEL

    kind: Literal["rag_result"] = "rag_result"
    answer: str = Field(min_length=1)
    sources: tuple[SourceRef, ...]
    grounded: bool
    top1_similarity: float = Field(ge=-1, le=1)
    strategy: Strategy
    threshold: float = Field(ge=-1, le=1)


class RagToolError(BaseModel):
    """Returned instead of raising into CrewAI; carries no internals (no exception text)."""

    model_config = _MODEL

    kind: Literal["rag_error"] = "rag_error"
    error: Literal["retrieval_unavailable"]


RAG_TOOL_OUTPUT: Final[TypeAdapter[RetrievalToolOutput | RagToolError]] = TypeAdapter(
    RetrievalToolOutput | RagToolError
)
