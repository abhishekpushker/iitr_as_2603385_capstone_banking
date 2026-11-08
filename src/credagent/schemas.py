"""Domain Pydantic models (§ARCH 7.11).

`LoanCategory` and `LoanStatus` mirror `dataset.CATEGORY_WEIGHTS` / `STATUS_WEIGHTS`. They are
shared through a unit test that asserts equality, not an import (dataset.py is stdlib-only).
Phase 3 adds the grounded-generation result (`SourceRef`, `GeneratedAnswer`); Phase 5 the tool
outputs (`LoanStatusFound` / `LoanStatusNotFound`, `RetrievalToolOutput` / `RagToolError`); Phase 6
the crew task outputs and `CrewResponse`, the Composer's structured answer.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from credagent.kb.chunking import Strategy

# The brief's given vocabularies, verbatim.
LoanCategory = Literal[
    "Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"
]  # fmt: skip
LoanStatus = Literal["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]


def loan_type(category: LoanCategory) -> str:
    """The product word of a category ("Home Loan" → "Home"), for running text such as
    "a Home loan" or "home loans"."""
    return category.removesuffix(" Loan")


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


# ------------------------------------------------------------------ crew task outputs (Phase 6)
# The Retrieval and Lookup agents' Final Answers. The Composer finds them in its task context by
# their `kind` (§ARCH 7.8.1 step 4), so `kind` is required, never defaulted.


class RetrievalTaskOutput(BaseModel):
    """`rag` is the tool's output, or `None` when the question needs no policy lookup."""

    model_config = _MODEL

    kind: Literal["retrieval_result"]
    rag: RetrievalToolOutput | RagToolError | None


class LookupFound(LoanStatusFound):
    kind: Literal["lookup_result"]


class LookupNotFound(BaseModel):
    """`no_record_id`: the question carries no id, so the tool was not called."""

    model_config = _MODEL

    kind: Literal["lookup_result"]
    found: Literal[False]
    error: RecordIdError | Literal["no_record_id"]


LOOKUP_TASK_OUTPUT: Final[TypeAdapter[LookupFound | LookupNotFound]] = TypeAdapter(
    LookupFound | LookupNotFound
)


# ------------------------------------------------------------------ CrewResponse (§ARCH 7.11)
Intent = Literal["policy", "status", "mixed", "clarify", "out_of_scope", "refused"]
# Phase 7: `ungrounded` (groundedness gate, §7.12.3) and `injection_detected` (stage 2).
RefusalReason = Literal[
    "budget_exceeded", "agent_error", "retrieval_unavailable", "ungrounded", "injection_detected"
]
_GROUNDED_WITHOUT_SOURCES: Final = frozenset({"status", "clarify"})


class ApplicationView(BaseModel):
    model_config = _MODEL

    record_id: str = Field(pattern=RECORD_ID_PATTERN)
    category: LoanCategory
    status: LoanStatus
    loan_amount_inr: int = Field(gt=0)
    days_since_created: int = Field(ge=0, le=30)
    flagged_for_fraud_review: bool


class Escalation(BaseModel):
    model_config = _MODEL

    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    recommended: bool


class CrewResponse(BaseModel):
    """The Composer's Final Answer (`Task.output_pydantic`). `grounded` means every claim comes
    from a retrieved source or, for `status`/`clarify`, from the application record or the
    request itself."""

    model_config = _MODEL

    kind: Literal["crew_response"] = "crew_response"
    intent: Intent
    answer: str = Field(min_length=1, max_length=4000)
    sources: tuple[SourceRef, ...]
    application: ApplicationView | None
    escalation: Escalation | None
    grounded: bool
    refused: bool
    refusal_reason: RefusalReason | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.refused != (self.refusal_reason is not None):
            raise ValueError("a refusal_reason is required exactly when refused is true")
        if self.refused != (self.intent == "refused"):
            raise ValueError("intent 'refused' is required exactly when refused is true")
        if self.refused and self.grounded:
            raise ValueError("a refusal cannot be grounded")
        if self.intent == "status" and self.application is None:
            raise ValueError("intent 'status' requires an application")
        if self.escalation is not None and self.application is None:
            raise ValueError("an escalation requires an application")
        if self.grounded and not self.sources and self.intent not in _GROUNDED_WITHOUT_SOURCES:
            raise ValueError("grounded requires sources unless intent is 'status' or 'clarify'")
        return self


# ------------------------------------------------------------------ the pipeline's answer (Phase 7)
# `Orchestrator.handle` returns `AskResponse` (§ARCH 6 step 11, 7.16). It lives here, not in
# `api`, because the pipeline may not import the API layer; `api/models.py` re-exports it.


class ReviewVerdict(BaseModel):
    """Exactly the brief's fields (§ARCH 7.11). Phase 7's review stage is a pass-through."""

    model_config = _MODEL

    approved: bool
    final_answer: str
    reason: str


class JudgeScores(BaseModel):
    """The LLM-as-judge reply (§ARCH 7.8.3, 7.18): the four rubric scores and why."""

    model_config = _MODEL

    accuracy: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    safety: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class ToolCall(BaseModel):
    """One `ToolCallLedger` entry as the client sees it (arguments are masked text)."""

    model_config = _MODEL

    agent: str
    tool: str
    args: dict[str, str]
    cache_hit: bool


class Usage(BaseModel):
    model_config = _MODEL

    estimated_tokens: int = Field(ge=0)
    budget_tokens: int = Field(ge=1)
    estimated_cost_usd: float = Field(ge=0)


class AskResponse(BaseModel):
    """`CrewResponse`'s answer fields (same consistency rules) plus the request's trace."""

    model_config = _MODEL

    trace_id: str
    session_id: str
    answer: str = Field(min_length=1, max_length=4000)
    intent: Intent
    grounded: bool
    refused: bool
    refusal_reason: RefusalReason | None
    sources: tuple[SourceRef, ...]
    application: ApplicationView | None
    escalation: Escalation | None
    review: ReviewVerdict | None  # None when no review ran (blocked input, clarification)
    tools_invoked: tuple[ToolCall, ...]
    cache_hit: bool
    usage: Usage
    timings_ms: dict[str, float]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        fields = set(CrewResponse.model_fields) - {"kind"}
        CrewResponse.model_validate({f: getattr(self, f) for f in fields})
        return self
