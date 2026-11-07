"""`Orchestrator.handle(raw_text, session_id, transport) -> AskResponse` (§ARCH 6): the only path
from the API (Phase 10) to the agent. Every stage records its duration in `ctx.timings_ms`.

```
 0  RequestContext (trace id, budget meter, ledger, timings) + TurnRecord
 1  mask        PiiMasker.mask(raw_text); the raw text is dropped here
 2  injection   InjectionDetector.detect(masked) → blocked: injection_detected refusal, not stored
 3  budget      BudgetPreflight.preflight(masked) → BudgetExceeded("preflight"), no crew call
 ── ConversationMemory.run_turn: RunnableWithMessageHistory, under the session's asyncio.Lock ──
 4  memory      MockEngine.resolve_followup(masked, history) → standalone query, or a
                clarification when it refers to something history cannot resolve (no crew)
 5  crew        crew_runner(ctx, query) in a worker thread under fail_after(REQUEST_TIMEOUT_S)
                → PipelineTimeout; afterwards meter.exceeded → BudgetExceeded("runtime")
 6  schema      parse_crew_output (raw validates, equals pydantic) + the two task outputs
 7  review      ReviewStage.review(ctx, query, draft, trace, lookup) → ReviewOutcome: the Autogen
                team (`AutogenReview`); skipped for refusals and clarifications, which claim
                nothing. A failed review fails closed: agent_error refusal, `review_error` logged
 8  groundedness GroundednessGuard.check(verdict.final_answer, draft, trace, lookup)
 9  budget_postcheck  meter.exceeded (e.g. review overran) → BudgetExceeded("runtime")
10  append      the Runnable stores HumanMessage(masked) + AIMessage(answer, {record_id,
                category}) — only when stages 4–9 succeed
11  response    AskResponse
12  log         RequestSink.emit(TurnRecord) in `finally`, exactly once, masked fields only
```

Stage 3's real policy (`governance.budget.BudgetPolicy`) arrives in Phase 9 behind
`BudgetPreflight`; stage 12's JSON-lines logger arrives in Phase 10 behind `RequestSink`.
Errors propagate as typed exceptions (`PipelineTimeout`, `BudgetExceeded`, `SchemaViolation`)
for the API to map to 504 / 413 / 500, after the record is emitted with its outcome and error
code.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anyio
import anyio.to_thread
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import ValidationError

from credagent.context import RequestContext, estimate_tokens
from credagent.crew.factory import SchemaViolation, parse_crew_output
from credagent.guardrails.groundedness import (
    GroundednessGuard,
    GuardDecision,
    RetrievalTrace,
    SentenceCheck,
)
from credagent.guardrails.injection import InjectionDetector
from credagent.guardrails.pii import PiiMasker, PiiType
from credagent.llm.engine import CLARIFY_ANSWER, MockEngine
from credagent.memory.session import ConversationMemory
from credagent.review.team import ReviewContext, ReviewOutcome, run_review
from credagent.schemas import (
    LOOKUP_TASK_OUTPUT,
    AskResponse,
    CrewResponse,
    LookupFound,
    LookupNotFound,
    RetrievalTaskOutput,
    RetrievalToolOutput,
    ReviewVerdict,
    ToolCall,
    Usage,
)
from credagent.settings import get_settings

Transport = Literal["http", "ws", "script"]
Outcome = Literal[
    "answered", "refused", "clarify", "blocked_injection", "budget_rejected", "timeout", "error"
]
CrewRunner = Callable[[RequestContext, str], Any]  # → a CrewOutput-shaped object
ChunkTexts = Callable[[Sequence[str]], Mapping[str, str]]  # chunk ids → texts


# ------------------------------------------------------------------ errors
class PipelineError(Exception):
    error_code: str = "INTERNAL_ERROR"


class PipelineTimeout(PipelineError):
    error_code = "PIPELINE_TIMEOUT"


class BudgetExceeded(PipelineError):
    error_code = "BUDGET_EXCEEDED"

    def __init__(
        self, stage: Literal["preflight", "runtime"], *, projected_tokens: int, budget_tokens: int
    ) -> None:
        super().__init__(f"budget exceeded at {stage}: {projected_tokens} > {budget_tokens}")
        self.stage = stage
        self.projected_tokens = projected_tokens
        self.budget_tokens = budget_tokens


# ------------------------------------------------------------------ pluggable stages
@dataclass(frozen=True)
class PreflightResult:
    exceeds: bool
    projected_tokens: int
    budget_tokens: int


class BudgetPreflight(Protocol):
    def preflight(self, masked_text: str) -> PreflightResult: ...


class PassThroughPreflight:
    """Stage 3 until Phase 9: reports the input's own token estimate and never rejects."""

    def __init__(self, budget_tokens: int) -> None:
        self.budget_tokens = budget_tokens

    def preflight(self, masked_text: str) -> PreflightResult:
        return PreflightResult(
            exceeds=False, projected_tokens=estimate_tokens(masked_text),
            budget_tokens=self.budget_tokens,
        )  # fmt: skip


class ReviewStage(Protocol):
    async def review(
        self,
        ctx: RequestContext,
        query: str,
        draft: CrewResponse,
        trace: RetrievalTrace | None,
        lookup: LookupFound | LookupNotFound | None,
    ) -> ReviewOutcome: ...


class AutogenReview:
    """Stage 7 (§ARCH 7.13): the reviewer/editor team on the draft answer. The context is the
    retrieved chunk texts and, when there is an application record, its JSON and the record's
    canonical statement (D-038)."""

    def __init__(self, engine: MockEngine | None = None) -> None:
        self._engine = MockEngine() if engine is None else engine

    def context_for(
        self, query: str, trace: RetrievalTrace | None, lookup: LookupFound | LookupNotFound | None
    ) -> ReviewContext:
        documents = [] if trace is None else list(trace.contexts)
        if isinstance(lookup, LookupFound):
            documents += [lookup.model_dump_json(exclude={"kind"}),
                          self._engine.record_statement(lookup)]  # fmt: skip
        return ReviewContext(question=query, documents=tuple(documents))

    async def review(
        self,
        ctx: RequestContext,
        query: str,
        draft: CrewResponse,
        trace: RetrievalTrace | None,
        lookup: LookupFound | LookupNotFound | None,
    ) -> ReviewOutcome:
        return await run_review(draft.answer, self.context_for(query, trace, lookup), ctx)


@dataclass
class TurnRecord:
    """Everything stage 12 logs about one request. Masked text and counts only."""

    trace_id: str
    session_id: str
    transport: Transport
    outcome: Outcome = "error"
    masked_text: str = ""
    pii_counts: dict[PiiType, int] = field(default_factory=dict)
    injection_score: float = 0.0
    injection_signals: tuple[str, ...] = ()
    standalone_query: str | None = None
    draft_intent: str | None = None  # the crew's, before the gate
    intent: str | None = None  # the one sent
    guard_reason: str | None = None
    top1_similarity: float | None = None
    guard_checks: tuple[SentenceCheck, ...] = ()
    review_approved: bool | None = None
    review_error: str | None = None  # the review failed closed (§ARCH 7.13)
    cache_hit: bool = False
    tools_invoked: tuple[ToolCall, ...] = ()
    tokens_estimated: int = 0
    cost_estimated_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    error_code: str | None = None


class RequestSink(Protocol):
    def emit(self, record: TurnRecord) -> None: ...


class NullSink:
    def emit(self, record: TurnRecord) -> None:
        return None


# ------------------------------------------------------------------ one turn's result
@dataclass
class _TurnResult:
    response: CrewResponse
    verdict: ReviewVerdict | None
    decision: GuardDecision | None


@contextmanager
def _stage(ctx: RequestContext, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        ctx.timings_ms[name] = (time.perf_counter() - start) * 1000


class Orchestrator:
    def __init__(
        self,
        *,
        crew_runner: CrewRunner,
        chunk_texts: ChunkTexts,
        memory: ConversationMemory,
        masker: PiiMasker | None = None,
        detector: InjectionDetector | None = None,
        preflight: BudgetPreflight | None = None,
        review: ReviewStage | None = None,
        guard: GroundednessGuard | None = None,
        engine: MockEngine | None = None,
        sink: RequestSink | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        price_per_1k_tokens_usd: float | None = None,
    ) -> None:
        s = get_settings()
        self.memory = memory
        self._crew_runner = crew_runner
        self._chunk_texts = chunk_texts
        self._masker = PiiMasker() if masker is None else masker
        self._detector = InjectionDetector() if detector is None else detector
        self._max_tokens = s.budget_max_tokens if max_tokens is None else max_tokens
        self._preflight = PassThroughPreflight(self._max_tokens) if preflight is None else preflight
        self._engine = MockEngine() if engine is None else engine
        self._review = AutogenReview(self._engine) if review is None else review
        self._guard = GroundednessGuard(self._engine) if guard is None else guard
        self._sink = NullSink() if sink is None else sink
        self._timeout_s = s.request_timeout_s if timeout_s is None else timeout_s
        self._price = (
            s.price_per_1k_tokens_usd if price_per_1k_tokens_usd is None
            else price_per_1k_tokens_usd
        )  # fmt: skip

    async def handle(
        self,
        raw_text: str,
        session_id: str,
        transport: Transport = "http",
        *,
        trace_id: str | None = None,
    ) -> AskResponse:
        ctx = RequestContext.new(session_id=session_id, trace_id=trace_id,
                                 max_tokens=self._max_tokens)  # fmt: skip
        record = TurnRecord(trace_id=ctx.trace_id, session_id=session_id, transport=transport)
        start = time.perf_counter()
        try:
            with _stage(ctx, "mask"):
                masked = self._masker.mask(raw_text)
            del raw_text  # stage 1: nothing below this line can reach the raw text
            record.masked_text, record.pii_counts = masked.text, dict(masked.counts_by_type)

            with _stage(ctx, "injection"):
                injection = self._detector.detect(masked.text)
            record.injection_score, record.injection_signals = injection.score, injection.signals
            if injection.blocked:
                record.outcome = "blocked_injection"
                refusal = self._engine.refusal("injection_detected")
                return self._respond(ctx, record, _TurnResult(refusal, None, None))

            with _stage(ctx, "budget"):
                pre = self._preflight.preflight(masked.text)
            if pre.exceeds:
                raise BudgetExceeded("preflight", projected_tokens=pre.projected_tokens,
                                     budget_tokens=pre.budget_tokens)  # fmt: skip

            holder: list[_TurnResult] = []
            turn = functools.partial(self._turn, ctx, record, holder)
            await self.memory.run_turn(session_id, masked.text, turn)
            (result,) = holder
            response = result.response
            record.outcome = (
                "refused" if response.refused
                else "clarify" if response.intent == "clarify" else "answered"
            )  # fmt: skip
            return self._respond(ctx, record, result)
        except PipelineError as exc:
            record.outcome = "budget_rejected" if isinstance(exc, BudgetExceeded) else "timeout"
            record.error_code = exc.error_code
            raise
        except SchemaViolation:
            record.outcome, record.error_code = "error", "SCHEMA_VIOLATION"
            raise
        except Exception:
            record.outcome, record.error_code = "error", "INTERNAL_ERROR"
            raise
        finally:
            entries = ctx.ledger.entries()
            ctx.timings_ms["rag"] = sum(e.duration_ms for e in entries if e.tool == "rag_lookup")
            ctx.timings_ms["total"] = (time.perf_counter() - start) * 1000
            record.tools_invoked = tuple(
                ToolCall(agent=e.agent, tool=e.tool, args=e.args, cache_hit=e.cache_hit)
                for e in entries
            )
            record.cache_hit = any(e.cache_hit for e in entries)
            record.tokens_estimated = ctx.budget.used_tokens
            record.cost_estimated_usd = self._cost(ctx.budget.used_tokens)
            record.timings_ms = dict(ctx.timings_ms)
            self._sink.emit(record)

    # ------------------------------------------------------------------ stages 4–9
    async def _turn(
        self,
        ctx: RequestContext,
        record: TurnRecord,
        holder: list[_TurnResult],
        masked: str,
        history: list[BaseMessage],
    ) -> AIMessage:
        with _stage(ctx, "memory"):
            follow = self._engine.resolve_followup(masked, history)
        record.standalone_query = follow.query
        if follow.clarify:
            clarify = CrewResponse(
                intent="clarify", answer=CLARIFY_ANSWER, sources=(), application=None,
                escalation=None, grounded=True, refused=False, refusal_reason=None,
            )  # fmt: skip
            holder.append(_TurnResult(clarify, None, None))
            record.intent = clarify.intent
            return AIMessage(content=clarify.answer,
                             additional_kwargs={"record_id": None, "category": None})  # fmt: skip

        with _stage(ctx, "crew"):
            output = await self._kickoff(ctx, follow.query)
        self._budget_check(ctx)

        with _stage(ctx, "schema"):
            draft = parse_crew_output(output)
            retrieval, lookup = _task_outputs(output, draft)
        record.draft_intent = draft.intent
        trace = await self._trace(retrieval)

        verdict: ReviewVerdict | None = None
        if not (draft.refused or draft.intent == "clarify"):  # those claim nothing to review
            with _stage(ctx, "review"):
                outcome = await self._review.review(ctx, follow.query, draft, trace, lookup)
            verdict = outcome.verdict
            record.review_approved, record.review_error = verdict.approved, outcome.error

        found = lookup if isinstance(lookup, LookupFound) else None
        decision: GuardDecision | None = None
        if record.review_error is not None:
            final = self._engine.refusal("agent_error")  # fail closed (§ARCH 7.13)
        else:
            with _stage(ctx, "groundedness"):
                answer = draft.answer if verdict is None else verdict.final_answer
                decision = self._guard.check(answer, draft, trace, found)
            record.guard_reason = decision.reason
            record.top1_similarity = decision.top1_similarity
            record.guard_checks = decision.checks
            final = decision.response

        with _stage(ctx, "budget_postcheck"):
            self._budget_check(ctx)

        record.intent = final.intent
        holder.append(_TurnResult(final, verdict, decision))
        return AIMessage(
            content=final.answer,
            additional_kwargs={
                "record_id": None if found is None else found.record_id,
                "category": follow.category if found is None else found.category,
            },
        )

    async def _kickoff(self, ctx: RequestContext, query: str) -> Any:
        runner = functools.partial(self._crew_runner, ctx, query)
        try:
            with anyio.fail_after(self._timeout_s):
                # abandon_on_cancel: the CrewAI thread cannot be interrupted; on timeout it runs
                # to completion on its own and its result is discarded (§ARCH 7.9).
                return await anyio.to_thread.run_sync(runner, abandon_on_cancel=True)
        except TimeoutError as exc:
            raise PipelineTimeout(f"crew did not finish within {self._timeout_s} s") from exc

    def _budget_check(self, ctx: RequestContext) -> None:
        if ctx.budget.exceeded:
            raise BudgetExceeded("runtime", projected_tokens=ctx.budget.used_tokens,
                                 budget_tokens=ctx.budget.max_tokens)  # fmt: skip

    async def _trace(self, retrieval: RetrievalTaskOutput | None) -> RetrievalTrace | None:
        rag = None if retrieval is None else retrieval.rag
        if not isinstance(rag, RetrievalToolOutput):
            return None
        ids = [s.chunk_id for s in rag.sources]
        texts = await anyio.to_thread.run_sync(self._chunk_texts, ids)
        return RetrievalTrace(
            top1_similarity=rag.top1_similarity, threshold=rag.threshold,
            contexts=tuple(texts[i] for i in ids if i in texts),
        )  # fmt: skip

    # ------------------------------------------------------------------ stage 11
    def _respond(self, ctx: RequestContext, record: TurnRecord, result: _TurnResult) -> AskResponse:
        with _stage(ctx, "response"):
            r = result.response
            entries = ctx.ledger.entries()
            return AskResponse(
                trace_id=ctx.trace_id, session_id=ctx.session_id, answer=r.answer,
                intent=r.intent, grounded=r.grounded, refused=r.refused,
                refusal_reason=r.refusal_reason, sources=r.sources, application=r.application,
                escalation=r.escalation, review=result.verdict,
                tools_invoked=tuple(
                    ToolCall(agent=e.agent, tool=e.tool, args=e.args, cache_hit=e.cache_hit)
                    for e in entries
                ),
                cache_hit=any(e.cache_hit for e in entries),
                usage=Usage(
                    estimated_tokens=ctx.budget.used_tokens, budget_tokens=ctx.budget.max_tokens,
                    estimated_cost_usd=self._cost(ctx.budget.used_tokens),
                ),
                timings_ms=dict(ctx.timings_ms),
            )  # fmt: skip

    def _cost(self, tokens: int) -> float:
        return round(tokens / 1000 * self._price, 8)


def _task_outputs(
    output: Any, draft: CrewResponse
) -> tuple[RetrievalTaskOutput | None, LookupFound | LookupNotFound | None]:
    """The Retrieval and Lookup task outputs. A refusal draft (budget overrun, protocol error)
    carries refusals there instead, so they are not read."""
    if draft.refused:
        return None, None
    try:
        first, second = (t.raw for t in output.tasks_output[:2])
        return (
            RetrievalTaskOutput.model_validate_json(first),
            LOOKUP_TASK_OUTPUT.validate_json(second),
        )
    except (ValidationError, ValueError, AttributeError) as exc:
        raise SchemaViolation("crew task outputs are not valid task results") from exc
