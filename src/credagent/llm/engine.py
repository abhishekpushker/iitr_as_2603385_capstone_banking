"""The deterministic mock-LLM engine (§ARCH 7.8). Pure and stateless: every output is a function
of the arguments, built on the `credagent.text` primitives. Phase 3 surface:

* `support_report(answer, context)`: the support rule shared by generation, the groundedness
  guard, the reviewer and the judge. A sentence S is supported by context C iff
  (a) `token_recall(S, best-matching context sentence) ≥ SUPPORT_RECALL` and
  (b) every number in S appears among the numbers of C (catches "0% interest for life").
* `generate_grounded(query, chunks, threshold=…)`: §ARCH 7.5. Below the threshold → the
  fallback. Otherwise extractive synthesis: split the chunks into sentences, drop repeats
  (overlapping fixed chunks, P16) and fragments contained in another sentence, score each by
  `RECALL_WEIGHT·query-token recall + SIMILARITY_WEIGHT·chunk similarity`, keep the best
  `MAX_ANSWER_SENTENCES` scoring ≥ `MIN_SENTENCE_SCORE`, and emit them verbatim in document
  order. The answer is grounded by construction.

Phase 6 (the Composer, §ARCH 7.8.1 step 5):

* `classify_intent(query, record_id)`: with the record id removed, a question whose remaining
  content tokens are all generic status words (`STATUS_TERMS`: "status", "application", the loan
  categories and statuses, …) asks only for a status: `status` with an id, `clarify` without one
  if it carries a status cue. Anything else is a policy question: `mixed` with an id, `policy`
  without.
* `compose(retrieval, lookup, query=…, record_id=…)`: the `CrewResponse` from the two task
  outputs. Inputs that contradict the intent (e.g. a skipped retrieval for a policy question)
  fail closed as an `agent_error` refusal.
* `refusal(reason)`: the schema-valid refusal for each `RefusalReason`.

Nothing here knows any query: behaviour comes only from the query tokens and the chunks (N7).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, get_args

from credagent.rag.retriever import NO_MATCH_SIMILARITY
from credagent.rag.store import RetrievedChunk
from credagent.schemas import (
    FALLBACK_ANSWER,
    ApplicationView,
    CrewResponse,
    Escalation,
    GeneratedAnswer,
    Intent,
    LoanCategory,
    LoanStatus,
    LookupFound,
    LookupNotFound,
    RagToolError,
    RefusalReason,
    RetrievalTaskOutput,
    SourceRef,
)
from credagent.text import (
    RECORD_ID_CANDIDATE,
    content_tokens,
    extract_numbers,
    normalize,
    sentence_spans,
)

SUPPORT_RECALL: Final = 0.6
RECALL_WEIGHT: Final = 0.7
SIMILARITY_WEIGHT: Final = 0.3
# Similarity alone contributes at most SIMILARITY_WEIGHT, so a sentence must share at least one
# query token to be selected (D-024).
MIN_SENTENCE_SCORE: Final = SIMILARITY_WEIGHT
MAX_ANSWER_SENTENCES: Final = 2

# ------------------------------------------------------------------ Composer vocabulary
_STATUS_WORDS: Final = frozenset(
    w.casefold() for v in get_args(LoanStatus) + get_args(LoanCategory) for w in v.split()
)
# Generic words of a status request; a question made only of these (plus stopwords and the id)
# asks for nothing a policy document could answer.
STATUS_TERMS: Final = _STATUS_WORDS | frozenset(
    {"status", "application", "applications", "loan", "loans", "check", "update", "updates",
     "progress", "stage", "track", "current", "currently", "latest", "yet", "now", "still",
     "please", "id"}
)  # fmt: skip
# Without an id, a status-only question must carry one of these to be a request for a status
# (and so get the clarifying answer) rather than a vague policy question.
STATUS_CUES: Final = frozenset(
    {"status", "update", "updates", "progress", "stage", "track", "submitted", "review",
     "approved", "rejected", "disbursed"}
)  # fmt: skip
CLARIFY_ANSWER: Final = "Please share your application id (format LA-0000)."
REFUSAL_ANSWERS: Final[dict[RefusalReason, str]] = {
    "budget_exceeded": (
        "This request is too large to process within the usage budget. Please shorten it and "
        "try again."
    ),
    "agent_error": (
        "Sorry, I couldn't complete this request. Please try again or contact Cred support."
    ),
    "retrieval_unavailable": (
        "I can't reach the policy documents right now. Please try again shortly."
    ),
}


def format_inr(amount: int) -> str:
    """Indian digit grouping: `2500000` → `₹25,00,000`."""
    digits = str(amount)
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join([*groups, tail])


def _age(days: int) -> str:
    return "today" if days == 0 else f"{days} day{'' if days == 1 else 's'} ago"


def _status_sentence(app: LookupFound) -> str:
    escalation = (
        f"It has been referred to a human specialist (escalation score {app.escalation_score} "
        f"≥ threshold {app.escalation_threshold})."
        if app.escalation_recommended
        else f"No escalation is needed (escalation score {app.escalation_score} < threshold "
        f"{app.escalation_threshold})."
    )
    return (
        f"Application {app.record_id} ({app.category} loan) is currently {app.status}. The loan "
        f"amount is {format_inr(app.loan_amount_inr)} and the application was created "
        f"{_age(app.days_since_created)}. {escalation}"
    )


def _not_found_sentence(lookup: LookupNotFound, record_id: str) -> str:
    if lookup.error == "not_found":
        return (
            f"I couldn't find a loan application with id {record_id}. Please check the id "
            "(format LA-0000)."
        )
    return f"{record_id} is not a valid application id. Please check the id (format LA-0000)."


def token_recall(sentence: str, reference: str) -> float:
    """Share of the sentence's content tokens found in `reference`. A sentence without content
    tokens makes no claim, so its recall is 1.0."""
    tokens = set(content_tokens(sentence))
    if not tokens:
        return 1.0
    return len(tokens & set(content_tokens(reference))) / len(tokens)


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    best_match: str | None
    recall: float
    numbers_missing: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.recall >= SUPPORT_RECALL and not self.numbers_missing


@dataclass(frozen=True)
class SupportReport:
    sentences: tuple[SentenceSupport, ...]

    @property
    def total(self) -> int:
        return len(self.sentences)

    @property
    def supported_count(self) -> int:
        return sum(s.supported for s in self.sentences)

    @property
    def ratio(self) -> float:
        """Supported / total; an answer with no sentences makes no claims (1.0)."""
        return self.supported_count / self.total if self.total else 1.0

    @property
    def all_supported(self) -> bool:
        return self.supported_count == self.total


@dataclass(frozen=True)
class _Candidate:
    text: str
    key: str  # normalised text, for dedupe and containment
    position: tuple[int, int]  # (document order, offset in the document body)
    chunk: RetrievedChunk


def _clamp(similarity: float) -> float:
    """float32 cosine distance can be a hair below 0 or above 2."""
    return min(1.0, max(-1.0, similarity))


def _split(text: str) -> list[tuple[int, str]]:
    return [(a, text[a:b]) for a, b in sentence_spans(text)]


class MockEngine:
    # ------------------------------------------------------------------ support rule
    def support_report(self, answer: str, context: Sequence[str]) -> SupportReport:
        context_sentences = [s for c in context for _, s in _split(c)]
        context_numbers = {n for c in context for n in extract_numbers(c)}
        out: list[SentenceSupport] = []
        for _, sentence in _split(answer):
            best: str | None = None
            recall = 0.0
            for candidate in context_sentences:
                r = token_recall(sentence, candidate)
                if best is None or r > recall:
                    best, recall = candidate, r
            missing = tuple(n for n in extract_numbers(sentence) if n not in context_numbers)
            out.append(SentenceSupport(sentence, best, recall, missing))
        return SupportReport(tuple(out))

    # ------------------------------------------------------------------ grounded generation
    def generate_grounded(
        self, query: str, chunks: Sequence[RetrievedChunk], *, threshold: float
    ) -> GeneratedAnswer:
        top1 = _clamp(max((c.similarity for c in chunks), default=NO_MATCH_SIMILARITY))
        if not chunks or top1 < threshold:
            return self._fallback(top1)

        query_tokens = set(content_tokens(query))
        scored = []
        for cand in self._candidates(chunks):
            recall = (
                len(query_tokens & set(content_tokens(cand.text))) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = RECALL_WEIGHT * recall + SIMILARITY_WEIGHT * _clamp(cand.chunk.similarity)
            if score >= MIN_SENTENCE_SCORE:
                scored.append((score, cand))
        if not scored:
            return self._fallback(top1)

        best = sorted(scored, key=lambda sc: (-sc[0], sc[1].position))[:MAX_ANSWER_SENTENCES]
        chosen = sorted((c for _, c in best), key=lambda c: c.position)
        sources: dict[str, SourceRef] = {}
        for c in chosen:
            sources.setdefault(
                c.chunk.chunk_id,
                SourceRef(
                    doc_id=c.chunk.doc_id, chunk_id=c.chunk.chunk_id,
                    similarity=_clamp(c.chunk.similarity),
                ),
            )  # fmt: skip
        return GeneratedAnswer(
            answer=" ".join(c.text for c in chosen),
            sources=tuple(sources.values()),
            grounded=True,
            top1_similarity=top1,
        )

    @staticmethod
    def _fallback(top1: float) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=FALLBACK_ANSWER, sources=(), grounded=False, top1_similarity=top1
        )

    @staticmethod
    def _candidates(chunks: Sequence[RetrievedChunk]) -> list[_Candidate]:
        """Sentences of the chunks in rank order, without exact repeats (the higher-ranked copy
        wins) or fragments contained in another sentence of the context (P16)."""
        doc_order: dict[str, int] = {}
        seen: dict[str, _Candidate] = {}
        for chunk in sorted(chunks, key=lambda c: (c.rank, c.chunk_id)):
            order = doc_order.setdefault(chunk.doc_id, len(doc_order))
            for offset, text in _split(chunk.text):
                key = normalize(text)
                if key not in seen:
                    seen[key] = _Candidate(text, key, (order, chunk.char_start + offset), chunk)
        keys = list(seen)
        return [
            cand
            for cand in seen.values()
            if not any(cand.key != other and cand.key in other for other in keys)
        ]

    # ------------------------------------------------------------------ Composer (Phase 6)
    def classify_intent(self, query: str, record_id: str | None) -> Intent:
        tokens = content_tokens(RECORD_ID_CANDIDATE.sub(" ", query))
        status_only = all(t in STATUS_TERMS for t in tokens)
        if record_id is not None:
            return "status" if status_only else "mixed"
        if status_only and any(t in STATUS_CUES for t in tokens):
            return "clarify"
        return "policy"

    def refusal(self, reason: RefusalReason) -> CrewResponse:
        return CrewResponse(
            intent="refused", answer=REFUSAL_ANSWERS[reason], sources=(), application=None,
            escalation=None, grounded=False, refused=True, refusal_reason=reason,
        )  # fmt: skip

    def compose(
        self,
        retrieval: RetrievalTaskOutput,
        lookup: LookupFound | LookupNotFound,
        *,
        query: str,
        record_id: str | None,
    ) -> CrewResponse:
        intent = self.classify_intent(query, record_id)
        needs_rag = intent in {"policy", "mixed"}
        needs_lookup = intent in {"status", "mixed"}
        looked_up = not (isinstance(lookup, LookupNotFound) and lookup.error == "no_record_id")
        rag = retrieval.rag
        if needs_rag != (rag is not None) or needs_lookup != looked_up:
            return self.refusal("agent_error")
        if intent == "clarify":
            return self._answer("clarify", CLARIFY_ANSWER, grounded=True)

        parts: list[str] = []
        application, escalation = None, None
        if needs_lookup:
            if isinstance(lookup, LookupFound):
                parts.append(_status_sentence(lookup))
                application, escalation = _application(lookup)
            else:
                parts.append(_not_found_sentence(lookup, record_id or ""))
        if intent == "status":
            return self._answer(
                "status" if application else "clarify", parts[0], grounded=True,
                application=application, escalation=escalation,
            )  # fmt: skip

        if rag is None or isinstance(rag, RagToolError):  # None cannot reach here (needs_rag)
            if intent == "policy":
                return self.refusal("retrieval_unavailable")
            parts.append(REFUSAL_ANSWERS["retrieval_unavailable"])
            sources: tuple[SourceRef, ...] = ()
            grounded = False
        else:
            parts.append(rag.answer)
            sources, grounded = rag.sources, rag.grounded
        if intent == "policy" and not grounded:
            intent = "out_of_scope"
        return self._answer(
            intent, " ".join(parts), grounded=grounded, sources=sources,
            application=application, escalation=escalation,
        )  # fmt: skip

    @staticmethod
    def _answer(
        intent: Intent,
        answer: str,
        *,
        grounded: bool,
        sources: tuple[SourceRef, ...] = (),
        application: ApplicationView | None = None,
        escalation: Escalation | None = None,
    ) -> CrewResponse:
        return CrewResponse(
            intent=intent, answer=answer, sources=sources, application=application,
            escalation=escalation, grounded=grounded, refused=False, refusal_reason=None,
        )  # fmt: skip


def _application(app: LookupFound) -> tuple[ApplicationView, Escalation]:
    return (
        ApplicationView(
            record_id=app.record_id, category=app.category, status=app.status,
            loan_amount_inr=app.loan_amount_inr, days_since_created=app.days_since_created,
            flagged_for_fraud_review=app.flagged_for_fraud_review,
        ),
        Escalation(
            score=app.escalation_score, threshold=app.escalation_threshold,
            recommended=app.escalation_recommended,
        ),
    )  # fmt: skip
