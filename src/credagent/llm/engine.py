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

Phase 7 (memory, §ARCH 7.10):

* `resolve_followup(query, history)`: a query without its own record id that refers back ("it"
  after a preposition or auxiliary, "its", "this/that/my application", "that loan", "the same
  one") is rewritten into a standalone query from the most recent record id and loan category in
  the history's AI messages (`additional_kwargs`). A possessive "its" with a noun phrase
  before it in the same sentence ("close my account in its first year") refers to that phrase
  and is not a reference (BUG-003); so is an "it" with such a phrase in an earlier clause of its
  sentence ("the card … and can it be waived"), and an extraposed "does it take/cost … to <verb>"
  (BUG-006). A status-only question takes the record id
  ("Has it been approved?" → "Has application LA-0007 been approved?"); any other question takes
  the loan category ("eligibility criteria for it" → "… for a Home loan"). Nothing to resolve →
  `clarify`. No side state: a fresh session has an empty history and so no state.

Phase 8 (the Autogen review, §ARCH 7.8.2):

* `review(draft, context)`: the Policy_Compliance_Reviewer's report: one line per draft sentence
  (the support rule above against the context) and, last, `REVIEW_JSON: {"unsupported": [...],
  "supported_count": n, "total": m}`. A context item that is a found record's JSON is not
  prose: the reviewer rebuilds that record's status sentences from it with the Composer's own
  template, and a draft sentence is supported by the record only if it equals one of them
  (D-039). The draft's wording of the record is never trusted.
* `edit(review_json, draft)`: the Final_Editor's `ReviewVerdict`. Nothing unsupported →
  approved, `final_answer` is the draft byte for byte. Otherwise the unsupported sentences are
  removed (the rest keep their order and text); nothing left → `FALLBACK_ANSWER`. A report that
  does not describe this draft (other sentences, another count) raises `ValueError`.

Nothing here knows any query: behaviour comes only from the query tokens and the chunks (N7).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, get_args

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from credagent.rag.retriever import NO_MATCH_SIMILARITY
from credagent.rag.store import RetrievedChunk
from credagent.schemas import (
    FALLBACK_ANSWER,
    RECORD_ID_PATTERN,
    ApplicationView,
    CrewResponse,
    Escalation,
    GeneratedAnswer,
    Intent,
    LoanCategory,
    LoanStatus,
    LoanStatusFound,
    LookupFound,
    LookupNotFound,
    RagToolError,
    RefusalReason,
    RetrievalTaskOutput,
    ReviewVerdict,
    SourceRef,
    loan_type,
)
from credagent.text import (
    RECORD_ID_CANDIDATE,
    content_tokens,
    extract_numbers,
    find_record_id,
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
# PII redaction tokens (§ARCH 7.12.1, e.g. "[PAN_REDACTED]") and the words that announce them
# ("My PAN is …", "a/c no. …") say nothing about what is asked, so intent ignores them (D-037).
REDACTION_TOKEN: Final = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*_REDACTED\]")
PII_WORDS: Final = frozenset(
    {"pan", "aadhaar", "aadhar", "bank", "account", "acct", "no", "number"}
)
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
    "ungrounded": "I can't answer that from the policy documents I have. Please contact support.",
    "injection_detected": (
        "I can't help with that request. Please ask a question about Cred's loans or banking "
        "policies."
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


def _status_sentence(app: LoanStatusFound) -> str:
    escalation = (
        f"It has been referred to a human specialist (escalation score {app.escalation_score} "
        f"≥ threshold {app.escalation_threshold})."
        if app.escalation_recommended
        else f"No escalation is needed (escalation score {app.escalation_score} < threshold "
        f"{app.escalation_threshold})."
    )
    return (
        f"Application {app.record_id} ({app.category}) is currently {app.status}. The loan "
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
    def support_report(
        self, answer: str, context: Sequence[str], *, exact: Collection[str] = ()
    ) -> SupportReport:
        """`exact`: sentences that support only an identical answer sentence (D-039)."""
        context_sentences = [s for c in context for _, s in _split(c)]
        context_numbers = {n for c in context for n in extract_numbers(c)}
        out: list[SentenceSupport] = []
        for _, sentence in _split(answer):
            if sentence in exact:
                out.append(SentenceSupport(sentence, sentence, 1.0, ()))
                continue
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
        # D-056: a question about one loan type is not answered with a sentence that names only
        # other loan types ("Fixed-rate personal loans …" does not answer a home-loan question).
        asked = mentioned_category(query)
        scored = []
        for cand in self._candidates(chunks):
            named = mentioned_categories(cand.text)
            if asked is not None and named and asked not in named:
                continue
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

    # ------------------------------------------------------------------ review (Phase 8)
    def context_support(self, answer: str, context: Sequence[str]) -> SupportReport:
        """The support rule against a review context: prose documents plus found-record JSON,
        whose status sentences support only an identical sentence (D-039). Shared by the
        reviewer and the judge (§ARCH 7.18)."""
        records, documents = _records(context)
        canonical = {s for r in records for _, s in _split(_status_sentence(r))}
        return self.support_report(answer, documents, exact=canonical)

    @staticmethod
    def context_numbers(context: Sequence[str]) -> set[str]:
        """Every number the context states: its documents' and its records' status sentences."""
        records, documents = _records(context)
        texts = [*documents, *(_status_sentence(r) for r in records)]
        return {n for t in texts for n in extract_numbers(t)}

    def review(self, draft: str, context: Sequence[str]) -> str:
        report = self.context_support(draft, context)
        lines = [
            f"Policy compliance review: {report.supported_count} of {report.total} sentences are "
            "supported by the retrieved context."
        ]
        for i, s in enumerate(report.sentences, start=1):
            detail = f"token recall {s.recall:.2f}"
            if s.numbers_missing:
                detail += "; numbers not in the context: " + ", ".join(s.numbers_missing)
            lines.append(f'{i}. {"supported" if s.supported else "UNSUPPORTED"} ({detail}): '
                         f'"{s.sentence}"')  # fmt: skip
        findings = ReviewFindings(
            unsupported=tuple(s.sentence for s in report.sentences if not s.supported),
            supported_count=report.supported_count, total=report.total,
        )  # fmt: skip
        lines.append(REVIEW_JSON_PREFIX + findings.model_dump_json())
        return "\n".join(lines)

    def edit(self, review_json: str, draft: str) -> ReviewVerdict:
        findings = ReviewFindings.model_validate_json(review_json)
        sentences = [s for _, s in _split(draft)]
        unsupported = set(findings.unsupported)
        if len(sentences) != findings.total or not unsupported <= set(sentences):
            raise ValueError("the review does not describe this draft")
        if not unsupported:
            return ReviewVerdict(
                approved=True, final_answer=draft,
                reason=f"All {findings.total} sentences are supported by the retrieved context.",
            )  # fmt: skip
        kept = [s for s in sentences if s not in unsupported]
        removed = "; ".join(f'"{s}"' for s in findings.unsupported)
        reason = (
            f"Removed {len(sentences) - len(kept)} of {findings.total} sentences not supported by "
            f"the retrieved context: {removed}"
        )
        if not kept:
            reason += " Nothing supported remained, so the safe fallback is sent."
        return ReviewVerdict(
            approved=False, final_answer=" ".join(kept) if kept else FALLBACK_ANSWER, reason=reason
        )

    # ------------------------------------------------------------------ Composer (Phase 6)
    # ------------------------------------------------------------------ memory (Phase 7)
    def is_status_only(self, query: str) -> bool:
        """With record ids and PII redactions removed, every content token is a generic status word
        (D-032, D-037)."""
        return all(t in STATUS_TERMS for t in _intent_tokens(query))

    def resolve_followup(self, query: str, history: Sequence[BaseMessage]) -> FollowUp:
        """§ARCH 7.10. State is read only from `history` (AI messages' `additional_kwargs`), so a
        fresh session has none. A query with its own record id is never rewritten."""
        record_id = find_record_id(query)
        refs = [m for m in _REFERENCE.finditer(query) if not _bound_in_sentence(query, m)]
        if record_id is not None or not refs:
            return FollowUp(query, record_id, mentioned_category(query), resolved=False,
                            clarify=False)  # fmt: skip
        last_id, last_category = _history_state(history)
        if self.is_status_only(_REFERENCE.sub(" ", query)):
            if last_id is None:
                return FollowUp(query, None, None, resolved=False, clarify=True)
            noun, rid, category = f"application {last_id}", last_id, last_category
        elif last_category is not None:
            noun, rid, category = _loan_phrase(last_category), None, last_category
        elif last_id is not None:
            noun, rid, category = f"application {last_id}", last_id, None
        else:
            return FollowUp(query, None, None, resolved=False, clarify=True)
        return FollowUp(_rewrite(query, refs, noun), rid, category, resolved=True, clarify=False)

    def classify_intent(self, query: str, record_id: str | None) -> Intent:
        tokens = _intent_tokens(query)
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


def _intent_tokens(query: str) -> list[str]:
    """The content tokens that say what is asked: no record id, no PII redaction (D-037)."""
    text = RECORD_ID_CANDIDATE.sub(" ", query)
    if not REDACTION_TOKEN.search(text):
        return content_tokens(text)
    return [t for t in content_tokens(REDACTION_TOKEN.sub(" ", text)) if t not in PII_WORDS]


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


# ------------------------------------------------------------------ review (Phase 8)
REVIEW_JSON_PREFIX: Final = "REVIEW_JSON: "


def _records(context: Sequence[str]) -> tuple[list[LoanStatusFound], list[str]]:
    """Split the review context into found-record JSON items and the other documents."""
    records: list[LoanStatusFound] = []
    documents: list[str] = []
    for item in context:
        try:
            records.append(LoanStatusFound.model_validate_json(item))
        except ValidationError:
            documents.append(item)
    return records, documents


class ReviewFindings(BaseModel):
    """The reviewer's machine-readable last line (§ARCH 7.8.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unsupported: tuple[str, ...]
    supported_count: int = Field(ge=0)
    total: int = Field(ge=0)


# ------------------------------------------------------------------ follow-up resolution (Phase 7)
# References to an earlier application or loan (§ARCH 7.10). "it" counts only after a
# preposition or an auxiliary ("for it", "does it"); "is it" never does, because "is it possible
# …" is far more often an expletive (D-036).
_REFERENCE: Final = re.compile(
    r"\b(?:(?P<poss>its)"
    r"|(?P<noun>(?:this|that|my|the same) (?:application|loan)|the same one)"
    r"|(?P<verb>for|on|about|of|with|to|from|does|did|do|will|would|can|could|has|have|had"
    r"|should|was|were) it)\b",
    re.IGNORECASE,
)
# A noun phrase ("my account", "the card") earlier in the same sentence: a possessive "its" after
# it refers to that phrase, not to an earlier turn (BUG-003).
_ANTECEDENT: Final = re.compile(r"\b(?:my|the|a|an|this|that|your|our) [a-z]", re.IGNORECASE)
_SENTENCE_END: Final = re.compile(r"[.?!](?:\s|$)")
# "it" in a later clause than such a phrase ("the card … and can it be waived") refers to that
# phrase; within one clause ("the eligibility criteria for it") it does not (BUG-006).
_CLAUSE_BREAK: Final = re.compile(r"[,;]|\b(?:and|or|but)\b", re.IGNORECASE)
# Extraposition: in "does it take/cost … to <verb>" the "it" stands for the to-clause (BUG-006).
_EXTRAPOSED: Final = re.compile(r"\s+(?:take|takes|took|cost|costs)\b[^.?!]*?\bto [a-z]",
                                re.IGNORECASE)  # fmt: skip
LOAN_CATEGORY: Final[TypeAdapter[LoanCategory]] = TypeAdapter(LoanCategory)
_RECORD_ID: Final = re.compile(RECORD_ID_PATTERN)
_CATEGORY_BY_TYPE: Final[dict[str, LoanCategory]] = {
    loan_type(c).casefold(): c for c in get_args(LoanCategory)
}
_CATEGORY_MENTION: Final = re.compile(
    r"\b(" + "|".join(loan_type(c) for c in get_args(LoanCategory)) + r") loans?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FollowUp:
    """A standalone query: `record_id` is the id it carries, `category` the loan type it is
    about (if any), `resolved` whether history rewrote it, `clarify` whether it refers to
    something history cannot resolve."""

    query: str
    record_id: str | None
    category: LoanCategory | None
    resolved: bool
    clarify: bool


def mentioned_categories(text: str) -> frozenset[LoanCategory]:
    """Every loan category the text names as "<type> loan(s)" ("home loans" → "Home Loan")."""
    return frozenset(
        _CATEGORY_BY_TYPE[m.group(1).casefold()] for m in _CATEGORY_MENTION.finditer(text)
    )


def mentioned_category(query: str) -> LoanCategory | None:
    """The one loan category the query names as "<type> loan", or `None`."""
    found = mentioned_categories(query)
    if len(found) != 1:
        return None
    (category,) = found
    return category


def _history_state(history: Sequence[BaseMessage]) -> tuple[str | None, LoanCategory | None]:
    """The most recent valid record id and loan category in AI messages' `additional_kwargs`."""
    record_id: str | None = None
    category: LoanCategory | None = None
    for message in reversed(history):
        if not isinstance(message, AIMessage):
            continue
        kwargs = message.additional_kwargs
        rid, cat = kwargs.get("record_id"), kwargs.get("category")
        if record_id is None and isinstance(rid, str) and _RECORD_ID.fullmatch(rid):
            record_id = rid
        if category is None and cat in get_args(LoanCategory):
            category = LOAN_CATEGORY.validate_python(cat)
        if record_id is not None and category is not None:
            break
    return record_id, category


def _loan_phrase(category: LoanCategory) -> str:
    word = loan_type(category)
    return f"{'an' if word[0] in 'AEIOU' else 'a'} {word} loan"


def _bound_in_sentence(query: str, ref: re.Match[str]) -> bool:
    """A possessive "its" with an antecedent noun phrase before it in its own sentence (BUG-003);
    an "it" with one in an earlier clause of its sentence, or an extraposed "it" (BUG-006)."""
    if ref["noun"]:
        return False
    start = max((e.end() for e in _SENTENCE_END.finditer(query, 0, ref.start())), default=0)
    antecedent = _ANTECEDENT.search(query, start, ref.start())
    if ref["poss"]:
        return antecedent is not None
    if _EXTRAPOSED.match(query, ref.end()):
        return True
    return antecedent is not None and (
        _CLAUSE_BREAK.search(query, antecedent.end(), ref.start()) is not None
    )


def _rewrite(query: str, refs: Sequence[re.Match[str]], noun: str) -> str:
    out, end = [], 0
    for m in refs:
        if m["poss"]:
            replacement = f"{noun}'s"
        elif m["verb"]:
            replacement = f"{m['verb']} {noun}"
        else:
            replacement = noun
        if m.start() == 0:
            replacement = replacement[0].upper() + replacement[1:]
        out += [query[end : m.start()], replacement]
        end = m.end()
    return "".join([*out, query[end:]])
