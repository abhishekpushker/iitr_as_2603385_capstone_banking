"""Output-side groundedness gate (§ARCH 7.12.3) — pipeline stage 8, after review.

`GroundednessGuard.check(final_answer, response, trace, lookup) -> GuardDecision` judges the
answer that will actually be sent (review may have removed sentences) and either passes it,
with `response.answer` replaced by `final_answer`, or converts it into a refusal.

* Refusals pass through, and so do `clarify` answers (the clarifying question, or "no
  application with id …", claim nothing; `not_applicable`).
* Answers that needed retrieval — intents `policy`, `mixed` and `out_of_scope` (a policy
  question whose retrieval fell short, D-036) — are refused when `trace` is missing
  (`retrieval_missing` → `retrieval_unavailable`) or `top1_similarity < threshold`
  (`below_threshold`).
* Then every sentence must be supported. A sentence supported by the retrieved chunk texts
  (`MockEngine.support_report`: token recall ≥ 0.6 and every number present) has basis `policy`.
  Otherwise, when there is an application record (`status` / `mixed`), a sentence that is
  *anchored* to the record (it names the record id, a loan status, "application"/"escalation",
  or carries a number) has basis `record` and must agree with it: every number is one of the
  record's (id, amount, age, escalation score, threshold), and every record id, loan status and
  loan category it names is the record's. A disagreement is `lookup_mismatch`; any other
  sentence is `unsupported_sentence`.

Every refusal is `ungrounded` ("I can't answer that from the policy documents I have. Please
contact support."), except a missing retrieval, which is `retrieval_unavailable`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

from pydantic import BaseModel, ConfigDict

from credagent.llm.engine import MockEngine
from credagent.schemas import (
    CrewResponse,
    Intent,
    LoanCategory,
    LoanStatus,
    LookupFound,
    RefusalReason,
)
from credagent.text import RECORD_ID_CANDIDATE, extract_numbers, normalize, split_sentences

GuardReason = Literal[
    "passed", "not_applicable", "below_threshold", "unsupported_sentence", "lookup_mismatch",
    "retrieval_missing",
]  # fmt: skip
Basis = Literal["policy", "record", "none"]

NEEDS_RETRIEVAL: Final[frozenset[Intent]] = frozenset({"policy", "mixed", "out_of_scope"})
HAS_RECORD: Final[frozenset[Intent]] = frozenset({"status", "mixed"})


def _phrases(values: Sequence[str]) -> dict[str, re.Pattern[str]]:
    return {v: re.compile(rf"\b{re.escape(normalize(v))}\b") for v in values}


_STATUSES: Final = _phrases(get_args(LoanStatus))
_CATEGORIES: Final = _phrases(get_args(LoanCategory))
_ANCHOR_WORDS: Final = re.compile(r"\b(?:application|escalation|escalated)\b")


@dataclass(frozen=True)
class RetrievalTrace:
    """What retrieval saw: the tool's top-1 similarity and threshold, and the texts of the chunks
    the answer was drawn from."""

    top1_similarity: float
    threshold: float
    contexts: tuple[str, ...]


class SentenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sentence: str
    basis: Basis
    supported: bool


class GuardDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason: GuardReason
    response: CrewResponse
    top1_similarity: float | None
    threshold: float | None
    checks: tuple[SentenceCheck, ...]


def _record_numbers(app: LookupFound) -> set[str]:
    values = (app.record_id, app.loan_amount_inr, app.days_since_created, app.escalation_score,
              app.escalation_threshold)  # fmt: skip
    return set(extract_numbers(" ".join(map(str, values))))


def _anchored(sentence: str) -> bool:
    norm = normalize(sentence)
    return bool(
        RECORD_ID_CANDIDATE.search(sentence)
        or extract_numbers(sentence)
        or _ANCHOR_WORDS.search(norm)
        or any(p.search(norm) for p in _STATUSES.values())
    )


def _agrees_with_record(sentence: str, app: LookupFound) -> bool:
    norm = normalize(sentence)
    return (
        set(extract_numbers(sentence)) <= _record_numbers(app)
        and all(m.upper() == app.record_id for m in RECORD_ID_CANDIDATE.findall(sentence))
        and all(v == app.status for v, p in _STATUSES.items() if p.search(norm))
        and all(v == app.category for v, p in _CATEGORIES.items() if p.search(norm))
    )


class GroundednessGuard:
    def __init__(self, engine: MockEngine | None = None) -> None:
        self._engine = MockEngine() if engine is None else engine

    def check(
        self,
        final_answer: str,
        response: CrewResponse,
        trace: RetrievalTrace | None,
        lookup: LookupFound | None,
    ) -> GuardDecision:
        top1 = None if trace is None else trace.top1_similarity
        threshold = None if trace is None else trace.threshold
        if response.refused or response.intent == "clarify":
            return GuardDecision(
                allowed=True, reason="not_applicable", response=response, top1_similarity=top1,
                threshold=threshold, checks=(),
            )  # fmt: skip
        needs_retrieval = response.intent in NEEDS_RETRIEVAL
        if needs_retrieval and trace is None:
            return self._refuse("retrieval_missing", top1, threshold, ())
        if trace is not None and needs_retrieval and trace.top1_similarity < trace.threshold:
            return self._refuse("below_threshold", top1, threshold, ())

        app = lookup if response.intent in HAS_RECORD else None
        contexts = trace.contexts if trace is not None and needs_retrieval else ()
        checks = self._sentence_checks(final_answer, contexts, app)
        if response.intent in HAS_RECORD and app is None:
            return self._refuse("lookup_mismatch", top1, threshold, checks)
        failed = [c for c in checks if not c.supported]
        if failed:
            reason: GuardReason = (
                "lookup_mismatch" if any(c.basis == "record" for c in failed)
                else "unsupported_sentence"
            )  # fmt: skip
            return self._refuse(reason, top1, threshold, checks)
        passed = CrewResponse.model_validate(response.model_dump() | {"answer": final_answer})
        return GuardDecision(
            allowed=True, reason="passed", response=passed, top1_similarity=top1,
            threshold=threshold, checks=checks,
        )  # fmt: skip

    def _sentence_checks(
        self, answer: str, contexts: tuple[str, ...], app: LookupFound | None
    ) -> tuple[SentenceCheck, ...]:
        policy = {
            s.sentence: s.supported for s in self._engine.support_report(answer, contexts).sentences
        }
        out = []
        for sentence in split_sentences(answer):
            if contexts and policy.get(sentence, False):
                out.append(SentenceCheck(sentence=sentence, basis="policy", supported=True))
            elif app is not None and _anchored(sentence):
                ok = _agrees_with_record(sentence, app)
                out.append(SentenceCheck(sentence=sentence, basis="record", supported=ok))
            else:
                out.append(SentenceCheck(sentence=sentence, basis="none", supported=False))
        return tuple(out)

    def _refuse(
        self,
        reason: GuardReason,
        top1: float | None,
        threshold: float | None,
        checks: tuple[SentenceCheck, ...],
    ) -> GuardDecision:
        refusal_reason: RefusalReason = (
            "retrieval_unavailable" if reason == "retrieval_missing" else "ungrounded"
        )
        return GuardDecision(
            allowed=False, reason=reason, response=self._engine.refusal(refusal_reason),
            top1_similarity=top1, threshold=threshold, checks=checks,
        )  # fmt: skip
