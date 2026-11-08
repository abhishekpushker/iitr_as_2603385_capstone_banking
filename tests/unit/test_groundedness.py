"""Output-side groundedness gate (§ARCH 7.12.3), pipeline stage 8."""

from __future__ import annotations

from typing import Any

import pytest

from credagent.guardrails.groundedness import GroundednessGuard, RetrievalTrace
from credagent.llm.engine import REFUSAL_ANSWERS, MockEngine
from credagent.schemas import (
    FALLBACK_ANSWER,
    CrewResponse,
    LookupFound,
    LookupNotFound,
    RetrievalTaskOutput,
    RetrievalToolOutput,
    SourceRef,
)

ENGINE = MockEngine()
GUARD = GroundednessGuard()
THRESHOLD = 0.567
KB08 = (
    "Floating-rate home loans taken by individuals carry no prepayment penalty. Fixed-rate "
    "personal loans charge a prepayment penalty of 4% of the prepaid principal during the first "
    "12 EMIs and 2% after that."
)
POLICY_ANSWER = "Floating-rate home loans taken by individuals carry no prepayment penalty."
SOURCE = SourceRef(doc_id="kb-08-prepayment-penalty", chunk_id="kb-08#s0", similarity=0.81)
UNGROUNDED = REFUSAL_ANSWERS["ungrounded"]


def trace(top1: float = 0.81, contexts: tuple[str, ...] = (KB08,)) -> RetrievalTrace:
    return RetrievalTrace(top1_similarity=top1, threshold=THRESHOLD, contexts=contexts)


def rag(
    answer: str = POLICY_ANSWER, top1: float = 0.81, grounded: bool = True
) -> RetrievalTaskOutput:
    return RetrievalTaskOutput(
        kind="retrieval_result",
        rag=RetrievalToolOutput(
            answer=answer, sources=(SOURCE,) if grounded else (), grounded=grounded,
            top1_similarity=top1, strategy="sentence", threshold=THRESHOLD,
        ),
    )  # fmt: skip


def found(**over: Any) -> LookupFound:
    base: dict[str, Any] = {
        "kind": "lookup_result", "found": True, "record_id": "LA-0007", "category": "Home Loan",
        "status": "Under Review", "loan_amount_inr": 2500000, "days_since_created": 12,
        "flagged_for_fraud_review": False, "escalation_score": 0.24,
        "escalation_recommended": False, "escalation_threshold": 0.5,
    }  # fmt: skip
    return LookupFound.model_validate(base | over)


NO_ID = LookupNotFound(kind="lookup_result", found=False, error="no_record_id")
POLICY_Q = "Is there a prepayment penalty on a floating-rate home loan?"
STATUS_Q = "What is the status of application LA-0007?"
MIXED_Q = "What is the status of LA-0007, and is there a prepayment penalty on a home loan?"


def policy_response() -> CrewResponse:
    return ENGINE.compose(rag(), NO_ID, query=POLICY_Q, record_id=None)


def status_response(app: LookupFound | None = None) -> CrewResponse:
    skipped = RetrievalTaskOutput(kind="retrieval_result", rag=None)
    return ENGINE.compose(skipped, app or found(), query=STATUS_Q, record_id="LA-0007")


def mixed_response(app: LookupFound | None = None) -> CrewResponse:
    return ENGINE.compose(rag(), app or found(), query=MIXED_Q, record_id="LA-0007")


def assert_refused(decision: Any, reason: str, refusal_reason: str = "ungrounded") -> None:
    assert not decision.allowed
    assert decision.reason == reason
    r = decision.response
    assert (r.intent, r.refused, r.refusal_reason, r.grounded) == (
        "refused", True, refusal_reason, False,
    )  # fmt: skip
    assert r.answer == REFUSAL_ANSWERS[refusal_reason]  # type: ignore[index]


# ------------------------------------------------------------------ policy: similarity threshold
def test_grounded_policy_answer_passes_unchanged() -> None:
    response = policy_response()
    decision = GUARD.check(response.answer, response, trace(), None)
    assert decision.allowed and decision.reason == "passed"
    assert decision.response == response
    assert [c.basis for c in decision.checks] == ["policy"]


def test_below_threshold_is_refused_even_if_every_sentence_is_supported() -> None:
    response = policy_response()
    decision = GUARD.check(response.answer, response, trace(top1=0.555), None)
    assert_refused(decision, "below_threshold")
    assert decision.top1_similarity == 0.555 and decision.threshold == THRESHOLD


def test_exactly_at_threshold_passes() -> None:
    response = policy_response()
    assert GUARD.check(response.answer, response, trace(top1=THRESHOLD), None).allowed


def test_out_of_scope_fallback_is_converted_into_the_ungrounded_refusal() -> None:
    oos = rag(FALLBACK_ANSWER, top1=0.555, grounded=False)
    response = ENGINE.compose(oos, NO_ID, query="gold-loan penalty?", record_id=None)
    assert response.intent == "out_of_scope"
    decision = GUARD.check(response.answer, response, trace(top1=0.555, contexts=()), None)
    assert_refused(decision, "below_threshold")
    assert (
        UNGROUNDED
        == "I can't answer that from the policy documents I have. Please contact support."
    )


# ------------------------------------------------------------ policy: every sentence supported
@pytest.mark.parametrize(
    "extra",
    [
        "Gold loans are approved within an hour.",  # no support in the context
        # a fabricated number
        "Fixed-rate personal loans charge a prepayment penalty of 0% for life.",
    ],
)
def test_one_unsupported_sentence_refuses_the_whole_answer(extra: str) -> None:
    response = policy_response()
    decision = GUARD.check(f"{response.answer} {extra}", response, trace(), None)
    assert_refused(decision, "unsupported_sentence")
    assert [c.supported for c in decision.checks] == [True, False]
    assert decision.checks[1].sentence == extra


def test_the_final_answer_is_checked_not_the_draft() -> None:
    """Review (stage 7) may remove sentences; the gate judges what will be sent."""
    draft = policy_response().model_copy(
        update={"answer": f"{POLICY_ANSWER} Gold loans are approved within an hour."}
    )
    decision = GUARD.check(POLICY_ANSWER, draft, trace(), None)
    assert decision.allowed
    assert decision.response.answer == POLICY_ANSWER


def test_policy_answer_without_retrieval_trace_fails_closed() -> None:
    response = policy_response()
    decision = GUARD.check(response.answer, response, None, None)
    assert_refused(decision, "retrieval_missing", refusal_reason="retrieval_unavailable")


# ------------------------------------------------------------------ lookup sentences vs the record
def test_status_answer_matching_the_record_passes() -> None:
    response = status_response()
    decision = GUARD.check(response.answer, response, None, found())
    assert decision.allowed
    assert {c.basis for c in decision.checks} == {"record"}
    assert len(decision.checks) == 3


@pytest.mark.parametrize(
    ("good", "bad"),
    [
        ("₹25,00,000", "₹26,00,000"),  # amount
        ("12 days ago", "2 days ago"),  # age
        ("Under Review", "Approved"),  # status
        ("(Home Loan)", "(Auto Loan)"),  # category
        ("Application LA-0007", "Application LA-0008"),  # record id
        ("score 0.24", "score 0.42"),  # escalation score
    ],
)
def test_lookup_sentence_that_contradicts_the_record_is_refused(good: str, bad: str) -> None:
    response = status_response()
    assert good in response.answer
    decision = GUARD.check(response.answer.replace(good, bad), response, None, found())
    assert_refused(decision, "lookup_mismatch")


def test_unanchored_sentence_in_a_status_answer_is_refused() -> None:
    response = status_response()
    decision = GUARD.check(
        f"{response.answer} You also qualify for a top-up loan.", response, None, found()
    )
    assert_refused(decision, "unsupported_sentence")


def test_status_answer_without_the_lookup_record_fails_closed() -> None:
    response = status_response()
    assert_refused(GUARD.check(response.answer, response, None, None), "lookup_mismatch")


def test_mixed_answer_checks_status_against_the_record_and_policy_against_the_chunks() -> None:
    response = mixed_response()
    decision = GUARD.check(response.answer, response, trace(), found())
    assert decision.allowed
    assert [c.basis for c in decision.checks] == ["record", "record", "record", "policy"]


def test_mixed_answer_with_a_wrong_amount_is_refused() -> None:
    response = mixed_response()
    bad = response.answer.replace("₹25,00,000", "₹2,50,000")
    assert_refused(GUARD.check(bad, response, trace(), found()), "lookup_mismatch")


def test_mixed_answer_below_threshold_is_refused() -> None:
    response = mixed_response()
    assert_refused(
        GUARD.check(response.answer, response, trace(top1=0.4), found()), "below_threshold"
    )


# ------------------------------------------------------------------ not applicable
def test_refusals_pass_through() -> None:
    response = ENGINE.refusal("budget_exceeded")
    decision = GUARD.check(response.answer, response, None, None)
    assert decision.allowed and decision.reason == "not_applicable"
    assert decision.response == response


def test_clarify_passes_through() -> None:
    skipped = RetrievalTaskOutput(kind="retrieval_result", rag=None)
    response = ENGINE.compose(skipped, NO_ID, query="What is my status?", record_id=None)
    assert response.intent == "clarify"
    decision = GUARD.check(response.answer, response, None, None)
    assert decision.allowed and decision.reason == "not_applicable"
