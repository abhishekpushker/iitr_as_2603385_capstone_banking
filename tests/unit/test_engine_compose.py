"""`MockEngine.classify_intent` / `compose` / `refusal` (§ARCH 7.8, 7.11) and
`text.find_record_id`. Every output validates as a `CrewResponse`."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent.llm.engine import CLARIFY_ANSWER, REFUSAL_ANSWERS, MockEngine, format_inr
from credagent.schemas import (
    FALLBACK_ANSWER,
    LookupFound,
    LookupNotFound,
    RagToolError,
    RetrievalTaskOutput,
    RetrievalToolOutput,
    SourceRef,
)
from credagent.text import find_record_id

ENGINE = MockEngine()
SOURCE = SourceRef(doc_id="kb-04", chunk_id="kb-04#s3", similarity=0.71)
GROUNDED = RetrievalTaskOutput(
    kind="retrieval_result",
    rag=RetrievalToolOutput(
        answer="Floating-rate home loans carry no prepayment penalty.", sources=(SOURCE,),
        grounded=True, top1_similarity=0.71, strategy="sentence", threshold=0.42,
    ),
)  # fmt: skip
FALLBACK = RetrievalTaskOutput(
    kind="retrieval_result",
    rag=RetrievalToolOutput(
        answer=FALLBACK_ANSWER, sources=(), grounded=False, top1_similarity=0.1,
        strategy="sentence", threshold=0.42,
    ),
)  # fmt: skip
RAG_ERROR = RetrievalTaskOutput(
    kind="retrieval_result", rag=RagToolError(error="retrieval_unavailable")
)
SKIPPED = RetrievalTaskOutput(kind="retrieval_result", rag=None)


def found(**over: Any) -> LookupFound:
    base: dict[str, Any] = {
        "kind": "lookup_result", "found": True, "record_id": "LA-0007", "category": "Home",
        "status": "Under Review", "loan_amount_inr": 2500000, "days_since_created": 12,
        "flagged_for_fraud_review": False, "escalation_score": 0.24,
        "escalation_recommended": False, "escalation_threshold": 0.5,
    }  # fmt: skip
    return LookupFound.model_validate(base | over)


def not_found(error: str) -> LookupNotFound:
    return LookupNotFound.model_validate({"kind": "lookup_result", "found": False, "error": error})


NO_ID = not_found("no_record_id")


# ------------------------------------------------------------------ record ids


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is the status of application LA-0007?", "LA-0007"),
        ("status of la-0042 please", "LA-0042"),
        ("(LA-1234)", "LA-1234"),
        ("LA-07 status", "LA-07"),  # malformed: the tool answers invalid_record_id
        ("first LA-0001 then LA-0002", "LA-0001"),
        ("XLA-0001", None),
        ("LA-0001X", None),
        ("LA-123456789", None),
        ("LA-٠٠٠١", None),  # Arabic-Indic digits are not ASCII digits
        ("What is the prepayment penalty?", None),
    ],
)
def test_find_record_id(text: str, expected: str | None) -> None:
    assert find_record_id(text) == expected


# ------------------------------------------------------------------ intent


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("What is the status of application LA-0007?", "status"),
        ("Check my home loan LA-0007", "status"),
        ("Is LA-0007 approved yet?", "status"),
        ("LA-0007", "status"),
        ("What is the status of LA-0007 and is there a prepayment penalty?", "mixed"),
        ("LA-0007: what documents do I need for KYC?", "mixed"),
        ("What is the status of my application?", "clarify"),
        ("Any update on my loan application?", "clarify"),
        ("Is there a prepayment penalty on home loans?", "policy"),
        ("Tell me about home loans", "policy"),
        ("What's the weather in Paris?", "policy"),
    ],
)
def test_classify_intent(query: str, intent: str) -> None:
    assert ENGINE.classify_intent(query, find_record_id(query)) == intent


# ------------------------------------------------------------------ compose


def test_policy_grounded() -> None:
    r = ENGINE.compose(GROUNDED, NO_ID, query="Is there a prepayment penalty?", record_id=None)
    assert (r.intent, r.grounded, r.refused) == ("policy", True, False)
    assert r.answer == GROUNDED.rag.answer  # type: ignore[union-attr]
    assert r.sources == (SOURCE,)
    assert r.application is None and r.escalation is None


def test_policy_fallback_is_out_of_scope() -> None:
    r = ENGINE.compose(FALLBACK, NO_ID, query="What's the weather?", record_id=None)
    assert (r.intent, r.answer, r.grounded, r.sources) == (
        "out_of_scope", FALLBACK_ANSWER, False, ()
    )  # fmt: skip


def test_policy_rag_error_is_a_refusal() -> None:
    r = ENGINE.compose(RAG_ERROR, NO_ID, query="Is there a prepayment penalty?", record_id=None)
    assert (r.intent, r.refused, r.refusal_reason) == ("refused", True, "retrieval_unavailable")
    assert r.answer == REFUSAL_ANSWERS["retrieval_unavailable"]


def test_status_found() -> None:
    r = ENGINE.compose(SKIPPED, found(), query="Status of LA-0007?", record_id="LA-0007")
    assert (r.intent, r.grounded, r.sources) == ("status", True, ())
    assert r.application is not None and r.application.record_id == "LA-0007"
    assert r.escalation is not None
    assert (r.escalation.score, r.escalation.threshold, r.escalation.recommended) == (
        0.24, 0.5, False
    )  # fmt: skip
    assert r.answer == (
        "Application LA-0007 (Home loan) is currently Under Review. The loan amount is "
        "₹25,00,000 and the application was created 12 days ago. No escalation is needed "
        "(escalation score 0.24 < threshold 0.5)."
    )


def test_status_escalated_mentions_the_specialist_but_never_the_fraud_flag() -> None:
    r = ENGINE.compose(
        SKIPPED,
        found(flagged_for_fraud_review=True, escalation_score=0.6, escalation_recommended=True,
              days_since_created=1),
        query="Status of LA-0007?", record_id="LA-0007",
    )  # fmt: skip
    assert "created 1 day ago" in r.answer
    assert "referred to a human specialist (escalation score 0.6 ≥ threshold 0.5)" in r.answer
    assert "fraud" not in r.answer.casefold()
    assert r.application is not None and r.application.flagged_for_fraud_review is True


def test_status_created_today() -> None:
    r = ENGINE.compose(SKIPPED, found(days_since_created=0), query="LA-0007", record_id="LA-0007")
    assert "created today" in r.answer


@pytest.mark.parametrize(
    ("error", "rid", "fragment"),
    [
        ("not_found", "LA-9999", "couldn't find a loan application with id LA-9999"),
        ("invalid_record_id", "LA-07", "LA-07 is not a valid application id"),
    ],
)
def test_status_unknown_or_invalid_id_is_a_polite_clarification(
    error: str, rid: str, fragment: str
) -> None:
    r = ENGINE.compose(SKIPPED, not_found(error), query=f"Status of {rid}?", record_id=rid)
    assert (r.intent, r.grounded, r.application, r.refused) == ("clarify", True, None, False)
    assert fragment in r.answer
    assert "(format LA-0000)" in r.answer


def test_clarify_without_id() -> None:
    r = ENGINE.compose(SKIPPED, NO_ID, query="What is the status of my application?",
                       record_id=None)  # fmt: skip
    assert (r.intent, r.answer, r.grounded) == ("clarify", CLARIFY_ANSWER, True)


def test_mixed_grounded() -> None:
    q = "What is the status of LA-0007 and is there a prepayment penalty?"
    r = ENGINE.compose(GROUNDED, found(), query=q, record_id="LA-0007")
    assert (r.intent, r.grounded, r.sources) == ("mixed", True, (SOURCE,))
    assert r.application is not None and r.escalation is not None
    assert r.answer.startswith("Application LA-0007 (Home loan) is currently Under Review.")
    assert r.answer.endswith(GROUNDED.rag.answer)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("retrieval", "tail"),
    [(FALLBACK, FALLBACK_ANSWER), (RAG_ERROR, REFUSAL_ANSWERS["retrieval_unavailable"])],
    ids=["fallback", "rag-error"],
)
def test_mixed_without_a_grounded_policy_part_keeps_the_status(
    retrieval: RetrievalTaskOutput, tail: str
) -> None:
    q = "What is the status of LA-0007 and is there a prepayment penalty?"
    r = ENGINE.compose(retrieval, found(), query=q, record_id="LA-0007")
    assert (r.intent, r.grounded, r.sources, r.refused) == ("mixed", False, (), False)
    assert r.application is not None
    assert r.answer.endswith(tail)


def test_mixed_with_unknown_id() -> None:
    q = "What is the status of LA-9999 and is there a prepayment penalty?"
    r = ENGINE.compose(GROUNDED, not_found("not_found"), query=q, record_id="LA-9999")
    assert (r.intent, r.application, r.escalation, r.grounded) == ("mixed", None, None, True)
    assert r.answer.startswith("I couldn't find a loan application with id LA-9999.")


# ------------------------------------------------------------------ refusals and formatting


@pytest.mark.parametrize("reason", sorted(REFUSAL_ANSWERS))
def test_refusal(reason: str) -> None:
    r = ENGINE.refusal(reason)  # type: ignore[arg-type]
    assert (r.intent, r.refused, r.refusal_reason, r.grounded) == ("refused", True, reason, False)
    assert r.answer == REFUSAL_ANSWERS[reason]  # type: ignore[index]


@pytest.mark.parametrize(
    ("amount", "text"),
    [(1, "₹1"), (999, "₹999"), (1000, "₹1,000"), (150000, "₹1,50,000"),
     (2500000, "₹25,00,000"), (123456789, "₹12,34,56,789")],
)  # fmt: skip
def test_format_inr_uses_indian_grouping(amount: int, text: str) -> None:
    assert format_inr(amount) == text


@given(st.integers(min_value=1, max_value=10**12))
def test_format_inr_round_trips(amount: int) -> None:
    assert int(format_inr(amount).removeprefix("₹").replace(",", "")) == amount


@pytest.mark.parametrize(
    ("retrieval", "lookup", "query"),
    [
        (SKIPPED, NO_ID, "Is there a prepayment penalty?"),  # policy question, retrieval skipped
        (GROUNDED, NO_ID, "Status of LA-0007?"),  # status question, lookup skipped
        (GROUNDED, found(), "What is the status of my application?"),  # clarify, but looked up
        (SKIPPED, found(), "Status of LA-0007 and the prepayment penalty?"),  # mixed, no rag
    ],
    ids=["policy-no-rag", "status-no-lookup", "clarify-with-lookup", "mixed-no-rag"],
)
def test_inputs_that_contradict_the_intent_fail_closed(
    retrieval: RetrievalTaskOutput, lookup: LookupFound | LookupNotFound, query: str
) -> None:
    r = ENGINE.compose(retrieval, lookup, query=query, record_id=find_record_id(query))
    assert (r.refused, r.refusal_reason) == (True, "agent_error")
