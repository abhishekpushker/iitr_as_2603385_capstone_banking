"""`CrewResponse` and the crew task outputs (§ARCH 7.11, 7.8.1 step 5): every validator, extra
fields forbidden, frozen, JSON round-trip."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from credagent.schemas import (
    LOOKUP_TASK_OUTPUT,
    ApplicationView,
    CrewResponse,
    Escalation,
    LookupFound,
    LookupNotFound,
    RetrievalTaskOutput,
    RetrievalToolOutput,
    SourceRef,
)

SOURCE = {"doc_id": "kb-04", "chunk_id": "kb-04#s3", "similarity": 0.71}
APP = {
    "record_id": "LA-0007", "category": "Home", "status": "Under Review",
    "loan_amount_inr": 2500000, "days_since_created": 12, "flagged_for_fraud_review": False,
}  # fmt: skip
ESC = {"score": 0.24, "threshold": 0.5, "recommended": False}


def response(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "crew_response", "intent": "policy", "answer": "Prepayment is free.",
        "sources": [SOURCE], "application": None, "escalation": None, "grounded": True,
        "refused": False, "refusal_reason": None,
    }  # fmt: skip
    return base | over


VALID = [
    response(),
    response(intent="status", sources=[], application=APP, escalation=ESC),
    response(intent="mixed", application=APP, escalation=ESC),
    response(intent="mixed", grounded=False, sources=[]),
    response(intent="clarify", sources=[], answer="Please share your application id."),
    response(intent="out_of_scope", sources=[], grounded=False),
    response(intent="refused", sources=[], grounded=False, refused=True,
             refusal_reason="budget_exceeded"),
]  # fmt: skip


@pytest.mark.parametrize("data", VALID, ids=lambda d: f"{d['intent']}-{d['grounded']}")
def test_valid_responses_round_trip_through_json(data: dict[str, Any]) -> None:
    model = CrewResponse.model_validate(data)
    assert CrewResponse.model_validate_json(model.model_dump_json()) == model
    assert model.model_dump(mode="json") == data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (response(refused=True, intent="refused", grounded=False, sources=[]), "refusal_reason"),
        (response(refusal_reason="agent_error"), "refusal_reason"),
        (response(intent="refused", grounded=False, sources=[]), "refused"),
        (response(refused=True, refusal_reason="agent_error", grounded=False), "refused"),
        (response(intent="status", sources=[]), "application"),
        (response(sources=[]), "grounded"),
        (response(intent="mixed", sources=[]), "grounded"),
        (response(escalation=ESC), "escalation"),
        (response(refused=True, intent="refused", refusal_reason="agent_error", sources=[]),
         "grounded"),
    ],
    ids=["refused-no-reason", "reason-not-refused", "intent-refused-not-refused",
         "refused-wrong-intent", "status-no-application", "grounded-no-sources",
         "mixed-grounded-no-sources", "escalation-no-application", "refused-grounded"],
)  # fmt: skip
def test_cross_field_validators(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CrewResponse.model_validate(data)


@pytest.mark.parametrize("intent", ["status", "clarify"])
def test_grounded_without_sources_is_allowed_for_status_and_clarify(intent: str) -> None:
    app = APP if intent == "status" else None
    assert CrewResponse.model_validate(response(intent=intent, sources=[], application=app))


@pytest.mark.parametrize(
    "data",
    [
        response(extra="x"),
        response(kind="other"),
        response(intent="chitchat"),
        response(answer=""),
        response(answer="x" * 4001),
        response(refused=True, intent="refused", grounded=False, sources=[],
                 refusal_reason="because"),
        response(sources=[SOURCE | {"similarity": 1.5}]),
        response(intent="status", sources=[], application=APP | {"record_id": "LA-7"}),
        response(intent="status", sources=[], application=APP | {"days_since_created": 31}),
        response(intent="status", sources=[], application=APP | {"loan_amount_inr": 0}),
        response(intent="status", sources=[], application=APP | {"category": "Gold"}),
        response(intent="status", sources=[], application=APP, escalation=ESC | {"score": 2}),
        response(intent="status", sources=[], application=APP | {"x": 1}),
    ],
    ids=["extra", "kind", "intent", "empty-answer", "long-answer", "reason", "similarity",
         "record-id", "days", "amount", "category", "score", "app-extra"],
)  # fmt: skip
def test_field_constraints_and_extra_fields_are_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CrewResponse.model_validate(data)


def test_answer_is_stripped_and_the_model_is_frozen() -> None:
    model = CrewResponse.model_validate(response(answer="  Prepayment is free.  "))
    assert model.answer == "Prepayment is free."
    with pytest.raises(ValidationError):
        model.answer = "x"  # type: ignore[misc]


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CrewResponse.model_validate_json('{"kind": "crew_response", "intent": "policy"')


@given(
    st.text(min_size=1, max_size=200).filter(lambda s: s.strip() == s and s),
    st.floats(min_value=-1, max_value=1, allow_nan=False),
)
def test_round_trip_property(answer: str, similarity: float) -> None:
    source = SourceRef(doc_id="d", chunk_id="d#1", similarity=similarity)
    model = CrewResponse.model_validate(response(answer=answer, sources=[source.model_dump()]))
    assert CrewResponse.model_validate_json(model.model_dump_json()) == model


def test_application_and_escalation_models_are_strict() -> None:
    assert ApplicationView.model_validate(APP).record_id == "LA-0007"
    assert Escalation.model_validate(ESC).recommended is False
    with pytest.raises(ValidationError):
        Escalation.model_validate(ESC | {"threshold": -0.1})


# ------------------------------------------------------------------ crew task outputs

RAG = RetrievalToolOutput(
    answer="Prepayment is free.", sources=(SourceRef(**SOURCE),), grounded=True,
    top1_similarity=0.71, strategy="sentence", threshold=0.42,
)  # fmt: skip


@pytest.mark.parametrize(
    "rag",
    [RAG, {"kind": "rag_error", "error": "retrieval_unavailable"}, None],
    ids=["result", "error", "skipped"],
)
def test_retrieval_task_output_round_trips(rag: Any) -> None:
    out = RetrievalTaskOutput.model_validate({"kind": "retrieval_result", "rag": rag})
    assert RetrievalTaskOutput.model_validate_json(out.model_dump_json()) == out


def test_retrieval_task_output_rejects_unknown_shapes() -> None:
    with pytest.raises(ValidationError):
        RetrievalTaskOutput.model_validate({"kind": "retrieval_result", "rag": {"kind": "x"}})
    with pytest.raises(ValidationError):
        RetrievalTaskOutput.model_validate({"kind": "retrieval_result", "rag": None, "x": 1})


def test_lookup_task_output_is_the_tool_output_plus_its_kind() -> None:
    found = {"kind": "lookup_result", "found": True, **APP, "escalation_score": 0.24,
             "escalation_recommended": False, "escalation_threshold": 0.5}  # fmt: skip
    assert isinstance(LOOKUP_TASK_OUTPUT.validate_python(found), LookupFound)
    for error in ("no_record_id", "invalid_record_id", "not_found"):
        out = LOOKUP_TASK_OUTPUT.validate_python(
            {"kind": "lookup_result", "found": False, "error": error}
        )
        assert isinstance(out, LookupNotFound)
        assert out.error == error
    with pytest.raises(ValidationError):
        LOOKUP_TASK_OUTPUT.validate_python({"found": False, "error": "not_found"})  # no kind
    with pytest.raises(ValidationError):
        LOOKUP_TASK_OUTPUT.validate_python({"kind": "lookup_result", "found": False, "error": "x"})


# ------------------------------------------------------------------ Phase 7: the pipeline's answer
def _ask(**over: Any) -> dict[str, Any]:
    from credagent.llm.engine import MockEngine

    refusal = MockEngine().refusal("ungrounded").model_dump(mode="json", exclude={"kind"})
    base = {
        "trace_id": "t1", "session_id": "s1", **refusal,
        "review": {"approved": True, "final_answer": refusal["answer"], "reason": "ok"},
        "tools_invoked": [{"agent": "retrieval_agent", "tool": "rag_lookup",
                           "args": {"query": "q"}, "cache_hit": False}],
        "cache_hit": False,
        "usage": {"estimated_tokens": 10, "budget_tokens": 8000, "estimated_cost_usd": 0.00002},
        "timings_ms": {"mask": 0.1},
    }  # fmt: skip
    return base | over


def test_review_verdict_has_exactly_the_brief_fields() -> None:
    from credagent.schemas import ReviewVerdict

    assert set(ReviewVerdict.model_fields) == {"approved", "final_answer", "reason"}
    with pytest.raises(ValidationError):
        ReviewVerdict.model_validate({"approved": True, "final_answer": "a", "reason": "r", "x": 1})


def test_ask_response_round_trips_and_mirrors_the_crew_response_fields() -> None:
    from credagent.schemas import AskResponse

    response = AskResponse.model_validate(_ask())
    assert AskResponse.model_validate_json(response.model_dump_json()) == response
    crew_fields = set(CrewResponse.model_fields) - {"kind"}
    assert crew_fields <= set(AskResponse.model_fields)


@pytest.mark.parametrize(
    "over",
    [
        {"refused": False},  # refused ⇔ refusal_reason, as in CrewResponse
        {"review": None, "extra": 1},
        {"usage": {"estimated_tokens": -1, "budget_tokens": 8000, "estimated_cost_usd": 0}},
    ],
)
def test_ask_response_is_strict(over: dict[str, Any]) -> None:
    from credagent.schemas import AskResponse

    with pytest.raises(ValidationError):
        AskResponse.model_validate(_ask(**over))
