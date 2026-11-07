"""`crew/{agents,tasks,factory}.py` (§ARCH 7.9, 7.14.1): real CrewAI kickoffs with a stub
retriever (no Chroma, sockets off). The same assertions against the real index are in
`tests/integration/test_crew.py`.

Exact LLM-call counts per query type detect hidden retries and converter re-asks (§ARCH 2.2.7).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from crewai import Crew

from credagent.cache.response_cache import ResponseCache
from credagent.context import RequestContext
from credagent.crew import templates
from credagent.crew.agents import MAX_ITER, MAX_RETRY_LIMIT, make_agent
from credagent.crew.factory import (
    CrewDeps,
    SchemaViolation,
    build_crew,
    parse_crew_output,
    run_crew,
)
from credagent.kb.chunking import Strategy
from credagent.llm.crewai_llm import MockCrewLLM
from credagent.observability.metrics import Metrics
from credagent.rag.calibration import RetrievalConfig
from credagent.rag.retriever import RetrievalResult
from credagent.rag.store import RetrievedChunk
from credagent.schemas import FALLBACK_ANSWER, CrewResponse
from credagent.tools.acl import LeastAutonomyViolation, audit_crew

TEXT = "Floating-rate home loans taken by individuals carry no prepayment penalty."
POLICY_Q = "Is there a prepayment penalty on a floating-rate home loan?"
STATUS_Q = "What is the status of application LA-0007?"
MIXED_Q = "What is the status of LA-0007, and is there a prepayment penalty on a home loan?"
UNKNOWN_Q = "What is the status of application LA-9999?"
CLARIFY_Q = "What is the status of my application?"


class FakeRetriever:
    def __init__(self, similarity: float = 0.8) -> None:
        self.similarity = similarity
        self.queries: list[str] = []

    def retrieve(self, query: str, strategy: Strategy, k: int | None = None) -> RetrievalResult:
        self.queries.append(query)
        chunk = RetrievedChunk(
            chunk_id="kb-08::sentence::000", doc_id="kb-08", text=TEXT,
            similarity=self.similarity, rank=1, char_start=0,
        )  # fmt: skip
        return RetrievalResult(chunks=(chunk,), top1_similarity=self.similarity, strategy=strategy)


def deps(retriever: FakeRetriever | None = None) -> CrewDeps:
    metrics = Metrics()
    config = RetrievalConfig(
        kb_version="kb-v1", top_k=3, gated_strategy="sentence", deployed_strategy="sentence",
        thresholds={"sentence": 0.5, "fixed": None},
    )  # fmt: skip
    return CrewDeps(
        retrieval_config=config, retriever=retriever or FakeRetriever(),
        cache=ResponseCache(8, 60, metrics=metrics), metrics=metrics,
    )  # fmt: skip


def run(query: str, d: CrewDeps | None = None) -> tuple[RequestContext, Crew, Any]:
    ctx = RequestContext.new(session_id="s", trace_id="t" * 32)
    crew = build_crew(ctx, query, d or deps())
    return ctx, crew, run_crew(crew)


# ------------------------------------------------------------------ agents (P4, least autonomy)


def test_built_agents_carry_every_override_and_one_llm_each() -> None:
    _, crew, _ = run(POLICY_Q)
    assert [a.role for a in crew.agents] == list(templates.ROLES.values())
    for agent in crew.agents:
        assert agent.cache is False  # P4
        assert agent.allow_delegation is False
        assert (agent.max_iter, agent.max_retry_limit, agent.verbose) == (
            MAX_ITER, MAX_RETRY_LIMIT, False
        )  # fmt: skip
        assert isinstance(agent.llm, MockCrewLLM)
        assert agent.llm.supports_function_calling() is False
    assert len({id(a.llm) for a in crew.agents}) == 3
    assert [a.llm.persona for a in crew.agents] == ["retrieval", "lookup", "composer"]
    assert [a.llm.response_format for a in crew.agents] == [None, None, CrewResponse]


def test_crew_flags_and_task_wiring() -> None:
    _, crew, _ = run(POLICY_Q)
    assert (crew.cache, crew.memory, crew.planning, crew.verbose) == (False, False, False, False)
    retrieval_task, lookup_task, compose_task = crew.tasks
    assert compose_task.context == [retrieval_task, lookup_task]
    assert compose_task.output_pydantic is CrewResponse
    assert [t.agent for t in crew.tasks] == crew.agents


def test_audit_passes_on_the_built_crew_and_maps_the_tools() -> None:
    _, crew, _ = run(POLICY_Q)
    assert audit_crew(crew) == {
        "retrieval_agent": ("rag_lookup",),
        "lookup_agent": ("check_loan_application_status",),
        "composer_agent": (),
    }


def test_make_agent_refuses_a_disallowed_tool_and_an_unknown_agent() -> None:
    _, crew, _ = run(POLICY_Q)
    lookup_tool = crew.agents[1].tools[0]
    llm = crew.agents[2].llm
    with pytest.raises(LeastAutonomyViolation):
        make_agent("composer_agent", [lookup_tool], llm=llm)
    with pytest.raises(LeastAutonomyViolation, match="unknown agent"):
        make_agent("planner_agent", [], llm=llm)


def test_each_request_gets_a_fresh_crew_tools_and_ledger() -> None:
    d = deps()
    ctx_a, crew_a, _ = run(POLICY_Q, d)
    ctx_b, crew_b, _ = run(POLICY_Q, d)
    assert crew_a is not crew_b
    assert crew_a.agents[0].tools[0] is not crew_b.agents[0].tools[0]
    assert (len(ctx_a.ledger), len(ctx_b.ledger)) == (1, 1)
    assert ctx_b.ledger.entries()[0].cache_hit is True  # the response cache is process-wide


# ------------------------------------------------------------------ kickoffs per query type


@pytest.mark.parametrize(
    ("query", "intent", "tools", "calls"),
    [
        (POLICY_Q, "policy", [("retrieval_agent", "rag_lookup")],
         {"retrieval": 2, "lookup": 1, "composer": 1}),
        (STATUS_Q, "status", [("lookup_agent", "check_loan_application_status")],
         {"retrieval": 1, "lookup": 2, "composer": 1}),
        (MIXED_Q, "mixed", [("retrieval_agent", "rag_lookup"),
                            ("lookup_agent", "check_loan_application_status")],
         {"retrieval": 2, "lookup": 2, "composer": 1}),
        (UNKNOWN_Q, "clarify", [("lookup_agent", "check_loan_application_status")],
         {"retrieval": 1, "lookup": 2, "composer": 1}),
        (CLARIFY_Q, "clarify", [], {"retrieval": 1, "lookup": 1, "composer": 1}),
    ],
    ids=["policy", "status", "mixed", "unknown-id", "clarify"],
)  # fmt: skip
def test_kickoff_per_query_type(
    query: str, intent: str, tools: list[tuple[str, str]], calls: dict[str, int]
) -> None:
    ctx, crew, result = run(query)
    response = parse_crew_output(result)
    assert response.intent == intent
    assert result.pydantic == CrewResponse.model_validate_json(result.raw)
    assert [(e.agent, e.tool) for e in ctx.ledger.entries()] == tools
    assert ctx.budget.calls() == dict(sorted(calls.items()))
    assert [a.llm.calls for a in crew.agents] == [
        calls["retrieval"], calls["lookup"], calls["composer"]
    ]  # fmt: skip
    assert "Observation:" not in result.raw
    assert ctx.budget.exceeded is False


def test_policy_answer_comes_from_the_retrieved_text() -> None:
    _, _, result = run(POLICY_Q)
    response = parse_crew_output(result)
    assert (response.answer, response.grounded) == (TEXT, True)
    assert [s.doc_id for s in response.sources] == ["kb-08"]


def test_below_threshold_is_out_of_scope_with_the_fallback() -> None:
    _, _, result = run("Can I get travel insurance with my card?", deps(FakeRetriever(0.1)))
    response = parse_crew_output(result)
    assert (response.intent, response.answer, response.grounded) == (
        "out_of_scope", FALLBACK_ANSWER, False
    )  # fmt: skip


def test_status_answer_and_ledger_arguments() -> None:
    ctx, _, result = run(STATUS_Q)
    response = parse_crew_output(result)
    assert response.application is not None and response.application.record_id == "LA-0007"
    assert response.answer.startswith("Application LA-0007 (")
    assert [e.args for e in ctx.ledger.entries()] == [{"record_id": "LA-0007"}]


def test_unknown_id_is_a_polite_not_found() -> None:
    _, _, result = run(UNKNOWN_Q)
    response = parse_crew_output(result)
    assert response.refused is False
    assert response.answer.startswith("I couldn't find a loan application with id LA-9999.")


def test_braces_in_the_query_are_not_interpolated() -> None:
    ctx, _, result = run("Is there a prepayment penalty {on} a {home_loan}?")
    assert parse_crew_output(result).intent == "policy"
    assert ctx.ledger.entries()[0].args == {
        "query": "Is there a prepayment penalty {on} a {home_loan}?"
    }


def test_budget_exceeded_mid_crew_returns_a_valid_refusal_without_raising() -> None:
    """P18: the first call already passes a tiny budget; nothing raises, the flag is set and
    every later call (the Composer included) refuses, so the crew output is still schema JSON."""
    ctx = RequestContext.new(session_id="s", trace_id="t" * 32, max_tokens=50)
    crew = build_crew(ctx, POLICY_Q, deps())
    result = crew.kickoff()
    response = parse_crew_output(result)
    assert (response.refused, response.refusal_reason) == (True, "budget_exceeded")
    assert ctx.budget.exceeded is True
    assert len(ctx.ledger) == 0  # no tool ran
    assert ctx.budget.calls() == {"composer": 1, "lookup": 1, "retrieval": 1}


# ------------------------------------------------------------------ parse_crew_output


def test_parse_rejects_raw_that_is_not_a_crew_response() -> None:
    with pytest.raises(SchemaViolation, match="not a valid"):
        parse_crew_output(SimpleNamespace(raw='{"kind": "crew_response"}', pydantic=None))


def test_parse_rejects_a_pydantic_that_differs_from_raw() -> None:
    _, _, result = run(POLICY_Q)
    other = parse_crew_output(result).model_copy(update={"answer": "different"})
    with pytest.raises(SchemaViolation, match="differs"):
        parse_crew_output(SimpleNamespace(raw=result.raw, pydantic=other))


def test_run_crew_refuses_a_streaming_result() -> None:
    with pytest.raises(TypeError, match="streaming"):
        run_crew(SimpleNamespace(kickoff=lambda: iter(())))  # type: ignore[arg-type]
