"""The crew end to end on the real index (§ARCH 7.9, IMPLEMENTATION_PLAN Phase 6): real CrewAI
kickoffs, the deployed collection, the committed configs.

The ledger proves which agent called which tool; exact per-persona LLM-call counts detect hidden
retries and converter re-asks; every `result.raw` validates and equals `result.pydantic`.
"""

from __future__ import annotations

from typing import Any

import pytest

import dataset
from credagent.cache.response_cache import ResponseCache
from credagent.context import RequestContext
from credagent.crew.factory import CrewDeps, build_crew, parse_crew_output
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import FALLBACK_ANSWER, CrewResponse
from credagent.settings import get_settings
from credagent.tools.acl import audit_crew
from scripts import indexer

POLICY_Q = "Is there a prepayment penalty on a floating-rate home loan?"
STATUS_Q = "What is the status of application LA-0007?"
MIXED_Q = "What is the status of LA-0007, and is there a prepayment penalty on a home loan?"
UNKNOWN_Q = "What is the status of application LA-9999?"
OUT_OF_SCOPE_Q = "Can I get travel insurance with my card?"
PREPAYMENT_DOC = "kb-08-prepayment-penalty"
RECORD = next(r for r in dataset.LOAN_APPLICATIONS if r["record_id"] == "LA-0007")


@pytest.fixture(scope="module")
def deps() -> CrewDeps:
    assert indexer.main([]) == 0  # a cold volume must not leave the crew reading an empty index
    s = get_settings()
    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    metrics = Metrics()
    retriever = Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=metrics,
    )  # fmt: skip
    return CrewDeps(
        retrieval_config=cal.load_retrieval_config(s.retrieval_config_path),
        retriever=retriever, cache=ResponseCache(64, 3600, metrics=metrics), metrics=metrics,
    )  # fmt: skip


def kickoff(deps: CrewDeps, query: str) -> tuple[RequestContext, Any, Any, CrewResponse]:
    ctx = RequestContext.new(session_id="integration")
    crew = build_crew(ctx, query, deps)
    result = crew.kickoff()
    return ctx, crew, result, parse_crew_output(result)


def test_policy_query_only_retrieval_calls_rag_lookup(deps: CrewDeps) -> None:
    ctx, _, _, response = kickoff(deps, POLICY_Q)
    assert [(e.agent, e.tool, e.args) for e in ctx.ledger.entries()] == [
        ("retrieval_agent", "rag_lookup", {"query": POLICY_Q})
    ]
    assert (response.intent, response.grounded, response.refused) == ("policy", True, False)
    assert response.sources[0].doc_id == PREPAYMENT_DOC
    assert response.application is None


def test_status_query_only_lookup_calls_the_status_tool(deps: CrewDeps) -> None:
    ctx, _, _, response = kickoff(deps, STATUS_Q)
    assert [(e.agent, e.tool, e.args) for e in ctx.ledger.entries()] == [
        ("lookup_agent", "check_loan_application_status", {"record_id": "LA-0007"})
    ]
    assert response.intent == "status"
    assert response.application is not None
    assert response.application.model_dump() == {
        k: RECORD[k] for k in response.application.model_dump()
    }
    assert response.escalation is not None
    assert f"is currently {RECORD['status']}" in response.answer


def test_mixed_query_invokes_both_tools(deps: CrewDeps) -> None:
    ctx, _, _, response = kickoff(deps, MIXED_Q)
    assert sorted((e.agent, e.tool) for e in ctx.ledger.entries()) == [
        ("lookup_agent", "check_loan_application_status"),
        ("retrieval_agent", "rag_lookup"),
    ]
    assert (response.intent, response.grounded) == ("mixed", True)
    assert response.application is not None and response.sources
    assert response.sources[0].doc_id == PREPAYMENT_DOC


def test_unknown_id_gets_a_polite_not_found(deps: CrewDeps) -> None:
    ctx, _, _, response = kickoff(deps, UNKNOWN_Q)
    assert [e.args for e in ctx.ledger.entries()] == [{"record_id": "LA-9999"}]
    assert (response.intent, response.refused, response.application) == ("clarify", False, None)
    assert response.answer == (
        "I couldn't find a loan application with id LA-9999. Please check the id (format LA-0000)."
    )


def test_out_of_scope_query_gets_the_fallback(deps: CrewDeps) -> None:
    _, _, _, response = kickoff(deps, OUT_OF_SCOPE_Q)
    assert (response.intent, response.answer, response.grounded, response.sources) == (
        "out_of_scope", FALLBACK_ANSWER, False, ()
    )  # fmt: skip


QUERY_TYPES = {
    "policy": (POLICY_Q, {"composer": 1, "lookup": 1, "retrieval": 2}),
    "status": (STATUS_Q, {"composer": 1, "lookup": 2, "retrieval": 1}),
    "mixed": (MIXED_Q, {"composer": 1, "lookup": 2, "retrieval": 2}),
    "unknown-id": (UNKNOWN_Q, {"composer": 1, "lookup": 2, "retrieval": 1}),
    "out-of-scope": (OUT_OF_SCOPE_Q, {"composer": 1, "lookup": 1, "retrieval": 2}),
    "clarify": ("What is the status of my application?",
                {"composer": 1, "lookup": 1, "retrieval": 1}),
}  # fmt: skip


@pytest.mark.parametrize("name", list(QUERY_TYPES))
def test_exact_llm_call_counts_per_query_type(deps: CrewDeps, name: str) -> None:
    query, expected = QUERY_TYPES[name]
    ctx, crew, _, _ = kickoff(deps, query)
    assert ctx.budget.calls() == expected
    assert sum(a.llm.calls for a in crew.agents) == sum(expected.values())


@pytest.mark.parametrize("name", list(QUERY_TYPES))
def test_raw_output_validates_and_equals_pydantic(deps: CrewDeps, name: str) -> None:
    _, _, result, response = kickoff(deps, QUERY_TYPES[name][0])
    assert CrewResponse.model_validate_json(result.raw) == result.pydantic == response
    assert "Observation:" not in result.raw


def test_every_built_agent_disables_cache_and_delegation_and_passes_the_audit(
    deps: CrewDeps,
) -> None:
    """P4: CrewAI's Agent.cache defaults to True and would hide repeated tool calls."""
    _, crew, _, _ = kickoff(deps, MIXED_Q)
    assert [(a.cache, a.allow_delegation) for a in crew.agents] == [(False, False)] * 3
    assert audit_crew(crew)["lookup_agent"] == ("check_loan_application_status",)
    holders = [a.role for a in crew.agents if any(
        t.name == "check_loan_application_status" for t in a.tools)]  # fmt: skip
    assert holders == ["Loan Application Lookup Specialist"]
