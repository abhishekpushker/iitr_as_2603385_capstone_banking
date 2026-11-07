"""`RagTool` (§ARCH 7.9, 7.15): cache → retriever → generator, JSON outputs that validate, the
ledger records every call with its cache hit, errors are returned (never raised, never cached),
and the tool is bound to the deployed collection."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from credagent.cache.response_cache import ResponseCache, cache_key
from credagent.kb.chunking import Strategy
from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag.calibration import RetrievalConfig
from credagent.rag.generator import GroundedGenerator
from credagent.rag.retriever import RetrievalResult
from credagent.rag.store import RetrievedChunk
from credagent.schemas import (
    FALLBACK_ANSWER,
    RAG_TOOL_OUTPUT,
    GeneratedAnswer,
    RagToolError,
    RetrievalToolOutput,
)
from credagent.tools.acl import LeastAutonomyViolation
from credagent.tools.ledger import ToolCallLedger
from credagent.tools.rag_tool import RagQuery, RagTool, rag_tool_for_deployment

TEXT = "Floating-rate home loans taken by individuals carry no prepayment penalty."
THRESHOLD = 0.5
KB = "kb-v1"


class FakeRetriever:
    def __init__(self, similarity: float = 0.8, *, fail: bool = False) -> None:
        self.similarity = similarity
        self.fail = fail
        self.calls: list[tuple[str, Strategy, int | None]] = []

    def retrieve(self, query: str, strategy: Strategy, k: int | None = None) -> RetrievalResult:
        self.calls.append((query, strategy, k))
        if self.fail:
            raise ConnectionError("chroma unreachable")
        chunk = RetrievedChunk(
            chunk_id="kb-01::sentence::000", doc_id="kb-01", text=TEXT,
            similarity=self.similarity, rank=1, char_start=0,
        )  # fmt: skip
        return RetrievalResult(chunks=(chunk,), top1_similarity=self.similarity, strategy=strategy)


Clock = Callable[[], float]


def make(
    retriever: FakeRetriever | None = None,
    *,
    cache: ResponseCache[GeneratedAnswer] | None = None,
    kb_version: str = KB,
    ledger: ToolCallLedger | None = None,
) -> tuple[RagTool, FakeRetriever, ToolCallLedger, Metrics]:
    retriever = FakeRetriever() if retriever is None else retriever
    metrics = Metrics()
    generator = GroundedGenerator(
        retriever, MockEngine(), strategy="sentence", threshold=THRESHOLD, top_k=3,
        metrics=metrics,
    )  # fmt: skip
    ledger = ToolCallLedger() if ledger is None else ledger
    if cache is None:
        cache = ResponseCache(8, 60, metrics=metrics, clock=lambda: 0.0)
    tool = RagTool(generator=generator, cache=cache, kb_version=kb_version, ledger=ledger)
    return tool, retriever, ledger, metrics


def test_tool_identity_and_args_schema() -> None:
    tool, *_ = make()
    assert tool.name == "rag_lookup"
    assert tool.args_schema is RagQuery
    assert tool.owner == "retrieval_agent"
    assert (tool.strategy, tool.threshold, tool.top_k, tool.kb_version) == (
        "sentence", THRESHOLD, 3, KB,
    )  # fmt: skip


@pytest.mark.parametrize("query", ["", "x" * 2001])
def test_query_length_is_bounded(query: str) -> None:
    with pytest.raises(ValidationError):
        RagQuery(query=query)


def test_in_scope_query_returns_a_grounded_answer_as_json() -> None:
    tool, retriever, _, _ = make()
    out = tool._run("prepayment penalty on floating-rate home loans")
    parsed = RetrievalToolOutput.model_validate_json(out)
    assert parsed.kind == "rag_result"
    assert parsed.grounded and parsed.answer == TEXT
    assert [s.chunk_id for s in parsed.sources] == ["kb-01::sentence::000"]
    assert (parsed.strategy, parsed.threshold, parsed.top1_similarity) == ("sentence", 0.5, 0.8)
    assert retriever.calls == [("prepayment penalty on floating-rate home loans", "sentence", 3)]
    RAG_TOOL_OUTPUT.validate_json(out)


def test_below_threshold_returns_the_fallback() -> None:
    tool, *_ = make(FakeRetriever(similarity=0.2))
    parsed = RetrievalToolOutput.model_validate_json(tool._run("what is the weather"))
    assert parsed.answer == FALLBACK_ANSWER and not parsed.grounded and parsed.sources == ()


def test_second_equivalent_query_is_a_cache_hit_and_the_ledger_shows_it() -> None:
    tool, retriever, ledger, metrics = make()
    assert tool.ledger is ledger
    first = tool._run("Prepayment penalty?")
    second = tool._run("  prepayment   PENALTY ")
    assert first == second
    assert len(retriever.calls) == 1
    assert [(e.agent, e.tool, dict(e.args), e.cache_hit) for e in ledger.entries()] == [
        ("retrieval_agent", "rag_lookup", {"query": "Prepayment penalty?"}, False),
        ("retrieval_agent", "rag_lookup", {"query": "  prepayment   PENALTY "}, True),
    ]
    snap = metrics.snapshot()
    assert (snap["cache_hits"], snap["cache_misses"], snap["generation_calls"]) == (1, 1, 1)


def test_fallback_results_are_cached_too() -> None:
    tool, retriever, ledger, _ = make(FakeRetriever(similarity=0.1))
    tool._run("weather")
    tool._run("weather")
    assert len(retriever.calls) == 1
    assert [e.cache_hit for e in ledger.entries()] == [False, True]


def test_cache_key_covers_kb_version_strategy_threshold_and_top_k() -> None:
    metrics = Metrics()
    cache: ResponseCache[GeneratedAnswer] = ResponseCache(8, 60, metrics=metrics)
    tool, *_ = make(cache=cache)
    tool._run("penalty")
    expected = cache_key(
        kb_version=KB, strategy="sentence", threshold=THRESHOLD, top_k=3, masked_query="penalty"
    )
    assert cache.list_keys() == [expected]


def test_a_new_kb_version_misses_a_shared_cache() -> None:
    cache: ResponseCache[GeneratedAnswer] = ResponseCache(8, 60, metrics=Metrics())
    old, old_retriever, *_ = make(cache=cache, kb_version="v1")
    new, new_retriever, ledger, _ = make(cache=cache, kb_version="v2")
    old._run("penalty")
    new._run("penalty")
    assert len(old_retriever.calls) == len(new_retriever.calls) == 1
    assert ledger.entries()[0].cache_hit is False


def test_retrieval_error_is_returned_as_json_not_raised_and_not_cached() -> None:
    retriever = FakeRetriever(fail=True)
    tool, _, ledger, _ = make(retriever)
    out = tool._run("penalty")
    assert RagToolError.model_validate_json(out) == RagToolError(
        kind="rag_error", error="retrieval_unavailable"
    )
    RAG_TOOL_OUTPUT.validate_json(out)
    assert "chroma" not in out  # internals never reach the agent
    retriever.fail = False
    assert RetrievalToolOutput.model_validate_json(tool._run("penalty")).grounded
    assert len(retriever.calls) == 2
    assert [e.cache_hit for e in ledger.entries()] == [False, False]


def test_public_run_path_validates_args_and_delegates() -> None:
    tool, *_ = make()
    assert json.loads(tool.run(query="penalty"))["kind"] == "rag_result"


@pytest.mark.parametrize("owner", ["lookup_agent", "composer_agent", "x"])
def test_tool_cannot_be_constructed_for_a_non_owner(owner: str) -> None:
    tool, *_ = make()
    with pytest.raises(LeastAutonomyViolation):
        RagTool(
            generator=tool.generator, cache=tool.cache, kb_version=KB, ledger=ToolCallLedger(),
            owner=owner,
        )  # fmt: skip


def test_owner_tamper_raises_on_run_and_records_nothing() -> None:
    tool, retriever, ledger, _ = make()
    tool.owner = "composer_agent"
    with pytest.raises(LeastAutonomyViolation):
        tool._run("penalty")
    assert ledger.entries() == () and retriever.calls == []


# ------------------------------------------------------------------ deployment binding
def config(deployed: Strategy | None, thresholds: dict[Strategy, float | None]) -> RetrievalConfig:
    return RetrievalConfig(
        kb_version="kb-v9", top_k=2, gated_strategy="sentence", deployed_strategy=deployed,
        thresholds=thresholds,
    )  # fmt: skip


def test_for_deployment_binds_the_deployed_strategy_threshold_and_top_k() -> None:
    retriever = FakeRetriever()
    tool = rag_tool_for_deployment(
        config("fixed", {"fixed": 0.42, "sentence": 0.5}), retriever=retriever,
        client=MockEngine(), cache=ResponseCache(8, 60, metrics=Metrics()),
        ledger=ToolCallLedger(), metrics=Metrics(),
    )  # fmt: skip
    assert (tool.strategy, tool.threshold, tool.top_k, tool.kb_version) == (
        "fixed", 0.42, 2, "kb-v9",
    )  # fmt: skip
    tool._run("penalty")
    assert retriever.calls == [("penalty", "fixed", 2)]


@pytest.mark.parametrize(
    ("deployed", "thresholds", "match"),
    [(None, {"sentence": 0.5}, "deployed"), ("fixed", {"fixed": None}, "threshold")],
)
def test_for_deployment_refuses_an_undeployable_config(
    deployed: Strategy | None, thresholds: dict[Strategy, float | None], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        rag_tool_for_deployment(
            config(deployed, thresholds), retriever=FakeRetriever(), client=MockEngine(),
            cache=ResponseCache(8, 60, metrics=Metrics()), ledger=ToolCallLedger(),
            metrics=Metrics(),
        )  # fmt: skip
