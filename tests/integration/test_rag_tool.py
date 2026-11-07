"""`RagTool` against the real index on the deployed collection (§ARCH 7.9): an in-scope
calibration query is answered from its expected document, an out-of-scope one gets the fallback,
and a repeat is a cache hit."""

from __future__ import annotations

import pytest

from credagent.cache.response_cache import ResponseCache
from credagent.llm.engine import MockEngine
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import FALLBACK_ANSWER, GeneratedAnswer, RetrievalToolOutput
from credagent.settings import get_settings
from credagent.tools.ledger import ToolCallLedger
from credagent.tools.rag_tool import RagTool, rag_tool_for_deployment
from scripts import indexer


@pytest.fixture(scope="module")
def tool() -> RagTool:
    assert indexer.main([]) == 0  # a cold volume must not leave the tool reading an empty index
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
    cache: ResponseCache[GeneratedAnswer] = ResponseCache(16, 60, metrics=metrics)
    return rag_tool_for_deployment(
        cal.load_retrieval_config(s.retrieval_config_path), retriever=retriever,
        client=MockEngine(), cache=cache, ledger=ToolCallLedger(), metrics=metrics,
    )  # fmt: skip


def test_tool_is_bound_to_the_deployed_collection(tool: RagTool) -> None:
    config = cal.load_retrieval_config(get_settings().retrieval_config_path)
    assert config.deployed_strategy is not None
    assert tool.strategy == config.deployed_strategy
    assert tool.threshold == config.threshold_for(config.deployed_strategy)


@pytest.mark.parametrize("q", cal.IN_SCOPE_QUERIES[:3], ids=lambda q: q.expected_doc_id)
def test_in_scope_query_is_grounded_in_its_document(tool: RagTool, q: cal.CalibrationQuery) -> None:
    out = RetrievalToolOutput.model_validate_json(tool._run(q.query))
    assert out.grounded and out.answer != FALLBACK_ANSWER
    assert out.sources[0].doc_id == q.expected_doc_id


def test_out_of_scope_query_gets_the_fallback(tool: RagTool) -> None:
    out = RetrievalToolOutput.model_validate_json(tool._run(cal.OUT_OF_SCOPE_QUERIES[0]))
    assert out.answer == FALLBACK_ANSWER and not out.grounded


def test_repeat_is_a_cache_hit(tool: RagTool) -> None:
    q = cal.IN_SCOPE_QUERIES[0].query
    assert tool._run(q) == tool._run(q.upper())
    assert tool.ledger.entries()[-1].cache_hit is True
