"""`rag_lookup` (§ARCH 7.9, 7.15): ResponseCache → Retriever → GroundedGenerator.

`_run(query)` builds the cache key from the masked query and the generator's bindings
(kb_version, strategy, threshold, top_k), computes on a miss, and returns
`RetrievalToolOutput` JSON. Any failure in retrieval or generation is returned as `RagToolError`
JSON with no internals; errors are never cached (the cache stores only completed results).
Every call is recorded in the ledger with its `cache_hit`.

Owner-bound like the lookup tool: only `retrieval_agent` may own it (§ARCH 7.14.1).
`rag_tool_for_deployment` binds it to the collection `config/retrieval.json` deploys.
"""

from __future__ import annotations

import time
from typing import Any, Final

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from credagent.cache.response_cache import ResponseCache, cache_key
from credagent.kb.chunking import Strategy
from credagent.observability.metrics import Metrics
from credagent.rag.calibration import RetrievalConfig
from credagent.rag.generator import GroundedGenerator, LLMClient
from credagent.rag.retriever import SupportsRetrieve
from credagent.schemas import GeneratedAnswer, RagToolError, RetrievalToolOutput
from credagent.tools.acl import assert_allowed
from credagent.tools.ledger import ToolCallLedger

TOOL_NAME: Final = "rag_lookup"
MAX_QUERY_CHARS: Final = 2000


class RagQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS,
        description="The member's policy question (already masked).",
    )  # fmt: skip


class RagTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Answer a loan or banking policy question from Cred's knowledge base. Returns JSON with "
        "a grounded answer and its sources, or the 'I don't know' fallback."
    )
    args_schema: type[BaseModel] = RagQuery
    owner: str = "retrieval_agent"

    _generator: GroundedGenerator = PrivateAttr()
    _cache: ResponseCache[GeneratedAnswer] = PrivateAttr()
    _kb_version: str = PrivateAttr()
    _ledger: ToolCallLedger = PrivateAttr()

    def __init__(
        self,
        *,
        generator: GroundedGenerator,
        cache: ResponseCache[GeneratedAnswer],
        kb_version: str,
        ledger: ToolCallLedger,
        owner: str = "retrieval_agent",
        **data: Any,
    ) -> None:
        super().__init__(owner=owner, **data)
        assert_allowed(owner, self)
        self._generator = generator
        self._cache = cache
        self._kb_version = kb_version
        self._ledger = ledger

    @property
    def generator(self) -> GroundedGenerator:
        return self._generator

    @property
    def cache(self) -> ResponseCache[GeneratedAnswer]:
        return self._cache

    @property
    def ledger(self) -> ToolCallLedger:
        return self._ledger

    @property
    def kb_version(self) -> str:
        return self._kb_version

    @property
    def strategy(self) -> Strategy:
        return self._generator.strategy

    @property
    def threshold(self) -> float:
        return self._generator.threshold

    @property
    def top_k(self) -> int:
        return self._generator.top_k

    def _run(self, query: str) -> str:
        assert_allowed(self.owner, self)  # runtime guard against a tampered owner
        start = time.perf_counter()
        key = cache_key(
            kb_version=self._kb_version, strategy=self.strategy, threshold=self.threshold,
            top_k=self.top_k, masked_query=query,
        )  # fmt: skip
        output: RetrievalToolOutput | RagToolError
        try:
            answer, hit = self._cache.get_or_compute(key, lambda: self._generator.answer(query))
        except Exception:  # noqa: BLE001  (a tool never raises into CrewAI; nothing is cached)
            output, hit = RagToolError(error="retrieval_unavailable"), False
        else:
            output = RetrievalToolOutput(
                answer=answer.answer, sources=answer.sources, grounded=answer.grounded,
                top1_similarity=answer.top1_similarity, strategy=self.strategy,
                threshold=self.threshold,
            )  # fmt: skip
        self._ledger.record(
            agent=self.owner, tool=self.name, args={"query": query},
            duration_ms=(time.perf_counter() - start) * 1000, cache_hit=hit,
        )  # fmt: skip
        return output.model_dump_json()


def rag_tool_for_deployment(
    config: RetrievalConfig,
    *,
    retriever: SupportsRetrieve,
    client: LLMClient,
    cache: ResponseCache[GeneratedAnswer],
    ledger: ToolCallLedger,
    metrics: Metrics,
) -> RagTool:
    """A `RagTool` on the deployed collection with its calibrated threshold and `top_k`."""
    strategy = config.deployed_strategy
    if strategy is None:
        raise ValueError("config/retrieval.json has no deployed_strategy (run make eval-chunking)")
    generator = GroundedGenerator(
        retriever, client, strategy=strategy, threshold=config.threshold_for(strategy),
        top_k=config.top_k, metrics=metrics,
    )  # fmt: skip
    return RagTool(generator=generator, cache=cache, kb_version=config.kb_version, ledger=ledger)
