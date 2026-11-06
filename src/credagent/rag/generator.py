"""Grounded generation (§ARCH 7.5): retrieve, then ask an `LLMClient` for an answer that uses only
the retrieved chunks.

`rag` never imports `llm` (layering, §ARCH 4), so the client is a protocol here. Its mock
implementation is `credagent.llm.engine.MockEngine`, which callers inject. The generator sees
only the (already masked) query and the retrieved chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from credagent.kb.chunking import Strategy
from credagent.observability.metrics import Metrics
from credagent.rag.retriever import RetrievalResult, SupportsRetrieve
from credagent.rag.store import RetrievedChunk
from credagent.schemas import GeneratedAnswer


class LLMClient(Protocol):
    def generate_grounded(
        self, query: str, chunks: Sequence[RetrievedChunk], *, threshold: float
    ) -> GeneratedAnswer: ...


class GroundedGenerator:
    def __init__(
        self,
        retriever: SupportsRetrieve,
        client: LLMClient,
        *,
        strategy: Strategy,
        threshold: float,
        top_k: int,
        metrics: Metrics,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be a cosine similarity in [-1, 1], got {threshold}")
        self._retriever = retriever
        self._client = client
        self.strategy: Strategy = strategy
        self.threshold = threshold
        self.top_k = top_k
        self._metrics = metrics

    def retrieve(self, masked_query: str) -> RetrievalResult:
        return self._retriever.retrieve(masked_query, self.strategy, self.top_k)

    def generate(self, masked_query: str, retrieval: RetrievalResult) -> GeneratedAnswer:
        answer = self._client.generate_grounded(
            masked_query, retrieval.chunks, threshold=self.threshold
        )
        self._metrics.incr("generation_calls")
        return answer

    def answer(self, masked_query: str) -> GeneratedAnswer:
        return self.generate(masked_query, self.retrieve(masked_query))
