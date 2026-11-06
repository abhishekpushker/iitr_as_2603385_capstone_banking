"""Retrieval (§ARCH 7.5): embed the (masked) query once, query one collection, return top-k.

`Retriever.retrieve(query, strategy, k=TOP_K) -> RetrievalResult{chunks, top1_similarity,
strategy}`. An empty collection yields no chunks and `top1_similarity = NO_MATCH_SIMILARITY`
(the cosine minimum), which is below any threshold. Each call counts one retrieval and one
embedding call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from credagent.kb.chunking import Strategy
from credagent.observability.metrics import Metrics
from credagent.rag.embedder import Encoder
from credagent.rag.store import RetrievedChunk, VectorStore
from credagent.settings import Settings

NO_MATCH_SIMILARITY: Final = -1.0


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunks: tuple[RetrievedChunk, ...]
    top1_similarity: float
    strategy: Strategy


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, strategy: Strategy, k: int | None = None) -> RetrievalResult: ...


def collections_from_settings(s: Settings) -> dict[Strategy, str]:
    return {"fixed": s.collection_fixed, "sentence": s.collection_sentence}


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        encoder: Encoder,
        collections: Mapping[Strategy, str],
        *,
        top_k: int,
        metrics: Metrics,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._collections = dict(collections)
        self._top_k = top_k
        self._metrics = metrics

    def collection_name(self, strategy: Strategy) -> str:
        return self._collections[strategy]

    def retrieve(self, query: str, strategy: Strategy, k: int | None = None) -> RetrievalResult:
        k = self._top_k if k is None else k
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        name = self._collections[strategy]
        (vector,) = self._encoder.encode([query])
        self._metrics.incr("embedding_calls")
        chunks = tuple(self._store.query(name, vector, k=k))
        self._metrics.incr("retrieval_calls")
        top1 = chunks[0].similarity if chunks else NO_MATCH_SIMILARITY
        return RetrievalResult(chunks=chunks, top1_similarity=top1, strategy=strategy)
