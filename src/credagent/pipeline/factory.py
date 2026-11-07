"""Wire the production `Orchestrator` (§ARCH 6, 7.9): Chroma-backed retriever on the deployed
collection, the response cache, a fresh crew per request, and the deployed collection's chunk
texts for the groundedness gate. Process-wide objects are built once; everything per request
comes from the `RequestContext` inside `Orchestrator.handle`.
"""

from __future__ import annotations

import functools
from typing import Any

from credagent.cache.response_cache import ResponseCache
from credagent.context import RequestContext
from credagent.crew.factory import CrewDeps, build_crew, run_crew
from credagent.memory.session import ConversationMemory, SessionStore
from credagent.observability.metrics import Metrics
from credagent.pipeline.orchestrator import (
    BudgetPreflight,
    CrewRunner,
    Orchestrator,
    RequestSink,
    ReviewStage,
)
from credagent.rag.calibration import load_retrieval_config
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import GeneratedAnswer
from credagent.settings import Settings, get_settings


def crew_runner(deps: CrewDeps) -> CrewRunner:
    """Stage 5: a fresh crew per request, kicked off synchronously (in the worker thread)."""

    def run(ctx: RequestContext, query: str) -> Any:
        return run_crew(build_crew(ctx, query, deps))

    return run


def build_orchestrator(
    settings: Settings | None = None,
    *,
    client: Any = None,
    store: SessionStore | None = None,
    sink: RequestSink | None = None,
    review: ReviewStage | None = None,
    metrics: Metrics | None = None,
    timeout_s: float | None = None,
    preflight: BudgetPreflight | None = None,
    max_tokens: int | None = None,
) -> Orchestrator:
    s = get_settings() if settings is None else settings
    if client is None:
        client = connect_http(
            s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
            backoff_cap_s=s.chroma_backoff_cap_s,
        )  # fmt: skip
    metrics = Metrics() if metrics is None else metrics
    vectors = VectorStore(client)
    retriever = Retriever(
        vectors, get_embedder(), collections_from_settings(s), top_k=s.top_k, metrics=metrics
    )
    config = load_retrieval_config(s.retrieval_config_path)
    if config.deployed_strategy is None:
        raise ValueError("config/retrieval.json has no deployed_strategy (run make eval-chunking)")
    cache: ResponseCache[GeneratedAnswer] = ResponseCache.from_settings(metrics=metrics)
    deps = CrewDeps(retrieval_config=config, retriever=retriever, cache=cache, metrics=metrics)
    collection = retriever.collection_name(config.deployed_strategy)
    return Orchestrator(
        crew_runner=crew_runner(deps),
        chunk_texts=functools.partial(vectors.documents, collection),
        memory=ConversationMemory(SessionStore.from_settings() if store is None else store),
        sink=sink, review=review, timeout_s=timeout_s, preflight=preflight, max_tokens=max_tokens,
    )  # fmt: skip
