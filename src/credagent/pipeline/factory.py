"""Wire the production pipeline (§ARCH 6, 7.9, 7.16): Chroma-backed retriever on the deployed
collection, the response cache, a fresh crew per request, and the deployed collection's chunk
texts for the groundedness gate. Process-wide objects are built once; everything per request
comes from the `RequestContext` inside `Orchestrator.handle`.

`build_services` returns everything the API holds (`Services`): the orchestrator plus the
objects its endpoints reach directly (sessions, counters, cache, the live kb_version, the
document service, the stage-12 record collector). `build_orchestrator` is its orchestrator.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from credagent.cache.response_cache import ResponseCache
from credagent.context import RequestContext
from credagent.crew.factory import CrewDeps, build_crew, run_crew
from credagent.kb.chunking import Strategy
from credagent.kb.version import KbVersion
from credagent.memory.session import ConversationMemory, SessionStore
from credagent.observability.metrics import Metrics
from credagent.pipeline.documents import DocumentService
from credagent.pipeline.orchestrator import (
    BudgetPreflight,
    CrewRunner,
    NullSink,
    Orchestrator,
    RecordCollector,
    RequestSink,
    ReviewStage,
)
from credagent.rag.calibration import RetrievalConfig, load_retrieval_config
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.schemas import GeneratedAnswer
from credagent.settings import Settings, get_settings


class IndexNotReady(RuntimeError):
    """Chroma answers, but the index is missing or was built from another KB (§ARCH 3.5)."""


def crew_runner(deps: CrewDeps) -> CrewRunner:
    """Stage 5: a fresh crew per request, kicked off synchronously (in the worker thread)."""

    def run(ctx: RequestContext, query: str) -> Any:
        return run_crew(build_crew(ctx, query, deps))

    return run


@dataclass(frozen=True)
class Services:
    orchestrator: Orchestrator
    sessions: SessionStore
    metrics: Metrics
    cache: ResponseCache[Any]
    vectors: VectorStore
    collections: Mapping[Strategy, str]
    kb: KbVersion
    documents: DocumentService
    collector: RecordCollector

    def readiness(self) -> dict[str, bool]:
        """`/readyz` checks (§ARCH 3.5): Chroma answers and both collections exist."""
        try:
            self.vectors.heartbeat()
            names = set(self.vectors.list_collections())
        except Exception:  # noqa: BLE001  (any failure means "not ready", never a 500)
            return {"chroma": False, "collections": False}
        return {"chroma": True, "collections": set(self.collections.values()) <= names}


def check_index(vectors: VectorStore, s: Settings, config: RetrievalConfig) -> None:
    """Both collections hold chunks and `kb_meta` records the KB version the retrieval config was
    calibrated on, so the cache keys and thresholds match the index being queried."""
    meta = vectors.read_meta(s.collection_meta)
    if meta is None or meta.get("kb_version") != config.kb_version:
        raise IndexNotReady("kb_meta is missing or differs from config/retrieval.json (re-index)")
    for name in collections_from_settings(s).values():
        if vectors.count(name) == 0:
            raise IndexNotReady(f"collection {name} is empty (run the indexer)")


def build_services(
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
    require_index: bool = True,
) -> Services:
    s = get_settings() if settings is None else settings
    if client is None:
        client = connect_http(
            s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
            backoff_cap_s=s.chroma_backoff_cap_s,
        )  # fmt: skip
    metrics = Metrics() if metrics is None else metrics
    vectors = VectorStore(client)
    collections = collections_from_settings(s)
    encoder = get_embedder()
    retriever = Retriever(vectors, encoder, collections, top_k=s.top_k, metrics=metrics)
    config = load_retrieval_config(s.retrieval_config_path)
    if config.deployed_strategy is None:
        raise ValueError("config/retrieval.json has no deployed_strategy (run make eval-chunking)")
    if require_index:
        check_index(vectors, s, config)
    cache: ResponseCache[GeneratedAnswer] = ResponseCache.from_settings(metrics=metrics)
    kb = KbVersion(config.kb_version)
    deps = CrewDeps(retrieval_config=config, retriever=retriever, cache=cache, metrics=metrics,
                    kb=kb)  # fmt: skip
    collector = RecordCollector()
    sessions = SessionStore.from_settings() if store is None else store
    orchestrator = Orchestrator(
        crew_runner=crew_runner(deps),
        chunk_texts=functools.partial(
            vectors.documents, retriever.collection_name(config.deployed_strategy)
        ),
        memory=ConversationMemory(sessions),
        sink=collector if sink is None else sink, metrics=metrics, review=review,
        timeout_s=timeout_s, preflight=preflight, max_tokens=max_tokens,
    )  # fmt: skip
    documents = DocumentService(vectors=vectors, encoder=encoder, collections=collections, kb=kb,
                                cache=cache)  # fmt: skip
    return Services(
        orchestrator=orchestrator, sessions=sessions, metrics=metrics, cache=cache,
        vectors=vectors, collections=collections, kb=kb, documents=documents, collector=collector,
    )  # fmt: skip


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
    """The orchestrator alone, for scripts and tests. No index check, and stage 12 is the given
    sink (default `NullSink`): scripts keep their own records."""
    return build_services(
        settings, client=client, store=store, sink=NullSink() if sink is None else sink,
        review=review, metrics=metrics, timeout_s=timeout_s, preflight=preflight,
        max_tokens=max_tokens, require_index=False,
    ).orchestrator  # fmt: skip
