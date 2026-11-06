"""Retriever over an EphemeralClient store and a fake encoder (§ARCH 7.5)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings as ChromaSettings

from credagent.observability.metrics import Metrics
from credagent.rag.retriever import NO_MATCH_SIMILARITY, Retriever, collections_from_settings
from credagent.rag.store import VectorStore
from credagent.settings import get_settings

_n = itertools.count()


def unit(v: Sequence[float]) -> list[float]:
    a = np.asarray(v, dtype=float)
    return [float(x) for x in a / np.linalg.norm(a)]


class FakeEncoder:
    VECTORS = {"emi": [1.0, 0.0, 0.0], "fee": [0.0, 1.0, 0.0], "both": [1.0, 1.0, 0.0]}

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [unit(self.VECTORS[t]) for t in texts]


@pytest.fixture
def store() -> Iterator[VectorStore]:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vs = VectorStore(client)
    yield vs
    for name in vs.list_collections():
        client.delete_collection(name)


@pytest.fixture
def names(store: VectorStore) -> dict[str, str]:
    n = next(_n)
    names = {"fixed": f"ut_fixed_{n}", "sentence": f"ut_sentence_{n}"}
    for strategy, name in names.items():
        store.upsert(
            name,
            ids=[f"kb-02::{strategy}::000", f"kb-03::{strategy}::000"],
            embeddings=[unit([1.0, 0.1, 0.0]), unit([0.0, 1.0, 0.2])],
            documents=["EMI text", "Fee text"],
            metadatas=[
                {"doc_id": "kb-02", "char_start": 0},
                {"doc_id": "kb-03", "char_start": 12},
            ],
        )
    store.collection(f"ut_empty_{n}")
    names["empty"] = f"ut_empty_{n}"
    return names


def test_retrieve_returns_ranked_chunks_and_top1(store: VectorStore, names: dict[str, str]) -> None:
    metrics, enc = Metrics(), FakeEncoder()
    r = Retriever(store, enc, {"fixed": names["fixed"], "sentence": names["sentence"]},
                  top_k=2, metrics=metrics)  # fmt: skip
    res = r.retrieve("fee", "sentence")
    assert res.strategy == "sentence"
    assert [c.doc_id for c in res.chunks] == ["kb-03", "kb-02"]
    assert [c.char_start for c in res.chunks] == [12, 0]
    assert res.top1_similarity == res.chunks[0].similarity
    assert res.top1_similarity == pytest.approx(float(np.dot(unit([0, 1, 0]), unit([0, 1, 0.2]))))
    snap = metrics.snapshot()
    assert (snap["retrieval_calls"], snap["embedding_calls"]) == (1, 1)


def test_k_overrides_the_default_and_strategy_picks_the_collection(
    store: VectorStore, names: dict[str, str]
) -> None:
    r = Retriever(store, FakeEncoder(), {"fixed": names["fixed"], "sentence": names["sentence"]},
                  top_k=2, metrics=Metrics())  # fmt: skip
    res = r.retrieve("emi", "fixed", k=1)
    assert [c.chunk_id for c in res.chunks] == ["kb-02::fixed::000"]
    assert r.collection_name("fixed") == names["fixed"]


def test_empty_collection_gives_no_match(store: VectorStore, names: dict[str, str]) -> None:
    r = Retriever(store, FakeEncoder(), {"fixed": names["empty"], "sentence": names["empty"]},
                  top_k=3, metrics=Metrics())  # fmt: skip
    res = r.retrieve("emi", "fixed")
    assert res.chunks == () and res.top1_similarity == NO_MATCH_SIMILARITY == -1.0


def test_rejects_non_positive_k(store: VectorStore, names: dict[str, str]) -> None:
    r = Retriever(store, FakeEncoder(), {"fixed": names["fixed"], "sentence": names["sentence"]},
                  top_k=3, metrics=Metrics())  # fmt: skip
    with pytest.raises(ValueError, match="k"):
        r.retrieve("emi", "fixed", k=0)


def test_collections_from_settings() -> None:
    s = get_settings()
    assert collections_from_settings(s) == {
        "fixed": s.collection_fixed, "sentence": s.collection_sentence,
    }  # fmt: skip
