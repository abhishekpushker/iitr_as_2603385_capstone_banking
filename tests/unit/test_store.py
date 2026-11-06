"""VectorStore over an injected chromadb client (§ARCH 7.4, P6, P7); EphemeralClient here."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from credagent.rag import store as store_module
from credagent.rag.store import RetrievedChunk, VectorStore, connect_http

_names = itertools.count()


@pytest.fixture
def store() -> Iterator[VectorStore]:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    vs = VectorStore(client)
    yield vs
    for name in vs.list_collections():
        client.delete_collection(name)


def _name() -> str:
    return f"unit_store_{next(_names)}"


def _unit(v: list[float]) -> list[float]:
    a = np.asarray(v, dtype=float)
    return list(a / np.linalg.norm(a))


ROWS = {
    "d1::s::000": ("d1", [1.0, 0.0, 0.0]),
    "d1::s::001": ("d1", [1.0, 1.0, 0.0]),
    "d2::s::000": ("d2", [0.0, 0.0, 1.0]),
}


def _fill(vs: VectorStore, name: str) -> None:
    vs.upsert(
        name,
        ids=list(ROWS),
        embeddings=[_unit(v) for _, v in ROWS.values()],
        documents=[f"text of {i}" for i in ROWS],
        metadatas=[
            {"doc_id": d, "index": n, "char_start": 10 * n}
            for n, (d, _) in enumerate(ROWS.values())
        ],
    )


def test_collection_is_cosine_without_embedding_function(store: VectorStore) -> None:
    col = store.collection(_name(), metadata={"strategy": "fixed"})
    assert col.configuration["hnsw"]["space"] == "cosine"
    assert col.configuration["embedding_function"] is None
    assert col.metadata == {"strategy": "fixed"}


def test_query_returns_similarity_sorted_desc(store: VectorStore) -> None:
    name = _name()
    _fill(store, name)
    q = _unit([1.0, 0.4, 0.0])
    got = store.query(name, q, k=3)
    assert [c.chunk_id for c in got] == ["d1::s::000", "d1::s::001", "d2::s::000"]
    assert [c.rank for c in got] == [1, 2, 3]
    for c in got:
        assert isinstance(c, RetrievedChunk)
        expected = float(np.dot(q, _unit(ROWS[c.chunk_id][1])))
        assert c.similarity == pytest.approx(expected, abs=1e-6)  # P7: 1 − cosine distance
        assert c.doc_id == ROWS[c.chunk_id][0]
        assert c.text == f"text of {c.chunk_id}"
        assert c.char_start == 10 * list(ROWS).index(c.chunk_id)


def test_ties_are_broken_by_chunk_id(store: VectorStore) -> None:
    name = _name()
    v = _unit([1.0, 1.0, 0.0])
    ids = ["b::s::000", "a::s::000", "c::s::000"]
    store.upsert(
        name, ids=ids, embeddings=[v] * 3, documents=ids,
        metadatas=[{"doc_id": i.split("::")[0], "char_start": 0} for i in ids],
    )  # fmt: skip
    assert [c.chunk_id for c in store.query(name, v, k=3)] == sorted(ids)


def test_k_larger_than_the_collection(store: VectorStore) -> None:
    name = _name()
    _fill(store, name)
    assert len(store.query(name, _unit([0.0, 1.0, 0.0]), k=10)) == 3


def test_upsert_is_idempotent_and_ids_count_delete(store: VectorStore) -> None:
    name = _name()
    _fill(store, name)
    _fill(store, name)
    assert store.count(name) == 3
    assert store.ids(name) == set(ROWS)
    store.delete(name, ["d2::s::000"])
    assert store.ids(name) == {"d1::s::000", "d1::s::001"}
    store.delete(name, [])  # no-op
    assert store.count(name) == 2


def test_upsert_rejects_mismatched_lengths(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        store.upsert(_name(), ids=["a"], embeddings=[], documents=["a"], metadatas=[{}])


def test_upsert_requires_explicit_embeddings(store: VectorStore) -> None:
    """P6: no record may reach Chroma without our embedding."""
    with pytest.raises(ValueError, match="explicit embeddings"):
        store.upsert(_name(), ids=[], embeddings=[], documents=[], metadatas=[])


def test_meta_roundtrip(store: VectorStore) -> None:
    name = _name()
    assert store.read_meta(name) is None
    store.write_meta(name, {"kb_version": "abc", "count_fixed": 3})
    store.write_meta(name, {"kb_version": "def", "count_fixed": 4})
    assert store.read_meta(name) == {"kb_version": "def", "count_fixed": 4}
    assert store.count(name) == 1


def test_list_collections(store: VectorStore) -> None:
    a, b = _name(), _name()
    store.collection(a)
    store.collection(b)
    assert {a, b} <= set(store.list_collections())


# ------------------------------------------------------------------ bounded connect retry


class _Flaky:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, **_: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("chroma not up yet")
        return _Heartbeat()


class _Heartbeat:
    def heartbeat(self) -> int:
        return 1


def test_connect_retries_with_capped_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky = _Flaky(failures=7)
    monkeypatch.setattr(store_module.chromadb, "HttpClient", flaky)
    sleeps: list[float] = []
    client = connect_http("chroma", 8000, attempts=10, backoff_cap_s=5.0, sleep=sleeps.append)
    assert isinstance(client, _Heartbeat)
    assert flaky.calls == 8
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_connect_gives_up_after_the_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky = _Flaky(failures=99)
    monkeypatch.setattr(store_module.chromadb, "HttpClient", flaky)
    sleeps: list[float] = []
    with pytest.raises(ConnectionError, match="after 10 attempts"):
        connect_http("chroma", 8000, attempts=10, backoff_cap_s=5.0, sleep=sleeps.append)
    assert flaky.calls == 10 and len(sleeps) == 9
