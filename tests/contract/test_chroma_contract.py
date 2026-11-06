"""chromadb 1.5.9 behaviour the vector store depends on (§ARCH 2.4, P6, P7, P8)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

import credagent  # noqa: F401
from credagent.settings import get_settings

VECTORS = {"a": [1.0, 0.0, 0.0], "b": [1.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
QUERY = [1.0, 0.5, 0.0]


@pytest.fixture
def client() -> Iterator[Any]:
    yield chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def _cosine_collection(client: Any, name: str) -> Any:
    return client.get_or_create_collection(
        name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


def test_client_version_is_pinned() -> None:
    assert chromadb.__version__ == "1.5.9"


def test_cosine_space_returns_cosine_distance(client: Any) -> None:
    """P7: similarity = 1 − distance, checked against numpy."""
    col = _cosine_collection(client, "contract_cosine")
    col.upsert(ids=list(VECTORS), embeddings=list(VECTORS.values()), documents=list(VECTORS))
    got = col.query(query_embeddings=[QUERY], n_results=3)
    q = np.asarray(QUERY)
    expected = {
        k: 1.0 - float(q @ np.asarray(v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        for k, v in VECTORS.items()
    }
    assert got["ids"][0] == sorted(expected, key=expected.__getitem__)
    for doc_id, dist in zip(got["ids"][0], got["distances"][0], strict=True):
        assert dist == pytest.approx(expected[doc_id], abs=1e-6)


def test_default_space_is_l2_so_cosine_must_be_explicit(client: Any) -> None:
    col = client.get_or_create_collection("contract_default", embedding_function=None)
    assert col.configuration["hnsw"]["space"] == "l2"
    assert _cosine_collection(client, "contract_cos2").configuration["hnsw"]["space"] == "cosine"


def test_no_embedding_function_is_attached(client: Any) -> None:
    col = _cosine_collection(client, "contract_noef")
    assert col._embedding_function is None
    assert col.configuration["embedding_function"] is None


@pytest.mark.parametrize("op", ["query", "add"])
def test_text_without_embeddings_fails_instead_of_downloading_a_model(client: Any, op: str) -> None:
    """P6: never fall back to the default ONNX embedder (it downloads at first use). This
    runs with sockets disabled, so a download attempt would also fail loudly."""
    col = _cosine_collection(client, "contract_text")
    with pytest.raises(ValueError, match="must provide an embedding function"):
        if op == "query":
            col.query(query_texts=["hello"], n_results=1)
        else:
            col.add(ids=["d"], documents=["no embedding"])


def test_explicit_embeddings_work_with_sockets_disabled(client: Any) -> None:
    col = _cosine_collection(client, "contract_offline")
    col.upsert(ids=["x"], embeddings=[[0.6, 0.8, 0.0]], documents=["x"], metadatas=[{"k": 1}])
    assert col.count() == 1
    assert col.get(ids=["x"], include=["metadatas"])["metadatas"] == [{"k": 1}]


@pytest.mark.integration
def test_client_and_server_interoperate_over_http() -> None:
    """P8 (amended, D-015): the server's reported version is independent of the package
    version (`chromadb/chroma:1.5.9` says "1.0.0"), so compatibility is proven functionally:
    the exact calls the vector store makes must round-trip through the real server."""
    s = get_settings()
    http = chromadb.HttpClient(
        host=s.chroma_host, port=s.chroma_port, settings=Settings(anonymized_telemetry=False)
    )
    assert http.heartbeat() > 0
    name = "contract_http_roundtrip"
    if name in {c.name for c in http.list_collections()}:
        http.delete_collection(name)  # leftover from an interrupted run on a warm volume
    col = _cosine_collection(http, name)
    try:
        assert col.configuration["hnsw"]["space"] == "cosine"
        assert col.configuration["embedding_function"] is None
        meta = [{"doc_id": k, "index": i} for i, k in enumerate(VECTORS)]
        col.upsert(
            ids=list(VECTORS), embeddings=list(VECTORS.values()), documents=list(VECTORS),
            metadatas=meta,
        )  # fmt: skip
        col.upsert(ids=["a"], embeddings=[VECTORS["a"]], documents=["a"], metadatas=[meta[0]])
        assert col.count() == len(VECTORS)  # upsert is idempotent by id
        got = col.query(query_embeddings=[QUERY], n_results=3, include=["distances", "metadatas"])
        q = np.asarray(QUERY)
        for doc_id, dist in zip(got["ids"][0], got["distances"][0], strict=True):
            v = np.asarray(VECTORS[doc_id])
            cosine = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
            assert dist == pytest.approx(1.0 - cosine, abs=1e-6)
        assert got["ids"][0] == ["b", "a", "c"]
        assert got["metadatas"][0][0] == {"doc_id": "b", "index": 1}
        col.delete(ids=["c"])
        assert sorted(col.get()["ids"]) == ["a", "b"]
        reopened = http.get_collection(name, embedding_function=None)
        assert reopened.configuration["hnsw"]["space"] == "cosine"
    finally:
        http.delete_collection(name)
    assert name not in {c.name for c in http.list_collections()}
