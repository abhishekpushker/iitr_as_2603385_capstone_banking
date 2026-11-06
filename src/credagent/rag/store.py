"""Vector store over an injected chromadb client (§ARCH 7.4; P6, P7, P8).

The same class serves production (`chromadb.HttpClient`, via `connect_http`) and unit tests
(`chromadb.EphemeralClient`). Collections never get an embedding function (P6: no ONNX download)
and always use cosine space, so `similarity = 1 − distance` (P7). Every write carries explicit
embeddings.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, ConfigDict

MetadataValue = str | int | float | bool
Metadata = Mapping[str, MetadataValue]

_COSINE: dict[str, Any] = {"hnsw": {"space": "cosine"}}
_META_ID = "kb_meta"
_META_VECTOR = [1.0]  # Chroma requires a document and an embedding on every record


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    rank: int


def connect_http(
    host: str,
    port: int,
    *,
    attempts: int,
    backoff_cap_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """`chromadb.HttpClient` once the server answers a heartbeat (§ARCH 3.5): up to `attempts`
    tries, sleeping 0.5 s, 1 s, 2 s, … between them, each sleep capped at `backoff_cap_s`."""
    last: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            sleep(min(backoff_cap_s, 0.5 * 2 ** (attempt - 1)))
        try:
            client = chromadb.HttpClient(
                host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
            )
            client.heartbeat()
        except Exception as exc:  # noqa: BLE001  (any failure to connect is retried, bounded)
            last = exc
        else:
            return client
    raise ConnectionError(
        f"chroma at {host}:{port} unreachable after {attempts} attempts"
    ) from last


class VectorStore:
    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------ collections
    def collection(self, name: str, metadata: Metadata | None = None) -> Any:
        return self._client.get_or_create_collection(
            name,
            embedding_function=None,
            configuration=_COSINE,
            metadata=dict(metadata) if metadata else None,
        )

    def list_collections(self) -> list[str]:
        return sorted(c.name for c in self._client.list_collections())

    def count(self, name: str) -> int:
        return int(self.collection(name).count())

    def ids(self, name: str) -> set[str]:
        return set(self.collection(name).get(include=[])["ids"])

    # ------------------------------------------------------------------ writes
    def upsert(
        self,
        name: str,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Metadata],
    ) -> None:
        if not len(ids) == len(embeddings) == len(documents) == len(metadatas):
            raise ValueError("ids, embeddings, documents and metadatas must have the same length")
        if not ids:
            raise ValueError("upsert needs at least one record with explicit embeddings")
        self.collection(name).upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[dict(m) for m in metadatas],
        )

    def delete(self, name: str, ids: Sequence[str]) -> None:
        if ids:
            self.collection(name).delete(ids=list(ids))

    # ------------------------------------------------------------------ query
    def query(self, name: str, embedding: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity, descending; ties broken by chunk_id."""
        got = self.collection(name).query(
            query_embeddings=[list(embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        rows = sorted(
            zip(got["ids"][0], got["documents"][0], got["metadatas"][0], got["distances"][0],
                strict=True),
            key=lambda r: (-(1.0 - r[3]), r[0]),
        )  # fmt: skip
        return [
            RetrievedChunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                text=doc,
                similarity=1.0 - dist,
                rank=rank,
            )  # fmt: skip
            for rank, (cid, doc, meta, dist) in enumerate(rows, start=1)
        ]

    # ------------------------------------------------------------------ meta record
    def write_meta(self, name: str, values: Metadata) -> None:
        self.collection(name).upsert(
            ids=[_META_ID], embeddings=[_META_VECTOR], documents=[_META_ID],
            metadatas=[dict(values)],
        )  # fmt: skip

    def read_meta(self, name: str) -> dict[str, MetadataValue] | None:
        got = self.collection(name).get(ids=[_META_ID], include=["metadatas"])
        return dict(got["metadatas"][0]) if got["ids"] else None
