"""Vector store over an injected chromadb client (§ARCH 7.4; P6, P7, P8).

The same class serves production (`chromadb.HttpClient`, via `connect_http`) and unit tests
(`chromadb.EphemeralClient`). Collections never get an embedding function (P6: no ONNX download)
and always use cosine space, so Chroma's nearest neighbours are the most similar (P7). The
similarity itself is the exact float64 cosine of the stored embedding and the query (= 1 − cosine
distance, without Chroma's state-dependent float32 rounding, BUG-005). Every write carries
explicit embeddings.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import chromadb
import numpy as np
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
    char_start: int  # offset of the chunk in its document body (orders extracted sentences)


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

    def ids_for_doc(self, name: str, doc_id: str) -> list[str]:
        return list(self.collection(name).get(where={"doc_id": doc_id}, include=[])["ids"])

    def heartbeat(self) -> int:
        """Raises when the server is unreachable (§ARCH 3.5 readiness)."""
        return int(self._client.heartbeat())

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
        """Top-k chunks by cosine similarity, descending; ties broken by chunk_id.

        Chroma picks the k nearest; the similarity is then computed here, in float64, from the
        stored embeddings (BUG-005): Chroma's own distance differs in the last float32 bits
        depending on whether its write buffer or HNSW answered, i.e. on the write history.
        """
        got = self.collection(name).query(
            query_embeddings=[list(embedding)],
            n_results=k,
            include=["documents", "metadatas", "embeddings"],
        )
        q = np.asarray(embedding, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))

        def cosine(stored: Any) -> float:
            e = np.asarray(stored, dtype=np.float64)
            return float(e @ q / (np.linalg.norm(e) * q_norm))

        rows = sorted(
            ((cid, doc, meta, cosine(emb)) for cid, doc, meta, emb in zip(
                got["ids"][0], got["documents"][0], got["metadatas"][0], got["embeddings"][0],
                strict=True)),
            key=lambda r: (-r[3], r[0]),
        )  # fmt: skip
        return [
            RetrievedChunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                text=doc,
                similarity=sim,
                rank=rank,
                char_start=int(meta["char_start"]),
            )  # fmt: skip
            for rank, (cid, doc, meta, sim) in enumerate(rows, start=1)
        ]

    def documents(self, name: str, ids: Sequence[str]) -> dict[str, str]:
        """`chunk_id → text` for the ids that exist (unknown ids are left out)."""
        if not ids:
            return {}
        got = self.collection(name).get(ids=list(ids), include=["documents"])
        return dict(zip(got["ids"], got["documents"], strict=True))

    # ------------------------------------------------------------------ meta record
    def write_meta(self, name: str, values: Metadata) -> None:
        self.collection(name).upsert(
            ids=[_META_ID], embeddings=[_META_VECTOR], documents=[_META_ID],
            metadatas=[dict(values)],
        )  # fmt: skip

    def read_meta(self, name: str) -> dict[str, MetadataValue] | None:
        got = self.collection(name).get(ids=[_META_ID], include=["metadatas"])
        return dict(got["metadatas"][0]) if got["ids"] else None
