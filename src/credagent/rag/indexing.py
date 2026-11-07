"""Index one document into both collections (§ARCH 7.4, 7.16). Shared by the one-shot indexer
(`scripts/indexer.py`) and `POST /v1/documents`, so a document added at runtime gets exactly the
chunks, ids and metadata the indexer would have written.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from credagent.kb.chunking import STRATEGIES, Chunk, ChunkSource, Strategy, chunk_document
from credagent.rag.embedder import Encoder
from credagent.rag.store import Metadata, VectorStore


class IndexableDocument(ChunkSource, Protocol):
    @property
    def title(self) -> str: ...


def chunk_metadata(chunk: Chunk, title: str, version: str) -> Metadata:
    return {
        "doc_id": chunk.doc_id,
        "strategy": chunk.strategy,
        "index": chunk.index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "kb_version": version,
        "title": title,
    }


def index_document(
    store: VectorStore,
    encoder: Encoder,
    collections: Mapping[Strategy, str],
    doc: IndexableDocument,
    version: str,
) -> dict[Strategy, int]:
    """Replace `doc`'s chunks in every collection (no orphans when it shrinks); chunk counts."""
    counts: dict[Strategy, int] = {}
    for strategy in STRATEGIES:
        name = collections[strategy]
        chunks = chunk_document(doc, strategy)
        store.delete(name, store.ids_for_doc(name, doc.doc_id))
        store.upsert(
            name,
            ids=[c.chunk_id for c in chunks],
            embeddings=encoder.encode([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[chunk_metadata(c, doc.title, version) for c in chunks],
        )
        counts[strategy] = len(chunks)
    return counts
