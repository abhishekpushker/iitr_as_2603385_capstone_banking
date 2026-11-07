"""One-shot indexer (§ARCH 3.5, 7.4). Run with `python -m scripts.indexer` (the `indexer` service).

load KB → chunk both ways → embed → upsert into the fixed and sentence collections → delete ids
no longer produced (no orphans after a KB change) → write `kb_version` and counts to `kb_meta`
→ print counts → exit 0. Chunk ids are deterministic, so a re-run changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from credagent.kb.chunking import STRATEGIES, Strategy, chunk_document, chunk_params
from credagent.kb.loader import load_kb
from credagent.kb.version import kb_version
from credagent.rag.embedder import Encoder, get_embedder
from credagent.rag.indexing import chunk_metadata
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import Settings, get_settings


@dataclass(frozen=True)
class CollectionNames:
    fixed: str
    sentence: str
    meta: str

    @classmethod
    def from_settings(cls, s: Settings) -> CollectionNames:
        return cls(fixed=s.collection_fixed, sentence=s.collection_sentence, meta=s.collection_meta)

    def for_strategy(self, strategy: Strategy) -> str:
        return self.fixed if strategy == "fixed" else self.sentence


@dataclass(frozen=True)
class IndexReport:
    kb_version: str
    counts: dict[str, int]
    deleted: dict[str, int]


def index_kb(
    kb_dir: Path, store: VectorStore, embedder: Encoder, names: CollectionNames
) -> IndexReport:
    docs = load_kb(kb_dir)
    version = kb_version(kb_dir)
    counts: dict[str, int] = {}
    deleted: dict[str, int] = {}
    for strategy in STRATEGIES:
        name = names.for_strategy(strategy)
        pairs = [(c, d) for d in docs for c in chunk_document(d, strategy)]
        store.upsert(
            name,
            ids=[c.chunk_id for c, _ in pairs],
            embeddings=embedder.encode([c.text for c, _ in pairs]),
            documents=[c.text for c, _ in pairs],
            metadatas=[chunk_metadata(c, d.title, version) for c, d in pairs],
        )
        stale = sorted(store.ids(name) - {c.chunk_id for c, _ in pairs})
        store.delete(name, stale)
        counts[strategy], deleted[strategy] = store.count(name), len(stale)
    store.write_meta(
        names.meta,
        {
            "kb_version": version,
            "collection_fixed": names.fixed,
            "collection_sentence": names.sentence,
            "count_fixed": counts["fixed"],
            "count_sentence": counts["sentence"],
            "doc_count": len(docs),
            "chunk_params": json.dumps(chunk_params(), sort_keys=True),
        },
    )
    return IndexReport(kb_version=version, counts=counts, deleted=deleted)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        s = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=s.kb_dir)
    args = parser.parse_args(argv)

    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    names = CollectionNames.from_settings(s)
    report = index_kb(args.kb_dir, VectorStore(client), get_embedder(), names)
    print(f"kb_version {report.kb_version}")
    for strategy in STRATEGIES:
        print(
            f"{names.for_strategy(strategy)}: {report.counts[strategy]} chunks "
            f"(stale deleted: {report.deleted[strategy]})"
        )
    print(f"{names.meta}: kb_version written")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
