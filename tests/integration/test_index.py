"""Indexer against the real chroma service (§ARCH 3.5, 7.4; Tasks 2–3).

Uses the production collections for the main checks (the indexer is idempotent, so re-running
it here is harmless) and throw-away `itest_*` collections for the orphan-cleanup scenario.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from credagent.kb.chunking import STRATEGIES, chunk_document
from credagent.kb.loader import load_kb
from credagent.kb.version import kb_version
from credagent.rag.embedder import Embedder, get_embedder
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import get_settings
from scripts import indexer
from scripts.indexer import CollectionNames, IndexReport, index_kb

KB_DIR = Path(__file__).resolve().parents[2] / "kb"
SAMPLE_QUERY = "Is there a penalty for prepaying a personal loan?"
METADATA_KEYS = {"doc_id", "strategy", "index", "char_start", "char_end", "kb_version", "title"}


@pytest.fixture(scope="module")
def store() -> VectorStore:
    s = get_settings()
    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    return VectorStore(client)


@pytest.fixture(scope="module")
def names() -> CollectionNames:
    return CollectionNames.from_settings(get_settings())


@pytest.fixture(scope="module")
def first_run(store: VectorStore, names: CollectionNames) -> IndexReport:
    """The real CLI entry point, exactly as the `indexer` service runs it; the report comes
    from a direct re-run (idempotent, so nothing is deleted the second time)."""
    assert indexer.main([]) == 0
    return index_kb(KB_DIR, store, get_embedder(), names)


def _expected_ids(kb_dir: Path) -> dict[str, list[str]]:
    docs = load_kb(kb_dir)
    return {s: [c.chunk_id for d in docs for c in chunk_document(d, s)] for s in STRATEGIES}


def test_both_collections_exist_with_the_expected_counts(
    first_run: IndexReport, store: VectorStore, names: CollectionNames
) -> None:
    expected = _expected_ids(KB_DIR)
    present = set(store.list_collections())
    assert {names.fixed, names.sentence, names.meta} <= present
    for strategy, name in (("fixed", names.fixed), ("sentence", names.sentence)):
        assert store.count(name) == len(expected[strategy]) == first_run.counts[strategy]
        assert store.ids(name) == set(expected[strategy])


def test_chunk_metadata_is_complete(
    first_run: IndexReport, store: VectorStore, names: CollectionNames
) -> None:
    for name in (names.fixed, names.sentence):
        got = store.collection(name).get(include=["metadatas", "documents"])
        for cid, meta, doc in zip(got["ids"], got["metadatas"], got["documents"], strict=True):
            assert set(meta) == METADATA_KEYS
            assert meta["kb_version"] == first_run.kb_version
            assert cid == f"{meta['doc_id']}::{meta['strategy']}::{meta['index']:03d}"
            assert int(meta["char_end"]) - int(meta["char_start"]) == len(doc)


def test_second_run_is_idempotent(
    first_run: IndexReport, store: VectorStore, names: CollectionNames
) -> None:
    before = {n: store.ids(n) for n in (names.fixed, names.sentence)}
    report = index_kb(KB_DIR, store, get_embedder(), names)
    assert report == IndexReport(
        kb_version=first_run.kb_version,
        counts=first_run.counts,
        deleted={"fixed": 0, "sentence": 0},
    )
    assert {n: store.ids(n) for n in (names.fixed, names.sentence)} == before


def test_meta_kb_version_is_stable(
    first_run: IndexReport, store: VectorStore, names: CollectionNames
) -> None:
    meta = store.read_meta(names.meta)
    assert meta is not None
    assert meta["kb_version"] == first_run.kb_version == kb_version(KB_DIR)
    assert meta["count_fixed"] == first_run.counts["fixed"]
    assert meta["count_sentence"] == first_run.counts["sentence"]
    index_kb(KB_DIR, store, get_embedder(), names)
    assert store.read_meta(names.meta) == meta


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_sample_query_returns_the_prepayment_doc_first(
    first_run: IndexReport, store: VectorStore, names: CollectionNames, strategy: str
) -> None:
    name = names.fixed if strategy == "fixed" else names.sentence
    (vector,) = get_embedder().encode([SAMPLE_QUERY])
    top = store.query(name, vector, k=3)
    assert len(top) == 3
    assert top[0].doc_id == "kb-08-prepayment-penalty"
    assert [c.rank for c in top] == [1, 2, 3]
    assert top[0].similarity >= top[1].similarity >= top[2].similarity


# ------------------------------------------------------------------ orphan cleanup


@pytest.fixture
def temp_names(store: VectorStore) -> Iterator[CollectionNames]:
    names = CollectionNames(fixed="itest_fixed", sentence="itest_sentence", meta="itest_meta")
    yield names
    for name in (names.fixed, names.sentence, names.meta):
        if name in store.list_collections():
            store._client.delete_collection(name)


def test_orphans_are_deleted_when_a_doc_is_removed(
    tmp_path: Path, store: VectorStore, temp_names: CollectionNames
) -> None:
    kb = tmp_path / "kb"
    shutil.copytree(KB_DIR, kb)
    embedder: Embedder = get_embedder()
    full = index_kb(kb, store, embedder, temp_names)
    removed = "kb-06-account-closure"
    removed_ids = {
        s: [i for i in _expected_ids(kb)[s] if i.startswith(removed)] for s in STRATEGIES
    }
    (kb / f"{removed}.md").unlink()

    report = index_kb(kb, store, embedder, temp_names)
    expected = _expected_ids(kb)
    assert report.deleted == {s: len(removed_ids[s]) for s in STRATEGIES}
    assert all(report.deleted.values())
    assert store.ids(temp_names.fixed) == set(expected["fixed"])
    assert store.ids(temp_names.sentence) == set(expected["sentence"])
    meta = store.read_meta(temp_names.meta)
    assert meta is not None and meta["kb_version"] == kb_version(kb) != full.kb_version


def test_cli_prints_counts(
    capsys: pytest.CaptureFixture[str], first_run: IndexReport, names: CollectionNames
) -> None:
    assert indexer.main([]) == 0
    out = capsys.readouterr().out
    assert f"kb_version {first_run.kb_version}" in out
    assert f"{names.fixed}: {first_run.counts['fixed']} chunks" in out
    assert f"{names.sentence}: {first_run.counts['sentence']} chunks" in out
