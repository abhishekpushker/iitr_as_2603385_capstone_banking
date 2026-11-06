"""Threshold calibration against the real index (§ARCH 7.5, Task 4).

Re-measuring yields byte-identical JSON; the gated (sentence) collection separates in-scope from
out-of-scope by ≥ MIN_GAP with `max_oos < threshold < min_in`; the calibration queries share
nothing with the demo, held-out, P/R or eval queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credagent.kb.chunking import chunk_params
from credagent.kb.version import kb_version
from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import get_settings
from scripts import calibrate_threshold, indexer
from tests.unit.test_query_sets import assert_calibration_disjoint

KB_DIR = Path(__file__).resolve().parents[2] / "kb"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    assert indexer.main([]) == 0  # the index is idempotent; make sure it reflects kb/
    s = get_settings()
    client = connect_http(
        s.chroma_host, s.chroma_port, attempts=s.chroma_connect_attempts,
        backoff_cap_s=s.chroma_backoff_cap_s,
    )  # fmt: skip
    return Retriever(
        VectorStore(client), get_embedder(), collections_from_settings(s), top_k=s.top_k,
        metrics=Metrics(),
    )  # fmt: skip


@pytest.fixture(scope="module")
def results(retriever: Retriever) -> dict[str, cal.CollectionCalibration]:
    return cal.calibrate(retriever, top_k=get_settings().top_k)


def _json(results: dict[str, cal.CollectionCalibration]) -> str:
    s = get_settings()
    return cal.dumps_config(
        cal.build_config(
            results, kb_version=kb_version(KB_DIR), top_k=s.top_k, chunk_params=chunk_params(),
            embed_model=s.embed_model_path.name,
        )
    )  # fmt: skip


def test_re_measurement_gives_identical_json(
    retriever: Retriever, results: dict[str, cal.CollectionCalibration]
) -> None:
    again = cal.calibrate(retriever, top_k=get_settings().top_k)
    assert _json(again) == _json(results)


def test_gated_collection_separates_with_the_minimum_gap(
    results: dict[str, cal.CollectionCalibration],
) -> None:
    gated = results[cal.GATED_STRATEGY]
    assert gated.threshold is not None
    assert gated.max_oos < gated.threshold < gated.min_in
    assert gated.gap >= cal.MIN_GAP
    assert gated.meets_min_gap


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_both_collections_are_measured_for_every_query(
    results: dict[str, cal.CollectionCalibration], strategy: str
) -> None:
    c = results[strategy]
    assert [m.query for m in c.in_scope] == [q.query for q in cal.IN_SCOPE_QUERIES]
    assert [m.query for m in c.out_of_scope] == list(cal.OUT_OF_SCOPE_QUERIES)
    assert all(-1.0 <= m.top1_similarity <= 1.0 for m in (*c.in_scope, *c.out_of_scope))


def test_in_scope_queries_retrieve_their_document_first(
    results: dict[str, cal.CollectionCalibration],
) -> None:
    gated = results[cal.GATED_STRATEGY]
    assert gated.in_scope_top1_correct == len(gated.in_scope)


def test_script_writes_the_same_json(
    tmp_path: Path, results: dict[str, cal.CollectionCalibration]
) -> None:
    out = tmp_path / "retrieval.json"
    assert calibrate_threshold.main(["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == _json(results)
    assert json.loads(out.read_text(encoding="utf-8"))["gated_strategy"] == "sentence"


def test_calibration_queries_are_disjoint_from_every_other_query_set() -> None:
    assert_calibration_disjoint()
