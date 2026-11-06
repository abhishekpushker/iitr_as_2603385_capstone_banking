"""Threshold calibration arithmetic and `config/retrieval.json` round trip (§ARCH 7.5).
The real measurement against Chroma is in tests/integration/test_calibration.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credagent.rag import calibration as cal
from credagent.rag.retriever import RetrievalResult
from credagent.rag.store import RetrievedChunk

SIMS = {
    # (strategy, query) → (top-1 similarity, top-1 doc)
    **{("sentence", q): (0.70 + i / 100, d) for i, (q, d) in enumerate(
        [("in a", "kb-01"), ("in b", "kb-02"), ("in c", "kb-03")])},
    ("sentence", "out x"): (0.30, "kb-09"),
    ("sentence", "out y"): (0.48, "kb-07"),
    ("fixed", "in a"): (0.52, "kb-01"),
    ("fixed", "in b"): (0.60, "kb-02"),
    ("fixed", "in c"): (0.61, "kb-04"),
    ("fixed", "out x"): (0.55, "kb-09"),
    ("fixed", "out y"): (0.20, "kb-07"),
}  # fmt: skip
IN_SCOPE = (
    cal.CalibrationQuery("in a", "kb-01"),
    cal.CalibrationQuery("in b", "kb-02"),
    cal.CalibrationQuery("in c", "kb-03"),
)
OUT = ("out x", "out y")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def collection_name(self, strategy: str) -> str:
        return f"kb_{strategy}_v1"

    def retrieve(self, query: str, strategy: str, k: int | None = None) -> RetrievalResult:
        self.calls.append((query, strategy, k))
        sim, doc = SIMS[(strategy, query)]
        c = RetrievedChunk(chunk_id=f"{doc}::x", doc_id=doc, text="t", similarity=sim, rank=1,
                           char_start=0)  # fmt: skip
        return RetrievalResult(chunks=(c,), top1_similarity=sim, strategy=strategy)  # type: ignore[arg-type]


@pytest.fixture
def results() -> dict[str, cal.CollectionCalibration]:
    return cal.calibrate(FakeRetriever(), in_scope=IN_SCOPE, out_of_scope=OUT, top_k=3)


def test_constants() -> None:
    assert cal.MIN_GAP == 0.05
    assert cal.GATED_STRATEGY == "sentence"
    assert len(cal.IN_SCOPE_QUERIES) >= 5 and len(cal.OUT_OF_SCOPE_QUERIES) >= 5
    texts = [q.query for q in cal.IN_SCOPE_QUERIES] + list(cal.OUT_OF_SCOPE_QUERIES)
    assert len(texts) == len(set(texts))


def test_separable_collection(results: dict[str, cal.CollectionCalibration]) -> None:
    s = results["sentence"]
    assert s.collection == "kb_sentence_v1"
    assert (s.max_oos, s.min_in) == (0.48, 0.70)
    assert s.gap == pytest.approx(0.22)
    assert s.threshold == round((0.48 + 0.70) / 2, 3) == 0.59
    assert s.meets_min_gap
    assert [m.top1_doc_id for m in s.in_scope] == ["kb-01", "kb-02", "kb-03"]
    assert all(m.expected_doc_id is not None for m in s.in_scope)
    assert s.in_scope_top1_correct == 3


def test_overlapping_collection_has_no_threshold(
    results: dict[str, cal.CollectionCalibration],
) -> None:
    f = results["fixed"]
    assert (f.max_oos, f.min_in) == (0.55, 0.52)
    assert f.gap == pytest.approx(-0.03)
    assert f.threshold is None and not f.meets_min_gap
    assert f.in_scope_top1_correct == 2


def test_positive_gap_below_minimum_keeps_the_midpoint_but_fails_the_gate() -> None:
    c = cal.CollectionCalibration(
        strategy="fixed", collection="c",
        in_scope=(cal.Measurement("q", 0.53, "kb-01", "kb-01"),),
        out_of_scope=(cal.Measurement("o", 0.50, "kb-02", None),),
    )  # fmt: skip
    assert c.threshold == round(1.03 / 2, 3) and not c.meets_min_gap


def test_retrieval_uses_top_k() -> None:
    r = FakeRetriever()
    cal.calibrate(r, in_scope=IN_SCOPE, out_of_scope=OUT, top_k=3)
    assert {k for *_, k in r.calls} == {3}
    assert len(r.calls) == 2 * (len(IN_SCOPE) + len(OUT))


def _config(results: dict[str, cal.CollectionCalibration]) -> dict[str, object]:
    return cal.build_config(
        results, kb_version="abc", top_k=3, chunk_params={"fixed": {"size": 1}},
        embed_model="all-MiniLM-L6-v2",
    )  # fmt: skip


def test_config_is_deterministic_json(results: dict[str, cal.CollectionCalibration]) -> None:
    text = cal.dumps_config(_config(results))
    again = cal.dumps_config(
        _config(cal.calibrate(FakeRetriever(), in_scope=IN_SCOPE, out_of_scope=OUT, top_k=3))
    )
    assert text == again and text.endswith("\n")
    data = json.loads(text)
    assert data["kb_version"] == "abc" and data["top_k"] == 3
    assert data["gated_strategy"] == "sentence" and data["deployed_strategy"] is None
    assert data["min_gap"] == 0.05
    sent = data["collections"]["sentence"]
    assert sent["threshold"] == 0.59 and sent["meets_min_gap"] is True
    assert sent["in_scope"][0] == {
        "query": "in a", "top1_similarity": 0.7, "top1_doc_id": "kb-01",
        "expected_doc_id": "kb-01",
    }  # fmt: skip
    assert data["collections"]["fixed"]["threshold"] is None


def test_load_round_trip(results: dict[str, cal.CollectionCalibration], tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    path.write_text(cal.dumps_config(_config(results)), encoding="utf-8")
    loaded = cal.load_retrieval_config(path)
    assert loaded.kb_version == "abc" and loaded.top_k == 3
    assert loaded.gated_strategy == "sentence" and loaded.deployed_strategy is None
    assert loaded.threshold_for("sentence") == 0.59
    with pytest.raises(ValueError, match="no threshold"):
        loaded.threshold_for("fixed")
