"""Chunking evaluation (§ARCH 7.6, Task 5): P/R arithmetic, report, choice and recommendation.

Every expected number below is computed by hand from the fake retrievals, not by the module.
The real measurement against Chroma is in tests/integration/test_chunking_eval.py.
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent import text
from credagent.kb.loader import load_kb
from credagent.rag import calibration as cal
from credagent.rag import evaluation as ev
from credagent.rag.retriever import NO_MATCH_SIMILARITY, RetrievalResult
from credagent.rag.store import RetrievedChunk

KB_DIR = Path(__file__).resolve().parents[2] / "kb"

# --------------------------------------------------------------------------- pure P/R math


def test_dedup_keeps_first_occurrence_in_rank_order() -> None:
    assert ev.dedup_doc_ids(["kb-08", "kb-07", "kb-08", "kb-01", "kb-07"]) == (
        "kb-08",
        "kb-07",
        "kb-01",
    )
    assert ev.dedup_doc_ids([]) == ()


def test_single_relevant_doc_retrieved_by_three_duplicate_chunks() -> None:
    s = ev.score(["kb-06"], ["kb-06", "kb-06", "kb-06"])
    assert s.retrieved == ("kb-06",)
    assert (s.hits, s.precision, s.recall) == (1, Fraction(1), Fraction(1))
    assert s.precision_arithmetic == "P = 1/1 = 1.000"
    assert s.recall_arithmetic == "R = 1/1 = 1.000"


def test_two_relevant_docs_both_retrieved() -> None:
    s = ev.score(["kb-11", "kb-12"], ["kb-12", "kb-11", "kb-12"])
    assert s.retrieved == ("kb-12", "kb-11")
    assert (s.precision, s.recall) == (Fraction(1), Fraction(1))
    assert (s.precision_arithmetic, s.recall_arithmetic) == ("P = 2/2 = 1.000", "R = 2/2 = 1.000")


def test_two_relevant_docs_one_retrieved_plus_a_distractor() -> None:
    s = ev.score(["kb-07", "kb-08"], ["kb-08", "kb-08", "kb-01"])
    assert s.retrieved == ("kb-08", "kb-01")
    assert (s.hits, s.precision, s.recall) == (1, Fraction(1, 2), Fraction(1, 2))
    assert (s.precision_arithmetic, s.recall_arithmetic) == ("P = 1/2 = 0.500", "R = 1/2 = 0.500")


def test_two_relevant_docs_one_retrieved_alone() -> None:
    s = ev.score(["kb-11", "kb-12"], ["kb-11", "kb-11", "kb-11"])
    assert (s.precision, s.recall) == (Fraction(1), Fraction(1, 2))
    assert (s.precision_arithmetic, s.recall_arithmetic) == ("P = 1/1 = 1.000", "R = 1/2 = 0.500")


def test_three_distinct_docs_one_relevant() -> None:
    s = ev.score(["kb-09"], ["kb-03", "kb-09", "kb-06"])
    assert (s.precision, s.recall) == (Fraction(1, 3), Fraction(1))
    assert s.precision_arithmetic == "P = 1/3 = 0.333"


def test_complete_miss() -> None:
    s = ev.score(["kb-13"], ["kb-02", "kb-02"])
    assert (s.hits, s.precision, s.recall) == (0, Fraction(0), Fraction(0))
    assert (s.precision_arithmetic, s.recall_arithmetic) == ("P = 0/1 = 0.000", "R = 0/1 = 0.000")


def test_empty_retrieval_defines_precision_zero_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="credagent.rag.evaluation"):
        s = ev.score(["kb-01"], [])
    assert s.retrieved == ()
    assert (s.hits, s.precision, s.recall) == (0, Fraction(0), Fraction(0))
    assert s.precision_arithmetic == "P = 0/0 := 0.000 (empty retrieval)"
    assert s.recall_arithmetic == "R = 0/1 = 0.000"
    assert any("empty retrieval" in r.getMessage() for r in caplog.records)
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_non_empty_retrieval_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="credagent.rag.evaluation"):
        ev.score(["kb-01"], ["kb-01"])
    assert caplog.records == []


@pytest.mark.parametrize("relevant", [[], ["kb-01", "kb-01"]])
def test_gold_must_be_non_empty_and_unique(relevant: list[str]) -> None:
    with pytest.raises(ValueError, match="relevant"):
        ev.score(relevant, ["kb-01"])


@given(
    relevant=st.sets(st.sampled_from([f"kb-{i:02d}" for i in range(1, 14)]), min_size=1),
    retrieved=st.lists(st.sampled_from([f"kb-{i:02d}" for i in range(1, 14)]), max_size=6),
)
def test_precision_and_recall_are_in_unit_interval(
    relevant: set[str], retrieved: list[str]
) -> None:
    s = ev.score(sorted(relevant), retrieved)
    assert 0 <= s.precision <= 1 and 0 <= s.recall <= 1
    assert s.hits == len(set(relevant) & set(retrieved))
    assert len(s.retrieved) == len(set(retrieved))


# --------------------------------------------------------------------------- fake retrieval

Q_JOINT = ev.PRQuery("joint q", ("kb-11", "kb-12"))
Q_PREPAY = ev.PRQuery("prepay q", ("kb-07", "kb-08"))
Q_CLOSE = ev.PRQuery("close q", ("kb-06",))
QUERIES = (Q_JOINT, Q_PREPAY, Q_CLOSE)

# (strategy, query) → ranked (doc_id, similarity) per chunk
HITS: dict[tuple[str, str], list[tuple[str, float]]] = {
    # sentence: P = 2/2, 1/2, 1/1 → 0.833 ; R = 2/2, 1/2, 1/1 → 0.833 ; docs 2, 2, 1
    ("sentence", "joint q"): [("kb-11", 0.80), ("kb-12", 0.79), ("kb-11", 0.60)],
    ("sentence", "prepay q"): [("kb-08", 0.70), ("kb-08", 0.65), ("kb-01", 0.50)],
    ("sentence", "close q"): [("kb-06", 0.90), ("kb-06", 0.70), ("kb-06", 0.60)],
    # fixed: P = 1/1, 2/3, 1/2 → 0.722 ; R = 1/2, 2/2, 1/1 → 0.833 ; docs 1, 3, 2
    ("fixed", "joint q"): [("kb-11", 0.70), ("kb-11", 0.65), ("kb-11", 0.60)],
    ("fixed", "prepay q"): [("kb-07", 0.60), ("kb-02", 0.55), ("kb-08", 0.50)],
    ("fixed", "close q"): [("kb-06", 0.80), ("kb-03", 0.40), ("kb-06", 0.30)],
}


class FakeRetriever:
    def __init__(self, hits: dict[tuple[str, str], list[tuple[str, float]]] = HITS) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str, int | None]] = []

    def collection_name(self, strategy: str) -> str:
        return f"kb_{strategy}_v1"

    def retrieve(self, query: str, strategy: Any, k: int | None = None) -> RetrievalResult:
        self.calls.append((query, strategy, k))
        ranked = self.hits.get((strategy, query), [])[: k or 3]
        chunks = tuple(
            RetrievedChunk(
                chunk_id=f"{doc}::{strategy}::{i:03d}", doc_id=doc, text=f"text {i}",
                similarity=sim, rank=i + 1, char_start=0,
            )
            for i, (doc, sim) in enumerate(ranked)
        )  # fmt: skip
        top1 = chunks[0].similarity if chunks else NO_MATCH_SIMILARITY
        return RetrievalResult(chunks=chunks, top1_similarity=top1, strategy=strategy)


@pytest.fixture
def results() -> dict[str, ev.CollectionEvaluation]:
    return ev.evaluate(FakeRetriever(), top_k=3, queries=QUERIES)


def test_evaluate_queries_both_collections_at_top_k() -> None:
    fake = FakeRetriever()
    out = ev.evaluate(fake, top_k=3, queries=QUERIES)
    assert set(out) == {"fixed", "sentence"}
    assert sorted(fake.calls) == sorted((q.query, s, 3) for q in QUERIES for s in out)
    assert out["sentence"].collection == "kb_sentence_v1"
    assert [qe.query for qe in out["fixed"].queries] == list(QUERIES)


def test_per_query_scores_hand_computed(results: dict[str, ev.CollectionEvaluation]) -> None:
    sent = [qe.score for qe in results["sentence"].queries]
    assert [s.retrieved for s in sent] == [("kb-11", "kb-12"), ("kb-08", "kb-01"), ("kb-06",)]
    assert [s.precision for s in sent] == [1, Fraction(1, 2), 1]
    assert [s.recall for s in sent] == [1, Fraction(1, 2), 1]
    fixed = [qe.score for qe in results["fixed"].queries]
    assert [s.retrieved for s in fixed] == [
        ("kb-11",),
        ("kb-07", "kb-02", "kb-08"),
        ("kb-06", "kb-03"),
    ]
    assert [s.precision for s in fixed] == [1, Fraction(2, 3), Fraction(1, 2)]
    assert [s.recall for s in fixed] == [Fraction(1, 2), 1, 1]
    assert [qe.top1_similarity for qe in results["fixed"].queries] == [0.70, 0.60, 0.80]


def test_macro_averages_hand_computed(results: dict[str, ev.CollectionEvaluation]) -> None:
    sent, fixed = results["sentence"], results["fixed"]
    assert sent.macro_precision == Fraction(5, 6) and sent.macro_recall == Fraction(5, 6)
    assert fixed.macro_precision == Fraction(13, 18) and fixed.macro_recall == Fraction(5, 6)
    assert sent.macro_f1 == Fraction(5, 6)
    # F1 = 2·(13/18)·(5/6) / (13/18 + 5/6) = (130/108) / (28/18) = 65/84
    assert fixed.macro_f1 == Fraction(65, 84)
    assert sent.mean_docs == Fraction(5, 3) and fixed.mean_docs == Fraction(2)
    assert sent.mean_top1 == pytest.approx((0.80 + 0.70 + 0.90) / 3)


def test_macro_arithmetic_is_literal(results: dict[str, ev.CollectionEvaluation]) -> None:
    fixed = results["fixed"]
    assert fixed.macro_precision_arithmetic == "macro P = (1.000 + 0.667 + 0.500) / 3 = 0.722"
    assert fixed.macro_recall_arithmetic == "macro R = (0.500 + 1.000 + 1.000) / 3 = 0.833"


def test_empty_collection_yields_zero_scores(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="credagent.rag.evaluation"):
        out = ev.evaluate(FakeRetriever({}), top_k=3, queries=(Q_CLOSE,), strategies=("fixed",))
    qe = out["fixed"].queries[0]
    assert qe.score.precision == 0 and qe.score.recall == 0
    assert qe.top1_similarity == NO_MATCH_SIMILARITY
    assert out["fixed"].macro_f1 == 0
    assert any("empty retrieval" in r.getMessage() for r in caplog.records)
    assert all("close q" not in r.getMessage() for r in caplog.records)  # no query text logged


def test_evaluate_needs_queries() -> None:
    with pytest.raises(ValueError, match="query"):
        ev.evaluate(FakeRetriever(), top_k=3, queries=())


# --------------------------------------------------------------------------- choice


def _uniform(sims: dict[str, float], docs: dict[str, list[str]]) -> FakeRetriever:
    """Each strategy returns `docs[strategy]` (top-1 similarity `sims[strategy]`) for Q_CLOSE."""
    return FakeRetriever(
        {(s, Q_CLOSE.query): [(d, sims[s] - i / 100) for i, d in enumerate(docs[s])] for s in sims}
    )


def test_choice_by_macro_f1(results: dict[str, ev.CollectionEvaluation]) -> None:
    assert ev.choose(results) == ev.Choice("sentence", "fixed", "macro_f1")


def test_choice_tie_on_f1_broken_by_mean_top1() -> None:
    fake = _uniform({"fixed": 0.81, "sentence": 0.80}, {"fixed": ["kb-06"], "sentence": ["kb-06"]})
    out = ev.evaluate(fake, top_k=3, queries=(Q_CLOSE,))
    assert ev.choose(out) == ev.Choice("fixed", "sentence", "mean_top1")


def test_choice_full_tie_falls_back_to_the_gated_strategy() -> None:
    same = [("kb-06", 0.8), ("kb-03", 0.7), ("kb-06", 0.6)]
    fake = FakeRetriever({("sentence", "close q"): same, ("fixed", "close q"): same})
    out = ev.evaluate(fake, top_k=3, queries=(Q_CLOSE,))
    assert ev.choose(out, gated="fixed") == ev.Choice("fixed", "sentence", "gated")
    assert ev.choose(out) == ev.Choice("sentence", "fixed", "gated")


def test_choice_fewer_docs_wins_on_full_tie_of_f1_and_top1() -> None:
    q = ev.PRQuery("q", ("kb-01",))
    q2 = ev.PRQuery("q2", ("kb-02",))
    fake = FakeRetriever(
        {
            # sentence: q → kb-01: P 1, R 1 ; q2 → nothing relevant: kb-05 → P 0, R 0  (docs 1, 1)
            ("sentence", "q"): [("kb-01", 0.8)],
            ("sentence", "q2"): [("kb-05", 0.6)],
            # fixed: q → kb-01: P 1, R 1 ; q2 → kb-05, kb-06: P 0, R 0  (docs 1, 2)
            ("fixed", "q"): [("kb-01", 0.8)],
            ("fixed", "q2"): [("kb-05", 0.6), ("kb-06", 0.5)],
        }
    )
    out = ev.evaluate(fake, top_k=3, queries=(q, q2))
    assert out["sentence"].macro_f1 == out["fixed"].macro_f1 == Fraction(1, 2)
    assert out["sentence"].mean_top1 == out["fixed"].mean_top1
    assert (out["sentence"].mean_docs, out["fixed"].mean_docs) == (1, Fraction(3, 2))
    assert ev.choose(out, gated="fixed") == ev.Choice("sentence", "fixed", "mean_docs")


def test_choose_needs_two_strategies(results: dict[str, ev.CollectionEvaluation]) -> None:
    with pytest.raises(ValueError, match="two"):
        ev.choose({"sentence": results["sentence"]})


# --------------------------------------------------------------------------- eligibility (D-028)

CALIB_OK = ev.CalibrationSummary(threshold=0.567, gap=0.1746, meets_min_gap=True)
CALIB_LOW = ev.CalibrationSummary(threshold=0.49, gap=0.0329, meets_min_gap=False)
CALIB_OK_FIXED = ev.CalibrationSummary(threshold=0.52, gap=0.08, meets_min_gap=True)
BOTH_OK = {"sentence": CALIB_OK, "fixed": CALIB_OK_FIXED}
FIXED_LOW = {"sentence": CALIB_OK, "fixed": CALIB_LOW}  # the Phase-3 measurement
DECIMAL = re.compile(r"\d+\.\d+")


def _tied_pr_fixed_higher_top1() -> dict[str, ev.CollectionEvaluation]:
    """The measured Phase-4 shape: identical P/R, fixed ahead on mean top-1 by 0.01."""
    fake = _uniform({"fixed": 0.81, "sentence": 0.80}, {"fixed": ["kb-06"], "sentence": ["kb-06"]})
    return ev.evaluate(fake, top_k=3, queries=(Q_CLOSE,))


def test_ineligible_strategy_is_excluded_before_the_tie_break_chain() -> None:
    out = _tied_pr_fixed_higher_top1()
    assert ev.choose(out).winner == "fixed"  # the unfiltered chain would pick fixed
    assert ev.choose(out, calibrations=FIXED_LOW) == ev.Choice(
        "sentence", "fixed", "only_eligible", excluded=("fixed",)
    )


def test_exclusion_overrides_a_better_p_r(results: dict[str, ev.CollectionEvaluation]) -> None:
    calibs = {"sentence": CALIB_LOW, "fixed": CALIB_OK_FIXED}
    assert ev.choose(results).winner == "sentence"  # sentence has the higher macro F1
    assert ev.choose(results, calibrations=calibs) == ev.Choice(
        "fixed", "sentence", "only_eligible", excluded=("sentence",)
    )


def test_all_eligible_keeps_the_chain_unchanged(
    results: dict[str, ev.CollectionEvaluation],
) -> None:
    assert ev.choose(results, calibrations=BOTH_OK) == ev.choose(results)
    out = _tied_pr_fixed_higher_top1()
    assert ev.choose(out, calibrations=BOTH_OK) == ev.Choice("fixed", "sentence", "mean_top1")


def test_no_eligible_strategy_raises(results: dict[str, ev.CollectionEvaluation]) -> None:
    with pytest.raises(ev.NoEligibleStrategyError, match=r"0\.033.*0\.05|fixed"):
        ev.choose(results, calibrations={"sentence": CALIB_LOW, "fixed": CALIB_LOW})
    assert issubclass(ev.NoEligibleStrategyError, ev.NotDeployableError)


def test_missing_calibration_counts_as_ineligible(
    results: dict[str, ev.CollectionEvaluation],
) -> None:
    calibs = {"fixed": CALIB_OK_FIXED}
    choice = ev.choose(results, calibrations=calibs)
    assert choice == ev.Choice("fixed", "sentence", "only_eligible", excluded=("sentence",))
    rec = ev.recommendation(results, choice, calibs, min_gap=0.05, top_k=3)
    assert "`sentence` is excluded before the P/R comparison because it has no calibration" in rec
    with pytest.raises(ev.NoEligibleStrategyError, match="fixed uncalibrated"):
        ev.choose(results, calibrations={})


def test_recommendation_states_the_eligibility_filter_with_measured_numbers() -> None:
    out = _tied_pr_fixed_higher_top1()
    choice = ev.choose(out, calibrations=FIXED_LOW)
    rec = ev.recommendation(out, choice, FIXED_LOW, min_gap=0.05, top_k=3)
    w, lo = out["sentence"], out["fixed"]
    assert 2 <= len(text.split_sentences(rec)) <= 3
    assert "`fixed`" in rec and "excluded" in rec and "`sentence`" in rec
    assert DECIMAL.findall(rec) == [
        _f(w.macro_precision), _f(w.macro_recall), _f(lo.macro_precision), _f(lo.macro_recall),
        "0.033", "0.05", "0.175", "0.05", "0.567",
    ]  # fmt: skip
    report = ev.render_report(out, choice, rec, top_k=3)
    assert "Excluded before the comparison" in report and "`kb_fixed_v1`" in report
    assert lo.macro_precision_arithmetic in report  # P/R of the excluded collection still shown


def _f(x: Fraction | float) -> str:
    return f"{float(x):.3f}"


def test_recommendation_numbers_parse_back_to_the_measured_values(
    results: dict[str, ev.CollectionEvaluation],
) -> None:
    choice = ev.choose(results, calibrations=BOTH_OK)
    rec = ev.recommendation(results, choice, BOTH_OK, min_gap=0.05, top_k=3)
    w, lo = results["sentence"], results["fixed"]
    assert 2 <= len(text.split_sentences(rec)) <= 3
    assert "`sentence`" in rec and "`kb_sentence_v1`" in rec and "`fixed`" in rec
    assert "3 in-scope queries" in rec and "top-3" in rec
    expected = [
        _f(w.macro_precision), _f(w.macro_recall), _f(lo.macro_precision), _f(lo.macro_recall),
        _f(w.macro_f1), _f(lo.macro_f1), _f(w.mean_top1), _f(lo.mean_top1),
        "0.175", "0.05", "0.567",
    ]  # fmt: skip
    assert DECIMAL.findall(rec) == expected
    assert expected[:6] == ["0.833", "0.833", "0.722", "0.833", "0.833", "0.774"]


@pytest.mark.parametrize(
    ("decided_by", "phrase"),
    [
        ("mean_top1", "mean top-1 similarity"),
        ("mean_docs", "distinct documents"),
        ("gated", "gated"),
    ],
)
def test_recommendation_names_the_tie_breaker(
    results: dict[str, ev.CollectionEvaluation], decided_by: Any, phrase: str
) -> None:
    rec = ev.recommendation(
        results, ev.Choice("sentence", "fixed", decided_by), BOTH_OK, min_gap=0.05, top_k=3
    )
    assert phrase in rec and "tie" in rec
    assert 2 <= len(text.split_sentences(rec)) <= 3
    w, lo = results["sentence"], results["fixed"]
    numbers = DECIMAL.findall(rec)
    assert numbers[:4] == [_f(w.macro_precision), _f(w.macro_recall), _f(lo.macro_precision),
                           _f(lo.macro_recall)]  # fmt: skip
    if decided_by == "mean_docs":
        assert _f(w.mean_docs) in numbers and _f(lo.mean_docs) in numbers


def test_recommendation_refuses_a_winner_below_the_minimum_gap(
    results: dict[str, ev.CollectionEvaluation],
) -> None:
    rec = ev.recommendation(
        results, ev.Choice("fixed", "sentence", "macro_f1"), FIXED_LOW, min_gap=0.05, top_k=3
    )
    assert "below the 0.05 minimum" in rec and "not deployed" in rec
    assert DECIMAL.findall(rec)[-2:] == ["0.033", "0.05"]


# --------------------------------------------------------------------------- report


def test_report_contains_the_literal_arithmetic(
    results: dict[str, ev.CollectionEvaluation],
) -> None:
    choice = ev.choose(results, calibrations=BOTH_OK)
    rec = ev.recommendation(results, choice, BOTH_OK, min_gap=0.05, top_k=3)
    report = ev.render_report(results, choice, rec, top_k=3)
    assert "Excluded before the comparison" not in report
    for c in results.values():
        assert f"`{c.collection}`" in report
        for qe in c.queries:
            assert qe.score.precision_arithmetic in report
            assert qe.score.recall_arithmetic in report
        assert c.macro_precision_arithmetic in report
        assert c.macro_recall_arithmetic in report
    for literal in ("P = 2/3 = 0.667", "R = 1/2 = 0.500", "P = 1/2 = 0.500", "R = 2/2 = 1.000"):
        assert literal in report
    assert "kb-07, kb-02, kb-08" in report  # rank order kept after dedup
    assert rec in report
    assert "`sentence`" in report.split("Recommendation", 1)[1]
    assert report == ev.render_report(results, choice, rec, top_k=3)  # deterministic
    assert report.endswith("\n")


# --------------------------------------------------------------------------- deployment


def _config() -> dict[str, Any]:
    return {
        "deployed_strategy": None,
        "gated_strategy": "sentence",
        "min_gap": 0.05,
        "collections": {
            "fixed": {"threshold": 0.49, "gap": 0.0329, "meets_min_gap": False},
            "sentence": {"threshold": 0.567, "gap": 0.1746, "meets_min_gap": True},
        },
    }


def test_calibration_summary_is_read_from_the_config() -> None:
    assert ev.calibration_of(_config(), "sentence") == CALIB_OK
    assert ev.calibration_of(_config(), "fixed") == CALIB_LOW


def test_deploy_sets_only_deployed_strategy() -> None:
    before = _config()
    after = ev.deploy(before, "sentence")
    assert after["deployed_strategy"] == "sentence"
    assert {k: v for k, v in after.items() if k != "deployed_strategy"} == {
        k: v for k, v in before.items() if k != "deployed_strategy"
    }
    assert before["deployed_strategy"] is None  # input not mutated
    assert cal.dumps_config(after) == cal.dumps_config(ev.deploy(after, "sentence"))


def test_deploy_refuses_a_strategy_below_the_minimum_gap() -> None:
    with pytest.raises(ev.NotDeployableError, match="fixed"):
        ev.deploy(_config(), "fixed")


# --------------------------------------------------------------------------- the query set


def test_pr_query_set_shape() -> None:
    """Task 5 scores "the same ≥5 queries from Task 4": one shared set (transcripts 05 and 06)."""
    assert len(ev.PR_QUERIES) == 10
    assert len({q.query for q in ev.PR_QUERIES}) == 10
    assert sum(len(q.relevant_doc_ids) == 2 for q in ev.PR_QUERIES) >= 2
    gold = {frozenset(q.relevant_doc_ids) for q in ev.PR_QUERIES}
    assert frozenset({"kb-11-joint-accounts", "kb-12-nri-eligibility"}) in gold
    assert frozenset({"kb-07-interest-rate-slabs", "kb-08-prepayment-penalty"}) in gold


def test_pr_gold_ids_exist_in_the_kb() -> None:
    ids = {d.doc_id for d in load_kb(KB_DIR)}
    for q in ev.PR_QUERIES:
        assert q.relevant_doc_ids and set(q.relevant_doc_ids) <= ids
        assert len(set(q.relevant_doc_ids)) == len(q.relevant_doc_ids)
