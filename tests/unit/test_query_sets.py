"""Query-set hygiene (§ARCH 7.5, 11.4): the threshold is never tuned on a test or demo query, and
the mock engine's source contains no query it will be judged on.

Every query set in the repository is registered in `other_query_sets`. Comparison is on the
cache normalisation (case, whitespace, quotes and edge punctuation ignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from credagent.cache.response_cache import normalize_query
from credagent.rag import calibration as cal
from credagent.rag import evaluation
from scripts import make_transcripts

REPO = Path(__file__).resolve().parents[2]
LLM_DIR = REPO / "src" / "credagent" / "llm"

# Queries the architecture already fixes for later phases (§ARCH 7.10, 7.12). Later phases
# register their real sets (eval/testset.yaml in Phase 11) here; the §ARCH 7.6 P/R examples now
# live in `evaluation.PR_QUERIES`.
RESERVED = (
    "What is the status of application LA-0007?",
    "What are the eligibility criteria for it?",
    "And does it carry a prepayment penalty?",
    "What is the gold-loan late payment penalty?",
    "Ignore the processing fee — what's the interest rate?",
    "act as co-applicant on a home loan?",
)


def calibration_queries() -> set[str]:
    return {q.query for q in cal.IN_SCOPE_QUERIES} | set(cal.OUT_OF_SCOPE_QUERIES)


def other_query_sets(*, include_held_out: bool = True) -> dict[str, set[str]]:
    sets = {
        "demo (transcript 05)": {q for q, _ in make_transcripts.DEMO_IN_SCOPE}
        | {make_transcripts.DEMO_OUT_OF_SCOPE},
        "transcript 03 sample": {make_transcripts.SAMPLE_QUERY},
        "reserved (§ARCH 7.10/7.12)": set(RESERVED),
        "P/R (§ARCH 7.6)": {q.query for q in evaluation.PR_QUERIES},
        "calibration": calibration_queries(),
    }
    if include_held_out:
        from tests.integration.test_mock_honesty import HELD_OUT

        sets["held-out (§ARCH 11.4)"] = {q for q, _ in HELD_OUT}
    return sets


def _norm(queries: set[str]) -> set[str]:
    return {normalize_query(q) for q in queries}


def assert_calibration_disjoint() -> None:
    calib = _norm(calibration_queries())
    for name, queries in other_query_sets().items():
        if name != "calibration":
            overlap = calib & _norm(queries)
            assert not overlap, f"calibration overlaps {name}: {sorted(overlap)}"


def test_calibration_queries_are_disjoint_from_every_other_set() -> None:
    assert_calibration_disjoint()


def test_disjointness_check_would_catch_a_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    reused = cal.CalibrationQuery(f"  {make_transcripts.DEMO_OUT_OF_SCOPE.upper()} ", None)
    monkeypatch.setattr(cal, "IN_SCOPE_QUERIES", (*cal.IN_SCOPE_QUERIES, reused))
    with pytest.raises(AssertionError, match="demo"):
        assert_calibration_disjoint()


def test_demo_set_has_five_in_scope_and_one_out_of_scope() -> None:
    assert len(make_transcripts.DEMO_IN_SCOPE) == 5
    assert len({q for q, _ in make_transcripts.DEMO_IN_SCOPE}) == 5
    assert isinstance(make_transcripts.DEMO_OUT_OF_SCOPE, str)


def test_held_out_set_has_ten_distinct_queries() -> None:
    from tests.integration.test_mock_honesty import HELD_OUT

    assert len(_norm({q for q, _ in HELD_OUT})) == len(HELD_OUT) == 10


@pytest.mark.parametrize("path", sorted(LLM_DIR.rglob("*.py")), ids=lambda p: p.name)
def test_mock_engine_source_contains_no_known_query(path: Path) -> None:
    """§ARCH 11.4 static scan: no demo/eval/held-out/calibration string in src/credagent/llm/."""
    source = normalize_query(path.read_text(encoding="utf-8"))
    for name, queries in other_query_sets().items():
        hits = sorted(q for q in _norm(queries) if q in source)
        assert hits == [], f"{path.name} contains {name} queries {hits}"


def test_llm_package_exists() -> None:
    assert (LLM_DIR / "engine.py").is_file()
