"""The evaluation report (§ARCH 7.18): a fresh run of `scripts/run_eval.py` over the 15-item test
set writes 15 rows × 4 scores and 4 averages; the averages are recomputed here independently;
the JSON validates and equals the committed `eval/results/eval_report.json`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from credagent.settings import get_settings
from scripts import indexer, run_eval

REPO = Path(__file__).resolve().parents[2]
DIMS = ("accuracy", "grounding", "completeness", "safety")


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert indexer.main([]) == 0
    path = tmp_path_factory.mktemp("eval")
    assert run_eval.main(["--out-dir", str(path)]) == 0
    return path


def test_report_has_15_rows_of_4_scores_and_independently_recomputed_averages(out: Path) -> None:
    raw = (out / "eval_report.json").read_text(encoding="utf-8")
    report = run_eval.EvalReport.model_validate_json(raw)
    ids = [i.id for i in run_eval.load_testset(get_settings().eval_testset_path)]
    assert [r.id for r in report.items] == ids
    for dim in DIMS:
        values = [getattr(r.scores, dim) for r in report.items]
        assert len(values) == 15
        assert getattr(report.averages, dim) == round(sum(values) / len(values), 6), dim

    md = (out / "eval_report.md").read_text(encoding="utf-8")
    rows = [line for line in md.splitlines() if re.match(r"\| E\d\d \|", line)]
    assert len(rows) == 15
    for dim in DIMS:
        assert f"| {dim} | {getattr(report.averages, dim):.3f} |" in md


def test_committed_report_matches_a_fresh_run(out: Path) -> None:
    for name in ("eval_report.json", "eval_report.md"):
        committed = (REPO / "eval" / "results" / name).read_text(encoding="utf-8")
        assert (out / name).read_text(encoding="utf-8") == committed, f"run `make bless`: {name}"
