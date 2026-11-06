"""Chunking evaluation against the real index (§ARCH 7.6, Task 5).

Every in-scope P/R query retrieves at least one document on both collections (so the P := 0
empty-retrieval rule never fires on them), re-measuring is deterministic, and the script sets
`deployed_strategy` to the winner, which must meet the calibration gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credagent.observability.metrics import Metrics
from credagent.rag import calibration as cal
from credagent.rag import evaluation as ev
from credagent.rag.embedder import get_embedder
from credagent.rag.retriever import Retriever, collections_from_settings
from credagent.rag.store import VectorStore, connect_http
from credagent.settings import get_settings
from scripts import calibrate_threshold, eval_chunking, indexer


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    assert indexer.main([]) == 0
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
def results(retriever: Retriever) -> dict[str, ev.CollectionEvaluation]:
    return ev.evaluate(retriever, top_k=get_settings().top_k)


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_no_in_scope_query_has_an_empty_retrieval(
    results: dict[str, ev.CollectionEvaluation], strategy: str
) -> None:
    c = results[strategy]
    assert [qe.query for qe in c.queries] == list(ev.PR_QUERIES)
    assert all(qe.score.retrieved for qe in c.queries)
    assert all(0 <= qe.score.precision <= 1 and 0 <= qe.score.recall <= 1 for qe in c.queries)


def test_re_measurement_renders_an_identical_report(
    retriever: Retriever, results: dict[str, ev.CollectionEvaluation]
) -> None:
    again = ev.evaluate(retriever, top_k=get_settings().top_k)
    choice = ev.choose(results)
    assert ev.choose(again) == choice
    rec = ev.recommendation(
        results,
        choice,
        {s: ev.CalibrationSummary(0.5, 0.1, True) for s in results},
        min_gap=0.05,
        top_k=3,  # fmt: skip
    )
    assert ev.render_report(again, choice, rec, top_k=3) == ev.render_report(
        results, choice, rec, top_k=3
    )


def test_script_deploys_the_winner(
    tmp_path: Path, results: dict[str, ev.CollectionEvaluation]
) -> None:
    config, report = tmp_path / "retrieval.json", tmp_path / "chunking_pr.md"
    assert calibrate_threshold.main(["--out", str(config)]) == 0
    before = json.loads(config.read_text(encoding="utf-8"))
    assert before["deployed_strategy"] is None

    assert eval_chunking.main(["--config", str(config), "--report", str(report)]) == 0
    after = json.loads(config.read_text(encoding="utf-8"))
    calibs = {s: ev.calibration_of(before, s) for s in results}
    winner = ev.choose(results, calibrations=calibs).winner
    assert after["deployed_strategy"] == winner == "sentence"  # fixed gap 0.033 < 0.05 (D-028)
    assert after["collections"][winner]["meets_min_gap"] is True
    assert {k: v for k, v in after.items() if k != "deployed_strategy"} == {
        k: v for k, v in before.items() if k != "deployed_strategy"
    }
    assert cal.load_retrieval_config(config).deployed_strategy == winner
    body = report.read_text(encoding="utf-8")
    assert "P = " in body and "macro P = " in body and f"`{winner}`" in body

    # idempotent: a second run leaves both files byte-identical
    first = (config.read_text(encoding="utf-8"), body)
    assert eval_chunking.main(["--config", str(config), "--report", str(report)]) == 0
    assert (config.read_text(encoding="utf-8"), report.read_text(encoding="utf-8")) == first


def test_script_deploys_nothing_when_no_strategy_meets_the_gap(tmp_path: Path) -> None:
    config, report = tmp_path / "retrieval.json", tmp_path / "chunking_pr.md"
    assert calibrate_threshold.main(["--out", str(config)]) == 0
    data = json.loads(config.read_text(encoding="utf-8"))
    for c in data["collections"].values():
        c["meets_min_gap"] = False
    config.write_text(cal.dumps_config(data), encoding="utf-8")
    before = config.read_text(encoding="utf-8")
    assert eval_chunking.main(["--config", str(config), "--report", str(report)]) == 1
    assert config.read_text(encoding="utf-8") == before
    assert json.loads(before)["deployed_strategy"] is None
    assert not report.exists()


def test_script_refuses_a_config_from_another_kb_version(tmp_path: Path) -> None:
    config, report = tmp_path / "retrieval.json", tmp_path / "chunking_pr.md"
    assert calibrate_threshold.main(["--out", str(config)]) == 0
    data = json.loads(config.read_text(encoding="utf-8"))
    data["kb_version"] = "0" * 64
    config.write_text(cal.dumps_config(data), encoding="utf-8")
    assert eval_chunking.main(["--config", str(config), "--report", str(report)]) == 1
    assert json.loads(config.read_text(encoding="utf-8"))["deployed_strategy"] is None
    assert not report.exists()
