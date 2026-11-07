"""Golden determinism for generated config and reports (§ARCH 11.8).

`config/retrieval.json` must equal a fresh calibration followed by the chunking evaluation (which
sets `deployed_strategy`) byte for byte, `eval/results/chunking_pr.md` must equal the freshly
rendered P/R report, `config/escalation.json` must equal a fresh build from the dataset, and
`config/budget.json` must equal a fresh measurement over the eval and demo queries.
Intentional changes are re-blessed with `make bless`.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from scripts import build_escalation_config, calibrate_threshold, eval_chunking, measure_budget

REPO = Path(__file__).resolve().parents[2]
COMMITTED_CONFIG = REPO / "config" / "retrieval.json"
COMMITTED_REPORT = REPO / "eval" / "results" / "chunking_pr.md"
COMMITTED_ESCALATION = REPO / "config" / "escalation.json"
COMMITTED_BUDGET = REPO / "config" / "budget.json"


def _diff(expected: str, actual: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), "committed", "regenerated", lineterm=""
        )
    )


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("generated")
    config, report = out / "retrieval.json", out / "chunking_pr.md"
    assert calibrate_threshold.main(["--out", str(config)]) == 0
    assert eval_chunking.main(["--config", str(config), "--report", str(report)]) == 0
    return config, report


def test_retrieval_config_matches_a_fresh_calibration_and_evaluation(
    regenerated: tuple[Path, Path],
) -> None:
    expected = COMMITTED_CONFIG.read_text(encoding="utf-8")
    actual = regenerated[0].read_text(encoding="utf-8")
    assert actual == expected, f"config/retrieval.json drifted:\n{_diff(expected, actual)}"


def test_escalation_config_matches_a_fresh_build(tmp_path: Path) -> None:
    expected = COMMITTED_ESCALATION.read_text(encoding="utf-8")
    out = tmp_path / "escalation.json"
    assert build_escalation_config.main(["--out", str(out)]) == 0
    actual = out.read_text(encoding="utf-8")
    assert actual == expected, f"config/escalation.json drifted:\n{_diff(expected, actual)}"


def test_chunking_report_matches_a_fresh_evaluation(regenerated: tuple[Path, Path]) -> None:
    expected = COMMITTED_REPORT.read_text(encoding="utf-8")
    actual = regenerated[1].read_text(encoding="utf-8")
    assert actual == expected, f"eval/results/chunking_pr.md drifted:\n{_diff(expected, actual)}"


def test_budget_config_matches_a_fresh_measurement(tmp_path: Path) -> None:
    expected = COMMITTED_BUDGET.read_text(encoding="utf-8")
    out = tmp_path / "budget.json"
    assert measure_budget.main(["--out", str(out)]) == 0
    actual = out.read_text(encoding="utf-8")
    assert actual == expected, f"config/budget.json drifted:\n{_diff(expected, actual)}"
