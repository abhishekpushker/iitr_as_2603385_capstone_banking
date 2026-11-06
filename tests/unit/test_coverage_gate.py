"""Separate line and branch coverage thresholds (§ARCH 11.3, §PLAN 3 step [9])."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import coverage_gate


def totals(lines: tuple[int, int], branches: tuple[int, int]) -> dict[str, int]:
    return {
        "covered_lines": lines[0], "num_statements": lines[1],
        "covered_branches": branches[0], "num_branches": branches[1],
    }  # fmt: skip


def test_thresholds_are_the_architecture_values() -> None:
    assert (coverage_gate.LINE_MIN, coverage_gate.BRANCH_MIN) == (90.0, 85.0)


def test_passes_at_the_thresholds() -> None:
    assert coverage_gate.problems(totals((90, 100), (85, 100))) == []


@pytest.mark.parametrize(
    ("lines", "branches", "failing"),
    [((89, 100), (100, 100), "line"), ((100, 100), (84, 100), "branch")],
)
def test_reports_each_shortfall(
    lines: tuple[int, int], branches: tuple[int, int], failing: str
) -> None:
    (problem,) = coverage_gate.problems(totals(lines, branches))
    assert problem.startswith(failing)


def test_no_branches_counts_as_full_branch_coverage() -> None:
    assert coverage_gate.problems(totals((95, 100), (0, 0))) == []


def test_cli_reads_coverage_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"totals": totals((95, 100), (80, 100))}), encoding="utf-8")
    assert coverage_gate.main([str(path)]) == 1
    assert "branch coverage 80.00% < 85.0%" in capsys.readouterr().err
    path.write_text(json.dumps({"totals": totals((95, 100), (90, 100))}), encoding="utf-8")
    assert coverage_gate.main([str(path)]) == 0
    assert "line 95.00% · branch 90.00%" in capsys.readouterr().out
