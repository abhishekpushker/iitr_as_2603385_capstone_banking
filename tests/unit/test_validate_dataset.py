"""`scripts/validate_dataset.py`: Markdown report, exit code, seed procedure (§ARCH 7.1)."""

from __future__ import annotations

import copy

import pytest

import dataset
from scripts import validate_dataset


def test_seed_procedure_starts_at_2026() -> None:
    assert validate_dataset.SEED_PROCEDURE_START == 2026


def test_seed_is_the_first_valid_seed_from_the_start() -> None:
    """The procedure: start at 2026, increment until `validate` passes — never edit records."""
    history = validate_dataset.seed_history(validate_dataset.SEED_PROCEDURE_START, dataset.SEED)
    assert [seed for seed, _ in history] == list(range(2026, dataset.SEED + 1))
    assert history[-1] == (dataset.SEED, [])
    assert all(problems for _, problems in history[:-1])


def test_seed_history_rejects_a_seed_before_the_start() -> None:
    with pytest.raises(ValueError, match="start"):
        validate_dataset.seed_history(2026, 2025)


def test_markdown_report_contains_every_figure() -> None:
    rep = dataset.report(dataset.LOAN_APPLICATIONS)
    md = validate_dataset.render_markdown(rep)
    assert md.startswith("## Dataset report")
    for cat, n in rep["category_counts"].items():
        assert f"| {cat} | {n} |" in md
    for status, n in rep["status_counts"].items():
        assert f"| {status} | {n} |" in md
    assert f"{rep['fraud_flagged']} / {rep['n_records']}" in md
    assert f"{rep['fraud_share']:.2%}" in md
    assert rep["sha256"] in md
    assert "**Valid:** yes" in md


def test_markdown_report_lists_problems() -> None:
    rep = dataset.report(copy.deepcopy(dataset.LOAN_APPLICATIONS[:5]))
    md = validate_dataset.render_markdown(rep)
    assert "**Valid:** no" in md
    for problem in rep["problems"]:
        assert f"- {problem}" in md


def test_main_prints_markdown_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert validate_dataset.main() == 0
    assert capsys.readouterr().out.startswith("## Dataset report")


def test_main_exits_non_zero_when_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    assert validate_dataset.main(copy.deepcopy(dataset.LOAN_APPLICATIONS[:5])) == 1
    assert "**Valid:** no" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("amount", "text"),
    [
        (5_000, "₹5,000"),
        (50_000, "₹50,000"),
        (4_000_000, "₹40,00,000"),
        (20_000_000, "₹2,00,00,000"),
        (12_345_678, "₹1,23,45,678"),
    ],
)
def test_inr_uses_indian_digit_grouping(amount: int, text: str) -> None:
    assert validate_dataset.inr(amount) == text
