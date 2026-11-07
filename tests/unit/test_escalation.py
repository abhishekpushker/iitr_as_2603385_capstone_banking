"""Escalation score and threshold (§ARCH 7.7, Task 6).

score = round(0.4·flag + 0.6·min(max(d, 0), 30)/30, 4); T = round(0.6·P80/30, 4) with the
nearest-rank P80 = sorted(days)[ceil(0.8·n) − 1]; escalate iff score ≥ T.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import dataset
from credagent.tools import escalation as esc

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "config" / "escalation.json"


def test_weights_are_the_architecture_values() -> None:
    assert (esc.FLAG_WEIGHT, esc.RECENCY_WEIGHT, esc.RECENCY_CAP_DAYS, esc.PERCENTILE) == (
        0.4, 0.6, 30, 80,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("days", "flagged", "expected"),
    [
        (0, False, 0.0),  # brand new, unflagged
        (0, True, 0.4),  # brand new, flagged: the flag weight alone
        (15, False, 0.3),  # 0.6 · 15/30
        (5, True, 0.5),  # 0.4 + 0.6 · 5/30 = 0.4 + 0.1
        (7, False, 0.14),  # 0.6 · 7/30
        (1, True, 0.42),  # 0.4 + 0.02
        (45, False, 0.6),  # clamped to 30 days
        (-3, True, 0.4),  # clamped to 0 days
    ],
)
def test_hand_computed_scores(days: int, flagged: bool, expected: float) -> None:
    assert esc.escalation_score(days, flagged) == expected


@given(st.integers(0, 30), st.booleans())
def test_score_is_within_unit_interval_and_rounded(days: int, flagged: bool) -> None:
    score = esc.escalation_score(days, flagged)
    assert 0.0 <= score <= 1.0
    assert score == round(score, 4)


@given(st.integers(-10_000, 10_000), st.booleans())
def test_out_of_range_days_are_clamped(days: int, flagged: bool) -> None:
    assert esc.escalation_score(days, flagged) == esc.escalation_score(
        min(max(days, 0), 30), flagged
    )


@given(st.integers(0, 29), st.booleans())
def test_score_is_monotonic_in_days(days: int, flagged: bool) -> None:
    assert esc.escalation_score(days + 1, flagged) > esc.escalation_score(days, flagged)


@given(st.integers(0, 30))
def test_score_is_monotonic_in_flag(days: int) -> None:
    flagged, unflagged = esc.escalation_score(days, True), esc.escalation_score(days, False)
    assert flagged > unflagged
    assert round(flagged - unflagged, 4) == esc.FLAG_WEIGHT


def test_all_days_and_flags_exhaustively_in_bounds() -> None:
    scores = [esc.escalation_score(d, f) for d in range(31) for f in (False, True)]
    assert min(scores) == 0.0 and max(scores) == 1.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([15, 20, 35, 40, 50], 40),  # n=5: ceil(4.0) − 1 = 3
        ([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], 5),  # n=10: sorted[7] of [1,1,2,3,3,4,5,5,6,9]
        ([1, 2, 3, 4, 5, 6, 7], 6),  # n=7: ceil(5.6) − 1 = 5
        ([42], 42),  # n=1
        ([7, 7, 7], 7),  # ties
    ],
)
def test_nearest_rank_percentile_on_fixture_lists(values: list[int], expected: int) -> None:
    assert esc.percentile_nearest_rank(values, 80) == expected


def test_percentile_of_empty_list_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        esc.percentile_nearest_rank([], 80)


@pytest.mark.parametrize("p", [0, 101, -5])
def test_percentile_rank_must_be_in_1_to_100(p: int) -> None:
    with pytest.raises(ValueError, match="percentile"):
        esc.percentile_nearest_rank([1, 2, 3], p)


@given(st.lists(st.integers(0, 30), min_size=1, max_size=500), st.integers(1, 100))
def test_percentile_matches_exact_rational_nearest_rank(values: list[int], p: int) -> None:
    """Integer arithmetic: no float `0.8 · n` can round across an integer."""
    rank = math.ceil(Fraction(p, 100) * len(values))
    assert esc.percentile_nearest_rank(values, p) == sorted(values)[rank - 1]


@pytest.mark.parametrize(("p80", "expected"), [(25, 0.5), (0, 0.0), (30, 0.6), (20, 0.4)])
def test_threshold_from_p80(p80: int, expected: float) -> None:
    assert esc.threshold_from_p80(p80) == expected


@pytest.mark.parametrize(
    ("score", "threshold", "expected"), [(0.5, 0.5, True), (0.4999, 0.5, False), (0.6, 0.5, True)]
)
def test_escalate_iff_score_at_least_threshold(
    score: float, threshold: float, expected: bool
) -> None:
    assert esc.should_escalate(score, threshold) is expected


@given(st.integers(0, 30))
def test_unflagged_application_at_p80_escalates(p80: int) -> None:
    """Interpretation (§ARCH 7.7): unflagged escalates exactly from the P80 age onward."""
    t = esc.threshold_from_p80(p80)
    assert esc.min_days_to_escalate(flagged=False, threshold=t) == p80


@pytest.mark.parametrize(("p80", "cutoff"), [(25, 5), (20, 0), (10, 0), (30, 10)])
def test_flagged_cutoff_is_30_t_minus_flag_over_recency(p80: int, cutoff: int) -> None:
    t = esc.threshold_from_p80(p80)
    assert esc.min_days_to_escalate(flagged=True, threshold=t) == cutoff
    assert cutoff == max(0, math.ceil(round(30 * (t - 0.4) / 0.6, 9)))


def test_unreachable_threshold_has_no_cutoff() -> None:
    assert esc.min_days_to_escalate(flagged=False, threshold=0.61) is None


# ------------------------------------------------------------------ config/escalation.json
def test_build_config_from_dataset() -> None:
    cfg = esc.build_config(dataset.LOAN_APPLICATIONS)
    days = [r["days_since_created"] for r in dataset.LOAN_APPLICATIONS]
    p80 = sorted(days)[math.ceil(0.8 * len(days)) - 1]
    assert cfg.n_records == len(days)
    assert cfg.p80_days_since_created == p80
    assert cfg.threshold == round(0.6 * p80 / 30, 4)
    assert (cfg.flag_weight, cfg.recency_weight, cfg.recency_cap_days, cfg.percentile) == (
        0.4, 0.6, 30, 80,
    )  # fmt: skip
    assert cfg.dataset_sha256 == dataset.sha256(dataset.LOAN_APPLICATIONS)
    assert cfg.unflagged_min_days == esc.min_days_to_escalate(False, cfg.threshold)
    assert cfg.flagged_min_days == esc.min_days_to_escalate(True, cfg.threshold)


def test_committed_config_threshold_equals_recomputed_threshold() -> None:
    committed = esc.load_escalation_config(COMMITTED)
    recomputed = esc.build_config(dataset.LOAN_APPLICATIONS)
    assert committed.threshold == recomputed.threshold
    assert committed == recomputed


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    cfg = esc.build_config(dataset.LOAN_APPLICATIONS)
    path = tmp_path / "escalation.json"
    path.write_text(esc.dumps_config(cfg), encoding="utf-8")
    assert esc.load_escalation_config(path) == cfg
    assert esc.dumps_config(cfg).endswith("\n")
    assert list(json.loads(esc.dumps_config(cfg))) == sorted(json.loads(esc.dumps_config(cfg)))


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    data = json.loads(esc.dumps_config(esc.build_config(dataset.LOAN_APPLICATIONS)))
    data["surprise"] = 1
    path = tmp_path / "escalation.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="surprise"):
        esc.load_escalation_config(path)


def test_config_rejects_a_threshold_inconsistent_with_p80(tmp_path: Path) -> None:
    data = json.loads(esc.dumps_config(esc.build_config(dataset.LOAN_APPLICATIONS)))
    data["threshold"] = 0.1
    path = tmp_path / "escalation.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        esc.load_escalation_config(path)


def test_build_config_on_empty_dataset_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        esc.build_config([])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("flag_weight", 0.5, "must be 1"),
        ("flagged_min_days", 4, "cut-offs"),
        ("unflagged_min_days", None, "cut-offs"),
    ],
)
def test_config_rejects_inconsistent_weights_and_cutoffs(
    field: str, value: object, match: str
) -> None:
    data = json.loads(esc.dumps_config(esc.build_config(dataset.LOAN_APPLICATIONS)))
    data[field] = value
    with pytest.raises(ValueError, match=match):
        esc.EscalationConfig.model_validate(data)
