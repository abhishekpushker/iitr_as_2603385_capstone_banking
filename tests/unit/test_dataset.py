"""`dataset.py` — the loan-application dataset (Task 1, §ARCH 7.1)."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, get_args

import pytest

import dataset
from credagent.schemas import LoanCategory, LoanStatus

REPO = Path(__file__).resolve().parents[2]

# Pinned after the first green run of the generator (seed procedure, docs/DECISIONS.md).
# A change here is a dataset change: regenerate transcripts and review the diff.
GOLDEN_SHA256 = "01a4304ab66ae63c78b3d740e2c075e707c31823f68c0eb8e4be1db841caa55d"

FIELDS = (
    "record_id",
    "category",
    "status",
    "loan_amount_inr",
    "days_since_created",
    "flagged_for_fraud_review",
)
CATEGORIES = ("Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan")
STATUSES = ("Submitted", "Under Review", "Approved", "Rejected", "Disbursed")
ID_PATTERN = re.compile(r"^LA-\d{4}$")

Records = list[dict[str, Any]]


def _sha256(records: Records) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


@pytest.fixture
def records() -> Records:
    """A private deep copy: mutating it must never touch `dataset.LOAN_APPLICATIONS`."""
    return copy.deepcopy(dataset.LOAN_APPLICATIONS)


# ---------------------------------------------------------------- constants (§ARCH 7.1)


def test_constants_match_the_architecture() -> None:
    assert dataset.N_RECORDS == 60
    assert dataset.CATEGORY_WEIGHTS == {
        "Personal Loan": 0.30, "Home Loan": 0.20, "Auto Loan": 0.20, "Education Loan": 0.15,
        "Business Loan": 0.15,
    }  # fmt: skip
    assert dataset.STATUS_WEIGHTS == {
        "Submitted": 0.20, "Under Review": 0.30, "Approved": 0.20, "Rejected": 0.15,
        "Disbursed": 0.15,
    }  # fmt: skip
    assert dataset.AMOUNT_RANGES_INR == {
        "Personal Loan": (50_000, 4_000_000),
        "Home Loan": (1_000_000, 20_000_000),
        "Auto Loan": (200_000, 3_000_000),
        "Education Loan": (100_000, 7_500_000),
        "Business Loan": (500_000, 10_000_000),
    }
    assert dataset.FRAUD_FLAG_PROBABILITY == 0.18
    assert isinstance(dataset.SEED, int)


def test_vocabulary_order_is_stable() -> None:
    assert tuple(dataset.CATEGORY_WEIGHTS) == CATEGORIES
    assert tuple(dataset.STATUS_WEIGHTS) == STATUSES
    assert tuple(dataset.AMOUNT_RANGES_INR) == CATEGORIES


@pytest.mark.parametrize("weights", ["CATEGORY_WEIGHTS", "STATUS_WEIGHTS"])
def test_weights_sum_to_one(weights: str) -> None:
    assert sum(getattr(dataset, weights).values()) == pytest.approx(1.0)


def test_amount_bounds_stay_below_the_bank_account_pattern() -> None:
    """Upper bounds have ≤ 8 digits, so no amount collides with the ≥ 9-digit masker (§7.12.1)."""
    for lo, hi in dataset.AMOUNT_RANGES_INR.values():
        assert 0 < lo < hi
        assert len(str(hi)) <= 8
        assert lo % 5_000 == 0
        assert hi % 5_000 == 0


def test_schema_literals_equal_dataset_vocabularies() -> None:
    """`LoanCategory`/`LoanStatus` are shared with dataset.py by assertion, not import (§7.11)."""
    assert get_args(LoanCategory) == tuple(dataset.CATEGORY_WEIGHTS)
    assert get_args(LoanStatus) == tuple(dataset.STATUS_WEIGHTS)


# ---------------------------------------------------------------- the generated dataset


def test_record_count() -> None:
    assert len(dataset.LOAN_APPLICATIONS) == dataset.N_RECORDS == 60
    assert len(dataset.LOAN_APPLICATIONS) >= 40


def test_every_category_has_at_least_three_records() -> None:
    counts = Counter(r["category"] for r in dataset.LOAN_APPLICATIONS)
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 3


def test_every_status_has_at_least_one_record() -> None:
    counts = Counter(r["status"] for r in dataset.LOAN_APPLICATIONS)
    assert set(counts) == set(STATUSES)


def test_fraud_share_is_within_band() -> None:
    flagged = sum(r["flagged_for_fraud_review"] for r in dataset.LOAN_APPLICATIONS)
    assert 0.10 <= flagged / len(dataset.LOAN_APPLICATIONS) <= 0.30


def test_ids_are_unique_sequential_and_pattern_valid() -> None:
    ids = [r["record_id"] for r in dataset.LOAN_APPLICATIONS]
    assert ids == [f"LA-{i:04d}" for i in range(1, 61)]
    assert all(ID_PATTERN.fullmatch(i) for i in ids)
    assert len(set(ids)) == len(ids)


def test_field_set_and_order() -> None:
    for record in dataset.LOAN_APPLICATIONS:
        assert tuple(record) == FIELDS


def test_field_types() -> None:
    for r in dataset.LOAN_APPLICATIONS:
        assert type(r["record_id"]) is str
        assert type(r["category"]) is str
        assert type(r["status"]) is str
        assert type(r["loan_amount_inr"]) is int
        assert type(r["days_since_created"]) is int
        assert type(r["flagged_for_fraud_review"]) is bool


def test_amounts_are_in_category_range_rounded_and_at_most_eight_digits() -> None:
    for r in dataset.LOAN_APPLICATIONS:
        lo, hi = dataset.AMOUNT_RANGES_INR[r["category"]]
        assert lo <= r["loan_amount_inr"] <= hi, r
        assert r["loan_amount_inr"] % 5_000 == 0, r
        assert len(str(r["loan_amount_inr"])) <= 8, r


def test_days_since_created_in_range() -> None:
    for r in dataset.LOAN_APPLICATIONS:
        assert 0 <= r["days_since_created"] <= 30


def test_validate_accepts_the_shipped_dataset() -> None:
    assert dataset.validate(dataset.LOAN_APPLICATIONS) == []


def test_golden_sha256() -> None:
    assert _sha256(dataset.LOAN_APPLICATIONS) == GOLDEN_SHA256


# ---------------------------------------------------------------- determinism


def test_generate_is_deterministic() -> None:
    assert dataset.generate(dataset.SEED) == dataset.generate(dataset.SEED)


def test_module_dataset_is_generate_of_seed() -> None:
    assert dataset.generate(dataset.SEED) == dataset.LOAN_APPLICATIONS


def test_a_different_seed_gives_a_different_dataset() -> None:
    assert dataset.generate(dataset.SEED + 1) != dataset.generate(dataset.SEED)


def test_generate_does_not_touch_the_global_random() -> None:
    """A local `random.Random(seed)` only (§ARCH 7.1)."""
    random.seed(12345)
    state = random.getstate()
    first = dataset.generate(dataset.SEED)
    assert random.getstate() == state
    random.seed(999)
    assert dataset.generate(dataset.SEED) == first


def test_generate_returns_fresh_lists() -> None:
    a = dataset.generate(dataset.SEED)
    a[0]["status"] = "mutated"
    assert dataset.generate(dataset.SEED)[0]["status"] != "mutated"


# ---------------------------------------------------------------- validate() catches each rule


def _problems(records: Records) -> str:
    problems = dataset.validate(records)
    assert problems, "validate() accepted an invalid dataset"
    return "\n".join(problems)


def test_validate_flags_too_few_records(records: Records) -> None:
    assert "records" in _problems(records[:39])


def test_validate_flags_an_empty_dataset() -> None:
    assert "records" in _problems([])


def test_validate_flags_a_thin_category(records: Records) -> None:
    keep = [r for r in records if r["category"] != "Auto Loan"]
    autos = [r for r in records if r["category"] == "Auto Loan"][:2]
    assert "Auto Loan" in _problems(keep + autos)


def test_validate_flags_a_missing_status(records: Records) -> None:
    for r in records:
        if r["status"] == "Rejected":
            r["status"] = "Approved"
    assert "Rejected" in _problems(records)


@pytest.mark.parametrize("flag", [True, False])
def test_validate_flags_fraud_share_outside_band(records: Records, flag: bool) -> None:
    for r in records:
        r["flagged_for_fraud_review"] = flag
    assert "fraud" in _problems(records)


def test_validate_flags_duplicate_ids(records: Records) -> None:
    records[1]["record_id"] = records[0]["record_id"]
    assert "duplicate" in _problems(records)


@pytest.mark.parametrize("bad_id", ["la-0001", "LA-01", "LA-00001", " LA-0001", "LA-000A"])
def test_validate_flags_malformed_ids(records: Records, bad_id: str) -> None:
    records[0]["record_id"] = bad_id
    assert "record_id" in _problems(records)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amount_inr", "500000"),
        ("loan_amount_inr", True),
        ("days_since_created", 3.0),
        ("flagged_for_fraud_review", 1),
        ("record_id", 1),
    ],
)
def test_validate_flags_wrong_types(records: Records, field: str, value: object) -> None:
    records[0][field] = value
    assert field in _problems(records)


@pytest.mark.parametrize(("field", "value"), [("category", "Gold"), ("status", "Pending")])
def test_validate_flags_unknown_vocabulary(records: Records, field: str, value: str) -> None:
    records[0][field] = value
    assert value in _problems(records)


def test_validate_flags_amount_outside_category_range(records: Records) -> None:
    lo, _ = dataset.AMOUNT_RANGES_INR[records[0]["category"]]
    records[0]["loan_amount_inr"] = lo - 5_000
    assert "loan_amount_inr" in _problems(records)


@pytest.mark.parametrize("days", [-1, 31])
def test_validate_flags_days_out_of_range(records: Records, days: int) -> None:
    records[0]["days_since_created"] = days
    assert "days_since_created" in _problems(records)


def test_validate_flags_field_order(records: Records) -> None:
    records[0] = dict(reversed(list(records[0].items())))
    assert "fields" in _problems(records)


def test_validate_flags_an_extra_field(records: Records) -> None:
    records[0]["applicant_name"] = "x"
    assert "fields" in _problems(records)


def test_validate_does_not_mutate_its_input(records: Records) -> None:
    before = copy.deepcopy(records)
    dataset.validate(records)
    assert records == before


# ---------------------------------------------------------------- report() and __main__


def test_report_summarises_the_dataset() -> None:
    rep = dataset.report(dataset.LOAN_APPLICATIONS)
    recs = dataset.LOAN_APPLICATIONS
    assert rep["n_records"] == 60
    assert rep["category_counts"] == {c: sum(r["category"] == c for r in recs) for c in CATEGORIES}
    assert rep["status_counts"] == {s: sum(r["status"] == s for r in recs) for s in STATUSES}
    flagged = sum(r["flagged_for_fraud_review"] for r in recs)
    assert rep["fraud_flagged"] == flagged
    assert rep["fraud_share"] == round(flagged / 60, 4)
    assert rep["sha256"] == _sha256(recs)
    assert sum(sum(row.values()) for row in rep["category_status"].values()) == 60
    for cat in CATEGORIES:
        amounts = [r["loan_amount_inr"] for r in recs if r["category"] == cat]
        assert rep["amount_inr"][cat]["min"] == min(amounts)
        assert rep["amount_inr"][cat]["max"] == max(amounts)
    days = [r["days_since_created"] for r in recs]
    assert rep["days_since_created"]["min"] == min(days)
    assert rep["days_since_created"]["max"] == max(days)
    assert rep["problems"] == []
    json.dumps(rep)  # JSON-serialisable


def test_report_lists_problems_of_an_invalid_dataset(records: Records) -> None:
    rep = dataset.report(records[:10])
    assert rep["problems"] == dataset.validate(records[:10]) != []


def test_main_prints_the_report_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert dataset.main() == 0
    assert json.loads(capsys.readouterr().out) == dataset.report(dataset.LOAN_APPLICATIONS)


def test_main_exits_non_zero_when_invalid(
    records: Records, capsys: pytest.CaptureFixture[str]
) -> None:
    assert dataset.main(records[:5]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["problems"]
    assert "INVALID" in captured.err


def test_running_the_module_prints_the_report() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dataset"], capture_output=True, text=True, check=False, cwd=REPO
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["sha256"] == _sha256(dataset.LOAN_APPLICATIONS)


# ---------------------------------------------------------------- isolation


def test_dataset_is_stdlib_only_and_imports_nothing_from_credagent() -> None:
    tree = ast.parse((REPO / "dataset.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module
            roots.add(node.module.split(".")[0])
    assert "credagent" not in roots
    assert roots <= sys.stdlib_module_names | {"__future__"}, roots - sys.stdlib_module_names


def test_importing_dataset_loads_no_credagent_module() -> None:
    code = (
        "import sys, dataset; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'credagent'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert proc.stdout.strip() == "[]"
