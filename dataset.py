"""Loan-application dataset (Task 1, §ARCH 7.1). Stdlib only; imports nothing from credagent.

Every record is drawn from a local `random.Random(seed)`, in a fixed order per record:
category, status, amount, days, fraud flag. Amounts are log-uniform within a per-category range
and rounded to the nearest ₹5,000. Ranges follow typical Indian retail-lending ticket sizes per
product, and log-uniform sampling reflects that small tickets are far more common than jumbo
ones. Upper bounds stay at 8 digits, so no amount can match the ≥ 9-digit bank-account masker.

Seed procedure: start at 2026 and increment until `validate` passes; never edit records
(the attempts are recorded in docs/DECISIONS.md and regenerated in transcripts/01).

`python -m dataset` prints the report as JSON and exits non-zero if the dataset is invalid.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

SEED = 2026
N_RECORDS = 60

CATEGORY_WEIGHTS: dict[str, float] = {
    "Personal Loan": 0.30,
    "Home Loan": 0.20,
    "Auto Loan": 0.20,
    "Education Loan": 0.15,
    "Business Loan": 0.15,
}
STATUS_WEIGHTS: dict[str, float] = {
    "Submitted": 0.20,
    "Under Review": 0.30,
    "Approved": 0.20,
    "Rejected": 0.15,
    "Disbursed": 0.15,
}
# Inclusive (min, max) in rupees; both ends are multiples of AMOUNT_ROUNDING_INR.
AMOUNT_RANGES_INR: dict[str, tuple[int, int]] = {
    "Personal Loan": (50_000, 4_000_000),  # ₹50,000 – ₹40,00,000
    "Home Loan": (1_000_000, 20_000_000),  # ₹10,00,000 – ₹2,00,00,000
    "Auto Loan": (200_000, 3_000_000),  # ₹2,00,000 – ₹30,00,000
    "Education Loan": (100_000, 7_500_000),  # ₹1,00,000 – ₹75,00,000
    "Business Loan": (500_000, 10_000_000),  # ₹5,00,000 – ₹1,00,00,000
}
AMOUNT_ROUNDING_INR = 5_000
FRAUD_FLAG_PROBABILITY = 0.18
MAX_DAYS_SINCE_CREATED = 30

# Validation rules (§ARCH 7.1).
MIN_RECORDS = 40
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
FRAUD_SHARE_BAND = (0.10, 0.30)
RECORD_ID_PATTERN = re.compile(r"^LA-\d{4}$")


@dataclass(frozen=True)
class LoanApplication:
    """One record. The field order here is the field order of every generated dict."""

    record_id: str
    category: str
    status: str
    loan_amount_inr: int
    days_since_created: int
    flagged_for_fraud_review: bool


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LoanApplication))
_FIELD_TYPES: dict[str, type] = {
    "record_id": str,
    "category": str,
    "status": str,
    "loan_amount_inr": int,
    "days_since_created": int,
    "flagged_for_fraud_review": bool,
}


def _log_uniform_amount(rng: random.Random, lo: int, hi: int) -> int:
    raw = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    rounded = round(raw / AMOUNT_ROUNDING_INR) * AMOUNT_ROUNDING_INR
    return min(max(rounded, lo), hi)


def generate(seed: int) -> list[dict[str, Any]]:
    """Return `N_RECORDS` fresh records for `seed`. Never touches the module-global `random`."""
    rng = random.Random(seed)  # noqa: S311 (reproducible synthetic data, not cryptography)
    categories = list(CATEGORY_WEIGHTS)
    statuses = list(STATUS_WEIGHTS)
    records: list[dict[str, Any]] = []
    for i in range(1, N_RECORDS + 1):
        category = rng.choices(categories, weights=list(CATEGORY_WEIGHTS.values()))[0]
        status = rng.choices(statuses, weights=list(STATUS_WEIGHTS.values()))[0]
        amount = _log_uniform_amount(rng, *AMOUNT_RANGES_INR[category])
        days = rng.randint(0, MAX_DAYS_SINCE_CREATED)
        flagged = rng.random() < FRAUD_FLAG_PROBABILITY
        record = LoanApplication(f"LA-{i:04d}", category, status, amount, days, flagged)
        records.append(asdict(record))
    return records


def sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """The golden fingerprint: sha256 of `json.dumps(records, sort_keys=True)`."""
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


def _record_problems(index: int, record: Mapping[str, Any]) -> list[str]:
    where = f"record {index} ({record.get('record_id')!r})"
    if tuple(record) != FIELDS:
        return [f"{where}: fields {list(record)} != {list(FIELDS)}"]
    problems = [
        f"{where}: {name} has type {type(record[name]).__name__}, expected {kind.__name__}"
        for name, kind in _FIELD_TYPES.items()
        if type(record[name]) is not kind
    ]
    if problems:
        return problems
    if not RECORD_ID_PATTERN.fullmatch(record["record_id"]):
        problems.append(f"{where}: record_id does not match {RECORD_ID_PATTERN.pattern}")
    if record["category"] not in CATEGORY_WEIGHTS:
        problems.append(f"{where}: unknown category {record['category']!r}")
    else:
        lo, hi = AMOUNT_RANGES_INR[record["category"]]
        if not lo <= record["loan_amount_inr"] <= hi:
            problems.append(
                f"{where}: loan_amount_inr {record['loan_amount_inr']} outside "
                f"[{lo}, {hi}] for {record['category']}"
            )
    if record["status"] not in STATUS_WEIGHTS:
        problems.append(f"{where}: unknown status {record['status']!r}")
    if not 0 <= record["days_since_created"] <= MAX_DAYS_SINCE_CREATED:
        problems.append(
            f"{where}: days_since_created {record['days_since_created']} outside "
            f"[0, {MAX_DAYS_SINCE_CREATED}]"
        )
    return problems


def validate(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return every rule the dataset breaks (§ARCH 7.1); an empty list means valid."""
    problems: list[str] = []
    if len(records) < MIN_RECORDS:
        problems.append(f"{len(records)} records < {MIN_RECORDS}")
    for index, record in enumerate(records):
        problems += _record_problems(index, record)

    ids = Counter(r.get("record_id") for r in records)
    problems += [f"duplicate record_id {rid!r} ({n}x)" for rid, n in ids.items() if n > 1]

    categories = Counter(r.get("category") for r in records)
    problems += [
        f"category {c!r}: {categories[c]} records < {MIN_PER_CATEGORY}"
        for c in CATEGORY_WEIGHTS
        if categories[c] < MIN_PER_CATEGORY
    ]
    statuses = Counter(r.get("status") for r in records)
    problems += [
        f"status {s!r}: {statuses[s]} records < {MIN_PER_STATUS}"
        for s in STATUS_WEIGHTS
        if statuses[s] < MIN_PER_STATUS
    ]

    if records:
        flagged = sum(r.get("flagged_for_fraud_review") is True for r in records)
        share = flagged / len(records)
        lo, hi = FRAUD_SHARE_BAND
        if not lo <= share <= hi:
            problems.append(
                f"fraud share {share:.4f} ({flagged}/{len(records)}) outside [{lo}, {hi}]"
            )
    return problems


def report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A JSON-serialisable summary of `records`, including `validate`'s problems."""
    n = len(records)
    flagged = sum(r["flagged_for_fraud_review"] is True for r in records)
    amounts: dict[str, dict[str, int]] = {}
    for cat in CATEGORY_WEIGHTS:
        values = sorted(r["loan_amount_inr"] for r in records if r["category"] == cat)
        if values:
            amounts[cat] = {
                "min": values[0],
                "median_low": values[(len(values) - 1) // 2],
                "max": values[-1],
            }
    days = [r["days_since_created"] for r in records]
    return {
        "n_records": n,
        "category_counts": {c: sum(r["category"] == c for r in records) for c in CATEGORY_WEIGHTS},
        "status_counts": {s: sum(r["status"] == s for r in records) for s in STATUS_WEIGHTS},
        "category_status": {
            c: {
                s: sum(r["category"] == c and r["status"] == s for r in records)
                for s in STATUS_WEIGHTS
            }
            for c in CATEGORY_WEIGHTS
        },
        "fraud_flagged": flagged,
        "fraud_share": round(flagged / n, 4) if n else 0.0,
        "amount_inr": amounts,
        "days_since_created": {
            "min": min(days, default=0),
            "max": max(days, default=0),
            "mean": round(sum(days) / n, 2) if n else 0.0,
        },
        "sha256": sha256(records),
        "problems": validate(records),
    }


LOAN_APPLICATIONS: list[dict[str, Any]] = generate(SEED)


def main(records: Sequence[Mapping[str, Any]] | None = None) -> int:
    rep = report(LOAN_APPLICATIONS if records is None else records)
    print(json.dumps(rep, indent=2))
    if rep["problems"]:
        print(f"INVALID dataset: {len(rep['problems'])} problem(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
