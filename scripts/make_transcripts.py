"""Regenerate the graded transcripts (§ARCH 7.19). Run with `python -m scripts.make_transcripts`.

Each generator fills a `Transcript` with inputs, outputs and ✔/✘ checks from real assertions.
Every transcript is written even when a check fails; the exit code is then 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import get_args

import dataset
from credagent.schemas import LoanCategory, LoanStatus
from credagent.settings import get_settings
from scripts import validate_dataset
from scripts.transcript_kit import Transcript

COMMAND = "make transcripts  (python -m scripts.make_transcripts)"

Generator = Callable[[Transcript], None]


def dataset_report(t: Transcript) -> None:
    records = dataset.LOAN_APPLICATIONS
    rep = dataset.report(records)
    n = len(records)

    t.section("Inputs")
    t.para(
        f"`SEED = {dataset.SEED}` · `N_RECORDS = {dataset.N_RECORDS}` · "
        f"`FRAUD_FLAG_PROBABILITY = {dataset.FRAUD_FLAG_PROBABILITY}` (Bernoulli per record). "
        "Each record draws, from a local `random.Random(SEED)` and in this order: category, "
        "status, amount (log-uniform within the category range, rounded to the nearest "
        f"₹{dataset.AMOUNT_ROUNDING_INR:,}), `days_since_created = randint(0, "
        f"{dataset.MAX_DAYS_SINCE_CREATED})`, fraud flag."
    )
    t.table(
        ["Category", "Weight", "Amount range"],
        [
            [c, w, f"{validate_dataset.inr(lo)} – {validate_dataset.inr(hi)}"]
            for (c, w), (lo, hi) in zip(
                dataset.CATEGORY_WEIGHTS.items(), dataset.AMOUNT_RANGES_INR.values(), strict=True
            )
        ],
    )
    t.table(["Status", "Weight"], list(dataset.STATUS_WEIGHTS.items()))

    t.section("Seed history")
    history = validate_dataset.seed_history(validate_dataset.SEED_PROCEDURE_START, dataset.SEED)
    t.para(
        f"Procedure (§ARCH 7.1): start at {validate_dataset.SEED_PROCEDURE_START}, increment "
        "until `validate` returns no problems. Records are never edited. Recomputed here:"
    )
    t.table(
        ["Seed", "Result"],
        [[s, "valid" if not p else "invalid: " + "; ".join(p)] for s, p in history],
    )

    t.raw(validate_dataset.render_markdown(rep))

    t.section("Records")
    t.table(
        list(dataset.FIELDS),
        [
            [r["record_id"], r["category"], r["status"], r["loan_amount_inr"],
             r["days_since_created"], r["flagged_for_fraud_review"]]
            for r in records
        ],
    )  # fmt: skip

    categories = Counter(r["category"] for r in records)
    statuses = Counter(r["status"] for r in records)
    share = sum(r["flagged_for_fraud_review"] for r in records) / n
    t.check("`validate(LOAN_APPLICATIONS)` returns no problems", dataset.validate(records) == [])
    t.check(f"record count {n} ≥ 40", n >= 40)
    t.check(
        f"every category has ≥ 3 records (min {min(categories.values())})",
        set(categories) == set(dataset.CATEGORY_WEIGHTS) and min(categories.values()) >= 3,
    )
    t.check(
        f"every status has ≥ 1 record ({len(statuses)}/{len(dataset.STATUS_WEIGHTS)} present)",
        set(statuses) == set(dataset.STATUS_WEIGHTS),
    )
    t.check(f"fraud share {share:.2%} within [10%, 30%]", 0.10 <= share <= 0.30)
    ids = [r["record_id"] for r in records]
    t.check(
        "record ids unique and match `^LA-\\d{4}$`",
        len(set(ids)) == n and all(re.fullmatch(r"LA-\d{4}", i) for i in ids),
    )
    t.check(
        "field set and order fixed in every record",
        all(tuple(r) == dataset.FIELDS for r in records),
    )
    t.check(
        "every amount within its category range and ≤ 8 digits",
        all(
            dataset.AMOUNT_RANGES_INR[r["category"]][0]
            <= r["loan_amount_inr"]
            <= dataset.AMOUNT_RANGES_INR[r["category"]][1]
            and len(str(r["loan_amount_inr"])) <= 8
            for r in records
        ),
    )
    t.check(
        "every `days_since_created` is an int in [0, 30]",
        all(
            type(r["days_since_created"]) is int and 0 <= r["days_since_created"] <= 30
            for r in records
        ),
    )
    t.check(
        "`generate(SEED)` twice is identical and equals `LOAN_APPLICATIONS`",
        dataset.generate(dataset.SEED) == dataset.generate(dataset.SEED) == records,
    )
    t.check("`generate(SEED + 1)` differs", dataset.generate(dataset.SEED + 1) != records)
    t.check(
        f"SEED {dataset.SEED} is the first valid seed from {validate_dataset.SEED_PROCEDURE_START}",
        history[-1] == (dataset.SEED, []) and all(p for _, p in history[:-1]),
    )
    t.check(
        "schema literals `LoanCategory` / `LoanStatus` equal the dataset vocabularies",
        get_args(LoanCategory) == tuple(dataset.CATEGORY_WEIGHTS)
        and get_args(LoanStatus) == tuple(dataset.STATUS_WEIGHTS),
    )


GENERATORS: dict[str, tuple[str, Generator]] = {
    "01_dataset_report": ("Dataset report (Task 1)", dataset_report),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("transcripts"))
    parser.add_argument("--only", action="append", choices=sorted(GENERATORS), default=None)
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc

    args.out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in args.only or sorted(GENERATORS):
        title, generate = GENERATORS[name]
        transcript = Transcript(name, title, command=COMMAND, settings=settings)
        generate(transcript)
        (args.out / f"{name}.md").write_text(transcript.render(), encoding="utf-8")
        failed += [f"{name}: ✘ {label}" for label in transcript.failed]
        print(f"{name}: {'FAILED' if transcript.failed else 'ok'}")
    for line in failed:
        print(line, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
