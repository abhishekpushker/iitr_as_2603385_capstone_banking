"""Print the dataset report as Markdown; exit non-zero if the dataset is invalid (§PLAN Phase 1).

Run with `python -m scripts.validate_dataset`. Also hosts the seed procedure (§ARCH 7.1):
start at `SEED_PROCEDURE_START` and increment until `dataset.validate` passes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import dataset
from credagent.settings import get_settings

SEED_PROCEDURE_START = 2026


def seed_history(start: int, seed: int) -> list[tuple[int, list[str]]]:
    """Every seed tried from `start` up to and including `seed`, with its validation problems."""
    if seed < start:
        raise ValueError(f"seed {seed} is before the procedure start {start}")
    return [(s, dataset.validate(dataset.generate(s))) for s in range(start, seed + 1)]


def inr(amount: int) -> str:
    """Indian digit grouping: 12345678 → ₹1,23,45,678."""
    digits = str(amount)
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:  # noqa: PLR2004 (lakh/crore groups are two digits)
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join([*groups, tail])


def render_markdown(rep: Mapping[str, Any]) -> str:
    n = rep["n_records"]
    lines = ["## Dataset report", ""]
    lines += [f"- **Records:** {n}", f"- **sha256:** `{rep['sha256']}`"]
    lines.append(
        f"- **Flagged for fraud review:** {rep['fraud_flagged']} / {n} = {rep['fraud_share']:.2%}"
    )
    days = rep["days_since_created"]
    lines.append(
        f"- **days_since_created:** min {days['min']}, max {days['max']}, mean {days['mean']}"
    )
    lines += ["", "### By category", "", "| Category | Records | Min ₹ | Median ₹ | Max ₹ |"]
    lines.append("|---|---|---|---|---|")
    for cat, count in rep["category_counts"].items():
        a = rep["amount_inr"].get(cat)
        cells = [inr(a["min"]), inr(a["median_low"]), inr(a["max"])] if a else ["–"] * 3
        lines.append(f"| {cat} | {count} | " + " | ".join(cells) + " |")
    lines += ["", "### By status", "", "| Status | Records |", "|---|---|"]
    lines += [f"| {status} | {count} |" for status, count in rep["status_counts"].items()]
    statuses = list(rep["status_counts"])
    lines += ["", "### Category × status", "", "| Category | " + " | ".join(statuses) + " |"]
    lines.append("|---|" + "---|" * len(statuses))
    for cat, row in rep["category_status"].items():
        lines.append(f"| {cat} | " + " | ".join(str(row[s]) for s in statuses) + " |")
    valid = not rep["problems"]
    lines += ["", f"**Valid:** {'yes' if valid else 'no'}"]
    lines += [f"- {problem}" for problem in rep["problems"]]
    return "\n".join(lines) + "\n"


def main(records: Sequence[Mapping[str, Any]] | None = None) -> int:
    try:
        get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    rep = dataset.report(dataset.LOAN_APPLICATIONS if records is None else records)
    print(render_markdown(rep), end="")
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
