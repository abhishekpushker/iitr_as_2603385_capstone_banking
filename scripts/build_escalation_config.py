"""Derive the escalation threshold from the dataset (§ARCH 7.7, Task 6).
`python -m scripts.build_escalation_config`.

P80 is the nearest-rank 80th percentile of `days_since_created`; `T = round(0.6 · P80 / 30, 4)`.
Writes `config/escalation.json`. Exits 1 without writing if the dataset is invalid.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import dataset
from credagent.settings import get_settings
from credagent.tools import escalation as esc


def main(argv: Sequence[str] | None = None) -> int:
    try:
        s = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=s.escalation_config_path)
    args = parser.parse_args(argv)

    records = dataset.LOAN_APPLICATIONS
    problems = dataset.validate(records)
    if problems:
        print(f"invalid dataset, not writing {args.out}: {problems}", file=sys.stderr)
        return 1
    config = esc.build_config(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(esc.dumps_config(config), encoding="utf-8")
    print(
        f"n = {config.n_records}; P80 = {config.p80_days_since_created} days; "
        f"T = {config.threshold}; unflagged escalates from day {config.unflagged_min_days}, "
        f"flagged from day {config.flagged_min_days}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
