"""Line and branch coverage thresholds, checked separately (§ARCH 11.3, §PLAN 3 step [9]).

`python -m scripts.coverage_gate /tmp/coverage.json` after `coverage json`. pytest-cov's
`fail_under` checks only the combined figure, so the branch threshold is enforced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

LINE_MIN: Final = 90.0
BRANCH_MIN: Final = 85.0


def _pct(covered: int, total: int) -> float:
    return 100.0 * covered / total if total else 100.0


def percentages(totals: Mapping[str, int]) -> tuple[float, float]:
    return (
        _pct(totals["covered_lines"], totals["num_statements"]),
        _pct(totals["covered_branches"], totals["num_branches"]),
    )


def problems(totals: Mapping[str, int]) -> list[str]:
    line, branch = percentages(totals)
    out = []
    if line < LINE_MIN:
        out.append(f"line coverage {line:.2f}% < {LINE_MIN}%")
    if branch < BRANCH_MIN:
        out.append(f"branch coverage {branch:.2f}% < {BRANCH_MIN}%")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    totals = json.loads(args.report.read_text(encoding="utf-8"))["totals"]
    line, branch = percentages(totals)
    print(f"coverage: line {line:.2f}% · branch {branch:.2f}% (min {LINE_MIN}% / {BRANCH_MIN}%)")
    found = problems(totals)
    for p in found:
        print(p, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
