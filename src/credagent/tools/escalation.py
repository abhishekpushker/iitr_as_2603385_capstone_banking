"""Escalation score and threshold (§ARCH 7.7, Task 6).

    recency = min(max(days_since_created, 0), 30) / 30
    score   = round(0.4 · flagged + 0.6 · recency, 4)          ∈ [0, 1]
    T       = round(0.6 · P80(days_since_created) / 30, 4)
    escalate iff score ≥ T

P80 is the nearest-rank percentile `sorted(days)[ceil(0.8·n) − 1]`, computed in integer
arithmetic so no float `0.8·n` can round across an integer. The weights, P80 and T are written to
`config/escalation.json` by `scripts/build_escalation_config.py`; the lookup tool reads them back.
Known limitation (documented): the score ignores status, so a flagged, disbursed loan still
scores on age alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

import dataset

FLAG_WEIGHT: Final = 0.4
RECENCY_WEIGHT: Final = 0.6
RECENCY_CAP_DAYS: Final = 30
PERCENTILE: Final = 80
SCORE_DECIMALS: Final = 4
_PERCENT: Final = 100


def escalation_score(
    days: int,
    flagged: bool,
    *,
    flag_weight: float = FLAG_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
    cap_days: int = RECENCY_CAP_DAYS,
) -> float:
    recency = min(max(days, 0), cap_days) / cap_days
    return round(flag_weight * int(flagged) + recency_weight * recency, SCORE_DECIMALS)


def percentile_nearest_rank(values: Sequence[int], p: int = PERCENTILE) -> int:
    """`sorted(values)[ceil(p/100 · n) − 1]`, with the ceiling taken in integers."""
    if not 1 <= p <= _PERCENT:
        raise ValueError(f"percentile must be in [1, 100], got {p}")
    if not values:
        raise ValueError("percentile of an empty list")
    rank = -(-p * len(values) // _PERCENT)  # ceil(p·n / 100)
    return sorted(values)[rank - 1]


def threshold_from_p80(
    p80_days: int, *, recency_weight: float = RECENCY_WEIGHT, cap_days: int = RECENCY_CAP_DAYS
) -> float:
    return round(recency_weight * p80_days / cap_days, SCORE_DECIMALS)


def should_escalate(score: float, threshold: float) -> bool:
    return score >= threshold


def min_days_to_escalate(
    flagged: bool,
    threshold: float,
    *,
    flag_weight: float = FLAG_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
    cap_days: int = RECENCY_CAP_DAYS,
) -> int | None:
    """The youngest age that escalates, found by evaluating the score itself (so rounding is
    exactly what the tool applies); `None` if no age within the cap reaches `threshold`."""
    for days in range(cap_days + 1):
        score = escalation_score(
            days, flagged, flag_weight=flag_weight, recency_weight=recency_weight,
            cap_days=cap_days,
        )  # fmt: skip
        if should_escalate(score, threshold):
            return days
    return None


class EscalationConfig(BaseModel):
    """`config/escalation.json`. Validation re-derives T and the cut-offs, so a hand-edited file
    with an inconsistent threshold is rejected on load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_records: int = Field(ge=1)
    flag_weight: float = Field(ge=0, le=1)
    recency_weight: float = Field(ge=0, le=1)
    recency_cap_days: int = Field(ge=1)
    percentile: int = Field(ge=1, le=100)
    percentile_method: Literal["nearest-rank: sorted(days)[ceil(p/100 * n) - 1]"]
    p80_days_since_created: int = Field(ge=0)
    threshold: float = Field(ge=0, le=1)
    unflagged_min_days: int | None
    flagged_min_days: int | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.flag_weight + self.recency_weight != 1.0:
            raise ValueError("flag_weight + recency_weight must be 1 (score ∈ [0, 1])")
        expected = threshold_from_p80(
            self.p80_days_since_created, recency_weight=self.recency_weight,
            cap_days=self.recency_cap_days,
        )  # fmt: skip
        if self.threshold != expected:
            raise ValueError(f"threshold {self.threshold} != round(w·P80/cap, 4) = {expected}")
        cutoffs = (self.min_days(flagged=False), self.min_days(flagged=True))
        if (self.unflagged_min_days, self.flagged_min_days) != cutoffs:
            raise ValueError(f"min-days cut-offs must be {cutoffs}")
        return self

    def score(self, days: int, flagged: bool) -> float:
        return escalation_score(
            days, flagged, flag_weight=self.flag_weight, recency_weight=self.recency_weight,
            cap_days=self.recency_cap_days,
        )  # fmt: skip

    def min_days(self, *, flagged: bool) -> int | None:
        return min_days_to_escalate(
            flagged, self.threshold, flag_weight=self.flag_weight,
            recency_weight=self.recency_weight, cap_days=self.recency_cap_days,
        )  # fmt: skip


def build_config(records: Sequence[Mapping[str, Any]]) -> EscalationConfig:
    if not records:
        raise ValueError("cannot derive P80 from an empty dataset")
    p80 = percentile_nearest_rank([int(r["days_since_created"]) for r in records], PERCENTILE)
    threshold = threshold_from_p80(p80)
    return EscalationConfig(
        dataset_sha256=dataset.sha256(records),
        n_records=len(records),
        flag_weight=FLAG_WEIGHT,
        recency_weight=RECENCY_WEIGHT,
        recency_cap_days=RECENCY_CAP_DAYS,
        percentile=PERCENTILE,
        percentile_method="nearest-rank: sorted(days)[ceil(p/100 * n) - 1]",
        p80_days_since_created=p80,
        threshold=threshold,
        unflagged_min_days=min_days_to_escalate(False, threshold),
        flagged_min_days=min_days_to_escalate(True, threshold),
    )


def dumps_config(config: EscalationConfig) -> str:
    return json.dumps(config.model_dump(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_escalation_config(path: Path) -> EscalationConfig:
    return EscalationConfig.model_validate_json(path.read_text(encoding="utf-8"))
