"""Chunking evaluation (§ARCH 7.6, Task 5): document-level precision/recall per collection.

For each collection, every P/R query retrieves its top-k chunks. The chunks are mapped to
`doc_id` and deduplicated with rank order kept. Then `P = |rel ∩ ret| / |ret|` and
`R = |rel ∩ ret| / |rel|`. An empty retrieval defines `P = 0` and logs a warning; it must not
happen for in-scope queries. P and R are exact `Fraction`s, so macro averages and ties are exact.

Eligibility (D-028): only strategies whose calibration meets `MIN_GAP` compete; the others are
still measured and reported. If none is eligible, nothing is deployed (`NoEligibleStrategyError`).
Choice among the eligible (D-027): the higher macro F1 (harmonic mean of macro P and macro R)
wins. On a tie, the higher mean top-1 similarity (rounded to 4 decimals, as stored in
`config/retrieval.json`) wins, then the fewer distinct documents retrieved per query, then the
gated strategy. `deploy` refuses any strategy below `MIN_GAP` (`NotDeployableError`).

The P/R queries are disjoint from the calibration queries (tests/unit/test_query_sets.py).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, Literal

from credagent.kb.chunking import STRATEGIES, Strategy
from credagent.rag.calibration import GATED_STRATEGY, CalibrationRetriever
from credagent.rag.store import RetrievedChunk

log = logging.getLogger(__name__)

_TOP1_DECIMALS: Final = 4  # tie-break precision, matching config/retrieval.json
_MIN_STRATEGIES: Final = 2


@dataclass(frozen=True)
class PRQuery:
    query: str
    relevant_doc_ids: tuple[str, ...]


# Written before measuring and never tuned on the results (D-027). Two queries need two documents.
PR_QUERIES: Final = (
    PRQuery("Can an NRI be a joint account holder?",
            ("kb-11-joint-accounts", "kb-12-nri-eligibility")),
    PRQuery("What prepayment penalty applies and what rate slab applies to a personal loan?",
            ("kb-07-interest-rate-slabs", "kb-08-prepayment-penalty")),
    PRQuery("What fee applies if I close my account in its first year?",
            ("kb-06-account-closure",)),
    PRQuery("What is charged when my savings balance falls below the required average?",
            ("kb-09-minimum-balance",)),
    PRQuery("What does an Under Review application status mean?",
            ("kb-13-application-status-glossary",)),
)  # fmt: skip


def _d(x: Fraction | float) -> str:
    return f"{float(x):.3f}"


def dedup_doc_ids(doc_ids: Iterable[str]) -> tuple[str, ...]:
    """Distinct doc ids in first-occurrence (rank) order."""
    return tuple(dict.fromkeys(doc_ids))


@dataclass(frozen=True)
class PRScore:
    relevant: tuple[str, ...]
    retrieved: tuple[str, ...]  # deduplicated, rank order

    @property
    def hits(self) -> int:
        return len(set(self.relevant) & set(self.retrieved))

    @property
    def precision(self) -> Fraction:
        return Fraction(self.hits, len(self.retrieved)) if self.retrieved else Fraction(0)

    @property
    def recall(self) -> Fraction:
        return Fraction(self.hits, len(self.relevant))

    @property
    def precision_arithmetic(self) -> str:
        if not self.retrieved:
            return "P = 0/0 := 0.000 (empty retrieval)"
        return f"P = {self.hits}/{len(self.retrieved)} = {_d(self.precision)}"

    @property
    def recall_arithmetic(self) -> str:
        return f"R = {self.hits}/{len(self.relevant)} = {_d(self.recall)}"


def score(relevant: Sequence[str], retrieved_doc_ids: Sequence[str]) -> PRScore:
    """P/R of one query; `retrieved_doc_ids` is per chunk, in rank order (duplicates allowed)."""
    if not relevant or len(set(relevant)) != len(relevant):
        raise ValueError(f"relevant doc ids must be non-empty and unique, got {list(relevant)}")
    s = PRScore(tuple(relevant), dedup_doc_ids(retrieved_doc_ids))
    if not s.retrieved:
        log.warning("empty retrieval: precision is undefined and set to 0")
    return s


@dataclass(frozen=True)
class QueryEvaluation:
    query: PRQuery
    chunks: tuple[RetrievedChunk, ...]
    score: PRScore
    top1_similarity: float


def _mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


@dataclass(frozen=True)
class CollectionEvaluation:
    strategy: Strategy
    collection: str
    queries: tuple[QueryEvaluation, ...]

    @property
    def macro_precision(self) -> Fraction:
        return _mean([q.score.precision for q in self.queries])

    @property
    def macro_recall(self) -> Fraction:
        return _mean([q.score.recall for q in self.queries])

    @property
    def macro_f1(self) -> Fraction:
        p, r = self.macro_precision, self.macro_recall
        return 2 * p * r / (p + r) if p + r else Fraction(0)

    @property
    def mean_top1(self) -> float:
        return sum(q.top1_similarity for q in self.queries) / len(self.queries)

    @property
    def mean_docs(self) -> Fraction:
        return _mean([Fraction(len(q.score.retrieved)) for q in self.queries])

    def _macro_arithmetic(self, name: str, values: Sequence[Fraction], mean: Fraction) -> str:
        terms = " + ".join(_d(v) for v in values)
        return f"macro {name} = ({terms}) / {len(values)} = {_d(mean)}"

    @property
    def macro_precision_arithmetic(self) -> str:
        values = [q.score.precision for q in self.queries]
        return self._macro_arithmetic("P", values, self.macro_precision)

    @property
    def macro_recall_arithmetic(self) -> str:
        values = [q.score.recall for q in self.queries]
        return self._macro_arithmetic("R", values, self.macro_recall)


def evaluate(
    retriever: CalibrationRetriever,
    *,
    top_k: int,
    queries: Sequence[PRQuery] = PR_QUERIES,
    strategies: Sequence[Strategy] = STRATEGIES,
) -> dict[Strategy, CollectionEvaluation]:
    if not queries:
        raise ValueError("at least one P/R query is required")
    out: dict[Strategy, CollectionEvaluation] = {}
    for strategy in strategies:
        rows = []
        for q in queries:
            result = retriever.retrieve(q.query, strategy, top_k)
            s = score(q.relevant_doc_ids, [c.doc_id for c in result.chunks])
            rows.append(QueryEvaluation(q, result.chunks, s, result.top1_similarity))
        out[strategy] = CollectionEvaluation(
            strategy, retriever.collection_name(strategy), tuple(rows)
        )
    return out


DecidedBy = Literal["macro_f1", "mean_top1", "mean_docs", "gated", "only_eligible"]


class NotDeployableError(ValueError):
    """The P/R winner's calibration gap is below MIN_GAP (a stop-and-ask condition)."""


class NoEligibleStrategyError(NotDeployableError):
    """No evaluated strategy meets the calibration gap: nothing may be deployed (D-028)."""


@dataclass(frozen=True)
class CalibrationSummary:
    threshold: float | None
    gap: float
    meets_min_gap: bool


def calibration_of(config: Mapping[str, Any], strategy: Strategy) -> CalibrationSummary:
    c = config["collections"][strategy]
    return CalibrationSummary(c["threshold"], float(c["gap"]), bool(c["meets_min_gap"]))


@dataclass(frozen=True)
class Choice:
    winner: Strategy
    runner_up: Strategy
    decided_by: DecidedBy
    excluded: tuple[Strategy, ...] = ()  # failed the calibration gap; never competed (D-028)


def choose(
    results: Mapping[Strategy, CollectionEvaluation],
    *,
    gated: Strategy = GATED_STRATEGY,
    calibrations: Mapping[Strategy, CalibrationSummary] | None = None,
) -> Choice:
    """Eligibility filter (D-028), then the unchanged P/R + tie-break chain (D-027).

    With `calibrations`, a strategy competes only if its calibration `meets_min_gap` (a missing
    calibration counts as not meeting it). Without them, every strategy competes.
    """
    if len(results) < _MIN_STRATEGIES:
        raise ValueError("choosing a strategy needs at least two evaluated collections")

    def key(c: CollectionEvaluation) -> tuple[Fraction, float, Fraction, bool]:
        return (-c.macro_f1, -round(c.mean_top1, _TOP1_DECIMALS), c.mean_docs, c.strategy != gated)

    def ranked(cs: Iterable[CollectionEvaluation]) -> list[CollectionEvaluation]:
        return sorted(cs, key=lambda c: (key(c), c.strategy))

    def eligible(s: Strategy) -> bool:
        if calibrations is None:
            return True
        c = calibrations.get(s)
        return c is not None and c.meets_min_gap

    excluded = tuple(sorted(s for s in results if not eligible(s)))
    pool = ranked(c for s, c in results.items() if eligible(s))
    if not pool:
        gaps = ", ".join(
            f"{s} gap {calibrations[s].gap:.3f}" if calibrations and s in calibrations else
            f"{s} uncalibrated"
            for s in excluded
        )  # fmt: skip
        raise NoEligibleStrategyError(f"no strategy meets the calibration gap ({gaps})")
    if len(pool) == 1:
        others = ranked(results[s] for s in excluded)
        return Choice(pool[0].strategy, others[0].strategy, "only_eligible", excluded)

    first, second = pool[:2]
    names: tuple[DecidedBy, ...] = ("macro_f1", "mean_top1", "mean_docs", "gated")
    a, b = key(first), key(second)
    fallback: DecidedBy = "gated"  # unreachable for distinct strategies; keeps `next` total
    decided_by = next((n for n, x, y in zip(names, a, b, strict=True) if x != y), fallback)
    return Choice(first.strategy, second.strategy, decided_by, excluded)


def _excluded_sentence(
    w: CollectionEvaluation, lo: CollectionEvaluation, c: CalibrationSummary | None, min_gap: float
) -> str:
    why = (
        f"its calibration gap of {_d(c.gap)} is below the {min_gap:.2f} minimum"
        if c is not None
        else "it has no calibration"
    )
    return (
        f"`{lo.strategy}` is excluded before the P/R comparison because {why}, so "
        f"`{w.strategy}` is the only eligible strategy."
    )


def recommendation(
    results: Mapping[Strategy, CollectionEvaluation],
    choice: Choice,
    calibrations: Mapping[Strategy, CalibrationSummary],
    *,
    min_gap: float,
    top_k: int,
) -> str:
    """2–3 sentences from a fixed template, filled only with the measured numbers."""
    w, lo = results[choice.winner], results[choice.runner_up]
    calibration = calibrations[choice.winner]
    first = (
        f"`{w.strategy}` chunking (`{w.collection}`) is recommended: across {len(w.queries)} "
        f"in-scope queries at top-{top_k} it reached macro precision {_d(w.macro_precision)} "
        f"and macro recall {_d(w.macro_recall)}, against {_d(lo.macro_precision)} and "
        f"{_d(lo.macro_recall)} for `{lo.strategy}` (`{lo.collection}`)."
    )
    second = {
        "macro_f1": f"Macro F1 decides it ({_d(w.macro_f1)} vs {_d(lo.macro_f1)}), with mean "
        f"top-1 similarity {_d(w.mean_top1)} vs {_d(lo.mean_top1)}.",
        "mean_top1": f"Macro F1 ties at {_d(w.macro_f1)}, so the first tie-breaker, mean top-1 "
        f"similarity, decides it ({_d(w.mean_top1)} vs {_d(lo.mean_top1)}).",
        "mean_docs": f"Macro F1 ({_d(w.macro_f1)}) and mean top-1 similarity "
        f"({_d(w.mean_top1)}) tie, so the second tie-breaker decides it: {_d(w.mean_docs)} vs "
        f"{_d(lo.mean_docs)} distinct documents retrieved per query.",
        "gated": f"Macro F1 ({_d(w.macro_f1)}), mean top-1 similarity ({_d(w.mean_top1)}) and "
        f"distinct documents per query ({_d(w.mean_docs)}) all tie, so the gated strategy is kept.",
        "only_eligible": _excluded_sentence(w, lo, calibrations.get(lo.strategy), min_gap),
    }[choice.decided_by]
    if calibration.meets_min_gap and calibration.threshold is not None:
        third = (
            f"Deploy it: its calibration gap of {_d(calibration.gap)} meets the {min_gap:.2f} "
            f"minimum, so its threshold {_d(calibration.threshold)} applies at runtime."
        )
    else:
        third = (
            f"Its calibration gap of {_d(calibration.gap)} is below the {min_gap:.2f} minimum, "
            "so it is not deployed until a recalibration closes the gap."
        )
    return f"{first} {second} {third}"


PR_HEADERS: Final = ("#", "query", "relevant", "retrieved (dedup, rank order)", "precision",
                     "recall")  # fmt: skip
AVERAGE_HEADERS: Final = ("strategy", "collection", "macro P", "macro R", "macro F1",
                          "mean top-1 sim", "mean docs retrieved")  # fmt: skip


def pr_rows(c: CollectionEvaluation) -> list[list[str]]:
    return [
        [str(i), qe.query.query, ", ".join(qe.query.relevant_doc_ids),
         ", ".join(qe.score.retrieved) or "—", qe.score.precision_arithmetic,
         qe.score.recall_arithmetic]
        for i, qe in enumerate(c.queries, start=1)
    ]  # fmt: skip


def average_rows(results: Mapping[Strategy, CollectionEvaluation]) -> list[list[str]]:
    return [
        [f"`{c.strategy}`", f"`{c.collection}`", _d(c.macro_precision), _d(c.macro_recall),
         _d(c.macro_f1), _d(c.mean_top1), _d(c.mean_docs)]
        for c in (results[s] for s in sorted(results))
    ]  # fmt: skip


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def render_report(
    results: Mapping[Strategy, CollectionEvaluation],
    choice: Choice,
    recommendation_text: str,
    *,
    top_k: int,
) -> str:
    """The deterministic P/R report (no timestamps), golden-diffed (§ARCH 11.8)."""
    lines = [
        "# Chunking evaluation — precision / recall (Task 5)",
        "",
        f"_Generated by `scripts/eval_chunking.py`. Do not edit by hand._ Top-{top_k} chunks per "
        "query → `doc_id` → deduplicated in rank order → `P = |rel ∩ ret| / |ret|`, "
        "`R = |rel ∩ ret| / |rel|` (`P := 0` when nothing is retrieved).",
        "",
    ]
    for s in sorted(results):
        c = results[s]
        lines += [f"## `{c.collection}` ({c.strategy})", ""]
        lines += _table(PR_HEADERS, pr_rows(c))
        lines += [f"- `{c.macro_precision_arithmetic}`", f"- `{c.macro_recall_arithmetic}`", ""]
    lines += ["## Macro averages", ""]
    lines += _table(AVERAGE_HEADERS, average_rows(results))
    if choice.excluded:
        lines += [
            "Excluded before the comparison (calibration gap below `min_gap`, D-028): "
            + ", ".join(f"`{results[s].collection}` ({s})" for s in choice.excluded)
            + ". Their P/R is reported above but they do not compete.",
            "",
        ]
    lines += [
        f"Winner: `{choice.winner}` (decided by `{choice.decided_by}`).",
        "",
        "## Recommendation",
        "",
        recommendation_text,
    ]
    return "\n".join(lines) + "\n"


def deploy(config: Mapping[str, Any], strategy: Strategy) -> dict[str, Any]:
    """A copy of `config/retrieval.json` with only `deployed_strategy` set."""
    if not calibration_of(config, strategy).meets_min_gap:
        raise NotDeployableError(
            f"{strategy!r} has calibration gap below min_gap: not deploying it (§ARCH 7.5)"
        )
    out = copy.deepcopy(dict(config))
    out["deployed_strategy"] = strategy
    return out
