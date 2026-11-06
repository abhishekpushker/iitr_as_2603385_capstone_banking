"""Similarity-threshold calibration (§ARCH 7.5, Task 4) and `config/retrieval.json`.

For each collection, the top-1 cosine similarity of every in-scope and out-of-scope calibration
query is measured through the production `Retriever`. Then `gap = min_in − max_oos` and
`threshold = round((max_oos + min_in) / 2, 3)`. A collection whose ranges overlap (gap ≤ 0) gets
no threshold: one is never picked inside an overlap. The gate (`gap ≥ MIN_GAP`) applies to
`GATED_STRATEGY`, the collection expected to be deployed. Phase 4 confirms the deployed
strategy from the P/R results and records it as `deployed_strategy`.

The calibration queries are disjoint from the demo, held-out, P/R and eval queries
(tests/unit/test_query_sets.py), so the threshold is not tuned on the test set.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from credagent.kb.chunking import STRATEGIES, Strategy
from credagent.rag.retriever import RetrievalResult

MIN_GAP: Final = 0.05
GATED_STRATEGY: Final[Strategy] = "sentence"
SCHEMA_VERSION: Final = 1
_SIM_DECIMALS: Final = 4


@dataclass(frozen=True)
class CalibrationQuery:
    query: str
    expected_doc_id: str | None


# Questions a customer could plausibly ask, one per topic area; none is reused elsewhere.
IN_SCOPE_QUERIES: Final = (
    CalibrationQuery("What credit score do I need for an auto loan?", "kb-01-loan-eligibility"),
    CalibrationQuery("How is the EMI on my loan calculated?", "kb-02-emi-calculation"),
    CalibrationQuery(
        "What is the forex markup on credit card purchases abroad?", "kb-03-credit-card-fees"
    ),
    CalibrationQuery("Which documents are required for KYC?", "kb-04-kyc-documents"),
    CalibrationQuery(
        "How long until a disputed transaction is finally resolved?", "kb-05-fraud-dispute"
    ),
    CalibrationQuery(
        "What minimum average monthly balance must a savings account keep?",
        "kb-09-minimum-balance",
    ),
    CalibrationQuery(
        "How long does a hard enquiry stay on my credit report?", "kb-10-credit-score-factors"
    ),
    CalibrationQuery("How many holders can a joint account have?", "kb-11-joint-accounts"),
    CalibrationQuery("Which accounts can NRIs open?", "kb-12-nri-eligibility"),
    CalibrationQuery(
        "How long does approval take after my application is submitted?",
        "kb-13-application-status-glossary",
    ),
)

# §ARCH 7.5's five examples plus three more. Near-domain traps (finance words, no KB answer):
# share price, mutual fund, exchange rate, fixed-deposit interest, a poem about loans.
OUT_OF_SCOPE_QUERIES: Final = (
    "What is Cred's share price today?",
    "Recommend a good mutual fund",
    "What's the weather in Mumbai?",
    "Write a poem about loans",
    "How do I reset my Netflix password?",
    "What is today's USD to INR exchange rate?",
    "What is the interest rate on a fixed deposit?",
    "Who won the cricket match yesterday?",
)


class CalibrationRetriever(Protocol):
    def collection_name(self, strategy: Strategy) -> str: ...

    def retrieve(self, query: str, strategy: Strategy, k: int | None = None) -> RetrievalResult: ...


@dataclass(frozen=True)
class Measurement:
    query: str
    top1_similarity: float
    top1_doc_id: str | None
    expected_doc_id: str | None

    def to_json(self, *, in_scope: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": self.query,
            "top1_similarity": round(self.top1_similarity, _SIM_DECIMALS),
            "top1_doc_id": self.top1_doc_id,
        }
        if in_scope:
            out["expected_doc_id"] = self.expected_doc_id
        return out


@dataclass(frozen=True)
class CollectionCalibration:
    strategy: Strategy
    collection: str
    in_scope: tuple[Measurement, ...]
    out_of_scope: tuple[Measurement, ...]

    @property
    def max_oos(self) -> float:
        return max(m.top1_similarity for m in self.out_of_scope)

    @property
    def min_in(self) -> float:
        return min(m.top1_similarity for m in self.in_scope)

    @property
    def gap(self) -> float:
        return self.min_in - self.max_oos

    @property
    def threshold(self) -> float | None:
        """Midpoint of the separating interval; None when the ranges overlap."""
        return round((self.max_oos + self.min_in) / 2, 3) if self.gap > 0 else None

    @property
    def meets_min_gap(self) -> bool:
        return self.gap >= MIN_GAP

    @property
    def in_scope_top1_correct(self) -> int:
        return sum(m.top1_doc_id == m.expected_doc_id for m in self.in_scope)

    def to_json(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "in_scope": [m.to_json(in_scope=True) for m in self.in_scope],
            "out_of_scope": [m.to_json(in_scope=False) for m in self.out_of_scope],
            "max_oos": round(self.max_oos, _SIM_DECIMALS),
            "min_in": round(self.min_in, _SIM_DECIMALS),
            "gap": round(self.gap, _SIM_DECIMALS),
            "threshold": self.threshold,
            "meets_min_gap": self.meets_min_gap,
            "in_scope_top1_correct": self.in_scope_top1_correct,
        }


def _measure(
    retriever: CalibrationRetriever,
    strategy: Strategy,
    query: str,
    expected: str | None,
    top_k: int,
) -> Measurement:
    result = retriever.retrieve(query, strategy, top_k)
    top_doc = result.chunks[0].doc_id if result.chunks else None
    return Measurement(query, result.top1_similarity, top_doc, expected)


def calibrate(
    retriever: CalibrationRetriever,
    *,
    top_k: int,
    in_scope: Sequence[CalibrationQuery] = IN_SCOPE_QUERIES,
    out_of_scope: Sequence[str] = OUT_OF_SCOPE_QUERIES,
    strategies: Sequence[Strategy] = STRATEGIES,
) -> dict[Strategy, CollectionCalibration]:
    return {
        strategy: CollectionCalibration(
            strategy=strategy,
            collection=retriever.collection_name(strategy),
            in_scope=tuple(
                _measure(retriever, strategy, q.query, q.expected_doc_id, top_k) for q in in_scope
            ),
            out_of_scope=tuple(_measure(retriever, strategy, q, None, top_k) for q in out_of_scope),
        )
        for strategy in strategies
    }


def build_config(
    results: Mapping[Strategy, CollectionCalibration],
    *,
    kb_version: str,
    top_k: int,
    chunk_params: Mapping[str, Any],
    embed_model: str,
    gated: Strategy = GATED_STRATEGY,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kb_version": kb_version,
        "embed_model": embed_model,
        "top_k": top_k,
        "chunk_params": dict(chunk_params),
        "min_gap": MIN_GAP,
        "gated_strategy": gated,
        "deployed_strategy": None,  # set in Phase 4 from the P/R results
        "collections": {s: results[s].to_json() for s in sorted(results)},
    }


def dumps_config(config: Mapping[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class RetrievalConfig:
    kb_version: str
    top_k: int
    gated_strategy: Strategy
    deployed_strategy: Strategy | None
    thresholds: Mapping[Strategy, float | None]

    def threshold_for(self, strategy: Strategy) -> float:
        t = self.thresholds.get(strategy)
        if t is None:
            raise ValueError(f"no threshold for {strategy!r}: its calibration ranges overlap")
        return t


def load_retrieval_config(path: Path) -> RetrievalConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    collections = data["collections"]
    return RetrievalConfig(
        kb_version=data["kb_version"],
        top_k=int(data["top_k"]),
        gated_strategy=cast(Strategy, data["gated_strategy"]),
        deployed_strategy=cast("Strategy | None", data["deployed_strategy"]),
        thresholds={cast(Strategy, s): c["threshold"] for s, c in collections.items()},
    )
