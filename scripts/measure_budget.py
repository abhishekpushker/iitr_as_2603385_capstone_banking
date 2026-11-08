"""Measure the budget constants (§ARCH 7.14.3) from real rendered prompts.
`python -m scripts.measure_budget`.

Every eval and demo query in the repository (the transcript and demo sets and the 15-item
evaluation test set, §ARCH 7.18) runs through the real pipeline (crew + Autogen review) with an
unlimited meter and no preflight. From the meter's per-call log:

* `PROMPT_OVERHEAD_TOKENS` = max over all calls of (prompt tokens − the request's input tokens),
  rounded up to the next 100;
* `MAX_OUTPUT_TOKENS` = max completion tokens of any call;
* `BUDGET_MAX_TOKENS` = ceil to the next 1,000 of max(2 × max actual tokens per request,
  1.2 × max preflight projection of a measured query); `BUDGET_MAX_COST_USD` = that × price/1k.

Writes `config/budget.json` (generated; re-bless, never hand-edit). Query text is not written,
only each query's source label and counts. Needs the index (`make budget` runs it first).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import anyio

from credagent.context import estimate_tokens
from credagent.governance.budget import N_LLM_CALLS, BudgetPolicy, PreflightResult, cost_usd
from credagent.memory.session import SessionStore
from credagent.pipeline.factory import build_orchestrator
from credagent.pipeline.orchestrator import TurnRecord
from credagent.settings import get_settings
from scripts import make_transcripts as mt
from scripts.run_eval import load_testset

UNLIMITED = 10**9


def conversations() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """`(session label, queries)`: one session per query, except transcript 10's three turns."""
    pan, aadhaar, account = mt.PII_PIPELINE_VALUES
    singles = [
        ("t03-sample", mt.SAMPLE_QUERY),
        *((f"t05-demo-{i}", q) for i, (q, _) in enumerate(mt.DEMO_IN_SCOPE, start=1)),
        ("t05-out-of-scope", mt.DEMO_OUT_OF_SCOPE),
        ("t08-policy", mt.CREW_POLICY_QUERY),
        ("t09-status", mt.CREW_STATUS_QUERY),
        ("t09-mixed", mt.CREW_MIXED_QUERY),
        ("t09-unknown", mt.CREW_UNKNOWN_QUERY),
        ("t09-clarify", mt.CREW_CLARIFY_QUERY),
        ("t13-pii", mt.PII_PIPELINE_QUERY.format(pan=pan, aadhaar=aadhaar, account=account)),
        ("t15-groundedness", mt.GROUNDEDNESS_QUERY),
        *((f"t19-eval-{i.id}", i.query) for i in load_testset(get_settings().eval_testset_path)),
    ]
    seen: set[str] = set()
    out: list[tuple[str, tuple[str, ...]]] = []
    for label, query in singles:
        if query not in seen:
            seen.add(query)
            out.append((label, (query,)))
    out.append(("t10-memory", (mt.MEMORY_T1, mt.MEMORY_T2, mt.MEMORY_T3)))
    return tuple(out)


def request_count() -> int:
    return sum(len(queries) for _, queries in conversations())


@dataclass(frozen=True)
class Row:
    source: str
    input_tokens: int
    llm_calls: int
    actual_tokens: int
    max_prompt_overhead_tokens: int  # max over this request's calls of prompt − input
    max_completion_tokens: int
    projected_tokens: int = 0  # filled in once the constants are known


class _NoPreflight:
    def preflight(self, masked_text: str) -> PreflightResult:
        return PreflightResult(exceeds=False, projected_tokens=0, budget_tokens=UNLIMITED)


class _Sink:
    def __init__(self) -> None:
        self.records: list[TurnRecord] = []

    def emit(self, record: TurnRecord) -> None:
        self.records.append(record)


def _row(source: str, record: TurnRecord) -> Row:
    input_tokens = estimate_tokens(record.masked_text)
    calls = record.llm_usage
    return Row(
        source=source, input_tokens=input_tokens, llm_calls=len(calls),
        actual_tokens=record.tokens_estimated,
        max_prompt_overhead_tokens=max((c.prompt_tokens - input_tokens for c in calls), default=0),
        max_completion_tokens=max((c.completion_tokens for c in calls), default=0),
    )  # fmt: skip


def run_queries() -> list[Row]:
    """Each request on a fixed trace id, so prompts (and so token counts) are reproducible."""
    sink = _Sink()
    orch = build_orchestrator(
        store=SessionStore.from_settings(), sink=sink, preflight=_NoPreflight(),
        max_tokens=UNLIMITED,
    )  # fmt: skip
    rows: list[Row] = []

    async def run() -> None:
        for session, queries in conversations():
            for i, query in enumerate(queries, start=1):
                source = session if len(queries) == 1 else f"{session}-t{i}"
                await orch.handle(query, f"budget-{session}", "script",
                                  trace_id=f"budget-{source}")  # fmt: skip
                rows.append(_row(source, sink.records[-1]))

    anyio.run(run)
    return rows


def _ceil_to(value: Fraction | int, step: int) -> int:
    return math.ceil(Fraction(value) / step) * step


def build_config(rows: Sequence[Row], price_per_1k_tokens_usd: float) -> dict[str, Any]:
    overhead_raw = max(r.max_prompt_overhead_tokens for r in rows)
    overhead = _ceil_to(overhead_raw, 100)
    max_output = max(r.max_completion_tokens for r in rows)
    policy = BudgetPolicy(
        prompt_overhead_tokens=overhead, max_output_tokens=max_output, max_tokens=UNLIMITED,
        max_cost_usd=float(UNLIMITED), price_per_1k_tokens_usd=price_per_1k_tokens_usd,
    )  # fmt: skip
    rows = [
        Row(**{**asdict(r), "projected_tokens": policy.project_tokens(r.input_tokens)})
        for r in rows
    ]
    max_actual = max(r.actual_tokens for r in rows)
    max_projection = max(r.projected_tokens for r in rows)
    budget = _ceil_to(max(Fraction(2 * max_actual), Fraction(6, 5) * max_projection), 1000)
    return {
        "budget_max_cost_usd": cost_usd(budget, price_per_1k_tokens_usd),
        "budget_max_tokens": budget,
        "max_output_tokens": max_output,
        "measured": {
            "llm_calls": sum(r.llm_calls for r in rows),
            "max_actual_total_tokens": max_actual,
            "max_completion_tokens": max_output,
            "max_normal_projection_tokens": max_projection,
            "max_prompt_overhead_tokens": overhead_raw,
            "requests": len(rows),
        },
        "n_llm_calls": N_LLM_CALLS,
        "price_per_1k_tokens_usd": price_per_1k_tokens_usd,
        "prompt_overhead_tokens": overhead,
        "queries": [asdict(r) for r in rows],
        "rule": (
            "PROMPT_OVERHEAD_TOKENS = ceil100(max(prompt - input)); MAX_OUTPUT_TOKENS = "
            "max(completion); BUDGET_MAX_TOKENS = ceil1000(max(2 * max_actual_total, "
            "1.2 * max_normal_projection)); BUDGET_MAX_COST_USD = BUDGET_MAX_TOKENS * price / 1000"
        ),
    }


def dumps_config(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        s = get_settings()
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"refusing to run: {exc}") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=s.budget_config_path)
    args = parser.parse_args(argv)

    config = build_config(run_queries(), s.price_per_1k_tokens_usd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_config(config), encoding="utf-8")
    m = config["measured"]
    print(
        f"{m['requests']} requests, {m['llm_calls']} LLM calls; overhead "
        f"{m['max_prompt_overhead_tokens']} → {config['prompt_overhead_tokens']}; max output "
        f"{config['max_output_tokens']}; max actual {m['max_actual_total_tokens']}; max "
        f"projection {m['max_normal_projection_tokens']}; budget {config['budget_max_tokens']} "
        f"tokens / ${config['budget_max_cost_usd']}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
