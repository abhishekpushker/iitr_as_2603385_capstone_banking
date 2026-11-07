"""Per-request state (§ARCH 6 step 0, D-031): `RequestContext`, the charge-only `BudgetMeter`
and the `ToolCallLedger` (§ARCH 7.9: which agent called which tool)."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from credagent.context import (
    BudgetMeter,
    LedgerEntry,
    RequestContext,
    ToolCallLedger,
    estimate_tokens,
)
from credagent.settings import get_settings


def test_records_in_order_and_returns_an_immutable_snapshot() -> None:
    ledger = ToolCallLedger()
    ledger.record(agent="lookup_agent", tool="check_loan_application_status",
                  args={"record_id": "LA-0001"}, duration_ms=1.5, cache_hit=False)  # fmt: skip
    ledger.record(agent="retrieval_agent", tool="rag_lookup", args={"query": "q"},
                  duration_ms=0.0, cache_hit=True)  # fmt: skip
    entries = ledger.entries()
    assert isinstance(entries, tuple)
    assert [(e.agent, e.tool, e.cache_hit) for e in entries] == [
        ("lookup_agent", "check_loan_application_status", False),
        ("retrieval_agent", "rag_lookup", True),
    ]
    assert len(ledger) == 2
    with pytest.raises(ValidationError):
        entries[0].agent = "x"  # type: ignore[misc]


def test_entry_rejects_negative_duration_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LedgerEntry(agent="a", tool="t", args={}, duration_ms=-1, cache_hit=False)
    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(
            {"agent": "a", "tool": "t", "args": {}, "duration_ms": 0, "cache_hit": False, "x": 1}
        )


def test_entry_args_are_copied() -> None:
    args = {"query": "q"}
    ledger = ToolCallLedger()
    ledger.record(agent="a", tool="t", args=args, duration_ms=0, cache_hit=False)
    args["query"] = "changed"
    assert ledger.entries()[0].args == {"query": "q"}


def test_concurrent_records_are_not_lost() -> None:
    ledger = ToolCallLedger()

    def worker() -> None:
        for _ in range(200):
            ledger.record(agent="a", tool="t", args={}, duration_ms=0, cache_hit=False)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.entries()) == 1600


# ------------------------------------------------------------------ BudgetMeter (charge-only)


@pytest.mark.parametrize(("chars", "tokens"), [(0, 0), (1, 1), (4, 1), (5, 2), (8, 2), (9, 3)])
def test_estimate_is_ceil_chars_over_four(chars: int, tokens: int) -> None:
    """§ARCH 2.2.8: the same heuristic CrewAI logs for a custom LLM."""
    assert estimate_tokens("x" * chars) == tokens


def test_meter_accumulates_per_persona_and_flags_only_past_the_limit() -> None:
    meter = BudgetMeter(max_tokens=10)
    assert meter.charge("retrieval", 6) is False
    assert meter.charge("composer", 4) is False  # exactly at the limit is still within it
    assert (meter.used_tokens, meter.exceeded) == (10, False)
    assert meter.charge("composer", 1) is True
    assert meter.exceeded is True
    assert meter.calls() == {"composer": 2, "retrieval": 1}
    assert meter.tokens_by_persona() == {"composer": 5, "retrieval": 6}


def test_exceeded_is_sticky() -> None:
    meter = BudgetMeter(max_tokens=1)
    meter.charge("a", 5)
    assert meter.charge("a", 0) is True
    assert meter.exceeded is True


def test_meter_rejects_negative_charges_and_limits() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        BudgetMeter(max_tokens=0)
    with pytest.raises(ValueError, match="negative"):
        BudgetMeter(max_tokens=5).charge("a", -1)


def test_concurrent_charges_are_not_lost() -> None:
    meter = BudgetMeter(max_tokens=10**9)

    def worker() -> None:
        for _ in range(500):
            meter.charge("p", 2)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (meter.used_tokens, meter.calls()) == (8000, {"p": 4000})


# ------------------------------------------------------------------ RequestContext


def test_new_context_has_fresh_state_and_the_settings_budget() -> None:
    a = RequestContext.new(session_id="s1")
    b = RequestContext.new(session_id="s1")
    assert a.trace_id != b.trace_id
    assert len(a.trace_id) == 32 and int(a.trace_id, 16) >= 0
    assert a.ledger is not b.ledger and a.budget is not b.budget
    assert a.budget.max_tokens == get_settings().budget_max_tokens  # from config/budget.json
    assert (len(a.ledger), a.budget.used_tokens, a.timings_ms) == (0, 0, {})


def test_context_accepts_a_fixed_trace_id_and_budget() -> None:
    ctx = RequestContext.new(session_id="s", trace_id="t-1", max_tokens=42)
    assert (ctx.trace_id, ctx.session_id, ctx.budget.max_tokens) == ("t-1", "s", 42)


def test_context_fields_cannot_be_rebound() -> None:
    ctx = RequestContext.new(session_id="s")
    with pytest.raises(AttributeError):
        ctx.trace_id = "x"  # type: ignore[misc]
