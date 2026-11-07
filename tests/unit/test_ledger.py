"""`ToolCallLedger` (§ARCH 7.9): the per-request record of which agent called which tool."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from credagent.tools.ledger import LedgerEntry, ToolCallLedger


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
