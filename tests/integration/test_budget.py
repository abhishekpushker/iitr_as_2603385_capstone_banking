"""The runtime budget cap on the real pipeline (§ARCH 7.14.3): every measured query fits in half
the budget, an oversized request is rejected at preflight with zero LLM calls, and a runtime
overrun is caught after kickoff without a crash or a stored turn."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from credagent.governance.budget import BudgetPolicy
from credagent.memory.session import SessionStore
from credagent.pipeline.factory import build_orchestrator
from credagent.pipeline.orchestrator import BudgetExceeded, Orchestrator, TurnRecord
from credagent.settings import get_settings
from scripts import indexer, measure_budget

POLICY_QUERY = "Is there a prepayment penalty on a floating-rate home loan?"
OVERSIZED = (POLICY_QUERY + " ") * 200  # ≈ 12,000 chars (§ARCH 7.14.3 demo)


class ListSink:
    def __init__(self) -> None:
        self.records: list[TurnRecord] = []

    def emit(self, record: TurnRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def indexed() -> None:
    assert indexer.main([]) == 0


@pytest.fixture
def built(indexed: None) -> Iterator[tuple[Orchestrator, SessionStore, ListSink]]:
    store, sink = SessionStore.from_settings(), ListSink()
    yield build_orchestrator(store=store, sink=sink), store, sink


@pytest.mark.anyio
async def test_every_measured_query_passes_preflight_and_uses_at_most_half_the_budget(
    built: tuple[Orchestrator, SessionStore, ListSink],
) -> None:
    orch, _, sink = built
    budget = get_settings().budget_max_tokens
    policy = BudgetPolicy()
    for session, queries in measure_budget.conversations():
        for query in queries:
            await orch.handle(query, session, "script")
            record = sink.records[-1]
            assert not policy.preflight(record.masked_text).exceeds, record.masked_text
            assert record.tokens_estimated <= budget / 2, (record.masked_text, budget)
    assert len(sink.records) == measure_budget.request_count()


@pytest.mark.anyio
async def test_an_oversized_request_is_rejected_at_preflight_with_zero_llm_calls(
    built: tuple[Orchestrator, SessionStore, ListSink],
) -> None:
    orch, store, sink = built
    assert 11_000 < len(OVERSIZED) < get_settings().max_message_chars
    with pytest.raises(BudgetExceeded) as info:
        await orch.handle(OVERSIZED, "oversized", "script")
    assert info.value.stage == "preflight"
    assert info.value.projected_tokens > info.value.budget_tokens
    (record,) = sink.records
    assert (record.outcome, record.error_code) == ("budget_rejected", "BUDGET_EXCEEDED")
    assert record.llm_usage == () and record.tokens_estimated == 0  # the mock LLM call counter
    assert record.tools_invoked == () and store.history_length("oversized") == 0


@pytest.mark.anyio
async def test_a_runtime_overrun_is_caught_after_kickoff_without_a_crash_or_memory_append(
    indexed: None,
) -> None:
    """Forced: the policy's constants are zeroed so preflight (7 × input tokens) passes, while
    the meter's tiny limit is crossed by the first real prompt."""
    limit = 200
    policy = BudgetPolicy(prompt_overhead_tokens=0, max_output_tokens=0, max_tokens=limit)
    store, sink = SessionStore.from_settings(), ListSink()
    orch = build_orchestrator(store=store, sink=sink, preflight=policy, max_tokens=limit)
    assert not policy.preflight(POLICY_QUERY).exceeds
    with pytest.raises(BudgetExceeded) as info:
        await orch.handle(POLICY_QUERY, "overrun", "script")
    assert info.value.stage == "runtime"
    (record,) = sink.records
    assert record.outcome == "budget_rejected" and len(record.llm_usage) >= 1
    assert record.tokens_estimated > limit
    assert store.history_length("overrun") == 0
