"""Per-request state (§ARCH 6 step 0, D-031). Bottom layer, beside `schemas`, so `llm`, `tools`
and later `pipeline` share one object without importing each other.

* `RequestContext`: trace id, session id, the meter, the ledger and stage timings.
* `BudgetMeter`: **charge-only**. Every mock-LLM call charges `ceil(chars/4)` tokens (the
  heuristic CrewAI logs for a custom LLM, §ARCH 2.2.8). Passing `max_tokens` sets the sticky
  `exceeded` flag; deciding what to do about it (preflight, HTTP 413) is policy, in
  `governance` (Phase 9).
* `ToolCallLedger` (§ARCH 7.9): one entry per tool call, `(agent, tool, args, duration_ms,
  cache_hit)`. It is how transcripts and `tools_invoked` prove which agent actually called which
  tool. `agent` is the owner key the ACL checks (`lookup_agent`, …), not the display role.
  Arguments are the tool's (already masked) inputs.

Each mutable object guards its updates and snapshots with one lock.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from credagent.settings import get_settings


def estimate_tokens(text: str) -> int:
    """`ceil(len/4)`: one token per four characters."""
    return math.ceil(len(text) / 4)


class BudgetMeter:
    def __init__(self, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be ≥ 1")
        self._lock = threading.Lock()
        self._max = max_tokens
        self._used = 0
        self._exceeded = False
        self._calls: dict[str, int] = {}
        self._tokens: dict[str, int] = {}

    @property
    def max_tokens(self) -> int:
        return self._max

    @property
    def used_tokens(self) -> int:
        with self._lock:
            return self._used

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self._exceeded

    def charge(self, persona: str, tokens: int) -> bool:
        """Add one call's tokens; return whether the budget is (now or already) exceeded."""
        if tokens < 0:
            raise ValueError("cannot charge negative tokens")
        with self._lock:
            self._used += tokens
            self._calls[persona] = self._calls.get(persona, 0) + 1
            self._tokens[persona] = self._tokens.get(persona, 0) + tokens
            self._exceeded = self._exceeded or self._used > self._max
            return self._exceeded

    def calls(self) -> dict[str, int]:
        """LLM calls per persona, sorted by persona."""
        with self._lock:
            return dict(sorted(self._calls.items()))

    def tokens_by_persona(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._tokens.items()))


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str
    tool: str
    args: dict[str, str]
    duration_ms: float = Field(ge=0)
    cache_hit: bool


class ToolCallLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[LedgerEntry] = []

    def record(
        self, *, agent: str, tool: str, args: Mapping[str, str], duration_ms: float, cache_hit: bool
    ) -> LedgerEntry:
        entry = LedgerEntry(
            agent=agent, tool=tool, args=dict(args), duration_ms=duration_ms, cache_hit=cache_hit
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True)
class RequestContext:
    trace_id: str
    session_id: str
    budget: BudgetMeter
    ledger: ToolCallLedger = field(default_factory=ToolCallLedger)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def new(
        cls, *, session_id: str, trace_id: str | None = None, max_tokens: int | None = None
    ) -> RequestContext:
        """A fresh context; `trace_id` and `max_tokens` default to a random UUID and
        `BUDGET_MAX_TOKENS`."""
        limit = get_settings().budget_max_tokens if max_tokens is None else max_tokens
        return cls(
            trace_id=uuid.uuid4().hex if trace_id is None else trace_id,
            session_id=session_id,
            budget=BudgetMeter(limit),
        )
