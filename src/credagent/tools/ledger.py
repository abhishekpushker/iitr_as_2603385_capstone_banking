"""`ToolCallLedger` (§ARCH 7.9): one entry per tool call, `(agent, tool, args, duration_ms,
cache_hit)`. It is how transcripts and `tools_invoked` prove which agent actually called which
tool. `agent` is the owner key the ACL checks (`lookup_agent`, …), not the display role.

Arguments are the tool's (already masked) inputs. One lock guards appends and snapshots.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


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
