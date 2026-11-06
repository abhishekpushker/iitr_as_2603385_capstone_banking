"""Thread-safe in-process counters (§ARCH 7.15), exposed later by `GET /v1/stats`.

A fixed set of named counters plus `llm_calls{persona}`. Every update and every snapshot holds
one lock, so concurrent increments are never lost and a snapshot is a consistent copy.
"""

from __future__ import annotations

import threading
from typing import Final

COUNTERS: Final = (
    "retrieval_calls",
    "embedding_calls",
    "generation_calls",
    "cache_hits",
    "cache_misses",
)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(COUNTERS, 0)
        self._llm_calls: dict[str, int] = {}

    def incr(self, name: str, n: int = 1) -> None:
        if name not in self._counts:
            raise KeyError(f"unknown counter {name!r}; known: {COUNTERS}")
        if n < 0:
            raise ValueError("counters only increase")
        with self._lock:
            self._counts[name] += n

    def incr_llm(self, persona: str) -> None:
        with self._lock:
            self._llm_calls[persona] = self._llm_calls.get(persona, 0) + 1

    def snapshot(self) -> dict[str, int | dict[str, int]]:
        with self._lock:
            return {**self._counts, "llm_calls": dict(sorted(self._llm_calls.items()))}

    def reset(self) -> None:
        with self._lock:
            self._counts = dict.fromkeys(COUNTERS, 0)
            self._llm_calls = {}


_instance = Metrics()


def get_metrics() -> Metrics:
    """The process-wide counters."""
    return _instance
