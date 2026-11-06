"""Response cache for the grounded-generation step (§ARCH 7.15, P13).

Key = `sha256(f"{kb_version}|{strategy}|{threshold}|{top_k}|{normalize_query(masked_query)}")`.
Keys are built from masked text only and are hex digests, so no PII is ever in a key. A new
`kb_version` (after `/v1/documents`) changes every key, and the cache is also `clear()`ed then.

`OrderedDict` LRU + TTL under one lock; the clock is injected (tests never sleep). Values are
computed outside the lock, so two concurrent misses on one key may both compute: harmless,
because generation is deterministic. A computation that raises is never cached. Fallback
answers are ordinary values and are cached.
"""

from __future__ import annotations

import hashlib
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable

from credagent.observability.metrics import Metrics
from credagent.settings import get_settings
from credagent.text import normalize

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
                         "‟": '"', "«": '"', "»": '"', "`": "'"})  # fmt: skip


def _is_edge_char(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def normalize_query(text: str) -> str:
    """NFKC, casefold, zero-width removal, whitespace collapse (`text.normalize`), quotes
    unified, leading/trailing punctuation stripped. Idempotent."""
    s = normalize(text).translate(_QUOTES)
    start, end = 0, len(s)
    while start < end and _is_edge_char(s[start]):
        start += 1
    while end > start and _is_edge_char(s[end - 1]):
        end -= 1
    return s[start:end]


def cache_key(
    *, kb_version: str, strategy: str, threshold: float, top_k: int, masked_query: str
) -> str:
    material = f"{kb_version}|{strategy}|{threshold}|{top_k}|{normalize_query(masked_query)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ResponseCache[V]:
    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        *,
        metrics: Metrics,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("max_entries must be ≥ 1 and ttl_seconds > 0")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._metrics = metrics
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, V]] = OrderedDict()

    @classmethod
    def from_settings(cls, *, metrics: Metrics) -> ResponseCache[V]:
        s = get_settings()
        return cls(s.cache_max_entries, s.cache_ttl_seconds, metrics=metrics)

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self._clock() >= entry[0]:
                del self._entries[key]
                entry = None
            if entry is None:
                self._metrics.incr("cache_misses")
                return None
            self._entries.move_to_end(key)
            self._metrics.incr("cache_hits")
            return entry[1]

    def put(self, key: str, value: V) -> None:
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get_or_compute(self, key: str, compute: Callable[[], V]) -> tuple[V, bool]:
        """`(value, hit)`. On a miss `compute()` runs outside the lock; if it raises, nothing
        is stored and the exception propagates."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = compute()
        self.put(key, value)
        return value, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def list_keys(self) -> list[str]:
        """Current keys, least recently used first (for the PII canary scan, §ARCH 11.6)."""
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
