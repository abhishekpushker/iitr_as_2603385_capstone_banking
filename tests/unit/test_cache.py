"""Response cache core (§ARCH 7.15, P13): key, normalisation, LRU + TTL with an injected clock,
errors never cached, thread safety, and no raw PII in keys."""

from __future__ import annotations

import re
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent.cache.response_cache import ResponseCache, cache_key, normalize_query
from credagent.observability.metrics import Metrics

# §ARCH 11.6 fabricated canaries
CANARIES = ("ABCPE1234F", "2345 6789 0123", "234567890123", "account no 123456789012345")
KEY_ARGS = {"kb_version": "v1", "strategy": "sentence", "threshold": 0.567, "top_k": 3}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(max_entries: int = 4, ttl: float = 60.0) -> tuple[ResponseCache[str], Clock, Metrics]:
    clock, metrics = Clock(), Metrics()
    return ResponseCache(max_entries, ttl, clock=clock, metrics=metrics), clock, metrics


# ------------------------------------------------------------------ normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  What is the  EMI?  ", "what is the emi"),
        ("WHAT IS THE EMI", "what is the emi"),
        ("“Is KYC needed?”", "is kyc needed"),
        ("‘Prepayment’ penalty?!", "prepayment' penalty"),
        ("...emi\tdate...", "emi date"),
        ("ＥＭＩ date", "emi date"),  # NFKC folds full-width letters
        ("emi\u200bdate", "emidate"),  # zero-width removed
    ],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


def test_quotes_are_normalised_inside_the_text() -> None:
    assert normalize_query("what’s the “annual fee” rule") == 'what\'s the "annual fee" rule'


@given(st.text())
def test_normalisation_is_idempotent(text: str) -> None:
    once = normalize_query(text)
    assert normalize_query(once) == once


# ------------------------------------------------------------------ key


def test_key_is_sha256_hex_and_ignores_surface_differences() -> None:
    a = cache_key(**KEY_ARGS, masked_query="What is the EMI?")
    b = cache_key(**KEY_ARGS, masked_query="  what is the emi ")
    assert a == b and re.fullmatch(r"[0-9a-f]{64}", a)


@pytest.mark.parametrize(
    "change",
    [{"kb_version": "v2"}, {"strategy": "fixed"}, {"threshold": 0.568}, {"top_k": 4}],
)
def test_key_includes_every_component(change: dict[str, object]) -> None:
    base = cache_key(**KEY_ARGS, masked_query="emi date")
    assert cache_key(**{**KEY_ARGS, **change}, masked_query="emi date") != base  # type: ignore[arg-type]


def test_key_differs_by_query() -> None:
    assert cache_key(**KEY_ARGS, masked_query="emi date") != cache_key(
        **KEY_ARGS, masked_query="annual fee"
    )


def test_keys_contain_no_raw_canary_pii() -> None:
    cache, _, _ = make(max_entries=16)
    for canary in CANARIES:
        key = cache_key(**KEY_ARGS, masked_query=f"my id is {canary}, what is the EMI?")
        cache.put(key, "answer")
    for key in cache.list_keys():
        assert re.fullmatch(r"[0-9a-f]{64}", key)
        assert not any(c.casefold() in key.casefold() for c in CANARIES)


# ------------------------------------------------------------------ hit / miss / metrics


def test_miss_then_hit() -> None:
    cache, _, metrics = make()
    assert cache.get("k") is None
    cache.put("k", "v")
    assert cache.get("k") == "v"
    assert (metrics.snapshot()["cache_misses"], metrics.snapshot()["cache_hits"]) == (1, 1)


def test_get_or_compute_computes_once() -> None:
    cache, _, metrics = make()
    calls: list[int] = []

    def compute() -> str:
        calls.append(1)
        return "answer"

    assert cache.get_or_compute("k", compute) == ("answer", False)
    assert cache.get_or_compute("k", compute) == ("answer", True)
    assert len(calls) == 1
    snap = metrics.snapshot()
    assert (snap["cache_misses"], snap["cache_hits"]) == (1, 1)


def test_errors_are_never_cached() -> None:
    cache, _, _ = make()

    def boom() -> str:
        raise RuntimeError("retrieval failed")

    with pytest.raises(RuntimeError):
        cache.get_or_compute("k", boom)
    assert len(cache) == 0
    assert cache.get_or_compute("k", lambda: "ok") == ("ok", False)


def test_clear_empties_the_cache() -> None:
    cache, _, _ = make()
    cache.put("a", "1")
    cache.clear()
    assert len(cache) == 0 and cache.get("a") is None


# ------------------------------------------------------------------ LRU


def test_lru_evicts_the_least_recently_used() -> None:
    cache, _, _ = make(max_entries=2)
    cache.put("a", "1")
    cache.put("b", "2")
    assert cache.get("a") == "1"  # a is now most recent
    cache.put("c", "3")
    assert cache.list_keys() == ["a", "c"]
    assert cache.get("b") is None


def test_put_existing_key_refreshes_without_growing() -> None:
    cache, _, _ = make(max_entries=2)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("a", "1b")
    cache.put("c", "3")
    assert cache.list_keys() == ["a", "c"] and cache.get("a") == "1b"


def test_default_size_comes_from_settings() -> None:
    cache: ResponseCache[str] = ResponseCache.from_settings(metrics=Metrics())
    assert (cache.max_entries, cache.ttl_seconds) == (512, 3600)


@pytest.mark.parametrize(("size", "ttl"), [(0, 1.0), (1, 0.0)])
def test_rejects_non_positive_limits(size: int, ttl: float) -> None:
    with pytest.raises(ValueError):
        ResponseCache(size, ttl, metrics=Metrics())


# ------------------------------------------------------------------ TTL (injected clock)


def test_entry_expires_after_ttl() -> None:
    cache, clock, metrics = make(ttl=60.0)
    cache.put("k", "v")
    clock.now = 1059.999
    assert cache.get("k") == "v"
    clock.now = 1060.0  # put at 1000.0 + ttl 60.0: expired exactly at the boundary
    assert cache.get("k") is None
    assert len(cache) == 0
    assert metrics.snapshot()["cache_misses"] == 1


def test_put_restarts_the_ttl() -> None:
    cache, clock, _ = make(ttl=60.0)
    cache.put("k", "v1")
    clock.now += 50
    cache.put("k", "v2")
    clock.now += 50
    assert cache.get("k") == "v2"


# ------------------------------------------------------------------ concurrency


def test_thread_safe_under_eight_threads() -> None:
    cache, _, metrics = make(max_entries=64)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []
    rounds = 500

    def worker(n: int) -> None:
        try:
            barrier.wait()
            for i in range(rounds):
                key = f"k{(n * 7 + i) % 100}"
                cache.get_or_compute(key, lambda key=key: key.upper())
                assert len(cache) <= 64
        except BaseException as exc:  # noqa: BLE001  (surfaced by the assertion below)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    snap = metrics.snapshot()
    assert snap["cache_hits"] + snap["cache_misses"] == 8 * rounds
    assert len(cache) <= 64
    assert all(cache.get(k) in (None, k.upper()) for k in cache.list_keys())
