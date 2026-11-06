"""Thread-safe counters (§ARCH 7.15)."""

from __future__ import annotations

import threading

import pytest

from credagent.observability.metrics import COUNTERS, Metrics, get_metrics


def test_counters_start_at_zero() -> None:
    snap = Metrics().snapshot()
    assert set(COUNTERS) == {
        "retrieval_calls", "embedding_calls", "generation_calls", "cache_hits", "cache_misses",
    }  # fmt: skip
    assert {k: snap[k] for k in COUNTERS} == dict.fromkeys(COUNTERS, 0)
    assert snap["llm_calls"] == {}


def test_incr_and_snapshot_is_a_copy() -> None:
    m = Metrics()
    m.incr("retrieval_calls")
    m.incr("embedding_calls", 3)
    m.incr_llm("composer")
    m.incr_llm("composer")
    snap = m.snapshot()
    assert (snap["retrieval_calls"], snap["embedding_calls"]) == (1, 3)
    assert snap["llm_calls"] == {"composer": 2}
    snap["llm_calls"]["composer"] = 99  # type: ignore[index]
    assert m.snapshot()["llm_calls"] == {"composer": 2}


def test_unknown_counter_and_negative_increment_are_rejected() -> None:
    m = Metrics()
    with pytest.raises(KeyError):
        m.incr("cache_hit")
    with pytest.raises(ValueError):
        m.incr("cache_hits", -1)


def test_reset() -> None:
    m = Metrics()
    m.incr("cache_hits")
    m.incr_llm("x")
    m.reset()
    assert m.snapshot() == Metrics().snapshot()


def test_process_wide_instance_is_shared() -> None:
    assert get_metrics() is get_metrics()


def test_increments_are_not_lost_under_eight_threads() -> None:
    m = Metrics()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(2000):
            m.incr("generation_calls")
            m.incr_llm("lookup")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = m.snapshot()
    assert snap["generation_calls"] == 16000
    assert snap["llm_calls"] == {"lookup": 16000}
