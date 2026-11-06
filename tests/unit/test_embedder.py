"""Embedder (§ARCH 7.4, P20): local model path only, 384-dim, normalised, deterministic."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from credagent.rag import embedder as embedder_module
from credagent.rag.embedder import EMBEDDING_DIM, Embedder, get_embedder
from credagent.settings import get_settings

SAMPLES = ["Is there a penalty for prepaying a personal loan?", "₹10,000 at 10.5% p.a.", "x"]


@pytest.fixture(scope="module")
def emb() -> Embedder:
    return get_embedder()


def test_dimension_is_384(emb: Embedder) -> None:
    vectors = emb.encode(SAMPLES)
    assert emb.dimension == EMBEDDING_DIM == 384
    assert [len(v) for v in vectors] == [384] * len(SAMPLES)
    assert all(isinstance(x, float) for x in vectors[0])


def test_vectors_are_unit_length(emb: Embedder) -> None:
    for v in emb.encode(SAMPLES):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_repeated_calls_are_identical(emb: Embedder) -> None:
    assert emb.encode(SAMPLES) == emb.encode(SAMPLES)
    # Batch composition (padding) moves values only at float32 rounding level, D-021.
    assert np.allclose(emb.encode(SAMPLES[:1]), emb.encode(SAMPLES)[:1], atol=1e-6)


def test_empty_input_needs_no_model_call(emb: Embedder) -> None:
    assert emb.encode([]) == []


def test_singleton_is_shared(emb: Embedder) -> None:
    assert get_embedder() is emb


def test_singleton_is_created_once_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class Fake:
        def __init__(self, path: Path, *, batch_size: int) -> None:
            created.append(self)

    monkeypatch.setattr(embedder_module, "_instance", None)
    monkeypatch.setattr(embedder_module, "Embedder", Fake)
    barrier = threading.Barrier(8)
    results: list[object] = []

    def worker() -> None:
        barrier.wait()
        results.append(get_embedder())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1 and all(r is created[0] for r in results)


def test_loads_from_the_local_path_with_sockets_disabled() -> None:
    """P20: the unit layer runs under pytest-socket; a hub lookup would raise here."""
    with pytest.raises(Exception, match="socket"):
        socket.create_connection(("1.1.1.1", 443), timeout=1)
    fresh = Embedder(get_settings().embed_model_path, batch_size=32)
    assert fresh.dimension == 384


@pytest.mark.parametrize(
    "path", ["sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "/nonexistent/model"]
)
def test_hub_names_and_missing_paths_are_rejected(path: str) -> None:
    with pytest.raises(FileNotFoundError, match="local model directory"):
        Embedder(Path(path), batch_size=32)


def test_torch_runs_single_threaded(emb: Embedder) -> None:
    import torch

    assert torch.get_num_threads() == 1


def test_vectors_match_across_processes(emb: Embedder) -> None:
    code = (
        "import json, sys; from credagent.rag.embedder import get_embedder; "
        "print(json.dumps(get_embedder().encode(json.loads(sys.argv[1]))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(SAMPLES)],
        capture_output=True, text=True, check=True, timeout=120,
    )  # fmt: skip
    other = json.loads(out.stdout)
    assert np.allclose(other, emb.encode(SAMPLES), atol=1e-6)
