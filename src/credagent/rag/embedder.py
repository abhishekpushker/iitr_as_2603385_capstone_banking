"""Sentence embeddings (§ARCH 7.4, P20): all-MiniLM-L6-v2 loaded from a local directory only.

The model is never resolved by hub name: `Embedder` refuses any path that is not an existing
directory, and loads with `local_files_only=True` on CPU. `encode` is serialised by a lock
(cheap at this scale; avoids torch thread contention) and returns unit-length vectors.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from credagent.settings import get_settings

EMBEDDING_DIM = 384


class Embedder:
    def __init__(self, model_path: Path, *, batch_size: int) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"{model_path} is not a local model directory")
        torch.set_num_threads(1)
        self._model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        self._batch_size = batch_size
        self._lock = threading.Lock()
        dim = self._model.get_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise ValueError(f"{model_path}: embedding dimension {dim} != {EMBEDDING_DIM}")
        self.dimension: int = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [[float(x) for x in row] for row in vectors]


_instance: Embedder | None = None
_instance_lock = threading.Lock()


def get_embedder() -> Embedder:
    """The process-wide embedder, created on first use."""
    global _instance  # noqa: PLW0603  (process-wide singleton, §ARCH 8)
    with _instance_lock:
        if _instance is None:
            s = get_settings()
            _instance = Embedder(s.embed_model_path, batch_size=s.embed_batch_size)
        return _instance
