"""sentence-transformers 6.2.0 behaviour the embedder relies on (§ARCH 2.1, 7.4; P20)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

import credagent  # noqa: F401  (egress-off env first)

MODEL_DIR = Path("/models/all-MiniLM-L6-v2")
TEXTS = ["Is there a penalty for prepaying a personal loan?", "₹10,000 at 10.5% p.a.", "x"]


@pytest.fixture(scope="module")
def model() -> SentenceTransformer:
    torch.set_num_threads(1)
    return SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)


def test_version_is_pinned() -> None:
    assert sentence_transformers.__version__ == "6.2.0"


def test_local_directory_loads_with_sockets_disabled(model: SentenceTransformer) -> None:
    assert model.get_embedding_dimension() == 384


def test_normalize_embeddings_gives_unit_vectors(model: SentenceTransformer) -> None:
    vectors = model.encode(TEXTS, normalize_embeddings=True, convert_to_numpy=True)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_same_batch_is_bit_identical(model: SentenceTransformer) -> None:
    a = model.encode(TEXTS, normalize_embeddings=True)
    b = model.encode(TEXTS, normalize_embeddings=True)
    assert np.array_equal(a, b)


def test_batch_composition_changes_values_only_at_float32_rounding(
    model: SentenceTransformer,
) -> None:
    """Padding to the longest text in a batch shifts values by ~1e-7, so a query embedded alone
    is not bit-identical to the same text embedded in a batch (D-021)."""
    alone = model.encode(TEXTS[:1], normalize_embeddings=True)[0]
    batched = model.encode(TEXTS, normalize_embeddings=True)[0]
    assert not np.array_equal(alone, batched)
    assert np.max(np.abs(alone - batched)) < 1e-6


def test_hub_name_is_not_resolved_offline() -> None:
    with pytest.raises(OSError):
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
