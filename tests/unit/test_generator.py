"""Grounded generation (§ARCH 7.5): threshold fallback, verbatim extractive synthesis, P16 dedupe,
determinism, and no state beyond the chunks passed in. Plus the `GroundedGenerator` wiring."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from credagent.llm.engine import (
    MAX_ANSWER_SENTENCES,
    MIN_SENTENCE_SCORE,
    RECALL_WEIGHT,
    SIMILARITY_WEIGHT,
    MockEngine,
)
from credagent.observability.metrics import Metrics
from credagent.rag.generator import GroundedGenerator, LLMClient
from credagent.rag.retriever import RetrievalResult
from credagent.rag.store import RetrievedChunk
from credagent.schemas import FALLBACK_ANSWER, GeneratedAnswer
from credagent.text import split_sentences

PREPAY = (
    "Floating-rate home loans taken by individuals carry no prepayment penalty. Fixed-rate "
    "personal loans charge a prepayment penalty of 4% of the prepaid principal during the first "
    "12 EMIs and 2% after that. You may foreclose a loan in full only after 6 EMIs have been "
    "paid. Auto loans allow free part-prepayment of up to 25% of the outstanding principal every "
    "year."
)
RATES = "Personal loan interest ranges from 10.5% to 24% p.a. by credit score band."
THRESHOLD = 0.5
engine = MockEngine()


def chunk(
    doc_id: str, text: str, similarity: float, rank: int, *, start: int = 0, index: int = 0
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::sentence::{index:03d}", doc_id=doc_id, text=text,
        similarity=similarity, rank=rank, char_start=start,
    )  # fmt: skip


def sentence_chunks(doc_id: str, body: str, sims: Sequence[float]) -> list[RetrievedChunk]:
    out, pos = [], 0
    for i, (s, sim) in enumerate(zip(split_sentences(body), sims, strict=False)):
        pos = body.index(s, pos)
        out.append(chunk(doc_id, s, sim, len(out) + 1, start=pos, index=i))
    return out


def context_sentences(chunks: Sequence[RetrievedChunk]) -> set[str]:
    return {s for c in chunks for s in split_sentences(c.text)}


def test_scoring_constants_are_the_architecture_values() -> None:
    assert (RECALL_WEIGHT, SIMILARITY_WEIGHT, MAX_ANSWER_SENTENCES) == (0.7, 0.3, 2)
    # A sentence needs at least one query token: similarity alone (≤ 1) gives at most 0.3.
    assert MIN_SENTENCE_SCORE == SIMILARITY_WEIGHT


# ------------------------------------------------------------------ fallback


def test_below_threshold_returns_the_fallback() -> None:
    chunks = [chunk("kb-08", PREPAY, THRESHOLD - 0.001, 1)]
    out = engine.generate_grounded("prepayment penalty personal loan", chunks, threshold=THRESHOLD)
    assert out == GeneratedAnswer(
        answer=FALLBACK_ANSWER, sources=(), grounded=False, top1_similarity=THRESHOLD - 0.001
    )
    assert FALLBACK_ANSWER == "I don't know based on the available policy documents."


def test_at_threshold_answers() -> None:
    chunks = [chunk("kb-08", PREPAY, THRESHOLD, 1)]
    out = engine.generate_grounded("prepayment penalty personal loan", chunks, threshold=THRESHOLD)
    assert out.grounded and out.answer != FALLBACK_ANSWER


def test_no_chunks_returns_the_fallback() -> None:
    out = engine.generate_grounded("anything at all", [], threshold=THRESHOLD)
    assert out.answer == FALLBACK_ANSWER and not out.grounded and out.top1_similarity == -1.0


def test_no_sentence_sharing_a_query_token_returns_the_fallback() -> None:
    chunks = [chunk("kb-07", RATES, 0.9, 1)]
    out = engine.generate_grounded("weather mumbai tomorrow", chunks, threshold=THRESHOLD)
    assert out.answer == FALLBACK_ANSWER and not out.grounded and out.sources == ()


# ------------------------------------------------------------------ extractive synthesis


def test_every_answer_sentence_is_verbatim_from_the_context() -> None:
    chunks = sentence_chunks("kb-08", PREPAY, [0.8, 0.7, 0.6, 0.55])
    out = engine.generate_grounded(
        "What prepayment penalty do personal loans charge?", chunks, threshold=THRESHOLD
    )
    assert out.grounded
    produced = split_sentences(out.answer)
    assert 1 <= len(produced) <= MAX_ANSWER_SENTENCES
    assert set(produced) <= context_sentences(chunks)
    assert engine.support_report(out.answer, [c.text for c in chunks]).all_supported


def test_the_best_scoring_sentences_are_kept_in_document_order() -> None:
    """Sentence 2 matches the query best, sentence 1 second; both are emitted, in body order."""
    chunks = sentence_chunks("kb-08", PREPAY, [0.6, 0.8, 0.6, 0.6])
    out = engine.generate_grounded(
        "personal loans prepayment penalty first 12 EMIs", chunks, threshold=THRESHOLD
    )
    s = split_sentences(PREPAY)
    assert out.answer == f"{s[0]} {s[1]}"
    assert [r.chunk_id for r in out.sources] == [chunks[0].chunk_id, chunks[1].chunk_id]


def test_document_order_follows_the_best_ranked_document_first() -> None:
    chunks = [
        chunk("kb-07", RATES, 0.8, 1),
        chunk("kb-08", split_sentences(PREPAY)[1], 0.7, 2, start=75),
    ]
    out = engine.generate_grounded(
        "personal loan interest and prepayment penalty", chunks, threshold=THRESHOLD
    )
    assert out.answer == f"{RATES} {split_sentences(PREPAY)[1]}"
    assert [r.doc_id for r in out.sources] == ["kb-07", "kb-08"]


def test_sources_carry_similarity_and_top1() -> None:
    chunks = sentence_chunks("kb-08", PREPAY, [0.81, 0.7])
    out = engine.generate_grounded("home loans prepayment penalty", chunks, threshold=THRESHOLD)
    assert out.top1_similarity == 0.81
    assert {r.similarity for r in out.sources} <= {0.81, 0.7}


def test_similarity_slightly_above_one_is_clamped() -> None:
    """Chroma's float32 cosine distance can be a hair below 0 (similarity a hair above 1)."""
    chunks = [chunk("kb-07", RATES, 1.0000001, 1)]
    out = engine.generate_grounded("personal loan interest", chunks, threshold=THRESHOLD)
    assert out.top1_similarity == 1.0 and out.sources[0].similarity == 1.0


# ------------------------------------------------------------------ P16 dedupe


def test_overlapping_fixed_chunks_never_repeat_a_sentence() -> None:
    """P16: two overlapping windows both contain sentence 2; it appears once in the answer."""
    s = split_sentences(PREPAY)
    first = f"{s[0]} {s[1]}"
    second = f"{s[1]} {s[2]}"
    chunks = [
        RetrievedChunk(chunk_id="kb-08::fixed::000", doc_id="kb-08", text=first,
                       similarity=0.8, rank=1, char_start=0),
        RetrievedChunk(chunk_id="kb-08::fixed::001", doc_id="kb-08", text=second,
                       similarity=0.79, rank=2, char_start=PREPAY.index(s[1])),
    ]  # fmt: skip
    out = engine.generate_grounded(
        "personal loans prepayment penalty principal EMIs", chunks, threshold=THRESHOLD
    )
    produced = split_sentences(out.answer)
    assert len(produced) == len(set(produced))
    assert produced.count(s[1]) == 1


def test_a_fragment_contained_in_a_full_sentence_is_dropped() -> None:
    """A fixed chunk can start mid-sentence; the fragment must not repeat the full sentence."""
    s = split_sentences(PREPAY)
    fragment = s[1][s[1].index("charge") :]
    chunks = [
        RetrievedChunk(chunk_id="kb-08::fixed::001", doc_id="kb-08", text=fragment,
                       similarity=0.82, rank=1, char_start=PREPAY.index(fragment)),
        RetrievedChunk(chunk_id="kb-08::fixed::000", doc_id="kb-08", text=f"{s[0]} {s[1]}",
                       similarity=0.8, rank=2, char_start=0),
    ]  # fmt: skip
    out = engine.generate_grounded(
        "charge prepayment penalty prepaid principal", chunks, threshold=THRESHOLD
    )
    assert fragment not in split_sentences(out.answer)
    assert s[1] in split_sentences(out.answer)


# ------------------------------------------------------------------ determinism / no state


def test_generation_is_deterministic() -> None:
    chunks = sentence_chunks("kb-08", PREPAY, [0.8, 0.7, 0.6, 0.55])
    query = "Can I foreclose or prepay my loan?"
    first = engine.generate_grounded(query, chunks, threshold=THRESHOLD)
    assert all(
        MockEngine().generate_grounded(query, chunks, threshold=THRESHOLD) == first
        for _ in range(5)
    )


def test_uses_only_the_given_chunks() -> None:
    """No global state: an earlier call with other chunks cannot leak into a later answer."""
    prepay = sentence_chunks("kb-08", PREPAY, [0.8, 0.7, 0.6, 0.55])
    rates = [chunk("kb-07", RATES, 0.8, 1)]
    query = "personal loan interest prepayment penalty"
    fresh = MockEngine().generate_grounded(query, rates, threshold=THRESHOLD)
    engine.generate_grounded(query, prepay, threshold=THRESHOLD)
    after = engine.generate_grounded(query, rates, threshold=THRESHOLD)
    assert after == fresh
    assert set(split_sentences(after.answer)) <= context_sentences(rates)
    assert {r.doc_id for r in after.sources} == {"kb-07"}


# ------------------------------------------------------------------ GroundedGenerator wiring


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str, int | None]] = []

    def retrieve(self, query: str, strategy: str, k: int | None = None) -> RetrievalResult:
        self.calls.append((query, strategy, k))
        return self.result


def test_generator_retrieves_then_calls_the_client_once() -> None:
    chunks = tuple(sentence_chunks("kb-08", PREPAY, [0.8, 0.7]))
    retriever = FakeRetriever(
        RetrievalResult(chunks=chunks, top1_similarity=0.8, strategy="sentence")
    )
    metrics = Metrics()
    client: LLMClient = engine
    gen = GroundedGenerator(
        retriever, client, strategy="sentence", threshold=THRESHOLD, top_k=2, metrics=metrics
    )
    out = gen.answer("home loans prepayment penalty")
    assert retriever.calls == [("home loans prepayment penalty", "sentence", 2)]
    assert out == engine.generate_grounded(
        "home loans prepayment penalty", chunks, threshold=THRESHOLD
    )
    assert metrics.snapshot()["generation_calls"] == 1


def test_generator_counts_a_fallback_as_a_generation_call() -> None:
    retriever = FakeRetriever(RetrievalResult(chunks=(), top1_similarity=-1.0, strategy="fixed"))
    metrics = Metrics()
    gen = GroundedGenerator(
        retriever, engine, strategy="fixed", threshold=THRESHOLD, top_k=3, metrics=metrics
    )
    assert gen.answer("q").answer == FALLBACK_ANSWER
    assert metrics.snapshot()["generation_calls"] == 1


def test_generator_rejects_a_threshold_outside_cosine_range() -> None:
    with pytest.raises(ValueError, match="threshold"):
        GroundedGenerator(
            FakeRetriever(RetrievalResult(chunks=(), top1_similarity=-1.0, strategy="fixed")),
            engine, strategy="fixed", threshold=1.5, top_k=3, metrics=Metrics(),
        )  # fmt: skip
