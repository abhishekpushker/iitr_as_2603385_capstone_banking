"""Chunking strategies (§ARCH 7.3, Task 3): ids, coverage, overlap, word boundaries, spans."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credagent import text
from credagent.kb.chunking import (
    FIXED_PARAMS,
    Chunk,
    FixedParams,
    chunk_document,
    chunk_fixed,
    chunk_params,
    chunk_sentences,
)
from credagent.kb.loader import DISCLAIMER, KbDocument, load_kb

KB_DIR = Path(__file__).resolve().parents[2] / "kb"


def _doc(body: str, doc_id: str = "kb-99-test") -> KbDocument:
    return KbDocument(
        doc_id=doc_id, title="T", topic="T", version=1, disclaimer=DISCLAIMER, body=body
    )


@pytest.fixture(scope="module")
def kb() -> list[KbDocument]:
    return load_kb(KB_DIR)


# ------------------------------------------------------------------ invariants (shared)


def _assert_word_boundaries(body: str, chunk: Chunk) -> None:
    start, end = chunk.char_start, chunk.char_end
    assert start == 0 or body[start - 1].isspace(), f"{chunk.chunk_id} starts mid-word"
    assert end == len(body) or body[end].isspace(), f"{chunk.chunk_id} ends mid-word"


def _assert_fixed_invariants(
    body: str, chunks: list[Chunk], params: FixedParams, *, expect_overlap: bool = True
) -> None:
    assert chunks, "a non-blank body yields at least one chunk"
    covered = [False] * len(body)
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.text and c.text == c.text.strip(), "no empty or padded chunks"
        assert body[c.char_start : c.char_end] == c.text, "span reconstructs the text"
        _assert_word_boundaries(body, c)
        longest_word = max(len(w) for w in c.text.split())
        assert len(c.text) <= max(params.size + params.max_extend, longest_word)
        for j in range(c.char_start, c.char_end):
            covered[j] = True
    assert all(covered[j] or body[j].isspace() for j in range(len(body))), "full coverage"
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.char_start < nxt.char_start, "progress"
        if expect_overlap:
            assert nxt.char_start < prev.char_end, f"{prev.chunk_id} → {nxt.chunk_id}: no overlap"


# ------------------------------------------------------------------ fixed-size


def test_fixed_params_match_the_architecture() -> None:
    assert FixedParams(size=160, overlap=40, max_extend=20) == FIXED_PARAMS


@pytest.mark.parametrize(
    "bad", [{"size": 0}, {"overlap": -1}, {"overlap": 160}, {"max_extend": -1}]
)
def test_fixed_params_are_validated(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        FixedParams(**{"size": 160, "overlap": 40, "max_extend": 20, **bad})


def test_fixed_chunks_over_the_real_kb(kb: list[KbDocument]) -> None:
    for doc in kb:
        chunks = chunk_fixed(doc)
        _assert_fixed_invariants(doc.body, chunks, FIXED_PARAMS)
        assert {c.doc_id for c in chunks} == {doc.doc_id}
        assert {c.strategy for c in chunks} == {"fixed"}


def test_fixed_short_body_is_one_chunk() -> None:
    (chunk,) = chunk_fixed(_doc("Short body. Two."))
    assert (chunk.text, chunk.char_start, chunk.char_end) == ("Short body. Two.", 0, 16)


def test_fixed_extends_to_the_word_end_within_the_cap() -> None:
    body = "a" * 158 + " " + "b" * 10 + " tail"
    first = chunk_fixed(_doc(body))[0]
    assert first.text == "a" * 158 + " " + "b" * 10  # 169 ≤ 180: extended, not split


def test_fixed_backs_off_when_the_word_exceeds_the_cap() -> None:
    body = "a" * 150 + " " + "b" * 40 + " tail"
    first = chunk_fixed(_doc(body))[0]
    assert first.text == "a" * 150  # extending would give 191 > 180, so back off


def test_fixed_never_splits_a_word_longer_than_the_cap() -> None:
    body = "x" * 300 + " end"
    chunks = chunk_fixed(_doc(body))
    assert chunks[0].text == "x" * 300
    assert chunks[-1].text.endswith("end")


def test_fixed_chunks_cover_overlapping_text() -> None:
    body = " ".join(f"word{i:03d}" for i in range(80))
    chunks = chunk_fixed(_doc(body))
    assert len(chunks) > 2
    _assert_fixed_invariants(body, chunks, FIXED_PARAMS)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.char_end - nxt.char_start <= FIXED_PARAMS.overlap + 8


_WORD = st.text(alphabet="abcdefghij0123456789.,%₹-", min_size=1, max_size=30)
_SEP = st.sampled_from([" ", "  ", "\n", " \t "])
_MAX_SEP = 3


@given(st.lists(st.tuples(_WORD, _SEP), min_size=1, max_size=120))
def test_fixed_invariants_hold_for_random_texts(parts: list[tuple[str, str]]) -> None:
    body = "".join(w + s for w, s in parts).strip()
    _assert_fixed_invariants(body, chunk_fixed(_doc(body)), FIXED_PARAMS)


@given(
    st.lists(st.tuples(_WORD, _SEP), min_size=1, max_size=60),
    st.integers(10, 80),
    st.integers(1, 9),
    st.integers(0, 10),
)
def test_fixed_invariants_hold_for_random_params(
    parts: list[tuple[str, str]], size: int, overlap: int, extend: int
) -> None:
    body = "".join(w + s for w, s in parts).strip()
    params = FixedParams(size=size, overlap=overlap, max_extend=extend)
    # A chunk holding a single word cannot overlap its successor without splitting that word.
    # Overlap is guaranteed once any word plus its separator fits twice into one chunk.
    fits_two_words = (max(len(w) for w, _ in parts) + _MAX_SEP) * 2 < size
    _assert_fixed_invariants(
        body, chunk_fixed(_doc(body), params), params, expect_overlap=fits_two_words
    )


# ------------------------------------------------------------------ sentence-based


def test_sentence_chunks_are_one_per_sentence(kb: list[KbDocument]) -> None:
    for doc in kb:
        chunks = chunk_sentences(doc)
        assert [c.text for c in chunks] == text.split_sentences(doc.body)
        for i, c in enumerate(chunks):
            assert c.index == i and c.strategy == "sentence"
            assert doc.body[c.char_start : c.char_end] == c.text


def test_sentence_chunks_keep_abbreviations_inside() -> None:
    chunks = chunk_sentences(_doc("Rates are 10.5% p.a. for top scores. Use e.g. PAN. Done"))
    assert [c.text for c in chunks] == [
        "Rates are 10.5% p.a. for top scores.",
        "Use e.g. PAN.",
        "Done",
    ]


@given(st.lists(st.tuples(_WORD, st.sampled_from([" ", ". ", "? ", "! "])), max_size=60))
def test_sentence_spans_reconstruct_random_texts(parts: list[tuple[str, str]]) -> None:
    body = "".join(w + s for w, s in parts).strip() or "x"
    for c in chunk_sentences(_doc(body)):
        assert body[c.char_start : c.char_end] == c.text and c.text.strip() == c.text


# ------------------------------------------------------------------ ids and determinism


def test_chunk_ids_are_deterministic_and_formatted(kb: list[KbDocument]) -> None:
    for strategy in ("fixed", "sentence"):
        first = [c.chunk_id for d in kb for c in chunk_document(d, strategy)]
        second = [c.chunk_id for d in load_kb(KB_DIR) for c in chunk_document(d, strategy)]
        assert first == second
        assert len(set(first)) == len(first)
    c = chunk_fixed(kb[0])[0]
    assert c.chunk_id == f"{kb[0].doc_id}::fixed::000"
    assert chunk_sentences(kb[0])[1].chunk_id == f"{kb[0].doc_id}::sentence::001"


def test_chunking_is_pure(kb: list[KbDocument]) -> None:
    doc = kb[7]
    assert chunk_fixed(doc) == chunk_fixed(doc)
    assert chunk_sentences(doc) == chunk_sentences(doc)


def test_unknown_strategy_is_rejected(kb: list[KbDocument]) -> None:
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_document(kb[0], "paragraph")  # type: ignore[arg-type]


def test_chunk_params_describe_both_strategies() -> None:
    params = chunk_params()
    assert params["fixed"] == {"size": 160, "overlap": 40, "max_extend": 20}
    assert params["sentence"]["abbreviations"] == sorted(text.ABBREVIATIONS)
