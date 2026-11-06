"""The deterministic mock-LLM engine (§ARCH 7.8). Pure and stateless: every output is a function
of the arguments, built on the `credagent.text` primitives. Phase 3 surface:

* `support_report(answer, context)`: the support rule shared by generation, the groundedness
  guard, the reviewer and the judge. A sentence S is supported by context C iff
  (a) `token_recall(S, best-matching context sentence) ≥ SUPPORT_RECALL` and
  (b) every number in S appears among the numbers of C (catches "0% interest for life").
* `generate_grounded(query, chunks, threshold=…)`: §ARCH 7.5. Below the threshold → the
  fallback. Otherwise extractive synthesis: split the chunks into sentences, drop repeats
  (overlapping fixed chunks, P16) and fragments contained in another sentence, score each by
  `RECALL_WEIGHT·query-token recall + SIMILARITY_WEIGHT·chunk similarity`, keep the best
  `MAX_ANSWER_SENTENCES` scoring ≥ `MIN_SENTENCE_SCORE`, and emit them verbatim in document
  order. The answer is grounded by construction.

Nothing here knows any query: behaviour comes only from the query tokens and the chunks (N7).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from credagent.rag.retriever import NO_MATCH_SIMILARITY
from credagent.rag.store import RetrievedChunk
from credagent.schemas import FALLBACK_ANSWER, GeneratedAnswer, SourceRef
from credagent.text import content_tokens, extract_numbers, normalize, sentence_spans

SUPPORT_RECALL: Final = 0.6
RECALL_WEIGHT: Final = 0.7
SIMILARITY_WEIGHT: Final = 0.3
# Similarity alone contributes at most SIMILARITY_WEIGHT, so a sentence must share at least one
# query token to be selected (D-024).
MIN_SENTENCE_SCORE: Final = SIMILARITY_WEIGHT
MAX_ANSWER_SENTENCES: Final = 2


def token_recall(sentence: str, reference: str) -> float:
    """Share of the sentence's content tokens found in `reference`. A sentence without content
    tokens makes no claim, so its recall is 1.0."""
    tokens = set(content_tokens(sentence))
    if not tokens:
        return 1.0
    return len(tokens & set(content_tokens(reference))) / len(tokens)


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    best_match: str | None
    recall: float
    numbers_missing: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.recall >= SUPPORT_RECALL and not self.numbers_missing


@dataclass(frozen=True)
class SupportReport:
    sentences: tuple[SentenceSupport, ...]

    @property
    def total(self) -> int:
        return len(self.sentences)

    @property
    def supported_count(self) -> int:
        return sum(s.supported for s in self.sentences)

    @property
    def ratio(self) -> float:
        """Supported / total; an answer with no sentences makes no claims (1.0)."""
        return self.supported_count / self.total if self.total else 1.0

    @property
    def all_supported(self) -> bool:
        return self.supported_count == self.total


@dataclass(frozen=True)
class _Candidate:
    text: str
    key: str  # normalised text, for dedupe and containment
    position: tuple[int, int]  # (document order, offset in the document body)
    chunk: RetrievedChunk


def _clamp(similarity: float) -> float:
    """float32 cosine distance can be a hair below 0 or above 2."""
    return min(1.0, max(-1.0, similarity))


def _split(text: str) -> list[tuple[int, str]]:
    return [(a, text[a:b]) for a, b in sentence_spans(text)]


class MockEngine:
    # ------------------------------------------------------------------ support rule
    def support_report(self, answer: str, context: Sequence[str]) -> SupportReport:
        context_sentences = [s for c in context for _, s in _split(c)]
        context_numbers = {n for c in context for n in extract_numbers(c)}
        out: list[SentenceSupport] = []
        for _, sentence in _split(answer):
            best: str | None = None
            recall = 0.0
            for candidate in context_sentences:
                r = token_recall(sentence, candidate)
                if best is None or r > recall:
                    best, recall = candidate, r
            missing = tuple(n for n in extract_numbers(sentence) if n not in context_numbers)
            out.append(SentenceSupport(sentence, best, recall, missing))
        return SupportReport(tuple(out))

    # ------------------------------------------------------------------ grounded generation
    def generate_grounded(
        self, query: str, chunks: Sequence[RetrievedChunk], *, threshold: float
    ) -> GeneratedAnswer:
        top1 = _clamp(max((c.similarity for c in chunks), default=NO_MATCH_SIMILARITY))
        if not chunks or top1 < threshold:
            return self._fallback(top1)

        query_tokens = set(content_tokens(query))
        scored = []
        for cand in self._candidates(chunks):
            recall = (
                len(query_tokens & set(content_tokens(cand.text))) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = RECALL_WEIGHT * recall + SIMILARITY_WEIGHT * _clamp(cand.chunk.similarity)
            if score >= MIN_SENTENCE_SCORE:
                scored.append((score, cand))
        if not scored:
            return self._fallback(top1)

        best = sorted(scored, key=lambda sc: (-sc[0], sc[1].position))[:MAX_ANSWER_SENTENCES]
        chosen = sorted((c for _, c in best), key=lambda c: c.position)
        sources: dict[str, SourceRef] = {}
        for c in chosen:
            sources.setdefault(
                c.chunk.chunk_id,
                SourceRef(
                    doc_id=c.chunk.doc_id, chunk_id=c.chunk.chunk_id,
                    similarity=_clamp(c.chunk.similarity),
                ),
            )  # fmt: skip
        return GeneratedAnswer(
            answer=" ".join(c.text for c in chosen),
            sources=tuple(sources.values()),
            grounded=True,
            top1_similarity=top1,
        )

    @staticmethod
    def _fallback(top1: float) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=FALLBACK_ANSWER, sources=(), grounded=False, top1_similarity=top1
        )

    @staticmethod
    def _candidates(chunks: Sequence[RetrievedChunk]) -> list[_Candidate]:
        """Sentences of the chunks in rank order, without exact repeats (the higher-ranked copy
        wins) or fragments contained in another sentence of the context (P16)."""
        doc_order: dict[str, int] = {}
        seen: dict[str, _Candidate] = {}
        for chunk in sorted(chunks, key=lambda c: (c.rank, c.chunk_id)):
            order = doc_order.setdefault(chunk.doc_id, len(doc_order))
            for offset, text in _split(chunk.text):
                key = normalize(text)
                if key not in seen:
                    seen[key] = _Candidate(text, key, (order, chunk.char_start + offset), chunk)
        keys = list(seen)
        return [
            cand
            for cand in seen.values()
            if not any(cand.key != other and cand.key in other for other in keys)
        ]
