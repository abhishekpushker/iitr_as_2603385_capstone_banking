"""Two chunking strategies (§ARCH 7.3, Task 3), both pure functions of a `KbDocument`.

* `fixed`: windows of `size` characters overlapping by about `overlap` characters. A window
  never splits a word: it extends to the next whitespace (up to `size + max_extend`), else backs
  off to the previous whitespace. A single word longer than that cap becomes its own chunk.
* `sentence`: one chunk per sentence from `text.sentence_spans`.

`chunk_id = f"{doc_id}::{strategy}::{index:03d}"`, so ids are deterministic and upserts idempotent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from credagent.text import ABBREVIATIONS, SENTENCE_FINAL_ABBREVIATIONS, sentence_spans

Strategy = Literal["fixed", "sentence"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "sentence")


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    strategy: Strategy
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


@dataclass(frozen=True)
class FixedParams:
    size: int = 160
    overlap: int = 40
    max_extend: int = 20

    def __post_init__(self) -> None:
        if self.size < 1 or not 0 <= self.overlap < self.size or self.max_extend < 0:
            raise ValueError(f"invalid fixed-size chunk params {self}")


FIXED_PARAMS = FixedParams()


class ChunkSource(Protocol):
    """What chunking reads: a `KbDocument`, or a document added via `/v1/documents`."""

    @property
    def doc_id(self) -> str: ...

    @property
    def body(self) -> str: ...


def _chunk(doc: ChunkSource, strategy: Strategy, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{strategy}::{index:03d}",
        doc_id=doc.doc_id,
        strategy=strategy,
        index=index,
        text=doc.body[start:end],
        char_start=start,
        char_end=end,
    )


def _window_end(body: str, start: int, p: FixedParams) -> int:
    """End of the window opening at `start` (a word start), snapped to a word boundary."""
    n = len(body)
    end = start + p.size
    if end >= n:
        end = n
    elif not body[end].isspace() and not body[end - 1].isspace():  # mid-word
        word_end = end
        while word_end < n and not body[word_end].isspace():
            word_end += 1
        if word_end <= start + p.size + p.max_extend:
            end = word_end
        else:
            back = end
            while back > start and not body[back - 1].isspace():
                back -= 1
            end = back if back > start else word_end  # one over-long word: keep it whole
    while body[end - 1].isspace():
        end -= 1
    return end


def _next_start(body: str, start: int, end: int, p: FixedParams) -> int:
    """Start of the next window: the last word start at or before `end - overlap` that lies
    inside the current chunk, else the last word start inside it, else the word after it."""
    word_starts = [
        i for i in range(start + 1, end) if body[i - 1].isspace() and not body[i].isspace()
    ]
    if word_starts:
        before = [i for i in word_starts if i <= end - p.overlap]
        return before[-1] if before else word_starts[-1]
    nxt = end
    while nxt < len(body) and body[nxt].isspace():
        nxt += 1
    return nxt


def chunk_fixed(doc: ChunkSource, params: FixedParams = FIXED_PARAMS) -> list[Chunk]:
    body, start = doc.body, 0
    chunks: list[Chunk] = []
    while start < len(body) and body[start].isspace():
        start += 1
    while start < len(body):
        end = _window_end(body, start, params)
        chunks.append(_chunk(doc, "fixed", len(chunks), start, end))
        if body[end:].strip() == "":
            break
        start = _next_start(body, start, end, params)
    return chunks


def chunk_sentences(doc: ChunkSource) -> list[Chunk]:
    return [_chunk(doc, "sentence", i, a, b) for i, (a, b) in enumerate(sentence_spans(doc.body))]


def chunk_document(doc: ChunkSource, strategy: Strategy) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc)
    if strategy == "sentence":
        return chunk_sentences(doc)
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def chunk_params() -> dict[str, Any]:
    """Everything that changes chunk output; hashed into `kb_version`."""
    return {
        "fixed": asdict(FIXED_PARAMS),
        "sentence": {
            "abbreviations": sorted(ABBREVIATIONS),
            "sentence_final": sorted(SENTENCE_FINAL_ABBREVIATIONS),
        },
    }
