"""Text primitives shared by chunking, retrieval, generation and the guards (§ARCH 7.3, 7.8).

* `clean`: NFKC + zero-width removal + whitespace collapse, case kept (KB loading).
* `normalize`: `clean` + casefold (every comparison).
* `strip_invisible`: zero-width and bidi-control characters removed, then NFKC; case, spacing and
  line breaks kept (the guardrails start here, so a PAN split by U+200B or a delimiter hidden
  behind U+202E is still seen).
* `content_tokens`: normalised word/number tokens minus a fixed stopword list.
* `extract_numbers`: canonical numbers (`₹1.5 lakh` → `150000`, `10,000` → `10000`, `2%`).
* `split_sentences` / `sentence_spans`: rule-based splitter aware of `p.a.`, `e.g.`, `i.e.`,
  `Rs.`, `No.`, decimals and ellipses (P15).
* `find_record_id` / `RECORD_ID_CANDIDATE`: an application-id-like token (`LA-` + 1–8 ASCII
  digits, any case). Well-formedness (`LA-0000`) is the lookup tool's job, so a malformed id
  reaches it and comes back as `invalid_record_id`.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

_ZERO_WIDTH_CHARS = "\u200b‌‍⁠﻿­"
_ZERO_WIDTH = dict.fromkeys(map(ord, _ZERO_WIDTH_CHARS))
# LRM, RLM, ALM and the embedding/override/isolate controls (U+202A–U+202E, U+2066–U+2069).
_BIDI_CHARS = "\u200e\u200f\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
INVISIBLE_CHARS = frozenset(_ZERO_WIDTH_CHARS + _BIDI_CHARS)
_INVISIBLE = dict.fromkeys(map(ord, INVISIBLE_CHARS))
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """NFKC, strip zero-width characters and collapse whitespace; case is preserved."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)).strip()


def normalize(text: str) -> str:
    """`clean` + casefold. Idempotent."""
    return clean(clean(text).casefold())


def strip_invisible(text: str) -> str:
    """Remove zero-width and bidi-control characters, then NFKC. Idempotent."""
    return unicodedata.normalize("NFKC", text.translate(_INVISIBLE))


# ------------------------------------------------------------------ tokens

# Negations (`no`, `not`, `never`) are deliberately kept: they flip a policy statement.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "could", "do",
        "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
        "its", "may", "me", "might", "my", "of", "on", "or", "our", "shall", "should", "so",
        "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to",
        "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
        "would", "you", "your",
    }
)  # fmt: skip
_TOKEN = re.compile(r"₹?\d+(?:,\d+)*(?:\.\d+)?%?|[^\W\d_]+(?:[-'][^\W\d_]+)*")


def content_tokens(text: str) -> list[str]:
    """Normalised tokens in order, without stopwords or single letters. Numbers keep their
    surface form (`10.5%`, `₹10,000`, `24`)."""
    return [
        t
        for t in _TOKEN.findall(normalize(text))
        if t not in STOPWORDS and not (len(t) == 1 and t.isalpha())
    ]


# ------------------------------------------------------------------ numbers

_MULTIPLIERS = {"lakh": 100_000, "lac": 100_000, "crore": 10_000_000}
_NUMBER = re.compile(
    r"(?<![\d.,])(?P<num>\d+(?:,\d+)*(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%)|\s+(?P<mult>lakhs?|lacs?|crores?)\b)?"
)


def _canonical(value: Decimal) -> str:
    return format(value.normalize(), "f")


def extract_numbers(text: str) -> list[str]:
    """Canonical numbers in order of appearance. Lakh/crore are expanded, digit grouping
    removed, trailing zeros dropped; percentages keep their `%`."""
    out: list[str] = []
    for m in _NUMBER.finditer(normalize(text)):
        value = Decimal(m["num"].replace(",", ""))
        if m["mult"]:
            value *= _MULTIPLIERS[m["mult"].rstrip("s")]
        out.append(_canonical(value) + ("%" if m["pct"] else ""))
    return out


# ------------------------------------------------------------------ sentence splitter

# Lower-case, without the final period. A period after one of these does not end a sentence...
ABBREVIATIONS = frozenset(
    {"p.a", "e.g", "i.e", "rs", "no", "nos", "mr", "mrs", "ms", "dr", "vs", "viz", "approx",
     "incl", "etc", "a.m", "p.m", "sr", "jr"}
)  # fmt: skip
# ...except these, which may also close a sentence when a capitalised sentence follows.
SENTENCE_FINAL_ABBREVIATIONS = frozenset({"p.a", "etc", "a.m", "p.m"})

_TERMINATOR = re.compile(r"(?:\.{3}|…|[.!?])+[\"'”’)\]]*")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_OPENERS = "\"'“‘(["


def _starts_sentence(text: str, pos: int) -> bool:
    """True if the next non-space character from `pos` can open a sentence."""
    m = _WS.match(text, pos)
    if m is None:
        return pos >= len(text)
    nxt = m.end()
    if nxt >= len(text):
        return True
    ch = text[nxt]
    return ch.isupper() or ch.isdigit() or ch == "₹" or ch in _OPENERS


def _word_before(text: str, pos: int) -> str:
    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:pos].lstrip(_OPENERS).casefold()


def _is_boundary(text: str, m: re.Match[str]) -> bool:
    if not _starts_sentence(text, m.end()):
        return False
    mark = m.group().rstrip("\"'”’)]")
    if mark == ".":
        word = _word_before(text, m.start())
        if word in ABBREVIATIONS:
            return word in SENTENCE_FINAL_ABBREVIATIONS and _next_is_upper(text, m.end())
    return True


def _next_is_upper(text: str, pos: int) -> bool:
    rest = text[pos:].lstrip()
    return bool(rest) and rest[0].isupper()


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _PARAGRAPH_BREAK.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """`(start, end)` offsets of each sentence in `text`, without surrounding whitespace.
    A blank line is a hard boundary; the last sentence needs no terminator."""
    spans: list[tuple[int, int]] = []

    def emit(a: int, b: int) -> None:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((a, b))

    for p_start, p_end in _paragraph_spans(text):
        start = p_start
        for m in _TERMINATOR.finditer(text, p_start, p_end):
            if m.end() == p_end or _is_boundary(text[:p_end], m):
                emit(start, m.end())
                start = m.end()
        emit(start, p_end)
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


# ------------------------------------------------------------------ record ids

RECORD_ID_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])LA-[0-9]{1,8}(?![A-Za-z0-9])", re.IGNORECASE)


def find_record_id(text: str) -> str | None:
    """The first application-id-like token, upper-cased, or `None`."""
    m = RECORD_ID_CANDIDATE.search(text)
    return m.group().upper() if m else None
