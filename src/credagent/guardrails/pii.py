"""PII masking (§ARCH 7.12.1) — pipeline stage 1, before anything else sees the text.

The text is first passed through `text.strip_invisible` (zero-width and bidi characters removed,
NFKC), so a PAN split by U+200B or written in full-width digits is still found. Five rules then
run in strict precedence order; a match is kept only if it overlaps no span an earlier rule
(or a longer match of the same rule) already took. Each kept span is replaced by its typed
token. The masked text is the stripped text with those replacements.

| # | rule | token |
|---|---|---|
| 1 | PAN: `AAAPA9999A` in any case, or in capitals with a space/hyphen between groups | PAN |
| 2 | Aadhaar, grouped: `[2-9]999 9999 9999` (space or hyphen) | AADHAAR |
| 3 | bank account after `a/c`, `acct`, `account` (+ `no`/`number`/`#`): the digits only | BANK |
| 4 | Aadhaar, contiguous: `[2-9]` + 11 digits | AADHAAR |
| 5 | bank account, bare: any remaining run of ≥ 9 digits (fail-safe) | BANK |

Boundaries are "not a letter/digit" lookarounds rather than `\\b`, so `_` cannot glue a value to
a neighbour and hide it. Over-masking is deliberate (D-035): no Verhoeff gate on Aadhaar, rule 5
also takes phone numbers and runs longer than 18 digits, while amounts (≤ 8 digits, §7.1),
`₹25,000`, `10.5%`, dates and `LA-0007` stay intact. Tokens contain no digits, so masking is
idempotent. `MaskResult` carries the masked text and counts per type, never the values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from credagent.text import strip_invisible

PiiType = Literal["PAN", "AADHAAR", "BANK_ACCOUNT"]
PII_TYPES: Final[tuple[PiiType, ...]] = ("PAN", "AADHAAR", "BANK_ACCOUNT")
TOKENS: Final[dict[PiiType, str]] = {
    "PAN": "[PAN_REDACTED]",
    "AADHAAR": "[AADHAAR_REDACTED]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT_REDACTED]",
}

_NOT_ALNUM_BEFORE = r"(?<![^\W_])"  # previous char is not a letter or digit (or there is none)
_NOT_ALNUM_AFTER = r"(?![^\W_])"
_NO_DIGIT_BEFORE = r"(?<!\d)"
_NO_DIGIT_AFTER = r"(?!\d)"


@dataclass(frozen=True)
class Rule:
    name: str
    pii_type: PiiType
    pattern: re.Pattern[str]
    group: int = 0  # the capture group that is masked (rule 3 keeps its keyword)


RULES: Final[tuple[Rule, ...]] = (
    Rule(
        "pan",
        "PAN",
        re.compile(
            _NOT_ALNUM_BEFORE
            + r"(?:(?i:[a-z]{3}[abcfghljpt][a-z][0-9]{4}[a-z])"
            + r"|[A-Z]{3}[ABCFGHLJPT][A-Z][ -]?[0-9]{4}[ -]?[A-Z])"
            + _NOT_ALNUM_AFTER
        ),
    ),
    Rule(
        "aadhaar_grouped",
        "AADHAAR",
        re.compile(_NO_DIGIT_BEFORE + r"[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}" + _NO_DIGIT_AFTER),
    ),
    Rule(
        "bank_account_keyword",
        "BANK_ACCOUNT",
        re.compile(
            _NOT_ALNUM_BEFORE
            + r"(?i:a/?c|acct|account)(?:\s*(?i:no|number|#))?\.?\s*[:\-#]?\s*"
            + r"(\d{9,18})"
            + _NO_DIGIT_AFTER
        ),
        group=1,
    ),
    Rule(
        "aadhaar_contiguous",
        "AADHAAR",
        re.compile(_NO_DIGIT_BEFORE + r"[2-9]\d{11}" + _NO_DIGIT_AFTER),
    ),
    Rule("bank_account_bare", "BANK_ACCOUNT", re.compile(_NO_DIGIT_BEFORE + r"\d{9,}")),
)


class MaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    counts_by_type: dict[PiiType, int]

    @property
    def total(self) -> int:
        return sum(self.counts_by_type.values())


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int, PiiType]]) -> bool:
    a, b = span
    return any(a < end and start < b for start, end, _ in taken)


class PiiMasker:
    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self._rules = rules

    def _spans(self, stripped: str) -> list[tuple[int, int, PiiType]]:
        taken: list[tuple[int, int, PiiType]] = []
        for rule in self._rules:
            spans = [m.span(rule.group) for m in rule.pattern.finditer(stripped)]
            for span in sorted(spans, key=lambda s: (s[0] - s[1], s[0])):  # longest first
                if not _overlaps(span, taken):
                    taken.append((*span, rule.pii_type))
        return taken

    def find(self, text: str) -> list[str]:
        """The raw PII values in `text`, in order (for the eval judge's safety check, §ARCH 7.18).
        Never logged or stored."""
        stripped = strip_invisible(text)
        return [stripped[a:b] for a, b, _ in sorted(self._spans(stripped))]

    def mask(self, text: str) -> MaskResult:
        stripped = strip_invisible(text)
        taken = self._spans(stripped)
        counts: dict[PiiType, int] = dict.fromkeys(PII_TYPES, 0)
        out = stripped
        for start, end, pii_type in sorted(taken, reverse=True):
            out = out[:start] + TOKENS[pii_type] + out[end:]
            counts[pii_type] += 1
        return MaskResult(text=out, counts_by_type=counts)
